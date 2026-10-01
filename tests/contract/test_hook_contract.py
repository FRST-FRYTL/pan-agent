"""Contract: the Hermes hook call sites still pass the kwargs PAN's hook subscriber reads (spec §8.2 #3).

Two layers: a static check of every ``invoke_hook("<name>", ...)`` call site PAN depends on, and a
live check that drives Hermes' real emitters through a real PluginManager with PAN registered via
the memory-provider loader (``plugins.memory._ProviderCollector``).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.contract

import pan.hermes  # noqa: E402
from pan.events.spool import EventSpool  # noqa: E402
from pan.hermes.hooks import HOOK_NAMES, HOOKS  # noqa: E402
from pan.hermes.provider import PanMemoryProvider  # noqa: E402

# hook -> (module with the call site, kwargs PAN reads)
CALL_SITES = {
    "post_tool_call": ("model_tools", {"tool_name", "args", "result", "duration_ms", "status", "error_type",
                                       "error_message"}),
    "subagent_start": ("tools.delegate_tool", {"parent_session_id", "parent_turn_id", "child_session_id",
                                               "child_subagent_id", "child_role", "child_goal"}),
    "subagent_stop": ("tools.delegate_tool_results", {"parent_session_id", "parent_turn_id", "child_session_id",
                                                      "child_role", "child_summary", "child_status",
                                                      "tool_call_history", "duration_ms"}),
    "on_session_end": ("agent.turn_finalizer", {"session_id", "turn_id", "completed", "failed", "interrupted",
                                                "turn_exit_reason", "model", "platform"}),
    "on_skill_lifecycle": ("tools.skill_usage", {"action", "skill_name", "provenance", "session_id", "task_id",
                                                 "use_count", "reused", "reuse_after_patch"}),
}


def _hook_call_kwargs(module: str, hook: str) -> list[set[str]]:
    """Keyword names of every call in ``module`` that passes the literal ``hook`` to an invoke helper."""
    source = Path(importlib.util.find_spec(module).origin).read_text(encoding="utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if "invoke_hook" not in func:
            continue
        if any(isinstance(a, ast.Constant) and a.value == hook for a in node.args):
            calls.append({kw.arg for kw in node.keywords if kw.arg})
    return calls


@pytest.mark.parametrize("hook", sorted(CALL_SITES))
def test_call_site_passes_kwargs_pan_reads(hook):
    module, needed = CALL_SITES[hook]
    calls = _hook_call_kwargs(module, hook)
    assert calls, f"no invoke_hook({hook!r}, ...) call left in {module}"
    assert any(needed <= kwargs for kwargs in calls), (
        f"{module} no longer passes {sorted(needed - set.union(*calls))} to {hook}")


def test_post_tool_call_ids_come_from_call_ids():
    import model_tools

    fields = {f.name for f in dataclasses.fields(model_tools._CallIds)}
    assert {"session_id", "tool_call_id", "turn_id", "task_id"} <= fields


def test_pan_hooks_are_valid_hermes_hooks():
    from hermes_cli.plugins import VALID_HOOKS

    assert set(HOOK_NAMES) <= VALID_HOOKS


def test_register_provides_provider_and_hooks():
    class Ctx:
        hooks: dict = {}
        provider = None

        def register_memory_provider(self, provider):
            self.provider = provider

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    ctx = Ctx()
    pan.hermes.register(ctx)
    assert isinstance(ctx.provider, PanMemoryProvider)
    assert set(ctx.hooks) == set(HOOK_NAMES)


def test_cron_platform_maps_to_cron_agent_context():
    from agent.agent_init import _memory_provider_init_kwargs

    class FakeAgent:
        session_id = "cron-sess"
        _session_db = None
        session_cwd = None

        def __getattr__(self, name):
            return None

    assert _memory_provider_init_kwargs(FakeAgent(), "cron")["agent_context"] == "cron"
    assert _memory_provider_init_kwargs(FakeAgent(), "telegram")["agent_context"] == "primary"


# -- live dispatch through Hermes' plugin manager ----------------------------------------------------

@pytest.fixture
def live(hermes_home, capture_hub, monkeypatch):
    """A fresh PluginManager with PAN loaded the way Hermes loads a memory provider, plus an
    initialized provider for session ``sess-live``."""
    import hermes_cli.plugins as plugins
    from plugins.memory import _ProviderCollector

    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    collector = _ProviderCollector("pan", register_skills=False)
    collector.collect(pan.hermes.register, source=pan.hermes.__file__)
    provider = collector.provider
    provider.initialize("sess-live", hermes_home=str(hermes_home), platform="cli", agent_context="primary")
    yield manager, provider, hermes_home / "pan" / "events.db"
    provider.shutdown()


def _events(path: Path):
    with EventSpool(path) as s:
        return s.claim(10_000, "contract")


def test_hooks_registered_through_provider_collector(live):
    manager, _, _ = live
    for name, callback in HOOKS.items():
        assert callback in manager.iter_hook_callbacks(name), f"{name} not registered via _ProviderCollector"


def test_live_post_tool_call(live):
    import model_tools

    _, _, path = live
    model_tools._emit_post_tool_call_hook(
        function_name="write_file", function_args={"path": "notes/a.md", "content": "hello"},
        result='{"bytes_written": 5}', task_id="task-1", session_id="sess-live", tool_call_id="tc-1",
        turn_id="turn-1", duration_ms=7)
    tool_call, file_change = _events(path)
    assert tool_call.content["tool"] == "write_file" and tool_call.content["status"] == "ok"
    assert tool_call.content["tool_call_id"] == "tc-1" and tool_call.content["turn_id"] == "turn-1"
    assert tool_call.content["duration_ms"] == 7
    assert file_change.content["path"] == "notes/a.md"


def test_live_subagent_stop(live):
    from tools.delegate_tool_results import _fire_subagent_stop_hooks

    _, _, path = live
    results = [{"task_index": 0, "summary": "flags found", "status": "completed", "duration_seconds": 1.5,
                "tool_trace": [{"tool": "terminal", "status": "ok", "args_bytes": 10, "result_bytes": 20}]}]
    _fire_subagent_stop_hooks(results, {0: SimpleNamespace(session_id="child-1")},
                              SimpleNamespace(session_id="sess-live", _current_turn_id="turn-1"))
    (event,) = _events(path)
    assert event.event_type.value == "subagent_stop" and event.parent_session_id == "sess-live"
    assert event.content["summary"] == "flags found" and event.content["duration_ms"] == 1500
    assert event.content["tool_call_history"][0]["tool_name"] == "terminal"


def test_live_skill_lifecycle(live):
    from tools.skill_usage import _emit_skill_lifecycle

    _, _, path = live
    _emit_skill_lifecycle("deploy", "used", record={"use_count": 2}, task_id="t", session_id="sess-live")
    (event,) = _events(path)
    assert event.event_type.value == "skill_event"
    assert event.content["skill_name"] == "deploy" and event.content["use_count"] == 2


def test_live_finalize_dedupes_with_provider_session_end(live):
    from hermes_cli.lifecycle import finalize_session

    _, provider, path = live
    finalize_session(session_id="sess-live", platform="cli", reason="session_boundary")
    provider.on_session_end([{"role": "user", "content": "bye"}])
    (event,) = _events(path)
    assert event.event_type.value == "session_end"
    assert event.content["kind"] == "close" and event.content["source"] == "on_session_finalize"
