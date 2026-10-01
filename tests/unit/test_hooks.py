"""Hook subscriber: registration and events from realistic Hermes hook payloads (no Hermes import).

Payloads mirror the kwargs at the Hermes call sites (see pan.hermes.hooks docstring); the contract
tests check that those call sites still pass them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pan.config import PanConfig
from pan.events.spool import EventSpool
from pan.hermes.capture import CaptureHub, SessionRecorder
from pan.hermes.hooks import HOOK_NAMES, file_changes, make_hooks, register_hooks


class FakeCtx:
    def __init__(self) -> None:
        self.hooks: dict[str, list] = {}

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


@pytest.fixture
def env(tmp_path: Path):
    hub = CaptureHub()
    rec = SessionRecorder(tmp_path / "events.db", "parent", config=PanConfig(), project="/repo", hub=hub)
    yield hub, make_hooks(hub), tmp_path / "events.db"
    rec.close()
    hub.clear()


def _events(path: Path):
    with EventSpool(path) as s:
        return s.claim(10_000, "test")


POST_TOOL_CALL = dict(  # model_tools._emit_post_tool_call_hook
    tool_name="terminal", args={"command": "docker ps"}, result='{"output": "vllm-main Up"}',
    task_id="task-1", session_id="parent", tool_call_id="tc-1", turn_id="turn-1", api_request_id="req-1",
    duration_ms=42, status="ok", error_type=None, error_message=None, middleware_trace=[],
    telemetry_schema_version=1)


def test_register_hooks_subscribes_all_hooks():
    ctx = FakeCtx()
    assert register_hooks(ctx) == 6
    assert set(ctx.hooks) == set(HOOK_NAMES) == {
        "post_tool_call", "subagent_start", "subagent_stop", "on_session_end", "on_session_finalize",
        "on_skill_lifecycle"}
    assert register_hooks(object()) == 0  # context without register_hook


def test_post_tool_call_event(env):
    hub, hooks, path = env
    hooks["post_tool_call"](**POST_TOOL_CALL)
    (event,) = _events(path)
    assert event.event_type.value == "tool_call" and event.actor.value == "tool"
    assert event.session_id == "parent" and event.project == "/repo"
    assert event.content["tool"] == "terminal" and event.content["status"] == "ok"
    assert event.content["result_excerpt"] == POST_TOOL_CALL["result"]
    assert event.content["duration_ms"] == 42
    assert event.source_refs == ["session:parent", "tool_call:tc-1"]


def test_result_excerpt_and_args_capped(env):
    hub, hooks, path = env
    hooks["post_tool_call"](**{**POST_TOOL_CALL, "tool_name": "write_file",
                               "args": {"path": "a.txt", "content": "c" * 20_000},
                               "result": "HEAD" + "r" * 20_000 + "TAIL"})
    tool_call, file_change = _events(path)
    excerpt = tool_call.content["result_excerpt"]
    assert len(excerpt.encode()) <= 8192 + 48
    assert excerpt.startswith("HEAD") and excerpt.endswith("TAIL") and "bytes omitted" in excerpt  # head + tail (M6)
    assert tool_call.content["result_bytes"] == 20_008
    assert len(tool_call.content["args"]["content"]) <= 8192 + 32
    assert file_change.event_type.value == "file_change"
    assert file_change.content == {"path": "a.txt", "op": "write", "tool": "write_file", "tool_call_id": "tc-1"}
    assert "path:a.txt" in file_change.source_refs


def test_failed_write_has_no_file_change(env):
    hub, hooks, path = env
    hooks["post_tool_call"](**{**POST_TOOL_CALL, "tool_name": "patch", "status": "error",
                               "args": {"path": "a.py", "old_string": "x", "new_string": "y"},
                               "error_type": "tool_error", "error_message": "no match"})
    (event,) = _events(path)
    assert event.content["status"] == "error" and event.content["error_message"] == "no match"


def test_file_changes_for_patch_modes():
    assert file_changes("patch", {"path": "a.py", "old_string": "x", "new_string": "y"}) == [
        {"path": "a.py", "op": "replace"}]
    v4a = ("*** Begin Patch\n*** Update File: src/a.py\n@@ x @@\n-a\n+b\n*** Add File: new.txt\n+hi\n"
           "*** Delete File: old.txt\n*** Move File: m1.py -> m2.py\n*** End Patch")
    assert file_changes("patch", {"mode": "patch", "patch": v4a}) == [
        {"path": "src/a.py", "op": "update"}, {"path": "new.txt", "op": "add"},
        {"path": "old.txt", "op": "delete"}, {"path": "m2.py", "op": "move", "from": "m1.py"}]
    assert file_changes("write_file", '{"path": "x.md", "content": ""}') == [{"path": "x.md", "op": "write"}]
    assert file_changes("read_file", {"path": "x"}) == []


def test_unknown_session_is_ignored(env):
    hub, hooks, path = env
    hooks["post_tool_call"](**{**POST_TOOL_CALL, "session_id": "cron-session"})
    hooks["on_skill_lifecycle"](action="used", skill_name="x", session_id="")
    assert _events(path) == []


def test_subagent_lifecycle_links_child(env):
    hub, hooks, path = env
    hooks["subagent_start"](parent_session_id="parent", parent_turn_id="turn-1", parent_subagent_id=None,
                            child_session_id="child", child_subagent_id="sa-1", child_role="leaf",
                            child_goal="find the vLLM flags")
    hooks["post_tool_call"](**{**POST_TOOL_CALL, "session_id": "child", "tool_call_id": "tc-c1"})
    hooks["subagent_stop"](parent_session_id="parent", parent_turn_id="turn-1", child_session_id="child",
                           child_role="leaf", child_summary="flags found", child_status="completed",
                           tool_call_history=[{"tool_name": "terminal", "tool_input": {}, "input_bytes": 10,
                                               "output_bytes": 20, "status": "ok"}], duration_ms=1234)
    hooks["post_tool_call"](**{**POST_TOOL_CALL, "session_id": "child"})  # after stop: unbound
    start, call, stop = _events(path)
    assert (start.event_type.value, start.session_id, start.parent_session_id) == ("subagent_start", "child", "parent")
    assert start.content["task"] == "find the vLLM flags" and start.actor.value == "subagent"
    assert call.parent_session_id == "parent" and call.metadata["agent_context"] == "subagent"
    assert stop.content["summary"] == "flags found" and stop.content["status"] == "completed"
    assert stop.content["tool_call_history"][0]["tool_name"] == "terminal"
    assert hub.lookup("parent") is not None and hub.lookup("child") is None


def test_session_end_turn_end_and_finalize_dedupe(env):
    hub, hooks, path = env
    turn_end = dict(session_id="parent", task_id="t", turn_id="turn-1", completed=True, failed=False,
                    interrupted=False, turn_exit_reason="text_response(stop)", model="primary", platform="cli")
    hooks["on_session_end"](**turn_end)
    hooks["on_session_end"](**turn_end)  # same turn → deduped
    hooks["on_session_end"](**{**turn_end, "turn_id": "turn-2"})
    hooks["on_session_end"](session_id="parent", completed=False, interrupted=True, model="primary",
                            platform="cli", reason="keyboard_interrupt")
    hooks["on_session_finalize"](session_id="parent", platform="cli", reason="session_boundary")
    hooks["on_session_finalize"](session_id="parent", platform="cli")
    events = _events(path)
    kinds = [(e.content["kind"], e.content["source"]) for e in events]
    assert kinds == [("turn_end", "on_session_end")] * 3 + [("close", "on_session_finalize")]
    assert events[0].content["turn_exit_reason"] == "text_response(stop)"
    assert events[2].content["interrupted"] is True


def test_skill_lifecycle(env):
    hub, hooks, path = env
    hooks["on_skill_lifecycle"](action="used", skill_name="deploy", provenance="local", task_id="t",
                                session_id="parent", use_count=3, reused=True, reuse_after_patch=None)
    (event,) = _events(path)
    assert event.event_type.value == "skill_event"
    assert event.content["skill_name"] == "deploy" and event.content["use_count"] == 3


def test_callbacks_never_raise(env):
    hub, hooks, path = env
    for name, cb in hooks.items():
        assert cb() is None
        assert cb(session_id="parent", args=object(), result=object(), tool_call_history="bad") is None


def test_hook_latency_under_5ms(env):
    hub, hooks, path = env
    cb = hooks["post_tool_call"]
    payload = {**POST_TOOL_CALL, "result": "x" * 50_000}
    start = time.perf_counter()
    for _ in range(100):
        cb(**payload)
    assert (time.perf_counter() - start) / 100 < 0.005
