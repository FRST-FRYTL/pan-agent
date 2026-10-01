"""Provider capture driven through Hermes' real MemoryManager (spec §4.1, §8.2 #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

from conftest import load_messages  # noqa: E402
from pan.events.spool import EventSpool  # noqa: E402
from pan.hermes.provider import PanMemoryProvider  # noqa: E402


def _manager(hermes_home: Path, agent_context: str = "primary", session_id: str = "s1"):
    from agent.memory_manager import MemoryManager

    manager = MemoryManager()
    manager.add_provider(PanMemoryProvider())
    manager.initialize_all(session_id=session_id, hermes_home=str(hermes_home), platform="cli",
                           agent_context=agent_context, cwd=str(hermes_home))
    return manager


def _events(hermes_home: Path):
    with EventSpool(hermes_home / "pan" / "events.db") as s:
        return s.claim(10_000, "contract")


def test_sync_turn_records_only_new_tail(hermes_home, capture_hub):
    messages = load_messages("two_turns_with_file_write")
    manager = _manager(hermes_home)
    manager.sync_all("Is vLLM running?", "Yes — up and healthy.", session_id="s1", messages=messages[:5])
    manager.sync_all("Write the serve flags to notes/vllm.md", "Done.", session_id="s1", messages=messages)
    manager.shutdown_all()  # drains the background worker, then closes the spool
    first, second = _events(hermes_home)
    assert first.event_type.value == second.event_type.value == "turn"
    assert first.content["user"] == "Is vLLM running?"
    assert [m.get("tool_call_id") for m in first.content["messages"] if m["role"] == "tool"] == ["tc-1"]
    assert [m.get("tool_call_id") for m in second.content["messages"] if m["role"] == "tool"] == ["tc-2"]
    assert second.content["messages"][0]["content"].startswith("Write the serve flags")
    assert first.metadata["platform"] == "cli" and first.metadata["hermes_version"]
    assert first.project == str(hermes_home.resolve())


def test_fixture_turn_with_tool_call(hermes_home, capture_hub):
    manager = _manager(hermes_home)
    manager.sync_all("Is vLLM running?", "Yes", session_id="s1", messages=load_messages("turn_with_tool_call"))
    manager.shutdown_all()
    (event,) = _events(hermes_home)
    assert [m["role"] for m in event.content["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert event.content["messages"][1]["tool_calls"][0]["name"] == "terminal"


@pytest.mark.parametrize("context", ["cron", "flush"])
def test_cron_and_flush_contexts_are_not_captured(hermes_home, capture_hub, context):
    from pan.hermes.hooks import HOOKS

    manager = _manager(hermes_home, agent_context=context, session_id="cron-1")
    manager.sync_all("run the report", "done", session_id="cron-1", messages=[{"role": "user", "content": "x"}])
    HOOKS["post_tool_call"](tool_name="terminal", args={}, result="ok", session_id="cron-1", status="ok")
    manager.on_session_end([])
    manager.shutdown_all()
    assert not (hermes_home / "pan" / "events.db").exists()


def test_lifecycle_events(hermes_home, capture_hub):
    manager = _manager(hermes_home)
    manager.on_memory_write("add", "user", "Prefers Markdown specs.",
                            metadata={"write_origin": "memory_tool", "session_id": "s1"})
    manager.on_pre_compress([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    manager.on_delegation(task="research flags", result="found", child_session_id="child-1")
    manager.on_session_end([{"role": "user", "content": "a"}])
    manager.on_session_end([{"role": "user", "content": "a"}])  # duplicate boundary
    manager.on_session_switch("s2", parent_session_id="s1", reset=True, reason="new_session")
    manager.shutdown_all()
    events = _events(hermes_home)
    assert [e.event_type.value for e in events] == [
        "l1_write", "compaction", "delegation", "session_end", "session_switch"]
    l1, compaction, delegation, end, switch = events
    assert l1.content["action"] == "add" and l1.content["metadata"]["write_origin"] == "memory_tool"
    assert compaction.content["message_count"] == 2
    assert delegation.content["child_session_id"] == "child-1"
    assert end.content == {"kind": "close", "source": "provider", "message_count": 1}
    assert switch.session_id == "s2" and switch.content["previous_session_id"] == "s1"
    assert switch.content["reason"] == "new_session" and switch.parent_session_id == "s1"
