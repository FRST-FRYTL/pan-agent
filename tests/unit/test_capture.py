"""SessionRecorder / CaptureHub: provider-side capture without Hermes."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import load_messages
from pan.config import CaptureConfig, PanConfig
from pan.events.spool import EventSpool
from pan.hermes.capture import CaptureHub, SessionRecorder, excerpt, new_messages_tail


@pytest.fixture
def hub():
    h = CaptureHub()
    yield h
    h.clear()


def _events(path: Path) -> list:
    with EventSpool(path) as s:
        return s.claim(10_000, "test")


def _recorder(tmp_path: Path, hub: CaptureHub, **kw) -> SessionRecorder:
    kw.setdefault("config", PanConfig())
    return SessionRecorder(tmp_path / "events.db", kw.pop("session_id", "s1"), hub=hub, **kw)


def test_tail_first_sync_starts_at_last_user_message():
    messages = load_messages("two_turns_with_file_write")
    tail = new_messages_tail(messages, None)
    assert tail[0] == {"role": "user", "content": "Write the serve flags to notes/vllm.md"}
    assert len(tail) == 4
    assert new_messages_tail(messages, 5) == messages[5:]
    assert new_messages_tail(messages, 99)[0]["role"] == "user"  # shrunk transcript → fall back
    assert all(m["role"] != "system" for m in new_messages_tail(messages, 0))


def test_turn_events_carry_only_new_messages(tmp_path: Path, hub: CaptureHub):
    messages = load_messages("two_turns_with_file_write")
    rec = _recorder(tmp_path, hub)
    rec.turn("Is vLLM running?", "Yes", session_id="s1", messages=messages[:5])
    rec.turn("Write the serve flags", "Done", session_id="s1", messages=messages)
    rec.close()
    first, second = _events(tmp_path / "events.db")
    assert [m["role"] for m in first.content["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert "tool_call:tc-1" in first.source_refs
    roles = [m["role"] for m in second.content["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert second.content["messages"][1]["tool_calls"][0]["name"] == "write_file"
    assert "tool_call:tc-2" in second.source_refs and "tool_call:tc-1" not in second.source_refs
    assert second.event_type.value == "turn" and second.actor.value == "main_agent"
    assert second.metadata["agent_context"] == "primary"


def test_tool_results_are_capped(tmp_path: Path, hub: CaptureHub):
    config = PanConfig(capture=CaptureConfig(tool_result_excerpt_bytes=100))
    rec = _recorder(tmp_path, hub, config=config)
    messages = [{"role": "user", "content": "go"}, {"role": "tool", "tool_call_id": "t", "content": "x" * 5000}]
    rec.turn("go", "ok", messages=messages)
    rec.close()
    (event,) = _events(tmp_path / "events.db")
    assert len(event.content["messages"][1]["content"]) < 200


@pytest.mark.parametrize("context", ["cron", "flush"])
def test_skipped_contexts_record_nothing(tmp_path: Path, hub: CaptureHub, context: str):
    rec = _recorder(tmp_path, hub, agent_context=context)
    assert not rec.enabled
    assert rec.turn("u", "a", messages=[]) is False
    assert hub.lookup("s1") is None
    rec.close()
    assert not (tmp_path / "events.db").exists()


def test_session_switch_rebinds_and_session_end_dedupes(tmp_path: Path, hub: CaptureHub):
    rec = _recorder(tmp_path, hub)
    rec.session_switch("s2", parent_session_id="s1", reset=False)
    assert hub.lookup("s2").parent_session_id == "s1"
    assert rec.session_end([{"role": "user", "content": "x"}]) is True
    assert rec.session_end([]) is False  # same (session, kind)
    rec.turn("again", "yes", messages=[{"role": "user", "content": "again"}])
    assert rec.session_end([]) is True  # resumed → may close again
    rec.close()
    assert hub.lookup("s1") is None and hub.lookup("s2") is None
    types = [(e.event_type.value, e.session_id) for e in _events(tmp_path / "events.db")]
    assert types == [("session_switch", "s2"), ("session_end", "s2"), ("turn", "s2"), ("session_end", "s2")]


def test_close_keeps_newer_binding_of_same_session(tmp_path: Path, hub: CaptureHub):
    old = _recorder(tmp_path, hub)
    new = _recorder(tmp_path, hub)  # e.g. gateway rebuilt the agent for the same session
    old.close()
    assert hub.lookup("s1") is not None
    new.close()
    assert hub.lookup("s1") is None


def test_other_provider_events(tmp_path: Path, hub: CaptureHub):
    rec = _recorder(tmp_path, hub)
    rec.memory_write("add", "user", "Prefers Markdown specs.", {"session_id": "s1", "write_origin": "tool"})
    rec.compaction([{"role": "user"}] * 3)
    rec.delegation("research X", "found Y", "child-1", {})
    rec.close()
    events = _events(tmp_path / "events.db")
    assert [e.event_type.value for e in events] == ["l1_write", "compaction", "delegation"]
    assert events[0].content["target"] == "user"
    assert events[1].content["message_count"] == 3
    assert "session:child-1" in events[2].source_refs


def test_excerpt_is_utf8_safe():
    text = "ä" * 100
    cut = excerpt(text, 51)
    assert cut.startswith("ä" * 25) and "bytes" in cut
    assert excerpt([{"type": "text", "text": "hi"}, {"type": "image_url"}], 100) == "hi\n[image_url]"
