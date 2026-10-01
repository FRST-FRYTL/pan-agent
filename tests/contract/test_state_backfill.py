"""Crash reconciliation against Hermes' state.db (M5, spec §9) with the pinned Hermes SessionDB.

A process that dies mid-turn leaves PAN's tool events but no ``turn`` event (``sync_turn`` never
ran). Hermes has already persisted the user message (and maybe the assistant's text) to state.db;
the daemon completes the episode from there — read-only.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import fixed_clock
from pan.events.schema import Actor, AgentEvent, EventType
from pan.events.spool import EventSpool
from pan.hermes import compat
from pan.memory.backfill import missing_turn
from pan.paths import PanPaths

pytestmark = pytest.mark.contract

SESSION = "20260923_120000_crash1"
USER = "Note for later: our vLLM server runs on port 8000."


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _state_db(home: Path, t0: float, *, assistant: str = "") -> Path:
    from hermes_state import SessionDB

    db_path = home / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session(SESSION, "cli")
        db.append_message(SESSION, "user", USER, timestamp=t0)
        db.append_message(SESSION, "assistant", "", tool_calls=[{"id": "c1", "type": "function", "function": {
            "name": "terminal", "arguments": json.dumps({"command": "ss -ltnp"})}}], timestamp=t0 + 1)
        db.append_message(SESSION, "tool", '{"output": "LISTEN 0 4096 *:8000"}', tool_call_id="c1",
                          tool_name="terminal", timestamp=t0 + 2)
        if assistant:
            db.append_message(SESSION, "assistant", assistant, timestamp=t0 + 3)
    finally:
        db.close()
    return db_path


def _tool_event(t: float) -> AgentEvent:
    return AgentEvent(event_type=EventType.TOOL_CALL, session_id=SESSION, actor=Actor.TOOL, ts=_iso(t),
                      content={"tool": "terminal", "args": {"command": "ss -ltnp"}, "status": "ok",
                               "result_excerpt": '{"output": "LISTEN 0 4096 *:8000"}', "tool_call_id": "c1"})


def test_read_session_messages_is_read_only_and_ordered(hermes_home):
    t0 = time.time() - 600
    db_path = _state_db(hermes_home, t0, assistant="Noted.")
    before = db_path.stat().st_mtime_ns
    msgs = compat.read_session_messages(db_path, SESSION)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert {"id", "role", "content", "timestamp", "tool_call_id"} <= set(msgs[0])
    assert db_path.stat().st_mtime_ns == before
    turn = missing_turn(msgs, first_event_ts=t0 + 1.5)
    assert turn["user"] == USER and turn["assistant"] == "Noted." and len(turn["message_ids"]) == 4
    assert missing_turn(msgs, first_event_ts=t0 + 1.5, captured_user=USER) is None   # already captured
    assert missing_turn(msgs, first_event_ts=t0 + 1.5, after_ts=t0 + 0.5) is None    # older than last turn
    assert missing_turn(msgs, first_event_ts=t0 - 10) is None                        # user msg after the tools


def test_daemon_completes_a_crashed_turn_from_state_db(hermes_home):
    from pan.daemon.memoryd import MemoryDaemon

    t0 = time.time() - 600
    _state_db(hermes_home, t0)  # the process died before the final answer
    paths = PanPaths.for_home(hermes_home)
    paths.ensure()
    with EventSpool(paths.events_db) as spool:
        spool.append(_tool_event(t0 + 1.5))
    d = MemoryDaemon(paths, clock=fixed_clock, worker_id="test")
    try:
        d.prepare()
        report = d.run_once(flush=True)  # --flush closes the turn-less group (like the idle timeout)
    finally:
        d.close()
    assert report.done == 2 and report.failed == 0  # tool event + the backfilled turn
    rec = [json.loads(line) for line in paths.curation_log.read_text().splitlines()][-1]
    assert rec["closed_by"] == "flush+state.db"
    assert rec["candidate"]["claims"][0] == "Our vLLM server runs on port 8000."
    with EventSpool(paths.events_db) as spool:
        turns = [r for r in spool.list_rows(session_id=SESSION) if r["event_type"] == "turn"]
    assert len(turns) == 1 and turns[0]["content"]["backfill"] == "state.db" and turns[0]["status"] == "done"
    assert any(ref.startswith("msg:") for ref in turns[0]["source_refs"])
    assert (paths.wiki / "operations/vllm-server.md").is_file()


def test_no_state_db_means_no_backfill(hermes_home):
    from pan.daemon.memoryd import MemoryDaemon

    paths = PanPaths.for_home(hermes_home)
    paths.ensure()
    with EventSpool(paths.events_db) as spool:
        spool.append(_tool_event(time.time() - 500))
    d = MemoryDaemon(paths, clock=fixed_clock, worker_id="test")
    try:
        d.prepare()
        report = d.run_once(flush=True)
    finally:
        d.close()
    assert report.done == 1
    rec = [json.loads(line) for line in paths.curation_log.read_text().splitlines()][-1]
    assert rec["closed_by"] == "flush"
