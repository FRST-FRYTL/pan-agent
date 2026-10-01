"""EventSpool: durability, claim/reclaim, failure handling, truncation, pruning, concurrency."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

import pytest

from pan.events.schema import Actor, AgentEvent, EventType
from pan.events.spool import MAX_CONTENT_BYTES, SCHEMA_VERSION, EventSpool, truncate_content


def _event(i: int = 0, session: str = "s1", **content) -> AgentEvent:
    return AgentEvent(event_type=EventType.TOOL_CALL, session_id=session, actor=Actor.TOOL,
                      content={"i": i, **content}, source_refs=[f"tool_call:tc-{i}"])


@pytest.fixture
def spool(tmp_path: Path):
    with EventSpool(tmp_path / "pan" / "events.db") as s:
        yield s


def test_schema_and_pragmas(spool: EventSpool):
    assert spool.schema_version == SCHEMA_VERSION
    conn = sqlite3.connect(spool.path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    columns = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    assert {"id", "ts", "session_id", "parent_session_id", "actor", "event_type", "content", "source_refs",
            "project", "metadata", "status", "attempts", "last_error", "claimed_by", "claimed_at"} <= columns
    conn.close()


def test_reopen_keeps_events(tmp_path: Path):
    path = tmp_path / "events.db"
    with EventSpool(path) as s:
        s.append(_event(1))
    with EventSpool(path) as s:
        assert s.stats()["new"] == 1


def test_newer_schema_is_refused(tmp_path: Path):
    path = tmp_path / "events.db"
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError):
        EventSpool(path)


def test_append_claim_round_trip(spool: EventSpool):
    events = [_event(i) for i in range(5)]
    for e in events:
        spool.append(e)
    claimed = spool.claim(3, "w1")
    assert [e.id for e in claimed] == [e.id for e in events[:3]]
    assert claimed[0] == events[0]
    assert spool.stats() == {"new": 2, "processing": 3, "done": 0, "dead": 0}
    assert spool.get(claimed[0].id)["claimed_by"] == "w1"
    assert [e.id for e in spool.claim(10, "w2")] == [e.id for e in events[3:]]
    assert spool.claim(10, "w2") == []
    assert spool.mark_done(e.id for e in claimed) == 3
    assert spool.stats()["done"] == 3


def test_ids_are_ordered_within_a_millisecond(spool: EventSpool):
    events = [_event(i) for i in range(200)]
    for e in events:
        spool.append(e)
    assert [e.content["i"] for e in spool.claim(500, "w")] == list(range(200))


def test_reclaim_expired_claim(spool: EventSpool):
    spool.append(_event(1))
    first = spool.claim(1, "crashed")
    assert spool.claim(1, "w2", reclaim_after=60) == []  # claim still fresh
    time.sleep(0.02)
    again = spool.claim(1, "w2", reclaim_after=0.01)
    assert [e.id for e in again] == [first[0].id]
    row = spool.get(first[0].id)
    assert row["claimed_by"] == "w2" and row["attempts"] == 1 and "crashed" in row["last_error"]


def test_repeatedly_expired_claim_goes_dead(spool: EventSpool):
    spool.append(_event(1))
    spool.claim(1, "w")
    for _ in range(2):
        time.sleep(0.02)
        spool.claim(1, "w", reclaim_after=0.01, max_attempts=3)
    time.sleep(0.02)
    assert spool.claim(1, "w", reclaim_after=0.01, max_attempts=3) == []
    assert spool.stats()["dead"] == 1


def test_mark_failed_retries_then_dead(spool: EventSpool):
    event = _event(1)
    spool.append(event)
    for attempt in (1, 2):
        assert [e.id for e in spool.claim(1, "w")] == [event.id]
        assert spool.mark_failed(event.id, f"boom {attempt}", max_attempts=3) == "new"
    spool.claim(1, "w")
    assert spool.mark_failed(event.id, "boom 3", max_attempts=3) == "dead"
    row = spool.get(event.id)
    assert row["status"] == "dead" and row["attempts"] == 3 and row["last_error"] == "boom 3"
    assert spool.claim(1, "w") == []
    assert spool.mark_failed("unknown", "x") is None


def test_large_payload_truncated_with_source_ref(spool: EventSpool):
    event = _event(1, session="big", result_excerpt="x" * (200 * 1024), tool="terminal")
    spool.append(event)
    row = spool.get(event.id)
    assert len(str(row["content"]).encode()) < MAX_CONTENT_BYTES + 1024
    assert row["content"]["tool"] == "terminal"  # structure kept
    assert row["content"]["_truncated"]["original_bytes"] > 200 * 1024
    assert "session:big" in row["source_refs"]


def test_truncate_many_fields_falls_back_to_excerpt():
    content = {f"k{i}": "y" * 300 for i in range(2000)}
    out, truncated = truncate_content(content, 16 * 1024)
    assert truncated and set(out) == {"_truncated", "excerpt"}
    assert truncate_content({"a": 1}) == ({"a": 1}, False)


def test_try_append_never_raises(tmp_path: Path):
    spool = EventSpool(tmp_path / "events.db")
    spool.close()
    assert spool.try_append(_event(1)) is False
    with EventSpool(tmp_path / "events.db") as s:
        e = _event(2)
        assert s.try_append(e) is True
        assert s.try_append(e) is False  # duplicate id


def test_prune_only_old_done(spool: EventSpool):
    for i in range(3):
        spool.append(_event(i))
    done = spool.claim(2, "w")
    spool.mark_done([done[0].id])
    spool.mark_failed(done[1].id, "x", max_attempts=1)  # dead, not pruned
    assert spool.prune(older_than_days=1) == 0
    assert spool.prune(older_than_days=0) == 1
    assert spool.stats() == {"new": 1, "processing": 0, "done": 0, "dead": 1}


# -- concurrency: several writer processes, one consumer ---------------------------------------------

def _writer(path: str, worker: int, count: int) -> None:
    with EventSpool(path) as s:
        for i in range(count):
            assert s.try_append(_event(i, session=f"w{worker}"))


def test_many_writer_processes_one_consumer(tmp_path: Path):
    path = str(tmp_path / "events.db")
    writers, per_writer = 4, 150
    ctx = mp.get_context("spawn")
    with EventSpool(path) as consumer:
        procs = [ctx.Process(target=_writer, args=(path, w, per_writer)) for w in range(writers)]
        for p in procs:
            p.start()
        seen: list[AgentEvent] = []
        deadline = time.time() + 60
        while len(seen) < writers * per_writer and time.time() < deadline:
            batch = consumer.claim(50, "consumer")
            consumer.mark_done(e.id for e in batch)
            seen.extend(batch)
            if not batch:
                time.sleep(0.01)
        for p in procs:
            p.join(30)
            assert p.exitcode == 0
        seen.extend(consumer.claim(10_000, "consumer"))
    assert len(seen) == len({e.id for e in seen}) == writers * per_writer
    for w in range(writers):  # per-writer order is preserved
        assert [e.content["i"] for e in seen if e.session_id == f"w{w}"] == list(range(per_writer))


def test_append_is_fast(spool: EventSpool):
    start = time.perf_counter()
    for i in range(200):
        spool.append(_event(i, result_excerpt="z" * 4096))
    assert (time.perf_counter() - start) / 200 < 0.005


def test_release_returns_events_without_counting_an_attempt(spool: EventSpool):
    for e in [_event(i) for i in range(3)]:
        spool.append(e)
    claimed = spool.claim(3, "w1")
    assert spool.release([claimed[0].id, "unknown"]) == 1
    assert spool.get(claimed[0].id)["status"] == "new" and spool.get(claimed[0].id)["attempts"] == 0
    assert [e.id for e in spool.claim(5, "w1")] == [claimed[0].id]


def test_history_and_list_rows(spool: EventSpool):
    events = [_event(i, session="a" if i % 2 else "b") for i in range(6)]
    for e in events:
        spool.append(e)
    spool.mark_done([e.id for e in spool.claim(4, "w1")])
    a_ids = [e.id for e in events if e.session_id == "a"]
    assert [e.id for e in spool.history("a")] == a_ids[:2]  # only done events by default
    assert [e.id for e in spool.history("a", before_id=a_ids[1], limit=5)] == a_ids[:1]
    assert [e.id for e in spool.history("a", statuses=())] == a_ids
    assert spool.history("a", event_types=["turn"]) == []
    rows = spool.list_rows(session_id="b", limit=2)
    assert [r["id"] for r in rows] == [e.id for e in events if e.session_id == "b"][-2:]
    assert "attempts" in rows[0] and isinstance(rows[0]["content"], dict)
    assert len(spool.list_rows(status="done")) == 4
