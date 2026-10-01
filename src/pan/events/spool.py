"""Durable event spool (integration spec §4.4): SQLite in WAL mode, many writer processes, one consumer.

Writers (agent processes) call :meth:`EventSpool.try_append` — a single INSERT that never raises.
The consumer (pan-memoryd) calls :meth:`claim` → process → :meth:`mark_done` / :meth:`mark_failed`.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from pan.events.schema import Actor, AgentEvent, EventType

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_CONTENT_BYTES = 64 * 1024
DEFAULT_RECLAIM_SECONDS = 300.0
STATUSES = ("new", "processing", "done", "dead")

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS events (
        id                TEXT PRIMARY KEY,
        ts                TEXT NOT NULL,
        session_id        TEXT NOT NULL,
        parent_session_id TEXT,
        actor             TEXT NOT NULL,
        event_type        TEXT NOT NULL,
        content           TEXT NOT NULL DEFAULT '{}',
        source_refs       TEXT NOT NULL DEFAULT '[]',
        project           TEXT,
        metadata          TEXT NOT NULL DEFAULT '{}',
        status            TEXT NOT NULL DEFAULT 'new'
                          CHECK (status IN ('new', 'processing', 'done', 'dead')),
        attempts          INTEGER NOT NULL DEFAULT 0,
        last_error        TEXT,
        claimed_by        TEXT,
        claimed_at        REAL,
        finished_at       REAL
    );
    CREATE INDEX IF NOT EXISTS events_status_id ON events (status, id);
    CREATE INDEX IF NOT EXISTS events_session ON events (session_id, id);
    """,
}

_COLUMNS = ("id", "ts", "session_id", "parent_session_id", "actor", "event_type", "content",
            "source_refs", "project", "metadata")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _cap_strings(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if len(raw) <= cap:
            return value
        return raw[:cap].decode("utf-8", "ignore") + f"…[truncated {len(raw) - cap} bytes]"
    if isinstance(value, dict):
        return {k: _cap_strings(v, cap) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cap_strings(v, cap) for v in value]
    return value


def truncate_content(content: dict[str, Any], max_bytes: int = MAX_CONTENT_BYTES) -> tuple[dict[str, Any], bool]:
    """``(content, truncated)`` with the serialized content at most ``max_bytes``.

    Long strings are shortened first (keeps the structure); if that is not enough the payload is
    replaced by a plain excerpt. A ``_truncated`` note records the original size.
    """
    text = _dumps(content)
    size = len(text.encode("utf-8"))
    if size <= max_bytes:
        return content, False
    note = {"original_bytes": size, "note": "payload truncated by PAN spool; full content in Hermes state.db"}
    cap = max_bytes // 4
    while cap >= 256:
        candidate = dict(_cap_strings(content, cap), _truncated=note)
        if len(_dumps(candidate).encode("utf-8")) <= max_bytes:
            return candidate, True
        cap //= 2
    excerpt = text.encode("utf-8")[: max_bytes // 2].decode("utf-8", "ignore")
    return {"_truncated": note, "excerpt": excerpt}, True


class EventSpool:
    """Append-only event queue in ``events.db``. Thread-safe; one connection per instance."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 2000,
                 max_content_bytes: int = MAX_CONTENT_BYTES) -> None:
        self.path = Path(path)
        self.max_content_bytes = max_content_bytes
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self.path), timeout=busy_timeout_ms / 1000, isolation_level=None, check_same_thread=False)
        try:
            self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._migrate()
        except BaseException:
            self.close()
            raise

    # -- lifecycle ---------------------------------------------------------------------------------

    def _migrate(self) -> None:
        conn = self._db()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has schema v{version}; this PAN supports v{SCHEMA_VERSION}")
        if version == SCHEMA_VERSION:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"{self.path} has schema v{version}; this PAN supports v{SCHEMA_VERSION}")
            for target in range(version + 1, SCHEMA_VERSION + 1):
                for statement in filter(str.strip, _MIGRATIONS[target].split(";")):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version={target}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._db().execute("PRAGMA user_version").fetchone()[0]

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError(f"EventSpool {self.path} is closed")
        return self._conn

    @property
    def closed(self) -> bool:
        return self._conn is None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "EventSpool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writer side -------------------------------------------------------------------------------

    def append(self, event: AgentEvent) -> None:
        """Insert one event (status ``new``). Oversized content is truncated. May raise."""
        content, truncated = truncate_content(event.content, self.max_content_bytes)
        refs = list(event.source_refs)
        if truncated and f"session:{event.session_id}" not in refs:
            refs.append(f"session:{event.session_id}")
        row = (event.id, event.ts, event.session_id, event.parent_session_id, event.actor.value,
               event.event_type.value, _dumps(content), _dumps(refs), event.project, _dumps(event.metadata))
        with self._lock:
            self._db().execute(f"INSERT INTO events ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                               row)

    def try_append(self, event: AgentEvent) -> bool:
        """:meth:`append` that logs and returns False instead of raising (the agent never waits on PAN)."""
        try:
            self.append(event)
            return True
        except Exception as exc:
            logger.warning("PAN spool: dropped %s event for session %s: %s",
                           getattr(event.event_type, "value", event.event_type), event.session_id, exc)
            return False

    def append_claimed(self, event: AgentEvent, worker_id: str) -> None:
        """Insert an event the consumer itself synthesizes (e.g. a turn backfilled from Hermes'
        ``state.db``) directly as ``processing`` for ``worker_id``, so it is marked done together
        with the episode it completes."""
        self.append(event)
        with self._lock:
            self._db().execute("UPDATE events SET status='processing', claimed_by=?, claimed_at=? WHERE id=?",
                               (worker_id, time.time(), event.id))

    # -- consumer side -----------------------------------------------------------------------------

    def claim(self, limit: int, worker_id: str, *, reclaim_after: float = DEFAULT_RECLAIM_SECONDS,
              max_attempts: int = 3) -> list[AgentEvent]:
        """Atomically move up to ``limit`` events to ``processing`` (oldest id first).

        ``processing`` rows whose claim is older than ``reclaim_after`` seconds (crashed worker) are
        claimed again; each expiry counts as an attempt, and a row that reaches ``max_attempts``
        becomes ``dead`` instead.
        """
        if limit <= 0:
            return []
        now = time.time()
        cutoff = now - reclaim_after
        with self._lock:
            conn = self._db()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "UPDATE events SET status='dead', attempts=attempts+1, finished_at=?, "
                    "last_error='claim expired (worker ' || COALESCE(claimed_by, '?') || ')' "
                    "WHERE status='processing' AND claimed_at < ? AND attempts+1 >= ?",
                    (now, cutoff, max_attempts))
                conn.execute(
                    "UPDATE events SET status='new', attempts=attempts+1, "
                    "last_error='claim expired (worker ' || COALESCE(claimed_by, '?') || ')' "
                    "WHERE status='processing' AND claimed_at < ?", (cutoff,))
                rows = conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM events WHERE status='new' ORDER BY id LIMIT ?",
                    (limit,)).fetchall()
                if rows:
                    conn.executemany(
                        "UPDATE events SET status='processing', claimed_by=?, claimed_at=? WHERE id=?",
                        [(worker_id, now, r[0]) for r in rows])
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return [_row_to_event(r) for r in rows]

    def mark_done(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        with self._lock:
            cur = self._db().executemany(
                "UPDATE events SET status='done', finished_at=?, last_error=NULL WHERE id=? AND status='processing'",
                [(time.time(), i) for i in ids])
            return cur.rowcount

    def mark_failed(self, event_id: str, error: str, max_attempts: int = 3) -> str | None:
        """Count a failed attempt: back to ``new``, or ``dead`` once ``max_attempts`` is reached.
        Returns the new status (None when the event is unknown)."""
        with self._lock:
            conn = self._db()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "UPDATE events SET attempts=attempts+1, last_error=?, claimed_by=NULL, claimed_at=NULL, "
                    "status=CASE WHEN attempts+1 >= ? THEN 'dead' ELSE 'new' END, "
                    "finished_at=CASE WHEN attempts+1 >= ? THEN ? ELSE NULL END WHERE id=?",
                    (str(error)[:4000], max_attempts, max_attempts, time.time(), event_id))
                row = conn.execute("SELECT status FROM events WHERE id=?", (event_id,)).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return row[0] if row else None

    def release(self, ids: Iterable[str]) -> int:
        """Give claimed events back (``processing`` → ``new``) without counting an attempt — for
        events the consumer cannot process *yet* (open episode, deferred decision)."""
        ids = list(ids)
        if not ids:
            return 0
        with self._lock:
            cur = self._db().executemany(
                "UPDATE events SET status='new', claimed_by=NULL, claimed_at=NULL WHERE id=? AND status='processing'",
                [(i,) for i in ids])
            return cur.rowcount

    # -- inspection / maintenance ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Event counts per status (every status present, zero when empty)."""
        with self._lock:
            rows = self._db().execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
        counts = dict.fromkeys(STATUSES, 0)
        counts.update({status: n for status, n in rows})
        return counts

    def get(self, event_id: str) -> dict[str, Any] | None:
        """One row as a dict including queue fields (status, attempts, last_error, …)."""
        with self._lock:
            conn = self._db()
            cur = conn.execute("SELECT * FROM events WHERE id=?", (event_id,))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        if row is None:
            return None
        d = dict(zip(names, row))
        for key in ("content", "source_refs", "metadata"):
            d[key] = json.loads(d[key])
        return d

    def history(self, session_id: str, *, before_id: str | None = None, limit: int = 20,
                event_types: Iterable[str] | None = None, statuses: Iterable[str] = ("done",)) -> list[AgentEvent]:
        """Up to ``limit`` most recent events of a session (older than ``before_id``), oldest first.
        The consumer uses it for cross-batch context (e.g. the turn a confirmation refers to)."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM events WHERE session_id=?"
        params: list[Any] = [session_id]
        if before_id:
            sql += " AND id < ?"
            params.append(before_id)
        types = list(event_types or [])
        if types:
            sql += f" AND event_type IN ({', '.join('?' * len(types))})"
            params += types
        sts = list(statuses)
        if sts:
            sql += f" AND status IN ({', '.join('?' * len(sts))})"
            params += sts
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._db().execute(sql, params).fetchall()
        return [_row_to_event(r) for r in reversed(rows)]

    def events_after(self, event_id: str, *, event_types: Iterable[str] = (), limit: int = 1000) -> list[AgentEvent]:
        """Events (any session, any status) with an id greater than ``event_id``, oldest first."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM events WHERE id > ?"
        params: list[Any] = [event_id]
        types = list(event_types)
        if types:
            sql += f" AND event_type IN ({', '.join('?' * len(types))})"
            params += types
        sql += " ORDER BY id LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._db().execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def last_event(self, session_id: str, event_type: str, *, before_id: str | None = None) -> AgentEvent | None:
        """Newest event of ``event_type`` in ``session_id`` (any status), optionally before ``before_id``."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM events WHERE session_id=? AND event_type=?"
        params: list[Any] = [session_id, event_type]
        if before_id:
            sql += " AND id < ?"
            params.append(before_id)
        with self._lock:
            row = self._db().execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()
        return _row_to_event(row) if row else None

    def list_rows(self, *, session_id: str | None = None, status: str | None = None,
                  limit: int = 50) -> list[dict[str, Any]]:
        """Most recent rows (newest last) with queue fields, for ``pan memory inspect`` / replay."""
        sql, params = "SELECT * FROM events", []
        where = []
        if session_id:
            where.append("session_id=?")
            params.append(session_id)
        if status:
            where.append("status=?")
            params.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            cur = self._db().execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        out = []
        for row in reversed(rows):
            d = dict(zip(names, row))
            for key in ("content", "source_refs", "metadata"):
                d[key] = json.loads(d[key])
            out.append(d)
        return out

    def prune(self, older_than_days: float) -> int:
        """Delete ``done`` events finished more than ``older_than_days`` ago; returns the count."""
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            cur = self._db().execute("DELETE FROM events WHERE status='done' AND finished_at < ?", (cutoff,))
            return cur.rowcount


def _row_to_event(row: tuple) -> AgentEvent:
    d = dict(zip(_COLUMNS, row))
    return AgentEvent(
        id=d["id"], ts=d["ts"], session_id=d["session_id"], parent_session_id=d["parent_session_id"],
        actor=Actor(d["actor"]), event_type=EventType(d["event_type"]), content=json.loads(d["content"]),
        source_refs=json.loads(d["source_refs"]), project=d["project"], metadata=json.loads(d["metadata"]))


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"
