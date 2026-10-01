"""Crash reconciliation against Hermes' ``state.db`` (integration spec §9, M5).

When an agent process dies mid-turn, PAN's capture has the turn's ``tool_call`` / ``file_change``
events (hooks fire during the turn) but never the ``turn`` event (``sync_turn`` runs after it), and
no ``close``. The daemon closes such a group after ``daemon.episode_idle_minutes`` (or on
``--flush``) — without the user's request and the assistant's text, the classifier sees only tool
output.

Hermes persists every message to ``state.db`` as the turn runs, so the missing text is usually
there. :func:`missing_turn` finds it, **read-only** (``compat.open_session_db_readonly``):

1. only for a session PAN captured (the episode's own ``session_id``) and only for a turn-less
   group closed by ``idle`` / ``flush``;
2. the user message is the newest ``role=user`` message written *before* the group's first event
   (1 s slack) and *after* the previous captured ``turn`` of the session — and its text must differ
   from that captured turn's user text (never a second copy of a turn PAN already has);
3. the assistant text is the last non-empty assistant message after it (before the next user
   message); tool results in between are included like ``sync_turn``'s message tail.

The daemon appends the result as a ``turn`` event (``content.backfill = "state.db"``, source refs
``msg:<state.db id>``) and completes the episode with it. Anything unexpected (no ``state.db``,
schema/API change, locked DB, no match) → no backfill; the episode is processed as before.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SLACK_SECONDS = 1.0
MAX_TEXT = 16384


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # multimodal parts
        return "\n".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in value)
    return "" if value is None else str(value)


def missing_turn(messages: List[Dict[str, Any]], *, first_event_ts: float, after_ts: float = 0.0,
                 captured_user: str = "") -> Optional[Dict[str, Any]]:
    """The turn a crashed process never synced (see module docstring), or None.

    ``messages`` = Hermes messages of the session in insertion order (``get_messages``)."""
    users = [i for i, m in enumerate(messages)
             if m.get("role") == "user" and after_ts < float(m.get("timestamp") or 0.0) <= first_event_ts + SLACK_SECONDS]
    if not users:
        return None
    start = users[-1]
    user = _text(messages[start].get("content")).strip()
    if not user or (captured_user and user == captured_user.strip()):
        return None
    tail: List[Dict[str, Any]] = []
    assistant = ""
    for m in messages[start + 1:]:
        role = m.get("role")
        if role == "user":
            break
        tail.append(m)
        if role == "assistant" and _text(m.get("content")).strip():
            assistant = _text(m.get("content")).strip()
    compact = [{"role": m.get("role"), "content": _text(m.get("content"))[:4096],
                **({"tool_call_id": m["tool_call_id"]} if m.get("tool_call_id") else {})}
               for m in [messages[start], *tail]]
    return {"user": user[:MAX_TEXT], "assistant": assistant[:MAX_TEXT], "messages": compact,
            "message_ids": [m.get("id") for m in [messages[start], *tail] if m.get("id") is not None]}


class StateDbBackfill:
    """Reads ``state.db`` through Hermes' read-only SessionDB (``pan.hermes.compat``)."""

    def __init__(self, state_db: Path, *, reader: Optional[Callable[[Path, str], List[Dict[str, Any]]]] = None) -> None:
        self.state_db = Path(state_db)
        self._reader = reader

    def available(self) -> bool:
        return self.state_db.is_file()

    def messages(self, session_id: str) -> List[Dict[str, Any]]:
        if self._reader is not None:
            return self._reader(self.state_db, session_id)
        from pan.hermes import compat
        return compat.read_session_messages(self.state_db, session_id)

    def missing_turn(self, session_id: str, *, first_event_ts: float, after_ts: float = 0.0,
                     captured_user: str = "") -> Optional[Dict[str, Any]]:
        if not session_id or not self.available():
            return None
        try:
            msgs = self.messages(session_id)
        except Exception as exc:  # locked, missing session, Hermes API change: no backfill
            logger.info("state.db backfill skipped for %s: %s", session_id, exc)
            return None
        return missing_turn(msgs, first_event_ts=first_event_ts, after_ts=after_ts, captured_user=captured_user)
