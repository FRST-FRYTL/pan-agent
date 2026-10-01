"""Group claimed spool events into episodes (integration spec §4.5): the unit the classifier sees.

An episode is one turn of one session: the ``tool_call`` / ``file_change`` events emitted while the
turn ran plus the ``turn`` event (user + assistant text) that ``sync_turn`` appends after it.

Event order in the spool (M1): hooks fire *during* the turn, ``sync_turn`` runs *after* it, and the
plugin ``on_session_end`` hook (``session_end`` kind ``turn_end``) may land before or after the
``turn`` event. Therefore:

- tool events collect in an open group; a ``turn`` event completes the oldest group of its session
  that has no turn yet (or forms an episode of its own);
- ``turn_end`` starts a new group for later tool events (the finished turn's group still waits for
  its ``turn`` event);
- ``session_end`` kind ``close`` and ``subagent_stop`` close every waiting group of the session;
- a group whose newest event is ``idle_seconds`` old (default 30 min) is closed without a turn
  (crash, interrupted turn, subagent sessions that never sync a turn);
- ``flush=True`` closes everything (replay, tests).

Groups that are not closed are returned as ``pending`` — the daemon releases them back to the spool.
Everything else (``l1_write``, ``session_switch``, ``compaction``, …) is returned as ``markers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from pan.events.schema import AgentEvent, EventType

DEFAULT_IDLE_SECONDS = 30 * 60
GROUPED_TYPES = frozenset({EventType.TOOL_CALL, EventType.FILE_CHANGE})


def parse_ts(ts: str) -> float:
    """ISO-8601 (``Z`` or offset) → epoch seconds; 0.0 when unparsable."""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# PAN read tools whose results are recalled wiki knowledge (not new observations).
RECALL_TOOLS = frozenset({"memory_search", "memory_read"})


@dataclass
class Episode:
    session_id: str
    events: List[AgentEvent] = field(default_factory=list)
    closed_by: str = ""  # "turn" | "close" | "subagent_stop" | "idle" | "flush"

    @property
    def id(self) -> str:
        """Id of the first event: stable across re-runs over the same events."""
        return self.events[0].id if self.events else ""

    @property
    def event_ids(self) -> List[str]:
        return [e.id for e in self.events]

    @property
    def turn(self) -> Optional[AgentEvent]:
        return next((e for e in self.events if e.event_type is EventType.TURN), None)

    @property
    def tool_calls(self) -> List[AgentEvent]:
        return [e for e in self.events if e.event_type is EventType.TOOL_CALL]

    @property
    def file_changes(self) -> List[AgentEvent]:
        return [e for e in self.events if e.event_type is EventType.FILE_CHANGE]

    @property
    def user_text(self) -> str:
        turn = self.turn
        return str((turn.content.get("user") if turn else "") or "")

    @property
    def assistant_text(self) -> str:
        turn = self.turn
        return str((turn.content.get("assistant") if turn else "") or "")

    @property
    def recalled_text(self) -> str:
        """What the agent read from PAN memory during this episode: ``memory_search`` /
        ``memory_read`` results and the turn's prefetch context (``turn.content["recall"]``)."""
        parts = [str(e.content.get("result_excerpt") or "") for e in self.tool_calls
                 if e.content.get("tool") in RECALL_TOOLS]
        turn = self.turn
        if turn is not None and turn.content.get("recall"):
            parts.append(str(turn.content["recall"]))
        return "\n".join(p for p in parts if p)

    @property
    def project(self) -> Optional[str]:
        return next((e.project for e in self.events if e.project), None)

    @property
    def last_ts(self) -> float:
        return max((parse_ts(e.ts) for e in self.events), default=0.0)


@dataclass
class Grouping:
    episodes: List[Episode] = field(default_factory=list)   # complete, ordered by first event id
    pending: List[AgentEvent] = field(default_factory=list)  # open groups: release, retry later
    markers: List[AgentEvent] = field(default_factory=list)  # non-content events: done after logging


def _session_end_kind(event: AgentEvent) -> str:
    return str(event.content.get("kind") or "")


def group_events(events: Iterable[AgentEvent], *, now: float, idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 flush: bool = False) -> Grouping:
    """Split ``events`` (any order; sorted by id here) into episodes, pending events and markers."""
    out = Grouping()
    # session → groups; each group is an Episode without turn yet. ``sealed`` groups take no new tools.
    waiting: Dict[str, List[Episode]] = {}
    sealed: Dict[str, set[int]] = {}

    def close(session: str, reason: str) -> None:
        for ep in waiting.pop(session, []):
            ep.closed_by = reason
            out.episodes.append(ep)
        sealed.pop(session, None)

    for event in sorted(events, key=lambda e: e.id):
        sid = event.session_id
        groups = waiting.setdefault(sid, [])
        if event.event_type in GROUPED_TYPES:
            if groups and id(groups[-1]) not in sealed.get(sid, set()):
                groups[-1].events.append(event)
            else:
                groups.append(Episode(sid, [event]))
        elif event.event_type is EventType.TURN:
            if groups:
                ep = groups.pop(0)
                sealed.get(sid, set()).discard(id(ep))
                ep.events.append(event)
            else:
                ep = Episode(sid, [event])
            ep.closed_by = "turn"
            out.episodes.append(ep)
        elif event.event_type is EventType.SESSION_END and _session_end_kind(event) != "close":
            if groups:
                sealed.setdefault(sid, set()).add(id(groups[-1]))
            out.markers.append(event)
        elif event.event_type is EventType.SESSION_END or event.event_type is EventType.SUBAGENT_STOP:
            out.markers.append(event)
            close(sid, "close" if event.event_type is EventType.SESSION_END else "subagent_stop")
        else:
            out.markers.append(event)

    for sid in list(waiting):
        for ep in waiting[sid]:
            if flush:
                ep.closed_by = "flush"
            elif now - ep.last_ts >= idle_seconds:
                ep.closed_by = "idle"
            else:
                out.pending.extend(ep.events)
                continue
            out.episodes.append(ep)
    out.episodes.sort(key=lambda ep: ep.id)
    return out


def previous_turn(episodes: Sequence[Episode], index: int) -> Optional[Episode]:
    """The closest earlier episode of the same session that has a turn (context for confirmations)."""
    session = episodes[index].session_id
    for ep in reversed(episodes[:index]):
        if ep.session_id == session and ep.turn is not None:
            return ep
    return None
