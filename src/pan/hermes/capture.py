"""Event capture shared by the provider and the plugin hooks (integration spec §4.1, §4.3).

Hermes hooks are process-global and carry only a ``session_id``; the provider knows the rest
(hermes_home, agent_context, cwd). :data:`HUB` joins the two: each :class:`SessionRecorder`
(one per provider instance) binds its session ids to a :class:`SessionBinding`, and hook callbacks
look the binding up by ``session_id``. Events for unknown sessions are dropped, so contexts that
never initialize the provider (or skip capture) produce nothing.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from pan import __version__
from pan.config import CaptureConfig, PanConfig
from pan.events.schema import Actor, AgentEvent, EventType
from pan.events.spool import EventSpool

logger = logging.getLogger(__name__)

_DEDUP_CAPACITY = 4096


def excerpt(value: Any, max_bytes: int) -> str:
    """``value`` as text, cut to ``max_bytes`` UTF-8 bytes (with a marker when cut)."""
    text = value if isinstance(value, str) else ("" if value is None else _to_text(value))
    raw = text.encode("utf-8", "replace")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", "ignore") + f"…[+{len(raw) - max_bytes} bytes]"


def head_tail_excerpt(value: Any, max_bytes: int) -> str:
    """Like :func:`excerpt`, but keeps the first and the last half (M6): long tool output (a deploy
    log, a file listing) often has its key facts at the start *and* its outcome at the end."""
    text = value if isinstance(value, str) else ("" if value is None else _to_text(value))
    raw = text.encode("utf-8", "replace")
    if len(raw) <= max_bytes:
        return text
    half = max_bytes // 2
    head = raw[:half].decode("utf-8", "ignore")
    tail = raw[len(raw) - half:].decode("utf-8", "ignore")
    return f"{head}…[{len(raw) - 2 * half} bytes omitted]…{tail}"


def cap_values(value: Any, max_bytes: int) -> Any:
    """Copy of a JSON-like structure with every string capped to ``max_bytes``."""
    if isinstance(value, str):
        return excerpt(value, max_bytes)
    if isinstance(value, dict):
        return {str(k): cap_values(v, max_bytes) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cap_values(v, max_bytes) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return excerpt(str(value), max_bytes)


def _to_text(value: Any) -> str:
    """Flatten OpenAI-style multimodal content (list of parts) or other objects to text."""
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or f"[{part.get('type', 'part')}]"))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


@dataclass(frozen=True)
class SessionBinding:
    """Where and how events of one session are recorded."""

    spool: EventSpool
    config: CaptureConfig
    agent_context: str = "primary"
    actor: Actor = Actor.MAIN_AGENT
    parent_session_id: Optional[str] = None
    project: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def emit(self, event_type: EventType, session_id: str, content: dict[str, Any], *,
             actor: Optional[Actor] = None, source_refs: Iterable[str] = (),
             parent_session_id: Optional[str] = None) -> bool:
        if self.spool.closed:
            return False
        refs = [f"session:{session_id}", *(r for r in source_refs if r)]
        event = AgentEvent(
            event_type=event_type, session_id=session_id, actor=actor or self.actor, content=content,
            source_refs=list(dict.fromkeys(refs)), parent_session_id=parent_session_id or self.parent_session_id,
            project=self.project, metadata=self.metadata)
        return self.spool.try_append(event)


class CaptureHub:
    """Process-wide registry: session id → binding, shared spools (refcounted), dedup keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionBinding] = {}
        self._spools: dict[Path, list] = {}
        self._seen: OrderedDict[tuple, None] = OrderedDict()

    def acquire_spool(self, path: Path) -> EventSpool:
        key = Path(path).resolve()
        with self._lock:
            entry = self._spools.get(key)
            if entry is None or entry[0].closed:
                entry = self._spools[key] = [EventSpool(key), 0]
            entry[1] += 1
            return entry[0]

    def release_spool(self, spool: EventSpool) -> None:
        with self._lock:
            key = spool.path
            entry = self._spools.get(key)
            if entry is None or entry[0] is not spool:
                spool.close()
                return
            entry[1] -= 1
            if entry[1] <= 0:
                del self._spools[key]
                spool.close()

    def bind(self, session_id: str, binding: SessionBinding) -> None:
        if session_id:
            with self._lock:
                self._sessions[session_id] = binding

    def unbind(self, session_id: str, binding: Optional[SessionBinding] = None) -> None:
        """Remove a session; with ``binding`` only if that exact binding is still the current one."""
        with self._lock:
            if binding is None or self._sessions.get(session_id) is binding:
                self._sessions.pop(session_id, None)

    def lookup(self, session_id: Optional[str]) -> Optional[SessionBinding]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def first_time(self, key: tuple) -> bool:
        """True the first time ``key`` is seen (bounded memory); used to dedupe session_end."""
        with self._lock:
            if key in self._seen:
                return False
            self._seen[key] = None
            if len(self._seen) > _DEDUP_CAPACITY:
                self._seen.popitem(last=False)
            return True

    def forget(self, key: tuple) -> None:
        with self._lock:
            self._seen.pop(key, None)

    def clear(self) -> None:
        """Drop all state (tests)."""
        with self._lock:
            spools = [entry[0] for entry in self._spools.values()]
            self._sessions.clear()
            self._spools.clear()
            self._seen.clear()
        for spool in spools:
            spool.close()


HUB = CaptureHub()

CLOSE = "close"
TURN_END = "turn_end"


def emit_session_end(session_id: str, kind: str, source: str, content: dict[str, Any], *,
                     dedup_key: Optional[tuple] = None, hub: CaptureHub = HUB) -> bool:
    """``session_end`` event, emitted at most once per ``dedup_key`` (default ``(session_id, kind)``;
    ``()`` disables dedup). The provider and the plugin hooks both report the same boundary."""
    binding = hub.lookup(session_id)
    key = (session_id, kind) if dedup_key is None else dedup_key
    if binding is None or (key and not hub.first_time(key)):
        return False
    return binding.emit(EventType.SESSION_END, session_id, {"kind": kind, "source": source, **content},
                        actor=Actor.SYSTEM)


def _compact_message(message: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.get("role")}
    if message.get("content") is not None:
        out["content"] = excerpt(message["content"], max_bytes)
    calls = message.get("tool_calls")
    if calls:
        out["tool_calls"] = [
            {"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
             "arguments": excerpt((c.get("function") or {}).get("arguments"), max_bytes)}
            for c in calls if isinstance(c, dict)]
    for key in ("tool_call_id", "name"):
        if message.get(key):
            out[key] = message[key]
    return out


def new_messages_tail(messages: list[dict[str, Any]], previous_len: Optional[int]) -> list[dict[str, Any]]:
    """Messages added since the last sync. Without a usable previous length (first sync, resumed
    or compressed transcript) the tail starts at the last user message."""
    if previous_len is not None and 0 <= previous_len <= len(messages):
        tail = messages[previous_len:]
    else:
        start = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                start = i
                break
        tail = messages[start:]
    return [m for m in tail if isinstance(m, dict) and m.get("role") != "system"]


class SessionRecorder:
    """Per-provider capture state: owns a spool reference and the provider's session bindings."""

    def __init__(self, spool_path: Path, session_id: str, *, config: PanConfig, agent_context: str = "primary",
                 platform: Optional[str] = None, project: Optional[str] = None,
                 parent_session_id: Optional[str] = None, hermes_version: str = "unknown",
                 hub: CaptureHub = HUB) -> None:
        self.hub = hub
        self.config = config
        self.session_id = session_id
        self.enabled = agent_context not in config.capture.skip_agent_contexts
        self._bound: dict[str, SessionBinding] = {}
        self._synced: dict[str, int] = {}
        self._lock = threading.Lock()
        self.spool = hub.acquire_spool(spool_path) if self.enabled else None
        self._binding = None if self.spool is None else SessionBinding(
            spool=self.spool, config=config.capture, agent_context=agent_context, project=project,
            parent_session_id=parent_session_id or None,
            metadata={"agent_context": agent_context, "platform": platform, "hermes_version": hermes_version,
                      "pan_version": __version__})
        self._bind(session_id)

    @property
    def max_bytes(self) -> int:
        return self.config.capture.tool_result_excerpt_bytes

    def _bind(self, session_id: str, parent_session_id: Optional[str] = None) -> None:
        if not session_id or self._binding is None:
            return
        binding = self._binding if not parent_session_id else replace(self._binding,
                                                                      parent_session_id=parent_session_id)
        self.hub.bind(session_id, binding)
        self._bound[session_id] = binding

    def _emit(self, event_type: EventType, content: dict[str, Any], *, session_id: str = "",
              actor: Actor = Actor.MAIN_AGENT, source_refs: Iterable[str] = ()) -> bool:
        sid = session_id or self.session_id
        binding = self._bound.get(sid) or self._binding
        if binding is None:
            return False
        return binding.emit(event_type, sid, content, actor=actor, source_refs=source_refs)

    # -- provider hooks --------------------------------------------------------------------------

    def turn(self, user: str, assistant: str, *, session_id: str = "",
             messages: Optional[list[dict[str, Any]]] = None, recall: str = "") -> bool:
        """``turn`` event; ``recall`` = the prefetch context PAN injected for this turn (the daemon
        must not mistake answers restating it for new knowledge)."""
        if self._binding is None:
            return False
        sid = session_id or self.session_id
        content: dict[str, Any] = {"user": excerpt(user, 16384), "assistant": excerpt(assistant, 16384)}
        if recall:
            content["recall"] = excerpt(recall, self.max_bytes)
        refs: list[str] = []
        if messages is not None:
            snapshot = list(messages)
            with self._lock:
                tail = new_messages_tail(snapshot, self._synced.get(sid))
                self._synced[sid] = len(snapshot)
            content["messages"] = [_compact_message(m, self.max_bytes) for m in tail]
            refs = [f"tool_call:{m['tool_call_id']}" for m in tail if m.get("role") == "tool" and m.get("tool_call_id")]
        self.hub.forget((sid, CLOSE))  # a resumed session can close again
        return self._emit(EventType.TURN, content, session_id=sid, source_refs=refs)

    def memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]]) -> bool:
        meta = dict(metadata or {})
        sid = str(meta.get("session_id") or "") or self.session_id
        return self._emit(EventType.L1_WRITE, {"action": action, "target": target, "content": content,
                                               "metadata": cap_values(meta, self.max_bytes)}, session_id=sid)

    def compaction(self, messages: list[dict[str, Any]]) -> bool:
        with self._lock:
            self._synced.pop(self.session_id, None)
        return self._emit(EventType.COMPACTION, {"message_count": len(messages or [])}, actor=Actor.SYSTEM)

    def delegation(self, task: str, result: str, child_session_id: str, extra: dict[str, Any]) -> bool:
        content = {"task": excerpt(task, self.max_bytes), "result": excerpt(result, self.max_bytes),
                   "child_session_id": child_session_id or None, **cap_values(extra, self.max_bytes)}
        return self._emit(EventType.DELEGATION, content,
                          source_refs=[f"session:{child_session_id}"] if child_session_id else ())

    def session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                       rewound: bool = False, extra: Optional[dict[str, Any]] = None) -> bool:
        old = self.session_id
        self.session_id = new_session_id or old
        self._bind(self.session_id, parent_session_id or None)
        with self._lock:
            if reset or rewound:
                self._synced.pop(self.session_id, None)
        self.hub.forget((self.session_id, CLOSE))
        content = {"previous_session_id": old, "new_session_id": self.session_id,
                   "parent_session_id": parent_session_id or None, "reset": reset, "rewound": rewound,
                   **cap_values(extra or {}, self.max_bytes)}
        return self._emit(EventType.SESSION_SWITCH, content, session_id=self.session_id, actor=Actor.SYSTEM,
                          source_refs=[f"session:{old}"] if old and old != self.session_id else ())

    def session_end(self, messages: Optional[list[dict[str, Any]]]) -> bool:
        return emit_session_end(self.session_id, CLOSE, "provider", {"message_count": len(messages or [])},
                                hub=self.hub)

    def close(self) -> None:
        for sid, binding in self._bound.items():
            self.hub.unbind(sid, binding)
        self._bound.clear()
        if self.spool is not None:
            self.hub.release_spool(self.spool)
            self.spool = None
            self._binding = None
