"""Memory Plane data model (spec v0.1 §11, §22; integration spec §4.4). Pure Python, no Hermes imports."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Actor(str, Enum):
    USER = "user"
    MAIN_AGENT = "main_agent"
    SUBAGENT = "subagent"
    TOOL = "tool"
    SYSTEM = "system"


class EventType(str, Enum):
    TURN = "turn"
    TOOL_CALL = "tool_call"
    FILE_CHANGE = "file_change"
    L1_WRITE = "l1_write"
    DELEGATION = "delegation"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    SESSION_SWITCH = "session_switch"
    SESSION_END = "session_end"
    COMPACTION = "compaction"
    SKILL_EVENT = "skill_event"


class Lifetime(str, Enum):
    TRANSIENT = "transient"
    SESSION = "session"
    LONG_TERM = "long_term"


class MemoryType(str, Enum):
    USER_PREFERENCE = "user_preference"
    PROJECT_FACT = "project_fact"
    ARCHITECTURE = "architecture"
    DECISION = "decision"
    LEARNING = "learning"
    PROCEDURE = "procedure"
    INCIDENT = "incident"
    ENVIRONMENT = "environment"
    CONFIGURATION = "configuration"
    NOISE = "noise"


class Destination(str, Enum):
    NONE = "none"
    USER = "user"
    CORE = "core"
    WIKI = "wiki"


class CuratorAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    IGNORE = "ignore"
    ESCALATE = "escalate"


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


_ulid_lock = threading.Lock()
_ulid_last = 0


def new_ulid() -> str:
    """Time-ordered 26-char ULID (48-bit ms timestamp + 80 random bits), monotonic within a process:
    ids minted in the same millisecond increment the previous one instead of drawing new randomness."""
    global _ulid_last
    with _ulid_lock:
        value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
        if value <= _ulid_last:
            value = _ulid_last + 1
        _ulid_last = value
    return "".join(_CROCKFORD[(value >> shift) & 31] for shift in range(125, -1, -5))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class AgentEvent:
    event_type: EventType
    session_id: str
    actor: Actor
    content: dict[str, Any] = field(default_factory=dict)
    source_refs: list[str] = field(default_factory=list)
    parent_session_id: str | None = None
    project: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_ulid)
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["actor"] = self.actor.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentEvent":
        d = dict(d)
        d["event_type"] = EventType(d["event_type"])
        d["actor"] = Actor(d["actor"])
        return cls(**d)


@dataclass
class MemoryClassification:
    should_remember: bool
    relevance: float
    importance: float
    confidence: float
    lifetime: Lifetime
    type: MemoryType
    destination: Destination


@dataclass
class MemoryCandidate:
    event_ids: list[str]
    classification: MemoryClassification
    normalized_claims: list[str]
    retrieval_query: str
    # PAN additions (M3), all optional:
    id: str = ""                                               # deterministic: "<episode id>:<type>"
    session_id: str = ""
    title: str = ""                                            # suggested page title
    tags: list[str] = field(default_factory=list)
    claim_evidence: list[str] = field(default_factory=list)    # per claim: "observed" | "inferred" | "reported" (T3)
    context: str = ""                                          # short human-readable context (e.g. the question)
    rationale: str = ""                                        # which classifier rule fired
    # M5 additions, optional:
    subject: str = ""                                          # what the facts are about ("vLLM server")
    supersedes: list[str] = field(default_factory=list)        # texts this candidate explicitly replaces


@dataclass
class CuratorDecision:
    candidate_id: str
    action: CuratorAction
    target_pages: list[str]
    rationale: str
    confidence: float
    provenance: list[str]
    patch: str = ""
    evidence: str = ""   # PAN addition: "observed" | "inferred" | "reported" | "mixed" (claims written by this decision)


def to_jsonable(value: Any) -> Any:
    """Dataclasses / enums / containers → plain JSON types (for logs and golden files)."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
