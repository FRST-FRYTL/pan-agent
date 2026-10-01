"""PanMemoryProvider — PAN's in-process memory provider for Hermes (integration spec §4.1).

Captures events into the durable spool (M1) and serves the read path — memory_search, memory_read,
prefetch, system prompt block (M2). Curation happens in pan-memoryd. The provider must never block or
raise into the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from pan import __version__
from pan.config import PanConfig, load_config
from pan.hermes import compat
from pan.hermes.capture import SessionRecorder
from pan.memory.claims import stem, token_set
from pan.paths import PanPaths

logger = logging.getLogger(__name__)

PROVIDER_NAME = "pan"
# The schemas go out with every request: no page-type filter (the agent never needed it; the reader
# still accepts ``type``) and no bounds the reader enforces anyway.
MEMORY_SEARCH_SCHEMA: Dict[str, Any] = {
    "name": "memory_search",
    "description": (
        "Search PAN's project memory wiki (facts, decisions, environment, systems, learnings from "
        "earlier sessions). Use before re-deriving or asking about project facts. Returns page ids, "
        "titles and the best-matching current facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
}

MEMORY_READ_SCHEMA: Dict[str, Any] = {
    "name": "memory_read",
    "description": ("Read a PAN wiki page by id (from memory_search or <memory-context>) or path: "
                    "its status and current facts, superseded ones listed last."),
    "parameters": {
        "type": "object",
        "properties": {
            "page": {"type": "string", "description": "Page id or wiki path."},
            "section": {"type": "string", "description": "Optional heading; return only that section."},
        },
        "required": ["page"],
    },
}

TOOL_NAMES = frozenset({MEMORY_SEARCH_SCHEMA["name"], MEMORY_READ_SCHEMA["name"]})

# Wording matters (seen in bench runs): "to record or correct a fact, just state it (or
# use the memory tool)" made Qwen3.8 hunt for a non-existent `pan_memory_save` tool through Hermes'
# Tool Search bridge until max-iterations; memory_read was first tried via `tool_call`. So: name
# the two tools, say they are called directly, and that PAN has no other tools. The agent keeps
# using its memory tool as usual (M6 it1: "stated facts need no tool call" made it stop saving
# while the rule classifier missed German and personal facts — ADR-007 (b) rejected); PAN
# captures on top and dedupes/reconciles. Static text — the block is part of the cached prompt.
# Told only "the user's statement wins", the agent preferred an older USER.md entry over a newer wiki
# update, hence "Newest wins". The markers ("(noted DATE)", "Assistant reported:" …) are explained
# where they appear (``marker_notes``: prefetch, memory_read, memory_search), not here: this block is
# on every request of every session. Per-turn material goes into prefetch, never here.
SYSTEM_PROMPT_BLOCK = (
    "PAN project memory: a curated wiki of facts, decisions, environment details and learnings from "
    "earlier sessions, updated automatically in the background from every conversation, including "
    "tool results. Your memory tool works as usual. When you need project knowledge you do not have, "
    "call memory_search (find pages) or memory_read (open a page by id); both are regular tools in "
    "your tool list, so call them directly by name. Relevant excerpts may already be in the user "
    "message inside <memory-context>. Before saying there is no record, check the user profile and "
    "memory above and open a listed page that may hold it. Newest wins: what the user says now beats "
    "memory; a later wiki fact beats an older USER.md/MEMORY.md entry. For advice, apply "
    "the user's remembered constraints, goals and preferences. For remembered facts use only values memory "
    "states or that follow from combining its facts (never invent them); keep status words as "
    "written (an idea is not a plan, proposed is not accepted). PAN has no other tools; never edit "
    "wiki files directly."
)


ACTIVE_PREFERENCES_HEADER = ("Active user preferences and constraints (follow each one that bears on "
                             "this request, including any format or length it names):")
ACTIVE_PREFERENCES_BUDGET = 500
_PREF_WRAPPER = re.compile(r'^User preference \(own words\):\s*"?(.*?)"?\s*$', re.S)
# How to answer (language, length, tone …) applies to every turn: ranked after entries that share words
# with the question, before the rest.
_PREF_STYLE = re.compile(r"\b(?:answers?|replies|responses?|language|tone|style|format|words?|sentences?|"
                         r"emojis?|english|german|concise|brief|short|formal|informal)\b", re.I)
_PREF_GENERIC = frozenset(stem(w) for w in """user users prefer prefers preference like likes want wants always never
please love loves enjoy enjoys dislike dislikes hate hates fan favourite favorite avoid avoids""".split())
# A quote without its context ("I think I'm sticking with this one from now on.") or talk about the
# conversation ("I recently told you about …") says nothing the agent can apply.
_PREF_FIRST_PERSON = re.compile(r"^\W*(?:I|I'm|I've|I'd|I'll|We|We're)\b")
_PREF_DEICTIC = re.compile(r"\b(?:this|that|these|those|it|one)\b", re.I)
_PREF_META = re.compile(r"^\W*(?:I|We)\s+(?:\w+\s+){0,2}(?:told|mentioned|asked|showed|said)(?:\s+to)?\s+you\b", re.I)
_PREF_FILLER = frozenset(stem(w) for w in """think guess one now going gonna really actually sure thing stuff way
time today yeah well still anyway""".split())
_PREF_NEGATION = re.compile(r"\b(?:no|not|never|cannot|avoids?|dislikes?|hates?)\b|n't\b", re.I)


def _negated(text: str) -> bool:
    return bool(_PREF_NEGATION.search(text))


def _pref_fragment(text: str) -> bool:
    """``text`` does not read as a statement: unbalanced quotes (a mangled quote), a question, talk
    about the conversation, or a first-person remark whose only object is "this/that/it"."""
    if text.count('"') % 2 or text.rstrip().endswith("?") or _PREF_META.search(text):
        return True
    if _PREF_FIRST_PERSON.search(text) and _PREF_DEICTIC.search(text):
        return len(token_set(text) - _PREF_FILLER) < 3
    return False


def format_preferences(prefs: List[str], query: str = "", budget: int = ACTIVE_PREFERENCES_BUDGET) -> str:
    """The active-preferences block: statements only, near-duplicates dropped (the entry that says
    more is kept), most relevant to ``query`` first (shared content words, then how-to-answer entries),
    as many as fit ``budget``."""
    cands = [(p, token_set(p) - _PREF_GENERIC) for p in prefs if p and not _pref_fragment(p)]
    keep = []
    for i, (p, toks) in enumerate(cands):
        dup = False
        for j, (q, other) in enumerate(cands):
            if i == j or min(len(toks), len(other)) < 2 or _negated(p) != _negated(q):
                continue
            shared = len(toks & other) / min(len(toks), len(other))
            if shared >= 0.8 and (len(other), j) > (len(toks), i):
                dup = True   # the other entry states this one (and more, or is newer)
                break
        if not dup:
            keep.append((i, p, toks))
    want = token_set(query or "") - _PREF_GENERIC
    ranked = sorted(keep, key=lambda k: (-(len(want & k[2]) + (0.5 if _PREF_STYLE.search(k[1]) else 0)), k[0]))
    lines = [ACTIVE_PREFERENCES_HEADER]
    used = len(lines[0])
    for _, p, _ in ranked:
        line = f"- {p}"
        if used + len(line) + 1 > budget:
            continue
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if len(lines) > 1 else ""


class PanMemoryProvider(MemoryProvider):
    _config: PanConfig = PanConfig()
    _reader: Any = None          # pan.memory.reader.Reader, opened in initialize (M2 read path)
    _prompt_block: Optional[str] = None

    def __init__(self) -> None:
        self._paths: Optional[PanPaths] = None
        self._session_id = ""
        self._agent_context = "primary"
        self._capture = False
        self._recall: Dict[str, str] = {}  # session id → prefetch context of the running turn

    # -- identity / availability ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return True  # local-only; no credentials or network needed

    # -- lifecycle -----------------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._paths = PanPaths.for_home(kwargs["hermes_home"])
        self._paths.ensure()
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context") or "primary"
        self._config = load_config(self._paths)
        self._capture = self._start_capture(session_id, kwargs)
        compat.check_hermes_pin()
        self._open_reader()
        logger.info("PAN %s initialized (session=%s, context=%s, capture=%s)",
                    __version__, session_id, self._agent_context, self._capture)

    def system_prompt_block(self) -> str:
        """Static pointer + wiki overview from index.md (≤ 800 chars); computed once per provider."""
        if self._prompt_block is None:
            block = SYSTEM_PROMPT_BLOCK
            try:
                overview = self._reader.wiki_overview() if self._reader is not None else ""
            except Exception:  # never break prompt building
                logger.debug("PAN wiki overview failed", exc_info=True)
                overview = ""
            if overview:
                block += "\n\nWiki pages:\n" + overview
            self._prompt_block = block
        return self._prompt_block

    def shutdown(self) -> None:
        self._stop_capture()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # -- tools ---------------------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMORY_SEARCH_SCHEMA, MEMORY_READ_SCHEMA]

    # -- read path (M2) ------------------------------------------------------------------------

    def _open_reader(self) -> None:
        """Open the read-only wiki/index reader. Never rebuilds; a missing index just means no recall."""
        from pan.memory.reader import Reader

        if self._reader is not None:
            self._reader.close()
        self._reader = Reader(self._paths, config=self._config.retrieval) if self._paths is not None else None
        self._prompt_block = None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name not in TOOL_NAMES:
            return json.dumps({"error": f"unknown PAN tool: {tool_name}"})
        if self._reader is None:
            return json.dumps({"error": "PAN memory provider is not initialized"})
        args = args if isinstance(args, dict) else {}
        try:
            if tool_name == "memory_search":
                result = self._reader.search(args.get("query", ""), type=args.get("type"),
                                             limit=args.get("limit", 5))
            else:
                result = self._reader.read(args.get("page", ""), section=args.get("section") or None)
        except Exception as exc:  # tools must answer with JSON, never raise into the loop
            logger.warning("PAN %s failed: %s", tool_name, exc, exc_info=True)
            result = {"error": f"{tool_name} failed: {exc}"}
        return json.dumps(result, ensure_ascii=False, default=str)

    # -- recall --------------------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "", **kwargs) -> str:
        """Budgeted FTS recall for the upcoming turn; "" when nothing scores above the threshold."""
        if self._reader is None:
            return ""
        prefs = self._active_preferences(query)
        try:
            cfg = self._config.retrieval
            budget = cfg.prefetch_budget_chars - (len(prefs) + 2 if prefs else 0)
            out = self._reader.prefetch(query, budget_chars=budget, min_score=cfg.prefetch_min_score)
        except Exception:
            logger.debug("PAN prefetch failed", exc_info=True)
            out = ""
        if prefs:
            out = prefs + ("\n\n" + out if out else "")
        # Remembered for the turn event (``recall``): Hermes calls prefetch at turn start and
        # sync_turn at its end, for the same session.
        self._recall[session_id or self._session_id] = out
        return out

    def _active_preferences(self, query: str = "") -> str:
        """Standing preferences from USER.md, repeated on the user turn (M6, benchmark runs: both systems
        obeyed "answer in English" / word limits poorly although USER.md was in the system prompt).
        The user message is not part of the cached prefix, so this costs no cache hits. Entries are
        ordered by relevance to ``query`` so the relevant ones fit the budget; near-duplicates and
        fragments that do not read as a statement are left out."""
        if self._paths is None:
            return ""
        try:
            from pan.memory.l1 import is_preference_entry
            text = (self._paths.hermes_home / "memories" / "USER.md").read_text(encoding="utf-8")
        except Exception:
            return ""
        entries = [e.strip() for e in text.split("§") if e.strip()]
        prefs = [" ".join(_PREF_WRAPPER.sub(r"\1", e).split()) for e in entries if is_preference_entry(e)]
        return format_preferences(prefs, query)

    # -- capture (M1: append AgentEvents to the spool; never blocks or raises) ----------------------

    _recorder: Optional[SessionRecorder] = None

    def _start_capture(self, session_id: str, kwargs: Dict[str, Any]) -> bool:
        """Open the spool and bind this session for the hooks; False when capture is skipped or fails."""
        self._stop_capture()  # re-initialize: release the previous spool reference
        try:
            config = self._config
            platform = kwargs.get("platform")
            cwd = kwargs.get("cwd") or (os.getcwd() if platform == "cli" else None)
            self._recorder = SessionRecorder(
                self._paths.events_db, session_id, config=config, agent_context=self._agent_context,
                platform=platform, project=_project_root(cwd), parent_session_id=kwargs.get("parent_session_id"),
                hermes_version=compat.hermes_version())
            return self._recorder.enabled
        except Exception as exc:
            logger.warning("PAN event capture disabled: %s", exc)
            self._recorder = None
            return False

    def _stop_capture(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            self._record(recorder.close)

    def _record(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.debug("PAN capture %s failed", getattr(fn, "__name__", fn), exc_info=True)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        recall = self._recall.pop(session_id or self._session_id, "")
        if self._recorder is not None:
            self._record(self._recorder.turn, user_content, assistant_content, session_id=session_id,
                         messages=messages, recall=recall)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if self._recorder is not None:
            self._record(self._recorder.memory_write, action, target, content, metadata)

    def on_pre_compress(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        if self._recorder is not None:
            self._record(self._recorder.compaction, messages)
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if self._recorder is not None:
            self._record(self._recorder.delegation, task, result, child_session_id, kwargs)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        self._session_id = new_session_id
        if self._recorder is not None:
            self._record(self._recorder.session_switch, new_session_id, parent_session_id=parent_session_id,
                         reset=reset, rewound=rewound, extra=kwargs)

    def on_session_end(self, messages: List[Dict[str, Any]], **kwargs) -> None:
        if self._recorder is not None:
            self._record(self._recorder.session_end, messages)


def _project_root(cwd: Optional[str]) -> Optional[str]:
    """Git repository root containing ``cwd`` (else ``cwd`` itself); None without a cwd."""
    if not cwd:
        return None
    try:
        start = Path(cwd).expanduser().resolve()
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return str(candidate)
        return str(start)
    except OSError:
        return str(cwd)
