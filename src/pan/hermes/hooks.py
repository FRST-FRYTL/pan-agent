"""General plugin hook subscriber (integration spec §4.3).

Registered from ``pan.hermes.register(ctx)``; for a memory provider Hermes' ``_ProviderCollector``
forwards ``ctx.register_hook`` to the plugin manager as a fallback hook group of this source.
Every callback accepts ``**kwargs`` (Hermes narrows payloads for narrow signatures), never raises,
and does one spool INSERT at most per event. Sessions unknown to :data:`pan.hermes.capture.HUB`
(never initialized by the provider, or capture skipped for cron/flush) are ignored.

Kwargs relied on (verified against the pinned Hermes; see tests/contract/test_hook_contract.py):

- ``post_tool_call``: tool_name, args, result, session_id, tool_call_id, turn_id, task_id,
  duration_ms, status, error_type, error_message (``model_tools._emit_post_tool_call_hook``)
- ``subagent_start``: parent_session_id, parent_turn_id, child_session_id, child_subagent_id,
  child_role, child_goal (``tools/delegate_tool.py``)
- ``subagent_stop``: parent_session_id, child_session_id, child_role, child_summary, child_status,
  tool_call_history, duration_ms (``tools/delegate_tool_results._fire_subagent_stop_hooks``)
- ``on_session_end``: session_id, turn_id, completed, failed, interrupted, turn_exit_reason, model,
  platform — fired at the end of EVERY ``run_conversation`` (``agent/turn_finalizer.py``) and for
  interrupted turns (``reason``), so PAN records it as ``session_end`` kind ``turn_end``
- ``on_session_finalize``: session_id, platform, reason (``hermes_cli.lifecycle.finalize_session``)
  — a real session boundary, recorded as kind ``close`` (deduped with the provider's on_session_end)
- ``on_skill_lifecycle``: action, skill_name, provenance, session_id, task_id, use_count, reused,
  reuse_after_patch (``tools/skill_usage._emit_skill_lifecycle``)
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import replace
from typing import Any, Callable

from pan.events.schema import Actor, EventType
from pan.hermes.capture import CLOSE, HUB, TURN_END, CaptureHub, cap_values, emit_session_end, excerpt, head_tail_excerpt

logger = logging.getLogger(__name__)

# Hermes file tools that modify files (tools/file_tools.py registry names).
FILE_WRITE_TOOLS = frozenset({"write_file", "patch"})

_V4A_OPS = [
    ("update", re.compile(r"^\*\*\*\s*Update\s+File:\s*(.+?)\s*$", re.M)),
    ("add", re.compile(r"^\*\*\*\s*Add\s+File:\s*(.+?)\s*$", re.M)),
    ("delete", re.compile(r"^\*\*\*\s*Delete\s+File:\s*(.+?)\s*$", re.M)),
]
_V4A_MOVE = re.compile(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+?)\s*$", re.M)


def _safe(fn: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> None:
        try:
            fn(**kwargs)
        except Exception:
            logger.debug("PAN hook %s failed", fn.__name__, exc_info=True)
        return None
    return wrapper


def _args_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def file_changes(tool_name: str, args: Any) -> list[dict[str, str]]:
    """``[{"path", "op", ...}]`` for file-modifying tool calls (write_file, patch replace / V4A)."""
    if tool_name not in FILE_WRITE_TOOLS:
        return []
    a = _args_dict(args)
    if tool_name == "write_file":
        return [{"path": str(a["path"]), "op": "write"}] if a.get("path") else []
    patch_text = a.get("patch")
    if a.get("mode") == "patch" or (patch_text and not a.get("path")):
        text = str(patch_text or "")
        changes = [{"path": m.group(1), "op": op} for op, rx in _V4A_OPS for m in rx.finditer(text)]
        changes += [{"path": m.group(2), "op": "move", "from": m.group(1)} for m in _V4A_MOVE.finditer(text)]
        return changes
    return [{"path": str(a["path"]), "op": "replace"}] if a.get("path") else []


def make_hooks(hub: CaptureHub = HUB) -> dict[str, Callable[..., None]]:
    """Hook name → callback, bound to ``hub`` (tests pass their own hub)."""

    @_safe
    def on_post_tool_call(**kw: Any) -> None:
        session_id = kw.get("session_id") or ""
        binding = hub.lookup(session_id)
        if binding is None:
            return
        limit = binding.config.tool_result_excerpt_bytes
        tool = str(kw.get("tool_name") or "")
        call_id = kw.get("tool_call_id") or ""
        result = kw.get("result")
        result_text = result if isinstance(result, str) else excerpt(result, limit * 4)
        content = {
            "tool": tool, "args": cap_values(_args_dict(kw.get("args")) or kw.get("args"), limit),
            "status": kw.get("status") or "unknown", "error_type": kw.get("error_type"),
            "error_message": excerpt(kw.get("error_message"), limit) if kw.get("error_message") else None,
            "result_excerpt": head_tail_excerpt(result_text, limit), "result_bytes": len(result_text.encode("utf-8", "replace")),
            "duration_ms": kw.get("duration_ms"), "tool_call_id": call_id or None,
            "turn_id": kw.get("turn_id") or None, "task_id": kw.get("task_id") or None,
        }
        refs = [f"tool_call:{call_id}"] if call_id else []
        binding.emit(EventType.TOOL_CALL, session_id, content, actor=Actor.TOOL, source_refs=refs)
        if kw.get("status") == "ok":
            for change in file_changes(tool, kw.get("args")):
                binding.emit(EventType.FILE_CHANGE, session_id, {**change, "tool": tool, "tool_call_id": call_id or None},
                             actor=Actor.TOOL, source_refs=[*refs, f"path:{change['path']}"])

    @_safe
    def on_subagent_start(**kw: Any) -> None:
        parent = kw.get("parent_session_id") or ""
        binding = hub.lookup(parent)
        if binding is None:
            return
        child = kw.get("child_session_id") or ""
        if child:
            hub.bind(child, replace(binding, actor=Actor.SUBAGENT, parent_session_id=parent,
                                    metadata={**binding.metadata, "agent_context": "subagent"}))
        content = {"child_session_id": child or None, "child_subagent_id": kw.get("child_subagent_id"),
                   "child_role": kw.get("child_role"), "task": excerpt(kw.get("child_goal"), 8192),
                   "parent_turn_id": kw.get("parent_turn_id") or None}
        binding.emit(EventType.SUBAGENT_START, child or parent, content, actor=Actor.SUBAGENT,
                     source_refs=[f"session:{parent}"], parent_session_id=parent)

    @_safe
    def on_subagent_stop(**kw: Any) -> None:
        parent = kw.get("parent_session_id") or ""
        child = kw.get("child_session_id") or ""
        child_binding = hub.lookup(child)
        binding = child_binding or hub.lookup(parent)
        if binding is None:
            return
        limit = binding.config.tool_result_excerpt_bytes
        content = {"child_session_id": child or None, "child_role": kw.get("child_role"),
                   "status": kw.get("child_status"), "summary": excerpt(kw.get("child_summary"), 16384),
                   "tool_call_history": cap_values(list(kw.get("tool_call_history") or [])[:200], limit),
                   "duration_ms": kw.get("duration_ms"), "parent_turn_id": kw.get("parent_turn_id") or None}
        binding.emit(EventType.SUBAGENT_STOP, child or parent, content, actor=Actor.SUBAGENT,
                     source_refs=[f"session:{parent}"] if parent else (), parent_session_id=parent or None)
        if child_binding is not None and child_binding.actor is Actor.SUBAGENT:
            hub.unbind(child, child_binding)

    @_safe
    def on_session_end(**kw: Any) -> None:
        session_id = kw.get("session_id") or ""
        turn_id = kw.get("turn_id") or ""
        content = {key: kw.get(key) for key in ("turn_id", "completed", "failed", "interrupted",
                                                "turn_exit_reason", "reason", "model", "platform")
                   if kw.get(key) is not None}
        emit_session_end(session_id, TURN_END, "on_session_end", content,
                         dedup_key=(session_id, TURN_END, turn_id) if turn_id else (), hub=hub)

    @_safe
    def on_session_finalize(**kw: Any) -> None:
        session_id = kw.get("session_id") or ""
        content = {key: kw.get(key) for key in ("platform", "reason") if kw.get(key) is not None}
        emit_session_end(session_id, CLOSE, "on_session_finalize", content, hub=hub)

    @_safe
    def on_skill_lifecycle(**kw: Any) -> None:
        session_id = kw.get("session_id") or ""
        binding = hub.lookup(session_id)
        if binding is None:
            return
        content = {key: kw.get(key) for key in ("action", "skill_name", "provenance", "task_id", "use_count",
                                                "reused", "reuse_after_patch")}
        binding.emit(EventType.SKILL_EVENT, session_id, cap_values(content, 1024), actor=Actor.SYSTEM,
                     source_refs=[f"skill:{kw.get('skill_name')}"] if kw.get("skill_name") else ())

    return {
        "post_tool_call": on_post_tool_call,
        "subagent_start": on_subagent_start,
        "subagent_stop": on_subagent_stop,
        "on_session_end": on_session_end,
        "on_session_finalize": on_session_finalize,
        "on_skill_lifecycle": on_skill_lifecycle,
    }


HOOKS = make_hooks()
HOOK_NAMES = tuple(HOOKS)


def register_hooks(ctx: Any) -> int:
    """Subscribe PAN's hooks through ``ctx.register_hook``; returns how many were registered."""
    register = getattr(ctx, "register_hook", None)
    if not callable(register):
        logger.debug("PAN: plugin context has no register_hook; event capture limited to provider hooks")
        return 0
    count = 0
    for name, callback in HOOKS.items():
        try:
            register(name, callback)
            count += 1
        except Exception as exc:
            logger.warning("PAN: could not register hook %s: %s", name, exc)
    return count
