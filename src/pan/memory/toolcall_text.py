"""Tool calls printed as text (M5): detect and strip them before claims are extracted.

Local models sometimes *print* a tool call instead of emitting a structured one (seen in benchmark runs,
Qwen3.8-27B): the assistant's final text is then e.g.

    memory replace: content="vLLM server runs on port 8010.", old_text="port 8000", target=memory
    Memory (user):
    <tool_call>
    <function=memory>
    <parameter=action>
    add
    </parameter>
    </function>
    </tool_call>
    {"name": "terminal", "arguments": {"command": "nvidia-smi"}}
    memory_search(query="vllm port")

None of this is knowledge; curated into the wiki it becomes junk pages ("Memory replace:
content=…"). :func:`strip_tool_call_text` removes such spans and reports whether it found any:

1. **markup blocks** — ``<tool_call>…</tool_call>``, ``<function=…>…</function>``,
   ``<parameter=…>…</parameter>``, ``<invoke …>``/``<tool_use>``/``<function_call>`` blocks, special
   tokens like ``<|tool_call_begin|>`` and ``[TOOL_CALLS] …`` (an unclosed block runs to the end);
2. **JSON call objects** (bare or fenced) — objects with ``name`` + ``arguments``/``parameters``/
   ``args``/``input``, ``function: {name, arguments}``, ``tool_calls: […]``, ``{"tool": <known
   tool>, …}`` or the memory tool's own shape (``action`` + ``target``/``content``/``old_text``);
3. **call lines** — a line that is a call ``name(arg=…)`` / ``functions.name(…)`` of a known tool,
   or starts with a known tool name (optionally followed by an action word) and continues with
   ``key=value`` pairs, a JSON object, or only a parenthesised target and a colon
   (``Memory (user):``). ``Memory: 128 GB`` stays (no call arguments).

Detection keys on tool names Hermes ships (:data:`KNOWN_TOOLS`) plus generic call syntax, so it
works without the session's tool list.
"""

from __future__ import annotations

import json
import re
from typing import Tuple

KNOWN_TOOLS = frozenset("""
memory memory_search memory_read session_search terminal process read_file write_file patch search_files
web_search web_extract web_crawl browser_navigate browser_click browser_type browser_snapshot browser_scroll
browser_back browser_press browser_close browser_vision browser_get_images browser_console delegate_task todo
execute_code skill_view skills_list skill_manage clarify vision_analyze text_to_speech image_generate cronjob
send_message mixture_of_agents file shell bash python run_command apply_patch
""".split())
_ARG_NAMES = ("content", "old_text", "new_text", "target", "action", "query", "command", "path", "page",
              "section", "limit", "type", "file_path", "pattern", "url", "code", "task", "args", "arguments")

_BLOCK = re.compile(
    r"<(?P<tag>tool_call|tool_calls|tool_use|function_call|function_calls|invoke|antml:invoke)\b[^>]*>"
    r".*?(?:</(?P=tag)>|\Z)", re.S | re.I)
_FUNCTION = re.compile(r"<function\s*=\s*[\w.-]+\s*>.*?(?:</function>|\Z)", re.S | re.I)
_PARAMETER = re.compile(r"<parameter\s*=\s*[\w.-]+\s*>.*?(?:</parameter>|\Z)", re.S | re.I)
_LONE_TAG = re.compile(r"</?(?:tool_call|tool_calls|tool_use|function_call|function_calls|function|parameter|"
                       r"invoke|arguments)\b[^>]*>|<\|[^|>\n]{0,40}\|>", re.I)
_MISTRAL = re.compile(r"\[TOOL_CALLS\].*\Z", re.S)
_SPECIAL_BLOCK = re.compile(r"<\|tool_calls?(?:_section)?_begin\|>.*?(?:<\|tool_calls?(?:_section)?_end\|>|\Z)", re.S)
_FENCE = re.compile(r"```[\w-]*\s*\n?(?P<body>.*?)```", re.S)
_TOOL_ALT = "|".join(sorted((re.escape(t) for t in KNOWN_TOOLS), key=len, reverse=True))
_CALL_LINE = re.compile(rf"^\W{{0,3}}(?:functions\.)?(?P<name>{_TOOL_ALT})\s*\((?P<args>.*)\)\s*[.;]?\s*$", re.I)
_GENERIC_CALL = re.compile(r"^\W{0,3}(?:functions\.)?[a-z_][a-z0-9_]*\s*\(\s*[a-z_]+\s*=.*\)\s*$", re.I)
_TOOL_LEAD = re.compile(
    rf"^\W{{0,3}}(?:functions\.)?(?P<name>{_TOOL_ALT})\b(?:\s*[.:]\s*|\s+)?"
    r"(?:(?:add|replace|remove|search|read|write|call|run|exec|query|open)\b)?\s*(?P<rest>.*)$", re.I)
_KV = re.compile(r"\b(?P<key>[a-z_][a-z0-9_]*)\s*[=:]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", re.I)
_PAREN_TARGET = re.compile(r"^\(\s*[\w.-]+\s*\)\s*:?\s*$")


def _is_call_object(obj: object) -> bool:
    if isinstance(obj, list):
        return bool(obj) and all(_is_call_object(o) for o in obj)
    if not isinstance(obj, dict):
        return False
    keys = {str(k).lower() for k in obj}
    if "tool_calls" in keys or "function_call" in keys:
        return True
    if "name" in keys and keys & {"arguments", "parameters", "args", "input"}:
        return True
    fn = obj.get("function")
    if isinstance(fn, dict) and "name" in fn:
        return True
    tool = obj.get("tool") or obj.get("tool_name") or obj.get("recipient_name")
    if isinstance(tool, str) and tool.split(".")[-1] in KNOWN_TOOLS:
        return True
    return "action" in keys and bool(keys & {"target", "old_text", "content"}) and \
        str(obj.get("action")).lower() in ("add", "replace", "remove")


def _json_spans(text: str):
    """(start, end) of top-level {...} / [...] spans that parse as JSON."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            close = "}" if ch == "{" else "]"
            depth, j, in_str, esc = 0, i, False, False
            while j < n:
                c = text[j]
                if in_str:
                    esc = (c == "\\") and not esc
                    if c == '"' and not esc:
                        in_str = False
                    elif c != "\\":
                        esc = False
                elif c == '"':
                    in_str = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n and text[j] == close:
                try:
                    yield i, j + 1, json.loads(text[i:j + 1])
                    i = j + 1
                    continue
                except ValueError:
                    pass
        i += 1


def _call_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _CALL_LINE.match(s) or _GENERIC_CALL.match(s):
        return True
    m = _TOOL_LEAD.match(s)
    if not m:
        return False
    rest = m.group("rest").strip()
    if _PAREN_TARGET.match(rest):
        return True                                   # "Memory (user):"
    if rest.startswith(("{", "[")):
        return any(_is_call_object(obj) for _, _, obj in _json_spans(rest))
    keys = [k.group("key").lower() for k in _KV.finditer(rest)]
    return any(k in _ARG_NAMES for k in keys) and ("=" in rest or len(keys) >= 2)


def strip_tool_call_text(text: str) -> Tuple[str, bool]:
    """``text`` without printed tool calls, and whether any were found (see module docstring)."""
    if not text:
        return "", False
    original = str(text)
    out = _BLOCK.sub("\n", original)
    out = _FUNCTION.sub("\n", out)
    out = _PARAMETER.sub("\n", out)
    out = _MISTRAL.sub("\n", out)
    out = _SPECIAL_BLOCK.sub("\n", out)
    out = _LONE_TAG.sub("\n", out)

    def fence(m: re.Match) -> str:
        body = m.group("body").strip()
        try:
            return "\n" if _is_call_object(json.loads(body)) else m.group(0)
        except ValueError:
            return "\n" if body and all(_call_line(ln) for ln in body.splitlines() if ln.strip()) else m.group(0)

    out = _FENCE.sub(fence, out)
    spans = [(a, b) for a, b, obj in _json_spans(out) if _is_call_object(obj)]
    for a, b in reversed(spans):
        out = out[:a] + "\n" + out[b:]
    kept = [ln for ln in out.splitlines() if not _call_line(ln)]
    found = "\n".join(kept) != "\n".join(original.splitlines())
    if not found:
        return original, False
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), True


def looks_like_tool_call_text(text: str) -> bool:
    return strip_tool_call_text(text)[1]
