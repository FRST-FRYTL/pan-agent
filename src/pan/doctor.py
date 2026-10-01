"""`pan doctor`: health check for PAN's OpenAI-compatible model endpoint (stdlib only).

Minimum capabilities:

- the agent (Hermes) needs **tool calling**: structured ``tool_calls`` with JSON arguments, streamed and
  non-streamed;
- PAN's LLM memory gate needs ``response_format: json_schema``, or at least ``json_object`` (then set
  ``models.gate.structured: json_object``).

Without a suitable endpoint PAN runs the rules-only gate (``classifier: {kind: rules}``).

Checks (names for ``--skip``), run in this order:

- ``models``: ``GET {base_url}/models`` answers and lists the configured model (without one, the listed
  ids are reported and the first is used for the other checks).
- ``tool_calls``: 6 streamed requests, each offering one small synthetic tool with a prompt that needs
  it. The SSE stream is reassembled; each answer must be a structured call with the right name and JSON
  arguments holding the required keys. Reports n/6, time to first token and total latency.
- ``tool_corruption``: one Hermes-shaped request (agent persona, tool-use rules, 7 realistic tool
  schemas) sent 5 times, not streamed, temperature 0. With a server prefix cache, attempts 2+ hit the
  cached prefix. Every answer must be a clean call: valid JSON arguments, no tool-call markup in
  ``content``, no truncated or garbled arguments, the same function each time. Some server builds
  corrupt tool calls when prefix caching or speculative decoding is on; this catches it.
- ``json_schema``: one gate-style request with ``response_format: json_schema`` (strict) whose output
  must validate; on HTTP 400 it retries with ``json_object`` and warns.

All prompts and schemas are synthetic. Each request has a timeout (default 120 s) and the whole run a
time budget (``TOTAL_BUDGET_S``). ``enable_thinking: false`` is only sent with ``--no-thinking``.

Exit codes: 0 all checks ok or warn, 1 a check failed, 2 the endpoint is unreachable.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator

DEFAULT_TIMEOUT_S = 120.0
TOTAL_BUDGET_S = 300.0            # the whole run; later requests get what is left
TOOL_MAX_TOKENS = 1024
CORRUPTION_ATTEMPTS = 5
CHECKS = ("models", "tool_calls", "tool_corruption", "json_schema")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the Authorization header to another URL. A 3xx
    surfaces as an HTTP error instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

HELP = """\
Check that an OpenAI-compatible endpoint can serve PAN.

Minimum capabilities:
  - tool calling (structured tool_calls): needed by the agent (Hermes)
  - response_format json_schema (or at least json_object): needed by the LLM memory gate
Without a suitable endpoint, PAN runs the rules-only gate (classifier: {kind: rules}).

checks: models, tool_calls (6 streamed probes), tool_corruption (5 identical Hermes-shaped
requests, exercises a prefix cache), json_schema.
exit codes: 0 ok/warn, 1 a check failed, 2 endpoint unreachable.
Defaults for --base-url/--model/--api-key come from models.gate in $HERMES_HOME/pan/config.yaml."""


# -- results ------------------------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str                  # ok | warn | fail | skip
    detail: str
    seconds: float = 0.0
    hint: str = ""               # fix hint, printed for warn and fail
    unreachable: bool = False    # the endpoint did not answer at all (exit code 2)
    data: dict[str, Any] = field(default_factory=dict)


# -- transport ----------------------------------------------------------------------------------------

class RequestError(Exception):
    """A failed request; ``kind`` is connection | timeout | http | invalid."""

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass
class StreamResult:
    content: str = ""
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    finish_reason: str | None = None
    ttft_s: float | None = None
    total_s: float = 0.0


@dataclass
class Endpoint:
    base_url: str
    model: str = ""
    api_key: str = "EMPTY"
    timeout: float = DEFAULT_TIMEOUT_S
    no_thinking: bool = False
    deadline: float = 0.0        # time.monotonic() limit for the whole run; 0 = none

    def _timeout(self) -> float:
        if not self.deadline:
            return self.timeout
        left = self.deadline - time.monotonic()
        if left <= 1:
            raise RequestError("timeout", "run time budget exhausted")
        return min(self.timeout, left)

    def _request(self, path: str, body: dict[str, Any] | None, read: Callable[[Any], Any]) -> Any:
        timeout = self._timeout()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        if body and body.get("stream"):
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(self.base_url.rstrip("/") + path, data=data, headers=headers,
                                     method="POST" if body is not None else "GET")
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return read(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200].decode("utf-8", "replace").strip() if exc.fp else ""
            raise RequestError("http", f"HTTP {exc.code} {detail}".strip(), exc.code) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise RequestError("timeout", f"no answer within {timeout:.0f}s") from None
            raise RequestError("connection", str(exc.reason)) from None
        except (socket.timeout, TimeoutError):
            raise RequestError("timeout", f"no answer within {timeout:.0f}s") from None
        except (ConnectionError, OSError) as exc:
            raise RequestError("connection", str(exc)) from None
        except http.client.HTTPException as exc:
            raise RequestError("invalid", f"broken HTTP response: {exc!r:.120}") from None

    def body(self, **fields: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "temperature": 0} if self.model else {"temperature": 0}
        body.update(fields)
        if self.no_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def get_json(self, path: str) -> Any:
        return self._request(path, None, lambda resp: _parse_json(resp.read()))

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """(Non-streamed) choice 0 of a chat completion: {"message": ..., "finish_reason": ...}."""
        data = self._request("/chat/completions", body, lambda resp: _parse_json(resp.read()))
        try:
            choice = data["choices"][0]
            if not isinstance(choice["message"], dict):
                raise TypeError
            return choice
        except (KeyError, IndexError, TypeError):
            raise RequestError("invalid", f"not a chat completion: {str(data)[:120]}") from None

    def chat_stream(self, body: dict[str, Any]) -> StreamResult:
        t0 = time.perf_counter()
        out = self._request("/chat/completions", body, lambda resp: read_stream(resp, t0))
        out.total_s = time.perf_counter() - t0
        return out


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        raise RequestError("invalid", f"not JSON: {raw[:120]!r}") from None


def iter_sse(lines: Iterable[bytes | str]) -> Iterator[str]:
    """The data payload of each server-sent event (multi-line data joined with a newline)."""
    data: list[str] = []
    for raw in lines:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def read_stream(lines: Iterable[bytes | str], t0: float) -> StreamResult:
    """Reassemble a streamed chat completion: content, ``tool_calls`` deltas (by index), finish reason."""
    out = StreamResult()
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for data in iter_sse(lines):
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            content.append(text)
            deltas = delta.get("tool_calls") or []
            if out.ttft_s is None and (text or deltas or delta.get("reasoning_content") or delta.get("reasoning")):
                out.ttft_s = time.perf_counter() - t0
            for tc in deltas:
                slot = calls.setdefault(int(tc.get("index") or 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                out.finish_reason = choice["finish_reason"]
    out.content = "".join(content)
    out.tool_calls = [calls[i] for i in sorted(calls)]
    return out


# -- judging a tool call ------------------------------------------------------------------------------

# Tool-call syntax that must never reach ``content`` (chat-template formats of common model families).
MARKUP = ("<tool_call", "</tool_call", "<function=", "<function_call", "</function", "<parameter=",
          "[TOOL_CALLS]", "<|")
_JSON_CALL = re.compile(r'\{\s*"(name|function)"\s*:\s*"')


def markup_in(text: str) -> str | None:
    for marker in MARKUP:
        if marker in text:
            return marker
    return "JSON tool call" if _JSON_CALL.search(text) else None


def _garbled(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_garbled(v) for v in value.values())
    if isinstance(value, list):
        return any(_garbled(v) for v in value)
    return isinstance(value, str) and ("\ufffd" in value or markup_in(value) is not None)


def judge_call(calls: list[dict[str, str]], content: str | None, finish: str | None, *, name: str,
               required: Iterable[str], allowed: Iterable[str] | None = None,
               expected: dict[str, Any] | None = None) -> str | None:
    """None when the first call is clean and matches, else the problem (short text)."""
    leak = markup_in(content or "")
    if leak:
        return f"tool-call markup in content ({leak})"
    truncated = " (truncated at max_tokens)" if finish == "length" else ""
    if not calls:
        return "no structured tool call" + truncated
    call = calls[0]
    if call.get("name") != name:
        return f"wrong function {call.get('name')!r:.40} (expected {name})"
    raw = call.get("arguments") or ""
    try:
        args = json.loads(raw)
    except ValueError:
        return f"arguments are not valid JSON{truncated}: {raw[:60]!r}"
    if not isinstance(args, dict):
        return f"arguments are not an object: {raw[:60]!r}"
    missing = [k for k in required if k not in args]
    if missing:
        return f"missing argument(s) {', '.join(missing)}"
    if allowed is not None and set(args) - set(allowed):
        return f"unknown argument(s) {', '.join(sorted(set(args) - set(allowed)))}"
    if _garbled(args):
        return f"garbled argument values: {raw[:60]!r}"
    for key, want in (expected or {}).items():
        if not _same(args.get(key), want):
            return f"wrong value for {key}: {args.get(key)!r:.40} (expected {want!r})"
    return None


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(got) - float(want)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(got).strip().removeprefix("./").lower() == str(want).lower()


# -- check: models ------------------------------------------------------------------------------------

def check_models(ep: Endpoint) -> CheckResult:
    try:
        data = ep.get_json("/models")
    except RequestError as exc:
        if exc.kind not in ("connection", "timeout"):
            return CheckResult("models", FAIL, f"GET /models: {exc}",
                               hint="the server answers but not as an OpenAI-compatible API: check --base-url "
                                    "(it usually ends in /v1)")
        return CheckResult("models", FAIL, f"unreachable: {exc}", unreachable=True,
                           hint=f"no answer from {ep.base_url}: is the server running? check --base-url "
                                "(models.gate.base_url)")
    ids = [str(m.get("id")) for m in (data.get("data") if isinstance(data, dict) else None) or []
           if isinstance(m, dict) and m.get("id")]
    if not ids:
        return CheckResult("models", WARN, "GET /models lists no models", data={"ids": ids},
                           hint="the server lists no models; set --model to the served name")
    if not ep.model:
        ep.model = ids[0]
        return CheckResult("models", WARN, f"no model configured; listed: {', '.join(ids)} (using {ids[0]})",
                           data={"ids": ids}, hint="set models.gate.model (or --model) to one of the listed ids")
    if ep.model not in ids:
        return CheckResult("models", FAIL, f"model {ep.model!r} not listed; listed: {', '.join(ids)}",
                           data={"ids": ids}, hint="set models.gate.model (or --model) to one of the listed ids, "
                                                   "or serve the model under that name")
    return CheckResult("models", OK, f"{ep.model} listed ({len(ids)} model(s))", data={"ids": ids})


# -- check: streamed tool-call probe ------------------------------------------------------------------

def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


ADD_TOOL = _fn("add", "Add two numbers and return the sum.",
               {"a": {"type": "number", "description": "first addend"},
                "b": {"type": "number", "description": "second addend"}}, ["a", "b"])
TIME_TOOL = _fn("get_time", "Current local time in an IANA time zone.",
                {"timezone": {"type": "string", "description": "IANA zone name, e.g. Europe/Paris"}},
                ["timezone"])

PROBE_SYSTEM = ("You are a precise assistant with access to tools. When a request needs a tool, call it with "
                "exactly the values the user gives. Never compute, guess or invent the result yourself; "
                "the tool is the only source of truth for it.")

# (tool, user prompt, expected arguments)
PROBE_CASES: list[tuple[dict[str, Any], str, dict[str, Any]]] = [
    (ADD_TOOL, "Use the add tool to work out 17 + 25.", {"a": 17, "b": 25}),
    (TIME_TOOL, "What time is it right now in the Asia/Tokyo time zone? Use the get_time tool.",
     {"timezone": "Asia/Tokyo"}),
    (ADD_TOOL, "Please add 3.5 and 8 with the add tool.", {"a": 3.5, "b": 8}),
    (TIME_TOOL, "Look up the current time for America/Chicago using get_time.", {"timezone": "America/Chicago"}),
    (ADD_TOOL, "Call add to sum 1200 and 34.", {"a": 1200, "b": 34}),
    (TIME_TOOL, "Which time is it in Europe/Lisbon at the moment? Call get_time.", {"timezone": "Europe/Lisbon"}),
]


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def check_tool_calls(ep: Endpoint) -> CheckResult:
    attempts = []
    for tool, prompt, expected in PROBE_CASES:
        fn = tool["function"]
        body = ep.body(stream=True, max_tokens=TOOL_MAX_TOKENS, tools=[tool], messages=[
            {"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": prompt}])
        try:
            res = ep.chat_stream(body)
        except RequestError as exc:
            attempts.append({"ok": False, "problem": str(exc), "kind": exc.kind})
            continue
        problem = judge_call(res.tool_calls, res.content, res.finish_reason, name=fn["name"],
                             required=fn["parameters"]["required"], allowed=fn["parameters"]["properties"],
                             expected=expected)
        attempts.append({"ok": problem is None, "problem": problem, "ttft_s": _round(res.ttft_s),
                         "total_s": _round(res.total_s)})
    n_ok = sum(a["ok"] for a in attempts)
    ttft = _median([a["ttft_s"] for a in attempts if a.get("ttft_s") is not None])
    total = _median([a["total_s"] for a in attempts if a.get("total_s") is not None])
    detail = f"{n_ok}/{len(attempts)} ok, TTFT median {_fmt_s(ttft)}, total median {_fmt_s(total)}"
    data = {"passed": n_ok, "n": len(attempts), "ttft_median_s": ttft, "total_median_s": total,
            "attempts": attempts}
    if all(a.get("kind") == "connection" for a in attempts):
        return CheckResult("tool_calls", FAIL, f"unreachable: {attempts[0]['problem']}", unreachable=True,
                           data=data, hint=f"no answer from {ep.base_url}: is the server running?")
    first = next((a["problem"] for a in attempts if not a["ok"]), None)
    if first:
        detail += f"; first problem: {first}"
    if n_ok == len(attempts):
        return CheckResult("tool_calls", OK, detail, data=data)
    status = WARN if n_ok == len(attempts) - 1 else FAIL
    return CheckResult("tool_calls", status, detail, data=data, hint=(
        "structured tool calls are unreliable: enable the server's tool-call parser for this model "
        "(vLLM: --enable-auto-tool-choice --tool-call-parser <parser>) and check the chat template; "
        "with thinking models try --no-thinking or a larger max_tokens budget"))


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _fmt_s(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


# -- check: repeated Hermes-shaped request (tool-call corruption) -------------------------------------

AGENT_SYSTEM = """\
You are Atlas, a careful software assistant that works inside a user's project directory. You can \
read and write files, run shell commands, search the workspace, look things up in your long-term \
memory and fetch web pages. You work step by step and you prefer doing over describing.

How you use tools:
- Use a tool whenever the answer depends on the state of the machine, the files or your memory. Do \
not guess file contents, command output or remembered facts.
- Call tools through the native tool-calling interface only. Never write a tool call as plain text, \
markup or a code block in your reply.
- Pass arguments exactly as the schema describes. Paths are relative to the project root unless the \
user gives an absolute path. Keep arguments minimal: leave optional fields out unless you need them.
- Prefer the most specific tool: read_file to look at a file, search_files to find one, terminal only \
for commands that have no dedicated tool.
- Make one tool call at a time when the next step depends on the result.
- Never run destructive commands (deleting, force-pushing, overwriting) without the user's explicit \
confirmation. Never print secrets such as API keys or passwords.

How you answer:
- Be brief and concrete. Lead with the result, then the detail that matters.
- When you changed something, say what and where. When a step failed, say why and what you will try \
next.
- If a request is ambiguous, ask one short question instead of guessing.

About memory:
- memory_search finds notes you stored in earlier sessions; memory_read opens one note by its id.
- Treat remembered facts as possibly outdated and prefer the current files when they disagree.

The user's workspace is a small documentation project with a docs/ folder, a notes/ folder and a \
few configuration files at the top level."""

AGENT_TOOLS = [
    _fn("read_file", "Read a text file from the workspace. Returns the content with line numbers.",
        {"path": {"type": "string", "description": "file path, relative to the project root"},
         "offset": {"type": "integer", "description": "first line to return (1-based)"},
         "limit": {"type": "integer", "description": "maximum number of lines"}}, ["path"]),
    _fn("write_file", "Create or overwrite a text file in the workspace.",
        {"path": {"type": "string", "description": "file path, relative to the project root"},
         "content": {"type": "string", "description": "the complete new file content"}}, ["path", "content"]),
    _fn("terminal", "Run a shell command in the project directory and return its output.",
        {"command": {"type": "string", "description": "the command line to run"},
         "timeout": {"type": "integer", "description": "seconds before the command is stopped"}}, ["command"]),
    _fn("search_files", "Search file names and contents in the workspace with a regular expression.",
        {"pattern": {"type": "string", "description": "regular expression"},
         "path": {"type": "string", "description": "directory to search (default: project root)"},
         "target": {"type": "string", "enum": ["content", "files"], "description": "what to match"}},
        ["pattern"]),
    _fn("memory_search", "Search long-term memory notes from earlier sessions.",
        {"query": {"type": "string", "description": "what to look for"},
         "limit": {"type": "integer", "description": "maximum number of hits"}}, ["query"]),
    _fn("memory_read", "Read one long-term memory note by its id.",
        {"page": {"type": "string", "description": "page id or path from memory_search"},
         "section": {"type": "string", "description": "only this heading"}}, ["page"]),
    _fn("web_fetch", "Fetch a web page and return its text.",
        {"url": {"type": "string", "description": "absolute http(s) URL"}}, ["url"]),
]

AGENT_USER = "Please open docs/getting-started.md and show me what it says."
AGENT_EXPECT = ("read_file", {"path": "docs/getting-started.md"})


def check_tool_corruption(ep: Endpoint, attempts: int = CORRUPTION_ATTEMPTS) -> CheckResult:
    name, expected = AGENT_EXPECT
    schema = next(t["function"]["parameters"] for t in AGENT_TOOLS if t["function"]["name"] == name)
    body = ep.body(stream=False, max_tokens=TOOL_MAX_TOKENS, tools=AGENT_TOOLS, messages=[
        {"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": AGENT_USER}])
    results: list[dict[str, Any]] = []
    for _ in range(attempts):
        t0 = time.perf_counter()
        try:
            choice = ep.chat(body)
        except RequestError as exc:
            results.append({"ok": False, "problem": str(exc), "kind": exc.kind,
                            "seconds": _round(time.perf_counter() - t0)})
            continue
        msg = choice["message"]
        calls = [{"name": (tc.get("function") or {}).get("name") or "",
                  "arguments": (tc.get("function") or {}).get("arguments") or ""}
                 for tc in msg.get("tool_calls") or [] if isinstance(tc, dict)]
        problem = judge_call(calls, msg.get("content"), choice.get("finish_reason"), name=name,
                             required=schema["required"], allowed=schema["properties"], expected=expected)
        results.append({"ok": problem is None, "problem": problem, "function": calls[0]["name"] if calls else None,
                        "seconds": _round(time.perf_counter() - t0)})
    marks = " ".join(f"#{i}:{'ok' if r['ok'] else 'FAIL'}" for i, r in enumerate(results, 1))
    n_ok = sum(r["ok"] for r in results)
    functions = {r["function"] for r in results if r.get("function")}
    corrupted = n_ok < len(results) or len(functions) > 1
    data = {"attempts": results, "passed": n_ok, "n": len(results), "corrupted": corrupted}
    if all(r.get("kind") == "connection" for r in results):
        return CheckResult("tool_corruption", FAIL, f"unreachable: {results[0]['problem']}", unreachable=True,
                           data=data, hint=f"no answer from {ep.base_url}: is the server running?")
    if not corrupted:
        return CheckResult("tool_corruption", OK, f"{n_ok}/{len(results)} clean ({name}) {marks}", data=data)
    first_bad = next((r["problem"] for r in results if not r["ok"]), None) or "function changed between attempts"
    detail = f"{n_ok}/{len(results)} clean, corrupted {marks}; {first_bad}"
    if results[0]["ok"]:
        hint = ("tool calls corrupted on repeated (prefix-cached) requests: see 'Tool-call corruption with prefix caching + MTP' in the spark-vllm-agent-cookbook field notes; "
                "try disabling speculative decoding or prefix caching")
    else:
        hint = ("tool calls are not clean even on a fresh request: check the server's tool-call parser and "
                "chat template, then try disabling speculative decoding or prefix caching")
    return CheckResult("tool_corruption", FAIL, detail, data=data, hint=hint)


# -- check: json_schema for the memory gate -----------------------------------------------------------

GATE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decision", "reason"],
               "properties": {"decision": {"type": "string", "enum": ["record", "drop"]},
                              "reason": {"type": "string"}}}

GATE_SYSTEM = ("You decide whether a short note from a conversation is worth keeping in long-term memory. "
               "Keep stable facts and preferences; drop small talk and one-off details. Answer with a JSON "
               'object only: {"decision": "record" or "drop", "reason": "<one short sentence>"}.')
GATE_USER = "Note: The user said they always want code examples written in Python 3.12 from now on."


def validate_gate(content: str | None) -> str | None:
    """None when ``content`` is a JSON object matching GATE_SCHEMA, else the problem."""
    try:
        obj = json.loads((content or "").strip())
    except ValueError:
        return f"output is not JSON: {(content or '')[:60]!r}"
    if not isinstance(obj, dict):
        return "output is not a JSON object"
    missing = [k for k in GATE_SCHEMA["required"] if k not in obj]
    if missing:
        return f"missing key(s) {', '.join(missing)}"
    extra = set(obj) - set(GATE_SCHEMA["properties"])
    if extra:
        return f"unexpected key(s) {', '.join(sorted(extra))}"
    if obj["decision"] not in GATE_SCHEMA["properties"]["decision"]["enum"]:
        return f"decision {obj['decision']!r:.30} not in record|drop"
    if not isinstance(obj["reason"], str):
        return "reason is not a string"
    return None


def _gate_body(ep: Endpoint, response_format: dict[str, Any]) -> dict[str, Any]:
    return ep.body(stream=False, max_tokens=300, response_format=response_format, messages=[
        {"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": GATE_USER}])


RULES_HINT = "the LLM memory gate will not work here; use the rules-only gate (classifier: {kind: rules})"


def check_json_schema(ep: Endpoint) -> CheckResult:
    fmt = {"type": "json_schema", "json_schema": {"name": "memory_decision", "schema": GATE_SCHEMA, "strict": True}}
    try:
        problem = validate_gate(ep.chat(_gate_body(ep, fmt))["message"].get("content"))
    except RequestError as exc:
        if exc.kind == "connection":
            return CheckResult("json_schema", FAIL, f"unreachable: {exc}", unreachable=True,
                               hint=f"no answer from {ep.base_url}: is the server running?")
        if exc.status != 400:
            return CheckResult("json_schema", FAIL, f"json_schema request failed: {exc}", hint=RULES_HINT)
        return _json_object_fallback(ep, str(exc))
    if problem:
        return CheckResult("json_schema", FAIL, f"json_schema accepted but output invalid: {problem}",
                           hint="structured output is not enforced; try models.gate.structured: json_object, "
                                "or the rules-only gate (classifier: {kind: rules})")
    return CheckResult("json_schema", OK, "json_schema output validates", data={"structured": "json_schema"})


def _json_object_fallback(ep: Endpoint, why: str) -> CheckResult:
    try:
        problem = validate_gate(ep.chat(_gate_body(ep, {"type": "json_object"}))["message"].get("content"))
    except RequestError as exc:
        return CheckResult("json_schema", FAIL, f"json_schema rejected ({why[:60]}); json_object failed: {exc}",
                           hint=RULES_HINT)
    if problem:
        return CheckResult("json_schema", FAIL, f"json_schema rejected; json_object output invalid: {problem}",
                           hint=RULES_HINT)
    return CheckResult("json_schema", WARN, "json_schema rejected (HTTP 400); json_object works",
                       data={"structured": "json_object"},
                       hint="the gate should use models.gate.structured: json_object")


# -- runner -------------------------------------------------------------------------------------------

RUNNERS: dict[str, Callable[[Endpoint], CheckResult]] = {
    "models": check_models, "tool_calls": check_tool_calls,
    "tool_corruption": check_tool_corruption, "json_schema": check_json_schema}


def run_checks(ep: Endpoint, skip: Iterable[str] = (), budget_s: float = TOTAL_BUDGET_S) -> list[CheckResult]:
    """Run every check not in ``skip``; after an unreachable result the rest are skipped."""
    skip = set(skip)
    ep.deadline = time.monotonic() + budget_s if budget_s else 0.0
    results: list[CheckResult] = []
    unreachable = False
    for name in CHECKS:
        if name in skip:
            results.append(CheckResult(name, SKIP, "skipped (--skip)"))
            continue
        if unreachable:
            results.append(CheckResult(name, SKIP, "endpoint unreachable"))
            continue
        t0 = time.perf_counter()
        result = RUNNERS[name](ep)
        result.seconds = round(time.perf_counter() - t0, 2)
        results.append(result)
        unreachable = result.unreachable
    return results


def exit_code(results: list[CheckResult]) -> int:
    if any(r.unreachable for r in results):
        return 2
    return 1 if any(r.status == FAIL for r in results) else 0


def render(results: list[CheckResult], ep: Endpoint) -> str:
    lines = [f"endpoint {ep.base_url}  model {ep.model or '-'}", ""]
    width = max(len(r.name) for r in results)
    for r in results:
        lines.append(f"{r.name:<{width}}  {r.status.upper():<4}  {r.seconds:6.1f}s  {r.detail}")
    hints = [r for r in results if r.hint and r.status in (FAIL, WARN)]
    if hints:
        lines.append("")
        lines.extend(f"fix ({r.name}): {r.hint}" for r in hints)
    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.description = HELP
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: models.gate.base_url)")
    p.add_argument("--model", default=None, help="model id (default: models.gate.model)")
    p.add_argument("--api-key", default=None, help="bearer token (default: models.gate.api_key)")
    p.add_argument("--skip", nargs="+", action="extend", default=[], type=_check_names, metavar="CHECK",
                   help=f"checks to leave out (space- or comma-separated): {', '.join(CHECKS)}")
    p.add_argument("--json", action="store_true", help="print the results as JSON")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per request (default: 120)")
    p.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs.enable_thinking=false with every chat request")


def _check_names(value: str) -> str:
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown check(s) {', '.join(unknown)} (choose from {', '.join(CHECKS)})")
    return ",".join(names)


def endpoint_from_args(args: argparse.Namespace, paths: Any) -> Endpoint:
    """CLI values over ``models.gate`` of the profile's PAN config (read-only)."""
    from pan.config import load_config

    gate = load_config(paths).models.gate
    return Endpoint(base_url=args.base_url or gate.base_url,
                    model=gate.model if args.model is None else args.model,
                    api_key=args.api_key or gate.api_key, timeout=args.timeout, no_thinking=args.no_thinking)


def run_cli(args: argparse.Namespace, paths: Any) -> int:
    ep = endpoint_from_args(args, paths)
    results = run_checks(ep, skip={n for item in args.skip for n in item.split(",") if n})
    code = exit_code(results)
    if args.json:
        print(json.dumps({"base_url": ep.base_url, "model": ep.model, "exit_code": code,
                          "checks": [asdict(r) for r in results]}, indent=2))
    else:
        print(render(results, ep))
    sys.stdout.flush()
    return code
