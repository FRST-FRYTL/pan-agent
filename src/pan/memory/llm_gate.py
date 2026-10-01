"""M9 LLM gate: one call per episode to a local OpenAI-compatible model (classifier design §4.1).

:class:`LocalLLMClassifier` implements the :class:`~pan.memory.classifier.Classifier` protocol
(config ``classifier.kind: llm``; ``shadow`` runs it next to the rules, which then still apply).
Per episode:

1. **Stage 0 (code)** — printed tool calls stripped from the assistant text, secrets redacted
   (:mod:`pan.memory.secrets`) in everything the model will see; episodes with nothing to record
   pre-skipped as noise (:func:`pre_skip`: empty, questions only — none carrying a first-person fact —,
   thank-you only; no tool call).
2. **one model call** (``models.gate``: temperature 0, thinking off, ``response_format:
   json_schema``) with the episode, the previous turn, the recalled text and the L1 snapshot, each
   section budgeted and truncated head+tail (logged). The answer is the ``GateDecision`` of
   :mod:`pan.memory.gate_prompt` (§3.2).
3. **deterministic post-checks** (§4.1.4), always on:

   - *grounding*: every claim must overlap its source (≥ 0.6 of its content tokens) and contain
     no value token (anything with a digit) that is not in the source — ``user_stated`` → this
     turn's user text (+ the previous user turn, for corrections), ``user_confirmed`` → previous
     assistant text + this user text, ``tool_observed`` → this episode's tool calls/outputs and file
     paths. A claim that fails is downgraded to ``assistant_inferred`` if it is grounded in the
     episode at all, otherwise dropped;
   - *inferred claims*: dropped when they echo recalled text / L1 / the user's words, talk about
     memory or the assistant itself, or are longer than a fact sentence (the rules' M4–M6
     filters); an inferred-only candidate needs a non-``other`` domain and tool output that
     supports it (F8: ungrounded world knowledge);
   - *assistant_reported* (T3): an inferred claim those checks would drop is kept when it is the
     assistant's own report of this turn (:func:`_reportable`: grounded in its text with every
     value; an action, decision or result — or a typed value about something the conversation
     named; no offer, plan, advice, hedge, question, memory talk or general knowledge; when a tool
     ran, its values must be in the tool output or the user's words). Label ``reported``, text
     prefixed ``Assistant reported:``, confidence ≤ 0.4; the curator never lets it strike an
     observed or inferred bullet, and it never reaches L1;
   - *secrets*: the output is scanned again; values are redacted;
   - *authorship*: preferences keep only ``user_stated`` claims (injection policy, §6.2);
     ``supersedes`` is only passed on when every claim is observed ("inferred never overrides
     observed" stays true in the curator);
   - *destination policy* (§3.3): preference → USER.md, volatile / non-long-term → none, else wiki;
     ≤ 4 candidates, ≤ 6 claims, ≤ 300 chars;
   - *subject*: the model's, when short and found in the episode; else :func:`facts.subject_of`.

4. **fallback**: timeout (one budget ``classifier.timeout_s`` per episode; not retried), connection
   error (``classifier.attempts`` tries within the budget), HTTP error, invalid
   output, or an open circuit breaker (``breaker_failures`` consecutive failed episodes →
   ``breaker_open_s`` without calls) → :class:`RuleClassifier` output with secrets redacted and
   ``rationale: "fallback:rules (<reason>); …"``.

Every call leaves ``last_info`` (version ``GATE_VERSION``, status, fallback reason, latency per call,
token usage, truncation, post-check notes, the parsed model output); the daemon writes it to the
curation log as ``gate``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pan.config import ClassifierConfig, GateModelConfig, PanConfig
from pan.events.schema import (AgentEvent, Destination, Lifetime, MemoryCandidate, MemoryClassification,
                               MemoryType)
from pan.memory import gate_prompt as gp
from pan.memory.claims import REPORTED_PREFIX, STOPWORDS, normalize, sentences, shorten, stem, token_set
from pan.memory.classifier import (_HEDGE, _OTHER_PARTY, INFERRED, MAX_CLAIM_CHARS, MEMORY_TOOLS, OBSERVED, REPORTED,
                                   Classifier, RuleClassifier, _clean, _command, _context, _failed, _inferred_ok, _ok, _recalled, _tags,
                                   _title_from_claim, _tool_output, _unack, classification, meta_talk)
from pan.memory.episodes import Episode
from pan.memory.classifier import _INTENT_LEAD, _MOVE_VERB, _PREF_ANY, _PREF_LEAD, _PREF_START
from pan.memory.facts import RETRACT_VERB, display_subject, distinctive_tokens, strip_lead, subject_of, subject_tags, values
from pan.memory.dates import annotate, bare, claim_date, stated_date, stated_offset, strip_today_line
from pan.memory.curator import different_occasion
from pan.memory.secrets import redact
from pan.memory.toolcall_text import strip_tool_call_text

logger = logging.getLogger("pan.memory.llm_gate")

GATE_VERSION = gp.GATE_VERSION
GROUNDING_OVERLAP = 0.6       # §4.1.4 (1): share of a claim's content tokens found in its source
TOOL_GROUNDING_OVERLAP = 0.5  # tool output: values must match exactly anyway; the prose around them is the model's
TOOL_SUPPORT_OVERLAP = 0.5    # §4.1.4 (2): inferred-only candidates need this much tool support
SUBJECT_OVERLAP = 0.5

# Section budgets in characters (≈ 4 chars/token), scaled down together when the whole input is
# over ``classifier.max_input_chars``.
BUDGETS = {"previous_user": 600, "previous_assistant": 1000, "known": 3000, "recalled": 3000, "user": 6000,
           "tool_call": 1500, "tools": 8000, "files": 600, "assistant": 4000}
MAX_TOOL_CALLS = 8
CACHE_SIZE = 256              # episodes whose result is kept (a deferred episode is re-classified every poll)

_OBSERVED_CLASSES = {"user_stated", "user_confirmed", "tool_observed"}
_KIND_TYPE = {"preference": MemoryType.USER_PREFERENCE, "decision": MemoryType.DECISION,
              "learning": MemoryType.LEARNING, "procedure": MemoryType.PROCEDURE, "incident": MemoryType.INCIDENT}
_FACT_TYPE = {"personal": MemoryType.PROJECT_FACT, "environment": MemoryType.ENVIRONMENT, "configuration": MemoryType.CONFIGURATION,
              "architecture": MemoryType.ARCHITECTURE, "project": MemoryType.PROJECT_FACT,
              "tooling": MemoryType.ENVIRONMENT, "other": MemoryType.PROJECT_FACT}
_NUMBERED = re.compile(r"^\s*\d+\|", re.M)


class GateError(Exception):
    """A failed gate call; ``reason`` is one of timeout | connection | http_error | invalid_output."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


# -- transport ----------------------------------------------------------------------------------------

class ChatClient:
    """Minimal stdlib client for ``POST {base_url}/chat/completions`` (non-streaming)."""

    def __init__(self, cfg: GateModelConfig, timeout_s: float) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.structured = cfg.structured
        mode = str(cfg.template_kwargs or "auto").strip().lower()
        self.template_kwargs = mode if mode in ("auto", "always", "never") else "auto"
        self.send_template_kwargs = self.template_kwargs != "never"
        self.dropped: List[str] = []   # request fields turned off after the endpoint rejected them

    def body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
                                "max_tokens": self.cfg.max_tokens, "stream": False}
        if self.send_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.cfg.thinking)}
        if self.structured == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "gate_decision", "schema": gp.output_schema(), "strict": True}}
        elif self.structured == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def _downgrade(self, exc: GateError) -> bool:
        """After an HTTP 400/422: turn off the request field the endpoint most likely rejected, for
        the rest of the run (True = retry). The field named in the error first; otherwise structured
        output (the M9 behaviour), then ``chat_template_kwargs`` (``template_kwargs: auto`` only — a
        vLLM/SGLang extension that hosted APIs may reject as an unknown field)."""
        text = str(exc)
        if exc.reason != "http_error" or not re.search(r"HTTP (?:400|422)\b", text):
            return False
        kwargs_ok = self.send_template_kwargs and self.template_kwargs == "auto"
        named_kwargs = "chat_template_kwargs" in text or "enable_thinking" in text
        if kwargs_ok and named_kwargs:
            field = "chat_template_kwargs"
        elif self.structured != "none" and not named_kwargs:
            field = "response_format"
        elif kwargs_ok:
            field = "chat_template_kwargs"
        elif self.structured != "none":
            field = "response_format"
        else:
            return False
        if field == "response_format":
            logger.warning("gate endpoint rejected response_format=%s; retrying with strict parsing only",
                           self.structured)
            self.structured = "none"
        else:
            logger.warning("gate endpoint rejected chat_template_kwargs; retrying without it (kept off)")
            self.send_template_kwargs = False
        self.dropped.append(field)
        return True

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """(content, usage). Raises :class:`GateError`. An HTTP 400/422 retries without the rejected
        field (:meth:`_downgrade`, at most two fields) and keeps it off: the server does not support it."""
        while True:   # each retry turns one field off for good, so this ends
            try:
                return self._post(self.body(messages))
            except GateError as exc:
                if not self._downgrade(exc):
                    raise

    def _post(self, body: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.cfg.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace") if hasattr(exc, "read") else ""
            raise GateError("http_error", f"HTTP {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise GateError("timeout", f"no answer within {self.timeout_s:.0f}s") from None
            raise GateError("connection", str(exc.reason)) from None
        except (socket.timeout, TimeoutError):
            raise GateError("timeout", f"no answer within {self.timeout_s:.0f}s") from None
        except (ConnectionError, OSError) as exc:
            raise GateError("connection", str(exc)) from None
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise GateError("invalid_output", f"not a chat completion: {raw[:200]!r}") from None
        usage = dict(data.get("usage") or {})
        usage["finish_reason"] = choice.get("finish_reason")
        return content, usage


# -- input ----------------------------------------------------------------------------------------------

@dataclass
class Sources:
    """Redacted episode texts the post-checks ground claims in."""
    user: str = ""
    previous_user: str = ""
    previous_assistant: str = ""
    tools: str = ""
    assistant: str = ""
    recall: str = ""
    tool_calls_ok: int = 0

    @property
    def everything(self) -> str:
        return "\n".join([self.user, self.previous_user, self.previous_assistant, self.tools, self.assistant])


@dataclass
class GateInput:
    prompt: str
    sources: Sources
    truncated: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    printed_call: bool = False


def _clip(text: str, limit: int, label: str, truncated: List[str]) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    truncated.append(label)
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head)
    return f"{text[:head].rstrip()} …[{len(text) - head - tail} chars cut]… {text[-tail:].lstrip()}"


_SALIENT = re.compile(r"\b(?:error|errors|fatal|fail(?:ed|ure)?|exception|traceback|denied|refused|rejected|"
                      r"exceeded|timeout|timed out|panic|critical|warn(?:ing)?|status=\w+|exit code [1-9])\b", re.I)


def _clip_output(text: str, limit: int, label: str, truncated: List[str]) -> str:
    """Like :func:`_clip`, but lines from the cut middle that look like errors / warnings / statuses
    are kept (a log's failure line is rarely in its first or last lines)."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head_n, tail_n = limit * 2 // 5, limit // 5
    head, middle, tail = text[:head_n], text[head_n:len(text) - tail_n], text[len(text) - tail_n:]
    keep, used = [], 0
    for line in middle.splitlines():
        line = line.strip()
        if line and _SALIENT.search(line) and used + len(line) <= limit - head_n - tail_n:
            keep.append(line)
            used += len(line) + 1
    truncated.append(label)
    cut = len(middle) - used
    body = "\n".join(keep)
    return (f"{head.rstrip()} …[{cut} chars cut" + (f"; {len(keep)} error/warning line(s) kept]…\n{body}\n…"
                                                    if keep else "]…") + f" {tail.lstrip()}")


def _args_text(call: AgentEvent) -> str:
    cmd = _command(call)
    if cmd:
        return f"`{cmd}`"
    args = call.content.get("args")
    if not args:
        return "(no args)"
    return json.dumps(args, ensure_ascii=False, sort_keys=True) if not isinstance(args, str) else args


def _call_output(call: AgentEvent) -> str:
    if _failed(call):
        text = call.content.get("error_message") or call.content.get("result_excerpt") or ""
    else:
        text = call.content.get("result_excerpt") or ""
    return _NUMBERED.sub("", _tool_output(str(text))).strip()


def _pick_calls(calls: Sequence[AgentEvent]) -> List[AgentEvent]:
    """At most ``MAX_TOOL_CALLS``: failures first, then the rest, kept in event order."""
    if len(calls) <= MAX_TOOL_CALLS:
        return list(calls)
    ranked = sorted(range(len(calls)), key=lambda i: (not _failed(calls[i]), i))[:MAX_TOOL_CALLS]
    return [calls[i] for i in sorted(ranked)]


def build_input(episode: Episode, previous: Optional[Episode], known: str, *,
                max_chars: int = ClassifierConfig().max_input_chars) -> GateInput:
    """The user message for the gate call plus the redacted sources for the post-checks."""
    secrets: List[str] = []

    def red(text: str) -> str:
        out, kinds = redact(text or "")
        secrets.extend(kinds)
        return out

    assistant, printed = strip_tool_call_text(episode.assistant_text)
    prev_assistant = strip_tool_call_text(previous.assistant_text)[0] if previous is not None else ""
    calls = [c for c in episode.tool_calls if c.content.get("tool") not in MEMORY_TOOLS]
    src = Sources(user=red(episode.user_text), previous_user=red(previous.user_text if previous else ""),
                  previous_assistant=red(prev_assistant), assistant=red(assistant),
                  recall=red(episode.recalled_text + "\n" + (known or "")),
                  tool_calls_ok=sum(1 for c in calls if _ok(c)))
    tool_texts = []
    for c in calls:
        tool_texts.append(" ".join([str(c.content.get("tool") or ""), red(_args_text(c)), red(_call_output(c))]))
    tool_texts += [str(f.content.get("path") or "") for f in episode.file_changes]
    src.tools = "\n".join(tool_texts)
    recalled_text, known_text = red(episode.recalled_text), red(known or "")

    def render(scale: float) -> Tuple[str, List[str]]:
        truncated: List[str] = []
        b = {k: max(200, int(v * scale)) for k, v in BUDGETS.items()}
        parts: List[str] = []
        if previous is not None and (src.previous_user or src.previous_assistant):
            parts.append("<previous_turn>\n<user>" + _clip(src.previous_user, b["previous_user"], "previous_user",
                                                           truncated)
                         + "</user>\n<assistant>" + _clip(src.previous_assistant, b["previous_assistant"],
                                                          "previous_assistant", truncated)
                         + "</assistant>\n</previous_turn>")
        if known_text.strip():
            parts.append("<known>\n" + _clip(known_text, b["known"], "known", truncated) + "\n</known>")
        if recalled_text.strip():
            parts.append("<recalled>\n" + _clip(recalled_text, b["recalled"], "recalled", truncated) + "\n</recalled>")
        ep = ["<episode>", "<user>" + _clip(src.user, b["user"], "user", truncated) + "</user>"]
        picked = _pick_calls(calls)
        if len(picked) < len(calls):
            truncated.append(f"tools:{len(calls) - len(picked)}_calls_omitted")
        if picked:
            # one or two reads (a notes file, a config) may use more of the budget
            cap = b["tools"] // 2 if len(picked) <= 2 else b["tool_call"]
            per_call = max(200, min(cap, b["tools"] // len(picked)))
            lines = []
            for n, c in enumerate(picked, 1):
                status = "error" if _failed(c) else (str(c.content.get("status") or "ok"))
                args = _clip(red(_args_text(c)), 200, f"tool{n}_args", truncated)
                out = _clip_output(red(_call_output(c)), per_call, f"tool{n}_output", truncated)
                lines.append(f"{n} {c.content.get('tool')} {args} {status}: {out}".rstrip(": "))
            ep.append("<tools>\n" + "\n".join(lines) + "\n</tools>")
        if episode.file_changes:
            files = "\n".join(f"{f.content.get('op') or 'change'} {f.content.get('path')}" for f in episode.file_changes)
            ep.append("<files>\n" + _clip(files, b["files"], "files", truncated) + "\n</files>")
        ep.append("<assistant>" + _clip(src.assistant, b["assistant"], "assistant", truncated) + "</assistant>")
        ep.append("</episode>")
        parts.append("\n".join(ep))
        return "\n".join(parts), truncated

    prompt, truncated = render(1.0)
    if len(prompt) > max_chars:
        prompt, truncated = render(max(0.1, max_chars / len(prompt) * 0.95))
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]
            truncated.append("input_hard_cut")
    return GateInput(prompt=prompt, sources=src, truncated=truncated, secrets=secrets, printed_call=printed)


def messages_for(gate_input: GateInput) -> List[Dict[str, str]]:
    return [{"role": "system", "content": gp.SYSTEM_PROMPT}, {"role": "user", "content": gate_input.prompt}]


def trivial(episode: Episode) -> bool:
    """Stage 0 pre-skip: nothing a gate could record."""
    return (not episode.user_text.strip() and not strip_tool_call_text(episode.assistant_text)[0].strip()
            and not any(c.content.get("tool") not in MEMORY_TOOLS for c in episode.tool_calls)
            and not episode.file_changes)


_STANDING = re.compile(r"\b(?:always|never|every time|whenever|from now on|going forward|by default|"
                       r"immer|nie|niemals|jedes mal|ab jetzt|ab sofort|k(?:ü|ue|u)nftig|grunds(?:ä|ae|a)tzlich)\b", re.I)
_CHANGE = re.compile(r"\b(?:mov(?:e|ed|es|ing)|migrat\w*|switch\w*|chang\w*|replac\w*|renam\w*|cancel\w*|"
                     r"retir\w*|upgrad\w*|downgrad\w*|extend\w*|postpon\w*|umgezogen|verschoben|verlegt|gewechselt|"
                     r"ge(?:ä|ae)ndert|ersetzt|abgesagt|gestrichen)\b", re.I)
_THANKS = re.compile(r"^\W*(?:thanks?(?: you| a lot)?|thx|cheers|great|nice|cool|perfect|danke(?: sch(?:ö|oe)n)?|"
                     r"super|prima|klasse)\W*$", re.I)


def pre_skip(episode: Episode) -> str:
    """Why the gate need not call the model for ``episode`` ("" = call it). Deterministic, cheap:
    no tool call or file change, and the user text is only questions (no typed value, no standing
    or remember wording, no change/retraction verb) or a bare thank-you. Such an episode can only
    yield assistant claims without tool support, which the post-checks drop anyway."""
    if trivial(episode):
        return "empty episode"
    if episode.file_changes or any(c.content.get("tool") not in MEMORY_TOOLS for c in episode.tool_calls):
        return ""
    user = strip_today_line(episode.user_text)
    if not user:
        return "assistant text only, no tool output"
    if _THANKS.match(user):
        return "thank-you only"
    sents = sentences(user, join_lines=False)
    if not sents or not all(x.rstrip().endswith("?") for x in sents):
        return ""
    for x in sents:
        if (values(x).slots or _STANDING.search(x) or _PREF_ANY.search(x) or _INTENT_LEAD.match(x)
                or _MOVE_VERB.search(x) or _CHANGE.search(x) or RETRACT_VERB.search(x)
                or (not _RECALL_Q.search(x) and (_OWN_FACT_MARK.search(x)
                                                 or (_OWN_THING.search(x) and _REQUEST.match(strip_lead(x) or x))))):
            return ""
    if any(len(re.findall(rf"\b{re.escape(m.group(1))}\b", user)) > 1 for m in _LIKE_NAME.finditer(user)):
        return ""
    return "questions only"


# A question that carries a first-person fact (a request about "my …": "What are good ways to keep my hallway
# clean with a dog that sheds?"; "I've …", "… series like Dark, which I just finished?"; "… a Labrador like
# Nala?" when Nala is named again): the gate is asked. Not "How do I …?", "Can you explain …?", "my question"
# or a question about the user's thing ("Which port does our cache use?").
# A question about what memory holds ("Can you remind me when my dental check-up is?", "Based on our last call,
# which vendor did we pick?"): no new fact, whatever first-person words it has.
_RECALL_Q = re.compile(r"\b(?:remind me|based on (?:our|my|what|the|all)|do you (?:remember|recall)|"
                       r"did I (?:tell|mention|say)|what did (?:I|we)|I (?:told|mentioned|shared|gave) you|(?:I've|I have|we've|we have) "
                       r"(?:told|mentioned|shared|given|discussed)|in (?:our|my) (?:last|previous|earlier)|"
                       r"erinner(?:e|st) (?:mich|du dich)|hab(?:e)? ich (?:dir )?(?:gesagt|erz(?:ä|ae)hlt))\b", re.I)
_OWN_THING = re.compile(r"\b(?i:my|our|mein\w*|unser\w*)\s+(?!(?i:own|question|questions|mind|understanding|options)\b)"
                        r"[A-Za-zÄÖÜäöü][\w-]*")
_OWN_FACT_MARK = re.compile(
    r"(?<!\bdo )(?<!\bdid )(?<!\bhave )(?<!\bam )\b(?:I've|I have|I'm|I am|I just|[Ww]e've|[Ww]e have|[Ww]e're|[Ww]e just|"
    r"[Ii]ch habe|[Ii]ch bin|[Ww]ir haben|[Ww]ir sind)\b(?!\s+(?:a question|no idea|been wondering|wondering|curious|"
    r"not sure|told|mentioned|shared|given|discussed|talked)\b)|"
    r",\s*(?:which|who|whom)\s+(?:I|we)\b(?!'d|'ll| would| will)")


# -- output parsing -------------------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_decision(content: str) -> Dict[str, Any]:
    """The gate's JSON object (strict: one object; fences and a think block tolerated)."""
    text = _THINK.sub("", content or "").strip()
    text = _FENCE.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise GateError("invalid_output", f"no JSON object in output: {text[:120]!r}")
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise GateError("invalid_output", f"invalid JSON ({exc}): {text[:120]!r}") from None
    if isinstance(obj, dict):
        obj = gp.from_wire(obj)
    if not isinstance(obj, dict) or not isinstance(obj.get("record"), bool) \
            or not isinstance(obj.get("candidates", []), list):
        raise GateError("invalid_output", "missing record/candidates")
    return obj


def _num(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


# -- post-checks ----------------------------------------------------------------------------------------

# German function words (claims.STOPWORDS is English only; they made German claims look ungrounded).
_DE_STOP = frozenset("""
der die das den dem des ein eine einen einem einer eines und oder aber ist sind war waren hat haben hatte
wird werden wurde von vom zu zum zur mit bei beim im in am an auf aus fur furs uber um als auch nicht noch
nur so wie wir ich du er sie es ihr uns unser unsere unseren unserem unserer mein meine meinen meinem
meiner dass da dort hier jetzt nun schon sehr bitte unter vor nach hinter neben zwischen gegen ohne durch
seit ab pro je user users nutzer nutzerin nutzers
""".split())
_WORD = re.compile(r"[a-z0-9]+")


_ISO_T = re.compile(r"(?<=\d)t(?=\d)")
_DASH = re.compile(r"[‒-―−]")   # figure / en / em dash, minus (normalize() drops them)
# Grouped amounts in either convention ("4.210,50" / "4,210.50") → plain digits ("4210.50"), as a CSV writes them.
_GROUPED_DE = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+),(\d+)(?![\d.,]*\d)")
_GROUPED_EN = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)\.(\d+)(?![\d.,]*\d)")


def _plain_numbers(text: str) -> str:
    text = _GROUPED_DE.sub(lambda m: m.group(1).replace(".", "") + "." + m.group(2), text)
    return _GROUPED_EN.sub(lambda m: m.group(1).replace(",", "") + "." + m.group(2), text)


def _gtokens(text: str) -> set:
    """Content tokens for grounding: claims' tokens minus English and German function words (an ISO
    timestamp's ``T`` separates date and time: "2026-09-21T14:40" holds the value "14"; a dash between
    words separates them: "great—Bento" is not one token)."""
    text = _ISO_T.sub(" ", normalize(_DASH.sub(" ", _plain_numbers(text or ""))))
    return {stem(t) for t in _WORD.findall(text) if t not in STOPWORDS and t not in _DE_STOP}


def _has_digit(t: str) -> bool:
    return any(ch.isdigit() for ch in t)


def _match(t: str, have: set) -> bool:
    """``t`` occurs in ``have``, tolerating a unit glued to a number ("700" ~ "700s") and inflection
    by prefix for words of ≥ 4 letters ("mittwoch" ~ "mittwochs"; the stem of "moved" is "mov", of
    "move" "move")."""
    if t in have:
        return True
    if _has_digit(t):
        return any((h.startswith(t) and h[len(t):].isalpha()) or
                   (t.startswith(h) and _has_digit(h) and t[len(h):].isalpha()) for h in have)
    if len(t) == 3 and t + "e" in have:
        return True
    return len(t) >= 4 and any(len(h) >= 4 and (h.startswith(t) or t.startswith(h)) for h in have)


def values_in(claim: str, source: str) -> bool:
    """Every value token (with a digit) of ``claim`` occurs in ``source``."""
    have = _gtokens(source)
    return all(_match(t, have) for t in _gtokens(claim) if _has_digit(t))


def grounded(claim: str, source: str, *, min_overlap: float = GROUNDING_OVERLAP) -> bool:
    """≥ ``min_overlap`` of the claim's content tokens and every value token (with a digit) in ``source``."""
    want = _gtokens(claim)
    if not want:
        return False
    have = _gtokens(source)
    if not all(_match(t, have) for t in want if _has_digit(t)):
        return False
    return sum(_match(t, have) for t in want) / len(want) >= min_overlap


def _overlap(text: str, source: str) -> float:
    want = _gtokens(text)
    if not want:
        return 0.0
    have = _gtokens(source)
    return sum(_match(t, have) for t in want) / len(want)


@dataclass
class _Claim:
    text: str
    evidence: str          # one of gate_prompt.EVIDENCE
    supersedes: str = ""

    @property
    def label(self) -> str:
        if self.evidence == "assistant_reported":
            return REPORTED
        return OBSERVED if self.evidence in _OBSERVED_CLASSES else INFERRED


def _source_sentence(claim: str, text: str) -> str:
    """The sentence of ``text`` the claim was taken from: the one holding the claim's values first
    (a sarcastic line and its honest correction share words, not numbers), then best token overlap."""
    best, score = "", (False, 0.0)
    has_values = any(_has_digit(t) for t in _gtokens(claim))
    for sent in sentences(text, join_lines=False):
        key = (has_values and values_in(claim, sent), _overlap(claim, sent))
        if key > score:
            best, score = sent, key
    return best


def _new_values_from_user(claim: str, user_ctx: str, recall: str) -> bool:
    """Every value of ``claim`` is the user's or in recalled memory / L1, and — when it has values — at
    least one is the user's (a year or unit the model copied from memory does not make it ungrounded)."""
    nums = [t for t in _gtokens(claim) if _has_digit(t)]
    user, mem = _gtokens(user_ctx), _gtokens(recall)
    if not all(_match(t, user) or _match(t, mem) for t in nums):
        return False
    return not nums or any(_match(t, user) for t in nums)


def _values_said(claim: str, user: str) -> bool:
    """``claim`` has values and every one of them is in ``user`` (the user's text of this turn)."""
    nums = [t for t in _gtokens(claim) if _has_digit(t)]
    have = _gtokens(user)
    return bool(nums) and all(_match(t, have) for t in nums)


def _values_sourced(claim: str, src: Sources) -> bool:
    """rules-r2 guard: every value token of an inferred claim occurs in what the user said, tool
    output or recalled memory — not only in the assistant's own text (no invented numbers)."""
    return values_in(claim, "\n".join([src.user, src.previous_user, src.tools, src.recall]))


_CLAUSE_SPLIT = re.compile(r";\s")   # enumerations ("… mit X und Y") keep their owner: split at ";" only
# Third party (precision first, G3): rules-r2's _OTHER_PARTY on the claim and its source clause, read
# more narrowly than the rules do, because the gate rewrites "my" as "their" ("The user bought a print for
# their hallway") and users talk about their own pets, clients and family:
#   - "(not ours)", "not my business", "nicht unser …" always count;
#   - a rule someone imposes on the user's own work or home binds the user ("the client's legal team
#     mandated …", "my manager wants me to …", "the landlord's no-pets policy"): not third party;
#   - a person's possessive or a person as owner ("my colleague's server", "my sister-in-law's café",
#     "Mein Kollege Timo hat …") counts; for the user's own family only unless the user acts in the same
#     sentence ("my dad's knees are bad, so I need …"; "My brother's van has 210,000 km" stays out);
#   - a bare possessive pronoun ("his", "their", "seine") counts only without a first-person or "the
#     user" owner in the text and with a person noun it can point to;
#   - reported speech needs a speaker (a name, a pronoun, a person word; "backup.yaml still says …" or
#     "… and said" has none) and counts unless what was said is about the user ("they said the spot I
#     picked gets no sun"); "He says the staging box has 64 GB" stays hearsay.
_THING_WORDS = frozenset("still also file config log logs docs doc readme page table output it this that which manual "
                         "spec label sign website site wiki error message screen display chart dashboard".split())
_SPEECH = re.compile(r"^(.*?)\s*\b(?:said|says|told(?: me)?|mentioned|thinks)$", re.I)
_PERSON = (r"(?:friend|colleague|co-?worker|neighbou?r|boss|manager|wife|husband|partner|girlfriend|boyfriend|"
           r"brother|sister|mother|father|mom|mum|dad|son|daughter|cousin|uncle|aunt|roommate|flatmate|client|"
           r"customer|teammate|kid|child|nephew|niece|grandma|grandpa|doctor|landlord|trainer|coach|teacher)s?")
_PERSON_DE = (r"(?:freund(?:in)?|kolleg(?:e|in|en)|nachbar(?:in|n)?|chef(?:in)?|frau|mann|bruder|schwester|mutter|"
              r"vater|sohn|tochter|cousin[e]?|onkel|tante|mitbewohner(?:in)?|kund(?:e|in|en)|neffe|nichte|oma|opa|"
              r"(?:ä|ae)rzt(?:in)?|vermieter(?:in)?|trainer(?:in)?|lehrer(?:in)?)")
_OWN_VERBS = (r"(?:has|have|owns|uses|runs|hosts|keeps|built|builds|is building|bought|drives|lives|works|grows|plays|"
              r"got|set up|sets up|installed|hat|besitzt|nutzt|benutzt|betreibt|baut|f(?:ä|ae)hrt|wohnt|arbeitet|kauft|"
              r"kaufte|hat sich)")
# Another person is the subject of the fact: "My neighbour grows tomatoes", "The user's colleague Tom uses a
# Mac", "Meine Schwester wohnt in Graz" (unless the user acts too: "my sister and I …").
_OTHER_SUBJECT = re.compile(
    rf"^\W*(?:(?:by the way|also|btw|übrigens|uebrigens|and),?\s+)?(?:(?:my|our|a|his|her|their|the user's|the)\s+"
    rf"(?:[\w-]+\s+)?{_PERSON}(?:-in-laws?)?|(?:mein|meine|meinem|meinen|unser|unsere|des nutzers)\s+(?:[\w-]+\s+)?"
    rf"{_PERSON_DE})(?:\s+[A-ZÄÖÜ][\w-]+)?(?:'s\s+[\w-]+)?\s+(?!and I\b|und ich\b){_OWN_VERBS}\b", re.I)
# rules-r2's person possessive misses in-laws ("my sister-in-law's café", "my in-laws' house")
_IN_LAW = re.compile(rf"\b(?:my|our|his|her|their|the user's|a)\s+(?:[\w-]+\s+)?(?:{_PERSON}-in-laws?|in-laws?)(?:'s|')",
                     re.I)
_NOT_OURS = re.compile(r"\(?\bnot ours\b\)?|\bnot (?:our|my)\s+[\w-]+|\bnone of (?:my|our) business\b|"
                       r"\bnicht (?:unser\w*|bei uns|mein\w*)\b", re.I)
_PRONOUN_POSS = re.compile(r"^(?:his|her|their|sein|seine|seinem|seinen|seiner)\s", re.I)
_PERSON_NOUN = re.compile(rf"\b{_PERSON}(?:-in-laws?)?\b|\b{_PERSON_DE}\b", re.I)
_FAMILY = re.compile(r"\b(?:mother|father|mom|mum|dad|parents?|son|daughter|kids?|child|children|wife|husband|partner|"
                     r"brother|sister|grandma|grandpa|grand(?:mother|father)|aunt|uncle|cousin|nephew|niece|siblings?|"
                     r"mutter|vater|mama|papa|eltern|sohn|tochter|kind|kinder|frau|mann|bruder|schwester|oma|opa|tante|"
                     r"onkel|neffe|nichte|geschwister)(?:-in-laws?)?(?:'s|s)?\b", re.I)
# The user does something in the sentence (an opinion is not an action: "I just think it's too loud").
_USER_DOES = re.compile(r"\b(?:I|I'm|I've|I'll|I'd|[Ww]e|[Ww]e're|[Ww]e've|[Ww]e'll|[Ii]ch|[Ww]ir)\b(?!\s+(?:just\s+|really\s+|"
                        r"also\s+|still\s+)?(?:think|guess|feel|believe|suppose|reckon|find|mean|hope|denke|glaube|"
                        r"finde|meine|hoffe)\b)")
# The user (or their team) owns or does it: first person in the source ("I", "my", "our", "mein", "wir";
# not the objects "me"/"us": "Tom showed me his NAS") or "the user" in the gate's rewrite.
_USER_OWNER = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|[Mm]y|[Mm]ine|[Ww]e|[Ww]e're|[Ww]e've|[Ww]e'd|[Oo]ur|[Oo]urs|"
                         r"[Ii]ch|[Mm]ein\w*|[Ww]ir|[Uu]nser\w*)\b|\b(?i:the user|der nutzer|die nutzerin|des nutzers|"
                         r"dem nutzer|den nutzer)\b")
_USER_ANY = re.compile(_USER_OWNER.pattern + r"|\b(?:[Mm]e|[Uu]s|[Mm]ich|[Mm]ir|[Uu]ns)\b")
_SPEAKER_WORDS = frozenset("he she they someone somebody everyone everybody er sie man jemand".split())
_NOT_SPEAKERS = frozenset("and but so also then just only still or und aber auch dann noch".split())
# Someone with a say over the user's work or home, and the wording of a rule or requirement.
_AUTHORITY = re.compile(r"\b(?:client|customer|boss|manager|employer|landlord|landlady|teacher|school|company|vp|"
                        r"director|cto|ceo|cfo|supervisor|hr|legal|compliance|kund(?:e|in|en)|chef(?:in)?|"
                        r"vorgesetzt\w*|arbeitgeber\w*|vermieter\w*|firma)(?:'s|s)?\b", re.I)
_REQUIRE = re.compile(r"\b(?:mandat\w*|requir\w*|requests?|insist\w*|demand\w*|stipulat\w*|polic(?:y|ies)|rules?|must|"
                      r"has to|have to|needs? to|not allowed|forbid\w*|forbade|ban(?:s|ned)?|"
                      r"verlangt|fordert|vorgabe\w*|vorschrift\w*|regel\w*|verbot\w*|muss|m(?:ü|ue)ssen|darf nicht|"
                      r"d(?:ü|ue)rfen nicht)\b", re.I)
_BINDS_USER = re.compile(
    r"\b(?:wants?|wanted|asks?|asked|needs?|needed|expects?|expected|tells?|told|requires?|required|forces?|forced|"
    r"allows?|allowed|lets?|instructs?|instructed)\s+(?:me|us|the user|the team|our team)\s+(?:not\s+)?to\b|"
    r"\b(?:mandat\w*|requir\w*|insist\w*|demand\w*|stipulat\w*)\s+(?:that\s+)?(?:we|I|the user|our)\b|"
    r"\b(?:will|m(?:ö|oe)chte|verlangt|fordert|erwartet|besteht darauf|schreibt vor),?\s+dass\s+(?:wir|ich|der nutzer)\b|"
    r"\b(?:bittet|zwingt|verpflichtet)\s+(?:uns|mich|den nutzer)\b", re.I)


def _binds_user(text: str) -> bool:
    """A rule or requirement laid on the user's own work or home ("the client's legal team mandated that
    all project files live in their SharePoint", "my manager wants me to …", "Mein Chef will, dass wir …")."""
    return bool(_BINDS_USER.search(text) or (_AUTHORITY.search(text) and _REQUIRE.search(text)))


def _speaker(before: str) -> bool:
    """``before`` (the text before "said/says/…") ends with someone who speaks: a capitalised name, a
    pronoun or a person word; not a thing ("backup.yaml still says") or a conjunction ("… and said")."""
    last = (before.split() or [""])[-1]
    if not last or re.search(r"[./]", last) or last.lower() in _THING_WORDS or last.lower() in _NOT_SPEAKERS:
        return False
    return (last[0].isupper() or last.lower() in _SPEAKER_WORDS
            or bool(re.fullmatch(_PERSON, last, re.I) or re.fullmatch(_PERSON_DE, last, re.I)))


def _third_party(text: str, *, family_ok: bool = False) -> bool:
    """``text`` states another person's own thing (see the comment above). ``family_ok``: the source
    sentence has the user acting, so a family member's fact is the user's context."""
    text = text or ""
    if _NOT_OURS.search(text):
        return True
    if _binds_user(text):
        return False
    owner = bool(_USER_OWNER.search(text))
    m = _OTHER_SUBJECT.search(text)
    if m and not (family_ok and _FAMILY.search(m.group(0))):
        return True
    for m in list(_OTHER_PARTY.finditer(text)) + list(_IN_LAW.finditer(text)):
        hit = m.group(0)
        if _PRONOUN_POSS.match(hit):
            if owner or not _PERSON_NOUN.search(text[:m.start()] + " " + text[m.end():]):
                continue   # "their hallway" in the rewrite of "my hallway"; "their fiscal year" of a company
            return True
        sp = _SPEECH.match(hit)
        if sp:
            if not _speaker(sp.group(1)) or _USER_ANY.search(text[m.end():]):
                continue   # a thing that "says", or what was said is about the user
            return True
        if family_ok and _FAMILY.search(hit):
            continue
        return True
    return False


# Wishes and irrealis about the user's own life (not facts; preferences may still be wishes):
_IRREALIS = re.compile(
    r"\b(?:some ?day|one day|(?<!asked )(?<!asks )(?<!wondered )(?<!wonders )if i (?:ever|get|got|had|win|won|were|could)|"
    r"dream(?:ing)? (?:of|about)|i wish i|"
    r"on my bucket list|irgendwann(?: mal)?|eines tages|falls ich|wenn ich (?:mal|je|irgendwann)|"
    r"ich tr(?:ä|ae)ume davon)\b", re.I)
# only on the claim itself: the same source sentence often carries a real fact plus "I'd like to …"
_WISH_CLAIM = re.compile(r"\b(?:would (?:love|like) to|'d (?:love|like) to|wants? to someday|w(?:ü|ue)rde (?:gern|gerne))\b",
                         re.I)


_USER_SUBJECT = re.compile(r"^\W*(?:the user|user|i|we|der nutzer|die nutzerin|ich|wir)\s+(?!'s\b)[a-zäöü]", re.I)
# a first-person subject, also in the diary style without one ("Just had the call with …", "Finished my review")
_FIRST_PERSON_SUBJ = re.compile(r"\b(?:I|I'm|I've|I'd|we|we're|we've|ich|wir)\b|"
                                r"^\W*(?:(?:just|finally|also|then|today|yesterday|recently)\s+)?"
                                r"(?:got|had|went|met|made|took|spent|bought|found|came|sent|saw|picked|wrapped up|"
                                r"\w{3,}ed)\b(?!\s+(?:is|are|was|were|has|have)\b)", re.I)


def _user_acts(claim: str, origin: str, user_text: str = "") -> bool:
    """The user is the actor ("The user got their sister a dress", "I met Sophia …") and the source
    sentence has a first-person subject: other people are only mentioned, the fact is the user's. With
    ``user_text``: or another sentence of it that holds half of the claim does ("My friend loved the
    gift! The mugs I made for her came out great." → "The user made mugs for a friend's gift")."""
    if not _USER_SUBJECT.match(claim or ""):
        return False
    if _FIRST_PERSON_SUBJ.search(origin or ""):
        return True
    return any(_FIRST_PERSON_SUBJ.search(s) and _overlap(claim, s) >= 0.5
               for s in sentences(strip_today_line(user_text), join_lines=False)) if user_text else False


# Not a statement of fact (G3, EN + DE): hypotheticals and undecided ideas, proposals, jokes and
# sarcasm. rules-r2's _HEDGE covers maybe/might/probably/hypothetically/would cost/… already.
# Proposals and hypotheticals bind their own clause ("We could move it, but the price is 600 euros");
# "undecided" markers, jokes and sarcasm the whole sentence ("The sync moves to Thursdays, nothing's
# decided"). "We should only / always / never …" is a standing team rule, not a proposal.
_PROPOSAL = (r"\b(?:if we (?:ever|were to|did|had)|what if|in case we ever|we could|should we|"
             r"we should(?!\s+(?:only|always|never|not|all)\b)|shall we|how about|was w(?:ä|ae)re,? wenn|falls wir|"
             r"wenn wir (?:mal|je|irgendwann)|sollten wir|wir k(?:ö|oe)nnten|wie w(?:ä|ae)re es)\b")
_UNDECIDED = (r"\b(?:thinking out loud|just thinking\s*[:,—-]|just (?:asking|wondering|an idea)|"
              r"nothing(?:'s| is| has been)? (?:planned|decided|settled|booked)(?: yet)?|not (?:yet )?decided|"
              r"undecided|no plans? (?:yet|to)|laut gedacht|nur (?:mal )?(?:so )?(?:gefragt|eine idee|überlegt|ueberlegt)|"
              r"nichts (?:ist )?(?:geplant|entschieden|beschlossen|fix)|noch (?:nicht|nichts) "
              r"(?:entschieden|beschlossen|geplant))\b|"
              r"^\W*(?:oh,?\s+(?:sure|great|wonderful|joy|yeah|fantastic|lovely|brilliant|perfect)|yeah,?\s+right|"
              r"(?:yeah|right|sure),?\s+because|as if|ja,?\s+genau|na\s+klar\s+doch|wie\s+toll|ha(?:ha)*|lol|"
              r"lmao|ja,?\s+klar|na\s+(?:toll|super|klasse|prima))\b|"
              r"\b(?:just kidding|kidding|only joking|haha|lol|lmao|/s|kleiner scherz|nur ein scherz|spa(?:ß|ss) beiseite|"
              r"just what i needed|genau was ich brauchte)\b|"
              r",\s*ha\s*[.!]*\s*$")
_NOT_A_FACT = re.compile(_PROPOSAL + "|" + _UNDECIDED, re.I)
_PROPOSAL_RE = re.compile(_PROPOSAL, re.I)
_UNDECIDED_RE = re.compile(_UNDECIDED, re.I)
# Clauses for the hedge / proposal / wish checks: at ";", ":", a dash, and before a conjunction that
# starts a new clause ("The price is $600, and I think I'll get it": the price is a plain assertion).
_OWN_CLAUSE_SPLIT = re.compile(
    r";\s+|:\s+|\s[—–-]\s|—|,\s+(?=(?:and|but|so|although|though|while|because|whereas|yet|which|who|saying|"
    r"adding|noting|hoping|thinking|aber|und|doch|also|weil|obwohl|denn|sondern)\b)|"
    r"\s+(?=(?:and|but|so|aber|und|doch)\s+(?:I|I'm|I'll|I'd|we|we're|we'll|ich|wir)\b)")


def _clause(claim: str, sentence: str) -> str:
    """The clause of ``sentence`` the claim was taken from (best overlap)."""
    parts = [p for p in _CLAUSE_SPLIT.split(sentence or "") if p and p.strip()]
    return max(parts, key=lambda p: _overlap(claim, p), default="")


# A concessive clause is a clause of its own: "while my handwriting can be messy, my strongest skill is …"
# (its "can be" does not hedge the main clause).
_CONCESSIVE = re.compile(r"\b(?:while|whilst|although|though|even though|even if|whereas|obwohl|obgleich|auch wenn|"
                         r"selbst wenn)\b[^,;.!?]*,", re.I)
# Firm evaluations and finds are not hedges: "I think this course is exactly what we need", "I found a
# workshop that would be perfect for the team" state the course / the workshop outright.
_FIRM_VIEW = re.compile(
    r"\bI (?:really |honestly |truly |definitely )?(?:think|believe|feel)\b(?=[^.;!?]*\b(?:is|are|'s|was|would be)\s+"
    r"(?:exactly|just|precisely|definitely|absolutely|really|clearly)?\s*(?:what (?:we|I|the team|they|you) "
    r"(?:need|needed|want|wanted|are looking for|were looking for)\b|perfect\b|ideal\b|spot on\b|"
    r"the (?:best|right|perfect|ideal)\b|a (?:great|perfect|good|solid) (?:fit|choice|match|option)\b))", re.I)
_FIND_EVAL = re.compile(r"\b(?:that|which|it|this)\s+(?:would|could|might|will)\s+be\s+(?:just\s+|really\s+|absolutely\s+)?"
                        r"(?:perfect|ideal|great|a (?:great|perfect|good) fit|useful|helpful|exactly what (?:we|I) need)\b",
                        re.I)


def _firm(text: str) -> str:
    """``text`` without firm evaluations and finds (:data:`_FIRM_VIEW`), for the hedge check."""
    return _FIND_EVAL.sub(" ", _FIRM_VIEW.sub(" ", text or ""))


# A trailing reported / reason clause of a claim ("The user sold the kayak, saying they would probably
# rent one instead"): a hedge there qualifies the reason, not the fact in front of it.
_TAIL_CLAUSE = re.compile(r",\s+(?:saying|adding|telling|noting|explaining|because|since|which)\b", re.I)


def _hedge_free_head(claim: str) -> str:
    """``claim`` cut before its first trailing clause (:data:`_TAIL_CLAUSE`) when only that tail hedges;
    "" otherwise (no tail, a hedged head, or a head too short to be a claim)."""
    m = _TAIL_CLAUSE.search(claim or "")
    if not m:
        return ""
    head = claim[:m.start()].rstrip(" ,;")
    if len(head.split()) < 3 or _HEDGE.search(_firm(head)) or not _HEDGE.search(_firm(claim[m.start():])):
        return ""
    return head + "."


def _own_clause(claim: str, sentence: str) -> str:
    """The clause of ``sentence`` holding the claim, for the hedge / proposal / wish checks: the one with
    all its values first, then best overlap (:data:`_OWN_CLAUSE_SPLIT`; a concessive clause is split off)."""
    sentence = _CONCESSIVE.sub(lambda m: m.group(0) + "; ", sentence or "")
    parts = [p for p in _OWN_CLAUSE_SPLIT.split(sentence) if p and p.strip()]
    has_values = any(_has_digit(t) for t in _gtokens(claim))
    return max(parts, key=lambda p: (has_values and values_in(claim, p), _overlap(claim, p)), default="")


# A question can still carry the user's own facts when it asks for help ("Can you suggest a catalog
# system for my 57 rare records?", "… considering my commute is 40 minutes each way?"): a request
# lead, no embedded question about the fact ("whether", "which", "ob"), and the claim's part is first person.
_REQUEST = re.compile(r"^\W*(?:(?:hey|hi|so|ok(?:ay)?|also|and|but|now|oh)\W+)*(?:(?:can|could|would|will) you\b|"
                      r"do you have (?:any|some)\b|any (?:tips|ideas|suggestions|recommendations|advice)\b|"
                      r"(?:what|which) (?:are|is|would be) (?:some|a few|the best|a good|good|(?:the )?(?:best|easiest|"
                      r"simplest))\b|how (?:can|do|should|could) I\b|"
                      r"(?:kannst|k(?:ö|oe)nntest|w(?:ü|ue)rdest) du\b|hast du (?:tipps|ideen|vorschl(?:ä|ae)ge)\b)", re.I)
_EMBEDDED_Q = re.compile(r"\b(?:whether|if|ob|falls|welche\w*|wann|wie ?viel\w*)\b|"
                         r"\b(?:what|which|where|when|who|how)\b(?!\s+to\b)", re.I)
_OWN_PART_SPLIT = re.compile(r",\s+|\s+(?=(?:for|about|with|considering|since|because|as|f(?:ü|ue)r|mit|da)\s+"
                             r"(?:my|our|I|mein\w*|unser\w*|ich)\b)")
_FIRST_PERSON = re.compile(r"\b(?:[Mm]y|[Oo]ur|I|I'm|I've|[Ww]e|[Ww]e're|[Mm]ein\w*|[Uu]nser\w*|[Ii]ch|[Ww]ir)\b")
# The user's own statement inside a question sentence, whatever is asked around it:
#   - a first-person clause of its own ("I've been sneezing a lot lately, could it be the dust?"; not "I wonder
#     …", "I'd …", nor the clause that asks: "Do you think I'm overtraining?");
#   - a first-person relative clause ("… series like Dark, which I just finished?"): its verb is in the claim;
#   - "like <Name>" for someone the user talks about as theirs ("I'm getting Nala a bed. Which size fits a
#     Labrador like Nala?"): the name is in a first-person sentence of this turn or follows "my";
#   - in a request, a circumstance next to the user's own words ("… keep my hallway clean, especially with
#     a dog that sheds?"): "with / having / given a …" when the request also says "my" / "I".
_Q_PART_SPLIT = re.compile(r",\s+|;\s+|\s[—–-]\s|\s+(?=(?:since|because|as|now that|given that)\s+(?:I|we)\b)")
_NOT_A_STATEMENT = (r"(?:think|thinking|thought|wonder\w*|guess|want\w*|need\w*|hope\w*|feel|believe|mean|curious|"
                    r"not sure|unsure|asking|trying to (?:understand|figure|decide|find out)|would|could|should|"
                    r"might|may|can|will|must)\b")
_OWN_CLAUSE_Q = re.compile(r"^[^\w'\"‘’“”]*(?:(?:and|but|so|also|since|because|as|now that|given that)\s+)?(?:I|[Ww]e)"
                           r"(?:'ve|'m|'re| have| am| are| had| was| were)?\s+(?:just\s+|recently\s+|already\s+|also\s+|"
                           r"finally\s+|really\s+)?(?!(?:am|are|was|were|have|had)\b|" + _NOT_A_STATEMENT + r")[a-z]\w*")
_OWN_RELATIVE = re.compile(r"(?:,\s*(?:which|who|whom)\s+(?:I|we)(?:'ve| have| had)?\s+(?:(?:just|recently|already|also|"
                           r"finally)\s+)?|\b(?:which|that)\s+(?:I|we)(?:'ve| have| had)?\s+(?:just|recently|already|"
                           r"finally)\s+)(?!" + _NOT_A_STATEMENT + r")([a-z]\w*)")
_LIKE_NAME = re.compile(r"\blike\s+([A-ZÄÖÜ][a-zäöüß]+)\b(?!\s*(?:[A-ZÄÖÜ]|'s\b|’s\b))")
_CIRCUMSTANCE = re.compile(r"^\W*(?:(?:especially|particularly|and|also|even)\s+)?(?:with|having|given|considering)\s+"
                           r"(?:a|an|two|three|four|five|\d+)\b", re.I)


def _own_fact_in_question(claim: str, lead: str, user_text: str = "") -> bool:
    """The claim's part of the question ``lead`` is the user's own statement (see above)."""
    parts = [p for p in _Q_PART_SPLIT.split(lead) if p and p.strip()]
    if not parts:
        return False
    part = max(parts, key=lambda p: _overlap(claim, p))
    if not part.rstrip().endswith("?") and _OWN_CLAUSE_Q.match(part) and _overlap(claim, part) >= 0.5:
        return True
    want = _gtokens(claim)
    for m in _OWN_RELATIVE.finditer(lead):
        verb = stem(m.group(1).lower())
        if any(_match(t, {verb}) for t in want) and grounded(claim, lead):
            return True
    for m in _LIKE_NAME.finditer(lead):
        name = m.group(1)
        if not re.search(rf"\b{re.escape(name)}\b", claim):
            continue
        own = re.search(rf"\b(?:[Mm]y|[Oo]ur)\s+(?:[\w-]+\s+){{0,2}}{re.escape(name)}\b", lead) or any(
            s.strip() != lead.strip() and re.search(rf"\b{re.escape(name)}\b", s) and _FIRST_PERSON.search(s)
            for s in sentences(strip_today_line(user_text), join_lines=False))
        if own and grounded(claim, user_text or lead):
            return True
    return False


def _asks_only(claim: str, sentence: str, user_text: str = "", *, own_facts: bool = True) -> bool:
    """``sentence`` is a question the claim cannot be taken from (rules-r2: declarative sentences only);
    the exceptions: the user's own facts inside a request for help (:data:`_REQUEST`) and, with
    ``own_facts`` (facts, not preferences), the user's own statement inside the question
    (:func:`_own_fact_in_question`)."""
    if not (sentence or "").rstrip().endswith("?"):
        return False
    lead = strip_lead(sentence) or sentence
    if own_facts and _own_fact_in_question(claim, lead, user_text):
        return False
    req = _REQUEST.match(lead)
    if not req or _EMBEDDED_Q.search(lead[req.end():]):
        return True
    rest = lead[req.end():]
    parts = [p for p in _OWN_PART_SPLIT.split(rest) if p and p.strip()]
    part = max(parts, key=lambda p: _overlap(claim, p), default="")
    if _FIRST_PERSON.search(part):
        return False
    return not (own_facts and _CIRCUMSTANCE.match(part) and _FIRST_PERSON.search(rest)
                and _USER_SUBJECT.match(claim or ""))


def _user_says(claim: str, src: Sources, *, preference: bool = False) -> bool:
    """``claim`` is grounded in this turn's user text (not the previous turn, not memory) and not taken
    from a question."""
    return grounded(claim, src.user) and not _asks_only(claim, _source_sentence(claim, src.user), src.user,
                                                        own_facts=not preference)


def _not_the_assistants(claim: str, src: Sources) -> bool:
    """The claim's content words that are not in the user's text of this turn are not in the assistant's
    reply either: the model rephrased what the user said; it did not take over the assistant's conclusion
    ("I love the new flat" + "So you moved recently!" → "The user moved to a new flat" stays inferred)."""
    have_user, have_assistant = _gtokens(src.user), _gtokens(src.assistant)
    return not any(_match(t, have_assistant) for t in _gtokens(claim) if not _match(t, have_user))


def _users_sentence(claim: str, user_text: str, *, preference: bool) -> str:
    """The user's own sentence for a claim that does not ground in it (the model translated or
    paraphrased too freely): for a fact, the declarative sentence holding all the claim's values; for
    a preference, the one sentence with standing/preference wording (or the one holding its values)."""
    nums = [t for t in _gtokens(claim) if _has_digit(t)]
    cands = [x for x in sentences(strip_today_line(user_text), join_lines=False) if not x.rstrip().endswith("?")]
    if preference:
        prefs = [x for x in cands if _STANDING.search(x) or _PREF_ANY.search(x) or _PREF_START.match(strip_lead(x))]
        if nums:
            prefs = [x for x in prefs if values_in(claim, x)] or prefs
        pick = prefs[0] if len(prefs) == 1 else ""
        return _PREF_LEAD.sub("", strip_lead(pick)).strip() if pick else ""
    if not nums:
        return ""
    hits = [x for x in cands if values_in(claim, x)]
    return strip_lead(hits[0]).strip() if len(hits) == 1 else ""


# -- assistant_reported (T3) ------------------------------------------------------------------------------
# What the assistant said it did, decided or found in this turn ("Final fix: raised the pool timeout to
# 45s", "I set up the dashboard", "Deployed everywhere.") is kept, labelled and low-confidence, instead of
# being dropped for lack of a tool output. Reports only: no offers, plans, advice, hedges, questions,
# memory talk or general knowledge (F8).
_REPORT = re.compile(
    r"^\W*(?:decided|decision|final fix|fix(?:ed)?|done|resolved|result|results|outcome|status|"
    r"entschieden|entscheidung|erledigt|ergebnis|l(?:ö|oe)sung)\s*[:—–-]|"
    r"\b(?:I|we)(?:'ve| have|'d| had)?\s+(?:just\s+|now\s+|also\s+|already\s+|finally\s+)?"
    r"(?:\w+ed|set|set up|built|wrote|ran|made|put|sent|took|found|chose|picked|shut|cut|split|bought|paid|left)\b|"
    r"^\W*(?:\w+ed|set up|built|wrote|ran|made|put|sent|found|chose|picked|shut down|cut|split)\b|"
    r"^\W*(?!(?:you|the user|user|he|she|they|du|sie|er)\b)\w+(?:[ -]\w+)?\s+(?:\w+ed|done|set up|written|built|sent|made)\b|"
    r"\b(?:is|are|was|were|has been|have been|now)\s+(?:now\s+)?(?:done|complete|completed|finished|deployed|"
    r"configured|installed|set up|booked|scheduled|live|running|fixed|resolved|merged|created|enabled|disabled|"
    r"updated|raised|lowered|in place)\b|"
    r"\b(?:ich habe|habe ich|wir haben|haben wir|ist jetzt|sind jetzt|wurde|wurden)\b|"
    r"^\W*(?:erledigt|eingerichtet|konfiguriert|installiert|erstellt|angelegt|gebucht|bestellt|ge(?:ä|ae)ndert|"
    r"aktualisiert|behoben|ausgerollt|gel(?:ö|oe)scht|entfernt|hinzugef(?:ü|ue)gt|geschrieben)\b", re.I)
_OFFER = re.compile(
    r"\b(?:I'll|I will|I can|I could|I'd|I would|let me|shall I|should I|want me to|would you like|do you want|"
    r"if you (?:want|like|prefer)|you (?:could|can|should|might|may)|I (?:suggest|recommend|propose|advise)|"
    r"we (?:could|should|can)|will|going to|plan(?:ning)? to|next step|"
    r"ich werde|ich kann|ich k(?:ö|oe)nnte|soll ich|m(?:ö|oe)chtest du|willst du|du (?:kannst|k(?:ö|oe)nntest|solltest)|"
    r"ich (?:empfehle|schlage)|wir (?:k(?:ö|oe)nnten|sollten)|werde[n]?|wird)\b", re.I)
_WORLD = re.compile(
    r"\b(?:typically|usually|generally|in general|commonly|normally|by default|is known (?:as|for)|known as|"
    r"stands for|refers to|is designed|is a (?:type|kind|form|popular|common)|found in|such as|for example|"
    r"e\.g\.|in most|most people|normalerweise|(?:ü|ue)blicherweise|in der regel|meistens|standardm(?:ä|ae)(?:ß|ss)ig|"
    r"bekannt als|steht f(?:ü|ue)r|zum beispiel|z\.\s?b\.)|\s[—–]\s(?:the|a|an|der|die|das|ein|eine)\s", re.I)
MAX_REPORTED_CHARS = 240


def _reportable(body: str, src: Sources) -> bool:
    """``body`` is the assistant's report of this turn: grounded in its own text (every value), from a
    declarative sentence that reports an action, decision or result — or states a typed value about
    something this conversation already named — and not an offer / plan / advice, a hedge, memory
    talk or general knowledge."""
    if not src.assistant.strip() or len(body) > MAX_REPORTED_CHARS or not grounded(body, src.assistant):
        return False
    if src.tools.strip() and not _values_sourced(body, src):
        return False   # a tool ran and does not show the value: the assistant made it up

    sent = _source_sentence(body, src.assistant)
    if not sent or sent.rstrip().endswith("?") or meta_talk(body) or meta_talk(sent):
        return False
    if _OFFER.search(sent) or _HEDGE.search(sent) or _HEDGE.search(body) or _NOT_A_FACT.search(sent):
        return False
    if _WORLD.search(sent) or _WORLD.search(body):
        return False
    if _REPORT.search(sent):
        return True
    context = "\n".join([src.user, src.previous_user, src.previous_assistant, src.tools])
    return bool(values(body).slots) and bool(distinctive_tokens(body) & _gtokens(context))


# The user corrects something ("actually", "not 12 but 14", "I meant", "correction") or says it changed.
_CORRECTION = re.compile(r"\b(?:actually|correction|i meant|i mean|my mistake|mistake|wrong|typo|not\b[^.;!?]{0,40}\bbut|"
                         r"instead|rather|no longer|any ?more|eigentlich|korrektur|falsch|stattdessen|nicht mehr)\b|"
                         + _CHANGE.pattern + "|" + RETRACT_VERB.pattern, re.I)


def _check_claim(raw: Any, src: Sources, has_previous: bool, notes: List[str], secrets: List[str],
                 third_party_guard: bool = True, preference: bool = False) -> Optional[_Claim]:
    if not isinstance(raw, dict):
        return None
    text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip()
    evidence = str(raw.get("evidence") or "assistant_inferred")
    if evidence not in gp.EVIDENCE or evidence == "assistant_reported":   # reported: set by the checks below only
        evidence = "assistant_inferred"
    text, kinds = redact(text)
    if kinds:
        secrets.extend(kinds)
        notes.append(f"secret redacted in claim ({', '.join(kinds)})")
    text = strip_lead(text) or text
    if not text:
        return None
    ok = True
    user_ctx = src.user + "\n" + src.previous_user
    if evidence == "user_stated":
        ok = grounded(text, user_ctx) and _overlap(text, src.user) > 0
        if not ok and _new_values_from_user(text, user_ctx, src.recall) and _overlap(text, src.user) >= 0.3 \
                and grounded(text, user_ctx + "\n" + src.recall):
            # an update phrased with the context the user did not repeat ("the Halvorsen deadline got
            # extended to 13 November" → "The Halvorsen Foundation grant proposal is due 13 November"):
            # the new values are the user's, the rest is from recalled memory / L1. A restatement of
            # memory is no update, unless every value of it is in what the user says now.
            ok = _overlap(text, src.recall) < 0.9 or not values_in(text, src.recall) or _values_said(text, src.user)
        if ok and _asks_only(text, _source_sentence(text, src.user), src.user, own_facts=not preference):
            ok = False   # a question is not a statement (rules-r2: declarative sentences only)
    elif evidence == "user_confirmed":
        ok = has_previous and grounded(text, src.previous_assistant + "\n" + src.user)
    elif evidence == "tool_observed":
        ok = grounded(text, src.tools, min_overlap=TOOL_GROUNDING_OVERLAP)
    # The user's own words under another label (the model called a fact of this turn "confirmed", "tool"
    # or "inferred"): relabelled user_stated instead of demoted and dropped as an echo of the user text.
    # Not for inferred preferences (authorship: the model's reading of a wish is not the user's rule).
    if ((not ok and evidence in ("user_confirmed", "tool_observed"))
            or (evidence == "assistant_inferred" and not preference and _not_the_assistants(text, src))) \
            and _user_says(text, src, preference=preference):
        notes.append(f"{evidence} claim grounded in this turn's user text → user_stated: {shorten(text, 60)}")
        evidence, ok = "user_stated", True
    if not ok and evidence in ("user_stated", "user_confirmed"):
        own = _users_sentence(text, src.user, preference=preference)
        if own:
            notes.append(f"claim replaced by the user's own sentence: {shorten(text, 50)} → {shorten(own, 50)}")
            text, ok, evidence = own, True, "user_stated"
    if not ok:
        notes.append(f"{evidence} claim not grounded in its source → assistant_inferred: {shorten(text, 60)}")
        evidence = "assistant_inferred"
    # rules-r2 precision guards (never weaker than the fallback): hedges / hypotheticals and facts
    # attributed to third parties, checked on the claim and on the user sentence it came from. Hedges,
    # proposals and wishes on the claim's own clause ("The price is $600, and I think I'll get it");
    # undecided markers and jokes on the whole sentence.
    sentence = _source_sentence(text, src.user) if evidence in ("user_stated", "user_confirmed") else ""
    origin, own = _clause(text, sentence), _own_clause(text, sentence)
    head = _hedge_free_head(text) if _HEDGE.search(_firm(text)) else ""
    if head:
        # the user's fact stands, the hedged reason after it goes
        origin, own = _clause(head, sentence), _own_clause(head, sentence)
        if not (own and _HEDGE.search(_firm(own))):
            notes.append(f"hedged trailing clause trimmed: {shorten(text, 60)} → {shorten(head, 50)}")
            text = head
    if _HEDGE.search(_firm(text)) or (own and _HEDGE.search(_firm(own))):
        notes.append(f"hedged/hypothetical claim dropped: {shorten(text, 60)}")
        return None
    if evidence == "user_stated" and (_NOT_A_FACT.search(text) or (own and _PROPOSAL_RE.search(own))
                                      or (sentence and _UNDECIDED_RE.search(sentence))):
        notes.append(f"hypothetical/proposal/joke claim dropped: {shorten(text, 60)}")
        return None
    if evidence == "user_stated" and not preference and (_IRREALIS.search(text) or _WISH_CLAIM.search(text)
                                                         or (own and _IRREALIS.search(own))):
        notes.append(f"wish/irrealis claim dropped: {shorten(text, 60)}")
        return None
    family_ok = bool(sentence and _USER_DOES.search(sentence))
    if third_party_guard and not _user_acts(text, origin, src.user if sentence else "") and (
            _third_party(text, family_ok=family_ok) or (origin and _third_party(origin, family_ok=family_ok))):
        notes.append(f"third-party claim dropped: {shorten(text, 60)}")
        return None
    if evidence == "assistant_inferred":
        body = _unack(text)
        if not grounded(body, src.everything):
            notes.append(f"ungrounded claim dropped: {shorten(text, 60)}")
            return None
        if _recalled(body, token_set(src.recall)) or _recalled(body, token_set(src.user)):
            notes.append(f"echo of recalled/L1/user text dropped: {shorten(text, 60)}")
            return None
        if not _inferred_ok(body) or not _values_sourced(body, src):
            if not _reportable(body, src):
                why = ("meta/first-person/long inferred claim dropped" if not _inferred_ok(body)
                       else "inferred claim with a value nobody else stated dropped")
                notes.append(f"{why}: {shorten(text, 60)}")
                return None
            notes.append(f"kept as assistant_reported: {shorten(body, 60)}")
            evidence = "assistant_reported"
        text = body
    supersedes = redact(re.sub(r"\s+", " ", str(raw.get("supersedes") or "")).strip())[0]
    if supersedes and not grounded(supersedes, "\n".join([src.previous_user, src.previous_assistant, src.recall,
                                                           src.user])):
        notes.append(f"old statement not found in context, not superseded: {shorten(supersedes, 60)}")
        supersedes = ""
    if supersedes and different_occasion(text, supersedes) and not _CORRECTION.search(src.user):
        # another outing with its own date ("we caught 9 on 6/19" next to "caught 7 on 6/12") adds a fact
        notes.append(f"old statement is about another occasion, not superseded: {shorten(supersedes, 60)}")
        supersedes = ""
    return _Claim(_clean(text), evidence, supersedes)


def _subject(raw: Any, claims: Sequence[_Claim], src: Sources) -> str:
    subj = re.sub(r"\s+", " ", str(raw or "")).strip(" .,:;\"'`")
    if subj and len(subj.split()) <= 6 and len(subj) <= 60 and _overlap(subj, src.everything) >= SUBJECT_OVERLAP:
        return display_subject(subj)
    for c in claims:
        s = subject_of(c.text)
        if s:
            return s
    return ""


@dataclass
class _Draft:
    mtype: MemoryType
    domain: str
    claims: List[_Claim]
    raw: Dict[str, Any]
    subject: str = ""


def post_check(decision: Dict[str, Any], episode: Episode, previous: Optional[Episode], gate_input: GateInput,
               notes: List[str], *, day: Optional[Any] = None) -> List[MemoryCandidate]:
    """GateDecision → MemoryCandidates (always ≥ 1: a noise record when nothing survives)."""
    src = gate_input.sources
    why = shorten(str(decision.get("why") or ""), 120, title=False)
    record_conf = _num(decision.get("record_confidence"), 0.5)
    secrets = list(gate_input.secrets)
    raw_cands = [c for c in decision.get("candidates") or [] if isinstance(c, dict)]
    if not decision.get("record"):
        if raw_cands:
            notes.append(f"record=false: {len(raw_cands)} candidate(s) ignored")
        raw_cands = []
    drafts: List[_Draft] = []
    for raw in raw_cands[:gp.MAX_CANDIDATES]:
        kind, domain = str(raw.get("kind") or ""), str(raw.get("domain") or "other")
        if domain not in gp.DOMAINS:
            domain = "other"
        mtype = _KIND_TYPE.get(kind) or (_FACT_TYPE[domain] if kind == "fact" else None)
        if mtype is None:
            notes.append(f"unknown kind {kind!r} dropped")
            continue
        claims = [c for c in (_check_claim(x, src, previous is not None, notes, secrets,
                                           third_party_guard=mtype is not MemoryType.USER_PREFERENCE,
                                           preference=mtype is MemoryType.USER_PREFERENCE)
                              for x in (raw.get("claims") or [])[:gp.MAX_CLAIMS]) if c is not None]
        if mtype is MemoryType.USER_PREFERENCE:
            dropped = [c for c in claims if c.evidence != "user_stated"]
            if dropped:
                notes.append(f"preference: {len(dropped)} claim(s) not user_stated dropped")
            claims = [c for c in claims if c.evidence == "user_stated"]
        if not claims:
            notes.append(f"{kind} candidate without surviving claims dropped")
            continue
        if not any(c.label == OBSERVED for c in claims):
            inferred = [c for c in claims if c.label == INFERRED]
            support = src.tool_calls_ok > 0 and all(_overlap(c.text, src.tools) >= TOOL_SUPPORT_OVERLAP for c in inferred)
            if inferred and (domain == "other" or not support):
                # F8 (no ungrounded world knowledge) stays: only the assistant's own reports survive
                kept = []
                for c in claims:
                    if c.label == INFERRED and _reportable(c.text, src):
                        notes.append(f"kept as assistant_reported (no tool support): {shorten(c.text, 60)}")
                        c.evidence = "assistant_reported"
                    if c.label != INFERRED:
                        kept.append(c)
                    else:
                        notes.append(f"inferred claim without tool support dropped: {shorten(c.text, 60)}")
                claims = kept
                if not claims:
                    notes.append(f"inferred-only {kind} candidate without tool support dropped")
                    continue
        draft = _Draft(mtype, domain, claims, raw, _subject(raw.get("subject"), claims, src))
        same = next((d for d in drafts if d.mtype is mtype and (mtype is MemoryType.USER_PREFERENCE
                                                                 or normalize(d.subject) == normalize(draft.subject))),
                    None)
        if same is not None:
            same.claims.extend(draft.claims)
        else:
            drafts.append(draft)

    out: List[MemoryCandidate] = []
    used_ids: set = set()
    for d in drafts:
        out.append(_candidate(d, episode, previous, record_conf, why, used_ids, day=day))
    sensitivity = str(decision.get("sensitivity") or "none")
    if secrets:
        sensitivity = "secret"
    decision["sensitivity"] = sensitivity if sensitivity in gp.SENSITIVITY else "none"
    if not out:
        reason = f"record=false: {why}" if not decision.get("record") else f"all candidates dropped: {why}"
        noise = MemoryCandidate(event_ids=list(episode.event_ids), classification=classification(MemoryType.NOISE),
                                normalized_claims=[], retrieval_query="", id=f"{episode.id}:noise",
                                session_id=episode.session_id, rationale=f"{GATE_VERSION}: {reason}")
        noise.classification.confidence = record_conf if not decision.get("record") else 0.5
        out.append(noise)
    return out


def claim_day(episode: Episode, previous: Optional[Episode], offset_days: Optional[int] = None):
    """Session date for the episode's claims (:mod:`pan.memory.dates`): the turn's event timestamp;
    a date the user states in this turn wins; otherwise the offset of a date the user stated earlier
    in the session (``offset_days``, or the previous turn's) shifts the event date."""
    turn = episode.turn
    ts = turn.ts if turn is not None else ""
    if stated_date(episode.user_text) is None and offset_days is None and previous is not None \
            and previous.turn is not None:
        offset_days = stated_offset(previous.user_text, previous.turn.ts)
    return claim_date(episode.user_text, ts, offset_days=offset_days)


# "2026-09-27", "2026-09-27T04:29:57"; clock times "04:29" / "04:29:57"
_ISO_DATE = re.compile(r"(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-])")
_CLOCK = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")


def _dated_record(text: str) -> bool:
    """The claim says when something happened: a date or a clock time (not only a relative word)."""
    plain = bare(text)
    slots = values(plain).slots
    return bool(_ISO_DATE.search(plain) or _CLOCK.search(plain) or "date" in slots or "time" in slots)


def _candidate(d: _Draft, episode: Episode, previous: Optional[Episode], record_conf: float, why: str,
               used_ids: set, *, day: Optional[Any] = None) -> MemoryCandidate:
    seen, claims = set(), []
    for c in d.claims:
        key = normalize(c.text)
        if key and key not in seen:
            seen.add(key)
            claims.append(c)
    claims = claims[:gp.MAX_CLAIMS]
    texts = [c.text[:MAX_CLAIM_CHARS] for c in claims]
    labels = [c.label for c in claims]
    only_reported = REPORTED in labels and OBSERVED not in labels
    # G3: relative dates resolved against the session date, which is appended (not for L1 preferences)
    if day is None:
        day = claim_day(episode, previous)
    texts = [annotate(t, day, stated=d.mtype is not MemoryType.USER_PREFERENCE) for t in texts]
    # the label is part of the claim text: every reader sees that the assistant said it (T3)
    shown = [REPORTED_PREFIX + t if lab == REPORTED else t for t, lab in zip(texts, labels)]
    raw = d.raw
    lifetime = str(raw.get("lifetime") or "long_term")
    volatile = bool(raw.get("volatile"))
    # A dated record read from a tool (a log line, a CSV row: "… at 04:29:57 on 2026-09-21") is a past
    # event, which stays true (prompt: "long … also past events"); the gate often chose session (T3, h3
    # tool_fact). Not for the user's own words: "at 15:00 today" may be a plan of the moment.
    dated = (lifetime != "long_term" or volatile) and d.mtype is not MemoryType.USER_PREFERENCE and \
        any(c.evidence == "tool_observed" and _dated_record(c.text) for c in claims)
    if dated:
        lifetime, volatile = "long_term", False
    lifetime_enum = {"transient": Lifetime.TRANSIENT, "session": Lifetime.SESSION}.get(lifetime, Lifetime.LONG_TERM)
    if volatile or lifetime_enum is not Lifetime.LONG_TERM:
        dest = Destination.NONE
    elif d.mtype is MemoryType.USER_PREFERENCE:
        dest = Destination.USER
    else:
        dest = Destination.WIKI
    cls = MemoryClassification(should_remember=dest is not Destination.NONE, relevance=round(record_conf, 3),
                               importance=round(_num(raw.get("importance"), 0.6), 3),
                               confidence=round(min(_num(raw.get("confidence"), 0.7), 0.4) if only_reported
                                                else _num(raw.get("confidence"), 0.7), 3), lifetime=lifetime_enum,
                               type=d.mtype, destination=dest)
    subject = d.subject
    title = shorten(re.sub(r"\s+", " ", str(raw.get("title") or "")).strip(), 70)
    if d.mtype is MemoryType.USER_PREFERENCE:
        title, subject = f"User preference: {texts[0]}", ""
    elif not title or not grounded(title, "\n".join(texts) + "\n" + subject, min_overlap=0.3):
        title = subject or _title_from_claim(bare(texts[0]))   # not the "(stated …)" suffix
    confirmed = previous is not None and any(c.evidence == "user_confirmed" for c in claims)
    event_ids = (previous.event_ids + episode.event_ids) if confirmed else list(episode.event_ids)
    if confirmed:
        context = (f'Proposed by the assistant in reply to "{shorten(previous.user_text, 160, title=False)}"; '
                   f'confirmed by the user: "{shorten(episode.user_text, 160, title=False)}".')
    elif any(c.evidence == "user_stated" for c in claims):
        context = ""   # the user's own sentence is the claim (see RuleClassifier._environment)
    else:
        context = _context(episode)
    supersedes = [c.supersedes for c in claims if c.supersedes] if raw.get("corrects") and \
        all(lab == OBSERVED for lab in labels) else []
    tags = _tags(texts + [subject])
    for t in subject_tags(subject):
        if t not in tags:
            tags.append(t)
    cid = f"{episode.id}:{d.mtype.value}"
    n = 2
    while cid in used_ids:
        cid, n = f"{episode.id}:{d.mtype.value}-{n}", n + 1
    used_ids.add(cid)
    evidence_classes = ", ".join(c.evidence for c in claims)
    return MemoryCandidate(
        event_ids=event_ids, classification=cls, normalized_claims=shown,
        retrieval_query=" ".join([title, *texts]), id=cid, session_id=episode.session_id, title=shorten(title),
        tags=tags[:10], claim_evidence=labels, context=context,
        rationale=f"{GATE_VERSION}: {why} [{d.domain}; {evidence_classes}]"
                  + ("; volatile/short-lived → none" if dest is Destination.NONE else "")
                  + ("; dated tool record → long_term" if dated else ""),
        subject=subject, supersedes=supersedes)


_NOTES_PATH = re.compile(r"(?:^|/)(?:notes?|chats?|transcripts?|conversations?|notizen)(?:/|[-_.])|"
                         r"(?:chat|transcript|conversation|notes?|notiz|verlauf)[^/]*\.(?:md|txt)$", re.I)
_ASSISTANT_HEADING = re.compile(r"^#+\s*(?:you|assistant|du|ki|bot)\b.*$|^#+.*\(assistant\).*$", re.I | re.M)
READ_TOOLS_ALL = frozenset({"read_file", "view_file", "cat", "open_file"})
MAX_NOTES_CANDIDATES = 3


def _keep_read_notes(cands: List[MemoryCandidate], episode: Episode, notes: List[str],
                     day: Optional[Any]) -> List[MemoryCandidate]:
    """A notes / chat-excerpt file the user had the agent read is the user's document: when the gate
    kept nothing from it, its statements are stored as tool-observed claims (seen on LongMemEval-style assistant-said items: the
    model called saved advice "world knowledge"; later "what did you tell me earlier?" had no answer).
    From the assistant part when the excerpt marks one; list items and sentences of 4+ words."""
    if any(OBSERVED in c.claim_evidence and "tool_observed" in c.rationale for c in cands):
        return cands
    reads = [c for c in episode.tool_calls if c.content.get("tool") in READ_TOOLS_ALL and _ok(c)]
    out = [c for c in cands if c.classification.type is not MemoryType.NOISE]
    added = 0
    for call in reads:
        args = call.content.get("args") if isinstance(call.content.get("args"), dict) else {}
        path = str(args.get("path") or args.get("file_path") or "")
        if not _NOTES_PATH.search(path):
            continue
        text = _NUMBERED.sub("", _tool_output(str(call.content.get("result_excerpt") or "")))
        heads = list(_ASSISTANT_HEADING.finditer(text))
        body = "\n".join(text[h.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))]
                         for i, h in enumerate(heads)) if heads else text
        items = []
        for sent in sentences(body):
            sent = redact(re.sub(r"\s+", " ", sent).strip())[0]
            if len(sent.split()) >= 4 and not sent.startswith("#"):
                items.append(shorten(f"Earlier chat ({path}): {sent}", MAX_CLAIM_CHARS, title=False))
        if not items:
            continue
        for n in range(0, min(len(items), gp.MAX_CLAIMS * MAX_NOTES_CANDIDATES), gp.MAX_CLAIMS):
            chunk = [annotate(x, day) for x in items[n:n + gp.MAX_CLAIMS]]
            cls = MemoryClassification(should_remember=True, relevance=0.7, importance=0.5, confidence=0.7,
                                       lifetime=Lifetime.LONG_TERM, type=MemoryType.PROJECT_FACT,
                                       destination=Destination.WIKI)
            out.append(MemoryCandidate(
                event_ids=list(episode.event_ids), classification=cls, normalized_claims=chunk,
                retrieval_query=" ".join([path, *chunk]), id=f"{episode.id}:notes-{added + 1}",
                session_id=episode.session_id, title=shorten(f"Earlier chat: {path}"), tags=_tags(chunk)[:10],
                claim_evidence=[OBSERVED] * len(chunk), context=_context(episode),
                rationale=f"{GATE_VERSION}: statements of a notes file the user had read [tool_observed]",
                subject=f"earlier chat {path}"))
            added += 1
    if added:
        notes.append(f"notes file statements kept ({added} candidate(s))")
        return out
    return cands


def _keep_explicit_preference(cands: List[MemoryCandidate], episode: Episode, notes: List[str]) -> List[MemoryCandidate]:
    """An explicit standing preference ("for all future chats …", "from now on …", "ab jetzt …") is
    never lost: when the gate kept none, rules-v2's preference rule (validated; the user's own words)
    adds it. validation failure shape: the model skipped a standing rule because the assistant declined it."""
    if any(c.classification.type is MemoryType.USER_PREFERENCE for c in cands):
        return cands
    # explicit standing wording only — not rules-v2's imperative branch ("Don't run anything to find out.")
    if not _PREF_ANY.search(strip_today_line(episode.user_text)):
        return cands
    pref = RuleClassifier()._preference(episode)
    if pref is None:
        return cands
    pref.normalized_claims = [redact(c)[0] for c in pref.normalized_claims]
    pref.rationale = f"{GATE_VERSION}: explicit standing preference (rules-v2 preference rule)"
    notes.append("explicit standing preference added by the rules-v2 preference rule")
    real = [c for c in cands if c.classification.type is not MemoryType.NOISE]
    return real + [pref]


def redact_obj(value: Any) -> Any:
    """``value`` (parsed JSON) with every string redacted — the raw model output goes to the log."""
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    return value


def redact_candidates(cands: List[MemoryCandidate]) -> List[str]:
    """Secret filter on finished candidates (used on the rules fallback); returns kinds found."""
    kinds: List[str] = []
    for c in cands:
        fixed = []
        for claim in c.normalized_claims:
            text, found = redact(claim)
            kinds.extend(found)
            fixed.append(text)
        c.normalized_claims = fixed
        for attr in ("title", "retrieval_query", "context", "subject"):
            text, found = redact(getattr(c, attr))
            kinds.extend(found)
            setattr(c, attr, text)
    return kinds


class RedactingRuleClassifier(RuleClassifier):
    """rules-v2 with the secret filter the gate and its fallback apply (``classifier.kind: rules``), so
    rules-only mode never writes a secret into the wiki or L1."""

    def classify(self, episode: Episode, previous: Optional[Episode] = None, *,
                 known: str = "") -> List[MemoryCandidate]:
        cands = super().classify(episode, previous, known=known)
        redact_candidates(cands)
        return cands


# -- the classifier ---------------------------------------------------------------------------------------

class LocalLLMClassifier:
    """The M9 gate (see module docstring). ``client`` and ``sleep`` are injectable for tests."""

    name = GATE_VERSION
    exclude_own_l1_writes = True   # the daemon leaves this turn's agent L1 writes out of ``known``

    def __init__(self, model: Optional[GateModelConfig] = None, config: Optional[ClassifierConfig] = None, *,
                 fallback: Optional[Classifier] = None, client: Optional[ChatClient] = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model or GateModelConfig()
        self.config = config or ClassifierConfig()
        self.fallback = fallback or RuleClassifier()
        self.client = client or ChatClient(self.model, self.config.timeout_s)
        self.clock = clock
        self.sleep = sleep
        self.last_info: Optional[Dict[str, Any]] = None
        self._failures = 0
        self._open_until = 0.0
        self._cache: "OrderedDict[str, Tuple[List[MemoryCandidate], Dict[str, Any]]]" = OrderedDict()
        self._session_offsets: Dict[str, int] = {}   # session → days between a user-stated date and the clock

    @staticmethod
    def _key(episode: Episode, previous: Optional[Episode], known: str) -> str:
        h = hashlib.sha1("\x1f".join([*episode.event_ids, *(previous.event_ids if previous else []), known]).encode())
        return h.hexdigest()

    def classify(self, episode: Episode, previous: Optional[Episode] = None, *,
                 known: str = "") -> List[MemoryCandidate]:
        """Candidates for ``episode`` (a successful result is cached per episode/previous/known)."""
        key = self._key(episode, previous, known)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.last_info = dict(hit[1], cached=True)
            return copy.deepcopy(hit[0])
        cands = self._classify(episode, previous, known)
        if self.last_info is not None and self.last_info.get("status") == "ok":
            self._cache[key] = (copy.deepcopy(cands), dict(self.last_info))
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return cands

    def _classify(self, episode: Episode, previous: Optional[Episode], known: str) -> List[MemoryCandidate]:
        info: Dict[str, Any] = {"version": GATE_VERSION, "model": self.model.model, "classifier": self.name}
        started = self.clock()
        try:
            skip = pre_skip(episode)
            if skip:
                info.update(status="skipped", skip_reason=skip)
                cands = [RuleClassifier()._noise(episode)]
                cands[0].rationale = f"{GATE_VERSION}: pre-skip ({skip})"
                return cands
            if self.clock() < self._open_until:
                raise GateError("circuit_open", f"{self._failures} consecutive failures")
            gate_input = build_input(episode, previous, known, max_chars=self.config.max_input_chars)
            info.update(prompt_chars=len(gp.SYSTEM_PROMPT) + len(gate_input.prompt), truncated=gate_input.truncated)
            if gate_input.secrets:
                info["secrets_redacted"] = sorted(set(gate_input.secrets))
            decision = self._call(gate_input, info)
            notes: List[str] = []
            turn = episode.turn
            offset = stated_offset(episode.user_text, turn.ts) if turn is not None else None
            if offset is not None:
                self._session_offsets[episode.session_id] = offset
            day = claim_day(episode, previous, self._session_offsets.get(episode.session_id))
            cands = post_check(decision, episode, previous, gate_input, notes, day=day)
            cands = _keep_explicit_preference(cands, episode, notes)
            cands = _keep_read_notes(cands, episode, notes, day)
            info.update(status="ok", output=redact_obj(decision), post_checks=notes)
            self._failures = 0
            return cands
        except GateError as exc:
            return self._fallback(episode, previous, known, exc, info)
        except Exception as exc:   # a bug in the gate must never lose an episode
            logger.exception("llm gate failed on episode %s", episode.id)
            return self._fallback(episode, previous, known, GateError("internal_error", repr(exc)), info)
        finally:
            info["latency_s"] = round(self.clock() - started, 3)
            self.last_info = info

    def _call(self, gate_input: GateInput, info: Dict[str, Any]) -> Dict[str, Any]:
        """One model call within the episode budget ``classifier.timeout_s``. A timeout is not
        retried (under load a second try mostly times out too and doubles the cost); a connection
        error is retried (``attempts``) with whatever budget is left."""
        calls: List[Dict[str, Any]] = []
        info["calls"] = calls
        messages = messages_for(gate_input)
        attempts = max(1, int(self.config.attempts))
        deadline = self.clock() + float(self.config.timeout_s)
        for attempt in range(1, attempts + 1):
            remaining = deadline - self.clock()
            if remaining <= 0.05:
                raise GateError("timeout", "episode budget spent")
            self.client.timeout_s = remaining
            t0 = self.clock()
            try:
                content, usage = self.client.complete(messages)
            except GateError as exc:
                calls.append({"attempt": attempt, "latency_s": round(self.clock() - t0, 3), "error": exc.reason})
                if exc.reason == "connection" and attempt < attempts:
                    self.sleep(min(2.0 * attempt, max(0.0, (deadline - self.clock()) / 2)))
                    continue
                raise
            if getattr(self.client, "dropped", None):
                info["dropped_fields"] = list(self.client.dropped)
            calls.append({"attempt": attempt, "latency_s": round(self.clock() - t0, 3),
                          "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"),
                          "finish_reason": usage.get("finish_reason")})
            return parse_decision(content)   # invalid output: no retry (temperature 0 repeats it)
        raise GateError("connection", "no attempt left")  # pragma: no cover

    def _fallback(self, episode: Episode, previous: Optional[Episode], known: str, exc: GateError,
                  info: Dict[str, Any]) -> List[MemoryCandidate]:
        if exc.reason != "circuit_open":
            self._failures += 1
            if self._failures >= max(1, int(self.config.breaker_failures)):
                self._open_until = self.clock() + float(self.config.breaker_open_s)
                logger.warning("llm gate: circuit breaker open for %.0fs after %d failures",
                               self.config.breaker_open_s, self._failures)
        logger.warning("llm gate fallback to %s on episode %s: %s", getattr(self.fallback, "name", "rules"),
                       episode.id, exc)
        cands = self.fallback.classify(episode, previous, known=known)
        kinds = redact_candidates(cands)
        for c in cands:
            c.rationale = f"fallback:rules ({exc.reason}); {c.rationale}"
        info.update(status="fallback", fallback_reason=exc.reason, error=str(exc)[:300],
                    classifier=getattr(self.fallback, "name", "rules"))
        if kinds:
            info["secrets_redacted"] = sorted(set(info.get("secrets_redacted", []) + kinds))
        return cands


class ShadowClassifier:
    """Shadow rollout (§4.1.5): the rules decide; the LLM gate runs too and its output is logged."""

    def __init__(self, gate: LocalLLMClassifier, rules: Optional[Classifier] = None) -> None:
        self.gate = gate
        self.rules = rules or RuleClassifier()
        self.name = getattr(self.rules, "name", "rules")
        self.last_info: Optional[Dict[str, Any]] = None

    def classify(self, episode: Episode, previous: Optional[Episode] = None, *,
                 known: str = "") -> List[MemoryCandidate]:
        shadow = self.gate.classify(episode, previous, known=known)
        info = dict(self.gate.last_info or {})
        info["mode"] = "shadow"
        info["classifier"] = self.name
        info["shadow"] = [{"id": c.id, "type": c.classification.type.value,
                           "destination": c.classification.destination.value, "claims": c.normalized_claims,
                           "claim_evidence": c.claim_evidence, "subject": c.subject} for c in shadow]
        cands = self.rules.classify(episode, previous, known=known)
        kinds = redact_candidates(cands)
        if kinds:
            info["secrets_redacted"] = sorted(set(kinds))
        self.last_info = info
        return cands


def build_classifier(config: PanConfig) -> Classifier:
    """``classifier.kind`` → the classifier the daemon uses (unknown kinds → rules, with a warning)."""
    kind = str(config.classifier.kind or "rules").strip().lower()
    if kind in ("rules", "rules-v2", ""):
        return RedactingRuleClassifier()
    gate = LocalLLMClassifier(config.models.gate, config.classifier)
    if kind in ("llm", GATE_VERSION):
        return gate
    if kind == "shadow":
        return ShadowClassifier(gate)
    logger.warning("unknown classifier.kind %r; using rules", kind)
    return RedactingRuleClassifier()
