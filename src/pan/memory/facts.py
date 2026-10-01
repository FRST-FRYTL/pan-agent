"""Fact structure for claims (M5): subjects, typed values and the contradiction rule.

Used by the classifier (page subjects / titles), the curator (superseding wiki bullets) and the L1
writer (replacing stale ``MEMORY.md`` entries). Everything here is deterministic and conservative.

**Values.** :func:`values` extracts typed, *slotted* values from a claim: ``port`` (``port 8000``,
``host:8000``, ``:8000``), ``host`` (``host db01``, ``db01:3300``), ``ip``, ``url``, ``path``
(``/etc/…``, ``~/…``), ``version`` (``v1.2``, ``0.21.4``), ``size`` (``128 GB``) and named ids
(tokens with letters *and* digits such as ``GB10``/``H100``/``Qwen3.8-27B``) — the latter only
when a slot noun (``gpu``, ``model``, ``cpu``, ``driver``, …) stands at most four words before
them (slot ``id:<noun>``). Bare numbers are never values (too ambiguous).

**Contradiction** (:func:`contradiction`): a newer claim contradicts an older text when

1. both have at least one slot in common and, for some common slot, the value sets are
   **disjoint** (values the new claim introduces with ``from`` — "moved from 8000 to 8010" — are
   the old values and are ignored);
2. they talk about the **same subject**: the *distinctive* tokens (content tokens minus values,
   slot words and generic words like server/instance/runs/moved/port) have a Jaccard similarity
   ≥ :data:`SUBJECT_MIN_JACCARD` and are not empty — or (M6) their subject phrases share the head
   noun and one's modifiers contain the other's ("vLLM server" ≈ "PAN vLLM server");
3. same polarity (both negated or neither).

``"vLLM server runs on port 8000."`` vs ``"We moved the vLLM server to port 8010."`` → contradiction
(slot ``port``, subject {vllm}); vs ``"Langfuse runs on port 3300."`` → no (different subject);
``"vLLM metrics exporter on port 9100"`` → no (Jaccard {vllm} / {vllm, metric, exporter} < 0.6).

**Rewrite.** When every conflicting slot has exactly one old and one new value, the old text is
rewritten in place (its own phrasing kept, e.g. an L1 entry the agent wrote); otherwise the caller
uses the new claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pan.memory.claims import normalize, stem, tokens

SUBJECT_MIN_JACCARD = 0.6

# Words that say nothing about *which* thing a fact is about (stemmed like claims.tokens).
_GENERIC_RAW = """
server servers service services instance instances host hosts hostname machine machines box node nodes system
project projects our user user's run runs running ran use uses using used move moved moving now set point points
configured located live lives switch switched change changed update updated new default currently port ports
also still available reachable hosted deployed serve serves served bound locally local address endpoint note later
fyi remember saved memory want wants one listen listens listening sit sits resides located stay stays go goes went
went got moving migrated migrate setup instead today already actually reports report called named value is has
installed install version versions path paths model models id name uses value values there here current
today yesterday tomorrow morning evening night week weekend recently earlier again finally just meanwhile
""".split()
GENERIC_WORDS = frozenset(stem(w) for w in _GENERIC_RAW) | frozenset(_GENERIC_RAW)

_NEGATION = re.compile(r"\b(?:not|no longer|never|isn't|aren't|wasn't|doesn't|don't|without)\b", re.I)

# Slot nouns that make a letters+digits token a value ("GPU is NVIDIA GB10", "model Qwen3.8-27B").
SLOT_NOUNS = ("gpu", "gpus", "cpu", "cpus", "model", "driver", "kernel", "image", "branch", "container",
              "chip", "card", "accelerator", "tag", "release", "distro", "os")

_URL = re.compile(r"\bhttps?://[^\s)\]>\"'`]+", re.I)
_IP = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{2,5}))?(?![\w.])")
_HOST_PORT = re.compile(r"(?<![\w./-])([a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*):(\d{2,5})\b", re.I)
_PORT = re.compile(r"\bports?\s*(?:number\s*)?(?:[:=#]|is|of|to|at|on)?\s*(\d{2,5})\b|(?<![\w.:/]):(\d{2,5})\b|"
                   r"\b(?:listens?|listening|runs?|running|serves?|served|reachable|available)\s+(?:now\s+)?on\s+"
                   r"(\d{4,5})\b", re.I)
_HOST = re.compile(r"\bhost(?:name)?\s*(?:[:=,]|is|called|named|to|at|on)?\s*`?([a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*)",
                   re.I)
_PATH = re.compile(r"(?<![\w/.:~-])((?:~|\.{1,2})?/(?:[\w.@+-]+/)*[\w.@+-]*[A-Za-z][\w.@+-]*/?)")
# Names: an identifier-shaped token (hyphen/underscore/dot inside, or letters+digits) right after a
# naming word — "the box is kestrel-dev", "I sign in as build-bot", "heißt nas-2", "called x_y" —
# or a capitalized proper name after called/named/heißt ("the project is called Larkspur"). Plain
# words ("as primary") and abbreviations ("called Mr. …") are not names.
_NAME = re.compile(r"\b(is|are|was|called|named|as|heißt|heisst|hei(?:ss|ß)en|lautet|ist)\s+[`'\"]?"
                   r"([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]+\d[\w-]*)(?![\w/-]|\.\w)|"
                   r"\b(called|named|heißt|heisst)\s+([A-Z][a-z][\w-]+)(?![\w-]|\.\w)")
# ``key = Value`` settings with an identifier-shaped value ("default_media = GlossPro-7", "theme = solar-2"):
# the same ``name`` slot as "the default media is GlossPro-7", so a config reading can update a stated value.
_ASSIGN = re.compile(r"\b[A-Za-z][\w.-]*\s*=\s*[`'\"]?([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]+\d[\w-]*)"
                     r"(?![\w/-]|\.\w)")
# Weekdays as schedule values ("deploys happen on Thursdays", "every Monday", "jeden Freitag").
_WEEKDAY = re.compile(r"\b(?:on|every|each|am|jeden|jede[nm]?|montags|dienstags)?\s*\b((?:mon|tues|wednes|thurs|fri|satur|sun)days?|"
                      r"(?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)s?)\b", re.I)
# Dates (M6): "October 12", "12 Oct 2026", "2026-10-12", "12. Oktober".
_MONTHS = (r"(?:jan(?:uary|uar)?|feb(?:ruary|ruar)?|m(?:ar(?:ch)?|ärz|aerz)|apr(?:il)?|ma[iy]|june?|juni|july?|juli|"
           r"aug(?:ust)?|sep(?:t(?:ember)?)?|o[ck]t(?:ober)?|nov(?:ember)?|de[cz](?:ember)?)")
_DATE = re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b|"
                   rf"\b\d{{1,2}}(?:st|nd|rd|th)?\.?\s+{_MONTHS}\.?(?:\s+\d{{4}})?\b|"
                   r"\b\d{4}-\d{2}-\d{2}\b", re.I)
# Clock times (M6): "standup is at 10:15", "backup runs at 3:00 am" — not timestamps (03:01:40).
_TIME = re.compile(r"\bat\s+(\d{1,2}[:.]\d{2})(?![:.\d])|(?<![\d:.T-])(\d{1,2}:\d{2})(?![:\d])\s*(?:am|pm|uhr|h)?\b", re.I)
_VERSION = re.compile(r"(?<![\w.])v?(\d+\.\d+(?:\.\d+)*(?:[-+][\w.]+)?)(?![\w.])", re.I)
# Numbers with thousands separators are one number: "7,500" / "1,234.5" (EN), "7.500" / "1.234,5"
# (DE) — never "7" and "500" (M6 generic-rules: a rewrite mixed "5,000" and "7,500" into "7,000").
NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
_NUMBER_TOKEN = re.compile(rf"(?<![\w.,]){NUMBER}(?![\w]|[.,]\d)")
_SIZE = re.compile(rf"(?<![\w.,])({NUMBER})\s?([KMGTP]i?B|[KMGT]B?)\b")
_ID = re.compile(r"(?<![\w.-])([A-Za-z][\w.-]*\d[\w.-]*|\d+[A-Za-z][\w.-]*)(?![\w-])")
_FROM_VALUE = re.compile(r"\bfrom\s+(?:port\s+|host\s+|version\s+)?`?([\w.:/~@+-]+)", re.I)
_WORD = re.compile(r"[A-Za-z][\w'-]*")
_HOST_STOP = frozenset({"is", "the", "a", "an", "at", "on", "and", "port", "machine", "this", "that", "our",
                        "localhost", "which", "with", "for", "in"})


# Measures (M6): a bare number is a value when an attribute phrase stands right before it
# ("API rate limit is 100 requests per minute", "rate_limit_per_min = 250", "burst = 40"). Slot
# ``measure:<attribute stems>``; two attributes match when one's stems contain the other's or their
# Jaccard ≥ SUBJECT_MIN_JACCARD ("rate limit" ≈ "API rate limit", but "billing rate limit" ≠ "auth
# rate limit"). Numbers glued to letters, dots, slashes, colons or hyphens are not measures.
_MEASURE_NUM = re.compile(rf"(?<![\w.,:/-])({NUMBER})(?![\w:/-]|[.,]\w|\s*[:/]\d)")
_MEASURE_SKIP = frozenset("""
is are was were be been set to now of at currently about around roughly approx approximately our the a an
per its their this that my your equals equal default by under over below above within than least most
der die das den dem des ein eine einen einem einer ist sind war mit bei von vom zu zum zur im am auf pro je
für fuer liegt liegen betraegt beträgt betragen
""".split())
_UNIT_WORDS = frozenset("""
ms s sec secs second seconds min mins minute minutes h hr hrs hour hours day days kb mb gb tb kib mib gib
req reqs request requests rps qps percent pct x times
""".split())
MEASURE_ATTR_WORDS = 3


def _measure_attr(before: str) -> Set[str]:
    """Stems of the (≤ MEASURE_ATTR_WORDS) content words right before a number, within its clause."""
    before = re.sub(r"[\s:=`*]+$", "", before)  # "limit: 100", "burst = 40", "`x` = 1"
    clause = re.split(r"[,;:.!?()\[\]`]\s|[,;:!?()\[\]`]$|\n", before)[-1]
    words = re.findall(r"[^\W\d_]+", clause)
    attr: List[str] = []
    for w in reversed(words):
        lw = w.lower()
        if lw in _MEASURE_SKIP or lw in _UNIT_WORDS:
            if attr:
                if lw in _MEASURE_SKIP and lw not in ("per",):
                    continue
                continue
            continue
        if lw in _VERBISH:
            break
        attr.append(stem(lw))
        if len(attr) >= MEASURE_ATTR_WORDS:
            break
    return set(attr)


_VERBISH = frozenset("""
has have had runs run ran uses use used moved changed raised lowered increased decreased reduced got takes took
says said shows showed reports reported contains contain and or but with without than then
""".split())


# Words that name the kind of quantity, not the thing it belongs to ("max upload" ≈ "upload size cap").
ATTRIBUTE_WORDS = frozenset(stem(w) for w in """
max maximum min minimum cap limit size count total default number num amount value threshold
""".split())


# Verbs that end up in a measure attribute ("Block C gets 30 minutes", "advised budgeting $25") but
# name no quantity: sharing only these is not the same attribute (T3: sibling items were struck).
WEAK_ATTRIBUTE_WORDS = frozenset(stem(w) for w in """
get gets give gives given grow grows take takes need needs advise advised advises recommend recommended
suggest suggested budget budgeting spend spends spent allot allotted plan planned
""".split())


def _attrs_match(a: str, b: str) -> bool:
    """Same measured attribute: shared stems, and not both sides with their own non-attribute
    qualifier ("rate limit" ≈ "API rate limit", "max upload" ≈ "upload size cap", but "billing retry
    limit" ≠ "auth retry limit"). Shared stems must name the thing, not only the kind of quantity
    ("limit") or a verb ("gets", "budgeting")."""
    ta, tb = set(a.split(":", 1)[1].split("-")), set(b.split(":", 1)[1].split("-"))
    if not ta or not tb or not ta & tb:
        return False
    if not (ta & tb) - ATTRIBUTE_WORDS - WEAK_ATTRIBUTE_WORDS:
        return False  # only "limit" / "gets" in common says nothing about which thing
    return not ((ta - tb - ATTRIBUTE_WORDS) and (tb - ta - ATTRIBUTE_WORDS))


@dataclass
class Values:
    """Slot → set of normalized values, plus the literal spans (for rewriting)."""
    slots: Dict[str, Set[str]] = field(default_factory=dict)
    spans: Dict[str, Dict[str, str]] = field(default_factory=dict)  # slot → value → literal text
    tokens: Set[str] = field(default_factory=set)                    # tokens of all value spans

    def add(self, slot: str, literal: str, value: Optional[str] = None) -> None:
        value = (value if value is not None else literal).strip().strip("`'\".,;").lower()
        if not value:
            return
        self.slots.setdefault(slot, set()).add(value)
        self.spans.setdefault(slot, {}).setdefault(value, literal.strip().strip("`'\".,;"))
        self.tokens.update(t for t in re.findall(r"[a-z0-9]+", value))


def _clean_text(text: str) -> str:
    # Markdown emphasis around values; underscores inside identifiers (rate_limit_per_min) stay
    return re.sub(r"(?<!\w)[*_]{1,3}|[*_]{1,3}(?!\w)|\*{1,3}", "", str(text or ""))


def values(text: str) -> Values:
    """Typed values in ``text`` (see module docstring)."""
    s = _clean_text(text)
    out = Values()
    taken: List[Tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for m in _URL.finditer(s):
        out.add("url", m.group(0))
        taken.append(m.span())
    for m in _IP.finditer(s):
        if free(*m.span()):
            out.add("ip", m.group(1))
            if m.group(2):
                out.add("port", m.group(2))
            taken.append(m.span())
    for m in _HOST_PORT.finditer(s):
        if free(*m.span()) and not m.group(1).lower() in ("port", "http", "https"):
            if m.group(1).lower() != "localhost":
                out.add("host", m.group(1))
            out.add("port", m.group(2))
            taken.append(m.span())
    for m in _PORT.finditer(s):
        if free(*m.span()):
            out.add("port", m.group(1) or m.group(2) or m.group(3))
            taken.append(m.span())
    for m in _HOST.finditer(s):
        if free(*m.span(1)) and m.group(1).lower() not in _HOST_STOP:
            out.add("host", m.group(1))
            taken.append(m.span(1))
    for m in _PATH.finditer(s):
        if free(*m.span(1)) and len(m.group(1)) > 2 and "/" in m.group(1).strip("/"):
            out.add("path", m.group(1).rstrip("/"))
            taken.append(m.span(1))
        elif free(*m.span(1)) and m.group(1).startswith(("/", "~/")) and len(m.group(1)) > 3:
            out.add("path", m.group(1).rstrip("/"))
            taken.append(m.span(1))
    for m in _DATE.finditer(s):
        if free(*m.span()):
            out.add("date", m.group(0))
            taken.append(m.span())
    for m in _WEEKDAY.finditer(s):
        if free(*m.span(1)):
            day = m.group(1).lower()
            out.add("day", m.group(1), value=day[:-1] if day.endswith("s") and not day.endswith("ss") else day)
            taken.append(m.span(1))
    for m in _TIME.finditer(s):
        if free(*m.span()):
            out.add("time", m.group(1) or m.group(2))
            taken.append(m.span())
    for m in _SIZE.finditer(s):
        if free(*m.span()):
            out.add("size", m.group(0), value=m.group(1) + m.group(2).upper())
            attr = _measure_attr(s[:m.start()])
            if attr:  # "upload size cap: 10 MB" is also a measure ("max_upload_mb = 25")
                out.add("measure:" + "-".join(sorted(attr)), m.group(1))
            taken.append(m.span())
    for m in _VERSION.finditer(s):
        if free(*m.span()):
            out.add("version", m.group(1))
            taken.append(m.span())
    for m in _MEASURE_NUM.finditer(s):
        if not free(*m.span()):
            continue
        attr = _measure_attr(s[:m.start()])
        if attr:
            out.add("measure:" + "-".join(sorted(attr)), m.group(1))
            taken.append(m.span())
    words = [(w.group(0).lower(), w.start()) for w in _WORD.finditer(s)]
    for m in _ID.finditer(s):
        if not free(*m.span()):
            continue
        before = [w for w, pos in words if pos < m.start()][-4:]
        noun = next((w for w in reversed(before) if w in SLOT_NOUNS), None)
        after = [w for w, pos in words if pos > m.end()][:2]
        noun = noun or next((w for w in after if w in SLOT_NOUNS), None)
        if noun:
            out.add(f"id:{noun.rstrip('s') if noun.endswith('s') and noun != 'os' else noun}", m.group(1))
            taken.append(m.span())
    for m in _NAME.finditer(s):
        g = 2 if m.group(2) else 4
        if free(*m.span(g)):
            out.add("name:as" if (m.group(1) or "").lower() == "as" else "name", m.group(g))
            taken.append(m.span(g))
    for m in _ASSIGN.finditer(s):
        if free(*m.span(1)):
            out.add("name", m.group(1))
            taken.append(m.span(1))
    return out


def _from_values(text: str) -> Set[str]:
    return {m.group(1).strip("`'\".,;").lower() for m in _FROM_VALUE.finditer(_clean_text(text))}


def distinctive_tokens(text: str, vals: Optional[Values] = None) -> Set[str]:
    """Content tokens that identify what ``text`` is about (values and generic words removed)."""
    vals = vals if vals is not None else values(text)
    out = set()
    for t in tokens(text):
        if t in GENERIC_WORDS or t in vals.tokens or any(c.isdigit() for c in t) or len(t) < 2:
            continue
        if t in SLOT_NOUNS or t.rstrip("s") in SLOT_NOUNS:
            continue
        out.add(t)
    return out


def same_subject(a: str, b: str) -> bool:
    ta, tb = distinctive_tokens(a), distinctive_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= SUBJECT_MIN_JACCARD


# Retractions (M6): "we cancelled daily standups entirely" / "X was decommissioned" / "no longer …".
RETRACT_VERB = re.compile(r"\b(?:cancel(?:l)?ed|dropped|discontinued|decommissioned|retired|abolished|shut down|"
                          r"no longer|got rid of|scrap(?:ped)?|scratch(?: that)?|(?<!don't )(?<!do not )(?<!never )forget(?! to)|"
                          r"never mind|called off|"
                          r"abgeschafft|eingestellt|gestrichen|abgesagt|storniert|vergiss(?! nicht)|streich|fällt weg|faellt weg|"
                          r"entfällt|entfaellt|gibt es nicht mehr|nicht mehr)\b", re.I)
# A retracted statement without a typed value must be short (a note, not a knowledge page paragraph).
RETRACT_MAX_WORDS = 20


def _retraction(new: str, old: str) -> Optional["Contradiction"]:
    """``new`` retracts ``old``: a retraction verb (EN + DE: cancelled, scratch, forget, never mind,
    vergiss, gestrichen, entfällt …), ``old`` has a typed value or is a short note, and ``new`` shares at
    least half (and ≥ 2, or all if fewer) of ``old``'s distinctive tokens ("Our team's daily standup
    is at 10:15." ← "we cancelled daily standups entirely"). ``old`` must not itself be a retraction."""
    if not RETRACT_VERB.search(new) or RETRACT_VERB.search(old):
        return None
    ov = values(old)
    if not ov.slots and len(old.split()) > RETRACT_MAX_WORDS:
        return None
    nv = values(new)
    common = set(nv.slots) & set(ov.slots)
    if any(not (nv.slots[k] & ov.slots[k]) for k in common):
        return None  # retracts another value ("no longer on port 8000" vs "… port 9000")
    if common and _restated(new, {v for vs in ov.slots.values() for v in vs}):
        return None  # "the default is X because Y was discontinued" restates X; it retracts Y
    tb = distinctive_tokens(old, ov)
    new_tokens = set(tokens(new))
    shared = tb & new_tokens
    # the old statement's subject named again ("the vendor sync" ← "scratch the vendor sync") or
    # most of its distinctive words
    subj = {t for t in tokens(subject_of(old)) if t not in GENERIC_WORDS}
    if not (subj and subj <= new_tokens) and (not tb or len(shared) < min(2, len(tb)) or 2 * len(shared) < len(tb)):
        return None
    return Contradiction(["retracted"], dict(ov.slots), {})


_CLAUSE = re.compile(r"[.;!?]\s|,\s|\s[—–-]\s|\b(?:because|since|as|after|and|but|so|weil|da|nachdem|und|aber)\b", re.I)


def _restated(new: str, old_values: Set[str]) -> bool:
    """Every value of the old statement stands in a clause of ``new`` without a retraction verb: the
    old statement is asserted again, not withdrawn ("the default is now MattePro-X because Satin-Y was
    discontinued" restates "the default is MattePro-X"). Restating only a name next to a retraction
    ("Rex's Friday visit is cancelled, Rex is fine") still retracts."""
    asserted: Set[str] = set()
    for clause in _CLAUSE.split(new):
        if clause and not RETRACT_VERB.search(clause):
            asserted |= {v for vs in values(clause).slots.values() for v in vs}
    return bool(old_values) and old_values <= asserted


# Words that mark a statement as an update without changing what it says ("now", "ab jetzt").
_UPDATE_WORDS = re.compile(r"\b(?:from now on|going forward|as of today|now|currently|these days|nowadays|anymore|"
                           r"ab jetzt|ab sofort|inzwischen|mittlerweile|jetzt|nun|neuerdings|aktuell)\b", re.I)
_SHAPE_TOKEN = re.compile(r"[\w][\w'.:/-]*")


def shape_update(new: str, old: str) -> Optional["Contradiction"]:
    """``new`` restates ``old`` with one short span replaced ("… in room B on Tuesdays" → "… now in
    room C on Tuesdays", "betreut Priya Nair" → "betreut jetzt Omar Haddad"): the same statement with a
    new value, even when the value has no typed shape. Update words are ignored; exactly one
    replaced span of ≤ 3 tokens on each side, ≥ 3 equal tokens around it, same polarity."""
    import difflib

    a = [t.lower().strip(".") for t in _SHAPE_TOKEN.findall(_UPDATE_WORDS.sub(" ", strip_lead(old)))]
    b = [t.lower().strip(".") for t in _SHAPE_TOKEN.findall(_UPDATE_WORDS.sub(" ", strip_lead(new)))]
    a, b = [t for t in a if t], [t for t in b if t]
    if not a or not b or bool(_NEGATION.search(new)) != bool(_NEGATION.search(old)):
        return None
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1 or ops[0][0] != "replace":
        return None
    _, i1, i2, j1, j2 = ops[0]
    if i2 - i1 > 3 or j2 - j1 > 3 or len(a) - (i2 - i1) < 3:
        return None
    old_span, new_span = " ".join(a[i1:i2]), " ".join(b[j1:j2])
    subj = set(normalize(subject_of(old)).split())
    if subj & set(a[i1:i2]):
        return None  # the subject itself changed: another thing, not a new value
    return Contradiction(["shape"], {"shape": {old_span}}, {"shape": {new_span}})


def _subject_in(subject: str, old_tokens: Set[str]) -> bool:
    ts = distinctive_tokens(subject)
    return bool(ts) and ts <= old_tokens


def _align_measures(nv: Values, ov: Values) -> None:
    """Rename ``ov``'s measure slots to the matching slot names of ``nv`` (see _attrs_match)."""
    for okey in [k for k in ov.slots if k.startswith("measure:") and k not in nv.slots]:
        nkey = next((k for k in nv.slots if k.startswith("measure:") and _attrs_match(k, okey)), None)
        if nkey is not None and nkey not in ov.slots:
            ov.slots[nkey] = ov.slots.pop(okey)
            ov.spans[nkey] = ov.spans.pop(okey)


def _qualified_subject(a: str, b: str, *, allow_bare: bool = False) -> bool:
    """Same subject up to extra qualifiers (M6): both subject phrases end in the same head noun and
    one's modifiers contain the other's non-empty ones — "vLLM server" ≈ "PAN vLLM server" ≈ "vLLM
    inference server" (a benchmark run: the agent's L1 entry said "PAN vLLM server"), but "vLLM server" ≠
    "vLLM metrics exporter" (other head) ≠ "Langfuse server" (disjoint modifiers)."""
    sa, sb = subject_of(a), subject_of(b)
    if not sa or not sb:
        return False
    wa, wb = normalize(sa).split(), normalize(sb).split()
    if not wa or not wb or stem(wa[-1]) != stem(wb[-1]):
        return False
    ma, mb = {stem(w) for w in wa[:-1]}, {stem(w) for w in wb[:-1]}
    if not ma or not mb:
        return allow_bare
    return ma <= mb or mb <= ma


@dataclass
class Contradiction:
    slots: List[str]                         # conflicting slots
    old_values: Dict[str, Set[str]]
    new_values: Dict[str, Set[str]]
    rewritten: Optional[str] = None          # old text with the new values, when unambiguous


def contradiction(new: str, old: str, *, on_page: bool = False, subject: str = "",
                  strict_measures: bool = False, distinct_names: bool = False) -> Optional[Contradiction]:
    """How ``new`` contradicts ``old`` (None = it does not; see module docstring).

    ``on_page``: ``old`` is a bullet of the page the curator already chose for ``new``'s subject, so
    a subject that *contains* the other's distinctive tokens also counts ("Kibana" vs "Kibana
    views"); ``subject`` = ``new``'s subject — its distinctive tokens inside ``old`` count too.
    Disjoint subjects on the same page ("auth" vs "billing") still don't. ``strict_measures``: a
    conflict of measures only also needs the same subject (the curator for LLM-gate claims, T3: two
    budgets on one page, "$60 for soil" and "$25 for pest control", are both valid)."""
    if not new or not old:
        return None
    retracted = _retraction(new, old)
    if retracted is not None:
        return retracted
    nv, ov = values(new), values(old)
    _align_measures(nv, ov)
    stale = _from_values(new)
    new_slots = {k: {v for v in vs if v not in stale} for k, vs in nv.slots.items()}
    new_slots = {k: vs for k, vs in new_slots.items() if vs}
    common = sorted(set(new_slots) & set(ov.slots))
    conflicting = [k for k in common if not (new_slots[k] & ov.slots[k])]
    if not conflicting:
        return None
    if bool(_NEGATION.search(new)) != bool(_NEGATION.search(old)):
        return None
    measures_only = all(k.startswith("measure:") for k in conflicting)
    if measures_only and distinct_names:
        na, nb = proper_names(new), proper_names(old)
        if na and nb and not na & nb:
            return None   # the same measure of another person or thing ("Jonas … number 9" vs "Mika … 17")
    ta, tb = distinctive_tokens(new, nv), distinctive_tokens(old, ov)
    if not measures_only or strict_measures:  # a matching measure attribute names the subject (rules)
        if not ta or not tb:
            # generic subjects only ("the box", "the machine"): the same head noun decides
            if not _qualified_subject(new, old, allow_bare=True):
                return None
            ta = tb = {"_"}
        contained = ta <= tb or tb <= ta
        if measures_only:
            # strict (gate claims): the page subject is too broad to tell two budgets apart; only the
            # claims' own words, or a new claim that restates part of the old one, do
            contained, subject = ta <= tb, ""
        if (len(ta & tb) / len(ta | tb) < SUBJECT_MIN_JACCARD and not _qualified_subject(new, old)
                and not (on_page and contained)
                and not (on_page and subject and _subject_in(subject, tb))):
            return None
    result = Contradiction(conflicting, {k: ov.slots[k] for k in conflicting}, {k: new_slots[k] for k in conflicting})
    if all(len(ov.slots[k]) == 1 and len(new_slots[k]) == 1 for k in conflicting):
        text = old
        for k in conflicting:
            (old_v,), (new_v,) = ov.slots[k], new_slots[k]
            lit = ov.spans[k][old_v]
            # whole-token replacement: "5" must not hit the "5" of "5,000" or of "port 5432"
            text = re.sub(rf"(?<![\w.,]){re.escape(lit)}(?![\w]|[.,]\d)", lambda _m: nv.spans[k][new_v], text, count=1)
        result.rewritten = text if text != old and _only_known_numbers(text, old, new) else None
    return result


_PROPER = re.compile(r"(?<=[\w,;:)] )([A-ZÄÖÜ][a-zäöüß]+(?:[ -][A-ZÄÖÜ][a-zäöüß]+)*)")


def proper_names(text: str) -> Set[str]:
    """Capitalized words that are not sentence-initial ("…, Jonas Hartl wears …") — English only
    (German capitalizes every noun, so a German text yields its nouns, which compare as well)."""
    return {m.group(1).lower() for m in _PROPER.finditer(_clean_text(text))}


def _only_known_numbers(text: str, *sources: str) -> bool:
    """Every number in ``text`` occurs as a whole number in one of ``sources`` (a rewrite must never
    synthesize a value)."""
    known = {m.group(0) for src in sources for m in _NUMBER_TOKEN.finditer(src)}
    return all(m.group(0) in known for m in _NUMBER_TOKEN.finditer(text))


# -- subjects (page titles) --------------------------------------------------------------------------

_FACT_LEAD = re.compile(
    r"^(?:\W*(?:note(?:\s+for\s+later)?|fyi|for the record|update|heads[- ]up|remember(?:\s+that)?|btw|"
    r"by the way|note to self|important|info|context|reminder|just so you know|for later(?: chats| sessions)?|"
    r"for (?:all )?future (?:chats|sessions|conversations)|zur info|info|übrigens|ubrigens|notiz|hinweis|"
    r"für später|fur spater|merk dir(?: bitte)?(?:,? dass)?|nur zur info|kurze info|(?:please )?note(?: that)?|"
    r"(?:please )?keep in mind(?: that)?|bear in mind(?: that)?|good to know|for your information)\b\s*[:,\-—–]?\s*)+",
    re.I)
_MOVE = re.compile(
    r"^(?:we|i|they|someone)\s+(?:have\s+|just\s+|now\s+|finally\s+)*(?:moved|changed|switched|migrated|set|put|"
    r"pointed|upgraded|downgraded|renamed|reconfigured|configured|bound|relocated|rebound|replaced)\s+"
    r"(?P<subj>.+?)\s+(?:to|from|on|onto|at|into|over\s+to|with|as)\b", re.I)
_COPULA = re.compile(
    r"^(?P<subj>.+?)\s+(?:is|are|was|were|has\s+been|have\s+been|now\s+runs?|runs?|listens|lives|sits|uses|serves|"
    r"points|has|have|resides|moved|got\s+moved|can\s+be\s+found|will\s+be|stays|reports?|"
    r"heißt|heisst|liegt|läuft|lauft|ist|sind|lautet|befindet\s+sich|"
    r"starts|begins|ends|happens|takes\s+place|opens|closes|expires|beginnt|endet|startet)\b", re.I)
_DETERMINER = re.compile(r"^(?:(?:the|our|my|your|this|that|these|those|a|an|its|their|his|her|"
                         r"user's|users'|the\s+user's|this\s+project's|the\s+project's|our\s+project's|"
                         r"unser|unsere|unseren|unserem|unserer|mein|meine|der|die|das|den|dem|ein|eine)\s+)+", re.I)
_TRAILING_PP = re.compile(r"\s+(?:for|of|in|on|at|from)\s+(?:this|the|our)\s+(?:project|repo|repository|team|"
                          r"machine|host|box|setup|environment|env|deployment|system|cluster|network|"
                          r"workspace|lab)\b.*$|\s+(?:here|there|now|currently)$", re.I)
_PRONOUN = frozenset({"it", "this", "that", "there", "we", "i", "you", "they", "he", "she", "which", "what", "es", "das", "dies", "er", "sie", "wir",
                      "everything", "something", "nothing", "one"})
_HOST_WORDS = frozenset({"machine", "host", "server", "box", "system", "computer", "node", "workstation"})
HARDWARE = ("gpu", "cpu", "ram", "memory", "disk", "ssd", "nvme", "storage", "network card")
_MD = re.compile(r"[`*_]+")


def strip_lead(sentence: str) -> str:
    """Drop "Note for later:", "FYI:", "Update:" … lead-ins."""
    return _FACT_LEAD.sub("", sentence.strip()).strip()


def display_subject(text: str) -> str:
    """Capitalize the first letter unless the first word has its own casing ("vLLM", "iPhone")."""
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    first = text.split(" ", 1)[0] if text else ""
    if first[:1].islower() and first == first.lower():
        return text[:1].upper() + text[1:]
    return text


def subject_of(sentence: str) -> str:
    """The noun phrase a fact sentence is about ("" when unclear). Examples:
    "Our vLLM server runs on port 8000." → "vLLM server";
    "Update: we moved the vLLM server to port 8010." → "vLLM server";
    "the Langfuse instance for this project runs on host db01" → "Langfuse instance";
    "The machine has an NVIDIA GB10 GPU." → "GPU"."""
    s = _MD.sub("", strip_lead(sentence)).strip()
    if not s:
        return ""
    m = _MOVE.match(s) or _COPULA.match(s)
    if not m:
        return ""
    subj = _DETERMINER.sub("", m.group("subj").strip())
    subj = _TRAILING_PP.sub("", subj).strip(" ,.:;—–-")
    words = subj.split()
    if not words or len(words) > 6 or words[0].lower() in _PRONOUN or not re.search(r"[A-Za-z]", subj):
        return ""
    if subj.lower() in _HOST_WORDS or (words[-1].lower() in _HOST_WORDS and len(words) == 1):
        low = s.lower()
        hw = next((h for h in HARDWARE if re.search(rf"\b{h}\b", low)), None)
        if hw is None:  # "The box is kestrel-dev": the host word itself when it is named
            return display_subject(subj) if _NAME.search(s) else ""
        return hw.upper() if hw in ("gpu", "cpu", "ram", "ssd", "nvme") else display_subject(hw)
    return display_subject(subj)


# Programs whose output describes one well-known subject (page title + tags).
COMMAND_TOPICS: Tuple[Tuple[re.Pattern, str, Tuple[str, ...]], ...] = (
    (re.compile(r"^nvidia-smi\b|^nvtop\b|\blspci\b.*\b(?:vga|nvidia|3d)\b", re.I), "GPU", ("gpu", "nvidia", "hardware")),
    (re.compile(r"^(?:lscpu|nproc)\b|/proc/cpuinfo", re.I), "CPU", ("cpu", "hardware")),
    (re.compile(r"^free\b|/proc/meminfo", re.I), "Memory (RAM)", ("ram", "memory", "hardware")),
    (re.compile(r"^(?:df|lsblk|du|mount|findmnt)\b", re.I), "Disks and filesystems", ("disk", "storage")),
    (re.compile(r"^(?:uname|hostnamectl|lsb_release)\b|/etc/os-release", re.I), "Operating system",
     ("os", "kernel", "host")),
    (re.compile(r"^(?:ss|netstat|lsof\s+-i)\b", re.I), "Listening ports", ("ports", "network")),
    (re.compile(r"^vllm\b", re.I), "vLLM server", ("vllm",)),
    (re.compile(r"^(?:docker|podman)(?:-compose)?\b", re.I), "Docker containers", ("docker",)),
    (re.compile(r"^systemctl\b", re.I), "systemd services", ("systemd",)),
    (re.compile(r"^kubectl\b", re.I), "Kubernetes", ("kubernetes",)),
)
_DOCKER_TARGET = re.compile(r"^(?:docker|podman)\s+(?:inspect|logs|restart|start|stop|exec(?:\s+-\w+)*)\s+"
                            r"([A-Za-z0-9][\w.-]*)", re.I)
_SYSTEMCTL_TARGET = re.compile(r"^systemctl\s+(?:--user\s+)?(?:status|restart|start|stop|enable|is-active)\s+"
                               r"([\w@.-]+)", re.I)


def command_topic(command: str) -> Tuple[str, Tuple[str, ...]]:
    """(subject, tags) for a shell command, ("", ()) when unknown."""
    cmd = re.sub(r"^\s*(?:sudo\s+|env\s+\S+=\S+\s+)*", "", command or "").strip()
    m = _DOCKER_TARGET.match(cmd)
    if m:
        return f"Docker container {m.group(1)}", ("docker",)
    m = _SYSTEMCTL_TARGET.match(cmd)
    if m:
        return f"systemd service {m.group(1)}", ("systemd",)
    for pattern, subject, tags in COMMAND_TOPICS:
        if pattern.search(cmd):
            return subject, tags
    return "", ()


def subject_key(subject: str) -> str:
    """Normalized subject for page matching ("The vLLM Server" ≈ "vLLM server")."""
    return " ".join(w for w in normalize(_DETERMINER.sub("", subject or "")).split() if w not in ("the", "our"))


def subject_tags(subject: str) -> List[str]:
    """Salient single-word tags from a subject ("Langfuse instance" → ["langfuse"])."""
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9.+-]*", normalize(subject)):
        w = w.strip(".+-")
        if len(w) > 1 and w not in GENERIC_WORDS and stem(w) not in GENERIC_WORDS and w not in out:
            out.append(w)
    return out
