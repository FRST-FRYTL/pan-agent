"""Claim text helpers shared by classifier, curator and L1 writer: sentences, tokens, overlap.

The IGNORE check ("claim already present", spec §4.8) is :func:`claim_present`: a claim is present
in a text when some window of up to ``WINDOW`` consecutive sentences of that text

1. contains at least ``PRESENT_MIN_OVERLAP`` of the claim's content tokens (stopwords dropped,
   light suffix stemming, so "enabled"/"enable" and "serves"/"served" match), and
2. contains *every* token of the claim that carries a digit (ports, versions, sizes, dates) —
   "port 8001" is new information next to "port 8000" even though the words overlap.

Windows keep long pages from matching by accident (a bag of all page words would contain almost
any claim); exact normalized substrings always count as present.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Set

PRESENT_MIN_OVERLAP = 0.75
# Claims of the ``reported`` evidence class (T3): what the assistant said it did or found, not verified.
# The label is part of the claim text, so every reader (wiki page, recall snippet, memory_read) sees it.
REPORTED_PREFIX = "Assistant reported: "
_REPORTED_RE = re.compile(r"^\s*Assistant reported:\s*", re.I)
WINDOW = 3

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have having he her
here hers him his how i if in into is it its itself just me my no nor not of on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they
this those through to too under until up very was we were what when where which while who whom why will
with would you your yours also about above after again all am any because before below between both
down during each few further more most off per via yes ok okay
t s d ll re ve m don doesn didn isn aren wasn weren won wouldn shouldn couldn hasn haven hadn
""".split())  # last line: fragments of contractions ("doesn't" -> "doesn", "t")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'`(\[_]?[A-Z0-9])")
_HARD_LINE_RE = re.compile(r"^(?:#{1,6}\s|\|)")
_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_FENCE_LINE_RE = re.compile(r"^(?:```|~~~)")
_MD_NOISE_RE = re.compile(r"[`*_>#|]+")


def normalize(text: str) -> str:
    """Lowercase ASCII, Markdown punctuation removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    text = _MD_NOISE_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    """Very light suffix stripping (enough to match inflections; not a real stemmer)."""
    if len(token) <= 4 or any(c.isdigit() for c in token):
        return token
    for suffix, repl in (("ies", "i"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: len(token) - len(suffix)] + repl
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokens(text: str) -> List[str]:
    """Content tokens (stemmed, stopwords dropped), in order, with repeats."""
    return [stem(t) for t in _TOKEN_RE.findall(normalize(text)) if t not in STOPWORDS]


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def sentences(text: str, *, join_lines: bool = True) -> List[str]:
    """Split prose or Markdown into sentences. Wrapped lines of a paragraph or list item are joined
    first (``join_lines=False``: every line break is a hard break, for chat messages); headings,
    table rows, list items and blank lines are hard breaks; bullet markers dropped."""
    out: List[str] = []
    block: List[str] = []

    def flush() -> None:
        if block:
            para = " ".join(block)
            out.extend(p.strip() for p in _SENTENCE_RE.split(para) if p.strip())
            block.clear()

    for line in str(text or "").splitlines():
        s = line.strip()
        if not s or _FENCE_LINE_RE.match(s):
            flush()
            continue
        if _HARD_LINE_RE.match(s):
            flush()
            out.append(s.lstrip("#").strip())
            continue
        item = _ITEM_RE.match(s)
        if item:
            flush()
            s = s[item.end():]
        block.append(s)
        if not join_lines:
            flush()
    flush()
    return [s for s in out if s]


def overlap(claim: str, text: str) -> float:
    """Share of the claim's content tokens that occur in ``text`` (0 when the claim has none)."""
    want = token_set(claim)
    if not want:
        return 0.0
    return len(want & token_set(text)) / len(want)


def _numeric(tokens_: Iterable[str]) -> Set[str]:
    return {t for t in tokens_ if any(c.isdigit() for c in t)}


def claim_present(claim: str, text: str, *, min_overlap: float = PRESENT_MIN_OVERLAP,
                  window: int = WINDOW) -> bool:
    """True when ``claim`` is already stated in ``text`` (see module docstring)."""
    want = token_set(claim)
    if not want:
        return True  # nothing to add
    norm_claim = normalize(claim).rstrip(".")
    if norm_claim and norm_claim in normalize(text):
        return True
    numeric = _numeric(want)
    sents = [token_set(s) for s in sentences(text)]
    for i in range(len(sents)):
        have: Set[str] = set()
        for j in range(i, min(i + window, len(sents))):
            have |= sents[j]
            if numeric <= have and len(want & have) / len(want) >= min_overlap:
                return True
    return False


def missing_claims(claims: Sequence[str], texts: Sequence[str]) -> List[int]:
    """Indexes of ``claims`` not present in any of ``texts``."""
    return [i for i, c in enumerate(claims) if not any(claim_present(c, t) for t in texts)]


def shorten(text: str, max_len: int = 70, *, title: bool = True) -> str:
    """Cut at a word boundary. ``title=True`` also drops trailing punctuation; quotes keep it and
    get an ellipsis when cut."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if title:
        text = text.rstrip(".;:,!?")
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return (cut.rstrip(".;:,!?-(") or text[:max_len]) + ("" if title else " …")


def unlabel(text: str) -> str:
    """``text`` without the :data:`REPORTED_PREFIX` label (for comparing facts)."""
    return _REPORTED_RE.sub("", text or "")
