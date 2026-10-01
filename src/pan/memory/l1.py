"""L1 writer (integration spec §4.9, D1): promote facts into Hermes' ``USER.md`` / ``MEMORY.md``.

Writes go through Hermes' own ``MemoryStore`` (``pan.hermes.compat.load_l1_store``): same file lock,
same entry format, so a running agent's store sees no "external drift". Policy:

- **add** new entries (preferences); PAN never removes or rewrites entries on its own initiative;
- **reconcile** (M5, :meth:`L1Writer.reconcile`, D1 "PAN owns automatic L1 writes"): when a newer
  observation contradicts a *factual* entry (:func:`pan.memory.facts.contradiction`: same subject,
  a common slot — port, host, path, version, GPU model, … — with a different value), that entry is
  superseded through ``MemoryStore.replace`` — rewritten in its own phrasing with the new value
  when unambiguous ("vLLM server runs on port 8000." → "… 8010."), else replaced by the new claim.
  If the new fact is already another entry, the stale one is removed (``MemoryStore.remove``).
  Preference entries (PAN's ``User preference (own words)`` wrapper, "prefers/wants/always/never
  …") are never touched; neither is an entry the caller marks as newer than the observation;
- **explicit** (T3, :meth:`L1Writer.reconcile_explicit`): an entry that states an old statement the
  gate explicitly replaces (``corrects`` + ``old``, the same match the curator uses to strike the wiki
  bullet) is replaced by the new claim that fits it best, even without a common typed value (a
  German price, a reworded plate number): the wiki and L1 stay in step;
- **dedupe** — an entry already stated by an existing entry of either L1 file (same polarity) is not
  added again (PAN's `User preference (own words): "…"` wrapper is ignored when comparing);
- **char limits** — if the entry does not fit, nothing is written and the result is ``l1_full``;
  the daemon then lets the curator put the fact into the wiki instead.

Hermes resolves the memories directory from ``HERMES_HOME``; every call runs inside
``compat.hermes_home_scope(hermes_home)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from pan.memory.claims import claim_present, overlap, sentences
from pan.memory.curator import l1_covered
from pan.memory.facts import contradiction, strip_lead, values

logger = logging.getLogger(__name__)

ADDED = "added"
SUPERSEDED = "superseded"      # entry replaced by the updated fact
REMOVED = "removed"            # stale entry removed (the updated fact is already another entry)
KEPT_NEWER = "kept_newer"      # contradicting entry was written after the observation: left alone
DUPLICATE = "duplicate"
L1_FULL = "l1_full"
DISABLED = "disabled"
ERROR = "error"
TARGETS = ("user", "memory")


@dataclass
class L1Result:
    status: str      # added | duplicate | l1_full | disabled | error | superseded | removed | kept_newer
    target: str
    entry: str
    message: str = ""
    usage: str = ""
    replacement: str = ""   # superseded: the entry's new text

    @property
    def ok(self) -> bool:
        return self.status in (ADDED, DUPLICATE)


# Tastes count too ("Enjoys hobbies that need focus", "can't stand crowds"): present-tense forms only,
# so a past event ("the friend loved the gift") stays a fact.
_PREFERENCE_ENTRY = re.compile(
    r"^\s*User preference \(own words\)|\b(?:prefers?|preference|wants?|likes?|dislikes?|hates?|"
    r"always|never|don't|do not|please|style|tone|language|answers?|replies|responses?|"
    r"enjoys?|loves?|favou?rites?|can't stand|cannot stand|a (?:big |huge )?fan of|"
    r"avoids?|(?:doesn't|does not) care for)\b", re.I)


def is_preference_entry(entry: str) -> bool:
    """Entries about how the user wants to be served — never superseded by PAN."""
    return bool(_PREFERENCE_ENTRY.search(entry or ""))


def fact_entry(claim: str) -> str:
    """L1 entry text for a claim that has no old entry to rewrite ("Update: we moved …" → "We moved …")."""
    text = strip_lead(claim).strip()
    return text[:1].upper() + text[1:] if text[:1].islower() and text.split(" ", 1)[0].islower() else text


def _states_old(entry: str, olds: Sequence[str]) -> bool:
    """``entry`` says no more than one of the explicitly replaced statements (it fits inside it). An
    entry that merely contains the old statement next to other facts is left alone: replacing the
    whole entry by one new claim would drop the other facts (T3 replay)."""
    parts = sentences(entry) or [entry]
    return any(old and all(claim_present(p, old) for p in parts) for old in olds)


# Sentence boundaries of an L1 entry, kept as separators so the other sentences survive verbatim.
_ENTRY_SPLIT = re.compile(r"((?<=[.!?])[ \t]+(?=[\"'`(\[_]?[A-Z0-9ÄÖÜ])|\s*\n\s*)")


def _replace_old_sentences(entry: str, olds: Sequence[str], new: str) -> Optional[str]:
    """``entry`` with the sentences that state one of ``olds`` replaced by ``new`` (the first one) or
    dropped (the rest); ``None`` when no sentence, or every sentence, states an old statement (the
    whole-entry path handles the latter). The other sentences are kept word for word."""
    parts = _ENTRY_SPLIT.split(entry)
    texts = parts[0::2]
    hit = [bool(t.strip()) and any(old and claim_present(t, old) for old in olds) for t in texts]
    if not any(hit) or all(h or not t.strip() for h, t in zip(hit, texts)):
        return None
    out: List[str] = []
    placed = False
    for i, text in enumerate(texts):
        if hit[i]:
            if placed:
                continue
            text, placed = new, True
        if out:
            out.append(parts[2 * i - 1] if i > 0 else " ")
        out.append(text)
    return "".join(out).strip()


def _records(entry: str, claim: str) -> bool:
    """``entry`` already states ``claim``, every typed value included."""
    want = values(claim).slots
    have = values(entry).slots
    return claim_present(claim, entry) and all(vs <= have.get(k, set()) for k, vs in want.items())


class L1Writer:
    def __init__(self, hermes_home: str | Path, *, store_factory: Optional[Callable[[], Any]] = None) -> None:
        self.hermes_home = Path(hermes_home).expanduser()
        self._factory = store_factory

    def _load(self) -> Any:
        from pan.hermes import compat
        return (self._factory or compat.load_l1_store)()

    def entries(self, target: str) -> List[str]:
        from pan.hermes import compat
        with compat.hermes_home_scope(self.hermes_home):
            return compat.l1_entries(self._load(), target)

    def add(self, target: str, entry: str) -> L1Result:
        """Add one entry (never raises; errors come back as ``status == "error"``)."""
        entry = (entry or "").strip()
        if target not in TARGETS:
            return L1Result(ERROR, target, entry, f"unknown L1 target {target!r}")
        if not entry:
            return L1Result(ERROR, target, entry, "empty entry")
        try:
            from pan.hermes import compat
            with compat.hermes_home_scope(self.hermes_home):
                store = self._load()
                if not compat.l1_target_enabled(store, target):
                    return L1Result(DISABLED, target, entry, f"L1 target {target} is disabled in Hermes config")
                existing = compat.l1_entries(store, target)
                # Both L1 files reach the prompt: a claim the agent already saved in the other
                # target (Qwen3.8 put a user preference into MEMORY.md in the M4 live run) counts.
                other = next(t for t in TARGETS if t != target)
                seen = existing + (compat.l1_entries(store, other) if compat.l1_target_enabled(store, other) else [])
                match = next((e for e in seen if l1_covered(entry, e)), None)
                if match is not None:
                    return L1Result(DUPLICATE, target, entry, f"already present: {match[:80]!r}")
                limit = compat.l1_char_limit(store, target)
                size = len(compat.l1_entry_delimiter().join(existing + [entry]))
                if size > limit:
                    logger.info("L1 %s full (%d/%d chars); leaving %r to the wiki", target, size, limit, entry[:60])
                    return L1Result(L1_FULL, target, entry, f"would need {size}/{limit} chars", f"{size}/{limit}")
                result = store.add(target, entry)
        except Exception as exc:  # Hermes missing / changed: the event is retried, then dead
            logger.warning("L1 write to %s failed: %s", target, exc)
            return L1Result(ERROR, target, entry, f"{type(exc).__name__}: {exc}")
        message = str(result.get("message") or result.get("error") or "")
        if result.get("success"):
            if "already exists" in message.lower():
                return L1Result(DUPLICATE, target, entry, message, str(result.get("usage") or ""))
            logger.info("L1 %s: added %r", target, entry[:80])
            return L1Result(ADDED, target, entry, message, str(result.get("usage") or ""))
        status = L1_FULL if "exceed" in message.lower() else ERROR
        return L1Result(status, target, entry, message, str(result.get("usage") or ""))

    def reconcile(self, claim: str, *, keep: Sequence[str] = (),
                  newer: Optional[Callable[[str], bool]] = None) -> List[L1Result]:
        """Supersede L1 entries that ``claim`` (a newer observation) contradicts; see module docstring.

        ``keep`` = entries never touched (e.g. the agent's own new entry); ``newer(entry)`` → True
        when the entry was written after the observation (then it is kept, status ``kept_newer``).
        Returns one result per contradicted entry (``superseded`` / ``removed`` / ``kept_newer`` /
        ``l1_full`` / ``error``); never raises."""
        claim = (claim or "").strip()
        if not claim:
            return []
        out: List[L1Result] = []
        try:
            from pan.hermes import compat
            with compat.hermes_home_scope(self.hermes_home):
                store = self._load()
                targets = [t for t in TARGETS if compat.l1_target_enabled(store, t)]
                for target in targets:
                    for entry in compat.l1_entries(store, target):
                        if entry in keep or is_preference_entry(entry):
                            continue
                        # a measure of another named person or thing is not this entry's (T3 replay)
                        c = contradiction(claim, entry, distinct_names=True)
                        if c is None:
                            continue
                        if newer is not None and newer(entry):
                            out.append(L1Result(KEPT_NEWER, target, entry, "entry is newer than the observation"))
                            continue
                        out.append(self._supersede(store, target, entry, c.rewritten or fact_entry(claim), targets))
        except Exception as exc:
            logger.warning("L1 reconcile failed: %s", exc)
            out.append(L1Result(ERROR, "", claim, f"{type(exc).__name__}: {exc}"))
        return out

    def reconcile_explicit(self, olds: Sequence[str], claims: Sequence[str], *, keep: Sequence[str] = (),
                           newer: Optional[Callable[[str], bool]] = None) -> List[L1Result]:
        """Replace L1 entries that state one of ``olds`` (statements a candidate explicitly
        supersedes) by the best-fitting new claim; same guards and results as :meth:`reconcile`."""
        olds = [o for o in olds if o and o.strip()]
        claims = [c.strip() for c in claims if c and c.strip()]
        if not olds or not claims:
            return []
        out: List[L1Result] = []
        try:
            from pan.hermes import compat
            with compat.hermes_home_scope(self.hermes_home):
                store = self._load()
                targets = [t for t in TARGETS if compat.l1_target_enabled(store, t)]
                for target in targets:
                    for entry in compat.l1_entries(store, target):
                        if entry in keep or is_preference_entry(entry):
                            continue
                        whole = _states_old(entry, olds)
                        if not whole and not any(claim_present(o, entry) for o in olds):
                            continue
                        if any(_records(entry, c) for c in claims):
                            continue   # the entry already records the new statement ("9035 (was 4172)")
                        best = max(claims, key=lambda c: overlap(c, entry))
                        # a multi-sentence entry: only the sentence that states the old statement is replaced
                        new = fact_entry(best) if whole else _replace_old_sentences(entry, olds, fact_entry(best))
                        if new is None:
                            continue
                        if newer is not None and newer(entry):
                            out.append(L1Result(KEPT_NEWER, target, entry, "entry is newer than the observation"))
                            continue
                        out.append(self._supersede(store, target, entry, new, targets))
        except Exception as exc:
            logger.warning("L1 explicit reconcile failed: %s", exc)
            out.append(L1Result(ERROR, "", "; ".join(olds)[:120], f"{type(exc).__name__}: {exc}"))
        return out

    def _supersede(self, store: Any, target: str, entry: str, new: str, targets: Sequence[str]) -> L1Result:
        from pan.hermes import compat
        current = [e for t in targets for e in compat.l1_entries(store, t) if e != entry]
        if any(l1_covered(new, e) for e in current):
            result = store.remove(target, entry)
            status, detail = REMOVED, f"stale entry removed; updated fact already in L1: {new[:80]!r}"
        else:
            entries = compat.l1_entries(store, target)
            size = len(compat.l1_entry_delimiter().join([new if e == entry else e for e in entries]))
            limit = compat.l1_char_limit(store, target)
            if size > limit:
                return L1Result(L1_FULL, target, entry, f"replacement would need {size}/{limit} chars")
            result = store.replace(target, entry, new)
            status, detail = SUPERSEDED, f"replaced by {new[:120]!r}"
        if not result.get("success"):
            return L1Result(ERROR, target, entry, str(result.get("error") or result.get("message") or result))
        logger.info("L1 %s: %s %r (%s)", target, status, entry[:80], detail)
        return L1Result(status, target, entry, detail, str(result.get("usage") or ""),
                        replacement=new if status == SUPERSEDED else "")
