"""Memory curator (integration spec §4.8, spec v0.1 §13). MVP: deterministic CREATE / UPDATE / IGNORE.

``Curator.curate(candidate)`` plans a change and returns a :class:`Curation` (the
:class:`CuratorDecision` plus the page to write); applying it — write, validate, commit, reindex —
is the daemon's job, so a curator never touches disk.

Policy of :class:`DeterministicCurator` for wiki candidates:

1. retrieve the top-k related pages (FTS) for ``candidate.retrieval_query``;
2. drop every claim already present in one of them (:func:`pan.memory.claims.claim_present`; the
   page without its bookkeeping — provenance lines, update headings, labels: :func:`live_text`);
   nothing left → **IGNORE**;
3. the best hit with ``score >= update_min_score`` whose type fits the candidate (decisions only
   update decisions; environment/learning facts update system/operations/learning/incident pages;
   superseded/deprecated/rejected pages are never targets) → **UPDATE**: the new claims are appended
   under ``## Updates`` with a dated provenance line;
4. otherwise → **CREATE** a page from the type template (decisions become the next
   ``decisions/ADR-NNN-<slug>.md``).

M5 refinements:

- **subject match first**: a page whose ``subject`` (frontmatter) or title equals the candidate's
  subject is the UPDATE target regardless of its FTS score (same subject = same page, also in
  a tiny wiki where scores are noisy); new pages are titled by the subject and store it;
- **supersession**: when a new *observed* claim (or any claim of a candidate that explicitly
  ``supersedes`` something, e.g. the agent's own ``memory replace``) contradicts a bullet of the
  target page (:func:`pan.memory.facts.contradiction`), that bullet is struck through
  (``- ~~old~~ (observed; superseded <date> by session:<id>)``); struck text never counts as
  "present" for the IGNORE check and is not indexed as a claim;
  For LLM-gate candidates (T3) a strike needs the gate's explicit ``supersedes`` or a typed-value
  contradiction of the *same attribute*: a conflict of measures only also needs the claims' own
  subjects to match (two budgets on one page are siblings, not an update; ``strict_measures``);
  a value conflict between claims about different occasions (other explicit dates, another place:
  :func:`different_occasion`) is never a supersession — only the gate's explicit ``supersedes`` strikes then;
- **inferred never overrides observed**: an inferred claim that contradicts a live observed
  bullet of the target page is dropped (the agent restating stale knowledge);
- **reported** claims (T3, what the assistant said it did; ``Assistant reported:`` label) never strike
  observed or inferred bullets; a newer report replaces only an older report of the same attribute,
  and an observed claim is not "present" because a report says it (it is added and confirms it);
- :meth:`DeterministicCurator.link_provenance` adds sources to a page without touching its body
  (used for ``l1_write`` events whose fact the wiki already holds).

User preferences go to L1 (``USER.md``) via :meth:`DeterministicCurator.plan_l1`; when L1 is full
the daemon calls :meth:`curate` with ``to_wiki=True`` and the preference lands on the single
``preferences/user-preferences.md`` page instead.

Every claim written is tagged ``(observed)`` or ``(inferred)``; every write adds ``session:<id>``
and ``event:<id>`` to the page's ``sources``.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from pan.events.schema import CuratorAction, CuratorDecision, Destination, MemoryCandidate, MemoryType
from pan.index.fts import Hit
from pan.memory.claims import REPORTED_PREFIX, claim_present, normalize, sentences, token_set, tokens, unlabel
from pan.memory.dates import bare
from pan.memory.facts import contradiction, display_subject, shape_update, subject_key, subject_tags, values
from pan.memory.retrieval import Retriever
from pan.wiki.frontmatter import Page
from pan.wiki.markdown import slugify
from pan.wiki.store import TYPE_DIRS, WikiStore, new_page

UPDATE_MIN_SCORE = 0.45
RETRIEVE_K = 5
UPDATES_HEADING = "## Updates"
PREFERENCES_PAGE = "preferences/user-preferences.md"
INACTIVE_STATUSES = frozenset({"superseded", "deprecated", "rejected"})

# M9: the LLM gate also labels facts as project / architecture; facts of any domain may update
# those pages too (same subject, same page — whichever domain the gate picked first).
_FACT_PAGES = frozenset({"learning", "incident", "operations", "procedure", "system", "environment",
                         "configuration", "project", "architecture"})
UPDATE_TARGET_TYPES: Dict[MemoryType, frozenset] = {
    MemoryType.DECISION: frozenset({"decision"}),
    MemoryType.LEARNING: _FACT_PAGES,
    MemoryType.ENVIRONMENT: _FACT_PAGES,
    MemoryType.CONFIGURATION: _FACT_PAGES,
    MemoryType.INCIDENT: _FACT_PAGES,
    MemoryType.PROCEDURE: _FACT_PAGES,
    MemoryType.PROJECT_FACT: _FACT_PAGES,
    MemoryType.ARCHITECTURE: _FACT_PAGES,
    MemoryType.USER_PREFERENCE: frozenset({"user_preference"}),
}
# Classifier label → page type for new pages.
PAGE_TYPE: Dict[MemoryType, str] = {
    MemoryType.USER_PREFERENCE: "user_preference", MemoryType.PROJECT_FACT: "project",
    MemoryType.ARCHITECTURE: "architecture", MemoryType.DECISION: "decision",
    MemoryType.LEARNING: "learning", MemoryType.PROCEDURE: "procedure", MemoryType.INCIDENT: "incident",
    MemoryType.ENVIRONMENT: "environment", MemoryType.CONFIGURATION: "configuration",
}
_FIRST_SECTION = {"environment": "Facts", "configuration": "Setting", "system": "Overview",
                  "operations": "Procedure", "procedure": "Steps", "architecture": "Overview",
                  "project": "Summary", "user_preference": "Preference", "incident": "Symptoms"}


@dataclass
class Curation:
    decision: CuratorDecision
    page: Optional[Page] = None          # page to write (CREATE / UPDATE)
    path: Optional[str] = None           # wiki-relative target path
    before: str = ""                     # target text before the change ("" for CREATE)
    l1_target: str = ""                  # "user" | "memory" for L1 plans
    l1_entries: List[str] = field(default_factory=list)
    new_claims: List[str] = field(default_factory=list)


class Curator(Protocol):
    name: str

    def curate(self, candidate: MemoryCandidate, *, to_wiki: bool = False) -> Curation:
        """Plan the wiki change for ``candidate`` (``to_wiki`` forces the wiki for L1 candidates)."""

    def plan_l1(self, candidate: MemoryCandidate) -> Curation:
        """Plan an L1 (USER.md / MEMORY.md) add for ``candidate``."""


def provenance(candidate: MemoryCandidate) -> List[str]:
    refs = [f"session:{candidate.session_id}"] if candidate.session_id else []
    return refs + [f"event:{i}" for i in candidate.event_ids]


def evidence_label(labels: Sequence[str]) -> str:
    kinds = set(labels)
    if not kinds:
        return ""
    return kinds.pop() if len(kinds) == 1 else "mixed"


def unified_patch(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}" if before else "/dev/null", tofile=f"b/{path}"))


def l1_entry(candidate: MemoryCandidate, claim: str) -> str:
    """USER.md / MEMORY.md entry text for one claim (compact, attributable)."""
    claim = claim.strip()
    if candidate.classification.type is MemoryType.USER_PREFERENCE:
        return f'User preference (own words): "{claim}"'
    return claim


_BULLET = re.compile(r"^(?P<lead>\s*[-*+]\s+)(?P<text>.*?)(?P<label>\s+\((?:observed|inferred|reported|mixed)[^)]*\))?\s*$")
_STRUCK = re.compile(r"~~.*?~~")
REPORTED = "reported"
_REPORTED_LINE = re.compile(r"^\s*[-*+]\s+.*\((?:reported)(?:;[^)]*)?\)\s*$", re.M)


# Page bookkeeping, not facts: provenance lines, "### <date> · session:<id>" update headings, evidence
# labels and strike notes, source ids. Their dates and ids must not make a claim's number "present" (a
# page updated on the 12th does not say "12 lessons").
_PROVENANCE_LINE = re.compile(r"^[ \t]*_Provenance:.*$", re.M)
_SOURCE_ID = re.compile(r"\b(?:session|event|l1):[\w.:-]+")
_ID_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(?:\d{4}-\d{2}-\d{2}|(?:session|event|l1):[\w.:-]+|[·, \t])+$", re.M)
_LABEL = re.compile(r"[ \t]*\((?:observed|inferred|reported|mixed)(?:;[^)\n]*)?\)")
# An ISO date is one value plus its year: day and month are no numbers of their own ("30 videos" is not
# in "2026-09-30"), so a "(stated …)" stamp can only say when — the same date or year in a claim — never
# a count.
_ISO_DATE = re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")


def _one_value_dates(text: str) -> str:
    return _ISO_DATE.sub(r"d\1\2\3 \1", text)


def fact_text(text: str) -> str:
    """``text`` (a page body or a bullet) without page bookkeeping (see :data:`_PROVENANCE_LINE`). A
    removed line leaves a word-free "·" line: sentences on both sides of it stay as far apart as before
    (``claim_present`` windows)."""
    text = _ID_HEADING.sub("·", _PROVENANCE_LINE.sub("·", text or ""))
    return _SOURCE_ID.sub("", _LABEL.sub("", text))


def fact_claim(claim: str) -> str:
    """A new claim as it is compared with :func:`live_text` (bare fact, dates as one value)."""
    return _one_value_dates(plain(claim))


def live_text(body: str, *, reported: bool = True) -> str:
    """``body`` as the text a new claim (:func:`fact_claim`) is compared with: without superseded
    (struck-through) text and page bookkeeping (:func:`fact_text`), dates as one value;
    ``reported=False`` also without the bullets of the ``reported`` class (an assistant's report
    does not make an observed fact "present")."""
    text = _STRUCK.sub("", body)
    if not reported:
        text = _REPORTED_LINE.sub("", text)
    return _one_value_dates(fact_text(text).replace(REPORTED_PREFIX, ""))


def plain(claim: str) -> str:
    """The fact of a claim or bullet: without ``(stated …)`` and the reported label."""
    return unlabel(bare(claim))


def page_bullets(body: str) -> List[tuple[int, str, str]]:
    """(line index, claim text, evidence label) for every live bullet of ``body``."""
    out = []
    for i, line in enumerate(body.split("\n")):
        if "~~" in line:
            continue
        m = _BULLET.match(line)
        if m and m.group("text").strip():
            label = (m.group("label") or "").strip(" ()").split(";")[0].strip()
            out.append((i, m.group("text").strip(), label))
    return out


def supersede_bullets(body: str, claims: Sequence[str], note: str,
                      subject: str = "", *, shape: bool = True, strict: bool = False,
                      labels: Optional[frozenset] = None) -> tuple[str, List[str], List[tuple[str, str]]]:
    """Strike through bullets of ``body`` (the page chosen for ``subject``) contradicted by one of
    ``claims``; returns (body, old texts, carried). ``carried`` = (clause, label) for the clauses of
    a struck bullet that do not contain a superseded value — **verbatim** substrings of the old
    bullet, never rewritten ("…, owned by Dana Kim" survives a new amount; see carry_clauses).
    ``strict`` (LLM-gate claims): a conflict of measures only also needs the same subject;
    ``labels``: only bullets with one of these evidence labels are candidates (reported claims only
    replace reported bullets)."""
    lines = body.split("\n")
    superseded: List[str] = []
    carried: List[tuple[str, str]] = []
    for i, text, label in page_bullets(body):
        if labels is not None and (label or "observed") not in labels:
            continue
        # "`conf/x.toml`: port = 7020" records what a file says; a statement about the service does
        # not make the file say something else — only a new reading of the same file does
        fm = _FILE_CLAIM.match(text)
        usable = [x for x in claims if not fm or (_FILE_CLAIM.match(x) and _FILE_CLAIM.match(x).group(1) == fm.group(1))]
        # a claim about another occasion (another date, another place) is a sibling, not an update
        hit = next((c for c in ((contradiction(plain(x), plain(text), on_page=True, subject=subject,
                                               strict_measures=strict)
                                 or _shape(plain(x), plain(text), words=shape))
                                if not different_occasion(x, text) else None for x in usable)
                    if c is not None), None)
        if hit is None:
            continue
        m = _BULLET.match(lines[i])
        assert m is not None
        lab = f"{label}; " if label else ""
        lines[i] = f"{m.group('lead')}~~{text}~~ ({lab}{note})"
        superseded.append(text)
        carried += [(c, label or "observed") for c in carry_clauses(text, hit, claims)]
    return "\n".join(lines), superseded, carried


_FILE_CLAIM = re.compile(r"^`([^`]+)`(?::| contains\b| reports:)")

# Dates that say which occasion a claim is about: "on 7/14", "2026-03-10", "March 3", "3. März".
_MONTHS = (r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sept|sep|"
           r"october|oct|november|nov|december|dec|januar|februar|m(?:ä|ae)rz|mai|juni|juli|oktober|okt|dezember|dez)\.?")
_MONTH_NO = {"jan": 1, "feb": 2, "mar": 3, "mär": 3, "mae": 3, "apr": 4, "may": 5, "mai": 5, "jun": 6, "jul": 7,
             "aug": 8, "sep": 9, "oct": 10, "okt": 10, "nov": 11, "dec": 12, "dez": 12}
_ISO_DAY = re.compile(r"(?<![\d-])\d{4}-(\d{2})-(\d{2})(?![\d-])")
_NUM_DAY = re.compile(r"\b(?:on|by|from|until|till|since|before|after|dated|am|vom|bis|zum)\s+(?:the\s+)?"
                      r"(\d{1,2})[/.](\d{1,2})(?:[/.](?:\d{2}|\d{4}))?(?![\d/])", re.I)
_NAMED_DAY = re.compile(rf"\b{_MONTHS}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?|"
                        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\.?\s+(?:of\s+)?{_MONTHS}(?![\w])(?:,?\s+\d{{4}}\b)?", re.I)
# "the trip to Lake Garda" / "our visit in Porto": where an occasion took place
_OCCASION = re.compile(r"\b(?:trip|visit|vacation|holiday|journey|tour|wedding|party|game|match|race|concert|festival|"
                       r"conference|outing|reunion|stay)\s+(?:to|in|at)\s+(?:the\s+)?"
                       r"([A-ZÄÖÜ][\w'-]*(?:\s+[A-ZÄÖÜ][\w'-]*)*)")


def event_days(text: str) -> set:
    """(month, day) of every explicit date in ``text`` (the ``(stated …)`` suffix is not one)."""
    text = bare(text or "")
    out = {(int(m.group(1)), int(m.group(2))) for m in _ISO_DAY.finditer(text)}
    out |= {(int(m.group(1)), int(m.group(2))) for m in _NUM_DAY.finditer(text)
            if 1 <= int(m.group(1)) <= 31 and 1 <= int(m.group(2)) <= 31}
    for m in _NAMED_DAY.finditer(text):
        num = _MONTH_NO.get((m.group(1) or m.group(4) or "").lower()[:3])
        day = int(m.group(2) or m.group(3))
        if num and 1 <= day <= 31:
            out.add((num, day))
    return out


_CLOCK = re.compile(r"(?<![\d:.])\d{1,2}[:.]\d{2}(?::\d{2})?\s*(?:am|pm|uhr|h)?(?![\d:])", re.I)
_AMOUNT = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?!\w)")
# the new claim says the thing itself changed ("moved to", "extended to", "von 184 auf 197 angepasst")
_CHANGED = re.compile(r"\b(?:mov(?:ed|es)|reschedul\w*|postpon\w*|push(?:ed)? back|extend\w*|extension|chang\w*|"
                      r"updat\w*|adjust\w*|rais(?:ed|es)|lower(?:ed|s)|increas\w*|decreas\w*|instead|no longer|now|"
                      r"verschoben|verlegt|verl(?:ä|ae)ngert|ge(?:ä|ae)ndert|angepasst|erh(?:ö|oe)ht|gesenkt|jetzt|nun)\b|"
                      r"\bfrom\b[^.;]{1,40}\bto\b|\bvon\b[^.;]{1,40}\bauf\b", re.I)


def _amounts(text: str) -> List[str]:
    """The numbers of ``text`` besides its dates and clock times."""
    for rx in (_ISO_DAY, _NUM_DAY, _NAMED_DAY, _CLOCK):
        text = rx.sub(" ", text)
    return sorted(_AMOUNT.findall(text))


def different_occasion(new: str, old: str) -> bool:
    """``new`` and ``old`` are about different occasions, not one fact that changed: both name where
    the occasion was and the places differ ("the trip to Lake Garda" / "the trip to Lake Como"), or both
    carry explicit dates that differ and a number besides the date and time differs too ("caught 7 trout on
    6/12" and "caught 9 trout on 6/19" are two outings; "the review is on 6/12" → "moved to 6/19" is one fact,
    and so is a new claim that says it changed: "the fee went from 40 to 45, due 6/19 now")."""
    new, old = plain(new), plain(old)
    pn, po = set(_OCCASION.findall(new)), set(_OCCASION.findall(old))
    if pn and po and not pn & po:
        return True
    dn, do = event_days(new), event_days(old)
    if not dn or not do or dn & do or _CHANGED.search(new):
        return False
    return _amounts(new) != _amounts(old)


def _shape(new: str, old: str, *, words: bool = True):
    """``shape_update``; with ``words=False`` only when the replaced span holds a number ("2 cards,
    not 4"), never a changed word (a sibling item: "road bike" next to "mountain bike")."""
    hit = shape_update(new, old)
    if hit is None or words:
        return hit
    spans = " ".join(v for vals in (*hit.old_values.values(), *hit.new_values.values()) for v in vals)
    return hit if re.search(r"\d", spans) else None


def already_stated(claim: str, text: str) -> bool:
    """IGNORE test: ``claim_present`` and, within one sentence of ``text``, every content word of 4+
    letters of the claim. Distinct items that differ by one word ("bought a road bike" next to "bought
    a mountain bike", a second museum visit) are new facts, not repeats (multi-session lists)."""
    if not claim_present(claim, text):
        return False
    long_words = {t for t in tokens(claim) if len(t) >= 4}
    if not long_words:
        return True
    return any(long_words <= token_set(sent) for sent in sentences(text))


def explicit_superseded(text: str, olds: Sequence[str]) -> bool:
    """A live bullet ``text`` is one of the statements a candidate explicitly replaces (M9 gate:
    ``corrects`` + the old statement): either one contains the other (claim_present: overlap and
    every number)."""
    text = plain(text)
    return any(old and (claim_present(text, o) or claim_present(o, text)) for old in olds for o in [plain(old)])


def strike_explicit(body: str, olds: Sequence[str], note: str) -> tuple[str, List[str]]:
    """Strike through the live bullets of ``body`` that ``explicit_superseded`` matches."""
    lines = body.split("\n")
    gone: List[str] = []
    for i, text, label in page_bullets(body):
        if not explicit_superseded(text, olds):
            continue
        m = _BULLET.match(lines[i])
        assert m is not None
        lab = f"{label}; " if label else ""
        lines[i] = f"{m.group('lead')}~~{text}~~ ({lab}{note})"
        gone.append(text)
    return "\n".join(lines), gone


def explicit_olds(candidate: MemoryCandidate) -> List[str]:
    """Old statements to strike: only for a candidate whose claims are all observed (the gate sets
    ``supersedes`` only then). Agent L1 ``replace`` candidates (inferred) keep the value-based
    supersession alone."""
    labels = list(candidate.claim_evidence)
    if not candidate.supersedes or not labels or any(lab != "observed" for lab in labels):
        return []
    return [str(x) for x in candidate.supersedes if str(x).strip()]


def _page_names(page: Page) -> set:
    named = " ".join([page.title, str(page.meta.get("subject") or ""), *page.tags])
    have = set(re.findall(r"[a-z0-9][a-z0-9.+-]*", normalize(named).replace("-", " ")))
    return have | {t.strip(".+") for t in have}


def carry_clauses(old: str, hit, claims: Sequence[str]) -> List[str]:
    """Clauses of ``old`` worth keeping after it was superseded: each is a verbatim substring of
    ``old``, has a typed value, contains none of the superseded values and is not already stated
    by the new claims. A comma between digits groups a number ("$7,000", "0,32 €"), it ends no clause."""
    stale = {normalize(v) for vs in hit.old_values.values() for v in vs}
    out = []
    for clause in re.split(r"(?<!\d),\s*(?:and\s+|und\s+)?|,(?!\d)\s*(?:and\s+|und\s+)?|;\s*", old.strip().rstrip(".")):
        clause = clause.strip()
        if len(clause.split()) < 3 or clause not in old:
            continue
        fact = bare(clause)   # the "(stated …)" stamp on the last clause is no value of it
        vals = values(fact).slots
        if not vals or any(normalize(v) in stale for vs in vals.values() for v in vs):
            continue
        if any(normalize(lit) in normalize(clause) for lit in stale):
            continue
        if any(claim_present(fact, plain(c)) for c in claims):
            continue
        out.append(clause)
    return out


def append_update(body: str, block: str) -> str:
    """Insert ``block`` at the end of the ``## Updates`` section (created at the end if missing)."""
    lines = body.rstrip("\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip().lower() == UPDATES_HEADING.lower()), None)
    if start is None:
        return "\n".join(lines) + f"\n\n{UPDATES_HEADING}\n\n{block.strip()}\n"
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^#{1,2}\s", lines[i])), len(lines))
    head, section, tail = lines[:start + 1], lines[start + 1:end], lines[end:]
    while section and not section[-1].strip():
        section.pop()
    new = head + section + [""] + block.strip().split("\n")
    if tail:
        new += [""] + tail
    return "\n".join(new) + "\n"


class DeterministicCurator:
    name = "deterministic-v1"

    def __init__(self, store: WikiStore, retriever: Retriever, *, today: Optional[Callable[[], str]] = None,
                 update_min_score: float = UPDATE_MIN_SCORE, k: int = RETRIEVE_K) -> None:
        self.store = store
        self.retriever = retriever
        self.today = today or (lambda: _dt.date.today().isoformat())
        self.update_min_score = update_min_score
        self.k = k

    # -- L1 ------------------------------------------------------------------------------------

    def plan_l1(self, candidate: MemoryCandidate) -> Curation:
        target = "user" if candidate.classification.destination is Destination.USER else "memory"
        entries = [l1_entry(candidate, c) for c in candidate.normalized_claims]
        decision = CuratorDecision(
            candidate_id=candidate.id, action=CuratorAction.CREATE, target_pages=[f"l1:{target}"],
            rationale=f"{candidate.classification.type.value} → L1 {target.upper()}.md (add-only)",
            confidence=candidate.classification.confidence, provenance=provenance(candidate),
            patch="".join(f"+{e}\n" for e in entries), evidence=evidence_label(candidate.claim_evidence))
        return Curation(decision=decision, l1_target=target, l1_entries=entries,
                        new_claims=list(candidate.normalized_claims))

    # -- wiki ----------------------------------------------------------------------------------

    def _hits(self, candidate: MemoryCandidate) -> List[Hit]:
        try:
            return self.retriever.retrieve(candidate.retrieval_query, k=self.k)
        except Exception:
            return []

    def curate(self, candidate: MemoryCandidate, *, to_wiki: bool = False) -> Curation:
        mtype = candidate.classification.type
        claims = list(candidate.normalized_claims)
        labels = list(candidate.claim_evidence) or ["inferred"] * len(claims)
        prov = provenance(candidate)

        if mtype is MemoryType.USER_PREFERENCE:
            existing = self.store.get_path(PREFERENCES_PAGE)
            pages: List[Page] = [existing] if existing is not None else []
            hits: List[Hit] = []
        else:
            hits = self._hits(candidate)
            pages = [p for p in (self.store.get(h.id) for h in hits) if p is not None]

        olds = explicit_olds(candidate)
        # the statements being replaced do not count as "already present" (a list that differs by one
        # name overlaps its old version almost completely)
        bodies = [strike_explicit(p.body, olds, "replaced")[0] if olds else p.body for p in pages]
        texts = [live_text(b) for b in bodies]
        confirmed = [live_text(b, reported=False) for b in bodies]
        # the "(stated <date>)" suffix is when, not what: compare the bare claim
        # LLM-gate claims are one item each: the strict test keeps distinct items apart (rules-v2
        # candidates keep the M3 presence test; rules are frozen)
        present = already_stated if candidate.rationale.startswith("llm-gate") else claim_present

        def known(i: int) -> List[str]:
            # a reported claim is present when anything says it; an observed/inferred one only when a
            # non-reported bullet does (it then confirms the report)
            return texts if i < len(labels) and labels[i] == REPORTED else confirmed

        new_idx = [i for i, c in enumerate(claims) if not any(present(fact_claim(c), t) for t in known(i))]
        if not new_idx:
            found = [p.id for p, t in zip(pages, texts) if any(present(fact_claim(c), t) for c in claims)]
            decision = CuratorDecision(
                candidate_id=candidate.id, action=CuratorAction.IGNORE, target_pages=found,
                rationale=f"all {len(claims)} claim(s) already present in {', '.join(found) or 'the wiki'}",
                confidence=0.9, provenance=prov, evidence=evidence_label(labels))
            return Curation(decision=decision)
        new_claims = [claims[i] for i in new_idx]
        new_labels = [labels[i] if i < len(labels) else "inferred" for i in new_idx]

        target = (self._superseded_page(candidate, pages) or self._subject_page(candidate)
                  or self._contradicted_page(candidate, new_claims, new_labels, pages)
                  or self._update_target(candidate, hits, pages))
        if target is not None:
            page, score = target
            observed = [b for _, b, lab in page_bullets(page.body) if lab == "observed"]
            # reported claims stay (labelled, dated, never striking anything observed)
            keep = [i for i, (c, lab) in enumerate(zip(new_claims, new_labels))
                    if lab in ("observed", REPORTED) or candidate.supersedes
                    or not any(contradiction(plain(c), plain(o)) and not different_occasion(c, o) for o in observed)]
            if not keep:
                decision = CuratorDecision(
                    candidate_id=candidate.id, action=CuratorAction.IGNORE, target_pages=[page.id],
                    rationale=f"inferred claim(s) contradict observed facts on {page.id}",
                    confidence=0.8, provenance=prov, evidence=evidence_label(new_labels))
                return Curation(decision=decision)
            new_claims = [new_claims[i] for i in keep]
            new_labels = [new_labels[i] for i in keep]
            return self._update(candidate, page, score, new_claims, new_labels, prov, len(claims))
        return self._create(candidate, new_claims, new_labels, prov, hits)

    def _superseded_page(self, candidate: MemoryCandidate, pages: Sequence[Page]) -> Optional[tuple[Page, float]]:
        """The page holding a statement the candidate explicitly replaces (``explicit_olds``). The
        old statement is also used as a retrieval query: an update in another language or under
        another subject still lands on the page it corrects."""
        olds = explicit_olds(candidate)
        if not olds:
            return None
        seen = {p.id for p in pages}
        cands = list(pages)
        for old in olds:
            try:
                hits = self.retriever.retrieve(old, k=self.k)
            except Exception:
                hits = []
            for h in hits:
                page = self.store.get(h.id)
                if page is not None and page.id not in seen:
                    seen.add(page.id)
                    cands.append(page)
        for page in cands:
            # an explicit old statement is precise: any content page that holds it (a withdrawn plan
            # may live on a decision page), not only the candidate type's usual update targets
            if page.status in INACTIVE_STATUSES or page.type in ("index", "user_preference"):
                continue
            if any(explicit_superseded(b, olds) for _, b, _ in page_bullets(page.body)):
                return page, 1.0
        return None

    def _subject_page(self, candidate: MemoryCandidate) -> Optional[tuple[Page, float]]:
        """The active page about ``candidate.subject`` (frontmatter ``subject`` or title), if any."""
        key = subject_key(candidate.subject)
        allowed = UPDATE_TARGET_TYPES.get(candidate.classification.type)
        if not key or candidate.classification.type is MemoryType.USER_PREFERENCE:
            return None
        for page in self.store.list_pages():
            if page.status in INACTIVE_STATUSES or page.type == "index":
                continue
            if allowed is not None and page.type not in allowed:
                continue
            if subject_key(str(page.meta.get("subject") or "")) == key or subject_key(page.title) == key:
                return page, 1.0
        return None

    def _contradicted_page(self, candidate: MemoryCandidate, claims: Sequence[str], labels: Sequence[str],
                           pages: Sequence[Page]) -> Optional[tuple[Page, float]]:
        """The retrieved page holding a live bullet that a new authoritative claim contradicts —
        the old fact lives there, so the update belongs there (whatever its FTS score)."""
        authoritative = [c for c, lab in zip(claims, labels) if lab == "observed" or candidate.supersedes]
        if not authoritative:
            return None
        allowed = UPDATE_TARGET_TYPES.get(candidate.classification.type)
        wanted = set(subject_tags(candidate.subject)) if candidate.subject else set()
        for page in pages:  # best hit first
            if page.status in INACTIVE_STATUSES or (allowed is not None and page.type not in allowed):
                continue
            # on a page that names the candidate's subject, a narrower/wider subject also counts
            on_page = bool(wanted) and bool(wanted & _page_names(page))
            strict = candidate.rationale.startswith("llm-gate")
            if any(contradiction(plain(c), plain(b), on_page=on_page, subject=candidate.subject if on_page else "",
                                 strict_measures=strict) and not different_occasion(c, b)
                   for _, b, _ in page_bullets(page.body) for c in authoritative):
                return page, 1.0
        return None

    def link_provenance(self, candidate: MemoryCandidate, page_id: str, extra: Sequence[str] = ()) -> Optional[Curation]:
        """UPDATE that only adds ``candidate``'s provenance (plus ``extra`` refs such as ``l1:memory``)
        to ``page_id``'s sources; None when there is nothing to add."""
        page = self.store.get(page_id)
        if page is None or page.path is None:
            return None
        prov = provenance(candidate) + [r for r in extra if r]
        sources = [str(x) for x in page.meta.get("sources") or []]
        missing = [r for r in prov if r not in sources]
        if not missing:
            return None
        before = page.to_text()
        meta = dict(page.meta)
        self._add_sources(meta, prov)
        updated = Page(meta=meta, body=page.body, path=page.path)
        decision = CuratorDecision(
            candidate_id=candidate.id, action=CuratorAction.UPDATE, target_pages=[page.id],
            rationale=f"fact already on {page.id}; provenance linked ({', '.join(missing)})",
            confidence=0.9, provenance=prov, patch=unified_patch(page.path, before, updated.to_text()),
            evidence=evidence_label(candidate.claim_evidence))
        return Curation(decision=decision, page=updated, path=page.path, before=before)

    def _update_target(self, candidate: MemoryCandidate, hits: Sequence[Hit],
                       pages: Sequence[Page]) -> Optional[tuple[Page, float]]:
        mtype = candidate.classification.type
        if mtype is MemoryType.USER_PREFERENCE:
            return (pages[0], 1.0) if pages else None
        allowed = UPDATE_TARGET_TYPES.get(mtype)
        by_id = {p.id: p for p in pages}
        # A candidate with a subject only updates a page that names it (title, tags or subject):
        # "vLLM server" facts do not become an update of the "DGX Spark host" page.
        wanted = set(subject_tags(candidate.subject)) if candidate.subject else set()
        for hit in hits:  # best first
            page = by_id.get(hit.id)
            if page is None or hit.score < self.update_min_score:
                continue
            if page.status in INACTIVE_STATUSES or (allowed is not None and page.type not in allowed):
                continue
            if wanted:
                named = " ".join([page.title, str(page.meta.get("subject") or ""), *page.tags])
                have = set(re.findall(r"[a-z0-9][a-z0-9.+-]*", normalize(named).replace("-", " ")))
                if not wanted & (have | {t.strip(".+") for t in have}):
                    continue
            return page, hit.score
        return None

    def _bullets(self, claims: Sequence[str], labels: Sequence[str]) -> List[str]:
        return [f"- {c} ({lab})" for c, lab in zip(claims, labels)]

    def _provenance_line(self, candidate: MemoryCandidate, prov: Sequence[str]) -> str:
        return f"_Provenance: pan-memoryd, {self.today()} · " + ", ".join(prov) + "_"

    @staticmethod
    def _add_sources(meta: dict, prov: Sequence[str]) -> None:
        sources = [str(s) for s in meta.get("sources") or []]
        meta["sources"] = sources + [p for p in prov if p not in sources]

    def _update(self, candidate: MemoryCandidate, page: Page, score: float, claims: List[str],
                labels: List[str], prov: List[str], total: int) -> Curation:
        before = page.to_text()
        today = self.today()
        authoritative = [c for c, lab in zip(claims, labels) if lab == "observed" or candidate.supersedes]
        note = f"superseded {today} by session:{candidate.session_id or '?'}"
        # LLM-gate candidates state corrections explicitly (corrects + old): no shape-based guess, which
        # would strike a sibling item ("bought a road bike" is not an update of "bought a mountain bike")
        gate = candidate.rationale.startswith("llm-gate")
        body, superseded, carried = (supersede_bullets(page.body, authoritative, note, candidate.subject, shape=not gate,
                                                       strict=gate)
                                     if authoritative else (page.body, [], []))
        reported = [c for c, lab in zip(claims, labels) if lab == REPORTED]
        if reported:   # a newer report replaces an older report of the same attribute, nothing else
            body, gone, _ = supersede_bullets(body, reported, note, candidate.subject, shape=False, strict=True,
                                              labels=frozenset({REPORTED}))
            superseded += gone
        olds = explicit_olds(candidate)
        if olds:
            body, gone = strike_explicit(body, olds, note)
            superseded += gone
        block = "\n".join([f"### {today} · session:{candidate.session_id or '?'}", "",
                           *self._bullets(claims, labels),
                           *[f"- {t} ({lab}; carried over)" for t, lab in carried],
                           "", self._provenance_line(candidate, prov)])
        meta = dict(page.meta)
        if str(meta.get("created", "")) <= today:
            meta["updated"] = today
        self._add_sources(meta, prov)
        updated = Page(meta=meta, body=append_update(body, block), path=page.path)
        path = page.path or ""
        how = "same subject" if score >= 1.0 and candidate.subject else f"score {score:.2f}"
        rationale = (f"{len(claims)} of {total} claim(s) new; best related page {page.id} "
                     f"({how}, type {page.type})")
        if superseded:
            rationale += f"; supersedes {len(superseded)} older claim(s): " + "; ".join(x[:60] for x in superseded)
        decision = CuratorDecision(
            candidate_id=candidate.id, action=CuratorAction.UPDATE, target_pages=[page.id],
            rationale=rationale,
            confidence=round(min(candidate.classification.confidence, 0.5 + score / 2), 2), provenance=prov,
            patch=unified_patch(path, before, updated.to_text()), evidence=evidence_label(labels))
        return Curation(decision=decision, page=updated, path=path, before=before, new_claims=claims)

    def _create_body(self, ptype: str, title: str, candidate: MemoryCandidate, claims: List[str],
                     labels: List[str], prov: List[str]) -> str:
        bullets = self._bullets(claims, labels)
        context = candidate.context.strip()
        parts = [f"# {title}", ""]
        if ptype == "decision":
            parts += ["## Context", "", context or "_Not recorded._", "", "## Decision", "", *bullets, "",
                      "## Consequences", "", "_Not recorded yet._", ""]
        elif ptype == "learning":
            parts += ["## Observation", ""] + ([context, ""] if context else []) + bullets[:1] + ["", "## Fix", ""]
            parts += (bullets[1:] or ["_Not recorded yet._"]) + ["", UPDATES_HEADING, ""]
        else:
            if context:
                parts += [context, ""]
            parts += [f"## {_FIRST_SECTION.get(ptype, 'Facts')}", "", *bullets, "", UPDATES_HEADING, ""]
        parts += [self._provenance_line(candidate, prov), ""]
        return "\n".join(parts)

    def _unique_path(self, path: str, page_id: str) -> tuple[str, str]:
        ids = {p.id for p in self.store.list_pages()}
        stem, n = path[:-3], 1
        cand_path, cand_id = path, page_id
        while (self.store.root / cand_path).exists() or cand_id in ids:
            n += 1
            cand_path, cand_id = f"{stem}-{n}.md", f"{page_id}-{n}"
        return cand_path, cand_id

    def _create(self, candidate: MemoryCandidate, claims: List[str], labels: List[str], prov: List[str],
                hits: Sequence[Hit]) -> Curation:
        mtype = candidate.classification.type
        ptype = PAGE_TYPE.get(mtype, "learning")
        today = self.today()
        observed = "observed" in labels
        common = dict(sources=prov, tags=list(candidate.tags), date=today,
                      confidence="medium" if observed else "low")
        # a title made from a claim: without the "(stated …)" suffix and the reported label
        named = plain(candidate.title or claims[0]).strip()
        if ptype == "decision":
            # User-confirmed → "accepted" (ADR lifecycle, spec §9).
            page = self.store.new_decision(named, status="accepted", body="", **common)
            number = page.meta["id"].rsplit("-", 1)[-1]
            title = f"ADR-{number}: {named}"
        elif mtype is MemoryType.USER_PREFERENCE:
            title = "User preferences"
            page = new_page(ptype, title, path=PREFERENCES_PAGE, page_id="preferences.user-preferences",
                            status="active", **common)
        else:
            title = display_subject(candidate.subject) if candidate.subject else named
            page = new_page(ptype, title, status="active" if observed else "draft", **common)
            if candidate.subject:
                page.meta["subject"] = display_subject(candidate.subject)
            path, page_id = self._unique_path(page.path or f"{TYPE_DIRS.get(ptype, ptype)}/{slugify(title)}.md",
                                              page.id)
            page.path, page.meta["id"] = path, page_id
        page.body = self._create_body(ptype, title, candidate, claims, labels, prov)
        path = page.path or ""
        related = ", ".join(f"{h.id} ({h.score:.2f})" for h in hits[:3]) or "none"
        decision = CuratorDecision(
            candidate_id=candidate.id, action=CuratorAction.CREATE, target_pages=[page.id],
            rationale=f"no related {ptype} page above {self.update_min_score:.2f} (related: {related})",
            confidence=candidate.classification.confidence, provenance=prov,
            patch=unified_patch(path, "", page.to_text()), evidence=evidence_label(labels))
        return Curation(decision=decision, page=page, path=path, before="", new_claims=claims)


_L1_WRAPPER = re.compile(r'^\s*User preference \(own words\):\s*"?(.*?)"?\s*$', re.S | re.I)


def l1_claim(entry: str) -> str:
    """The claim inside an L1 entry (``l1_entry``'s wrapper removed; other entries unchanged)."""
    m = _L1_WRAPPER.match(entry or "")
    return m.group(1) if m else (entry or "")


def same_claim(a: str, b: str) -> bool:
    """Loose equality used for L1 dedupe (normalized text or mutual containment), ignoring PAN's
    ``User preference (own words): "…"`` wrapper — the agent's ``memory`` tool may already have
    saved the same preference in its own phrasing (M4 live run)."""
    a, b = l1_claim(a), l1_claim(b)
    return normalize(a) == normalize(b) or (claim_present(a, b) and claim_present(b, a))


_NEGATION = re.compile(r"\b(?:never|not|no|don't|do not|doesn't|avoid|stop|without)\b", re.I)


def l1_covered(new: str, existing: str) -> bool:
    """True when the L1 entry ``existing`` already states ``new`` (L1 dedupe, one direction).

    ``new``'s claim must be present in ``existing`` (:func:`claim_present`) — an existing entry that
    says more ("User prefers dates in ISO 8601 format in all responses") covers a shorter new claim,
    but a more specific new claim is still added — and both must have the same polarity ("never X"
    is not covered by "always X")."""
    a, b = l1_claim(new), l1_claim(existing)
    if normalize(a) == normalize(b):
        return True
    if bool(_NEGATION.search(a)) != bool(_NEGATION.search(b)):
        return False
    return claim_present(a, b)
