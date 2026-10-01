"""Agent-side reader (spec §2 "Reader"): read-only access to the derived index and the wiki.

Backs ``memory_search``/``memory_read``/``prefetch``/``system_prompt_block`` and the CLI. It never
writes and never rebuilds the index, so it is safe inside the agent process and works while
``pan-memoryd`` is down.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pan.index.fts import FtsIndex, Hit, present_claim, present_heading
from pan.memory.retrieval import (PREFETCH_BUDGET_CHARS, PREFETCH_K, PREFETCH_MIN_SCORE,
                                  FtsRetriever, Retriever, build_retriever, focus_query, format_prefetch,
                                  marker_notes)
from pan.paths import PanPaths
from pan.wiki.frontmatter import FrontmatterError, parse
from pan.wiki.markdown import extract_section, split_sections, strip_links
from pan.wiki.store import INDEX_PAGES_HEADING, INDEX_PAGES_INTRO, WikiStore

logger = logging.getLogger(__name__)

SECTIONS_CAP_CHARS = 800
READ_CAP_CHARS = 20_000
SEARCH_MAX_LIMIT = 20
# A hint to fall back to session_search here was tried and removed: it improved recall of tool-derived
# facts, but sent the agent into raw transcripts on noise questions (hypotheticals stated as facts).
NO_MATCH_NOTE = "No matching wiki pages."


def _pages_section(body: str) -> Optional[List[str]]:
    """The lines of the daemon-maintained ``## Pages`` section (its intro line dropped), or None."""
    lines = body.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == INDEX_PAGES_HEADING), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return [ln for ln in lines[start + 1:end] if ln.strip() != INDEX_PAGES_INTRO]


def index_overview(index_md: str, cap: int = SECTIONS_CAP_CHARS) -> str:
    """Condensed top-level sections of ``index.md`` (tables → bullets, links → text), ≤ ``cap``.
    An index with the daemon's ``## Pages`` section is shown as that page list only: the rest is the
    wiki template (area descriptions), the same in every wiki, and was ~100 tokens on every request
    of every session, even with no page in the wiki (an index without links lists nothing)."""
    try:
        _, body = parse(index_md)
    except FrontmatterError:
        body = index_md
    pages = _pages_section(body)
    if pages is None and "](" not in body:
        return ""   # the template of a wiki without pages: nothing to list
    lines: List[str] = []
    in_table = False
    for raw in (pages if pages is not None else body.splitlines()):
        line = strip_links(raw).strip()
        if not line.startswith("|"):
            in_table = False
        if not line or line.startswith("# ") or line.startswith(">"):
            continue
        if line.startswith("|"):
            first_row, in_table = not in_table, True
            cells = [c.strip().replace("`", "") for c in line.strip("|").split("|")]
            if first_row or all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # header or separator row
            cells = [c for c in cells if c]
            if cells:
                lines.append(f"- {cells[0]}" + (f": {', '.join(cells[1:])}" if len(cells) > 1 else ""))
        elif line.startswith("#"):
            lines.append(line.lstrip("#").strip() + ":")
        else:
            lines.append(line)
    out: List[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > cap:
            if cap - used > 20:
                out.append(ln[:cap - used - 2].rstrip() + "…")
            break
        out.append(ln)
        used += len(ln) + 1
    return "\n".join(out)


RERANK_DOC_CHARS = 2000

# -- memory_read presentation --------------------------------------------------------------------
# The stored page carries provenance for audits (event ids, `_Provenance:` lines, struck-through old
# claims inline, "(observed)" on every bullet). Shown raw, ~70 % of a read was provenance and the agent
# mixed a struck value with its replacement in one answer; so the agent gets the page's current facts,
# with superseded ones listed separately at the end. The file itself is unchanged (`pan wiki read`).
READ_META_KEYS = ("type", "status", "updated")
SUPERSEDED_HEADING = "Replaced by a newer statement (no longer current; still true for that earlier time or event):"
UPDATES_LABEL = "Updates (newer than the facts above; newest last)"
_LINE_ITEM = re.compile(r"^(?P<lead>\s*[-*+]\s+)(?P<text>.*?)\s*$")
_LINE_LABEL = re.compile(r"\s*\((?P<label>observed|inferred|reported|mixed)(?P<note>;[^)]*)?\)\s*$")
_STRUCK_ITEM = re.compile(r"^\s*[-*+]\s+~~(?P<text>.+?)~~\s*(?:\((?P<note>[^)]*)\))?\s*$")
_STRUCK_SPAN = re.compile(r"~~.*?~~")
_SUPERSEDED_ON = re.compile(r"superseded (\d{4}-\d{2}-\d{2})")
_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_LABEL_TEXT = {"inferred": " (inferred)", "mixed": " (partly inferred)"}
_REPORTED = re.compile(r"^\s*Assistant reported:", re.I)
_PROVENANCE = re.compile(r"^\s*_Provenance:.*_\s*$")
_PLACEHOLDER = re.compile(r"^\s*_Not recorded(?: yet)?\._\s*$")
_NOTED_END = re.compile(r"\(noted \d{4}-\d{2}-\d{2}\)$")


def _present_item(text: str, label: str) -> str:
    if label == "reported" and not _REPORTED.match(text):
        text = "Assistant reported: " + text
    text, extra = present_claim(text), _LABEL_TEXT.get(label, "")
    if extra and _NOTED_END.search(text):   # "… (noted D) (inferred)" → "… (noted D, inferred)"
        return text[:-1] + ", " + extra.strip(" ()") + ")"
    return text + extra


def render_page(body: str) -> str:
    """``body`` as ``memory_read`` shows it: current facts only, no provenance lines or ids, the
    ``(stated …)`` stamp as ``(noted …)``, no "(observed)" label (inferred/reported stay marked),
    empty sections dropped; superseded claims follow under :data:`SUPERSEDED_HEADING`."""
    out: List[tuple] = []   # (heading level or 0, text)
    old: List[str] = []
    for line in body.splitlines():
        if _PROVENANCE.match(line) or _PLACEHOLDER.match(line):
            continue
        h = _HEADING_LINE.match(line)
        if h:
            level, heading = len(h.group(1)), present_heading(h.group(2))
            if level == 2 and heading.lower() == "updates":
                heading = "Updates"
            out.append((level, f"{h.group(1)} {heading}"))
            continue
        m = _STRUCK_ITEM.match(line)
        if m:
            note = m.group("note") or ""
            label = note.split(";")[0].strip()
            when = _SUPERSEDED_ON.search(note)
            item = _present_item(m.group("text").strip(), label)
            if when:   # "… (noted D1)" → "… (noted D1; superseded D2)"
                item = (item[:-1] + f"; superseded {when.group(1)})" if item.endswith(")") and "(noted " in item
                        else item + f" (superseded {when.group(1)})")
            old.append(f"- {item}")
            continue
        if "~~" in line:   # a struck span inside a line: drop the old text
            line = _STRUCK_SPAN.sub("", line)
            if not line.strip(" -*+"):
                continue
        m = _LINE_ITEM.match(line)
        if m and m.group("text"):
            text, label = m.group("text"), ""
            lab = _LINE_LABEL.search(text)
            if lab:
                text, label = text[:lab.start()], lab.group("label")
            out.append((0, f"{m.group('lead')}{_present_item(text.strip(), label)}"))
            continue
        out.append((0, present_claim(line.rstrip())))
    kept: List[str] = []
    facts = False   # current content before the ``## Updates`` section
    for i, (level, text) in enumerate(out):
        if level > 1:   # a heading with nothing under it (up to the next heading of its level) is dropped
            rest = next((j for j in range(i + 1, len(out)) if 0 < out[j][0] <= level), len(out))
            if not any(t.strip() and not lv for lv, t in out[i + 1:rest]):
                continue
            if text == "## Updates" and facts:
                text = f"## {UPDATES_LABEL}"
        facts = facts or (not level and bool(text.strip()) and not text.startswith("User asked:"))
        if text.strip() or (kept and kept[-1].strip()):
            kept.append(text)
    body_text = "\n".join(kept).strip()
    if old:
        body_text += "\n\n" + "\n".join([SUPERSEDED_HEADING, *old])
    return body_text.strip()


class Reader:
    """``config`` (a ``RetrievalConfig``) selects the retrieval pipeline; without it, FTS only."""

    def __init__(self, paths: PanPaths, retriever: Optional[Retriever] = None, config: Any = None) -> None:
        self.paths = paths
        self.store = WikiStore(paths.wiki)
        self.index = FtsIndex(paths.index_db, readonly=True)
        self.config = config
        if retriever is None and config is not None:
            retriever = build_retriever(paths, config, purpose="reader", fts_index=self.index,
                                        doc_text=self.rerank_text)
        self.retriever = retriever or FtsRetriever(self.index)

    def rerank_text(self, hit: Hit) -> str:
        """What the reranker reads for ``hit``: the page's current content (card, claims, prose)."""
        from pan.index.vectors import page_units

        page = self.store.get_path(hit.path) if hit.path else None
        if page is None:
            return f"{hit.title}: {hit.snippet}"
        units = page_units(page)
        text = "\n".join([units[0].text] + [("- " if u.kind == "claim" else "") + u.text for u in units[1:]])
        return text[:RERANK_DOC_CHARS]

    def _accept(self, hit: Hit, default: float, rerank_min: Optional[float] = None) -> bool:
        """Threshold per stage: a reranked hit passes on its reranker relevance, or on a strong
        pre-rerank score (``rerank_keep_score``: bge-reranker-v2-m3 scores ~20 % of relevant
        DE-question/EN-page pairs near 0, dev-v1 layer A); other hits on ``default``."""
        cfg = self.config
        if cfg is None or hit.via != "rerank":
            return hit.score >= default
        floor = cfg.rerank_min_score if rerank_min is None else rerank_min
        return hit.score >= floor or hit.prior >= cfg.rerank_keep_score

    def close(self) -> None:
        self.index.close()

    # -- search / prefetch ---------------------------------------------------------------------

    def search(self, query: str, type: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}
        try:
            limit = max(1, min(SEARCH_MAX_LIMIT, int(limit)))
        except (TypeError, ValueError):
            limit = 5
        type = None if type in (None, "", "any") else str(type)
        result: Dict[str, Any] = {"query": query, "type": type or "any"}
        if not self.index.available():
            result.update(results=[], note="PAN wiki index not available (run `pan index rebuild`).")
            return result
        hits = self.retriever.retrieve(query, limit, type=type)
        if self.config is not None:  # dense/rerank always return something: keep only plausible hits
            floor = self.config.search_min_score
            hits = [h for h in hits if h.via == "fts" or self._accept(h, floor, floor)]
        result["results"] = [
            {"id": h.id, "path": h.path, "title": h.title, "type": h.type, "status": h.status,
             "confidence": h.confidence, "score": h.score, "section": present_heading(h.section), "snippet": h.snippet}
            for h in hits]
        notes = marker_notes(" ".join(h.snippet for h in hits))
        if notes or not hits:
            result["note"] = notes or NO_MATCH_NOTE
        return result

    def prefetch(self, query: str, budget_chars: int = PREFETCH_BUDGET_CHARS,
                 min_score: float = PREFETCH_MIN_SCORE) -> str:
        if not query or not query.strip():
            return ""
        cfg = self.config
        hits = self.retriever.retrieve(focus_query(query), cfg.prefetch_k if cfg is not None else PREFETCH_K)
        floor = None
        reranked = [h.score for h in hits if h.via == "rerank"]
        if cfg is not None and reranked:
            floor = max(cfg.rerank_min_score, cfg.rerank_rel_floor * max(reranked))
        accepted = [h for h in hits if self._accept(h, min_score, floor)]
        if hits:  # score and route per hit, for tuning the thresholds from real traces
            logger.info("PAN prefetch: %s", "; ".join(
                f"{h.id} {h.via} {h.score:.4f}" + (f"/{h.prior:.3f}" if h.via == "rerank" else "")
                + (" +" if h in accepted else " -") for h in hits))
        return format_prefetch(accepted, budget_chars, 0.0)

    # -- read ----------------------------------------------------------------------------------

    def read(self, page: str, section: Optional[str] = None, *, raw: bool = False) -> Dict[str, Any]:
        """The page for the agent (``memory_read``): id, title, type, status, updated and the
        :func:`render_page` content. ``raw=True`` (CLI): the full frontmatter and the stored text."""
        ref = str(page or "").strip()
        if not ref:
            return {"error": "page must be a non-empty string (id or wiki path)"}
        found = None
        path = self.index.lookup(ref)  # fast id → path via the index
        if path:
            found = self.store.get_path(path)
            if found is not None and found.id != ref:
                found = None  # index stale; fall back to scanning the wiki
        if found is None:
            found = self.store.get(ref)
        if found is None:
            return {"page": ref, "found": False, "error": f"no wiki page with id or path {ref!r}"}
        if raw:
            result: Dict[str, Any] = {"page": ref, "found": True, "id": found.id, "path": found.path,
                                      "title": found.title, "frontmatter": found.meta}
        else:
            result = {"found": True, "id": found.id, "title": found.title,
                      **{k: str(found.meta[k]) for k in READ_META_KEYS if found.meta.get(k) not in (None, "")}}
        content = found.body.strip()
        if section:
            part = extract_section(found.body, section)
            if part is None:  # a heading as shown to the agent ("2026-09-30" for "2026-09-30 · session:…")
                want = section.strip().lstrip("#").strip().lower()
                parts = [extract_section(found.body, s.heading) for s in split_sections(found.body)
                         if s.level > 0 and present_heading(s.heading).lower() == want]
                part = "\n\n".join(dict.fromkeys(p for p in parts if p)) or None
            if part is None:
                result["error"] = f"section {section!r} not found"
                heads = [s.heading if raw else present_heading(s.heading)
                         for s in split_sections(found.body) if s.level > 0]
                result["sections"] = list(dict.fromkeys(heads))
                return result
            result["section"] = section
            content = part
        if not raw:
            content = render_page(content)
        if len(content) > READ_CAP_CHARS:
            content = content[:READ_CAP_CHARS] + "\n…"
            result["truncated"] = True
        result["content"] = content
        notes = "" if raw else marker_notes(content)
        if notes:
            result["note"] = notes
        return result

    # -- system prompt -------------------------------------------------------------------------

    def wiki_overview(self, cap: int = SECTIONS_CAP_CHARS) -> str:
        try:
            text = self.store.index_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""
        return index_overview(text, cap)


def reader_for_home(hermes_home: str | Path) -> Reader:
    return Reader(PanPaths.for_home(hermes_home))
