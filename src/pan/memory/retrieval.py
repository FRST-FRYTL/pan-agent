"""Retrieval interface (spec §4.7, §12). FTS5, plus dense vectors and a reranker (M7–M8, ADR-012).

``Retriever`` is what the daemon's curator and the agent-side reader depend on. Every implementation
returns ``Hit`` objects, best first, with a normalized ``score`` in [0, 1] — callers and thresholds
(``prefetch_min_score``, the curator's UPDATE threshold) never see raw BM25 or cosine values.

- :class:`FtsRetriever`   FTS5 (lexical; always available).
- :class:`DenseRetriever` multilingual embeddings from the local sidecar over the derived
  ``vectors.db``; cosine is mapped to [0, 1] with a per-model floor/ceiling.
- :class:`FusedRetriever` weighted reciprocal-rank fusion of several retrievers for the *order*; the
  best component score is the hit's ``score`` (so an FTS-only setup and a fused one share thresholds).
- :class:`RerankRetriever` a cross-encoder over the fused candidates, within a latency budget; its
  relevance becomes the score (``via="rerank"``, thresholded with ``rerank_min_score``).

Every stage degrades: a sidecar that is down or slow leaves FTS (or the fused order) in place.
:func:`build_retriever` wires the stages from ``RetrievalConfig``.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from pan.index.fts import (CURATION_UNKNOWN_TERM_WEIGHT, SNIPPET_MAX_CHARS, SNIPPET_MAX_ITEMS,
                           UNKNOWN_TERM_WEIGHT, FtsIndex, Hit, is_context_line, present_claim, present_heading)

if TYPE_CHECKING:
    from pan.config import RetrievalConfig
    from pan.index.embed import EmbedClient
    from pan.index.vectors import VectorIndex
    from pan.paths import PanPaths

__all__ = ["Hit", "Retriever", "FtsRetriever", "DenseRetriever", "FusedRetriever", "RerankRetriever",
           "build_retriever", "focus_query", "format_prefetch", "marker_notes", "CURATION_UNKNOWN_TERM_WEIGHT",
           "PREFETCH_BUDGET_CHARS", "PREFETCH_MIN_SCORE"]

logger = logging.getLogger(__name__)

# Defaults from spec §5 (`retrieval.*` in pan/config.yaml). TODO: read from pan.config after merge.
PREFETCH_BUDGET_CHARS = 1500
PREFETCH_MIN_SCORE = 0.2
PREFETCH_K = 5


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5, *, type: Optional[str] = None) -> List[Hit]:
        """Top-``k`` pages for ``query`` (best first). Must not raise on odd input."""


class FtsRetriever(Retriever):
    """FTS5 retrieval. ``unknown_weight``: how much query terms the wiki does not contain lower the
    score (``pan.index.fts.UNKNOWN_TERM_WEIGHT`` for recall, ``CURATION_UNKNOWN_TERM_WEIGHT`` for
    the curator's "is this already covered?" question)."""

    def __init__(self, index: FtsIndex, *, unknown_weight: Optional[float] = None) -> None:
        self.index = index
        self.unknown_weight = UNKNOWN_TERM_WEIGHT if unknown_weight is None else unknown_weight

    def retrieve(self, query: str, k: int = 5, *, type: Optional[str] = None) -> List[Hit]:
        return self.index.search(query, type=type, limit=k, unknown_weight=self.unknown_weight)


class FusedRetriever(Retriever):
    """Weighted reciprocal-rank fusion of several retrievers: ``rrf(d) = Σ w_i / (k + rank_i(d))``.

    Order = fused rank; ``score`` = the best component score of the page (thresholds keep their
    meaning); snippet/section come from the component that ranked the page best. A component that
    raises is skipped (e.g. an embedding service that is down) — retrieval degrades to the rest."""

    def __init__(self, retrievers: Sequence[Tuple[Retriever, float]], *, k: int = 60, depth: int = 20) -> None:
        if not retrievers:
            raise ValueError("FusedRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.k = k
        self.depth = depth

    def retrieve(self, query: str, k: int = 5, *, type: Optional[str] = None) -> List[Hit]:
        fused: Dict[str, float] = {}
        best: Dict[str, Tuple[int, Hit]] = {}
        top_score: Dict[str, float] = {}
        for retriever, weight in self.retrievers:
            try:
                hits = retriever.retrieve(query, max(k, self.depth), type=type)
            except Exception:
                continue
            for rank, hit in enumerate(hits, start=1):
                fused[hit.id] = fused.get(hit.id, 0.0) + weight / (self.k + rank)
                if hit.id not in best or rank < best[hit.id][0]:
                    best[hit.id] = (rank, hit)
                top_score[hit.id] = max(top_score.get(hit.id, 0.0), hit.score)
        order = sorted(fused, key=lambda i: (-fused[i], -top_score[i], best[i][1].path))
        return [replace(best[i][1], score=top_score[i]) for i in order[:k]]


def _dense_snippet(units: Sequence[tuple]) -> Tuple[str, str, str]:
    """(heading, anchor, snippet) from a page's best-matching units: the best one, plus the next
    claim when both fit ``SNIPPET_MAX_CHARS`` (mirrors the FTS claim snippet). The "User asked"
    context line of a page is never the snippet."""
    ranked = [u for u in units if u[1] != "card" and not (u[1] == "text" and is_context_line(u[4]))]
    if not ranked:
        return "", "", ""
    best = ranked[0]
    chosen = [best[4]]
    for u in ranked[1:SNIPPET_MAX_ITEMS]:
        if u[1] == "claim" and best[1] == "claim" and len(" · ".join(chosen + [u[4]])) <= SNIPPET_MAX_CHARS:
            chosen.append(u[4])
    snippet = present_claim(" · ".join(chosen))
    if len(snippet) > SNIPPET_MAX_CHARS:
        snippet = snippet[:SNIPPET_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return best[2], best[3], snippet


class DenseRetriever(Retriever):
    """Embedding retrieval over the derived vector index. ``score`` = cosine mapped linearly from
    [``floor``, ``ceiling``] to [0, 1] (per-model calibration, ADR-012). Raises when the sidecar
    is unavailable (the fused retriever then skips it); returns [] when the index is empty or was
    built with another model than the sidecar serves."""

    def __init__(self, index: "VectorIndex", client: "EmbedClient", *, floor: float = 0.3,
                 ceiling: float = 0.8, timeout_s: Optional[float] = None) -> None:
        self.index = index
        self.client = client
        self.floor = floor
        self.ceiling = max(ceiling, floor + 1e-6)
        self.timeout_s = timeout_s

    def compatible(self) -> bool:
        have = self.index.model()
        if have is None:
            return False
        served = self.client.embed_model_id()
        if served is None:
            return False
        dim = str(self.client.dimensions or served.get("dim", ""))
        if (have["model_id"], have["revision"], have["dim"]) != (served.get("model_id"), served.get("revision"), dim):
            logger.warning("PAN vectors were built with %s@%s/%s but the sidecar serves %s@%s/%s; dense retrieval "
                           "off until `pan index rebuild`", have["model_id"], have["revision"][:12], have["dim"],
                           served.get("model_id"), str(served.get("revision"))[:12], dim)
            return False
        return True

    def calibrate(self, sim: float) -> float:
        return round(min(1.0, max(0.0, (sim - self.floor) / (self.ceiling - self.floor))), 4)

    def retrieve(self, query: str, k: int = 5, *, type: Optional[str] = None) -> List[Hit]:
        query = str(query or "").strip()
        if not query or k <= 0 or not self.compatible():
            return []
        qvec = self.client.embed([query], "query", timeout=self.timeout_s)[0]
        hits = []
        for m in self.index.search(qvec, k, type=type):
            section, anchor, snippet = _dense_snippet(m.units)
            hits.append(Hit(id=m.id, path=m.path, title=m.title, type=m.type, status=m.status,
                            score=self.calibrate(m.sim), bm25=0.0, section=section, anchor=anchor,
                            snippet=snippet, confidence=m.confidence, updated=m.updated, via="dense"))
        return hits


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
MAX_RERANK_QUERIES = 3
MAX_RERANK_PARAGRAPHS = 2
# An absolute filesystem path (``/a/b/c``, ``~/a/b``), not part of a URL or a relative path.
_ABS_PATH_RE = re.compile(r"(?<![\w/~.:])~?/(?:[^\s/]+/)+([^\s/]*)")


def focus_query(query: str) -> str:
    """The user message as a prefetch query: absolute filesystem paths shortened to their last
    component. A path's directories (a working directory a client puts in front of the request, a
    checkout location) say where something lives, not what the turn is about; their many tokens lifted
    every page's reranker score just above the prefetch floor, and the page the request was about
    lost against them."""
    return _ABS_PATH_RE.sub(lambda m: m.group(1), str(query or ""))


def rerank_queries(query: str) -> List[str]:
    """The texts a page is reranked against: the whole message plus, for a longer multi-sentence
    message, each of its questions (up to ``MAX_RERANK_QUERIES`` in total), and for a message of
    several paragraphs its last ``MAX_RERANK_PARAGRAPHS`` paragraphs. A cross-encoder scores a
    focused question well but a chatty turn ("…here is the context. What do I answer? No need to open
    the file.") near 0 even for the right page (seen on a validation run), and the same holds for
    a request behind a preamble (the environment a client describes first); the best of these scores
    is the page's relevance."""
    text = str(query or "")
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH_RE.split(text) if p.strip()]
    query = " ".join(text.split())
    parts = [p.strip() for p in _SENTENCE_RE.split(query) if p.strip()]
    out = [query]
    if len(parts) >= 2:
        questions = [p for p in parts if p.endswith("?") and len(p) >= 12]
        for q in sorted(questions, key=len, reverse=True):
            if len(out) >= MAX_RERANK_QUERIES:
                break
            if q != query:
                out.append(q)
    if len(paragraphs) >= 2:
        out += [p for p in paragraphs[::-1] if len(p) >= 12 and p not in out][:MAX_RERANK_PARAGRAPHS]
    return out


class RerankRetriever(Retriever):
    """Cross-encoder reranking of ``base``'s top-``depth`` candidates. ``doc_text(hit)`` gives the
    text the reranker reads (the reader passes the page's live content). The reranker's relevance
    becomes the hit's ``score`` (``via="rerank"``). Over the ``timeout_s`` budget, or when the
    sidecar fails, the base order and scores are returned unchanged."""

    def __init__(self, base: Retriever, client: "EmbedClient", doc_text: Callable[[Hit], str], *,
                 depth: int = 20, timeout_s: float = 0.5, segments: bool = True) -> None:
        self.base = base
        self.client = client
        self.doc_text = doc_text
        self.depth = depth
        self.timeout_s = timeout_s
        self.segments = segments

    def retrieve(self, query: str, k: int = 5, *, type: Optional[str] = None) -> List[Hit]:
        from pan.index.embed import EmbedError

        hits = self.base.retrieve(query, max(k, self.depth), type=type)
        if not hits:
            return []
        t0 = time.monotonic()
        try:
            docs = [self.doc_text(h) for h in hits]
            scores = self.client.rerank(query, docs, timeout=self.timeout_s)
        except (EmbedError, OSError, ValueError) as exc:
            logger.debug("PAN rerank skipped (%s)", exc)
            return hits[:k]
        for extra in (rerank_queries(query)[1:] if self.segments else []):
            left = self.timeout_s - (time.monotonic() - t0)
            if left <= 0.05:
                break
            try:  # a segment over the budget only loses its own contribution
                scores = [max(a, b) for a, b in zip(scores, self.client.rerank(extra, docs, timeout=left))]
            except (EmbedError, OSError, ValueError):
                break
        logger.debug("PAN rerank of %d candidates took %.0f ms", len(hits), 1000 * (time.monotonic() - t0))
        ranked = sorted(zip(scores, range(len(hits)), hits), key=lambda x: (-x[0], x[1]))
        return [replace(h, score=round(float(s), 4), via="rerank", prior=h.score) for s, _, h in ranked[:k]]


def build_retriever(paths: "PanPaths", cfg: "RetrievalConfig", *, purpose: str = "reader",
                    fts_index: Optional[FtsIndex] = None, readonly: bool = True,
                    doc_text: Optional[Callable[[Hit], str]] = None) -> Retriever:
    """The configured pipeline. ``purpose``: ``reader`` (prefetch, memory_search) or ``curator``
    (FTS with the curation unknown-term weight; reranking only if ``curator_rerank``)."""
    from pan.index.embed import EmbedClient
    from pan.index.vectors import VectorIndex

    index = fts_index or FtsIndex(paths.index_db, readonly=readonly)
    fts = FtsRetriever(index, unknown_weight=CURATION_UNKNOWN_TERM_WEIGHT if purpose == "curator" else None)
    mode = cfg.curator_mode if purpose == "curator" else cfg.mode
    if mode != "hybrid":
        return fts
    client = EmbedClient(cfg.embed_url, timeout_s=cfg.timeout_s, embed_model=cfg.embed_model,
                         rerank_model=cfg.rerank_model, dimensions=cfg.embed_dimensions,
                         query_prefix=cfg.embed_query_prefix)
    curator = purpose == "curator"
    dense = DenseRetriever(VectorIndex(paths.vectors_db, readonly=readonly), client,
                           floor=cfg.curator_dense_floor if curator else cfg.dense_floor,
                           ceiling=cfg.curator_dense_ceiling if curator else cfg.dense_ceiling,
                           timeout_s=max(cfg.timeout_s, 10.0) if curator else cfg.timeout_s)
    fused: Retriever = FusedRetriever([(fts, 1.0), (dense, cfg.dense_weight)], k=cfg.rrf_k,
                                      depth=max(cfg.rerank_depth, 20))
    rerank = cfg.curator_rerank if purpose == "curator" else cfg.rerank
    if rerank and doc_text is not None:
        return RerankRetriever(fused, client, doc_text, depth=cfg.rerank_depth, timeout_s=cfg.rerank_timeout_s,
                               segments=cfg.rerank_segments)
    return fused


PREFETCH_HEADER = "Possibly relevant PAN wiki pages (open with memory_read):"
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
# What the presentation markers mean, said where they appear (prefetch, memory_read, memory_search)
# rather than in the static system prompt, which every request carried whether or not a marker was in
# sight. Unexplained, the agent read the recall stamp as content ("since the 30th" from a claim noted
# that day).
MARKER_NOTES = (
    (re.compile(r"\(noted \d{4}"),
     '"(noted DATE)" is when a fact was recorded, not when it happened: never derive dates from it.'),
    (re.compile(r"Assistant reported:"), '"Assistant reported:" marks the assistant\'s own account.'),
    (re.compile(r"\binferred\)"), '"(inferred)" facts were not said outright.'),
    (re.compile(r"\bsuperseded\b", re.I),
     "A superseded fact is no longer current but still answers questions about that earlier time."),
)


def marker_notes(text: str) -> str:
    """The explanations of the markers that occur in ``text`` ("" when none does)."""
    return " ".join(note for pattern, note in MARKER_NOTES if pattern.search(text or ""))


def format_prefetch(hits: Sequence[Hit], budget_chars: int = PREFETCH_BUDGET_CHARS,
                    min_score: float = PREFETCH_MIN_SCORE) -> str:
    """Compact recall block of at most ``budget_chars`` characters; "" if no hit is good enough."""
    good = [h for h in hits if h.score >= min_score]
    if not good or budget_chars <= len(PREFETCH_HEADER) + 20:
        return ""
    lines = [PREFETCH_HEADER]
    used = len(PREFETCH_HEADER)
    for h in good:
        # a dated update heading says no more than the snippet's "(noted …)" stamp: no label
        section = present_heading(h.section)
        shown = section and section != h.title and not _DATE_ONLY.fullmatch(section)
        where = f" § {section}" if shown else ""
        status = "" if h.status in ("", "active") else f", {h.status}"   # the default status says nothing
        head = f"- [{h.id}] {h.title}{where} ({h.type}{status})"
        line = f"{head}: {h.snippet}" if h.snippet else head
        room = budget_chars - used - 1  # newline
        if len(line) > room:
            if len(head) + 8 > room:
                break
            line = line[:room - 1].rstrip() + "…"
        lines.append(line)
        used += len(line) + 1
    notes = marker_notes("\n".join(lines[1:]))
    while notes and len(lines) > 2 and len("\n".join(lines)) + 1 + len(notes) > budget_chars:
        lines.pop()   # the notes count against the budget: the last hit makes room
        notes = marker_notes("\n".join(lines[1:]))
    if notes and len("\n".join(lines)) + 1 + len(notes) <= budget_chars:
        lines.append(notes)
    return "\n".join(lines) if len(lines) > 1 else ""
