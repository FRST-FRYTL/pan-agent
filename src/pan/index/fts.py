"""Derived SQLite FTS5 index over the wiki (spec §4.7). Deletable; ``rebuild`` recreates it.

Tables:
- ``pages``        one row per page (id, path, title, type, status, …, content hash)
- ``pages_fts``    page-level FTS rows (title, tags, claims, body) → candidate set + BM25 tie-break;
                   ``claims`` = the page's list items (the curated claims) plus its ``subject``
- ``sections_fts`` one row per heading section → snippets that point at a heading
- ``meta``         schema version, page count and a hash of the indexed wiki state

Contents depend only on the wiki (no timestamps), so the same wiki yields an identical index.
Readers open the file read-only and never write or rebuild.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pan.wiki.frontmatter import Page
from pan.wiki.markdown import split_sections

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2"
TOKENIZER = "porter unicode61 remove_diacritics 2"
MAX_QUERY_TERMS = 24
# BM25 column weights: pages_fts(id, title, tags, claims, body), sections_fts(page_id, ord, anchor, heading, body)
PAGE_WEIGHTS = (0.0, 8.0, 4.0, 2.0, 1.0)
SECTION_WEIGHTS = (0.0, 0.0, 0.0, 4.0, 1.0)
# Relevance score (M5, see ``FtsIndex._search``): idf-weighted query-term coverage. A term found only
# in the body (incl. claims) counts BODY_ONLY_WEIGHT of a term in the title or tags; a query term the wiki
# does not contain at all counts UNKNOWN_TERM_WEIGHT of its (maximal) idf in the denominator.
BODY_ONLY_WEIGHT = 0.7
UNKNOWN_TERM_WEIGHT = 0.2   # recall: chat queries mix topics ("From now on … Also: why do tool calls fail?")
# Curation asks "does this page already cover the candidate?": new words must weigh more, else a
# decision about the bench's results index "updates" the ADR about FTS retrieval.
CURATION_UNKNOWN_TERM_WEIGHT = 0.5
STRONG_COLUMNS = ("title", "tags")  # claims: BM25 weight only (a list item is still body text)
SNIPPET_TOKENS = 24
# Snippets from curated claims (M6): the best-matching current list items of the page, at most
# SNIPPET_MAX_ITEMS and SNIPPET_MAX_CHARS. In early benchmark runs the FTS window snippet showed the
# page's "User asked: …" context line or an old value (two benchmark scenarios) instead of the fact.
SNIPPET_MAX_ITEMS = 2
SNIPPET_MAX_CHARS = 420  # two log-line or clause facts must fit (was 240)
# "(observed)" / "(inferred)" labels are for readers of the page; in a snippet Qwen3.8 took the
# labelled bullet for an L1 entry and tried `memory replace` with it (M6 check run).
_EVIDENCE_RE = re.compile(r"\s*\((?:observed|inferred|reported|mixed)(?:;[^)]*)?\)(?=\s*$)")
_CLAIM_SECTIONS = frozenset({"facts", "updates", "summary", "preference", "status", "setting"})
_NON_CLAIM_SECTIONS = frozenset({"related", "sources", "see also", "links", "references", "provenance"})
# The curator's "User asked: …" line records what prompted a page; it is context, never a fact — a
# snippet that showed it (a file path from the question) hid the page's claims.
_CONTEXT_LINE_RE = re.compile(r'^\s*User asked:\s*"')
_PROVENANCE_LINE_RE = re.compile(r"^\s*_Provenance:.*_\s*$")
_PLACEHOLDER_RE = re.compile(r"^\s*_Not recorded(?: yet)?\._\s*$")
# "(stated 2026-09-30)" is the day the claim was noted, not when it happened; shown as "stated" the agent
# read it as content ("since 30 September") and it is wrong for tool-observed claims.
_STATED_RE = re.compile(r"\(stated (\d{4}-\d{2}-\d{2})\)")


def is_context_line(text: str) -> bool:
    return bool(_CONTEXT_LINE_RE.match(text or ""))


def present_claim(text: str) -> str:
    """A stored claim as the agent sees it: the ``(stated <date>)`` stamp becomes ``(noted <date>)``
    (a resolved date inside the claim, "yesterday (2026-09-29)", stays)."""
    return _STATED_RE.sub(r"(noted \1)", text)


_SOURCE_ID_RE = re.compile(r"\s*[·,;]?\s*\b(?:session|event):[\w.:-]+")


def present_heading(heading: str) -> str:
    """A section heading without provenance ids ("2026-09-30 · session:…" → "2026-09-30")."""
    return _SOURCE_ID_RE.sub("", heading or "").strip(" ·,;")

_TERM_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset("""
a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my no not of on or our should so than that the their them then there these they this to us was
we were what when where which who why will with would you your please pls thanks thank hi hello
about above after again all also am any because been before being both each few just more most
other over same some such too very here now only own up out off tell show give know like want get
let lets one
answer answers reply respond sentence briefly short remember recall without exactly currently
""".split())

_SCHEMA = f"""
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pages (
    id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, title TEXT NOT NULL, type TEXT NOT NULL,
    status TEXT NOT NULL, confidence TEXT NOT NULL, updated TEXT NOT NULL, tags TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE VIRTUAL TABLE pages_fts USING fts5(id UNINDEXED, title, tags, claims, body, tokenize='{TOKENIZER}');
CREATE VIRTUAL TABLE sections_fts USING fts5(
    page_id UNINDEXED, ord UNINDEXED, anchor UNINDEXED, heading, body, tokenize='{TOKENIZER}'
);
"""


@dataclass(frozen=True)
class Hit:
    id: str
    path: str
    title: str
    type: str
    status: str
    score: float              # normalized relevance in [0, 1]
    bm25: float               # raw FTS5 bm25 (lower = better)
    section: str = ""         # heading of the best-matching section ("" = page preamble)
    anchor: str = ""
    snippet: str = ""
    confidence: str = ""
    updated: str = ""
    via: str = "fts"          # which stage produced ``score``: fts | dense | rerank (ADR-012)
    prior: float = 0.0        # for ``via="rerank"``: the fused FTS/dense score before reranking

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def page_hash(page: Page) -> str:
    return hashlib.sha256(page.to_text().encode("utf-8")).hexdigest()


_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_STRUCK_RE = re.compile(r"~~.*?~~")


def live_body(text: str) -> str:
    """``text`` with superseded (``~~struck~~``) claims replaced by a marker: the index — and so
    snippets and prefetch — only carries current facts; the page keeps the history (M5)."""
    return _STRUCK_RE.sub("[superseded]", text)


def page_claims(page: Page) -> str:
    """Text of the page's list items (curated claims; superseded ``~~…~~`` spans dropped) plus its
    ``subject`` — indexed in their own column so a fact matches as strongly as a title word."""
    items = []
    for line in page.body.splitlines():
        m = _ITEM_RE.match(line)
        if m:
            text = _STRUCK_RE.sub("", m.group(1)).strip()
            if text:
                items.append(text)
    subject = str(page.meta.get("subject") or "").strip()
    return "\n".join(([subject] if subject else []) + items)


def query_terms(query: str) -> List[str]:
    """Lower-cased word tokens; stopwords dropped (unless nothing else is left); deduplicated.

    Only word characters survive, so FTS5 operators, quotes and parentheses in user input can never
    reach the MATCH expression."""
    words = [w.lower() for w in _TERM_RE.findall(query or "")]
    words = [w[:64] for w in words if w.strip("_")]
    # 1–2 digit numbers ("2 + 2", "3 sentences") carry no topic; ports, versions and years do.
    content = [w for w in words if w not in STOPWORDS and not (w.isdigit() and len(w) <= 2)]
    out: List[str] = []
    for w in content or words:
        if w not in out:
            out.append(w)
    return out[:MAX_QUERY_TERMS]


def match_expression(terms: Sequence[str]) -> str:
    """OR of quoted terms (quoted = literal, so words like AND/NEAR are plain tokens)."""
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms)


class FtsIndex:
    """FTS5 index at ``db_path``. ``readonly=True`` for the agent-side reader (never writes)."""

    def __init__(self, db_path: str | Path, *, readonly: bool = False) -> None:
        self.db_path = Path(db_path)
        self.readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None
        self._stat: Optional[tuple[int, int]] = None

    # -- connections ---------------------------------------------------------------------------

    def _file_id(self) -> Optional[tuple[int, int]]:
        try:
            st = os.stat(self.db_path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino)

    def _connect(self) -> Optional[sqlite3.Connection]:
        """Current connection; reopened when ``rebuild`` replaced the file. None if unusable."""
        fid = self._file_id()
        if self._conn is not None and fid == self._stat:
            return self._conn
        self.close()
        if fid is None:
            if self.readonly:
                return None
            self._create(self.db_path)
            fid = self._file_id()
        try:
            if self.readonly:
                conn = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True,
                                       timeout=2.0, check_same_thread=False)
                conn.execute("PRAGMA query_only = 1")
            else:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
            version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        except sqlite3.Error as exc:
            logger.warning("PAN index %s unusable (%s); run `pan index rebuild`", self.db_path, exc)
            return None
        if not version or version[0] != SCHEMA_VERSION:
            logger.warning("PAN index %s has schema %s (want %s); run `pan index rebuild`",
                           self.db_path, version and version[0], SCHEMA_VERSION)
            conn.close()
            return None
        self._conn, self._stat = conn, fid
        return conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn, self._stat = None, None

    def available(self) -> bool:
        return self._connect() is not None

    @staticmethod
    def _create(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.executescript(_SCHEMA)
            conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)",
                             [("schema_version", SCHEMA_VERSION), ("tokenizer", TOKENIZER),
                              ("page_count", "0"), ("wiki_hash", hashlib.sha256(b"").hexdigest())])
            conn.commit()
        finally:
            conn.close()

    def _writer(self) -> sqlite3.Connection:
        if self.readonly:
            raise PermissionError("index opened read-only")
        conn = self._connect()
        if conn is None:
            raise RuntimeError(f"index {self.db_path} unusable; rebuild it")
        return conn

    # -- writes --------------------------------------------------------------------------------

    @staticmethod
    def _insert(conn: sqlite3.Connection, page: Page) -> None:
        assert page.path is not None, "page must have a wiki-relative path"
        tags = " ".join(page.tags)
        conn.execute(
            "INSERT INTO pages(id, path, title, type, status, confidence, updated, tags, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (page.id, page.path, page.title, page.type, page.status,
             str(page.meta.get("confidence", "")), str(page.meta.get("updated", "")), tags,
             page_hash(page)))
        conn.execute("INSERT INTO pages_fts(id, title, tags, claims, body) VALUES (?, ?, ?, ?, ?)",
                     (page.id, page.title, tags, page_claims(page), live_body(page.body).strip()))
        for ord_, sec in enumerate(split_sections(page.body)):
            conn.execute("INSERT INTO sections_fts(page_id, ord, anchor, heading, body) "
                         "VALUES (?, ?, ?, ?, ?)", (page.id, ord_, sec.anchor, sec.heading, live_body(sec.text)))

    @staticmethod
    def _delete(conn: sqlite3.Connection, page_id: str) -> int:
        n = conn.execute("DELETE FROM pages WHERE id = ?", (page_id,)).rowcount
        conn.execute("DELETE FROM pages_fts WHERE id = ?", (page_id,))
        conn.execute("DELETE FROM sections_fts WHERE page_id = ?", (page_id,))
        return n

    @staticmethod
    def _refresh_meta(conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT path, hash FROM pages ORDER BY path").fetchall()
        digest = hashlib.sha256("".join(f"{p}\0{h}\n" for p, h in rows).encode()).hexdigest()
        conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                         [("page_count", str(len(rows))), ("wiki_hash", digest)])

    def rebuild(self, wiki_root: str | Path) -> int:
        """Recreate the index from the wiki (atomic file replace). Returns the page count."""
        from pan.wiki.store import WikiStore

        if self.readonly:
            raise PermissionError("index opened read-only")
        pages = WikiStore(wiki_root).list_pages()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".index.", suffix=".db", dir=self.db_path.parent)
        os.close(fd)
        os.unlink(tmp)
        try:
            self._create(Path(tmp))
            conn = sqlite3.connect(tmp)
            try:
                seen: set[str] = set()
                for page in pages:  # sorted by path → deterministic
                    if not page.id or page.id in seen:
                        logger.warning("index: skipping %s (missing or duplicate id %r)", page.path, page.id)
                        continue
                    seen.add(page.id)
                    self._insert(conn, page)
                self._refresh_meta(conn)
                conn.commit()
                conn.execute("VACUUM")
            finally:
                conn.close()
            self.close()
            os.replace(tmp, self.db_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return len(seen)

    def update(self, pages: Iterable[Page]) -> int:
        """Insert or replace pages (matched by id, or by path if the id changed)."""
        conn = self._writer()
        n = 0
        with conn:
            for page in pages:
                for (old_id,) in conn.execute("SELECT id FROM pages WHERE path = ? OR id = ?",
                                              (page.path, page.id)).fetchall():
                    self._delete(conn, old_id)
                self._insert(conn, page)
                n += 1
            self._refresh_meta(conn)
        return n

    def remove(self, ids: Iterable[str]) -> int:
        conn = self._writer()
        with conn:
            n = sum(self._delete(conn, i) for i in ids)
            self._refresh_meta(conn)
        return n

    def remove_paths(self, paths: Iterable[str]) -> int:
        """Remove pages by wiki-relative path (for deleted files whose id is gone)."""
        conn = self._writer()
        with conn:
            n = 0
            for p in paths:
                for (old_id,) in conn.execute("SELECT id FROM pages WHERE path = ?", (p,)).fetchall():
                    n += self._delete(conn, old_id)
            self._refresh_meta(conn)
        return n

    # -- reads ---------------------------------------------------------------------------------

    def lookup(self, page_id: str) -> Optional[str]:
        """Wiki-relative path for ``page_id`` (None if unknown or index unavailable)."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute("SELECT path FROM pages WHERE id = ?", (page_id,)).fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row else None

    def search(self, query: str, type: Optional[str] = None, limit: int = 5, *,
               unknown_weight: float = UNKNOWN_TERM_WEIGHT) -> List[Hit]:
        """BM25 search. Never raises on odd input; returns [] when nothing matches.

        ``type=None``/"any" searches knowledge pages (``index`` pages only when asked for)."""
        terms = query_terms(query)
        if not terms or limit <= 0:
            return []
        if type in ("", "any"):
            type = None
        conn = self._connect()
        if conn is None:
            return []
        try:
            return self._search(conn, terms, type, int(limit), unknown_weight)
        except sqlite3.Error as exc:
            logger.warning("PAN index search failed for %r: %s", query, exc)
            return []

    def _search(self, conn: sqlite3.Connection, terms: List[str], type: Optional[str],
                limit: int, unknown_weight: float = UNKNOWN_TERM_WEIGHT) -> List[Hit]:
        """Score = idf-weighted coverage of the query terms (in [0, 1]); BM25 breaks ties.

        Raw BM25 is unusable as an absolute score for small wikis: FTS5 clamps a term's idf to ~0
        when it occurs in half of the pages or more, so with one or two pages every hit scored 0.00
        (e.g. "which GPU does this machine have?"). The score here uses a smoothed idf
        ``ln(1 + (N - df + 0.5) / (df + 0.5))`` (always > 0), counts a term fully when it occurs in
        the title/tags and ``BODY_ONLY_WEIGHT`` when only in the body, and divides by the idf
        mass of *all* query terms (unknown terms at ``unknown_weight``) — so a hit on one
        generic word of a long query stays low, and a page matching the distinctive words scores
        high regardless of wiki size."""
        expr = match_expression(terms)
        weights = ", ".join(str(w) for w in PAGE_WEIGHTS)
        sql = (f"SELECT p.id, p.path, p.title, p.type, p.status, p.confidence, p.updated, "
               f"bm25(pages_fts, {weights}) AS r FROM pages_fts JOIN pages p ON p.id = pages_fts.id "
               f"WHERE pages_fts MATCH ?" + (" AND p.type = ?" if type else " AND p.type != 'index'") +
               " ORDER BY r, p.path LIMIT ?")
        params: list[Any] = [expr] + ([type] if type else []) + [max(limit * 5, 50)]
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []
        candidates = {r[0] for r in rows}
        n_pages = max(1, conn.execute("SELECT count(*) FROM pages WHERE type != 'index'").fetchone()[0])
        total = 0.0
        gained: Dict[str, float] = {c: 0.0 for c in candidates}
        for term in terms:
            q = match_expression([term])
            ids = {r[0] for r in conn.execute(
                "SELECT pages_fts.id FROM pages_fts JOIN pages p ON p.id = pages_fts.id "
                "WHERE pages_fts MATCH ? AND p.type != 'index'", (q,))}
            df = len(ids)
            idf = math.log(1.0 + (n_pages - df + 0.5) / (df + 0.5))
            if not df:
                total += unknown_weight * idf
                continue
            total += idf
            hit = ids & candidates
            if not hit:
                continue
            strong_expr = " OR ".join(f"{col} : {q}" for col in STRONG_COLUMNS)
            strong = {r[0] for r in conn.execute("SELECT id FROM pages_fts WHERE pages_fts MATCH ?",
                                                 (strong_expr,))}
            for pid in hit:
                gained[pid] += idf * (1.0 if pid in strong else BODY_ONLY_WEIGHT)
        scored = []
        for pid, path, title, ptype, status, conf, updated, r in rows:
            score = round(min(1.0, gained[pid] / total), 4) if total > 0 else 0.0
            scored.append((score, float(r), path, pid, title, ptype, status, conf, updated))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        hits = []
        for score, r, path, pid, title, ptype, status, conf, updated in scored[:limit]:
            best = self._claim_snippet(conn, terms, pid)
            section, anchor, snippet = best if best is not None else self._best_section(conn, expr, pid)
            hits.append(Hit(id=pid, path=path, title=title, type=ptype, status=status,
                            score=score, bm25=round(r, 6), section=section, anchor=anchor,
                            snippet=snippet, confidence=conf, updated=updated))
        return hits

    @staticmethod
    def _claim_snippet(conn: sqlite3.Connection, terms: Sequence[str], page_id: str) -> Optional[tuple[str, str, str]]:
        """(heading, anchor, snippet) from the page's current list items (see SNIPPET_MAX_ITEMS), or
        None when the page has none. Items are ranked by matched query stems, then observed over
        inferred, then later (newer: ``## Updates`` follows ``## Facts``) first; superseded items
        (``[superseded]`` in the index) are never shown. When no item matches a query term (the page
        matched on its title or tags), its first current item of a claim section (``## Facts``,
        ``## Summary``, ``## Updates`` …) is the snippet — or, on a page without prose, its first item:
        the "User asked" context line is never a snippet. Link lists (``## Related`` …) are skipped."""
        from pan.memory.claims import stem

        want = {stem(t) for t in terms}
        items = []
        prose = False
        rows = conn.execute("SELECT heading, anchor, body FROM sections_fts WHERE page_id = ? "
                            "ORDER BY CAST(ord AS INTEGER)", (page_id,)).fetchall()
        for heading, anchor, body in rows:
            if heading.strip().lower() in _NON_CLAIM_SECTIONS:
                continue
            for line in body.splitlines():
                m = _ITEM_RE.match(line)
                if not m:
                    prose = prose or bool(line.strip() and not is_context_line(line)
                                          and not _PROVENANCE_LINE_RE.match(line) and not _PLACEHOLDER_RE.match(line))
                    continue
                raw = " ".join(m.group(1).split())
                label = _EVIDENCE_RE.search(raw)
                observed = label is None or "(observed" in label.group(0)
                text = _EVIDENCE_RE.sub("", raw).strip()
                if not text or "[superseded]" in text:
                    continue
                have = {stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())}
                items.append((len(want & have), observed, len(items), heading, anchor, present_claim(text)))
        if not items:
            return None
        ranked = sorted(items, key=lambda i: (-i[0], not i[1], -i[2]))
        if ranked[0][0] == 0:
            facts = [i for i in items if i[3].strip().lower() in _CLAIM_SECTIONS] or ([] if prose else items)
            if not facts:
                return None  # prose page matched on title/tags: the FTS window snippet
            ranked = [facts[0]]
        chosen = [ranked[0]]
        for item in ranked[1:SNIPPET_MAX_ITEMS]:
            if item[0] > 0 and len(" · ".join(c[5] for c in chosen + [item])) <= SNIPPET_MAX_CHARS:
                chosen.append(item)
        snippet = " · ".join(c[5] for c in chosen)
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[:SNIPPET_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
        return chosen[0][3], chosen[0][4], snippet

    @staticmethod
    def _best_section(conn: sqlite3.Connection, expr: str, page_id: str) -> tuple[str, str, str]:
        weights = ", ".join(str(w) for w in SECTION_WEIGHTS)
        # a section that is only the "User asked" context line is never the snippet
        not_context = "NOT (body LIKE 'User asked: \"%' AND instr(body, char(10)) = 0)"
        row = conn.execute(
            f"SELECT heading, anchor, snippet(sections_fts, 4, '', '', '…', {SNIPPET_TOKENS}), body, "
            f"CAST(ord AS INTEGER) "
            f"FROM sections_fts WHERE sections_fts MATCH ? AND page_id = ? AND {not_context} "
            f"ORDER BY bm25(sections_fts, {weights}), CAST(ord AS INTEGER) LIMIT 1",
            (expr, page_id)).fetchone()
        if row is None:  # matched only title/tags: first non-empty section
            row = conn.execute("SELECT heading, anchor, '', body, CAST(ord AS INTEGER) FROM sections_fts "
                               f"WHERE page_id = ? AND {not_context} ORDER BY CAST(ord AS INTEGER) LIMIT 1",
                               (page_id,)).fetchone()
            if row is None:
                return "", "", ""
        heading, anchor, snippet, body, ord_ = row
        if not snippet.strip():
            if not body.strip():  # heading-only section: take the text that follows it
                nxt = conn.execute("SELECT body FROM sections_fts WHERE page_id = ? AND "
                                   "CAST(ord AS INTEGER) > ? AND body != '' ORDER BY CAST(ord AS INTEGER) "
                                   "LIMIT 1", (page_id, ord_)).fetchone()
                body = nxt[0] if nxt else ""
            snippet = body[:200].rstrip() + ("…" if len(body) > 200 else "")
        return heading, anchor, present_claim(" ".join(snippet.split()))

    def status(self, wiki_root: str | Path | None = None) -> Dict[str, Any]:
        """Index state; with ``wiki_root`` also which pages are missing/changed/extra (staleness)."""
        info: Dict[str, Any] = {"path": str(self.db_path), "exists": self.db_path.exists()}
        conn = self._connect()
        info["usable"] = conn is not None
        if conn is None:
            return info
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        info.update({
            "schema_version": meta.get("schema_version"),
            "tokenizer": meta.get("tokenizer"),
            "pages": int(meta.get("page_count", 0)),
            "sections": conn.execute("SELECT count(*) FROM sections_fts").fetchone()[0],
            "wiki_hash": meta.get("wiki_hash"),
            "size_bytes": self.db_path.stat().st_size,
        })
        if wiki_root is not None:
            from pan.wiki.store import WikiStore

            indexed = dict(conn.execute("SELECT path, hash FROM pages").fetchall())
            current = {p.path: page_hash(p) for p in WikiStore(wiki_root).list_pages()}
            info["missing"] = sorted(set(current) - set(indexed))
            info["extra"] = sorted(set(indexed) - set(current))
            info["changed"] = sorted(p for p in set(current) & set(indexed) if current[p] != indexed[p])
            info["stale"] = bool(info["missing"] or info["extra"] or info["changed"])
        return info

    def dump(self) -> List[tuple]:
        """All index rows in a canonical order (used to check rebuild determinism)."""
        conn = self._connect()
        if conn is None:
            return []
        out: List[tuple] = []
        for table, order in (("meta", "key"), ("pages", "id"), ("pages_fts", "id"),
                             ("sections_fts", "page_id, CAST(ord AS INTEGER)")):
            out.extend((table, *row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"))
        return out


def rebuild(db_path: str | Path, wiki_root: str | Path) -> int:
    idx = FtsIndex(db_path)
    try:
        return idx.rebuild(wiki_root)
    finally:
        idx.close()
