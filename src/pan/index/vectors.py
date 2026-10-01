"""Derived dense-vector index over the wiki (spec §12 hybrid retrieval, ADR-012). Deletable; rebuilt
from the wiki like the FTS index (``pan index rebuild``). The wiki stays the source of truth.

A page is embedded as several *units*:
- ``card``    title + subject + tags (what the page is about);
- ``claim``   every current list item (curated claims; ``~~superseded~~`` items are skipped), prefixed
              with the page title for context;
- ``text``    prose outside list items, in chunks of ≈ ``CHUNK_CHARS``.
A page's dense similarity is its best unit; the best ``claim`` units become the snippet, so a German
question over an English page still gets the matching fact as its snippet.

Storage: SQLite ``vectors.db`` next to ``index.db`` (``meta``, ``pages``, ``units`` with float32
vectors). Vectors are unit-normalised by the sidecar, so cosine = dot product. Search is brute force
in pure Python (the agent venv has no numpy): ≈ 25 µs per 1024-d unit — fine for a personal wiki of a
few thousand units; beyond that move the scoring into the sidecar (ADR-012, open question).

Writers (``pan-memoryd``, the CLI) call :meth:`VectorIndex.sync` with the current pages; it embeds
only units whose text it has not embedded before (by text hash) and drops pages that are gone. The
index records the embedding model id, revision and dimension; a different model means a full
re-embed, and a reader whose sidecar serves a different model ignores the index (no mixed spaces).
"""

from __future__ import annotations

import hashlib
import logging
import operator
import os
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pan.index.fts import _EVIDENCE_RE, _ITEM_RE, _NON_CLAIM_SECTIONS, _STRUCK_RE, page_hash
from pan.wiki.frontmatter import Page
from pan.wiki.markdown import split_sections

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
CHUNK_CHARS = 700
EMBED_BATCH = 64
_PROVENANCE_RE = re.compile(r"^\s*_Provenance:.*_\s*$")

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pages (
    id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, title TEXT NOT NULL, type TEXT NOT NULL,
    status TEXT NOT NULL, confidence TEXT NOT NULL, updated TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TABLE units (
    page_id TEXT NOT NULL, ord INTEGER NOT NULL, kind TEXT NOT NULL, heading TEXT NOT NULL,
    anchor TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (page_id, ord)
);
CREATE INDEX units_hash ON units(text_hash);
"""


@dataclass(frozen=True)
class Unit:
    kind: str       # card | claim | text
    heading: str
    anchor: str
    text: str       # display text (snippet)
    embed_text: str  # what is embedded

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()


def _chunks(text: str, size: int = CHUNK_CHARS) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + 1 + len(sentence) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
        while len(cur) > size:  # one very long sentence
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return out


def page_units(page: Page) -> List[Unit]:
    """The embedding units of ``page`` (deterministic order)."""
    title = page.title.strip()
    subject = str(page.meta.get("subject") or "").strip()
    card = title + (f" — {subject}" if subject and subject.lower() != title.lower() else "")
    if page.tags:
        card += " (" + ", ".join(page.tags) + ")"
    units = [Unit("card", "", "", card, card)]
    for sec in split_sections(page.body):
        if sec.heading.strip().lower() in _NON_CLAIM_SECTIONS:
            continue
        prose: List[str] = []
        for line in sec.text.splitlines():
            m = _ITEM_RE.match(line)
            if m:
                item = _EVIDENCE_RE.sub("", " ".join(_STRUCK_RE.sub("", m.group(1)).split())).strip()
                if item and len(item) > 2:
                    units.append(Unit("claim", sec.heading, sec.anchor, item, f"{title}: {item}"))
            elif line.strip() and not _PROVENANCE_RE.match(line) and not line.lstrip().startswith("#"):
                prose.append(_STRUCK_RE.sub("", line))
        for chunk in _chunks(" ".join(prose)):
            if len(chunk) >= 20:
                where = f"{title} — {sec.heading}" if sec.heading and sec.heading != title else title
                units.append(Unit("text", sec.heading, sec.anchor, chunk, f"{where}: {chunk}"))
    return units


def _pack(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


@dataclass(frozen=True)
class PageMatch:
    id: str
    path: str
    title: str
    type: str
    status: str
    confidence: str
    updated: str
    sim: float                               # best unit similarity (cosine)
    units: Tuple[Tuple[float, str, str, str, str], ...]  # (sim, kind, heading, anchor, text), best first


class VectorIndex:
    """Vector index at ``db_path``. ``readonly=True`` for the agent-side reader (never writes)."""

    def __init__(self, db_path: str | Path, *, readonly: bool = False) -> None:
        self.db_path = Path(db_path)
        self.readonly = readonly
        self._cache_key: Optional[tuple] = None
        self._meta: Dict[str, str] = {}
        self._pages: Dict[str, tuple] = {}
        self._units: List[tuple] = []   # (page_id, kind, heading, anchor, text, vec)

    # -- file ----------------------------------------------------------------------------------

    def _key(self) -> Optional[tuple]:
        try:
            st = os.stat(self.db_path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)

    def _connect(self) -> sqlite3.Connection:
        if self.readonly:
            conn = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2.0,
                                   check_same_thread=False)
            conn.execute("PRAGMA query_only = 1")
            return conn
        new = not self.db_path.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        if new:
            conn.executescript(_SCHEMA)
            conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            conn.commit()
        return conn

    def _load(self) -> bool:
        """(Re)load vectors into memory when the file changed. False when unusable."""
        key = self._key()
        if key is None:
            self._cache_key, self._units, self._pages, self._meta = None, [], {}, {}
            return False
        if key == self._cache_key:
            return bool(self._meta)
        try:
            conn = self._connect()
            try:
                meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
                if meta.get("schema_version") != SCHEMA_VERSION:
                    logger.warning("PAN vectors %s: schema %s (want %s); run `pan index rebuild`",
                                   self.db_path, meta.get("schema_version"), SCHEMA_VERSION)
                    meta = {}
                pages = {r[0]: r for r in conn.execute(
                    "SELECT id, path, title, type, status, confidence, updated FROM pages")}
                units = [(pid, kind, heading, anchor, text, _unpack(vec)) for pid, kind, heading, anchor, text, vec
                         in conn.execute("SELECT page_id, kind, heading, anchor, text, vec FROM units "
                                         "ORDER BY page_id, ord")]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("PAN vectors %s unusable (%s); run `pan index rebuild`", self.db_path, exc)
            return False
        self._cache_key, self._meta, self._pages, self._units = key, meta, pages, units
        return bool(meta)

    def model(self) -> Optional[Dict[str, str]]:
        """{model_id, revision, dim} the vectors were made with (None if empty/unusable)."""
        if not self._load() or not self._meta.get("model_id"):
            return None
        return {k: self._meta.get(k, "") for k in ("model_id", "revision", "dim")}

    def close(self) -> None:
        self._cache_key, self._units, self._pages, self._meta = None, [], {}, {}

    # -- writes --------------------------------------------------------------------------------

    def sync(self, pages: Iterable[Page], client, *, model: Optional[Dict[str, object]] = None,
             remove_missing: bool = True) -> Dict[str, int]:
        """Make the index match ``pages`` (all current wiki pages): embed new/changed pages, drop
        missing ones (``remove_missing=False``: only upsert ``pages``). Raises :class:`pan.index.embed.EmbedError` when the sidecar is unavailable
        (the index is then left as it was)."""
        from pan.index.embed import EmbedError

        if self.readonly:
            raise PermissionError("vector index opened read-only")
        model = model or client.embed_model_id()
        if not model:
            raise EmbedError("no embedding model available")
        want = {"model_id": str(model.get("model_id", "")), "revision": str(model.get("revision", "")),
                "dim": str(client.dimensions or model.get("dim", ""))}
        pages = [p for p in pages if p.id and p.type != "index"]
        stats = {"pages": len(pages), "embedded_pages": 0, "embedded_units": 0, "reused_units": 0, "removed": 0}
        conn = self._connect()
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if any(meta.get(k, "") != v for k, v in want.items()):
                if meta.get("model_id"):
                    logger.info("PAN vectors: model changed (%s → %s); re-embedding everything",
                                meta.get("model_id"), want["model_id"])
                with conn:
                    conn.execute("DELETE FROM units")
                    conn.execute("DELETE FROM pages")
            have = dict(conn.execute("SELECT path, hash FROM pages").fetchall())
            ids = {p.id for p in pages}
            paths = {p.path for p in pages}
            todo, seen = [], set()
            for page in pages:
                if page.id in seen:
                    continue
                seen.add(page.id)
                if have.get(page.path) != page_hash(page):
                    todo.append(page)
            gone = [pid for (pid, path) in conn.execute("SELECT id, path FROM pages")
                    if pid not in ids or path not in paths] if remove_missing else []
            # embed outside the write transaction; reuse vectors of identical texts
            plan: List[Tuple[Page, List[Unit]]] = [(p, page_units(p)) for p in todo]
            known: Dict[str, bytes] = {}
            hashes = {u.text_hash for _, units in plan for u in units}
            for h in hashes:
                row = conn.execute("SELECT vec FROM units WHERE text_hash = ? LIMIT 1", (h,)).fetchone()
                if row is not None:
                    known[h] = row[0]
            missing: Dict[str, str] = {}
            for _, units in plan:
                for u in units:
                    if u.text_hash not in known:
                        missing.setdefault(u.text_hash, u.embed_text)
            items = list(missing.items())
            for i in range(0, len(items), EMBED_BATCH):
                batch = items[i:i + EMBED_BATCH]
                vecs = client.embed([t for _, t in batch], "document")
                for (h, _), v in zip(batch, vecs):
                    known[h] = _pack(v)
            stats["embedded_units"] = len(items)
            stats["reused_units"] = sum(len(u) for _, u in plan) - len(items)
            with conn:
                for pid in gone:
                    conn.execute("DELETE FROM units WHERE page_id = ?", (pid,))
                    conn.execute("DELETE FROM pages WHERE id = ?", (pid,))
                for page, units in plan:
                    for (old,) in conn.execute("SELECT id FROM pages WHERE path = ? OR id = ?",
                                               (page.path, page.id)).fetchall():
                        conn.execute("DELETE FROM units WHERE page_id = ?", (old,))
                        conn.execute("DELETE FROM pages WHERE id = ?", (old,))
                    conn.execute("INSERT INTO pages(id, path, title, type, status, confidence, updated, hash) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (page.id, page.path, page.title, page.type, page.status,
                                  str(page.meta.get("confidence", "")), str(page.meta.get("updated", "")),
                                  page_hash(page)))
                    conn.executemany(
                        "INSERT INTO units(page_id, ord, kind, heading, anchor, text, text_hash, vec) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [(page.id, n, u.kind, u.heading, u.anchor, u.text, u.text_hash, known[u.text_hash])
                         for n, u in enumerate(units)])
                conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", list(want.items()))
            stats["embedded_pages"] = len(plan)
            stats["removed"] = len(gone)
        finally:
            conn.close()
        return stats

    def rebuild(self, wiki_root: str | Path, client) -> Dict[str, int]:
        """Re-embed the whole wiki into a fresh file (atomic replace)."""
        from pan.wiki.store import WikiStore

        if self.readonly:
            raise PermissionError("vector index opened read-only")
        tmp = self.db_path.with_name(f".{self.db_path.name}.rebuild")
        if tmp.exists():
            tmp.unlink()
        try:
            stats = VectorIndex(tmp).sync(WikiStore(wiki_root).list_pages(), client)
            os.replace(tmp, self.db_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self.close()
        return stats

    # -- reads ---------------------------------------------------------------------------------

    def search(self, qvec: Sequence[float], limit: int = 5, *, type: Optional[str] = None,
               units_per_page: int = 3) -> List[PageMatch]:
        """Pages by best unit similarity (``type=None`` excludes nothing but index pages, which
        are never embedded)."""
        if limit <= 0 or not self._load() or not self._units:
            return []
        q = array("f", qvec)
        if len(q) != len(self._units[0][5]):
            logger.warning("PAN vectors: query dim %d != index dim %d", len(q), len(self._units[0][5]))
            return []
        mul = operator.mul
        per_page: Dict[str, List[tuple]] = {}
        for pid, kind, heading, anchor, text, vec in self._units:
            per_page.setdefault(pid, []).append((sum(map(mul, q, vec)), kind, heading, anchor, text))
        out: List[PageMatch] = []
        for pid, units in per_page.items():
            row = self._pages.get(pid)
            if row is None or (type and row[3] != type):
                continue
            units.sort(key=lambda u: -u[0])
            out.append(PageMatch(id=pid, path=row[1], title=row[2], type=row[3], status=row[4],
                                 confidence=row[5], updated=row[6], sim=units[0][0],
                                 units=tuple(units[:max(1, units_per_page)])))
        out.sort(key=lambda m: (-m.sim, m.path))
        return out[:limit]

    def status(self, wiki_root: str | Path | None = None) -> Dict[str, object]:
        info: Dict[str, object] = {"path": str(self.db_path), "exists": self.db_path.exists()}
        if not self._load():
            info["usable"] = False
            return info
        info.update(usable=True, model_id=self._meta.get("model_id", ""),
                    revision=self._meta.get("revision", ""), dim=self._meta.get("dim", ""),
                    pages=len(self._pages), units=len(self._units),
                    size_bytes=self.db_path.stat().st_size)
        if wiki_root is not None:
            from pan.wiki.store import WikiStore

            conn = self._connect()
            try:
                indexed = dict(conn.execute("SELECT path, hash FROM pages").fetchall())
            finally:
                conn.close()
            current = {p.path: page_hash(p) for p in WikiStore(wiki_root).list_pages() if p.type != "index"}
            info["missing"] = sorted(set(current) - set(indexed))
            info["extra"] = sorted(set(indexed) - set(current))
            info["changed"] = sorted(p for p in set(current) & set(indexed) if current[p] != indexed[p])
            info["stale"] = bool(info["missing"] or info["extra"] or info["changed"])
        return info


def sync_wiki(paths, cfg, *, rebuild: bool = False, pages: Optional[Iterable[Page]] = None,
              client=None) -> Optional[Dict[str, int]]:
    """Bring ``paths.vectors_db`` up to date with the wiki using the ``RetrievalConfig`` ``cfg``.
    None when dense retrieval is off; raises ``EmbedError`` when the sidecar is unavailable."""
    from pan.index.embed import EmbedClient
    from pan.wiki.store import WikiStore

    if "hybrid" not in (cfg.mode, cfg.curator_mode):
        return None
    client = client or EmbedClient(cfg.embed_url, timeout_s=max(cfg.timeout_s, 30.0), embed_model=cfg.embed_model,
                                   dimensions=cfg.embed_dimensions)
    idx = VectorIndex(paths.vectors_db)
    if rebuild:
        return idx.rebuild(paths.wiki, client)
    if pages is None:
        return idx.sync(WikiStore(paths.wiki).list_pages(), client)
    return idx.sync(pages, client, remove_missing=False)
