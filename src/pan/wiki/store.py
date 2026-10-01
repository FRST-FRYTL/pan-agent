"""WikiStore — canonical L2 knowledge as Markdown + frontmatter in a git repo (spec §3, §4.5).

Reads are safe from any process. Writes (``write``/``delete``) are atomic per file and meant for
the single writer ``pan-memoryd``; committing is the caller's job (see ``pan.wiki.git``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence

from pan.wiki import git as _git
from pan.wiki.frontmatter import FrontmatterError, Page, validate_meta
from pan.wiki.markdown import compact_slug, links, slugify

logger = logging.getLogger(__name__)

# Skeleton directories (spec §3).
WIKI_DIRS = ("architecture", "decisions", "systems", "learnings", "incidents", "operations", "projects")

# Page type → directory for new pages.
TYPE_DIRS: Dict[str, str] = {
    "architecture": "architecture",
    "decision": "decisions",
    "system": "systems",
    "configuration": "systems",
    "learning": "learnings",
    "incident": "incidents",
    "operations": "operations",
    "procedure": "operations",
    "environment": "operations",
    "project": "projects",
    "user_preference": "preferences",  # normally L1 (USER.md); created lazily if ever used
    "bench": "bench",
}

INDEX_TEMPLATE = """\
---
id: index
type: index
status: active
created: {today}
updated: {today}
confidence: high
sources: []
related: []
tags: [index]
---

# PAN Knowledge Wiki

Curated long-term knowledge, maintained by `pan-memoryd`. Every page has YAML frontmatter
(id, type, status, confidence, sources).

| Area | Contents |
|---|---|
| architecture/ | How systems are designed |
| decisions/ | ADRs: decisions and their rationale |
| systems/ | Systems, services and their configuration |
| learnings/ | Things found out the hard way |
| incidents/ | Failures, causes and fixes |
| operations/ | Runbooks, procedures and environments |
| projects/ | Project facts and status |
"""

# Body templates per type; ``{title}`` is substituted.
TEMPLATES: Dict[str, str] = {
    "decision": "# {title}\n\n## Context\n\n## Decision\n\n## Consequences\n",
    "learning": "# {title}\n\n## Observation\n\n## Fix\n\n## Updates\n",
    "incident": "# {title}\n\n## Symptoms\n\n## Cause\n\n## Resolution\n\n## Updates\n",
    "system": "# {title}\n\n## Overview\n\n## Configuration\n\n## Updates\n",
    "configuration": "# {title}\n\n## Setting\n\n## Reason\n\n## Updates\n",
    "environment": "# {title}\n\n## Facts\n\n## Updates\n",
    "operations": "# {title}\n\n## Procedure\n\n## Updates\n",
    "procedure": "# {title}\n\n## Steps\n\n## Updates\n",
    "architecture": "# {title}\n\n## Overview\n\n## Components\n\n## Updates\n",
    "project": "# {title}\n\n## Summary\n\n## Status\n\n## Updates\n",
    "user_preference": "# {title}\n\n## Preference\n\n## Updates\n",
}
_DEFAULT_TEMPLATE = "# {title}\n\n## Updates\n"
# Decision slugs longer than this (a whole sentence as title) are compacted to content words (M6).
COMPACT_SLUG_OVER = 32
_ADR_RE = re.compile(r"^ADR-(\d{3,})-", re.I)
_SKIP_LINK = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.I)  # URLs, mailto:, anchors, absolute


INDEX_PAGES_HEADING = "## Pages"
INDEX_PAGES_INTRO = "_Maintained by `pan-memoryd`: every page it creates is linked here, grouped by area._"


def _link_text(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("[", "(").replace("]", ")")).strip()


def link_in_index(index_text: str, page_path: str, title: str, *, today: Optional[str] = None) -> Optional[str]:
    """``index.md`` text with a link to ``page_path`` under ``## Pages`` / ``### <area>/`` (sorted by
    title; section and area heading created when missing). None when the page is already linked
    anywhere in the index (hand-maintained tables count) — nothing to do. ``today`` updates the
    frontmatter's ``updated`` date."""
    if f"]({page_path})" in index_text or f"](./{page_path})" in index_text:
        return None
    try:
        page = Page.from_text(index_text, path="index.md")
    except FrontmatterError:
        return None
    area = page_path.split("/", 1)[0] + "/" if "/" in page_path else "./"
    entry = f"- [{_link_text(title)}]({page_path})"
    lines = page.body.rstrip("\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip() == INDEX_PAGES_HEADING), None)
    if start is None:
        lines += ["", INDEX_PAGES_HEADING, "", INDEX_PAGES_INTRO]
        start = len(lines) - 3
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^#{1,2}\s", lines[i])), len(lines))
    area_heading = f"### {area}"
    areas = [(i, lines[i].strip()) for i in range(start + 1, end) if lines[i].startswith("### ")]
    here = next((i for i, h in areas if h == area_heading), None)
    if here is None:
        after = next((i for i, h in areas if h > area_heading), None)
        at = after if after is not None else end
        block = [area_heading, "", entry, ""]
        while at > start + 1 and not lines[at - 1].strip() and at == end:
            at -= 1
        if at == end or after is None:
            block = ([""] if lines[at - 1].strip() else []) + block
        lines[at:at] = block
    else:
        a_end = next((i for i in range(here + 1, end + 1) if i == end or lines[i].startswith("### ")), end)
        items = [i for i in range(here + 1, a_end) if lines[i].startswith("- ")]
        titles = [(lines[i][3:].split("](", 1)[0].casefold(), i) for i in items]
        pos = next((i for t, i in titles if t > _link_text(title).casefold()), None)
        if pos is None:
            pos = (items[-1] + 1) if items else here + 2
            if not items:
                lines[here + 1:here + 1] = [""]
        lines[pos:pos] = [entry]
    body = "\n".join(lines).rstrip("\n") + "\n"
    body = re.sub(r"\n{3,}", "\n\n", body)
    meta = dict(page.meta)
    if today and str(meta.get("created", "")) <= today:
        meta["updated"] = today
    return Page(meta=meta, body=body, path="index.md").to_text()


class WikiError(ValueError):
    pass


@dataclass(frozen=True)
class Issue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def today() -> str:
    return _dt.date.today().isoformat()


def new_page(page_type: str, title: str, *, page_id: Optional[str] = None, path: Optional[str] = None,
             sources: Sequence[str] = (), related: Sequence[str] = (), tags: Sequence[str] = (),
             status: str = "draft", confidence: str = "medium", body: Optional[str] = None,
             date: Optional[str] = None) -> Page:
    """A new page from the type template. Default path ``<type dir>/<slug>.md``, id ``<dir>.<slug>``."""
    directory = TYPE_DIRS.get(page_type, page_type)
    slug = slugify(title)
    date = date or today()
    meta = {
        "id": page_id or f"{directory}.{slug}",
        "type": page_type,
        "status": status,
        "created": date,
        "updated": date,
        "confidence": confidence,
        "sources": list(sources),
        "related": list(related),
        "tags": list(tags),
    }
    if body is None:
        body = TEMPLATES.get(page_type, _DEFAULT_TEMPLATE).format(title=title)
    return Page(meta=meta, body=body, path=path or f"{directory}/{slug}.md")


class WikiStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.git = _git.GitRepo(self.root)

    # -- setup ---------------------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.md"

    def exists(self) -> bool:
        return self.index_path.is_file()

    def init(self, *, git: bool = True) -> bool:
        """Create the skeleton (index.md + area dirs) and, if git is available, a repo with an
        initial commit by ``pan-memoryd``. Idempotent; returns True if anything was created."""
        created = False
        self.root.mkdir(parents=True, exist_ok=True)
        for d in WIKI_DIRS:
            keep = self.root / d / ".gitkeep"
            if not keep.parent.is_dir():
                keep.parent.mkdir(parents=True)
                keep.touch()
                created = True
        if not self.index_path.exists():
            self._atomic_write(self.index_path, INDEX_TEMPLATE.format(today=today()))
            created = True
        if git:
            if not _git.git_available():
                logger.warning("git not found; wiki at %s is not versioned", self.root)
            else:
                if self.git.init():
                    created = True
                if self.git.head() is None:
                    self.git.commit(["."], "wiki: initialize")
        return created

    # -- read ----------------------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def page_files(self) -> List[Path]:
        if not self.root.is_dir():
            return []
        files = [p for p in self.root.rglob("*.md")
                 if ".git" not in p.relative_to(self.root).parts and p.is_file()]
        return sorted(files, key=lambda p: self._rel(p))

    def load(self, path: str | Path) -> Page:
        """Parse one file (raises FrontmatterError / OSError)."""
        full = self.resolve_path(path)
        if full is None:
            raise WikiError(f"path outside the wiki: {path}")
        return Page.from_text(full.read_text(encoding="utf-8"), path=self._rel(full))

    def list_pages(self) -> List[Page]:
        """All parseable pages, sorted by path. Broken pages are skipped (see ``validate``)."""
        pages = []
        for f in self.page_files():
            try:
                pages.append(Page.from_text(f.read_text(encoding="utf-8"), path=self._rel(f)))
            except (FrontmatterError, OSError, UnicodeDecodeError) as exc:
                logger.warning("skipping wiki page %s: %s", f, exc)
        return pages

    def resolve_path(self, ref: str | Path) -> Optional[Path]:
        """Map a wiki-relative path (with or without ``.md``, optional ``wiki/`` prefix) to a file
        inside the wiki. None if it escapes the root."""
        s = str(ref).strip().replace("\\", "/")
        if s.startswith("wiki/"):
            s = s[5:]
        s = s.lstrip("/")
        if not s:
            return None
        candidate = (self.root / s)
        if candidate.suffix != ".md" and not candidate.is_file():
            candidate = candidate.with_name(candidate.name + ".md")
        try:
            rel = candidate.resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return None
        return self.root / rel if rel.suffix == ".md" else None

    def get(self, ref: str) -> Optional[Page]:
        """Page by id or by wiki-relative path. None if not found."""
        ref = (ref or "").strip()
        if not ref:
            return None
        looks_like_path = "/" in ref or ref.endswith(".md")
        if looks_like_path:
            page = self.get_path(ref)
            if page is not None:
                return page
        for page in self.list_pages():
            if page.id == ref:
                return page
        return None if looks_like_path else self.get_path(ref)

    def get_path(self, ref: str) -> Optional[Page]:
        """Page at a wiki-relative path (no id lookup)."""
        full = self.resolve_path(ref)
        if full is None or not full.is_file():
            return None
        try:
            return Page.from_text(full.read_text(encoding="utf-8"), path=self._rel(full))
        except (FrontmatterError, OSError, UnicodeDecodeError):
            return None

    # -- write ---------------------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def write(self, page: Page) -> str:
        """Validate and atomically write ``page``; returns its wiki-relative path.

        Raises WikiError on schema errors, a path outside the wiki, or an id already used by a
        different page."""
        errors = validate_meta(page.meta)
        if errors:
            raise WikiError(f"invalid page {page.id or page.path!r}: " + "; ".join(errors))
        rel = page.path or f"{TYPE_DIRS.get(page.type, page.type)}/{slugify(page.id.split('.')[-1])}.md"
        full = self.resolve_path(rel)
        if full is None or not rel.endswith(".md"):
            raise WikiError(f"invalid page path {rel!r}")
        rel = self._rel(full)
        for other in self.list_pages():
            if other.id == page.id and other.path != rel:
                raise WikiError(f"id {page.id!r} already used by {other.path}")
        self._atomic_write(full, page.to_text())
        page.path = rel
        return rel

    def link_in_index(self, page: Page, *, today: Optional[str] = None) -> bool:
        """Link ``page`` from ``index.md`` (see :func:`link_in_index`); True when index.md changed."""
        if page.path is None or not self.index_path.is_file():
            return False
        text = self.index_path.read_text(encoding="utf-8")
        new = link_in_index(text, page.path, page.title, today=today)
        if new is None or new == text:
            return False
        self._atomic_write(self.index_path, new)
        return True

    def delete(self, ref: str) -> Optional[str]:
        page = self.get(ref)
        if page is None or page.path is None:
            return None
        (self.root / page.path).unlink()
        return page.path

    def next_adr_number(self) -> int:
        nums = [int(m.group(1)) for p in (self.root / "decisions").glob("ADR-*.md")
                if (m := _ADR_RE.match(p.name))]
        return max(nums, default=0) + 1

    def new_decision(self, title: str, **kwargs) -> Page:
        """Decision page following the wiki conventions: ``decisions/ADR-NNN-slug.md`` (compact slug)."""
        n = self.next_adr_number()
        slug = slugify(title)
        if len(slug) > COMPACT_SLUG_OVER:
            slug = compact_slug(title)
        return new_page("decision", f"ADR-{n:03d}: {title}", page_id=f"decisions.adr-{n:03d}",
                        path=f"decisions/ADR-{n:03d}-{slug}.md", **kwargs)

    # -- validation ----------------------------------------------------------------------------

    def _link_ok(self, page_path: str, target: str, ids: set[str]) -> bool:
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            return True
        if target in ids:
            return True
        page_dir = PurePosixPath(page_path).parent
        root = self.root.resolve()
        for base in (page_dir, PurePosixPath(".")):
            try:
                full = (self.root / base / target).resolve()
            except OSError:
                continue
            # Links may point outside the wiki (e.g. to repo files); they only have to exist.
            if full.exists() and (base == page_dir or full.is_relative_to(root)):
                return True
        return False

    def validate(self) -> List[Issue]:
        """Schema, unique ids, ``related`` entries and relative Markdown links resolve."""
        issues: List[Issue] = []
        if not self.exists():
            issues.append(Issue("index.md", "missing (run `pan wiki init`)"))
        pages: List[Page] = []
        for f in self.page_files():
            rel = self._rel(f)
            try:
                pages.append(Page.from_text(f.read_text(encoding="utf-8"), path=rel))
            except (FrontmatterError, OSError, UnicodeDecodeError) as exc:
                issues.append(Issue(rel, str(exc)))
        seen: Dict[str, str] = {}
        for page in pages:
            issues.extend(Issue(page.path or "?", e) for e in validate_meta(page.meta))
            if page.id:
                if page.id in seen:
                    issues.append(Issue(page.path or "?", f"duplicate id {page.id!r} (also {seen[page.id]})"))
                else:
                    seen[page.id] = page.path or "?"
        ids = set(seen)
        for page in pages:
            assert page.path is not None
            for rel in page.meta.get("related") or []:
                if isinstance(rel, str) and not self._link_ok(page.path, rel, ids):
                    issues.append(Issue(page.path, f"related entry does not resolve: {rel}"))
            for target in links(page.body):
                if _SKIP_LINK.match(target):
                    continue
                if not self._link_ok(page.path, target, ids):
                    issues.append(Issue(page.path, f"broken link: {target}"))
        return sorted(issues, key=lambda i: (i.path, i.message))


def iter_changed(store: WikiStore, paths: Iterable[str]) -> List[Page]:
    """Parse the given wiki-relative paths that still exist (helper for incremental reindex)."""
    out = []
    for p in paths:
        page = store.get_path(p)
        if page is not None:
            out.append(page)
    return out
