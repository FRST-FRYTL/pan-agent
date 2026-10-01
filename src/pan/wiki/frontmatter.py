"""Wiki page model: YAML frontmatter + Markdown body (schema: spec v0.1 §9).

``parse``/``serialize`` round-trip a page; ``validate_meta`` checks the schema. Serialization is
deterministic (fixed key order, fixed list styles) so the daemon's git diffs stay minimal.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# Page types of the spec's wiki conventions, extended with the runtime wiki's classifier labels (spec §4.6).
PAGE_TYPES = frozenset({
    "architecture", "decision", "system", "learning", "incident", "operations", "bench", "index",
    "project", "procedure", "environment", "configuration", "user_preference",
})
# Page statuses plus the ADR lifecycle (proposed … deprecated) for decision pages.
PAGE_STATUSES = frozenset({
    "draft", "active", "proposed", "accepted", "rejected", "superseded", "deprecated",
})
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})

REQUIRED_KEYS = ("id", "type", "status", "created", "updated", "confidence")
LIST_KEYS = ("sources", "related", "tags")
KEY_ORDER = REQUIRED_KEYS + LIST_KEYS
FLOW_LIST_KEYS = frozenset({"tags"})

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_FM_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)
_H1_RE = re.compile(r"^#[ \t]+(.+?)[ \t#]*$", re.M)


class FrontmatterError(ValueError):
    """The page has no parseable frontmatter block."""


@dataclass
class Page:
    """One wiki page. ``path`` is relative to the wiki root (POSIX), ``None`` until placed."""

    meta: Dict[str, Any]
    body: str
    path: Optional[str] = None

    @property
    def id(self) -> str:
        return str(self.meta.get("id", ""))

    @property
    def type(self) -> str:
        return str(self.meta.get("type", ""))

    @property
    def status(self) -> str:
        return str(self.meta.get("status", ""))

    @property
    def tags(self) -> List[str]:
        return [str(t) for t in self.meta.get("tags") or []]

    @property
    def title(self) -> str:
        m = _H1_RE.search(self.body)
        return m.group(1).strip() if m else self.id

    def to_text(self) -> str:
        return serialize(self.meta, self.body)

    @classmethod
    def from_text(cls, text: str, path: Optional[str] = None) -> "Page":
        meta, body = parse(text)
        return cls(meta=meta, body=body, path=path)


def _normalize(value: Any) -> Any:
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    return value


def parse(text: str) -> tuple[Dict[str, Any], str]:
    """Split ``text`` into (meta, body). Dates are normalized to ISO strings."""
    text = text.lstrip("﻿")
    m = _FM_RE.match(text)
    if not m:
        raise FrontmatterError("missing YAML frontmatter block (--- ... ---)")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise FrontmatterError("frontmatter must be a YAML mapping")
    body = text[m.end():]
    return _normalize(meta), body


def _scalar(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value  # keep dates unquoted, like hand-written pages
    out = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10_000).strip()
    return out[:-4].rstrip() if out.endswith("\n...") else out  # drop YAML's document-end marker


def serialize(meta: Dict[str, Any], body: str) -> str:
    """Render frontmatter in schema key order (unknown keys after, sorted) followed by the body."""
    meta = _normalize(dict(meta))
    keys = [k for k in KEY_ORDER if k in meta] + sorted(k for k in meta if k not in KEY_ORDER)
    lines = ["---"]
    for key in keys:
        value = meta[key]
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            elif key in FLOW_LIST_KEYS:
                lines.append(f"{key}: [{', '.join(_scalar(v) for v in value)}]")
            else:
                lines.append(f"{key}:")
                lines.extend(f"  - {_scalar(v)}" for v in value)
        elif isinstance(value, dict):
            dumped = yaml.safe_dump({key: value}, default_flow_style=False, allow_unicode=True,
                                    sort_keys=True).rstrip()
            lines.append(dumped)
        else:
            lines.append(f"{key}: {_scalar(value)}")
    lines.append("---")
    body = body if body.startswith("\n") else "\n" + body
    if not body.endswith("\n"):
        body += "\n"
    return "\n".join(lines) + "\n" + body


def _is_date(value: Any) -> bool:
    try:
        _dt.date.fromisoformat(str(value))
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)))
    except ValueError:
        return False


def validate_meta(meta: Dict[str, Any]) -> List[str]:
    """Return schema violations (empty list = valid)."""
    errors: List[str] = []
    for key in REQUIRED_KEYS:
        if meta.get(key) in (None, ""):
            errors.append(f"missing required field '{key}'")
    page_id = meta.get("id")
    if page_id not in (None, "") and (not isinstance(page_id, str) or not ID_RE.match(page_id)):
        errors.append(f"invalid id {page_id!r} (letters, digits, '.', '-', '_', ':')")
    for key, allowed in (("type", PAGE_TYPES), ("status", PAGE_STATUSES),
                         ("confidence", CONFIDENCE_LEVELS)):
        value = meta.get(key)
        if value not in (None, "") and (not isinstance(value, str) or value not in allowed):
            errors.append(f"invalid {key} {value!r} (allowed: {', '.join(sorted(allowed))})")
    for key in ("created", "updated"):
        value = meta.get(key)
        if value not in (None, "") and not _is_date(value):
            errors.append(f"invalid {key} {value!r} (expected YYYY-MM-DD)")
    if _is_date(meta.get("created")) and _is_date(meta.get("updated")) \
            and str(meta["updated"]) < str(meta["created"]):
        errors.append("'updated' is before 'created'")
    for key in LIST_KEYS:
        value = meta.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, (str, int, float)) for v in value):
            errors.append(f"'{key}' must be a list of strings")
    return errors
