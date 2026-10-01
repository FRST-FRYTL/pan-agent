"""Small Markdown helpers: headings/sections, links and slugs (no Markdown library needed)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


# A provenance stamp or evidence label at the end of a claim ("… (stated 2026-09-30)", "… (observed)") is
# not part of a name made from it.
_STAMP = re.compile(r"(?:\s*\((?:(?:stated|noted|said)\b[^()]*|observed|inferred|reported)(?:;[^()]*)?\))+\s*$", re.I)


def slugify(text: str, max_len: int = 64) -> str:
    """ASCII, lowercase, hyphen-separated slug ("vLLM Tool-Calling!" → "vllm-tool-calling"; a trailing
    provenance stamp such as "(stated 2026-09-30)" is left out)."""
    text = _STAMP.sub("", text or "") or text or ""
    text = text.replace("ß", "ss").replace("ẞ", "SS")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "page"


_SLUG_STOPWORDS = frozenset("""
a an the and or of for to in on at by with from as is are be will we i our us you your use uses using used
decide decided decision go going let lets let's that this these those it its should shall must
""".split())


def compact_slug(text: str, max_words: int = 6, max_len: int = 48) -> str:
    """Short slug for titles made from a whole sentence (M6: ADR paths were the full user sentence):
    content words only, at most ``max_words`` words and ``max_len`` characters, cut at a word
    ("We use SQLite, not Postgres, for the bench results index" → "sqlite-not-postgres-bench-results-index")."""
    words = [slugify(w) for w in re.findall(r"[^\W_][\w.+'-]*", text or "") if w.lower() not in _SLUG_STOPWORDS]
    out: List[str] = []
    for w in words:
        if w == "page" or not w:
            continue
        if len("-".join(out + [w])) > max_len or len(out) >= max_words:
            break
        out.append(w)
    return "-".join(out) or "page"


@dataclass(frozen=True)
class Section:
    level: int          # 0 = preamble before the first heading
    heading: str
    anchor: str
    start: int          # line index of the heading (or 0 for the preamble)
    end: int            # line index where this section's *own* text ends (next heading)
    text: str           # own text, without the heading line and without subsections


def split_sections(body: str) -> List[Section]:
    """Split ``body`` at every heading (fenced code blocks are skipped). Empty preamble is dropped."""
    lines = body.splitlines()
    marks: list[tuple[int, int, str]] = []  # (line, level, heading)
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            marks.append((i, len(m.group(1)), m.group(2).strip()))
    sections: List[Section] = []
    first = marks[0][0] if marks else len(lines)
    pre = "\n".join(lines[:first]).strip()
    if pre:
        sections.append(Section(0, "", "", 0, first, pre))
    seen: dict[str, int] = {}
    for n, (i, level, heading) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        anchor = slugify(heading, max_len=80)
        if anchor in seen:  # GitHub-style de-duplication
            seen[anchor] += 1
            anchor = f"{anchor}-{seen[anchor]}"
        else:
            seen[anchor] = 0
        sections.append(Section(level, heading, anchor, i, end, "\n".join(lines[i + 1:end]).strip()))
    return sections


def extract_section(body: str, heading: str) -> Optional[str]:
    """Return the section titled ``heading`` (case-insensitive, or its anchor) incl. subsections."""
    want = heading.strip().lstrip("#").strip()
    want_l, want_anchor = want.lower(), slugify(want, max_len=80)
    sections = [s for s in split_sections(body) if s.level > 0]
    match = next((s for s in sections if s.heading.lower() == want_l), None) \
        or next((s for s in sections if s.anchor == want_anchor), None)
    if match is None:
        return None
    lines = body.splitlines()
    end = len(lines)
    for s in sections:
        if s.start > match.start and s.level <= match.level:
            end = s.start
            break
    return "\n".join(lines[match.start:end]).strip()


def links(body: str) -> List[str]:
    """Markdown link targets in ``body`` (images and fenced code excluded)."""
    out: List[str] = []
    in_fence = False
    for line in body.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"`[^`]*`", "", line)  # ignore inline code
        out.extend(m.group(2) for m in _LINK_RE.finditer(line))
    return out


def strip_links(text: str) -> str:
    """Replace ``[text](url)`` by ``text``."""
    return _LINK_RE.sub(lambda m: m.group(1), text)
