"""Frontmatter parsing/serialization, schema validation and Markdown helpers."""

from __future__ import annotations

import pytest

from conftest import FIXTURES
from pan.wiki.frontmatter import FrontmatterError, Page, parse, serialize, validate_meta
from pan.wiki.markdown import extract_section, links, slugify, split_sections

VALID = {
    "id": "learnings.x", "type": "learning", "status": "active", "created": "2026-09-01",
    "updated": "2026-09-02", "confidence": "high", "sources": ["session:s1"], "related": [],
    "tags": ["a", "b"],
}


def test_parse_seed_page_normalizes_dates():
    text = (FIXTURES / "wiki" / "learnings" / "vllm-tool-calling.md").read_text()
    meta, body = parse(text)
    assert meta["id"] == "learnings.vllm-tool-calling"
    assert meta["created"] == "2026-09-23"  # str, not datetime.date
    assert body.lstrip().startswith("# vLLM serves without tool calling")
    assert validate_meta(meta) == []


def test_round_trip_is_stable():
    text = serialize(VALID, "\n# Title\n\nBody text.\n")
    meta, body = parse(text)
    assert meta == VALID
    assert serialize(meta, body) == text
    assert "created: 2026-09-01\n" in text  # dates stay unquoted
    assert "tags: [a, b]\n" in text
    assert "sources:\n  - session:s1\n" in text


def test_serialize_key_order_and_quoting():
    meta = dict(VALID, zzz="x", aaa="y: z", tags=["needs: quote"])
    text = serialize(meta, "# T\n")
    keys = [ln.split(":")[0] for ln in text.splitlines()[1:] if ln and not ln.startswith((" ", "-"))]
    assert keys[:9] == ["id", "type", "status", "created", "updated", "confidence", "sources",
                        "related", "tags"]
    assert keys[9:11] == ["aaa", "zzz"]
    assert parse(text)[0]["tags"] == ["needs: quote"]
    assert parse(text)[0]["aaa"] == "y: z"


@pytest.mark.parametrize("text", ["no frontmatter", "---\n: : :\n---\n", "---\n- a list\n---\nbody"])
def test_parse_errors(text):
    with pytest.raises(FrontmatterError):
        parse(text)


@pytest.mark.parametrize("change, expected", [
    ({"id": None}, "missing required field 'id'"),
    ({"type": "blog"}, "invalid type"),
    ({"status": "done"}, "invalid status"),
    ({"confidence": "certain"}, "invalid confidence"),
    ({"created": "23.09.2026"}, "invalid created"),
    ({"updated": "2026-08-01"}, "'updated' is before 'created'"),
    ({"tags": "a, b"}, "'tags' must be a list"),
    ({"id": "has space"}, "invalid id"),
    ({"type": ["learning"]}, "invalid type"),
])
def test_validate_meta_errors(change, expected):
    errors = validate_meta({**VALID, **change})
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize("page_type", ["procedure", "environment", "configuration", "user_preference",
                                       "bench", "index", "decision"])
def test_runtime_types_allowed(page_type):
    assert validate_meta({**VALID, "type": page_type}) == []


def test_page_title_falls_back_to_id():
    assert Page(meta={"id": "x.y"}, body="no heading").title == "x.y"
    assert Page(meta={"id": "x.y"}, body="\n# Real Title #\n").title == "Real Title"


def test_slugify():
    assert slugify("vLLM Tool-Calling!") == "vllm-tool-calling"
    assert slugify("Größe über alles") == "grosse-uber-alles"
    assert slugify("!!!") == "page"
    # a provenance stamp or evidence label is not part of the name
    assert slugify("The user adopted a ferret. (stated 2026-04-02)") == "the-user-adopted-a-ferret"
    assert slugify("Kiln temperature is 1240 C (observed)") == "kiln-temperature-is-1240-c"
    assert slugify("Budget (2026)") == "budget-2026"


BODY = """\
Intro line.

# Title

## Setup
Install it.

### Details
Deep detail.

```bash
# not a heading
```

## Setup
Second setup.
"""


def test_sections_skip_code_and_dedupe_anchors():
    secs = split_sections(BODY)
    assert [(s.level, s.heading) for s in secs] == [(0, ""), (1, "Title"), (2, "Setup"),
                                                     (3, "Details"), (2, "Setup")]
    assert [s.anchor for s in secs if s.level] == ["title", "setup", "details", "setup-1"]
    assert "# not a heading" in secs[3].text


def test_extract_section_includes_subsections():
    part = extract_section(BODY, "setup")
    assert part.startswith("## Setup") and "Deep detail." in part and "Second setup." not in part
    assert extract_section(BODY, "## Details").startswith("### Details")
    assert extract_section(BODY, "missing") is None


def test_links_ignore_code_and_images():
    body = "See [a](a.md) and ![img](x.png), `[no](code.md)`.\n```\n[no](fence.md)\n```\n[b](../b.md#x)"
    assert links(body) == ["a.md", "../b.md#x"]
