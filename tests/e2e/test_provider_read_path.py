"""PanMemoryProvider read path (M2): tools, prefetch and system prompt block against a real wiki/index."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pan.hermes.provider import ACTIVE_PREFERENCES_HEADER, SYSTEM_PROMPT_BLOCK, PanMemoryProvider
from pan.memory.retrieval import PREFETCH_BUDGET_CHARS
from pan.paths import PanPaths

pytestmark = pytest.mark.e2e


def _provider(home: Path) -> PanMemoryProvider:
    p = PanMemoryProvider()
    p.initialize("sess-m2", hermes_home=str(home), platform="cli", agent_context="primary")
    return p


def _call(p: PanMemoryProvider, name: str, **args) -> dict:
    out = p.handle_tool_call(name, args)
    assert isinstance(out, str)
    return json.loads(out)


def test_memory_search_and_read(indexed_home: Path):
    p = _provider(indexed_home)
    res = _call(p, "memory_search", query="which retrieval backend did we choose", type="decision")
    assert [r["id"] for r in res["results"]] == ["decisions.adr-001"]
    res = _call(p, "memory_search", query="vllm", limit=2)
    assert 1 <= len(res["results"]) <= 2
    page = _call(p, "memory_read", page="learnings.vllm-tool-calling")
    assert page["found"] and page["status"] == "active" and "frontmatter" not in page
    assert "--tool-call-parser" in page["content"]
    sec = _call(p, "memory_read", page="operations/restart-vllm.md", section="Rollback")
    assert sec["content"].startswith("## Rollback")


@pytest.mark.parametrize("name, args", [
    ("memory_search", {}),
    ("memory_search", {"query": 'foo AND ("'}),
    ("memory_search", {"query": "x", "limit": "many", "type": 42}),
    ("memory_read", {}),
    ("memory_read", {"page": None}),
    ("memory_read", {"page": "../../config.yaml"}),
    ("unknown_tool", {"query": "x"}),
])
def test_tools_always_return_json(indexed_home: Path, name: str, args: dict):
    res = _call(_provider(indexed_home), name, **args)
    assert isinstance(res, dict)


def _wiki(prefetch: str) -> str:
    """The wiki part of a prefetch (M6: active USER.md preferences may precede it)."""
    if prefetch.startswith(ACTIVE_PREFERENCES_HEADER):
        return prefetch.split("\n\n", 1)[1] if "\n\n" in prefetch else ""
    return prefetch


def test_active_preferences_lead_every_prefetch_within_the_budget(indexed_home: Path):
    p = _provider(indexed_home)
    out = p.prefetch("what's the weather like today?")
    assert out.startswith(ACTIVE_PREFERENCES_HEADER) and "concise answers" in out
    assert "User preference (own words)" not in out
    full = p.prefetch("tool calls fail on vllm, is the tool call parser missing?")
    assert full.startswith(ACTIVE_PREFERENCES_HEADER) and "learnings.vllm-tool-calling" in full
    assert len(full) <= PREFETCH_BUDGET_CHARS


def test_tools_on_empty_home(hermes_home: Path):
    p = _provider(hermes_home)
    res = _call(p, "memory_search", query="vllm")
    assert res["results"] == [] and "note" in res
    assert _call(p, "memory_read", page="index")["found"] is False
    assert _wiki(p.prefetch("vllm tool calling")) == ""
    assert p.system_prompt_block() == SYSTEM_PROMPT_BLOCK


def test_uninitialized_provider_is_safe():
    p = PanMemoryProvider()
    assert "error" in json.loads(p.handle_tool_call("memory_search", {"query": "x"}))
    assert p.prefetch("x") == ""
    assert p.system_prompt_block() == SYSTEM_PROMPT_BLOCK


def test_prefetch_budget_threshold_and_speed(indexed_home: Path):
    p = _provider(indexed_home)
    start = time.perf_counter()
    out = p.prefetch("tool calls fail on vllm, is the tool call parser missing?", session_id="sess-m2")
    elapsed = time.perf_counter() - start
    assert "learnings.vllm-tool-calling" in out
    assert 0 < len(out) <= PREFETCH_BUDGET_CHARS
    assert elapsed < 0.2
    assert _wiki(p.prefetch("what's the weather like today?")) == ""
    long_query = " ".join(["vllm tool memory port docker postgres disk spool retrieval"] * 50)
    assert len(p.prefetch(long_query)) <= PREFETCH_BUDGET_CHARS


def test_prefetch_does_not_create_or_rebuild_index(indexed_home: Path):
    paths = PanPaths.for_home(indexed_home)
    paths.index_db.unlink()
    p = _provider(indexed_home)
    assert _wiki(p.prefetch("vllm tool calling")) == ""
    assert not paths.index_db.exists()


def test_system_prompt_block_includes_wiki_sections_and_is_stable(indexed_home: Path):
    p = _provider(indexed_home)
    block = p.system_prompt_block()
    assert block.startswith(SYSTEM_PROMPT_BLOCK)
    overview = block[len(SYSTEM_PROMPT_BLOCK):]
    assert "decisions/" in overview and "learnings/" in overview
    assert len(overview) <= 800 + len("\n\nWiki pages:\n")
    assert _provider(indexed_home).system_prompt_block() == block  # same wiki → same block
    assert p.system_prompt_block() is block


def test_system_prompt_block_caps_large_index(indexed_home: Path):
    index = PanPaths.for_home(indexed_home).wiki / "index.md"
    index.write_text(index.read_text() + "".join(f"| area{i}/ | [p](x{i}.md) |\n" for i in range(300)))
    block = _provider(indexed_home).system_prompt_block()
    assert len(block) - len(SYSTEM_PROMPT_BLOCK) <= 800 + len("\n\nWiki pages:\n")
