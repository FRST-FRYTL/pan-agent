"""Reader (search/read/prefetch/overview), retrieval formatting and the `pan wiki`/`pan index` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FIXTURES
from pan.cli import main
from pan.index.fts import Hit
from pan.memory.reader import Reader, index_overview
from pan.memory.retrieval import FtsRetriever, Retriever, format_prefetch
from pan.paths import PanPaths


@pytest.fixture
def reader(indexed_home: Path):
    r = Reader(PanPaths.for_home(indexed_home))
    yield r
    r.close()


def _hit(i: int, score: float = 0.9, snippet: str = "x" * 300) -> Hit:
    return Hit(id=f"p.{i}", path=f"p/{i}.md", title=f"Page {i}", type="learning", status="active",
               score=score, bm25=-1.0, section="S", snippet=snippet)


def test_fts_retriever_implements_interface(reader: Reader):
    assert isinstance(reader.retriever, FtsRetriever) and isinstance(reader.retriever, Retriever)
    assert reader.retriever.retrieve("postgres", 1)[0].id == "systems.langfuse"


@pytest.mark.parametrize("budget", [150, 200, 500, 1500])
def test_format_prefetch_respects_budget(budget: int):
    out = format_prefetch([_hit(i) for i in range(10)], budget_chars=budget, min_score=0.2)
    assert len(out) <= budget
    assert out.count("\n- [") >= 1


@pytest.mark.parametrize("budget", [0, 30, 60, 100])
def test_format_prefetch_tiny_budget(budget: int):
    assert len(format_prefetch([_hit(1)], budget_chars=budget, min_score=0.2)) <= budget


def test_format_prefetch_threshold_and_empty():
    assert format_prefetch([], 1500, 0.2) == ""
    assert format_prefetch([_hit(1, score=0.1)], 1500, 0.2) == ""
    assert format_prefetch([_hit(1)], 10, 0.2) == ""
    out = format_prefetch([_hit(1, 0.9), _hit(2, 0.1)], 1500, 0.2)
    assert "[p.1]" in out and "[p.2]" not in out


def test_search_result_shape(reader: Reader):
    res = reader.search("why do vllm tool calls fail", limit=3)
    assert res["results"][0]["id"] == "learnings.vllm-tool-calling"
    assert set(res["results"][0]) == {"id", "path", "title", "type", "status", "confidence",
                                      "score", "section", "snippet"}
    assert len(res["results"]) <= 3
    assert reader.search("x", limit=999)["type"] == "any"
    assert reader.search("x", limit="bogus")  # coerced, no exception
    assert "error" in reader.search("   ")
    assert reader.search("kubernetes")["results"] == []


def test_read_by_id_path_and_section(reader: Reader):
    res = reader.read("decisions.adr-001")
    assert res["found"] and res["status"] == "accepted" and res["type"] == "decision"
    assert "frontmatter" not in res and "path" not in res  # no sources/event lists for the agent
    assert res["content"].startswith("# ADR-001")
    raw = reader.read("decisions.adr-001", raw=True)  # `pan wiki read`: the stored page
    assert raw["path"] == "decisions/ADR-001-sqlite-fts5-retrieval.md" and raw["frontmatter"]["sources"]
    assert reader.read("decisions/ADR-001-sqlite-fts5-retrieval.md")["id"] == "decisions.adr-001"
    sec = reader.read("decisions.adr-001", section="Decision")
    assert sec["content"].startswith("## Decision") and "Consequences" not in sec["content"]
    missing = reader.read("decisions.adr-001", section="Nope")
    assert "error" in missing and "Context" in missing["sections"]
    assert reader.read("nope")["found"] is False
    assert reader.read("../../../etc/passwd")["found"] is False
    assert "error" in reader.read("")


def test_prefetch_relevant_and_irrelevant(reader: Reader):
    out = reader.prefetch("tool calls fail with vllm, what is wrong?")
    assert "learnings.vllm-tool-calling" in out and len(out) <= 1500
    assert reader.prefetch("tell me a joke about cats") == ""
    assert reader.prefetch("") == ""
    assert len(reader.prefetch("vllm memory port disk tool docker postgres", budget_chars=300)) <= 300


def test_index_overview_capped_and_stable():
    text = (FIXTURES / "wiki" / "index.md").read_text()
    out = index_overview(text)
    assert out.startswith("- decisions/: ADR-001: SQLite FTS5 for retrieval")
    assert "Area" not in out and "](" not in out and "---" not in out
    assert out == index_overview(text)
    big = text + "\n".join(f"| area{i}/ | page {i} |" for i in range(200))
    assert len(index_overview(big)) <= 800


def test_reader_without_wiki_or_index(hermes_home: Path):
    r = Reader(PanPaths.for_home(hermes_home))
    assert r.search("vllm")["results"] == [] and "note" in r.search("vllm")
    assert r.read("index")["found"] is False
    assert r.prefetch("vllm") == ""
    assert r.wiki_overview() == ""


# -- CLI ------------------------------------------------------------------------------------------

def test_cli_wiki_init_validate_and_index(hermes_home: Path, capsys):
    assert main(["index", "rebuild"]) == 1  # no wiki yet
    assert main(["wiki", "init"]) == 0
    assert main(["wiki", "validate"]) == 0
    assert main(["index", "rebuild"]) == 0
    capsys.readouterr()
    assert main(["index", "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["pages"] == 1 and status["stale"] is False


def test_cli_search_read_and_rebuild_reproducible(indexed_home: Path, capsys):
    assert main(["wiki", "search", "--json", "unified", "memory"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["results"][0]["id"] == "systems.dgx-spark-host"

    (PanPaths.for_home(indexed_home).index_db).unlink()
    assert main(["wiki", "search", "unified memory"]) == 0
    assert "not available" in capsys.readouterr().out
    assert main(["index", "rebuild"]) == 0
    capsys.readouterr()
    assert main(["wiki", "search", "--json", "unified", "memory"]) == 0
    assert json.loads(capsys.readouterr().out) == first

    assert main(["wiki", "search", "unified", "memory"]) == 0
    assert "systems.dgx-spark-host" in capsys.readouterr().out
    assert main(["wiki", "read", "systems.langfuse"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("---\nid: systems.langfuse\n") and "# Langfuse" in out
    assert main(["wiki", "read", "systems.langfuse", "--section", "Credentials"]) == 0
    assert "## Credentials" in capsys.readouterr().out
    assert main(["wiki", "read", "missing.page"]) == 1
    assert main(["wiki", "validate"]) == 0


def test_cli_validate_fails_on_errors(indexed_home: Path, capsys):
    wiki = PanPaths.for_home(indexed_home).wiki
    (wiki / "learnings" / "bad.md").write_text("---\nid: bad\n---\n# Bad\n")
    assert main(["wiki", "validate"]) == 1
    assert "learnings/bad.md: missing required field 'type'" in capsys.readouterr().out
