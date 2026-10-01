"""Derived FTS5 index: determinism, incremental updates, ranking, query robustness."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pan.index.fts import FtsIndex, match_expression, query_terms, rebuild
from pan.wiki.store import WikiStore, new_page


@pytest.fixture
def index(seed_wiki: Path, tmp_path: Path):
    idx = FtsIndex(tmp_path / "index.db")
    idx.rebuild(seed_wiki)
    yield idx
    idx.close()


def test_sqlite_has_fts5():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(a, tokenize='porter unicode61')")


def test_rebuild_is_deterministic(seed_wiki: Path, tmp_path: Path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    assert rebuild(a, seed_wiki) == rebuild(b, seed_wiki) == 7
    ia, ib = FtsIndex(a, readonly=True), FtsIndex(b, readonly=True)
    assert ia.dump() == ib.dump() and ia.dump()
    assert ia.status()["wiki_hash"] == ib.status()["wiki_hash"]
    ia.close()
    ib.close()
    assert a.read_bytes() == b.read_bytes()


def test_rebuild_after_delete_reproduces_results(seed_wiki: Path, tmp_path: Path):
    db = tmp_path / "index.db"
    rebuild(db, seed_wiki)
    queries = ["vllm tool calling", "unified memory", "postgres", "disk full"]
    before = [FtsIndex(db, readonly=True).search(q) for q in queries]
    db.unlink()
    rebuild(db, seed_wiki)
    after = [FtsIndex(db, readonly=True).search(q) for q in queries]
    assert before == after and all(before)


@pytest.mark.parametrize("query, expected", [
    ("why do tool calls fail on vllm", "learnings.vllm-tool-calling"),
    ("which retrieval backend did we decide on", "decisions.adr-001"),
    ("how much unified memory does the host have", "systems.dgx-spark-host"),
    ("restart the vllm container", "operations.restart-vllm"),
    ("postgres database for tracing", "systems.langfuse"),
    ("events dropped because the disk was full", "incidents.2026-09-20-spool-disk-full"),
])
def test_ranking_on_seed_wiki(index: FtsIndex, query: str, expected: str):
    hits = index.search(query)
    assert hits and hits[0].id == expected, [(h.id, h.score) for h in hits]
    assert 0.2 <= hits[0].score <= 1.0


def test_hit_fields_and_section_snippet(index: FtsIndex):
    hit = index.search("gpu memory utilization budget")[0]
    assert hit.id == "systems.dgx-spark-host"
    assert hit.path == "systems/dgx-spark-host.md" and hit.type == "system" and hit.status == "active"
    assert hit.section == "Memory budget" and hit.anchor == "memory-budget"
    assert "unified memory" in hit.snippet
    assert all(0.0 <= h.score <= 1.0 for h in index.search("vllm"))


def test_type_filter_and_limit(index: FtsIndex):
    assert {h.type for h in index.search("vllm port", type="system")} == {"system"}
    assert index.search("vllm", type="any")
    assert len(index.search("the vllm tool port memory disk", limit=2)) <= 2
    assert all(h.type != "index" for h in index.search("learnings decisions systems"))
    assert index.search("test seed", type="index")[0].id == "index"


def test_irrelevant_query_scores_low(index: FtsIndex):
    assert index.search("kubernetes helm chart") == []
    assert all(h.score < 0.2 for h in index.search("hello, thanks"))


@pytest.mark.parametrize("query", [
    'foo AND (', '"unbalanced', "NEAR(a b", "*", "a OR", "-", ":", "col:vllm", "^vllm", "{x}",
    "", "   ", "'); DROP TABLE pages; --", "\x00", "ü" * 500, "AND OR NOT", "vllm*",
])
def test_weird_queries_never_raise(index: FtsIndex, query: str):
    hits = index.search(query)
    assert isinstance(hits, list)


def test_operators_are_literal_terms():
    assert query_terms('foo NEAR (bar') == ["foo", "near", "bar"]
    assert query_terms('foo AND (bar') == ["foo", "bar"]  # "and" is a stopword
    assert match_expression(["foo", "and"]) == '"foo" OR "and"'
    assert query_terms("the and of") == ["the", "and", "of"]  # only stopwords → keep them
    assert len(query_terms(" ".join(f"w{i}" for i in range(100)))) == 24


def test_incremental_update_and_remove(seed_wiki: Path, tmp_path: Path):
    db = tmp_path / "index.db"
    idx = FtsIndex(db)
    idx.rebuild(seed_wiki)
    store = WikiStore(seed_wiki)
    page = new_page("learning", "Kubernetes is not used", date="2026-09-23",
                    body="# Kubernetes is not used\n\nEverything runs in plain Docker on one host.\n")
    store.write(page)
    assert idx.status(seed_wiki)["missing"] == [page.path]
    assert idx.update([page]) == 1
    assert idx.search("kubernetes")[0].id == page.id
    assert idx.status(seed_wiki)["stale"] is False

    page.body = "# Kubernetes is not used\n\nNomad was evaluated too.\n"
    store.write(page)
    assert idx.status(seed_wiki)["changed"] == [page.path]
    idx.update([page])
    assert idx.search("nomad")[0].id == page.id
    assert idx.status()["pages"] == 8

    # incremental result == full rebuild of the same wiki
    full = tmp_path / "full.db"
    rebuild(full, seed_wiki)
    assert FtsIndex(full, readonly=True).dump() == idx.dump()

    assert idx.remove([page.id]) == 1
    assert idx.search("nomad") == []
    store.delete(page.id)
    assert idx.status(seed_wiki)["stale"] is False


def test_update_handles_id_change_and_remove_paths(index: FtsIndex, seed_wiki: Path):
    store = WikiStore(seed_wiki)
    page = store.get("systems.langfuse")
    page.meta["id"] = "systems.langfuse-v2"
    index.update([page])
    assert index.lookup("systems.langfuse") is None
    assert index.lookup("systems.langfuse-v2") == "systems/langfuse.md"
    assert index.remove_paths(["systems/langfuse.md"]) == 1
    assert index.lookup("systems.langfuse-v2") is None


def test_readonly_index_missing_or_foreign(tmp_path: Path):
    ro = FtsIndex(tmp_path / "missing.db", readonly=True)
    assert ro.search("anything") == [] and not ro.available()
    assert ro.status() == {"path": str(tmp_path / "missing.db"), "exists": False, "usable": False}
    assert not (tmp_path / "missing.db").exists()  # read-only never creates the file
    with pytest.raises(PermissionError):
        ro.rebuild(tmp_path)
    foreign = tmp_path / "foreign.db"
    sqlite3.connect(foreign).execute("CREATE TABLE x(a)").connection.commit()
    assert FtsIndex(foreign, readonly=True).search("x") == []


def test_reader_sees_rebuild(seed_wiki: Path, tmp_path: Path):
    db = tmp_path / "index.db"
    rebuild(db, seed_wiki)
    reader = FtsIndex(db, readonly=True)
    assert reader.search("nomad") == []
    store = WikiStore(seed_wiki)
    store.write(new_page("learning", "Nomad scheduler", date="2026-09-23"))
    rebuild(db, seed_wiki)  # atomic replace → reader reopens transparently
    assert reader.search("nomad")[0].id == "learnings.nomad-scheduler"


def test_empty_wiki(tmp_path: Path):
    store = WikiStore(tmp_path / "wiki")
    store.init(git=False)
    idx = FtsIndex(tmp_path / "index.db")
    assert idx.rebuild(store.root) == 1  # just index.md
    assert idx.search("vllm") == []
