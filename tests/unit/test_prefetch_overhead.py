"""Per-request overhead: prefetch query focus and gate, marker notes where markers appear, the page-list
overview, and the compact tool schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.config import RetrievalConfig
from pan.index.fts import Hit
from pan.memory.reader import Reader, index_overview
from pan.memory.retrieval import Retriever, focus_query, format_prefetch, marker_notes, rerank_queries
from pan.paths import PanPaths
from pan.wiki.store import WikiStore, new_page


def _provider():
    return pytest.importorskip("pan.hermes.provider", reason="the provider module needs Hermes Agent")


def _hit(i: int, score: float, prior: float = 0.0, via: str = "rerank", snippet: str = "a fact",
         status: str = "active") -> Hit:
    return Hit(id=f"p.{i}", path=f"p/{i}.md", title=f"Page {i}", type="project", status=status,
               score=score, bm25=0.0, snippet=snippet, via=via, prior=prior)


class _Fixed(Retriever):
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def retrieve(self, query, k=5, *, type=None):
        self.queries.append((query, k))
        return self.hits[:k]


def _reader(tmp_path: Path, hits) -> Reader:
    return Reader(PanPaths.for_home(tmp_path), retriever=_Fixed(hits), config=RetrievalConfig())


# -- query focus -------------------------------------------------------------------------------------


def test_focus_query_shortens_absolute_paths_only():
    q = "Your working directory is: /srv/work/runs/job-7/workspace\n\nTask: summarise notes/inbox.md daily."
    out = focus_query(q)
    assert "/srv" not in out and "workspace" in out and "notes/inbox.md" in out and "\n\n" in out
    assert focus_query("see https://example.org/a/b and ~/src/app/main.py") == \
        "see https://example.org/a/b and main.py"
    assert focus_query("on 30/09/2026, and/or 1/2 of it") == "on 30/09/2026, and/or 1/2 of it"
    assert focus_query("Wie heißt mein Hund?") == "Wie heißt mein Hund?"


def test_rerank_queries_add_the_paragraphs_of_a_long_message():
    q = "You are running inside the desktop app. Your working directory is: workspace\n\nTask: What dose does Shadow get?"
    out = rerank_queries(q)
    assert out[0] == " ".join(q.split())
    assert "Task: What dose does Shadow get?" in out
    assert "You are running inside the desktop app. Your working directory is: workspace" in out
    assert rerank_queries("Wie heißt mein Hund?") == ["Wie heißt mein Hund?"]


# -- prefetch gate ---------------------------------------------------------------------------------


def test_prefetch_relative_floor_drops_noise_next_to_a_clear_match(tmp_path: Path):
    r = _reader(tmp_path, [_hit(1, 0.5), _hit(2, 0.003), _hit(3, 0.003, prior=0.45), _hit(4, 0.001)])
    out = r.prefetch("what dose does shadow get?")
    assert "[p.1]" in out and "[p.3]" in out           # the match, and a hit kept on its fused score
    assert "[p.2]" not in out and "[p.4]" not in out   # 0.003 < 1 % of 0.5


def test_prefetch_weak_best_hit_keeps_the_absolute_floor(tmp_path: Path):
    r = _reader(tmp_path, [_hit(1, 0.02), _hit(2, 0.004), _hit(3, 0.001)])
    out = r.prefetch("wie war das nochmal?")
    assert "[p.1]" in out and "[p.2]" in out and "[p.3]" not in out


def test_prefetch_takes_prefetch_k_candidates_of_the_focused_query(tmp_path: Path):
    r = _reader(tmp_path, [_hit(i, 0.9) for i in range(1, 7)])
    out = r.prefetch("Directory: /a/b/c/work\n\nTask: x")
    assert r.retriever.queries == [("Directory: work\n\nTask: x", RetrievalConfig().prefetch_k)]
    assert out.count("\n- [") == 4


def test_prefetch_lines_omit_the_default_status():
    out = format_prefetch([_hit(1, 0.9), _hit(2, 0.9, status="proposed")], 1500, 0.0)
    assert "[p.1] Page 1 (project):" in out and "(project, proposed)" in out


# -- marker notes --------------------------------------------------------------------------------------


def test_marker_notes_only_for_markers_present():
    assert marker_notes("plain fact") == ""
    notes = marker_notes("Assistant reported: moved the job (noted 2026-09-20, inferred)")
    assert '"(noted DATE)"' in notes and '"Assistant reported:"' in notes and '"(inferred)"' in notes
    assert "superseded" not in notes
    assert "earlier time" in marker_notes("- old value (noted 2026-09-01; superseded 2026-09-20)")


def test_prefetch_block_explains_its_markers_within_budget():
    hits = [_hit(i, 0.9, snippet="x" * 200 + " (noted 2026-09-20)") for i in range(8)]
    for budget in (300, 700, 1500):
        out = format_prefetch(hits, budget, 0.0)
        assert len(out) <= budget and out.count("\n- [") >= 1
        assert out.endswith("never derive dates from it.") or budget < 500
    assert "never derive" not in format_prefetch([_hit(1, 0.9)], 1500, 0.0)


def test_memory_read_notes_its_markers(tmp_path: Path):
    paths = PanPaths.for_home(tmp_path)
    store = WikiStore(paths.wiki)
    store.init(git=False)
    page = new_page("project", "Shadow", date="2026-09-20",
                    body="# Shadow\n\n## Facts\n\n- Shadow gets 25 mg carprofen (stated 2026-09-20) (observed)\n")
    store.write(page)
    r = Reader(paths)
    res = r.read(page.id)
    assert res["found"] and "(noted 2026-09-20)" in res["content"]
    assert "never derive dates" in res["note"]
    assert "note" not in r.read(page.id, raw=True)


def test_static_block_leaves_the_marker_legend_to_the_notes():
    SYSTEM_PROMPT_BLOCK = _provider().SYSTEM_PROMPT_BLOCK
    assert "(noted DATE)" not in SYSTEM_PROMPT_BLOCK and "Assistant reported" not in SYSTEM_PROMPT_BLOCK
    for kept in ("open a listed page", "Newest wins", "never invent them", "keep status words",
                 "call them directly by name", "PAN has no other tools"):
        assert kept in SYSTEM_PROMPT_BLOCK


# -- overview and schemas --------------------------------------------------------------------------------


def test_overview_is_the_page_list_of_a_daemon_index(tmp_path: Path):
    store = WikiStore(tmp_path / "wiki")
    store.init(git=False)
    assert index_overview(store.index_path.read_text()) == ""   # template only: nothing to list
    page = new_page("project", "Shadow (pet)", date="2026-09-24")
    store.write(page)
    store.link_in_index(page)
    out = index_overview(store.index_path.read_text())
    assert out == "projects/:\n- Shadow (pet)"


def test_tool_schemas_are_compact():
    MEMORY_SEARCH_SCHEMA, MEMORY_READ_SCHEMA = _provider().MEMORY_SEARCH_SCHEMA, _provider().MEMORY_READ_SCHEMA
    props = MEMORY_SEARCH_SCHEMA["parameters"]["properties"]
    assert set(props) == {"query", "limit"} and MEMORY_SEARCH_SCHEMA["parameters"]["required"] == ["query"]
    assert "Use before re-deriving" in MEMORY_SEARCH_SCHEMA["description"]
    assert set(MEMORY_READ_SCHEMA["parameters"]["properties"]) == {"page", "section"}
