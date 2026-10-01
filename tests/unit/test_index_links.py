"""M5: pages the daemon creates are linked from wiki/index.md, grouped by area (WikiStore.link_in_index)."""

from __future__ import annotations

import shutil
from pathlib import Path

from conftest import FIXTURES
from pan.wiki.frontmatter import Page
from pan.wiki.store import INDEX_PAGES_HEADING, WikiStore, new_page


def test_link_in_index_groups_by_area_and_sorts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("pan.wiki.store.today", lambda: "2026-09-24")  # init() dates index.md by the clock
    store = WikiStore(tmp_path / "wiki")
    store.init(git=False)
    pages = [new_page("environment", "vLLM server", date="2026-09-24"),
             new_page("environment", "GPU", date="2026-09-24"),
             new_page("learning", "Tool call [parser] missing", date="2026-09-24")]
    for p in pages:
        store.write(p)
        assert store.link_in_index(p, today="2026-09-24")
    assert not store.link_in_index(pages[0])  # already linked: idempotent
    text = store.index_path.read_text()
    section = text[text.index(INDEX_PAGES_HEADING):]
    assert section.index("### learnings/") < section.index("### operations/")
    assert section.index("[GPU](operations/gpu.md)") < section.index("[vLLM server](operations/vllm-server.md)")
    assert "[Tool call (parser) missing](learnings/tool-call-parser-missing.md)" in section
    assert Page.from_text(text).meta["updated"] == "2026-09-24"
    assert store.validate() == []  # every link resolves


def test_link_in_index_respects_hand_written_links(tmp_path: Path):
    wiki = tmp_path / "wiki"
    shutil.copytree(FIXTURES / "wiki", wiki)
    store = WikiStore(wiki)
    before = store.index_path.read_text()
    assert not store.link_in_index(store.get("learnings.vllm-tool-calling"))  # linked in the table
    assert store.index_path.read_text() == before


def test_index_overview_lists_linked_pages(tmp_path: Path):
    from pan.memory.reader import index_overview

    store = WikiStore(tmp_path / "wiki")
    store.init(git=False)
    page = new_page("environment", "GPU", date="2026-09-24")
    store.write(page)
    store.link_in_index(page)
    overview = index_overview(store.index_path.read_text())
    assert "operations/:" in overview and "- GPU" in overview
