"""WikiStore: init skeleton, write/get, validation, templates; git helper."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import FIXTURES
from pan.wiki.frontmatter import Page
from pan.wiki.git import GitRepo, git_available
from pan.wiki.store import WIKI_DIRS, WikiError, WikiStore, new_page

needs_git = pytest.mark.skipif(not git_available(), reason="git not installed")


@needs_git
def test_init_creates_skeleton_and_repo(tmp_path: Path):
    store = WikiStore(tmp_path / "wiki")
    assert store.init() is True
    assert store.exists()
    for d in WIKI_DIRS:
        assert (store.root / d).is_dir()
    assert store.git.is_repo() and store.git.head()
    assert not store.git.is_dirty()
    assert "pan-memoryd <pan-memoryd@localhost>" in store.git.show()
    assert store.validate() == []
    assert store.init() is False  # idempotent
    assert len(store.git.log()) == 1


def test_init_without_git(tmp_path: Path):
    store = WikiStore(tmp_path / "wiki")
    store.init(git=False)
    assert store.exists() and not store.git.is_repo()


def test_write_and_get_by_id_and_path(tmp_path: Path):
    store = WikiStore(tmp_path)
    store.init(git=False)
    page = new_page("learning", "Docker needs --gpus all", sources=["session:s1"], tags=["docker"],
                    date="2026-09-23")
    rel = store.write(page)
    assert rel == "learnings/docker-needs-gpus-all.md"
    assert not list((tmp_path / "learnings").glob(".*.tmp"))  # atomic write left no temp file
    by_id = store.get("learnings.docker-needs-gpus-all")
    assert by_id is not None and by_id.path == rel and by_id.title == "Docker needs --gpus all"
    for ref in (rel, rel[:-3], "wiki/" + rel):
        assert store.get(ref).id == page.id
    assert "## Observation" in by_id.body  # type template
    assert store.get("nope") is None
    assert store.get("../../etc/passwd") is None
    assert [p.id for p in store.list_pages()] == ["index", page.id]


def test_write_rejects_invalid_and_duplicate(tmp_path: Path):
    store = WikiStore(tmp_path)
    store.init(git=False)
    bad = new_page("learning", "x")
    bad.meta["confidence"] = "certain"
    with pytest.raises(WikiError, match="invalid confidence"):
        store.write(bad)
    store.write(new_page("learning", "One", page_id="dup.id"))
    with pytest.raises(WikiError, match="already used"):
        store.write(new_page("learning", "Two", page_id="dup.id"))
    with pytest.raises(WikiError, match="invalid page path"):
        store.write(new_page("learning", "Esc", path="../outside.md"))


def test_rewrite_same_page_is_allowed(tmp_path: Path):
    store = WikiStore(tmp_path)
    store.init(git=False)
    page = new_page("system", "Postgres")
    store.write(page)
    page.body += "\nMore.\n"
    store.write(page)
    assert "More." in store.get(page.id).body


def test_new_decision_numbering(tmp_path: Path):
    store = WikiStore(tmp_path)
    store.init(git=False)
    first = store.new_decision("Use FTS5")
    assert first.path == "decisions/ADR-001-use-fts5.md" and first.id == "decisions.adr-001"
    store.write(first)
    second = store.new_decision("Second one")
    assert second.path.startswith("decisions/ADR-002-") and "## Decision" in second.body


def test_seed_wiki_is_valid(seed_wiki: Path):
    assert WikiStore(seed_wiki).validate() == []


def test_validate_reports_errors(seed_wiki: Path):
    store = WikiStore(seed_wiki)
    (seed_wiki / "learnings" / "broken.md").write_text("no frontmatter here\n")
    dup = store.get("systems.langfuse")
    (seed_wiki / "systems" / "langfuse-copy.md").write_text(dup.to_text())
    page = store.get("operations.restart-vllm")
    page.meta["related"] = ["systems/missing.md"]
    page.meta["status"] = "finished"
    page.body += "\nSee [gone](../nowhere.md) and [web](https://example.org) and [id](#rollback).\n"
    (seed_wiki / page.path).write_text(page.to_text())
    messages = [str(i) for i in store.validate()]
    assert any("broken.md" in m and "frontmatter" in m for m in messages)
    assert any("duplicate id 'systems.langfuse'" in m for m in messages)
    assert any("related entry does not resolve: systems/missing.md" in m for m in messages)
    assert any("broken link: ../nowhere.md" in m for m in messages)
    assert any("invalid status 'finished'" in m for m in messages)
    assert not any("example.org" in m or "#rollback" in m for m in messages)
    assert len(store.list_pages()) == 8  # broken page skipped, duplicate still listed


def test_related_accepts_page_ids(seed_wiki: Path):
    store = WikiStore(seed_wiki)
    page = store.get("systems.langfuse")
    page.meta["related"] = ["systems.dgx-spark-host"]
    store.write(page)
    assert store.validate() == []


# -- git helper -----------------------------------------------------------------------------------

@needs_git
def test_git_commit_only_given_paths(tmp_path: Path):
    repo = GitRepo(tmp_path / "r")
    assert repo.init() is True and repo.head() is None
    assert repo.diff() == ""
    (repo.root / "a.md").write_text("a\n")
    (repo.root / "b.md").write_text("b\n")
    assert repo.dirty_paths() == ["a.md", "b.md"]
    sha = repo.commit(["a.md"], "add a")
    assert sha and repo.head() == sha
    assert repo.dirty_paths() == ["b.md"]           # untouched file left alone
    assert not repo.is_dirty(["a.md"]) and repo.is_dirty(["b.md"])
    assert repo.commit(["a.md"], "nothing") is None  # no changes → no commit
    assert "pan-memoryd <pan-memoryd@localhost>" in repo.show(sha)


@needs_git
def test_git_edit_delete_diff_and_author(tmp_path: Path):
    repo = GitRepo(tmp_path / "r")
    repo.init()
    (repo.root / "d").mkdir()
    (repo.root / "d" / "p.md").write_text("one\n")
    repo.commit(["d/p.md"], "create")
    (repo.root / "d" / "p.md").write_text("two\n")
    assert repo.is_dirty(["d"])
    assert "+two" in repo.diff(paths=["d/p.md"])
    repo.commit(["d/p.md"], "update", author="Curator Test <curator@localhost>")
    assert "Curator Test <curator@localhost>" in repo.show()
    (repo.root / "d" / "p.md").unlink()
    sha = repo.commit(["d/p.md"], "delete")
    assert sha and not repo.is_dirty()
    assert [line.split(" ", 1)[1] for line in repo.log()] == ["delete", "update", "create"]


@needs_git
def test_store_write_then_commit(tmp_path: Path):
    store = WikiStore(tmp_path / "wiki")
    store.init()
    rel = store.write(new_page("incident", "GPU hang", date="2026-09-23"))
    assert store.git.dirty_paths() == [rel]
    assert store.git.commit([rel], "curator: CREATE incidents/gpu-hang")
    assert not store.git.is_dirty()


def test_page_copy_from_fixture(tmp_path: Path):
    """Pages loaded from disk keep their exact serialization (minimal git diffs)."""
    src = FIXTURES / "wiki" / "systems" / "langfuse.md"
    page = Page.from_text(src.read_text(), path="systems/langfuse.md")
    store = WikiStore(tmp_path)
    shutil.copy(src, tmp_path / "x.md")
    assert page.to_text() == src.read_text()
    assert store.get("x.md").id == "systems.langfuse"
