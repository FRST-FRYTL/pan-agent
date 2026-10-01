"""L1Writer against the pinned Hermes MemoryStore (integration spec §4.9, contract test 5)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pan.hermes import compat
from pan.memory.l1 import ADDED, DUPLICATE, ERROR, L1_FULL, L1Writer

pytestmark = pytest.mark.contract

ENTRY = 'User preference (own words): "Write meeting notes as Markdown files in the repo."'


def _user_md(home: Path) -> str:
    return (home / "memories" / "USER.md").read_text(encoding="utf-8")


def test_add_goes_through_hermes_store_without_drift(hermes_home):
    writer = L1Writer(hermes_home)
    before = _user_md(hermes_home)
    assert writer.add("user", ENTRY).status == ADDED
    text = _user_md(hermes_home)
    assert text.startswith(before) and ENTRY in text  # add-only: the user's entry stays first
    agent_store = compat.load_l1_store()  # a running agent's store
    result = agent_store.add("user", "Works in Europe/Berlin.")
    assert result.get("success"), f"agent store saw drift after PAN's write: {result}"
    assert not list((hermes_home / "memories").glob("*.bak.*"))
    assert writer.entries("user") == ["Prefers concise answers with concrete next steps.", ENTRY,
                                      "Works in Europe/Berlin."]


def test_duplicates_are_not_added(hermes_home):
    writer = L1Writer(hermes_home)
    assert writer.add("user", ENTRY).status == ADDED
    assert writer.add("user", ENTRY).status == DUPLICATE
    # same claim with different punctuation/case → also a duplicate
    assert writer.add("user", ENTRY.lower().replace('"', "")).status == DUPLICATE
    assert _user_md(hermes_home).count("Write meeting notes") == 1


def test_full_l1_returns_l1_full_and_writes_nothing(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "memory:\n  provider: pan\n  memory_char_limit: 2200\n  user_char_limit: 80\n")
    writer = L1Writer(hermes_home)
    before = _user_md(hermes_home)
    result = writer.add("user", ENTRY)
    assert result.status == L1_FULL and "/80" in result.message
    assert _user_md(hermes_home) == before


def test_scope_sets_and_restores_hermes_home(hermes_home, tmp_path, monkeypatch):
    other = tmp_path / "other"
    (other / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "unrelated"))
    assert L1Writer(other).add("memory", "PAN wiki lives in pan/wiki.").status == ADDED
    assert "PAN wiki lives" in (other / "memories" / "MEMORY.md").read_text()
    assert os.environ["HERMES_HOME"] == str(tmp_path / "unrelated")
    assert not (tmp_path / "unrelated").exists()


def test_errors_are_returned_not_raised(hermes_home):
    writer = L1Writer(hermes_home)
    assert writer.add("nope", "x").status == ERROR
    assert writer.add("user", "   ").status == ERROR
    broken = L1Writer(hermes_home, store_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    result = broken.add("user", ENTRY)
    assert result.status == ERROR and "boom" in result.message


def test_preference_already_saved_by_the_agent_is_a_duplicate(hermes_home):
    """M4 live run: Qwen3.8 saved the preference itself via Hermes' memory tool, in its own phrasing."""
    assert compat.load_l1_store().add("user", "Prefers all dates written in ISO 8601 format (YYYY-MM-DD).")["success"]
    writer = L1Writer(hermes_home)
    entry = 'User preference (own words): "Always write dates in ISO 8601 format (YYYY-MM-DD)."'
    assert writer.add("user", entry).status == DUPLICATE
    assert writer.add("user", 'User preference (own words): "Always write dates as DD.MM.YYYY."').status == ADDED


def test_claim_saved_in_the_other_l1_file_is_a_duplicate(hermes_home):
    """M4 live run: the agent saved the user's preference into MEMORY.md (target "memory")."""
    assert compat.load_l1_store().add("memory", "Always write dates in ISO 8601 format (YYYY-MM-DD).")["success"]
    writer = L1Writer(hermes_home)
    before = _user_md(hermes_home)
    result = writer.add("user", 'User preference (own words): "Always write dates in ISO 8601 format (YYYY-MM-DD)."')
    assert result.status == DUPLICATE and _user_md(hermes_home) == before
