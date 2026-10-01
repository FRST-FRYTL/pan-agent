"""L1Writer.reconcile against the pinned Hermes MemoryStore (M5, D1: PAN owns automatic L1 writes).

In a benchmark run: the agent saved "vLLM server runs on port 8000." with its memory tool;
the user later moved the server to 8010; the wiki was updated but the stale L1 entry stayed and the
model answered 8000. PAN now supersedes such entries through ``MemoryStore.replace``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.hermes import compat
from pan.memory.l1 import KEPT_NEWER, L1_FULL, REMOVED, SUPERSEDED, L1Writer

pytestmark = pytest.mark.contract

STALE = "vLLM server runs on port 8000."
UPDATE = "We moved the vLLM server to port 8010."


def _memory_md(home: Path) -> str:
    return (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")


def _agent_add(target: str, entry: str) -> None:
    assert compat.load_l1_store().add(target, entry)["success"]


def test_stale_fact_is_replaced_in_its_own_phrasing_without_drift(hermes_home):
    _agent_add("memory", STALE)
    writer = L1Writer(hermes_home)
    (result,) = writer.reconcile(UPDATE)
    assert result.status == SUPERSEDED and result.entry == STALE
    assert result.replacement == "vLLM server runs on port 8010."
    entries = writer.entries("memory")
    assert "vLLM server runs on port 8010." in entries and STALE not in entries
    assert entries[0].startswith("Primary inference runs on a local DGX Spark")  # other entries untouched, in order
    # a running agent's store still accepts writes (no external drift) and sees the new entry
    agent = compat.load_l1_store()
    assert agent.add("memory", "Langfuse runs on spark01:3300.")["success"]
    assert not list((hermes_home / "memories").glob("*.bak.*"))
    assert writer.reconcile(UPDATE) == []  # idempotent


def test_preferences_and_unrelated_entries_are_never_touched(hermes_home):
    _agent_add("user", "Prefers the vLLM server on port 8000 for demos.")
    _agent_add("memory", "Langfuse runs on host spark01 at port 3300.")
    before_user = (hermes_home / "memories" / "USER.md").read_text()
    before_memory = _memory_md(hermes_home)
    assert L1Writer(hermes_home).reconcile(UPDATE) == []
    assert (hermes_home / "memories" / "USER.md").read_text() == before_user
    assert _memory_md(hermes_home) == before_memory


def test_stale_entry_is_removed_when_the_new_fact_is_already_there(hermes_home):
    _agent_add("memory", STALE)
    _agent_add("memory", "vLLM server runs on port 8010.")  # the agent added instead of replacing
    (result,) = L1Writer(hermes_home).reconcile(UPDATE)
    assert result.status == REMOVED
    entries = L1Writer(hermes_home).entries("memory")
    assert STALE not in entries and entries.count("vLLM server runs on port 8010.") == 1


def test_entry_written_after_the_observation_is_kept(hermes_home):
    _agent_add("memory", STALE)
    (result,) = L1Writer(hermes_home).reconcile(UPDATE, newer=lambda entry: entry == STALE)
    assert result.status == KEPT_NEWER and STALE in _memory_md(hermes_home)


def test_replacement_that_does_not_fit_is_reported_not_written(hermes_home):
    _agent_add("memory", "vLLM on ports 8000 and port 8001.")  # two old values: no in-place rewrite
    before = _memory_md(hermes_home)
    (hermes_home / "config.yaml").write_text(
        f"memory:\n  provider: pan\n  memory_char_limit: {len(before) + 2}\n  user_char_limit: 1375\n")
    (result,) = L1Writer(hermes_home).reconcile("We moved the vLLM server to port 8010 this morning.")
    assert result.status == L1_FULL and "/" in result.message
    assert _memory_md(hermes_home) == before


def test_qualified_agent_phrasing_is_reconciled(hermes_home):
    """M6, a benchmark run: the agent wrote "PAN vLLM server …" (context leaked into its
    phrasing); the Jaccard subject rule alone ({vllm} vs {pan, vllm}) missed it."""
    _agent_add("memory", "PAN vLLM server runs on port 8000.")
    (result,) = L1Writer(hermes_home).reconcile(UPDATE)
    assert result.status == SUPERSEDED and result.replacement == "PAN vLLM server runs on port 8010."


def test_explicit_supersedes_replaces_an_entry_without_a_common_value(hermes_home):
    """T3: the gate names the old statement (``corrects`` + ``old``) that the agent saved in its own
    words; no typed value links old and new (another attribute word, a German amount), so only the
    explicit link can keep L1 in step with the wiki."""
    stale = "Bike club: Abrechnung mit 0,40 € pro Stunde."
    _agent_add("user", stale)
    writer = L1Writer(hermes_home)
    new = "Der Stundenpreis des Fahrradclubs beträgt jetzt 0,35 € pro Stunde."
    assert writer.reconcile(new) == []          # the value path finds no common attribute
    (result,) = writer.reconcile_explicit([stale], [new, "Die Änderung gilt ab Oktober."])
    assert result.status == SUPERSEDED and result.entry == stale and result.replacement == new
    assert stale not in writer.entries("user") and new in writer.entries("user")
    # a preference entry is never touched, even when named
    pref = "Prefers invoices as PDF, never as Word files."
    _agent_add("user", pref)
    assert writer.reconcile_explicit([pref], ["Invoices go out as Word files now."]) == []


def test_explicit_supersedes_keeps_a_newer_entry(hermes_home):
    stale = "The shared drive is mounted at /mnt/team."
    _agent_add("memory", stale)
    (result,) = L1Writer(hermes_home).reconcile_explicit([stale], ["The shared drive is mounted at /srv/team."],
                                                         newer=lambda entry: True)
    assert result.status == KEPT_NEWER and stale in L1Writer(hermes_home).entries("memory")


def test_explicit_supersedes_leaves_multi_fact_and_current_entries_alone(hermes_home):
    """T3 replay: an entry that holds the old statement *next to other facts*, or already records the
    change, is not replaced by one new claim (that dropped still-valid facts)."""
    old = "The lab's door code is 4172."
    multi = "Lab: my advisor is Dr. Ines Vogt, the committee meets in May, and the lab's door code is 4172."
    current = "The lab's door code is 9035 (was 4172)."
    _agent_add("memory", multi)
    _agent_add("memory", current)
    writer = L1Writer(hermes_home)
    assert writer.reconcile_explicit([old], ["The lab's door code is 9035."]) == []
    assert multi in writer.entries("memory") and current in writer.entries("memory")


def test_explicit_supersedes_replaces_only_the_old_sentence_of_a_multi_sentence_entry(hermes_home):
    """An entry with the old statement as one of its sentences: that sentence is replaced, the other
    sentences stay word for word (replacing the whole entry dropped them, skipping it kept L1 stale)."""
    two = "Ticketing: moving to Zammad 6 (approved in May). Jira stays the tracker for now."
    _agent_add("memory", two)
    writer = L1Writer(hermes_home)
    (result,) = writer.reconcile_explicit(["Ticketing: moving to Zammad 6 (approved in May)."],
                                          ["The Zammad 6 move is paused."])
    assert result.status == SUPERSEDED
    assert result.replacement == "The Zammad 6 move is paused. Jira stays the tracker for now."
    assert two not in writer.entries("memory") and result.replacement in writer.entries("memory")
    # German, three sentences, the old one in the middle; a newer entry is still kept
    de = "Der Nutzer wohnt in Kassel.\nSein Auto ist ein blauer Golf.\nEr fährt jeden Tag zur Arbeit."
    _agent_add("user", de)
    (kept,) = writer.reconcile_explicit(["Das Auto des Nutzers ist ein blauer Golf."],
                                        ["Das Auto des Nutzers ist jetzt ein roter Passat."], newer=lambda e: True)
    assert kept.status == KEPT_NEWER and de in writer.entries("user")
    (result,) = writer.reconcile_explicit(["Sein Auto ist ein blauer Golf."],
                                          ["Das Auto des Nutzers ist jetzt ein roter Passat."])
    assert result.replacement == ("Der Nutzer wohnt in Kassel.\nDas Auto des Nutzers ist jetzt ein roter Passat.\n"
                                  "Er fährt jeden Tag zur Arbeit.")


def test_measure_of_another_subject_does_not_rewrite_an_entry(hermes_home):
    """A sentence-final number is a measure now; L1 reconcile needs the same subject for measures."""
    entry = "In the club team, Jonas Hartl wears jersey number 9."
    _agent_add("memory", entry)
    assert L1Writer(hermes_home).reconcile("In the club team, Mika Brandt wears jersey number 17.") == []
    assert entry in L1Writer(hermes_home).entries("memory")
