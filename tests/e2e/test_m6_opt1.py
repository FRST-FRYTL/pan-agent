"""M6 opt-1: capture/supersession mechanisms found missing in a benchmark iteration.

Fixtures are worded differently from the bench scenarios on purpose (the bench is a test set):
German facts, personal profile facts, facts read from files/tool output, config values that
supersede an earlier user claim, enumerations, retractions, task-scoped "don't …" instructions.
Real Hermes MemoryStore (tmp HERMES_HOME), seed wiki, fixed clock, daemon run after each session.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import pytest

from conftest import fixed_clock
from pan.daemon.memoryd import MemoryDaemon
from pan.events.schema import Actor, AgentEvent, EventType, MemoryType
from pan.events.spool import EventSpool
from pan.hermes import compat
from pan.memory.classifier import RuleClassifier, split_enumeration
from pan.memory.episodes import Episode
from pan.memory.facts import contradiction, values
from pan.memory.l1 import L1Writer
from pan.memory.reader import Reader
from pan.paths import PanPaths

_ids = count(1)
clf = RuleClassifier()


def _id() -> str:
    return f"01K0M6O{next(_ids):019d}"


def _turn(session: str, user: str, assistant: str) -> dict:
    return AgentEvent(id=_id(), ts="2026-09-24T10:00:00Z", event_type=EventType.TURN, session_id=session,
                      actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant}).to_dict()


def _read(session: str, path: str, content: str, tool: str = "read_file") -> dict:
    numbered = "\n".join(f"{i}|{ln}" for i, ln in enumerate(content.splitlines(), 1))
    return AgentEvent(id=_id(), ts="2026-09-24T10:00:00Z", event_type=EventType.TOOL_CALL, session_id=session,
                      actor=Actor.TOOL, content={"tool": tool, "args": {"path": path}, "status": "ok",
                                                 "result_excerpt": json.dumps({"content": numbered})}).to_dict()


def _session(home: Path, events: list[dict]) -> None:
    paths = PanPaths.for_home(home)
    with EventSpool(paths.events_db) as spool:
        for raw in events:
            spool.append(AgentEvent.from_dict(raw))
    d = MemoryDaemon(paths, clock=fixed_clock, worker_id="test")
    d.prepare()
    try:
        d.drain()
    finally:
        d.close()


def _prefetch(home: Path, q: str) -> str:
    r = Reader(PanPaths.for_home(home))
    try:
        return r.prefetch(q)
    finally:
        r.close()


def _episode(*events: dict) -> Episode:
    return Episode("s", [AgentEvent.from_dict(e) for e in events], closed_by="turn")


# -- classifier -----------------------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "Kurze Info: Die Staging-Datenbank läuft auf db-stage-2 unter Port 5433.",
    "Nur zur Info, unser Build-Server heißt ci-nord und steht im Keller.",
])
def test_german_environment_facts_are_captured(user):
    (c,) = clf.classify(_episode(_turn("s", user, "Alles klar.")))
    assert c.classification.type is MemoryType.ENVIRONMENT and c.claim_evidence == ["observed"]


def test_german_preference_is_captured():
    (c,) = clf.classify(_episode(_turn("s", "Schreib mir künftig bitte immer Code-Kommentare auf Englisch.", "Okay.")))
    assert c.classification.type is MemoryType.USER_PREFERENCE


def test_personal_profile_facts_go_to_user_md():
    user = "Quick context for the future: my terminal is kitty and I use Emacs keybindings everywhere. What's a good font?"
    cands = clf.classify(_episode(_turn("s", user, "Try JetBrains Mono.")))
    assert [c.classification.type for c in cands] == [MemoryType.USER_PREFERENCE]
    assert "kitty" in cands[0].normalized_claims[0]


@pytest.mark.parametrize("user", ["Don't touch that script again.", "Never mind, don't run it for now.",
                                  "Do not print the whole file this time."])
def test_task_scoped_instructions_are_not_preferences(user):
    assert [c.classification.type for c in clf.classify(_episode(_turn("s", user, "OK.")))] == [MemoryType.NOISE]


def test_standing_dont_is_still_a_preference():
    (c,) = clf.classify(_episode(_turn("s", "Don't use emojis in commit messages.", "Understood.")))
    assert c.classification.type is MemoryType.USER_PREFERENCE


def test_enumeration_splits_into_one_claim_per_item():
    assert split_enumeration("the cache listens on 6380, the queue on 5673 and the UI on 8443") == [
        "The cache listens on 6380.", "The queue listens on 6380".replace("6380", "5673") + ".",
        "The UI listens on 8443."]
    assert split_enumeration("metrics now runs on 9101, and 9100 is free") == ["Metrics now runs on 9101."]
    assert split_enumeration("Our vLLM server runs on port 8000, which is fine.") == [
        "Our vLLM server runs on port 8000, which is fine."]


def test_tool_grounded_assistant_facts_are_observed():
    log = "\n".join([f"2026-09-20T01:{i:02d}:00Z INFO worker: heartbeat" for i in range(30)]
                    + ["2026-09-20T01:31:00Z INFO release: shipped image tag v7.3.1-hotfix to eu-west"])
    ep = _episode(_read("s", "var/release.log", log),
                  _turn("s", "Skim var/release.log: did the rollout go out?",
                        "Yes. The release shipped image tag v7.3.1-hotfix to eu-west at 01:31. "
                        "I think it might also have touched us-east."))
    (c,) = clf.classify(ep)
    assert c.normalized_claims == ["The release shipped image tag v7.3.1-hotfix to eu-west at 01:31."]
    assert c.claim_evidence == ["observed"] and c.title == "var/release.log"


def test_config_and_tiny_files_become_claims():
    ep = _episode(_read("s", "/work/x/workspace/settings/worker.ini", "[pool]\n# tuned 2026-08\nmax_workers = 12\nqueue = jobs\n"),
                  _read("s", "/work/x/workspace/.node-version", "20.11.1"),
                  _turn("s", "What do the worker settings and the node pin say?", "12 workers; Node 20.11.1."))
    (c,) = clf.classify(ep)
    assert "`settings/worker.ini`: max_workers = 12" in c.normalized_claims
    assert "`.node-version` contains `20.11.1`." in c.normalized_claims


# -- facts -------------------------------------------------------------------------------------------

def test_measure_values_and_their_contradictions():
    assert values("The upload size cap is 20 MB").slots  # size slot
    assert "measure:burst" in values("burst = 40").slots
    assert contradiction("`gateway.yaml`: request_timeout = 45", "Request timeout: 30 seconds.") is not None
    assert contradiction("billing retry limit is 5", "auth retry limit is 3") is None
    assert contradiction("The nightly job starts at 02:30.", "The nightly job starts at 01:00.") is not None


def test_retractions():
    old = "Our weekly retro is on Fridays at 16:00."
    assert contradiction("We dropped the weekly retro altogether.", old) is not None
    assert contradiction("We dropped the monthly demo.", old) is None


# -- daemon end-to-end ------------------------------------------------------------------------------

pytestmark = pytest.mark.e2e


def test_config_value_read_later_supersedes_the_users_earlier_number(pan_home):
    with compat.hermes_home_scope(pan_home):
        assert compat.load_l1_store().add("memory", "Upload size cap: 10 MB per file.")["success"]
    _session(pan_home, [_turn("a", "FYI: our upload size cap is 10 MB per file.", "Got it.")])
    _session(pan_home, [_read("b", "conf/uploads.yaml", "# bumped last week\nmax_upload_mb: 25\n"),
                        _turn("b", "Open conf/uploads.yaml: does it match the upload cap I mentioned?",
                              "No: the file sets max_upload_mb: 25, you said 10.")])
    memory = L1Writer(pan_home).entries("memory")
    assert not any("10 MB" in e for e in memory), memory
    out = _prefetch(pan_home, "What upload cap is configured at the moment?")
    assert "25" in out


def test_retraction_supersedes_the_old_fact_in_l1_and_wiki(pan_home):
    with compat.hermes_home_scope(pan_home):
        assert compat.load_l1_store().add("memory", "The weekly retro is on Fridays at 16:00.")["success"]
    _session(pan_home, [_turn("a", "Note for later: our weekly retro meeting is on Fridays at 16:00.", "Noted.")])
    _session(pan_home, [_turn("b", "Scrap that: we abolished the weekly retro meeting, feedback goes into a doc now.",
                              "Understood.")])
    memory = L1Writer(pan_home).entries("memory")
    assert not any("16:00" in e for e in memory), memory
    wiki = "\n".join(p.read_text() for p in PanPaths.for_home(pan_home).wiki.rglob("*.md"))
    assert "~~" in wiki and "abolished" in wiki


def test_enumeration_update_strikes_only_the_changed_item(pan_home):
    _session(pan_home, [_turn("a", "Service map: the cache listens on 6380, the queue on 5673 and the UI on 8443.",
                              "Noted.")])
    _session(pan_home, [_turn("b", "Heads-up: the queue now listens on 5680.", "OK.")])
    out = _prefetch(pan_home, "Which port is the queue on, and the UI?")
    assert "5680" in out and "5673" not in out
    wiki = "\n".join(p.read_text() for p in PanPaths.for_home(pan_home).wiki.rglob("*.md"))
    assert "- The UI listens on 8443." in wiki and "~~The cache" not in wiki


def test_german_fact_is_found_by_an_english_question(pan_home):
    _session(pan_home, [_turn("a", "Zur Info: Unser Grafana-Dashboard für die Latenzen heißt spark-latency und "
                                   "läuft auf monitor-01.", "Verstanden.")])
    out = _prefetch(pan_home, "What is our latency dashboard in Grafana called, and which host runs it?")
    assert "spark-latency" in out


def test_same_session_correction_rewrites_the_previous_fact():
    first = _turn("c", "Please note: the office move happens on November 3.", "Noted.")
    ep1 = _episode(first)
    ep2 = Episode("c", [AgentEvent.from_dict(_turn("c", "Oops, I misread the mail, it's November 10.", "Fixed."))],
                  closed_by="turn")
    (c,) = clf.classify(ep2, ep1)
    assert c.normalized_claims == ["The office move happens on November 10."]
    assert c.subject == "Office move"
    assert clf.classify(ep2)[0].classification.type is MemoryType.NOISE  # no previous fact: nothing to correct


@pytest.mark.parametrize("first,second,stale", [
    ("FYI: the design review happens in room B on Tuesdays.", "FYI: the design review now happens in room C on Tuesdays.",
     "room B"),
    ("Zur Info: das Billing-API betreut Priya Nair.", "Zur Info: das Billing-API betreut jetzt Omar Haddad.", "Priya"),
])
def test_new_value_supersedes_instead_of_sitting_next_to_the_old_bullet(pan_home, first, second, stale):
    _session(pan_home, [_turn("a", first, "OK.")])
    _session(pan_home, [_turn("b", second, "OK.")])
    live = [ln for p in PanPaths.for_home(pan_home).wiki.rglob("*.md") if p.name != "index.md"
            for ln in p.read_text().splitlines() if ln.startswith("- ") and "~~" not in ln]
    assert not any(stale in ln for ln in live), live


@pytest.mark.parametrize("first,second,stale", [
    ("Note for later: the vendor sync happens every Thursday at 15:00.", "Scratch the vendor sync, it has been cancelled.",
     "15:00"),
    ("Note: the design review is on Tuesdays in room B.", "Actually, never mind the design review, we dropped it.",
     "room B"),
    ("FYI: the relay runs on port 7020.", "Forget what I said about the relay port, it's actually 7045.", "7020"),
    ("Zur Info: das Team-Frühstück ist jeden Freitag um 9:00.", "Vergiss das Team-Frühstück, das wurde gestrichen.",
     "9:00"),
    ("Zur Info: das Relay läuft auf Port 7020.", "Korrektur: das Relay läuft eigentlich auf Port 7045.", "7020"),
])
def test_retractions_and_corrections_across_sessions_supersede(pan_home, first, second, stale):
    _session(pan_home, [_turn("a", first, "OK.")])
    _session(pan_home, [_turn("b", second, "OK.")])
    live = [ln for p in PanPaths.for_home(pan_home).wiki.rglob("*.md") if p.name != "index.md"
            for ln in p.read_text().splitlines() if ln.startswith("- ") and "~~" not in ln and stale in ln]
    assert not live, live


@pytest.mark.parametrize("second", [
    "Don't forget the vendor sync on Thursday at 15:00.",
    "Forget it, I'll write the summary myself.",
    "Vergiss nicht den Vendor-Sync am Donnerstag.",
])
def test_reminders_and_unrelated_forgets_retract_nothing(pan_home, second):
    _session(pan_home, [_turn("a", "Note for later: the vendor sync happens every Thursday at 15:00.", "OK.")])
    _session(pan_home, [_turn("b", second, "OK.")])
    live = [ln for p in PanPaths.for_home(pan_home).wiki.rglob("*.md") if p.name != "index.md"
            for ln in p.read_text().splitlines() if ln.startswith("- ") and "~~" not in ln and "15:00" in ln]
    assert live
