"""Rules iteration R2 (m6/rules-r2): language-level cues, EN + DE, precision guards. Own wording."""

from __future__ import annotations

from itertools import count

import pytest

from pan.events.schema import Actor, AgentEvent, EventType, MemoryType
from pan.memory.classifier import RuleClassifier
from pan.memory.episodes import Episode

clf = RuleClassifier()
_n = count(1)


def _cands(user: str, assistant: str = "OK.", previous: str = ""):
    def ep(text):
        ev = AgentEvent(id=f"01K0R2X{next(_n):019d}", ts="2026-09-24T10:00:00Z", event_type=EventType.TURN,
                        session_id="s", actor=Actor.MAIN_AGENT, content={"user": text, "assistant": assistant})
        return Episode("s", [ev], closed_by="turn")
    return clf.classify(ep(user), ep(previous) if previous else None)


def _types(user: str, **kw):
    return [c.classification.type for c in _cands(user, **kw)]


def _claims(user: str, **kw):
    return [x for c in _cands(user, **kw) for x in c.normalized_claims]


# -- R2-1 attribution ------------------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "Note for later: our team's cache runs on port 6390.",
    "My laptop's SSH daemon listens on port 2222.",
    "We moved the metrics server to port 9101.",
    "Zur Info: unser Build-Server läuft auf Port 8088.",
])
def test_first_party_facts_pass(user):
    assert _types(user) == [MemoryType.ENVIRONMENT]


@pytest.mark.parametrize("user", [
    "My colleague's project uses the Redis server on port 6380.",
    "My friend Tom's NAS server is on host 192.0.2.20.",
    "His server listens on port 8081.",
    "Their staging API runs on port 9000.",
    "Note for later: Jana's team (not ours) runs its API on port 7300.",
    "My colleague said the build server runs on port 8089.",
    "Mein Kollege betreibt seinen Server auf Port 6380.",
    "Bei meinem Freund läuft der Server auf Port 8080.",
])
def test_third_party_attribution_blocks_capture(user):
    assert _types(user) == [MemoryType.NOISE]


# -- R2-2 hypotheticals ----------------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "Hypothetically, a second build server would cost 120 euros a month on port 9000.",
    "In theory we could run the cache server on port 6390, but nothing is decided.",
    "Angenommen, wir ziehen um, dann würde der Server auf Port 8443 laufen.",
    "Theoretisch würde ein zweiter Server 90 Euro im Monat kosten.",
])
def test_hypotheticals_are_noise(user):
    assert _types(user) == [MemoryType.NOISE]


# -- R2-3 asks mid-sentence ------------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "Take a look at deploy/web.yaml and tell me which port nginx uses.",
    "Open /etc/hosts and let me know whether db01 is listed.",
    "Schau dir mal deploy/web.yaml an und sag mir, welchen Port nginx nutzt.",
])
def test_asks_are_not_facts(user):
    assert _types(user) == [MemoryType.NOISE]


# -- R2-4 typed values inside tasks ------------------------------------------------------------------

@pytest.mark.parametrize("user,needle", [
    ("Write a changelog line for this: the scraper now waits 30 seconds between pages.", "30 seconds"),
    ("Formulier mir eine Commit-Nachricht: der Exporter nutzt jetzt Port 9464.", "9464"),
    ("Draft a reply to Anna. The payroll run is on 2026-10-28.", "2026-10-28"),
])
def test_typed_values_in_passing_are_facts(user, needle):
    assert any(needle in c for c in _claims(user))


@pytest.mark.parametrize("user", ["Oops, typo, it's 2026-11-03.", "Is the exporter on port 9464?",
                                  "Das ist 2026-11-03, glaube ich."])
def test_subjectless_or_question_values_are_noise(user):
    assert _types(user) == [MemoryType.NOISE]


# -- R2-5 names --------------------------------------------------------------------------------------

def test_named_machine_and_login_inside_a_task_are_captured():
    claims = _claims("Give me an ssh alias for our build box. Just the line. The box is osprey-ci and I log in as deploy-7.")
    assert any("osprey-ci" in c and "deploy-7" in c for c in claims)


# -- R2-6 notes without typed values --------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "Remember that: the spare office keys are in the blue drawer at reception.",
    "FYI the quarterly review happens in the large meeting room.",
    "Merk dir, dass die Ersatzschlüssel in der blauen Schublade am Empfang liegen.",
    "Zur Info: das Quartalsreview findet im großen Besprechungsraum statt.",
])
def test_explicit_notes_are_kept(user):
    assert _types(user) == [MemoryType.ENVIRONMENT]


@pytest.mark.parametrize("user", [
    "Remember the release checklist for later.",
    "By the way, the coffee here is great today.",
    "Übrigens, das Wetter ist heute schön.",
    "FYI, maybe the review moves to the small room.",
])
def test_non_notes_stay_noise(user):
    assert _types(user) == [MemoryType.NOISE]


# -- R2-7 restated statement with a new value supersedes ------------------------------------------------

from pan.memory.facts import shape_update  # noqa: E402


@pytest.mark.parametrize("new,old", [
    ("The design review now happens in room C on Tuesdays.", "The design review happens in room B on Tuesdays."),
    ("Das Billing-API betreut jetzt Omar Haddad.", "Das Billing-API betreut Priya Nair."),
])
def test_shape_update(new, old):
    assert shape_update(new, old) is not None


@pytest.mark.parametrize("new,old", [
    ("The design review happens in room B and C on Tuesdays.", "The design review happens in room B on Tuesdays."),
    ("The standup happens in room B on Tuesdays.", "The design review happens in room B on Tuesdays."),
    ("The design review does not happen in room C on Tuesdays.", "The design review happens in room B on Tuesdays."),
])
def test_not_a_shape_update(new, old):
    assert shape_update(new, old) is None


# -- R2-9 no topic nouns needed ---------------------------------------------------------------------------

@pytest.mark.parametrize("user", [
    "The weekly sync is at 09:30.",
    "Unser Archiv-Laufwerk heißt vault-3 und hängt an /mnt/archiv.",
    "FYI the design review happens in room B.",
])
def test_capture_does_not_depend_on_topic_nouns(user):
    assert _types(user) == [MemoryType.ENVIRONMENT]


# -- R2 restatements that invent values -------------------------------------------------------------------

@pytest.mark.parametrize("user,assistant,bad", [
    ("FYI, cache.conf is outdated: we moved the cache server to port 6400 yesterday.",
     "Got it — the cache server is on port 6400 and cache.conf (still showing 6399) needs updating.", "6399"),
    ("Zur Info: der Cache-Server läuft jetzt auf Port 6400.",
     "Notiert — der Cache-Server läuft auf Port 6400, vorher war es Port 6390.", "6390"),
])
def test_restatements_that_invent_values_are_dropped(user, assistant, bad):
    assert not any(bad in c for c in _claims(user, assistant=assistant))
