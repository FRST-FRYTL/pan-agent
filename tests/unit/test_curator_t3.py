"""T3: supersession only on an explicit gate ``supersedes`` or a typed-value contradiction of the same
attribute — never between sibling items on one page — and a config reading that does update a stated
value. Own wording (ADR-009: no benchmark text)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.events.schema import CuratorAction, Destination, Lifetime, MemoryCandidate, MemoryClassification, MemoryType
from pan.index.fts import FtsIndex
from pan.memory.curator import DeterministicCurator, page_bullets, supersede_bullets
from pan.memory.facts import contradiction, values
from pan.memory.retrieval import FtsRetriever
from pan.wiki.store import INDEX_TEMPLATE, WikiStore


@pytest.fixture
def env(tmp_path: Path):
    root = tmp_path / "wiki"
    store = WikiStore(root)
    store.init(git=False)
    store.index_path.write_text(INDEX_TEMPLATE.format(today="2026-09-23"))
    index = FtsIndex(tmp_path / "index.db")
    index.rebuild(root)
    curator = DeterministicCurator(store, FtsRetriever(index), today=lambda: "2026-09-25")

    def apply(cur):
        store.write(cur.page)
        index.update([store.get_path(cur.path)])
        return store.get_path(cur.path)
    yield curator, apply
    index.close()


def cand(claims, *, subject: str, evidence="observed", event: str = "01K0T3C0000000000000000001",
         supersedes=(), mtype: MemoryType = MemoryType.PROJECT_FACT, gate: bool = True) -> MemoryCandidate:
    claims = [claims] if isinstance(claims, str) else list(claims)
    evidence = [evidence] * len(claims) if isinstance(evidence, str) else list(evidence)
    cls = MemoryClassification(should_remember=True, relevance=0.9, importance=0.6, confidence=0.8,
                               lifetime=Lifetime.LONG_TERM, type=mtype, destination=Destination.WIKI)
    return MemoryCandidate(event_ids=[event], classification=cls, normalized_claims=claims,
                           retrieval_query=" ".join([subject, *claims]), id=f"{event}:{mtype.value}",
                           session_id=event[-4:], title=subject, tags=[], claim_evidence=evidence,
                           rationale="llm-gate-v3: test" if gate else "rules-v2: test",
                           subject=subject, supersedes=list(supersedes))


def live(page):
    return [b for _, b, _ in page_bullets(page.body)]


# -- over-eager supersession (curator-only strikes the gate did not ask for) ------------------------------

BUDGET_A = "In an earlier chat the assistant advised budgeting about $70 for a set of garden tools."
BUDGET_B = "In an earlier chat the assistant advised budgeting $30 for bird seed for the winter."


def test_two_budgets_on_one_page_are_siblings_not_an_update(env):
    curator, apply = env
    apply(curator.curate(cand(BUDGET_A, subject="garden budget (earlier chat)")))
    cur = curator.curate(cand(BUDGET_B, subject="garden budget (earlier chat)", event="01K0T3C0000000000000000002"))
    assert cur.decision.action is CuratorAction.UPDATE and "supersedes" not in cur.decision.rationale
    page = apply(cur)
    assert set(live(page)) >= {BUDGET_A, BUDGET_B}


def test_measure_conflict_needs_the_same_subject_for_gate_claims():
    assert contradiction(BUDGET_B, BUDGET_A, on_page=True, subject="bird seed plan", strict_measures=True) is None
    # the same attribute of the same thing is still an update
    old, new = "The staging database pool size limit is 25.", "The staging database pool size limit is 40."
    hit = contradiction(new, old, on_page=True, subject="staging database", strict_measures=True)
    assert hit is not None and hit.rewritten == new


def test_verb_only_attribute_is_not_the_same_attribute():
    old = "Bed A grows basil, gets 45 minutes of water per cycle, starts at 06:00."
    new = "Bed C gets 20 minutes of water per cycle."
    body = f"## Facts\n\n- {old} (observed)\n"
    out, gone, _ = supersede_bullets(body, [new], "superseded", "Bed C", shape=False, strict=True)
    assert gone == [] and out == body


def test_explicit_gate_supersedes_still_strikes(env):
    curator, apply = env
    old = "The pottery class meets in room 4 on Thursdays."
    apply(curator.curate(cand(old, subject="pottery class")))
    new = "The pottery class now meets in the east studio on Thursdays."
    page = apply(curator.curate(cand(new, subject="pottery class", supersedes=[old], event="01K0T3C0000000000000000002")))
    assert live(page) == [new]


# -- missing supersession: a file / tool reading of the same attribute ------------------------------------

def test_config_assignment_is_a_name_value():
    assert values("In conf/editor.ini the editor has default_theme = Nord-2 for new windows.").slots["name"] == {"nord-2"}


def test_file_reading_supersedes_the_stated_value(env):
    curator, apply = env
    old = "In the user's studio, the editor's default theme for new windows is Solar-1."
    apply(curator.curate(cand(old, subject="editor", mtype=MemoryType.CONFIGURATION)))
    new = "In conf/editor.ini the editor has default_theme = Nord-2 for new windows."
    cur = curator.curate(cand(new, subject="editor", mtype=MemoryType.CONFIGURATION, event="01K0T3C0000000000000000002"))
    assert "supersedes 1 older claim" in cur.decision.rationale
    assert live(apply(cur)) == [new]


def test_restated_value_next_to_a_retraction_is_not_retracted():
    current = "The default font for invoices is Inter-400."
    note = "The default font for invoices is Inter-400 because Robo-300 was discontinued by the vendor."
    assert contradiction(note, current, on_page=True) is None
    # a real retraction of the same value still counts
    assert contradiction("We no longer run the vLLM server on port 8000.", "The vLLM server runs on port 8000.")


def test_sentence_final_number_and_german_attribute():
    slots = values("The club car has the plate number KX 902.").slots
    assert [v for k, v in slots.items() if k.startswith("measure:")] == [{"902"}]
    slots = values("Der Stundenpreis beträgt 12 Euro.").slots
    assert list(slots) == ["measure:stundenprei"]   # stemmed; "beträgt" is a skipped word, not "betr" + "gt"


# -- reported claims on the wiki (T3) -----------------------------------------------------------------------

REP = "Assistant reported: "


def test_reported_claim_never_strikes_an_observed_fact(env):
    curator, apply = env
    observed = "The reverse proxy listens on port 8443."
    apply(curator.curate(cand(observed, subject="reverse proxy", mtype=MemoryType.ENVIRONMENT)))
    report = REP + "The reverse proxy now listens on port 9443."
    cur = curator.curate(cand(report, subject="reverse proxy", evidence="reported", mtype=MemoryType.ENVIRONMENT,
                              event="01K0T3C0000000000000000002"))
    assert "supersedes" not in cur.decision.rationale and cur.decision.evidence == "reported"
    page = apply(cur)
    assert set(live(page)) == {observed, report}
    assert "(reported)" in page.body


def test_observed_claim_confirming_a_report_is_added(env):
    curator, apply = env
    report = REP + "The nightly export writes to /srv/exports."
    apply(curator.curate(cand(report, subject="nightly export", evidence="reported", mtype=MemoryType.ENVIRONMENT)))
    observed = "The nightly export writes to /srv/exports."
    cur = curator.curate(cand(observed, subject="nightly export", mtype=MemoryType.ENVIRONMENT,
                              event="01K0T3C0000000000000000002"))
    assert cur.decision.action is CuratorAction.UPDATE and observed in apply(cur).body
    # the same report again is already present
    again = curator.curate(cand(report, subject="nightly export", evidence="reported", mtype=MemoryType.ENVIRONMENT,
                                event="01K0T3C0000000000000000003"))
    assert again.decision.action is CuratorAction.IGNORE


def test_newer_report_replaces_an_older_report_only(env):
    curator, apply = env
    observed = "The build cache size limit is 20 GB."
    apply(curator.curate(cand([observed, REP + "The build cache eviction window is 7 days."],
                              evidence=["observed", "reported"], subject="build cache",
                              mtype=MemoryType.CONFIGURATION)))
    newer = REP + "The build cache eviction window is 3 days."
    page = apply(curator.curate(cand(newer, subject="build cache", evidence="reported", mtype=MemoryType.CONFIGURATION,
                                     event="01K0T3C0000000000000000002")))
    assert set(live(page)) == {observed, newer}


def test_observed_update_strikes_a_report(env):
    curator, apply = env
    report = REP + "The staging database pool size limit is 25."
    apply(curator.curate(cand(report, subject="staging database", evidence="reported", mtype=MemoryType.CONFIGURATION)))
    observed = "The staging database pool size limit is 40."
    page = apply(curator.curate(cand(observed, subject="staging database", mtype=MemoryType.CONFIGURATION,
                                     event="01K0T3C0000000000000000002")))
    assert live(page) == [observed]


# -- claims about different occasions are siblings; titles made from a claim carry no stamp -----------------

def test_another_dated_outing_is_not_an_update(env):
    curator, apply = env
    first = "The user picked 6 kg of plums at the Hollerhof orchard with Mia on 8/09. (stated 2026-08-12)"
    apply(curator.curate(cand(first, subject="plum picking")))
    second = "The user picked 9 kg of plums with Mia on 8/23. (stated 2026-08-25)"
    cur = curator.curate(cand(second, subject="plum picking", event="01K0T3C0000000000000000002"))
    assert "supersedes" not in cur.decision.rationale
    assert set(live(apply(cur))) == {first, second}


def test_different_occasion_needs_other_dates_and_another_difference():
    from pan.memory.curator import different_occasion
    assert different_occasion("The user ran 12 km on the trail on 2026-05-03.",
                              "The user ran 8 km on the trail on 2026-04-19.")
    assert different_occasion("The user stayed 3 nights on the trip to Lake Bled.",
                              "The user stayed 5 nights on the trip to Lake Ohrid.")
    assert not different_occasion("The pool fee went from 40 to 45 euros, due 6/19 now.", "The pool fee is 40 euros, due on 6/12.")
    # one appointment that moved, the same day restated, or no date at all: not a different occasion
    assert not different_occasion("The dentist check-up moved to May 14.", "The dentist check-up is on May 7.")
    assert not different_occasion("The user ran 12 km on 2026-05-03.", "The user ran 8 km on 2026-05-03.")
    assert not different_occasion("The rent is 950 euros.", "The rent is 900 euros.")


def test_page_title_and_path_from_a_claim_have_no_stamp(env):
    curator, _ = env
    claim = "The user adopted a ferret called Biscuit. (stated 2026-04-02)"
    c = cand(claim, subject="")
    c.title = claim
    cur = curator.curate(c)
    assert cur.decision.action is CuratorAction.CREATE
    assert "stated" not in cur.page.title and "stated" not in cur.path, (cur.page.title, cur.path)


def test_retraction_that_restates_only_the_name_still_retracts():
    old = "Rex's grooming appointment at Pawlish is on Friday."
    new = "Rex's Friday grooming appointment at Pawlish is cancelled; Rex is called Rex-2 on the booking now."
    assert contradiction(new, old, on_page=True) is not None
