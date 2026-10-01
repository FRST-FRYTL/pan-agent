"""M9 G2: explicit supersession from the gate (corrects + old statement). Own wording."""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.events.schema import CuratorAction, Destination, Lifetime, MemoryCandidate, MemoryClassification, MemoryType
from pan.index.fts import FtsIndex
from pan.memory.curator import DeterministicCurator, page_bullets
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


def cand(claims, *, subject: str, evidence: str = "observed", event: str = "01K0G2C0000000000000000001",
         supersedes=(), mtype: MemoryType = MemoryType.PROJECT_FACT) -> MemoryCandidate:
    claims = [claims] if isinstance(claims, str) else list(claims)
    cls = MemoryClassification(should_remember=True, relevance=0.9, importance=0.6, confidence=0.8,
                               lifetime=Lifetime.LONG_TERM, type=mtype, destination=Destination.WIKI)
    return MemoryCandidate(event_ids=[event], classification=cls, normalized_claims=claims,
                           retrieval_query=" ".join([subject, *claims]), id=f"{event}:{mtype.value}",
                           session_id=event[-4:], title=subject, tags=[], claim_evidence=[evidence] * len(claims),
                           rationale="llm-gate-v3: test",
                           subject=subject, supersedes=list(supersedes))


def live(page):
    return [b for _, b, _ in page_bullets(page.body)]


def test_member_swap_strikes_the_old_list_on_its_page(env):
    curator, apply = env
    old = "The hiking group for the June trip is four people: me, Sam, Ravi and Noor."
    apply(curator.curate(cand(old, subject="June hiking group")))
    new = "The hiking group for the June trip is four people: me, Sam, Ravi and Tove."
    cur = curator.curate(cand(new, subject="Tove", supersedes=[old], event="01K0G2C0000000000000000002"))
    assert cur.decision.action is CuratorAction.UPDATE and "supersedes 1 older claim" in cur.decision.rationale
    page = apply(cur)
    assert live(page) == [new]


def test_update_in_another_language_lands_on_the_corrected_page(env):
    curator, apply = env
    old = "The workshop printer is rated for 120 pages per minute."
    apply(curator.curate(cand(old, subject="workshop printer", mtype=MemoryType.ENVIRONMENT)))
    new = "Der Werkstattdrucker schafft jetzt nur noch 80 Seiten pro Minute."
    cur = curator.curate(cand(new, subject="Werkstattdrucker", supersedes=[old], mtype=MemoryType.ENVIRONMENT,
                              event="01K0G2C0000000000000000002"))
    assert cur.decision.action is CuratorAction.UPDATE and cur.path.endswith("workshop-printer.md")
    assert live(apply(cur)) == [new]


def test_withdrawal_strikes_the_plan(env):
    curator, apply = env
    old = "Project quillmap will drop Windows builds in release 4.0."
    apply(curator.curate(cand(old, subject="quillmap")))
    new = "Project quillmap keeps Windows builds for now; no new date."
    cur = curator.curate(cand(new, subject="quillmap Windows builds", supersedes=[old],
                              event="01K0G2C0000000000000000002"))
    assert live(apply(cur)) == [new]


def test_inferred_candidate_with_supersedes_keeps_the_old_behaviour(env):
    """Agent L1 replace (inferred): no explicit strike; value-based supersession only."""
    curator, apply = env
    old = "The hiking group for the June trip is four people: me, Sam, Ravi and Noor."
    apply(curator.curate(cand(old, subject="June hiking group")))
    new = "June hiking group: me, Sam, Ravi and Tove."
    cur = curator.curate(cand(new, subject="June hiking group", evidence="inferred", supersedes=[old],
                              event="01K0G2C0000000000000000002"))
    assert cur.decision.action is CuratorAction.IGNORE or old in live(cur.page)


def test_unrelated_old_text_strikes_nothing(env):
    curator, apply = env
    keep = "The lab freezer is set to minus 80 degrees."
    apply(curator.curate(cand(keep, subject="lab freezer", mtype=MemoryType.ENVIRONMENT)))
    cur = curator.curate(cand("The lab fridge is set to 4 degrees.", subject="lab fridge", mtype=MemoryType.ENVIRONMENT,
                              supersedes=["The lab fridge was set to 6 degrees."], event="01K0G2C0000000000000000002"))
    if cur.page is not None and cur.path.endswith("lab-freezer.md"):
        assert keep in live(cur.page)


def test_withdrawn_plan_on_a_decision_page_is_struck(env):
    curator, apply = env
    old = "Project quillmap will drop Windows builds in release 4.0."
    apply(curator.curate(cand(old, subject="quillmap Windows builds", mtype=MemoryType.DECISION)))
    new = "Project quillmap keeps Windows builds for now."
    cur = curator.curate(cand(new, subject="quillmap", supersedes=[old], event="01K0G2C0000000000000000002"))
    assert cur.path.startswith("decisions/") and old not in live(apply(cur))


def test_same_fact_stated_on_two_days_is_present_and_items_stay_separate(env):
    curator, apply = env
    apply(curator.curate(cand("The user owns three kayaks. (stated 2024-03-01)", subject="kayaks")))
    again = curator.curate(cand("The user owns three kayaks. (stated 2024-03-20)", subject="kayaks",
                                event="01K0G2C0000000000000000002"))
    assert again.decision.action is CuratorAction.IGNORE
    apply(curator.curate(cand("The user bought 4 paddles for the club. (stated 2024-03-02)", subject="club paddles",
                              event="01K0G2C0000000000000000003")))
    cur = curator.curate(cand("The user spent 180 euros on the club paddles. (stated 2024-03-09)", subject="club paddles",
                              event="01K0G2C0000000000000000004"))
    page = apply(cur)
    assert any("4 paddles" in b for b in live(page)) and any("180 euros" in b for b in live(page))


def test_distinct_items_that_differ_by_one_word_are_both_kept(env):
    curator, apply = env
    apply(curator.curate(cand("The user bought a mountain bike. (stated 2024-03-01)", subject="bikes")))
    cur = curator.curate(cand("The user bought a road bike. (stated 2024-03-15)", subject="bikes",
                              event="01K0G2C0000000000000000002"))
    assert cur.decision.action is CuratorAction.UPDATE
    page = apply(cur)
    assert any("mountain bike" in b for b in live(page)) and any("road bike" in b for b in live(page))
    again = curator.curate(cand("The user bought a new road bike. (stated 2024-03-20)", subject="bikes",
                                event="01K0G2C0000000000000000003"))
    assert again.decision.action is CuratorAction.IGNORE


def test_page_dates_and_stamps_are_no_counts(env):
    """The provenance line and update heading carry the write date (the 25th here), a "(stated …)" stamp
    the day a fact was said: neither makes "25 chapters" present."""
    curator, apply = env
    apply(curator.curate(cand("The user has read 12 chapters of the pottery handbook. (stated 2024-03-25)",
                              subject="pottery handbook")))
    apply(curator.curate(cand("The user keeps the pottery handbook in the studio. (stated 2024-03-26)",
                              subject="pottery handbook", event="01K0G2C0000000000000000002")))
    cur = curator.curate(cand("The user has read 25 chapters of the pottery handbook. (stated 2024-04-02)",
                              subject="pottery handbook", event="01K0G2C0000000000000000003"))
    assert cur.decision.action is CuratorAction.UPDATE
    assert any("25 chapters" in b for b in live(apply(cur)))


def test_a_stamp_still_says_when_a_fact_was_stated(env):
    curator, apply = env
    apply(curator.curate(cand("The choir moves to the new rehearsal hall on October 3. (stated 2024-09-01)",
                              subject="choir rehearsal hall")))
    again = curator.curate(cand("The choir moves to the new rehearsal hall on October 3, 2024.",
                                subject="choir rehearsal hall", event="01K0G2C0000000000000000002"))
    assert again.decision.action is CuratorAction.IGNORE
