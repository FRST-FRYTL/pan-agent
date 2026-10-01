"""LLM-gate post-checks: whose fact it is (third party vs. the user's own), evidence labels of the
user's own words, hedges on the claim's own clause, requests that carry facts. Own wording."""

from __future__ import annotations

import pytest

from pan.events.schema import Actor, AgentEvent, EventType, MemoryType
from pan.memory.classifier import OBSERVED
from pan.memory.dates import bare
from pan.memory.episodes import Episode
from pan.memory.llm_gate import build_input, post_check


def _turn(n: int, user: str, assistant: str = "OK.", **extra) -> AgentEvent:
    return AgentEvent(id=f"01K0ATTR{n:018d}", ts="2026-09-24T10:00:00Z", event_type=EventType.TURN, session_id="s",
                      actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant, **extra})


def _check(user: str, *claims, kind: str = "fact", domain: str = "personal", recall: str = ""):
    ep = Episode("s", [_turn(1, user, recall=recall)] if recall else [_turn(1, user)], closed_by="turn")
    dec = {"why": "test", "record": True, "record_confidence": 0.9, "language": "en", "sensitivity": "none",
           "candidates": [{"kind": kind, "domain": domain, "lifetime": "long_term", "volatile": False,
                           "corrects": False, "importance": 0.7, "confidence": 0.8, "subject": "", "title": "",
                           "claims": [{"text": t, "evidence": e, "supersedes": ""} for t, e in claims]}]}
    notes: list = []
    cands = post_check(dec, ep, None, build_input(ep, None, ""), notes)
    kept = [bare(t) for c in cands if c.classification.type is not MemoryType.NOISE for t in c.normalized_claims]
    labels = [lab for c in cands if c.classification.type is not MemoryType.NOISE for lab in c.claim_evidence]
    return kept, labels, notes


# -- the user's own things, pets, clients and family are not third party ----------------------------------------

@pytest.mark.parametrize("user,claim", [
    # the gate writes "their" for "my"
    ("The oak bookshelf for my study was 340 euros.", "The user bought an oak bookshelf for their study for 340 euros."),
    ("Unser Hund Fips bekommt gegen seine Allergie jetzt zweimal täglich 5 mg Cetirizin.",
     "Der Hund des Nutzers, Fips, bekommt gegen seine Allergie zweimal täglich 5 mg Cetirizin."),
    # a client account's "their" is the company's, and the account is the user's work
    ("Had the renewal call with Norvik Freight. Their contract renews on 1 March at 48,000 dollars.",
     "Norvik Freight's contract renews on 1 March at 48,000 dollars."),
    # a rule laid on the user's own work or home
    ("The client's procurement team requires that every invoice we send carries a PO number.",
     "The client's procurement team requires a PO number on every invoice the user sends."),
    ("My manager wants me to send the capacity report every Friday by 16:00.",
     "The user's manager wants the user to send the capacity report every Friday by 16:00."),
    # a family member's fact the user acts on
    ("My mum's hip surgery is on 12 May, so I'm taking that week off.", "The user's mum has hip surgery on 12 May."),
    # reported speech about the user, and a conjunction before "said" is no speaker
    ("Dr. Brandt said my blood pressure is 128 over 82.", "Dr. Brandt said the user's blood pressure is 128 over 82."),
    ("The landlord checked the boiler in our flat and said it now runs at 1.5 bar.",
     "The boiler in the user's flat now runs at 1.5 bar."),
    # the user acts in the neighbouring sentence
    ("My cousin loved the birthday present! The 3 candles I poured for her came out great.",
     "The user poured 3 candles for a cousin's birthday present."),
])
def test_the_users_own_things_are_kept(user, claim):
    kept, labels, notes = _check(user, (claim, "user_stated"))
    assert kept == [claim] and labels == [OBSERVED], notes


@pytest.mark.parametrize("user,claim", [
    ("My sister-in-law's bakery opens at 6:30 every day.", "The bakery opens at 6:30 every day."),
    ("My sister-in-law's bakery runs its till on a tablet with 2 GB RAM, not my business really.",
     "The bakery till runs on a tablet with 2 GB RAM."),
    ("Mein Kollege Jonas hat einen Mini-PC mit 64 GB RAM.", "Jonas hat einen Mini-PC mit 64 GB RAM."),
    ("Our neighbour's heat pump runs at 55 degrees.", "The heat pump runs at 55 degrees."),
    ("My friend Ravi's Pi-hole blocks 30 percent of the traffic.", "The Pi-hole blocks 30 percent of the traffic."),
    ("Lena says her new flat has 3 rooms.", "Lena's new flat has 3 rooms."),
    ("My brother's boat has 2 engines, I just think it's too loud.", "The user's brother's boat has 2 engines."),
    ("Their team (not ours) deploys with 4 replicas.", "The team deploys with 4 replicas."),
])
def test_other_peoples_own_things_stay_out(user, claim):
    kept, _, notes = _check(user, (claim, "user_stated"))
    assert kept == [] and any("third-party" in n for n in notes), notes


# -- the user's own words under another evidence label ---------------------------------------------------------

@pytest.mark.parametrize("evidence", ["tool_observed", "user_confirmed", "assistant_inferred"])
def test_a_fact_in_this_turns_user_text_is_user_stated(evidence):
    kept, labels, notes = _check("I switched my gym membership to the 29 euro plan.",
                                 ("The user switched their gym membership to the 29 euro plan.", evidence))
    assert kept == ["The user switched their gym membership to the 29 euro plan."] and labels == [OBSERVED]
    assert any("→ user_stated" in n for n in notes)


def test_an_echo_of_memory_or_an_inferred_preference_is_still_dropped():
    recall = "The user's gym membership is the 29 euro plan."
    kept, _, _ = _check("Which gym plan am I on?", ("The user's gym membership is the 29 euro plan.",
                                                    "assistant_inferred"), recall=recall)
    assert kept == []
    kept, _, _ = _check("I switched my gym membership to the 29 euro plan.",
                        ("The user prefers the 29 euro gym plan.", "assistant_inferred"), kind="preference")
    assert kept == []


def test_a_restated_update_keeps_its_wording_when_the_values_are_the_users():
    recall = "The choir concert in St. Anna church on Linden Street starts at 20:00; doors open at 19:00."
    claim = "The choir concert in St. Anna church on Linden Street starts at 19:00."
    kept, labels, notes = _check("Small correction: the choir concert starts at 19:00, not 20:00.",
                                 (claim, "user_stated"), recall=recall)
    assert kept == [claim] and labels == [OBSERVED] and not any("own sentence" in n for n in notes)


# -- requests for help carry the user's facts; questions about the fact do not ------------------------------------

def test_a_request_for_help_carries_the_users_fact():
    kept, _, _ = _check("Can you suggest a filing system for my 212 vinyl singles?",
                        ("The user has 212 vinyl singles.", "user_stated"))
    assert kept == ["The user has 212 vinyl singles."]
    kept, _, _ = _check("Could you plan my reading list, considering my commute is about 35 minutes each way?",
                        ("The user's commute is about 35 minutes each way.", "user_stated"))
    assert kept == ["The user's commute is about 35 minutes each way."]


@pytest.mark.parametrize("user,claim", [
    ("Can you check whether our backup job still runs at 02:00?", "The backup job runs at 02:00."),
    ("Does my router use channel 11?", "The user's router uses channel 11."),
    ("Could you tell me which port our wiki uses, 8081?", "The wiki uses port 8081."),
])
def test_questions_about_the_fact_are_not_statements(user, claim):
    assert _check(user, (claim, "user_stated"))[0] == []


# -- hedges, proposals and wishes on the claim's own clause ----------------------------------------------------------

def test_a_plain_clause_next_to_a_hedged_one_is_kept():
    user = "The rail ticket was 85 euros, and I think I'll book the early train."
    kept, _, _ = _check(user, ("The rail ticket was 85 euros.", "user_stated"),
                        ("The user will book the early train.", "user_stated"))
    assert kept == ["The rail ticket was 85 euros."]
    user = "My uncle left his old drill with me, saying I'd probably use it more."
    kept, _, _ = _check(user, ("The user's uncle left his old drill with the user.", "user_stated"))
    assert kept == ["The user's uncle left his old drill with the user."]
    user = "My aunt asked if I could pick up the cake on the 9th instead."
    kept, _, _ = _check(user, ("The user picks up the cake on the 9th.", "user_stated"))
    assert kept == ["The user picks up the cake on the 9th."]


@pytest.mark.parametrize("user,claim,kept", [
    ("My neighbour gave me her old sewing machine, saying I'd probably use it more than she does.",
     "The user's neighbour gave the user her old sewing machine, saying the user would probably use it more.",
     "The user's neighbour gave the user her old sewing machine."),
    ("I sold the rowing machine last week, because I might get a bike trainer instead.",
     "The user sold the rowing machine, because the user might get a bike trainer instead.",
     "The user sold the rowing machine."),
])
def test_a_hedge_in_a_trailing_reason_clause_trims_the_claim(user, claim, kept):
    got, _, notes = _check(user, (claim, "user_stated"))
    assert got == [kept], notes


@pytest.mark.parametrize("user,claim,kept", [
    ("We should only book the big meeting room when more than 8 people attend.",
     "The team books the big meeting room only when more than 8 people attend.", True),
    ("We should try booking the big meeting room for 8 people.", "The team books the big meeting room for 8 people.",
     False),
    ("The sync moves to Thursdays at 15:00, but nothing is decided yet.", "The sync moves to Thursdays at 15:00.", False),
    ("Just thinking out loud: the garage needs a 16 A socket.", "The garage needs a 16 A socket.", False),
])
def test_standing_rules_are_not_proposals_and_undecided_stays_out(user, claim, kept):
    assert (_check(user, (claim, "user_stated"), domain="project")[0] == [claim]) is kept


def test_carry_over_does_not_split_grouped_numbers():
    from pan.memory.curator import carry_clauses
    from pan.memory.facts import Contradiction
    old = "The print shop job runs on port 9410, the monthly budget is $7,000, and the ink costs 0,32 € per page."
    hit = Contradiction(["port"], {"port": {"9410"}}, {"port": {"9420"}})
    assert carry_clauses(old, hit, ["The print shop job now runs on port 9420."]) == [
        "the monthly budget is $7,000", "the ink costs 0,32 € per page"]


def _decision(*claims, kind: str = "fact", corrects: bool = False):
    return {"why": "t", "record": True, "record_confidence": 0.9, "language": "en", "sensitivity": "none",
            "candidates": [{"kind": kind, "domain": "personal", "lifetime": "long_term", "volatile": False,
                            "corrects": corrects, "importance": 0.7, "confidence": 0.8, "subject": "", "title": "",
                            "claims": [{"text": t, "evidence": e, "supersedes": s} for t, e, s in claims]}]}


def _post(user: str, dec: dict, assistant: str = "OK.", known: str = ""):
    ep = Episode("s", [_turn(1, user, assistant)], closed_by="turn")
    notes: list = []
    return post_check(dec, ep, None, build_input(ep, None, known), notes), notes


# -- the user's own statement inside a question -----------------------------------------------------------------

@pytest.mark.parametrize("user,claim", [
    ("I'm ordering Pepper a new harness. Which size would fit a Border Collie like Pepper?",
     "The user has a Border Collie named Pepper."),
    ('Could you suggest crime dramas similar to "Broadchurch" and "Mare of Easttown", which I just finished?',
     'The user just finished watching "Mare of Easttown".'),
    ("What are some good ways to keep my hallway clean, especially with a dog that sheds a lot?",
     "The user has a dog that sheds a lot."),
    ("I've been getting headaches every afternoon, could it be my monitor?",
     "The user has been getting headaches every afternoon."),
])
def test_a_first_person_fact_inside_a_question_is_user_stated(user, claim):
    kept, labels, notes = _check(user, (claim, "user_stated"))
    assert kept == [claim] and labels == [OBSERVED], notes


@pytest.mark.parametrize("user,claim", [
    ("Do you think I'm overtraining?", "The user is overtraining."),
    ("Is Postgres faster than MySQL for writes?", "Postgres is faster than MySQL for writes."),
    ("Can you recommend a camera like Fujifilm?", "The user has a Fujifilm camera."),
    ("What is the order of these: 'I renewed my passport', 'I booked the ferry'?", "The user renewed their passport."),
    ("I'm wondering if my plan is too ambitious, is it?", "The user's plan is too ambitious."),
])
def test_questions_without_a_statement_of_the_users_stay_out(user, claim):
    assert _check(user, (claim, "user_stated"))[0] == []


def test_a_preference_inside_a_question_stays_as_it_was():
    kept, _, _ = _check("Could you suggest podcasts with long interviews, which I really enjoy?",
                        ("The user enjoys podcasts with long interviews.", "user_stated"), kind="preference")
    assert kept == []


def test_an_inferred_conclusion_of_the_assistant_is_not_relabelled():
    cands, notes = _post("My cat Juno adores the new flat, she naps in the sun all day.",
                         _decision(("The user moved to a new flat.", "assistant_inferred", "")),
                         assistant="Sounds like the move went well for her!")
    assert all(c.classification.type is MemoryType.NOISE for c in cands)
    assert not any("→ user_stated" in n for n in notes)


# -- concessive clauses, firm evaluations and finds are not hedges ------------------------------------------------

@pytest.mark.parametrize("user,claim", [
    ("My mentor once wrote that although my estimates could be optimistic, my best trait is staying calm in outages.",
     "The user's best trait is staying calm in outages."),
    ("I think a two-day Terraform bootcamp is exactly what we need.",
     "The user thinks a two-day Terraform bootcamp is exactly what the team needs."),
    ("I found a Go concurrency workshop that would be perfect for the platform team.",
     "The user found a Go concurrency workshop for the platform team."),
])
def test_concessions_evaluations_and_finds_keep_the_fact(user, claim):
    kept, _, notes = _check(user, (claim, "user_stated"), domain="project")
    assert kept == [claim], notes


@pytest.mark.parametrize("user,claim", [
    ("I think the NAS has 16 GB RAM.", "The NAS has 16 GB RAM."),
    ("Although it's early, the offsite could be in Porto.", "The offsite is in Porto."),
    ("I might move to Leipzig next spring.", "The user might move to Leipzig next spring."),
    ("I probably still have a tile cutter somewhere, my brother says it's in the garage.",
     "The user probably owns a tile cutter, saying it is in the garage."),
])
def test_real_hedges_still_drop(user, claim):
    assert _check(user, (claim, "user_stated"), domain="project")[0] == []


# -- a claim about another occasion supersedes nothing ------------------------------------------------------------

def test_another_dated_outing_does_not_supersede_the_first():
    old = "The user saw 4 herons at the Elbe marshes with Ines on 5/03."
    (cand,), notes = _post("We went back with Ines on 5/17 and saw 11 herons that day!",
                           _decision(("The user saw 11 herons with Ines on 5/17.", "user_stated", old), corrects=True),
                           known=old)
    assert cand.supersedes == [] and any("another occasion" in n for n in notes)
    # a correction of the same outing still supersedes
    (cand,), _ = _post("Correction: it was 5/04, not 5/03, and we saw 6 herons.",
                       _decision(("The user saw 6 herons with Ines on 5/04.", "user_stated", old), corrects=True),
                       known=old)
    assert cand.supersedes == [old]


def test_a_page_title_from_a_claim_has_no_stamp():
    (cand,), _ = _post("I work at Harbourline Logistics.",
                       _decision(("The user works at Harbourline Logistics.", "user_stated", "")))
    assert "(stated" in cand.normalized_claims[0] and "stated" not in cand.title
