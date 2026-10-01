"""T3: the ``assistant_reported`` evidence class — what the assistant said it did, decided or found in a
turn is kept, labelled and low-confidence, instead of being dropped for lack of a tool output. Own
wording (ADR-009); the shapes follow the Hermes session_search_schema history (text-only reports)."""

from __future__ import annotations

from pan.events.schema import Actor, AgentEvent, EventType, MemoryType
from pan.memory.claims import REPORTED_PREFIX
from pan.memory.classifier import OBSERVED, REPORTED
from pan.memory.dates import bare
from pan.memory.episodes import Episode
from pan.memory.llm_gate import build_input, grounded, post_check


def _ev(n: int, etype: EventType, content: dict) -> AgentEvent:
    return AgentEvent(id=f"01K0REPT{n:018d}", ts="2026-09-24T10:00:00Z", event_type=etype, session_id="s",
                      actor=Actor.MAIN_AGENT, content=content)


def _turn(n: int, user: str, assistant: str) -> AgentEvent:
    return _ev(n, EventType.TURN, {"user": user, "assistant": assistant})


def _tool(n: int, result: str, command: str) -> AgentEvent:
    return _ev(n, EventType.TOOL_CALL, {"tool": "terminal", "args": {"command": command}, "status": "ok",
                                        "result_excerpt": result, "error_message": None})


def _check(claims, *events, previous=None, kind="fact", domain="environment", subject=""):
    ep = Episode("s", list(events), closed_by="turn")
    dec = {"why": "t", "record": True, "record_confidence": 0.9, "language": "en", "sensitivity": "none",
           "candidates": [{"kind": kind, "domain": domain, "lifetime": "long_term", "volatile": False,
                           "corrects": False, "importance": 0.7, "confidence": 0.8, "subject": subject, "title": "",
                           "claims": [{"text": t, "evidence": e, "supersedes": ""} for t, e in claims]}]}
    notes: list = []
    return post_check(dec, ep, previous, build_input(ep, previous, ""), notes), notes


def _reported(cands):
    return [bare(c)[len(REPORTED_PREFIX):] for x in cands for c, lab in zip(x.normalized_claims, x.claim_evidence)
            if lab == REPORTED]


def test_decision_the_assistant_made_is_kept_as_reported():
    cands, notes = _check([("The docs site migration will use rsync to move to the new host.", "user_confirmed")],
                          _turn(1, "Downtime matters, pick the copy method.",
                                "Decided: move the docs site to the new host using rsync, cutover Saturday night."))
    (c,) = cands
    assert c.claim_evidence == [REPORTED] and c.normalized_claims[0].startswith(REPORTED_PREFIX)
    assert c.classification.confidence <= 0.4 and c.classification.type is MemoryType.ENVIRONMENT
    assert any("kept as assistant_reported" in n for n in notes)


def test_terse_action_report_with_a_value_is_kept():
    cands, _ = _check([("The final fix raised the worker pool timeout to 45s on the queue.", "user_confirmed")],
                      _turn(1, "what exactly did you change? write it down",
                            "Final fix: raised the worker pool timeout to 45s on the queue and re-ran the backlog."))
    assert _reported(cands) == ["The final fix raised the worker pool timeout to 45s on the queue."]


def test_value_about_something_the_conversation_named_is_kept():
    prev = Episode("s", [_turn(1, "I want charts for the weather station", "I set up a chartbox instance for the sensors.")],
                   closed_by="turn")
    cands, _ = _check([("The chartbox dashboard is on port 4100 of the shed pi.", "tool_observed")],
                      _turn(2, "where do I see it", "The chartbox dashboard is on port 4100 of the shed pi."),
                      previous=prev, subject="chartbox")
    assert _reported(cands) == ["The chartbox dashboard is on port 4100 of the shed pi."]


def test_inferred_only_candidate_without_tool_support_keeps_reports_only():
    cands, notes = _check([("Deployed the patched build to every node.", "assistant_inferred")],
                          _turn(1, "ship it everywhere then", "Deployed the patched build to every node."),
                          domain="other")
    assert _reported(cands) == ["Deployed the patched build to every node."]
    assert any("no tool support" in n for n in notes)


def test_offers_plans_advice_hedges_and_questions_stay_out():
    for assistant in ("I'll raise the pool timeout to 45s tomorrow.",
                      "You could raise the pool timeout to 45s.",
                      "I recommend raising the pool timeout to 45s.",
                      "Maybe the pool timeout was raised to 45s.",
                      "Should I raise the pool timeout to 45s?"):
        cands, _ = _check([("The pool timeout is 45s.", "user_confirmed")], _turn(1, "and the timeout", assistant))
        assert _reported(cands) == [], assistant
        assert cands[0].classification.type is MemoryType.NOISE


def test_world_knowledge_stays_out():
    """F8: a general statement from the assistant is not a report of this conversation's work."""
    cands, _ = _check([("The RX-9 board uses a Kestrel X4 chip, the controller found in most mesh routers.",
                        "tool_observed")],
                      _turn(1, "what board is this?",
                            "The RX-9 board uses a Kestrel X4 chip, the controller found in most mesh routers."))
    assert _reported(cands) == []


def test_tool_ran_and_does_not_show_the_value():
    """A number the assistant adds to a tool result is made up, not reported."""
    cands, _ = _check([("The queue service is running with 12 workers.", "assistant_inferred")],
                      _tool(1, "queue active", "systemctl status queue"),
                      _turn(2, "is the queue up?", "The queue service is running with 12 workers."))
    assert _reported(cands) == [] and cands[0].classification.type is MemoryType.NOISE


def test_memory_talk_and_echo_stay_out():
    cands, _ = _check([("I saved the port 8123 to memory.", "assistant_inferred")],
                      _turn(1, "note the port", "I saved the port 8123 to memory."))
    assert _reported(cands) == []


def test_model_cannot_claim_reported_itself():
    """``assistant_reported`` in the model output is treated as inferred and checked like it."""
    cands, _ = _check([("The tides app shows 14 stations.", "assistant_reported")],
                      _turn(1, "hm", "Nice weather today."))
    assert _reported(cands) == []


def test_observed_claims_are_unchanged():
    cands, _ = _check([("The backup disk is mounted at /mnt/vault.", "user_stated")],
                      _turn(1, "The backup disk is mounted at /mnt/vault.", "Noted."))
    assert cands[0].claim_evidence == [OBSERVED] and not cands[0].normalized_claims[0].startswith(REPORTED_PREFIX)


def test_iso_timestamp_hour_is_a_value_token():
    """Grounding (tool_fact): "2026-03-02T09:15" in a CSV holds the hour 09 and the minute 15."""
    assert grounded("On 2026-03-02 the pump in zone B4 reached 71.5 kPa at 09:15.",
                    "ts,zone,kpa\n2026-03-02T09:15,B4,71.5")


def test_reports_about_the_user_or_from_memory_talk_stay_out():
    """The report must be the assistant's own action; a memory-talk sentence is never a report."""
    for assistant, claim in (("You proposed the name Juno for the new kitten.", "The user has a kitten named Juno."),
                             ("Updated my persistent memory: all flyers should show the May 3rd opening.",
                              "All flyers should show the May 3rd opening.")):
        cands, _ = _check([(claim, "tool_observed")], _turn(1, "ok", assistant))
        assert _reported(cands) == [], assistant


def test_grouped_amounts_match_across_number_conventions():
    """Grounding (tool_fact): "3.120,75 €" in a German claim is the CSV's 3120.75."""
    assert grounded("Die Rechnung R-77 beträgt 3.120,75 € netto.", "id,netto\nR-77,3120.75")
    assert grounded("Invoice R-77 is 3,120.75 net.", "id,net\nR-77,3120.75")


def _check_life(claims, *events, life="session", kind="incident"):
    ep = Episode("s", list(events), closed_by="turn")
    dec = {"why": "t", "record": True, "record_confidence": 0.9, "language": "en", "sensitivity": "none",
           "candidates": [{"kind": kind, "domain": "environment", "lifetime": life, "volatile": life != "long_term",
                           "corrects": False, "importance": 0.7, "confidence": 0.8, "subject": "", "title": "",
                           "claims": [{"text": t, "evidence": e, "supersedes": ""} for t, e in claims]}]}
    notes: list = []
    return post_check(dec, ep, None, build_input(ep, None, ""), notes)


def test_dated_tool_record_is_a_past_event_not_session_state():
    """h3 tool_fact shape: the gate marks a log line it read as life=session; a dated record stays true."""
    log = "2026-03-02 04:12:09 kiln-2 fault OVERTEMP code K-771\n2026-03-02 04:15:40 kiln-2 back to normal"
    (c,) = _check_life([("On 2026-03-02 at 04:12:09 kiln-2 raised fault OVERTEMP with code K-771.", "tool_observed")],
                       _tool(1, log, "cat logs/kiln.log"),
                       _turn(2, "which kiln faulted last night and with what code?", "Kiln-2, code K-771."))
    assert c.classification.destination.value == "wiki" and "dated tool record" in c.rationale


def test_iso_datetime_and_seconds_count_as_dated():
    from pan.memory.llm_gate import _dated_record
    assert _dated_record("Job KX-12 aborted at 2026-04-07T03:18:44 on bench 4.")
    assert _dated_record("Job KX-12 aborted at 03:18:44.")
    assert not _dated_record("Job KX-12 is running on bench 4.")
    assert not _dated_record("Release 2.14 is installed.")


def test_momentary_state_and_user_plans_stay_session():
    (c,) = _check_life([("The kiln-2 service is active.", "tool_observed")],
                       _tool(1, "active (running)", "service-status kiln-2"), _turn(2, "is kiln-2 up?", "Yes."))
    assert c.classification.destination.value == "none"
    (c,) = _check_life([("The user has a call with the supplier at 15:00 today.", "user_stated")],
                       _turn(1, "I have a call with the supplier at 15:00 today, remind me what to ask.", "Sure."),
                       kind="fact")
    assert c.classification.destination.value == "none"
