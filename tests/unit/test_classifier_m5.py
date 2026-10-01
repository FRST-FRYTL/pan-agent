"""RuleClassifier M5 rules: user-stated facts, subjects, echo/ack filter, printed tool calls, L1 recall.

Episodes are shaped like multi-session benchmark runs (Qwen3.8-27B), see tests/fixtures/events/.
"""

from __future__ import annotations

import pytest

from conftest import scenario_episodes
from pan.events.schema import Actor, AgentEvent, Destination, EventType, MemoryType
from pan.memory.classifier import INFERRED, OBSERVED, RuleClassifier
from pan.memory.episodes import Episode, previous_turn

clf = RuleClassifier()


def _turn(user: str, assistant: str = "", n: int = 1, session: str = "s") -> Episode:
    ev = AgentEvent(id=f"01K0CM5{n:019d}", ts="2026-09-23T10:00:00Z", event_type=EventType.TURN,
                    session_id=session, actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant})
    return Episode(session, [ev], closed_by="turn")


def _types(cands):
    return [(c.classification.type, c.classification.destination) for c in cands]


def _scenario(name):
    eps = scenario_episodes(name)
    return [clf.classify(ep, previous_turn(eps, i)) for i, ep in enumerate(eps)]


@pytest.mark.parametrize("user,claim,subject", [
    ("Note for later: our vLLM server runs on port 8000.", "Our vLLM server runs on port 8000.", "vLLM server"),
    ("Update: we moved the vLLM server to port 8010.", "We moved the vLLM server to port 8010.", "vLLM server"),
    ("FYI: the Langfuse instance for this project runs on host spark01 at port 3300.",
     "The Langfuse instance for this project runs on host spark01 at port 3300.", "Langfuse instance"),
])
def test_user_stated_environment_facts(user, claim, subject):
    cands = clf.classify(_turn(user, "Noted."))
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    c = cands[0]
    assert c.normalized_claims == [claim] and c.claim_evidence == [OBSERVED]
    assert c.subject == subject and c.title == subject
    assert c.rationale == "user states an environment fact"


@pytest.mark.parametrize("user", [
    "By the way, my neighbour's cat is called Mr. Whiskers. What is 17 times 3?",   # seed-scenario distractor
    "Run `nvidia-smi` and tell me which GPU this machine has.",                      # a request
    "Which port does our vLLM server use?",                                          # a question
    "Maybe the server runs on port 9000.",                                           # hedged
    "Summarize what vLLM does.",
])
def test_not_user_facts(user):
    assert _types(clf.classify(_turn(user))) == [(MemoryType.NOISE, Destination.NONE)]


@pytest.mark.parametrize("assistant", [
    "Saved: vLLM server on port 8000.",                            # M4 bench run
    "Noted — vLLM server on port 8000 is saved to memory.",        # M3 bench run
    "Got it — noted that this project's vLLM server runs on port 8000.",
])
def test_acknowledgements_do_not_become_claims(assistant):
    c = clf.classify(_turn("Note for later: our vLLM server runs on port 8000.", assistant))[0]
    assert c.normalized_claims == ["Our vLLM server runs on port 8000."]


def test_restating_the_user_only_counts_when_the_user_stated_a_fact():
    ep = _turn("Always use port 8000 for vLLM.", "Noted. The vLLM server listens on port 8000.")
    assert _types(clf.classify(ep)) == [(MemoryType.USER_PREFERENCE, Destination.USER),
                                         (MemoryType.ENVIRONMENT, Destination.WIKI)]


def test_answer_from_l1_is_not_a_new_fact():
    """M3 bench run, session 3: the stale L1 entry answered the question."""
    ep = _turn("Which port does our vLLM server use? Answer from memory in one sentence.",
               "Our vLLM server runs on port 8000.")
    assert _types(clf.classify(ep, known="vLLM server runs on port 8000.")) == [(MemoryType.NOISE, Destination.NONE)]
    assert _types(clf.classify(ep)) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]


def test_printed_tool_call_is_dropped_but_the_users_fact_is_kept():
    ep = _turn("Update: we moved the vLLM server to port 8010.",
               'memory replace: content="vLLM server runs on port 8010.", old_text="port 8000", target=memory')
    c = clf.classify(ep)[0]
    assert c.normalized_claims == ["We moved the vLLM server to port 8010."]
    assert c.claim_evidence == [OBSERVED]


def test_printed_tool_call_alone_is_noise_with_a_reason():
    cands = clf.classify(_turn("Please remember the vLLM docs link.",
                               "<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n"
                               "<parameter=content>\nvLLM server docs on port 8000\n</parameter>\n</function>\n"
                               "</tool_call>"))
    assert _types(cands) == [(MemoryType.NOISE, Destination.NONE)]
    assert "printed a tool call" in cands[0].rationale


def test_gpu_fact_gets_a_subject_title_and_tags():
    (cands,) = _scenario("gpu_fact")
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]  # the empty-args retry is no learning
    c = cands[0]
    assert c.subject == "GPU" and c.title == "GPU"
    assert {"gpu", "nvidia", "hardware"} <= set(c.tags)
    assert c.normalized_claims[-1] == "GPU (`nvidia-smi --query-gpu=name --format=csv,noheader`) reports: NVIDIA GB10"
    # M6: the explanation's value (GB10) is in the nvidia-smi output → grounded, observed
    assert c.claim_evidence == [OBSERVED, OBSERVED]


def test_toolcall_text_scenario():
    tct1, tct2, tct3 = _scenario("toolcall_text")
    assert _types(tct1) == [(MemoryType.NOISE, Destination.NONE)]
    assert _types(tct2) == [(MemoryType.DECISION, Destination.WIKI)]
    assert _types(tct3) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    assert tct3[0].normalized_claims == ["The Grafana container listens on port 3100."]
    assert tct3[0].subject == "Grafana container"


def test_fact_update_scenario():
    s1, s2, _ = _scenario("fact_update")
    assert [c.normalized_claims for c in s1] == [["Our vLLM server runs on port 8000."]]  # no memory-tool learning
    assert [c.normalized_claims for c in s2] == [["We moved the vLLM server to port 8010."]]
    assert all(c.subject == "vLLM server" for c in s1 + s2)
