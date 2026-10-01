"""RuleClassifier (integration spec §4.6): fixture scenarios → expected labels (tests/fixtures/expected)."""

from __future__ import annotations

import json

import pytest

from conftest import scenario_episodes
from pan.events.schema import Actor, AgentEvent, Destination, EventType, Lifetime, MemoryType
from pan.memory.classifier import INFERRED, OBSERVED, RuleClassifier
from pan.memory.episodes import Episode, previous_turn

clf = RuleClassifier()


def classify_scenario(name: str):
    eps = scenario_episodes(name)
    return [c for i, ep in enumerate(eps) for c in clf.classify(ep, previous_turn(eps, i))]


def _turn(user: str, assistant: str = "", n: int = 1, session: str = "s") -> Episode:
    ev = AgentEvent(id=f"01K0CLS{n:019d}", ts="2026-09-23T10:00:00Z", event_type=EventType.TURN,
                    session_id=session, actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant})
    return Episode(session, [ev], closed_by="turn")


def _tool(n: int, command: str, status: str = "ok", result: str = "", **extra) -> AgentEvent:
    return AgentEvent(id=f"01K0CLS{n:019d}", ts="2026-09-23T10:00:00Z", event_type=EventType.TOOL_CALL,
                      session_id="s", actor=Actor.TOOL,
                      content={"tool": "terminal", "args": {"command": command}, "status": status,
                               "result_excerpt": result, **extra})


def _types(cands):
    return [(c.classification.type, c.classification.destination) for c in cands]


# -- the golden scenarios (tests/fixtures/expected/README.md) --------------------------------------

def test_preference_scenario():
    cands = classify_scenario("preference")
    assert _types(cands) == [(MemoryType.USER_PREFERENCE, Destination.USER)]
    c = cands[0]
    assert c.normalized_claims == ["Write meeting notes as Markdown files in the repo, don't e-mail them unless I ask."]
    assert c.claim_evidence == [OBSERVED]
    assert c.classification.should_remember and c.classification.lifetime is Lifetime.LONG_TERM
    assert c.id == "01K0PREF000000000000000001:user_preference" and c.session_id == "sess-pref"


def test_env_fact_scenario():
    cands = classify_scenario("env_fact")
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    c = cands[0]
    assert c.claim_evidence == [INFERRED, OBSERVED]
    assert "started without --enable-auto-tool-choice" in c.normalized_claims[0]
    assert c.normalized_claims[1].startswith("`docker inspect vllm-main` reports: vllm serve")
    assert "Hermes needs it." not in c.normalized_claims  # no infra noun + state verb
    assert set(c.event_ids) == {"01K0ENV0000000000000000001", "01K0ENV0000000000000000002"}
    assert "vllm" in c.tags and "vllm" in c.retrieval_query.lower()


def test_decision_scenario_needs_the_confirmation_turn():
    cands = classify_scenario("decision")
    assert _types(cands) == [(MemoryType.NOISE, Destination.NONE), (MemoryType.DECISION, Destination.WIKI)]
    d = cands[1]
    assert d.normalized_claims == ["Run curation in a separate background service (pan-memoryd); "
                                   "the plugin only writes events to a durable spool."]
    assert d.title == "Run curation in a separate background service (pan-memoryd)"
    assert d.event_ids == ["01K0DEC0000000000000000001", "01K0DEC0000000000000000002"]
    assert "confirmed by the user" in d.context


def test_duplicate_scenario():
    cands = classify_scenario("duplicate")
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    assert cands[0].normalized_claims[0].startswith("vLLM serves RedHatAI/Qwen3.6-35B-A3B-NVFP4")


def test_noise_scenario():
    cands = classify_scenario("noise")
    assert _types(cands) == [(MemoryType.NOISE, Destination.NONE)] * 2
    assert all(not c.classification.should_remember and c.normalized_claims == [] for c in cands)


# -- rules in isolation ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I prefer tabs over spaces in Makefiles.",
    "Always run the linter before committing.",
    "Never push directly to the main branch.",
    "Don't add emojis to commit messages please.",
    "Going forward, use uv instead of pip for installs.",
])
def test_preference_statements(text):
    assert _types(clf.classify(_turn(text))) == [(MemoryType.USER_PREFERENCE, Destination.USER)]


@pytest.mark.parametrize("text", [
    "Should I always use uv here?",       # question
    "Don't!",                             # too short
    "why do tool calls fail?",
    "I don't know why the build is slow.",
])
def test_not_preferences(text):
    assert _types(clf.classify(_turn(text))) == [(MemoryType.NOISE, Destination.NONE)]


def test_preference_lead_in_is_stripped():
    c = clf.classify(_turn("From now on, answer in English."))[0]
    assert c.normalized_claims == ["Answer in English."]


def test_confirmation_without_proposal_is_noise():
    prev = _turn("hi", "Hello! How can I help?", n=1)
    assert _types(clf.classify(_turn("yes", n=2), prev)) == [(MemoryType.NOISE, Destination.NONE)]


def test_negated_confirmation_is_not_a_decision():
    prev = _turn("db?", "I suggest we use Postgres for the event store.", n=1)
    cands = clf.classify(_turn("no, not Postgres", n=2), prev)
    assert _types(cands) == [(MemoryType.NOISE, Destination.NONE)]
    cands = clf.classify(_turn("ok, sounds good", n=2), prev)
    assert _types(cands) == [(MemoryType.DECISION, Destination.WIKI)]
    assert cands[0].normalized_claims == ["We use Postgres for the event store."]


def test_user_stated_decision():
    cands = clf.classify(_turn("Decision: the wiki stays a separate git repo per profile."))
    assert _types(cands) == [(MemoryType.DECISION, Destination.WIKI)]
    assert cands[0].normalized_claims == ["The wiki stays a separate git repo per profile."]


def test_error_followed_by_fix_is_a_learning():
    turn = _turn("start the proxy", "It failed because port 80 needs root; binding to 8080 fixed it.", n=3)
    events = [_tool(1, "nginx -c /srv/proxy.conf", status="error",
                    error_message="nginx: [emerg] bind() to 0.0.0.0:80 failed (13: Permission denied)"),
              _tool(2, "nginx -c /srv/proxy.conf -g 'listen 8080;'", result="started"), *turn.events]
    cands = clf.classify(Episode("s", events, closed_by="turn"))
    assert _types(cands) == [(MemoryType.LEARNING, Destination.WIKI)]
    c = cands[0]
    assert c.normalized_claims[0].startswith("`nginx -c /srv/proxy.conf` failed: nginx: [emerg] bind()")
    assert c.normalized_claims[1] == "Fixed by `nginx -c /srv/proxy.conf -g 'listen 8080;'`."
    assert c.claim_evidence == [OBSERVED, OBSERVED, INFERRED]
    assert c.title.startswith("nginx error:")


def test_error_without_fix_is_noise():
    ep = Episode("s", [_tool(1, "ls /nope", status="error", result="No such file")], closed_by="idle")
    assert _types(clf.classify(ep)) == [(MemoryType.NOISE, Destination.NONE)]


def test_infra_command_success_is_environment():
    ep = Episode("s", [_tool(1, "systemctl --user status langfuse", result="Active: active (running) since Mon")],
                 closed_by="idle")
    cands = clf.classify(ep)
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    assert cands[0].claim_evidence == [OBSERVED]


def test_non_infra_command_is_noise():
    ep = Episode("s", [_tool(1, "python -c 'print(1)'", result="1")], closed_by="idle")
    assert _types(clf.classify(ep)) == [(MemoryType.NOISE, Destination.NONE)]


def test_config_file_change_is_configuration():
    change = AgentEvent(id="01K0CLS0000000000000000005", ts="2026-09-23T10:00:00Z",
                        event_type=EventType.FILE_CHANGE, session_id="s", actor=Actor.TOOL,
                        content={"path": "deploy/vllm.yaml", "op": "write", "tool": "write_file"})
    cands = clf.classify(Episode("s", [change], closed_by="idle"))
    assert _types(cands) == [(MemoryType.CONFIGURATION, Destination.WIKI)]
    assert cands[0].normalized_claims == ["`deploy/vllm.yaml` was written by the agent."]


def test_hedged_explanations_are_not_facts():
    ep = _turn("why?", "The server might be running on port 9000.")
    assert _types(clf.classify(ep)) == [(MemoryType.NOISE, Destination.NONE)]


def test_preference_and_fact_in_one_episode():
    ep = _turn("Always use port 8000 for vLLM.", "Noted. The vLLM server listens on port 8000.")
    assert _types(clf.classify(ep)) == [(MemoryType.USER_PREFERENCE, Destination.USER),
                                         (MemoryType.ENVIRONMENT, Destination.WIKI)]


# -- recall is not observation (M4 live run: answers restating the wiki were written back) ----------

WIKI_READ = json.dumps({"id": "learnings.vllm-tool-calling", "found": True, "content": (
    "vLLM on the DGX Spark serves `RedHatAI/Qwen3.6-35B-A3B-NVFP4` as `primary` on port 8000, but was "
    "started without `--enable-auto-tool-choice --tool-call-parser`, so tool calls fail. Hermes needs "
    "tool calling. The fix: restart the vllm-main container with --tool-call-parser hermes.")})
# The real Qwen3.8 answer from the first live run (it had just called memory_search + memory_read).
ECHO = ("Tool calls fail because the vLLM container on the DGX Spark was started without "
        "`--enable-auto-tool-choice --tool-call-parser hermes`, so the server doesn't parse or emit "
        "structured tool calls.")


def _recall_episode(assistant: str, *, tool: str = "memory_read", result: str = WIKI_READ,
                    status: str = "ok", recall: str = "") -> Episode:
    read = AgentEvent(id="01K0CLS0000000000000000001", ts="2026-09-23T10:00:00Z", event_type=EventType.TOOL_CALL,
                      session_id="s", actor=Actor.TOOL,
                      content={"tool": tool, "args": {"page": "learnings.vllm-tool-calling"}, "status": status,
                               "result_excerpt": result})
    content = {"user": "Why do tool calls fail on our vLLM server?", "assistant": assistant}
    if recall:
        content["recall"] = recall
    turn = AgentEvent(id="01K0CLS0000000000000000002", ts="2026-09-23T10:00:01Z", event_type=EventType.TURN,
                      session_id="s", actor=Actor.MAIN_AGENT, content=content)
    return Episode("s", [read, turn], closed_by="turn")


def test_answer_restating_recalled_wiki_is_noise():
    assert _types(clf.classify(_recall_episode(ECHO))) == [(MemoryType.NOISE, Destination.NONE)]


def test_same_answer_without_recall_is_still_a_fact():
    ep = _turn("Why do tool calls fail on our vLLM server?", ECHO)
    assert _types(clf.classify(ep)) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]


def test_prefetch_recall_counts_like_a_memory_read():
    recall = "Possibly relevant PAN wiki pages (open with memory_read):\n- [learnings.vllm-tool-calling] " + WIKI_READ
    ep = _recall_episode(ECHO, tool="read_file", result="unrelated file content", recall=recall)
    assert _types(clf.classify(ep)) == [(MemoryType.NOISE, Destination.NONE)]


def test_new_numbers_beyond_recall_are_kept():
    new = "The vLLM server now listens on port 8001 with the tool-call parser enabled."
    cands = clf.classify(_recall_episode(new))
    assert _types(cands) == [(MemoryType.ENVIRONMENT, Destination.WIKI)]
    assert cands[0].normalized_claims == [new]


def test_failed_memory_tool_is_not_a_learning():
    read_fail = _recall_episode("Found it.", status="error", result="no wiki page with id 'x'")
    search_ok = AgentEvent(id="01K0CLS0000000000000000003", ts="2026-09-23T10:00:00Z",
                           event_type=EventType.TOOL_CALL, session_id="s", actor=Actor.TOOL,
                           content={"tool": "memory_read", "args": {"page": "y"}, "status": "ok",
                                    "result_excerpt": "{}"})
    ep = Episode("s", [read_fail.events[0], search_ok, read_fail.events[1]], closed_by="turn")
    assert _types(clf.classify(ep)) == [(MemoryType.NOISE, Destination.NONE)]


def test_contraction_fragments_are_not_content_tokens():
    from pan.memory.claims import token_set

    assert token_set("The server doesn't parse it; we can't") == {"server", "pars"}
