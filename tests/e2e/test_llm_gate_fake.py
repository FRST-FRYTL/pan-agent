"""M9 LLM gate against the scripted fake OpenAI server: request shape, fallback paths, circuit
breaker, and the daemon writing the gate info into the curation log (classifier design §5.2:
recorded-response tests, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import fixed_clock, spool_scenarios
from pan.config import ClassifierConfig, GateModelConfig
from pan.daemon.memoryd import MemoryDaemon
from pan.events.schema import Actor, AgentEvent, Destination, EventType, MemoryType
from pan.memory.dates import bare
from pan.memory.episodes import Episode
from pan.memory.llm_gate import GATE_VERSION, LocalLLMClassifier
from pan.paths import PanPaths

sys.path.insert(0, str(Path(__file__).parent))
from fake_openai import FakeOpenAI  # noqa: E402

pytestmark = pytest.mark.e2e

PREF_USER = "From now on, write meeting notes as Markdown files in the repo, don't e-mail them unless I ask."


def _reply(*cands, record=True, why="standing preference") -> dict:
    return {"content": json.dumps({"why": why, "record": record, "record_confidence": 0.9, "language": "en",
                                   "sensitivity": "none", "candidates": list(cands)})}


PREF_REPLY = _reply({"kind": "preference", "domain": "other", "lifetime": "long_term", "volatile": False,
                     "corrects": False, "importance": 0.7, "confidence": 0.9, "subject": "", "title": "Specs",
                     "claims": [{"text": "Write meeting notes as Markdown files in the repo, don't e-mail them "
                                         "unless I ask.", "evidence": "user_stated", "supersedes": ""}]})


def _episode(user: str = PREF_USER, assistant: str = "Understood.", n: int = 1) -> Episode:
    ev = AgentEvent(id=f"01K0GATEFAKE{n:014d}", ts="2026-09-24T10:00:00Z", event_type=EventType.TURN,
                    session_id="s", actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant})
    return Episode("s", [ev], closed_by="turn")


def _gate(server: FakeOpenAI, **cfg) -> LocalLLMClassifier:
    return LocalLLMClassifier(GateModelConfig(base_url=server.base_url, model="fake-gate"),
                              ClassifierConfig(**cfg), sleep=lambda s: None)


def test_request_shape_and_success():
    with FakeOpenAI([], plain_script=[PREF_REPLY]) as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode(), None, known="MEMORY.md: vLLM on 8000")
        body = server.requests[0]
    assert body["model"] == "fake-gate" and body["temperature"] == 0 and body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 400
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["required"][0] == "record"
    system, user = body["messages"]
    assert system["role"] == "system" and "memory gate" in system["content"]
    assert "<episode>" in user["content"] and "<known>" in user["content"] and PREF_USER in user["content"]
    assert c.classification.type is MemoryType.USER_PREFERENCE and c.classification.destination is Destination.USER
    info = gate.last_info
    assert info["version"] == GATE_VERSION and info["status"] == "ok" and info["classifier"] == GATE_VERSION
    assert info["calls"][0]["prompt_tokens"] == 100 and info["latency_s"] >= 0
    assert info["output"]["why"] == "standing preference" and info["post_checks"] == []


def test_result_is_cached_for_the_same_episode():
    with FakeOpenAI([], plain_script=[PREF_REPLY]) as server:
        gate = _gate(server)
        gate.classify(_episode())
        again = gate.classify(_episode())
        assert len(server.requests) == 1
    assert again[0].classification.type is MemoryType.USER_PREFERENCE and gate.last_info["cached"] is True


@pytest.mark.parametrize("step,reason,calls", [
    ({"delay": 1.5, "content": "{}"}, "timeout", 1),   # a timeout is not retried (episode budget)
    ({"status": 500, "error": "engine dead"}, "http_error", 1),
    ({"content": "Sure! Here is what I think: record it."}, "invalid_output", 1),
    ({"content": json.dumps({"record": "maybe"})}, "invalid_output", 1),
])
def test_fallback_to_rules(step, reason, calls):
    with FakeOpenAI([], plain_script=[step, step]) as server:
        gate = _gate(server, timeout_s=0.5, attempts=2)
        cands = gate.classify(_episode())
        n = len(server.requests)
    assert n == calls
    info = gate.last_info
    assert info["status"] == "fallback" and info["fallback_reason"] == reason and info["classifier"] == "rules-v2"
    # the rules still find the preference
    assert [c.classification.type for c in cands] == [MemoryType.USER_PREFERENCE]
    assert cands[0].rationale.startswith(f"fallback:rules ({reason}); ")


def test_http_400_drops_response_format_once():
    with FakeOpenAI([], plain_script=[{"status": 400, "error": "json_schema not supported"}, PREF_REPLY,
                                      PREF_REPLY]) as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode())
        gate.classify(_episode(user=PREF_USER + " Thanks.", n=2))
        bodies = server.requests
    assert "response_format" in bodies[0] and "response_format" not in bodies[1] and "response_format" not in bodies[2]
    assert gate.last_info["status"] == "ok" and c.classification.type is MemoryType.USER_PREFERENCE


@pytest.mark.parametrize("status,message", [
    (400, None),                                                        # OpenAI: names the field
    (422, "body.chat_template_kwargs: Extra inputs are not permitted"),  # pydantic-style validation
])
def test_hosted_api_rejecting_chat_template_kwargs(status, message):
    """A hosted API that rejects the vLLM/SGLang extension: one retry without it, then it stays off
    (structured output is kept); no fallback to the rules."""
    with FakeOpenAI([], plain_script=[PREF_REPLY, PREF_REPLY], reject_fields=("chat_template_kwargs",),
                    reject_status=status, reject_message=message) as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode())
        gate.classify(_episode(user=PREF_USER + " Thanks.", n=2))
        bodies = server.requests
    assert len(bodies) == 3 and "chat_template_kwargs" in bodies[0]
    assert all("chat_template_kwargs" not in b and b["response_format"]["type"] == "json_schema" for b in bodies[1:])
    assert gate.last_info["status"] == "ok" and gate.last_info["dropped_fields"] == ["chat_template_kwargs"]
    assert c.classification.type is MemoryType.USER_PREFERENCE and c.rationale.startswith(GATE_VERSION)


def test_unnamed_400_drops_response_format_then_chat_template_kwargs():
    """An error that names no field: structured output goes first (M9 behaviour), then the kwargs."""
    with FakeOpenAI([], plain_script=[PREF_REPLY], reject_fields=("chat_template_kwargs",),
                    reject_message="Bad request") as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode())
        bodies = server.requests
    assert len(bodies) == 3 and "response_format" in bodies[0] and "chat_template_kwargs" in bodies[1]
    assert "response_format" not in bodies[1] and "response_format" not in bodies[2]
    assert "chat_template_kwargs" not in bodies[2]
    assert gate.last_info["dropped_fields"] == ["response_format", "chat_template_kwargs"]
    assert c.classification.type is MemoryType.USER_PREFERENCE


@pytest.mark.parametrize("mode,sent,falls_back", [("never", False, False), ("always", True, True)])
def test_template_kwargs_modes(mode, sent, falls_back):
    """``never``: the field is not sent at all; ``always``: it is never dropped (a rejecting endpoint
    then falls back to the rules, as before this setting existed)."""
    with FakeOpenAI([], plain_script=[PREF_REPLY], reject_fields=("chat_template_kwargs",)) as server:
        gate = LocalLLMClassifier(GateModelConfig(base_url=server.base_url, model="fake-gate", template_kwargs=mode),
                                  ClassifierConfig(), sleep=lambda s: None)
        gate.classify(_episode())
        bodies = server.requests
    assert ("chat_template_kwargs" in bodies[0]) is sent
    assert (gate.last_info["status"] == "fallback") is falls_back


def test_default_request_body_unchanged():
    """The default path (local vLLM) sends exactly the M9 request shape: key order and values."""
    from pan.memory.llm_gate import ChatClient
    body = ChatClient(GateModelConfig(), 90).body([{"role": "user", "content": "x"}])
    assert list(body) == ["model", "messages", "temperature", "max_tokens", "stream", "chat_template_kwargs",
                          "response_format"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["model"] == "primary"


def test_circuit_breaker_skips_calls_while_open():
    now = [1000.0]
    with FakeOpenAI([], plain_script=[{"status": 500}] * 5) as server:
        gate = LocalLLMClassifier(GateModelConfig(base_url=server.base_url),
                                  ClassifierConfig(breaker_failures=2, breaker_open_s=60, attempts=1),
                                  clock=lambda: now[0], sleep=lambda s: None)
        for i in range(4):
            gate.classify(_episode(n=i + 1))
        assert len(server.requests) == 2
        assert gate.last_info["fallback_reason"] == "circuit_open"
        now[0] += 61
        gate.classify(_episode(n=9))
        assert len(server.requests) == 3


def test_secret_never_reaches_model_or_log():
    key = "sk-" + "t3st" + "Q9x7Lm2Pv8Rz4Kd1"
    user = f"My API key for the staging server is {key}, the server runs on port 9000."
    reply = _reply({"kind": "fact", "domain": "environment", "lifetime": "long_term", "volatile": False,
                    "corrects": False, "importance": 0.6, "confidence": 0.8, "subject": "staging server",
                    "title": "Staging server", "claims": [
                        {"text": "The staging server runs on port 9000.", "evidence": "user_stated", "supersedes": ""},
                        {"text": f"The staging API key is {key}.", "evidence": "user_stated", "supersedes": ""}]})
    with FakeOpenAI([], plain_script=[reply]) as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode(user=user))
        sent = json.dumps(server.requests)
    assert key not in sent and "[secret:api_key]" in sent
    assert key not in json.dumps(c.normalized_claims) and "The staging server runs on port 9000." in [bare(x) for x in c.normalized_claims]
    assert gate.last_info["output"]["sensitivity"] == "secret" and key not in json.dumps(gate.last_info)


# -- daemon integration --------------------------------------------------------------------------------------------

def _log(home: Path) -> list:
    path = PanPaths.for_home(home).curation_log
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _configure(home: Path, base_url: str, kind: str) -> None:
    # Via Hermes' config.yaml `pan:` section, as a benchmark harness would write it.
    import yaml
    cfg_path = home / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["pan"] = {"classifier": {"kind": kind, "timeout_s": 5}, "models": {"gate": {"base_url": base_url}}}
    cfg_path.write_text(yaml.safe_dump(cfg))


def test_daemon_logs_gate_info_and_applies_llm_result(pan_home: Path):
    with FakeOpenAI([], plain_script=[PREF_REPLY]) as server:
        _configure(pan_home, server.base_url, "llm")
        spool_scenarios(pan_home, ["preference"])
        daemon = MemoryDaemon(PanPaths.for_home(pan_home), clock=fixed_clock, worker_id="test")
        daemon.prepare()
        try:
            daemon.drain(flush=True)
        finally:
            daemon.close()
        assert len(server.requests) == 1
    (rec,) = [r for r in _log(pan_home) if r.get("episode_id")]
    assert rec["classifier"] == GATE_VERSION and rec["outcome"] == "l1_added"
    assert rec["gate"]["version"] == GATE_VERSION and rec["gate"]["status"] == "ok"
    assert isinstance(rec["gate"]["latency_s"], float) and rec["gate"]["calls"][0]["latency_s"] >= 0
    assert rec["candidate"]["rationale"].startswith(GATE_VERSION + ": ")
    user_md = (pan_home / "memories" / "USER.md").read_text()
    assert "Markdown files in the repo" in user_md


def test_daemon_fallback_when_endpoint_down(pan_home: Path):
    _configure(pan_home, "http://127.0.0.1:9/v1", "llm")   # nothing listens on port 9
    spool_scenarios(pan_home, ["preference"])
    daemon = MemoryDaemon(PanPaths.for_home(pan_home), clock=fixed_clock, worker_id="test")
    daemon.classifier.sleep = lambda s: None
    daemon.prepare()
    try:
        daemon.drain(flush=True)
    finally:
        daemon.close()
    (rec,) = [r for r in _log(pan_home) if r.get("episode_id")]
    assert rec["classifier"] == "rules-v2" and rec["gate"]["status"] == "fallback"
    assert rec["gate"]["fallback_reason"] == "connection" and rec["outcome"] == "l1_added"


def test_daemon_shadow_mode_applies_rules_and_logs_llm(pan_home: Path):
    with FakeOpenAI([], plain_script=[_reply(record=False, why="shadow says noise")]) as server:
        _configure(pan_home, server.base_url, "shadow")
        spool_scenarios(pan_home, ["preference"])
        daemon = MemoryDaemon(PanPaths.for_home(pan_home), clock=fixed_clock, worker_id="test")
        daemon.prepare()
        try:
            daemon.drain(flush=True)
        finally:
            daemon.close()
    (rec,) = [r for r in _log(pan_home) if r.get("episode_id")]
    assert rec["classifier"] == "rules-v2" and rec["outcome"] == "l1_added"
    # the scripted gate said "noise"; the explicit "From now on …" preference is kept anyway (G3)
    assert rec["gate"]["mode"] == "shadow" and rec["gate"]["shadow"][0]["type"] == "user_preference"


def test_gate_known_excludes_the_agents_own_l1_write_of_the_same_turn(pan_home: Path):
    from conftest import apply_agent_l1_writes
    noise = _reply(record=False, why="n")
    with FakeOpenAI([], plain_script=[noise] * 6) as server:
        _configure(pan_home, server.base_url, "llm")
        spool_scenarios(pan_home, ["fact_update"])
        apply_agent_l1_writes(pan_home, ["fact_update"])   # the agent saved "vLLM server runs on port 8000."
        daemon = MemoryDaemon(PanPaths.for_home(pan_home), clock=fixed_clock, worker_id="test")
        daemon.prepare()
        try:
            daemon.drain(flush=True)
        finally:
            daemon.close()
        prompts = [r["messages"][1]["content"] for r in server.requests]

    def known(p: str) -> str:
        return p.split("<known>")[1].split("</known>")[0] if "<known>" in p else ""
    first = next(p for p in prompts if "Note for later: our vLLM server runs on port 8000." in p)
    later = next(p for p in prompts if "Which port does our vLLM server use?" in p)
    assert "port 8000" not in known(first)    # written by the agent in this very turn
    assert "port 8000" in known(later)        # an older entry: a real echo source


def test_pre_skipped_episode_makes_no_request():
    with FakeOpenAI([], plain_script=[PREF_REPLY]) as server:
        gate = _gate(server)
        (c,) = gate.classify(_episode(user="Which port does our cache listen on?", n=5))
        assert server.requests == []
    assert c.classification.type is MemoryType.NOISE and gate.last_info["status"] == "skipped"
    assert gate.last_info["skip_reason"] == "questions only"


def test_claim_dates_come_from_the_event_timestamp_unless_the_user_states_one():
    with FakeOpenAI([], plain_script=[_reply({"kind": "fact", "domain": "personal", "lifetime": "long_term",
                                              "volatile": False, "corrects": False, "importance": 0.5,
                                              "confidence": 0.8, "subject": "fence", "title": "",
                                              "claims": [{"text": "The user painted the fence yesterday.",
                                                          "evidence": "user_stated", "supersedes": ""}]},
                                             why="fact")] * 2) as server:
        gate = _gate(server)
        plain = _episode(user="I painted the fence yesterday, it looks great.", n=11)   # event ts 2026-09-24
        (c,) = gate.classify(plain)
        assert c.normalized_claims == ["The user painted the fence yesterday (2026-09-23). (stated 2026-09-24)"]
        dated = _episode(user="Today is 11 March 2024. I painted the fence yesterday, it looks great.", n=12)
        (c,) = gate.classify(dated)
        assert c.normalized_claims == ["The user painted the fence yesterday (2024-03-10). (stated 2024-03-11)"]
