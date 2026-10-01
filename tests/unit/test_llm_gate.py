"""M9 LLM gate: input building, output parsing and the deterministic post-checks (no model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pan.config import PanConfig, load_config, parse_config
from pan.events.schema import Actor, AgentEvent, Destination, EventType, Lifetime, MemoryType
from pan.memory import gate_prompt as gp
from pan.memory.classifier import INFERRED, OBSERVED, RuleClassifier
from pan.memory.dates import bare
from pan.memory.episodes import Episode
from pan.memory.llm_gate import (GATE_VERSION, GateError, LocalLLMClassifier, ShadowClassifier, build_classifier,
                                 build_input, grounded, parse_decision, post_check, trivial)
from pan.memory.secrets import contains_secret, redact
from pan.paths import PanPaths

# Built at runtime so the secret scanner of the repo (gitleaks) never sees a token-shaped literal.
FAKE_KEY = "sk-" + "t3st" + "Q9x7Lm2Pv8Rz4Kd1"


def _ev(n: int, etype: EventType, content: dict, session: str = "s") -> AgentEvent:
    return AgentEvent(id=f"01K0GATE{n:018d}", ts="2026-09-24T10:00:00Z", event_type=etype, session_id=session,
                      actor=Actor.MAIN_AGENT, content=content)


def _turn(n: int, user: str, assistant: str = "", **extra) -> AgentEvent:
    return _ev(n, EventType.TURN, {"user": user, "assistant": assistant, **extra})


def _tool(n: int, tool: str = "terminal", status: str = "ok", result: str = "", **args) -> AgentEvent:
    return _ev(n, EventType.TOOL_CALL, {"tool": tool, "args": args, "status": status, "result_excerpt": result,
                                        "error_message": result if status == "error" else None})


def _episode(*events: AgentEvent) -> Episode:
    return Episode("s", list(events), closed_by="turn")


def _decision(*cands: dict, record: bool = True, why: str = "test") -> dict:
    return {"why": why, "record": record, "record_confidence": 0.9, "language": "en", "sensitivity": "none",
            "candidates": list(cands)}


def _cand(kind: str, claims, *, domain: str = "environment", subject: str = "", title: str = "",
          lifetime: str = "long_term", volatile: bool = False, corrects: bool = False) -> dict:
    return {"kind": kind, "domain": domain, "lifetime": lifetime, "volatile": volatile, "corrects": corrects,
            "importance": 0.7, "confidence": 0.8, "subject": subject, "title": title,
            "claims": [{"text": t, "evidence": e, "supersedes": s} for t, e, *rest in claims
                       for s in [rest[0] if rest else ""]]}


def _check(decision: dict, episode: Episode, previous: Episode = None, known: str = ""):
    gi = build_input(episode, previous, known)
    notes: list = []
    return post_check(decision, episode, previous, gi, notes), notes


# -- input -----------------------------------------------------------------------------------------------

def test_input_sections_redaction_and_recall():
    ep = _episode(
        _tool(1, "memory_search", result="The backup job writes to /mnt/old.", query="backup"),
        _tool(2, "terminal", result="active (running)", command="systemctl status backupd"),
        _ev(3, EventType.FILE_CHANGE, {"path": "/etc/backupd.conf", "op": "replace"}),
        _turn(4, f"My token for the backup API is {FAKE_KEY}. Is the service up?",
              'Yes, backupd is running.\n{"name": "terminal", "arguments": {"command": "ls"}}',
              recall="Prefetched: backupd listens on 7300."))
    gi = build_input(ep, None, "USER.md: prefers short answers")
    assert FAKE_KEY not in gi.prompt and "[secret:api_key]" in gi.prompt and gi.secrets == ["api_key"]
    assert "<recalled>" in gi.prompt and "/mnt/old" in gi.prompt and "7300" in gi.prompt
    assert "<known>" in gi.prompt and "prefers short answers" in gi.prompt
    tools = gi.prompt.split("<tools>")[1].split("</tools>")[0]
    assert "memory_search" not in tools and "`systemctl status backupd` ok: active (running)" in tools
    assert "replace /etc/backupd.conf" in gi.prompt
    assert '"arguments"' not in gi.prompt and gi.printed_call  # printed tool call stripped
    assert FAKE_KEY not in gi.sources.user and "/etc/backupd.conf" in gi.sources.tools


def test_input_truncation_is_logged_and_capped():
    long_out = "\n".join(f"{i}|line {i} of the log" for i in range(3000))
    ep = _episode(_tool(1, "read_file", result=json.dumps({"content": long_out}), path="app.log"),
                  _turn(2, "x " * 5000, "ok"))
    gi = build_input(ep, None, "", max_chars=12000)
    assert len(gi.prompt) <= 12000
    assert "user" in gi.truncated and "tool1_output" in gi.truncated
    assert "chars cut]" in gi.prompt and "|line" not in gi.prompt   # read_file line numbers removed


def test_truncated_tool_output_keeps_error_lines():
    lines = [f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z INFO worker: heartbeat {i}" for i in range(400)]
    lines[200] = "2026-01-01T00:03:20Z ERROR export: write rejected, quota exceeded on bucket 'b1'"
    ep = _episode(_tool(1, "read_file", result="\n".join(lines), path="app.log"), _turn(2, "why did it fail?", "quota"))
    gi = build_input(ep, None, "")
    assert "quota exceeded on bucket 'b1'" in gi.prompt and "error/warning line(s) kept" in gi.prompt
    assert "tool1_output" in gi.truncated


def test_trivial_episode_is_pre_skipped():
    assert trivial(_episode(_tool(1, "memory_search", result="x", query="q")))
    assert not trivial(_episode(_turn(1, "yes")))


# -- parsing ----------------------------------------------------------------------------------------------

def test_parse_decision_tolerates_fences_and_think():
    obj = _decision()
    assert parse_decision("```json\n" + json.dumps(obj) + "\n```")["record"] is True
    assert parse_decision("<think>hmm</think>" + json.dumps(obj))["why"] == "test"
    for bad in ("", "no json here", '{"record": "yes"}', '{"record": true, "candidates": {}}', "{broken"):
        with pytest.raises(GateError) as exc:
            parse_decision(bad)
        assert exc.value.reason == "invalid_output"


def test_schema_is_strict():
    schema = gp.output_schema()
    cand = schema["properties"]["items"]["items"]
    assert schema["additionalProperties"] is False and cand["additionalProperties"] is False
    # only default-valued fields are optional (compact output: omitted unless they apply)
    assert set(schema["properties"]) - set(schema["required"]) == {"sens"}
    assert set(cand["properties"]) - set(cand["required"]) == {"corrects"}
    claim = cand["properties"]["claims"]["items"]
    assert set(claim["properties"]) - set(claim["required"]) == {"old"}
    assert cand["properties"]["kind"]["enum"] == list(gp.KINDS)
    assert cand["properties"]["claims"]["items"]["properties"]["ev"]["enum"] == ["user", "confirmed", "tool", "inferred"]


def test_wire_format_maps_to_internal_fields():
    wire = {"record": True, "conf": 0.8, "why": "w", "sens": "none", "items": [
        {"kind": "fact", "domain": "project", "life": "session", "corrects": True, "subject": "s",
         "claims": [{"text": "t", "ev": "tool", "old": "o"}]}]}
    d = gp.from_wire(wire)
    (c,) = d["candidates"]
    assert d["record_confidence"] == 0.8 and c["lifetime"] == "session" and c["volatile"] is True
    assert c["claims"] == [{"text": "t", "evidence": "tool_observed", "supersedes": "o"}]
    assert parse_decision(json.dumps(wire))["candidates"][0]["corrects"] is True


# -- post-checks ----------------------------------------------------------------------------------------------

def test_grounded_requires_overlap_and_all_values():
    src = "Our billing service now listens on port 7002."
    assert grounded("The billing service listens on port 7002.", src)
    assert not grounded("The billing service listens on port 7003.", src)      # value not in source
    assert not grounded("Billing is written in Rust and deployed weekly.", src)


def test_preference_grounded_goes_to_user_md():
    ep = _episode(_turn(1, "From now on, keep commit messages under 60 characters.", "Will do."))
    cands, _ = _check(_decision(_cand("preference", [("Keep commit messages under 60 characters.", "user_stated")],
                                      domain="other")), ep)
    (c,) = cands
    assert c.classification.type is MemoryType.USER_PREFERENCE and c.classification.destination is Destination.USER
    assert c.claim_evidence == [OBSERVED] and c.id == f"{ep.id}:user_preference"
    assert c.rationale.startswith(GATE_VERSION)


def test_preference_paraphrase_or_inferred_is_not_stored():
    ep = _episode(_turn(1, "Bitte antworte ab jetzt immer auf Deutsch.", "Alles klar."))
    cands, notes = _check(_decision(_cand("preference", [("User wants German replies.", "user_stated")],
                                          domain="other")), ep)
    # G3: a paraphrase that does not ground falls back to the user's own preference sentence
    assert cands[0].classification.destination is Destination.USER
    assert cands[0].normalized_claims == ["Bitte antworte ab jetzt immer auf Deutsch."]
    assert any("own sentence" in n for n in notes)
    cands, notes = _check(_decision(_cand("preference", [("User wants German replies.", "assistant_inferred")],
                                          domain="other")), ep)
    assert [c.classification.type for c in cands] == [MemoryType.NOISE]   # never from the assistant
    # German, close to the user's words: accepted
    cands, _ = _check(_decision(_cand("preference", [("Antworte immer auf Deutsch.", "user_stated")],
                                      domain="other")), ep)
    assert cands[0].classification.destination is Destination.USER


def test_one_off_instruction_is_not_long_term():
    ep = _episode(_turn(1, "Don't run the tests now, just commit the change.", "Committed."))
    cands, _ = _check(_decision(_cand("preference", [("Don't run the tests now, just commit the change.",
                                                      "user_stated")], domain="other", lifetime="session")), ep)
    (c,) = cands
    assert c.classification.destination is Destination.NONE and not c.classification.should_remember
    assert c.classification.lifetime is Lifetime.SESSION


def test_secret_in_claim_is_redacted():
    user = f"My API key for the staging server is {FAKE_KEY}, the server runs on port 9000."
    ep = _episode(_turn(1, user, "Noted."))
    # the model sees the redacted text; even if it copied a key, the output scan removes it
    dec = _decision(_cand("fact", [(f"The staging API key is {FAKE_KEY}.", "user_stated"),
                                   ("The staging server runs on port 9000.", "user_stated")],
                          subject="staging server"))
    cands, notes = _check(dec, ep)
    blob = json.dumps([c.normalized_claims for c in cands])
    assert FAKE_KEY not in blob and "[secret:api_key]" in blob
    assert "The staging server runs on port 9000." in [bare(x) for x in cands[0].normalized_claims]
    assert dec["sensitivity"] == "secret" and any("secret redacted" in n for n in notes)


def test_hallucinated_value_is_dropped():
    ep = _episode(_turn(1, "Note for later: our Grafana container listens on port 3100.", "Noted."))
    dec = _decision(_cand("fact", [("The Grafana container listens on port 3000.", "user_stated")],
                          subject="Grafana container"))
    cands, notes = _check(dec, ep)
    assert [c.classification.type for c in cands] == [MemoryType.NOISE]
    assert any("ungrounded" in n for n in notes)


def test_inferred_world_knowledge_without_tool_support_is_dropped():
    ep = _episode(_tool(1, "terminal", result="NVIDIA GB10", command="nvidia-smi --query-gpu=name"),
                  _turn(2, "which GPU is this?",
                        "This machine has an NVIDIA GB10 GPU, the Grace Blackwell chip used in compact AI boxes."))
    dec = _decision(_cand("fact", [("The GB10 is the Grace Blackwell chip used in compact AI boxes.",
                                    "assistant_inferred")], subject="GPU"))
    cands, notes = _check(dec, ep)
    assert [c.classification.type for c in cands] == [MemoryType.NOISE]
    dec = _decision(_cand("fact", [("This machine has an NVIDIA GB10 GPU.", "tool_observed")], subject="GPU"))
    cands, _ = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.ENVIRONMENT and cands[0].claim_evidence == [OBSERVED]
    assert cands[0].subject == "GPU"


def test_inferred_claim_supported_by_tool_output_survives():
    ep = _episode(_tool(1, "terminal", result="vllm serve m --max-model-len 32768", command="docker inspect vllm"),
                  _turn(2, "why do tool calls fail?",
                        "The vLLM server was started without the tool-call parser, so tool calling is off."))
    dec = _decision(_cand("learning", [("The vLLM server was started without the tool-call parser.",
                                        "assistant_inferred")], domain="configuration", subject="vLLM server"))
    cands, _ = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.NOISE   # no tool support for the "without" claim
    dec = _decision(_cand("fact", [("The vLLM server runs with --max-model-len 32768.", "assistant_inferred")],
                          subject="vLLM server"))
    cands, _ = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.ENVIRONMENT and cands[0].claim_evidence == [INFERRED]


def test_echo_of_recalled_text_is_dropped():
    ep = _episode(_turn(1, "what port does the cache use?", "The Redis cache listens on port 6380.",
                        recall="The Redis cache listens on port 6380."))
    dec = _decision(_cand("fact", [("The Redis cache listens on port 6380.", "assistant_inferred")]))
    cands, notes = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.NOISE and any("echo" in n or "ungrounded" in n for n in notes)


def test_several_candidates_per_episode():
    user = "Decision: we use SQLite for the job queue. Also note our Grafana container listens on port 3100."
    ep = _episode(_turn(1, user, "OK."))
    dec = _decision(_cand("decision", [("We use SQLite for the job queue.", "user_stated")], domain="architecture",
                          title="SQLite for the job queue"),
                    _cand("fact", [("Our Grafana container listens on port 3100.", "user_stated")],
                          subject="Grafana container"))
    cands, _ = _check(dec, ep)
    assert [c.classification.type for c in cands] == [MemoryType.DECISION, MemoryType.ENVIRONMENT]
    assert cands[1].subject == "Grafana container" and cands[1].context == ""
    assert len({c.id for c in cands}) == 2


def test_confirmed_decision_links_previous_turn():
    prev = Episode("s", [_turn(1, "Where should metrics go?",
                               "Proposal: we keep the metrics in TimescaleDB instead of InfluxDB.")], closed_by="turn")
    ep = _episode(_turn(2, "Passt, machen wir so."))
    dec = _decision(_cand("decision", [("We keep the metrics in TimescaleDB instead of InfluxDB.", "user_confirmed")],
                          domain="architecture", title="Metrics in TimescaleDB"))
    cands, _ = _check(dec, ep, prev)
    (c,) = cands
    assert c.classification.type is MemoryType.DECISION and c.claim_evidence == [OBSERVED]
    assert c.event_ids == prev.event_ids + ep.event_ids and "confirmed by the user" in c.context
    cands, _ = _check(dec, ep, None)   # no previous turn: cannot be a confirmation
    assert cands[0].classification.type is MemoryType.NOISE


def test_correction_passes_supersedes_only_when_all_observed():
    prev = Episode("s", [_turn(1, "The release freeze starts on May 12.", "Noted.")], closed_by="turn")
    ep = _episode(_turn(2, "Scratch that, it starts on May 19.", "Updated."))
    dec = _decision(_cand("fact", [("The release freeze starts on May 19.", "user_stated",
                                    "The release freeze starts on May 12.")],
                          domain="project", corrects=True, subject="release freeze"))
    cands, _ = _check(dec, ep, prev)
    (c,) = cands
    assert c.classification.type is MemoryType.PROJECT_FACT and c.claim_evidence == [OBSERVED]
    assert c.supersedes == ["The release freeze starts on May 12."]


def test_volatile_state_is_not_stored_and_record_false_is_noise():
    ep = _episode(_turn(1, "FYI the Langfuse instance is down right now.", "Thanks."))
    dec = _decision(_cand("fact", [("The Langfuse instance is down right now.", "user_stated")], volatile=True))
    (c,) = _check(dec, ep)[0]
    assert c.classification.destination is Destination.NONE
    (n,) = _check(_decision(record=False, why="chit-chat"), ep)[0]
    assert n.classification.type is MemoryType.NOISE and n.id == f"{ep.id}:noise" and "chit-chat" in n.rationale


def test_unknown_kind_and_garbage_claims_are_ignored():
    ep = _episode(_turn(1, "Our NAS backup target is called vault-02.", "ok"))
    dec = _decision({"kind": "gossip", "claims": []}, _cand("fact", [("Our NAS backup target is called vault-02.",
                                                                      "user_stated")], domain="bogus"))
    dec["candidates"][1]["claims"].append("not a dict")
    cands, notes = _check(dec, ep)
    assert [c.classification.type for c in cands] == [MemoryType.PROJECT_FACT]   # domain "bogus" → other
    assert any("unknown kind" in n for n in notes)


# -- secrets --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    (f"key is {FAKE_KEY}", "api_key"),
    ("postgres://app:" + "Pw" + "9secret" + "@localhost:5432/app", "url_credentials"),
    ("password=" + "hunter" + "22x", "credential"),
    ("Das Passwort lautet " + "Sommer" + "2031!", "credential"),
    ("Authorization: Bearer " + "abcDEF123" + "ghiJKL456mno", "bearer"),
])
def test_secrets_are_redacted(text, kind):
    out, kinds = redact(text)
    assert kind in kinds and "[secret:" in out and "[secret:[secret" not in out


@pytest.mark.parametrize("text", ["token limit is 128000", "max_tokens: 700", "The primary key is customer_id",
                                  "api_key: EMPTY", "the key is stored in ~/.config/app/key",
                                  "vLLM runs on port 8000"])
def test_no_false_positive_secrets(text):
    assert not contains_secret(text)


# -- config ------------------------------------------------------------------------------------------------------

def test_shipped_default_is_the_llm_gate(pin_rules_classifier, monkeypatch):
    import pan.config
    assert pin_rules_classifier == "llm"
    monkeypatch.setattr(pan.config, "DEFAULT_CLASSIFIER_KIND", pin_rules_classifier)
    assert PanConfig().classifier.kind == "llm" and isinstance(build_classifier(PanConfig()), LocalLLMClassifier)
    assert isinstance(build_classifier(PanConfig()).fallback, RuleClassifier)


def test_classifier_config_defaults_and_nested_models():
    cfg = PanConfig()   # pinned to rules by conftest
    assert cfg.classifier.kind == "rules" and cfg.models.gate.model == "primary"
    assert cfg.models.gate.base_url == "http://localhost:8000/v1" and cfg.models.gate.thinking is False
    cfg = parse_config({"classifier": {"kind": "llm", "timeout_s": "30"},
                        "models": {"gate": {"model": "gate-model", "temperature": 0, "bogus": 1}}})
    assert cfg.classifier.kind == "llm" and cfg.classifier.timeout_s == 30.0
    assert cfg.models.gate.model == "gate-model" and cfg.models.gate.base_url == "http://localhost:8000/v1"


def test_hermes_config_pan_section_is_a_lower_layer(tmp_path: Path):
    paths = PanPaths.for_home(tmp_path)
    paths.root.mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: pan\npan:\n  classifier:\n    kind: llm\n  models:\n    gate:\n      model: g1\n")
    assert load_config(paths).classifier.kind == "llm" and load_config(paths).models.gate.model == "g1"
    paths.config.write_text("classifier:\n  kind: shadow\ndaemon:\n  poll_seconds: 5\n")
    cfg = load_config(paths)
    assert cfg.classifier.kind == "shadow" and cfg.models.gate.model == "g1" and cfg.daemon.poll_seconds == 5.0


def test_default_config_file_does_not_mask_hermes_section(tmp_path: Path):
    from pan.config import write_default_config
    paths = PanPaths.for_home(tmp_path)
    assert write_default_config(paths)   # what `pan setup` writes
    (tmp_path / "config.yaml").write_text("pan:\n  classifier:\n    kind: llm\n")
    assert load_config(paths).classifier.kind == "llm"


def test_build_classifier_kinds():
    assert isinstance(build_classifier(PanConfig()), RuleClassifier)
    gate = build_classifier(parse_config({"classifier": {"kind": "llm"}}))
    assert isinstance(gate, LocalLLMClassifier) and gate.name == GATE_VERSION
    shadow = build_classifier(parse_config({"classifier": {"kind": "shadow"}}))
    assert isinstance(shadow, ShadowClassifier) and shadow.name == "rules-v2"
    assert isinstance(build_classifier(parse_config({"classifier": {"kind": "magic"}})), RuleClassifier)


# -- recorded responses on the golden fixtures (§5.2: every fixture ≥ rules) ----------------------------------

def _fixture(name: str):
    from conftest import scenario_episodes
    from pan.memory.episodes import previous_turn
    eps = scenario_episodes(name)
    return [(ep, previous_turn(eps, i)) for i, ep in enumerate(eps)]


def _gate_types(name: str, index: int, decision: dict, known: str = ""):
    ep, prev = _fixture(name)[index]
    return _check(decision, ep, prev, known)[0], ep, prev


def test_fixture_decision_confirmed():
    cands, ep, prev = _gate_types("decision", 1, _decision(_cand("decision", [
        ("Run curation in a separate background service (pan-memoryd); the plugin only writes events to a "
         "durable spool.", "user_confirmed")], domain="architecture", title="Run curation in a separate service")))
    assert [c.classification.type for c in cands] == [MemoryType.DECISION]
    assert cands[0].claim_evidence == [OBSERVED] and cands[0].event_ids == prev.event_ids + ep.event_ids


def test_fixture_fact_update_and_answer_from_l1():
    eps = _fixture("fact_update")
    turns = [(ep, prev) for ep, prev in eps if ep.turn is not None]
    (ep1, p1), (ep2, p2), (ep3, p3) = turns
    c1 = _check(_decision(_cand("fact", [("Our vLLM server runs on port 8000.", "user_stated")],
                                subject="vLLM server")), ep1, p1)[0]
    assert c1[0].subject == "vLLM server" and c1[0].claim_evidence == [OBSERVED]
    dec2 = _decision(_cand("fact", [("We moved the vLLM server to port 8010.", "user_stated",
                                     "vLLM server runs on port 8000.")], corrects=True, subject="vLLM server"))
    c2 = _check(json.loads(json.dumps(dec2)), ep2, p2, known="vLLM server runs on port 8000.")[0]
    assert [bare(x) for x in c2[0].normalized_claims] == ["We moved the vLLM server to port 8010."]
    assert c2[0].supersedes == ["vLLM server runs on port 8000."]
    c2, notes = _check(dec2, ep2, p2)   # the old statement is nowhere in the context: not passed on
    assert c2[0].supersedes == [] and any("not found in context" in n for n in notes)
    c3 = _check(_decision(_cand("fact", [("Our vLLM server runs on port 8000.", "assistant_inferred")])),
                ep3, p3, known="vLLM server runs on port 8000.")[0]
    assert [c.classification.type for c in c3] == [MemoryType.NOISE]   # answer from L1 is an echo


def test_fixture_gpu_and_printed_calls():
    cands, _, _ = _gate_types("gpu_fact", 0, _decision(_cand("fact", [
        ("The machine has an NVIDIA GB10 GPU.", "tool_observed")], subject="GPU")))
    assert cands[0].classification.type is MemoryType.ENVIRONMENT and cands[0].claim_evidence == [OBSERVED]
    # the docs link existed only inside a printed tool call: nothing grounds it
    cands, _, _ = _gate_types("toolcall_text", 0, _decision(_cand("fact", [
        ("vLLM server docs: https://docs.vllm.ai", "assistant_inferred")], domain="tooling")))
    assert cands[0].classification.type is MemoryType.NOISE
    cands, _, _ = _gate_types("toolcall_text", 1, _decision(_cand("decision", [
        ("We use SQLite, not Postgres, for the bench results index.", "user_stated")], domain="architecture")))
    assert cands[0].classification.type is MemoryType.DECISION
    cands, _, _ = _gate_types("toolcall_text", 2, _decision(_cand("fact", [
        ("The Grafana container listens on port 3100.", "user_stated")], subject="Grafana container")))
    assert cands[0].classification.type is MemoryType.ENVIRONMENT and cands[0].subject == "Grafana container"


# -- rules-r2 precision guards also bind the gate --------------------------------------------------------------

def test_hypothetical_and_third_party_claims_are_dropped():
    ep = _episode(_turn(1, "Hypothetically, a second GPU node would cost 9000 euros.", "Sounds pricey."))
    cands, notes = _check(_decision(_cand("fact", [("A second GPU node would cost 9000 euros.", "user_stated")])), ep)
    assert cands[0].classification.type is MemoryType.NOISE and any("hedged" in n for n in notes)
    ep = _episode(_turn(1, "My neighbour's NAS (not ours) exports /volume1 over NFS; we have no NAS.", "OK."))
    dec = _decision(_cand("fact", [("The NAS exports /volume1 over NFS.", "user_stated")], subject="NAS"))
    cands, notes = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.NOISE and any("third-party" in n for n in notes)


def test_inferred_claim_must_not_introduce_a_value():
    ep = _episode(_tool(1, "terminal", result="scheduler active", command="systemctl status scheduler"),
                  _turn(2, "is the scheduler up?", "The scheduler service is running with 16 workers."))
    dec = _decision(_cand("fact", [("The scheduler service is running with 16 workers.", "assistant_inferred")],
                          subject="scheduler service"))
    cands, notes = _check(dec, ep)
    assert cands[0].classification.type is MemoryType.NOISE and any("value nobody else" in n for n in notes)


# -- G2 grounding / attribution (own wording; validation failure shapes) ----------------------------------

def test_first_party_fact_next_to_an_object_pronoun_is_kept():
    ep = _episode(_turn(1, "Write a note for the team: since Omar Haddad became our release captain in June he "
                           "approves the tags, so ping him too.", "Done."))
    dec = _decision(_cand("fact", [("Omar Haddad became our release captain in June.", "user_stated")],
                          domain="project", subject="release captain"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.PROJECT_FACT
    ep = _episode(_turn(1, "FYI: deploy.toml still says 3 replicas, but that's outdated; we run 5 now.", "OK."))
    dec = _decision(_cand("fact", [("We run 5 replicas now.", "user_stated")], subject="replicas"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.ENVIRONMENT   # "still says" is not hearsay


def test_other_peoples_things_stay_out_even_inside_an_enumeration():
    ep = _episode(_turn(1, "Meine Kollegin Ines baut gerade ein NAS mit acht Platten und 10-GbE-Karte. "
                           "Mein eigenes NAS hat zwei Platten.", "Klingt gut."))
    dec = _decision(_cand("fact", [("Ines' NAS hat eine 10-GbE-Karte.", "user_stated"),
                                   ("Mein eigenes NAS hat zwei Platten.", "user_stated")], subject="NAS"))
    (c,), notes = _check(dec, ep)
    assert [bare(x) for x in c.normalized_claims] == ["Mein eigenes NAS hat zwei Platten."] and any("third-party" in n for n in notes)


def test_values_with_units_and_german_claims_ground_in_tool_output():
    out = json.dumps({"sensors": [{"id": "P2", "offset_bar": -0.12, "range": [0, 16]}]})
    ep = _episode(_tool(1, "read_file", result=out, path="anlage/drucksensoren.json"),
                  _turn(2, "Lies die Datei und nenn mir Offset und Bereich von P2.", "Offset -0,12 bar, Bereich 0 bis 16."))
    dec = _decision(_cand("fact", [("Sensor P2 hat einen Offset von -0,12 bar.", "tool_observed"),
                                   ("Der Messbereich von Sensor P2 ist 0 bis 16.", "tool_observed")],
                          domain="configuration", subject="Sensor P2"))
    (c,) = _check(dec, ep)[0]
    assert c.claim_evidence == [OBSERVED, OBSERVED]
    ep = _episode(_tool(1, "read_file", result="2026-03-01 outage duration=640s min_charge=41%", path="ups.log"),
                  _turn(2, "longest outage?", "640 seconds, lowest charge 41 %."))
    dec = _decision(_cand("fact", [("The longest outage in ups.log lasted 640 seconds.", "tool_observed")]))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]
    dec = _decision(_cand("fact", [("The longest outage in ups.log lasted 650 seconds.", "tool_observed")]))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE   # a wrong value never grounds


def test_update_may_carry_context_from_memory_but_not_new_values():
    ep = _episode(_turn(1, "The Brightwater deadline moved to 3 March. The page limit is unchanged.", "Noted.",
                        recall="The Brightwater Trust application is due 14 February, max 12 pages."))
    dec = _decision(_cand("fact", [("The Brightwater Trust application is due 3 March.", "user_stated")],
                          domain="project", corrects=True, subject="Brightwater Trust application"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]
    dec = _decision(_cand("fact", [("The Brightwater Trust application is due 3 March, max 10 pages.", "user_stated")],
                          domain="project", subject="Brightwater Trust application"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE   # "10" is nobody's value
    ep = _episode(_turn(1, "Is the Brightwater Trust application still due 14 February?", "Yes.",
                        recall="The Brightwater Trust application is due 14 February, max 12 pages."))
    dec = _decision(_cand("fact", [("The Brightwater Trust application is due 14 February.", "user_stated")],
                          domain="project"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE   # recall echoed back, not a statement


# -- G2 pre-skip (no model call) -------------------------------------------------------------------------------

@pytest.mark.parametrize("user,skip", [
    ("Which port does the ticket service use?", True),
    ("Welcher Rechner steht im Keller? Und seit wann?", True),
    ("thanks!", True),
    ("Danke schön", True),
    ("ok, go ahead", False),                                        # may confirm a proposal
    ("Could you always answer in German from now on?", False),      # standing wording in a question
    ("Is it true that the ticket service moved to port 7100?", False),   # typed value
    ("Why did we move the queue to the new host?", False),          # change verb
    ("The ticket service moved to port 7100. Can you update the docs?", False),
    # a first-person fact inside the question: the gate is asked
    ("What are some easy ways to keep my car clean with a toddler who spills everything?", False),
    ("Can you suggest sci-fi novels like Hyperion, which I just finished?", False),
    ("I've got a sore knee, can you suggest some low-impact workouts?", False),
    ("I'm ordering Pepper a new harness, which size fits a Border Collie like Pepper?", False),
    # pure questions, questions about the user's thing and memory questions stay skipped
    ("Can you explain how DNS caching works?", True),
    ("How do I list open files on Linux?", True),
    ("What kind of bike do I have?", True),
    ("Can you remind me when my dentist appointment is?", True),
    ("Which region does our object storage live in?", True),
])
def test_pre_skip_only_for_questions_and_thanks(user, skip):
    from pan.memory.llm_gate import pre_skip
    assert bool(pre_skip(_episode(_turn(1, user, "Sure.")))) is skip


def test_pre_skip_never_with_tool_output_or_file_changes():
    from pan.memory.llm_gate import pre_skip
    ep = _episode(_tool(1, "terminal", result="v2.4.1", command="app --version"), _turn(2, "Which version is it?", "2.4.1"))
    assert pre_skip(ep) == ""
    ep = _episode(_ev(1, EventType.FILE_CHANGE, {"path": "a.yaml", "op": "write"}), _turn(2, "Done?", "Yes."))
    assert pre_skip(ep) == ""


def test_update_may_copy_a_year_from_memory_but_needs_a_user_value():
    ep = _episode(_turn(1, "The Brightwater deadline moved to 3 March.", "Noted.",
                        recall="The Brightwater Trust application is due 14 February 2027."))
    ok = _decision(_cand("fact", [("The Brightwater Trust application is due 3 March 2027.", "user_stated")],
                         domain="project"))
    assert _check(ok, ep)[0][0].claim_evidence == [OBSERVED]
    old = _decision(_cand("fact", [("The Brightwater Trust application was due 14 February 2027.", "user_stated")],
                          domain="project"))
    assert _check(old, ep)[0][0].classification.type is MemoryType.NOISE


# -- G3 noise precision (own wording, EN + DE) -----------------------------------------------------------------

@pytest.mark.parametrize("user,claim", [
    ("Just thinking out loud: if we ever put the NAS in the attic we'd need a second fan. Nothing's planned.",
     "The NAS would need a second fan in the attic."),
    ("Should our choir move rehearsals to Thursdays at 18:00? Only asking, nothing is decided.",
     "Choir rehearsals move to Thursdays at 18:00."),
    ("Oh sure, the printer queue takes 'just' an hour, lightning fast. (Really it's about 4 minutes.)",
     "The printer queue takes an hour."),
    ("Ha, the old toaster is basically our most reliable build agent.", "The old toaster is our build agent."),
    ("We should rename the repo to Tortoise, ha.", "The repo is renamed to Tortoise."),
    ("Mein Nachbar betreibt sein Heimnetz mit einem 48-Port-Switch; wir haben gar keinen Switch.",
     "Das Heimnetz hat einen 48-Port-Switch."),
    ("Wenn wir mal umziehen, bräuchten wir wahrscheinlich 3 Router. Noch nichts entschieden.",
     "Wir brauchen 3 Router."),
    ("Mein Kollege Timo baut gerade einen Rechner mit 128 GB RAM und zwei GPUs.", "Der Rechner hat 128 GB RAM."),
    ("He says the staging box has 64 GB of RAM.", "The staging box has 64 GB of RAM."),
])
def test_noise_statements_are_not_recorded(user, claim):
    ep = _episode(_turn(1, user, "Got it."))
    dec = _decision(_cand("fact", [(claim, "user_stated")]))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE


def test_honest_part_after_sarcasm_and_own_part_next_to_a_colleagues_are_kept():
    ep = _episode(_turn(1, "Oh sure, the printer queue takes 'just' an hour, lightning fast. "
                           "(Really it's about 4 minutes.)", "OK."))
    dec = _decision(_cand("fact", [("The printer queue takes about 4 minutes.", "user_stated")], subject="printer queue"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]
    ep = _episode(_turn(1, "Mein Kollege Timo baut einen Rechner mit 128 GB RAM. Mein eigener hat 32 GB RAM.", "OK."))
    dec = _decision(_cand("fact", [("Mein eigener Rechner hat 32 GB RAM.", "user_stated")], subject="Rechner"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]
    ep = _episode(_turn(1, "Heads-up: deploy.toml still says 3 workers, but we run 5 now.", "OK."))
    dec = _decision(_cand("fact", [("We run 5 workers now.", "user_stated")], subject="workers"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]    # a file "says", not a person


# -- G3 personal facts, dates, notes files -----------------------------------------------------------------------

def test_personal_fact_with_people_mentioned_is_the_users_and_gets_dates():
    ep = _episode(_turn(1, "[Today is Mon 2024-03-11] For my nephew's graduation I got him a fountain pen last "
                           "Saturday. Any tips for a card text?", "Sure."))
    dec = _decision(_cand("fact", [("The user got their nephew a fountain pen for his graduation last Saturday.",
                                    "user_stated")], domain="personal", subject="graduation gift"))
    (c,) = _check(dec, ep)[0]
    assert c.claim_evidence == [OBSERVED] and c.classification.type is MemoryType.PROJECT_FACT
    assert c.normalized_claims == ["The user got their nephew a fountain pen for his graduation last Saturday "
                                   "(2024-03-09). (stated 2024-03-11)"]


def test_someone_elses_fact_without_the_user_acting_stays_out():
    ep = _episode(_turn(1, "[Today is Mon 2024-03-11] My brother's van has 210,000 km on it. Should I buy it?", "Hm."))
    dec = _decision(_cand("fact", [("The user's brother's van has 210,000 km on it.", "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE


def test_preferences_get_no_stated_suffix():
    ep = _episode(_turn(1, "[Today is Mon 2024-03-11] Lately I always cook with fresh coriander from my balcony.", "Nice."))
    dec = _decision(_cand("preference", [("The user always cooks with fresh coriander from their balcony.",
                                         "user_stated")], domain="personal"))
    (c,) = _check(dec, ep)[0]
    assert c.classification.destination is Destination.USER and "(stated" not in c.normalized_claims[0]


def test_a_read_notes_file_is_shown_nearly_whole():
    notes = "# Chat excerpt\n\n## You (assistant)\n\n" + "\n".join(f"{i}. Step {i}: do thing {i} for {i + 5} minutes."
                                                                     for i in range(1, 90))
    assert 2500 < len(notes) < 4000
    ep = _episode(_tool(1, "read_file", result=notes, path="notes/chat.md"), _turn(2, "Summarise notes/chat.md.", "Done."))
    gi = build_input(ep, None, "")
    assert "tool1_output" not in gi.truncated and "Step 45: do thing 45 for 50 minutes." in gi.prompt


def test_translated_claims_fall_back_to_the_users_own_sentence():
    ep = _episode(_turn(1, "Ab sofort: Antworte mir bitte immer auf Englisch, höchstens 60 Wörter. Was ist DNS?", "OK."))
    dec = _decision(_cand("preference", [("Always answer in English, at most 60 words.", "user_stated")], domain="other"))
    (c,) = _check(dec, ep)[0]
    assert c.classification.destination is Destination.USER
    assert c.normalized_claims == ["Antworte mir bitte immer auf Englisch, höchstens 60 Wörter."]
    ep = _episode(_turn(1, "Mein Nachbar hat eine Solaranlage. Mein eigener Router hängt an einer 12-V-USV.", "OK."))
    dec = _decision(_cand("fact", [("The user's own router is powered by a 12-V UPS battery.", "user_stated")],
                          domain="personal"))
    (c,) = _check(dec, ep)[0]
    assert [bare(x) for x in c.normalized_claims] == ["Mein eigener Router hängt an einer 12-V-USV."]
    ep = _episode(_turn(1, "Mein Nachbar hat eine 12-V-USV am Router.", "OK."))   # the only sentence is someone else's
    dec = _decision(_cand("fact", [("The user's router is connected to a 12-V UPS.", "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE


def test_later_turns_of_a_session_keep_the_session_date():
    from pan.memory.llm_gate import claim_day
    import datetime as dt
    first = Episode("s", [_turn(1, "Today is 11 March 2024. I repainted my bike yesterday.", "Nice.")], closed_by="turn")
    second = Episode("s", [_turn(2, "It took me three hours.", "OK.")], closed_by="turn")
    # the event timestamps of these turns are 2026-09-24: the stated date wins, later turns keep the offset
    assert claim_day(first, None) == dt.date(2024, 3, 11)
    assert claim_day(second, first) == dt.date(2024, 3, 11)
    assert claim_day(second, None, (dt.date(2024, 3, 11) - dt.date(2026, 9, 24)).days) == dt.date(2024, 3, 11)
    assert claim_day(second, None) == dt.date(2026, 9, 24)   # no stated date anywhere: the event timestamp


def test_explicit_standing_preference_survives_a_gate_that_skipped_it():
    from pan.memory.llm_gate import _keep_explicit_preference
    ep = _episode(_turn(1, "For all future chats: sign every answer with a short haiku.", "I'd rather not do that."))
    cands, notes = _check(_decision(record=False, why="assistant declined"), ep)
    out = _keep_explicit_preference(cands, ep, notes)
    assert [c.classification.type for c in out] == [MemoryType.USER_PREFERENCE]
    assert out[0].classification.destination is Destination.USER
    ep = _episode(_turn(1, "Can you sign this answer with a haiku?", "Sure."))   # no standing wording
    cands, notes = _check(_decision(record=False), ep)
    assert [c.classification.type for c in _keep_explicit_preference(cands, ep, notes)] == [MemoryType.NOISE]


def test_statements_of_a_read_notes_file_are_kept_when_the_gate_skips_them():
    from pan.memory.llm_gate import _keep_read_notes
    notes = ("# Chat excerpt (2024-02-02)\n\n## Me\n\nHow do I get rid of limescale in the shower?\n\n"
             "## You (assistant)\n\n1. Spray white vinegar on the glass and wait 15 minutes.\n"
             "2. Wipe with a microfibre cloth and rinse with warm water.\n")
    ep = _episode(_tool(1, "read_file", result=notes, path="notes/chat-2024-02-02.md"),
                  _turn(2, "[Today is Fri 2024-02-09] I saved an earlier chat in notes/chat-2024-02-02.md, summarise it.",
                        "Summary: vinegar, wait, wipe."))
    cands, gnotes = _check(_decision(record=False, why="world knowledge"), ep)
    out = _keep_read_notes(cands, ep, gnotes, None)
    claims = [x for c in out for x in c.normalized_claims]
    assert any("15 minutes" in x for x in claims) and all(x.startswith("Earlier chat (notes/") for x in claims)
    assert not any("How do I get rid" in x for x in claims)          # the user's own question is not advice
    assert out[0].claim_evidence == [OBSERVED, OBSERVED]
    ep2 = _episode(_tool(1, "read_file", result="ERROR disk full at 03:00", path="logs/app.log"), _turn(2, "why?", "Disk."))
    cands, gnotes = _check(_decision(record=False), ep2)
    assert [c.classification.type for c in _keep_read_notes(cands, ep2, gnotes, None)] == [MemoryType.NOISE]


def test_a_musing_lead_in_is_not_the_same_as_thinking_about_doing_something():
    ep = _episode(_turn(1, "I was just thinking about visiting Omar, my old flatmate, who's currently in Porto.", "Nice."))
    dec = _decision(_cand("fact", [("The user's old flatmate Omar is currently in Porto.", "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.PROJECT_FACT   # a stated fact, not musing
    ep = _episode(_turn(1, "Just thinking, maybe we'd move the NAS to the attic someday.", "OK."))
    dec = _decision(_cand("fact", [("The NAS moves to the attic.", "user_stated")]))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE


def test_preference_safety_net_ignores_task_imperatives():
    from pan.memory.llm_gate import _keep_explicit_preference
    ep = _episode(_turn(1, "Which port does the proxy use? Don't run anything to find out.", "Not recorded."))
    cands, notes = _check(_decision(record=False), ep)
    assert [c.classification.type for c in _keep_explicit_preference(cands, ep, notes)] == [MemoryType.NOISE]


# -- G3 precision pass 2: personal facts vs other people's, wishes, sarcasm (own wording, EN + DE) ----------

@pytest.mark.parametrize("user,claim", [
    ("My neighbour grows chillies on her balcony. Any tips for mine?", "The neighbour grows chillies on her balcony."),
    ("By the way, my colleague Omar drives an old Saab.", "The user's colleague Omar drives an old Saab."),
    ("Meine Schwester wohnt seit Mai in Graz.", "Die Schwester des Nutzers wohnt in Graz."),
    ("My doctor said I have mild asthma, does that matter for running?", "The user has mild asthma."),
    ("Someday I'd love to hike the whole Kungsleden.", "The user will hike the Kungsleden."),
    ("If I win the raffle I'll buy a sailboat.", "The user will buy a sailboat."),
    ("Irgendwann möchte ich einen Kutter restaurieren.", "Der Nutzer restauriert einen Kutter."),
    ("Oh lovely, the dishwasher leaked again, just what I needed.", "The dishwasher leaked again."),
    ("Yeah, because I totally have 5 free evenings a week.", "The user has 5 free evenings a week."),
])
def test_noise_under_the_personal_facts_rule(user, claim):
    ep = _episode(_turn(1, user, "Hm."))
    dec = _decision(_cand("fact", [(claim, "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].classification.type is MemoryType.NOISE


@pytest.mark.parametrize("user,claim", [
    ("My sister and I hiked the Kungsleden last summer.", "The user hiked the Kungsleden with their sister last summer."),
    ("I bought my dad a cordless drill for his birthday.", "The user bought their dad a cordless drill for his birthday."),
    ("Meine Schwester und ich waren im Mai in Graz.", "Der Nutzer war im Mai mit seiner Schwester in Graz."),
    ("The manual says the filter lasts 6 months, and I changed it in May.", "The user changed the filter in May."),
])
def test_the_users_own_facts_with_people_mentioned_are_kept(user, claim):
    ep = _episode(_turn(1, user, "Nice."))
    dec = _decision(_cand("fact", [(claim, "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]


def test_a_real_fact_next_to_a_wish_in_the_same_sentence_is_kept():
    ep = _episode(_turn(1, "I just booked the ferry to Hvar for June 3rd, and I'd like to make sure I pack light.", "OK."))
    dec = _decision(_cand("fact", [("The user booked the ferry to Hvar for June 3rd.", "user_stated")], domain="personal"))
    assert _check(dec, ep)[0][0].claim_evidence == [OBSERVED]


def test_rules_only_mode_redacts_secrets():
    """classifier.kind: rules must apply the same secret filter as the gate and its fallback."""
    ep = _episode(_tool(1, "read_file", result=f"DATABASE_HOST=db.example.com\nOPENAI_API_KEY={FAKE_KEY}\n",
                        path=".env"),
                  _turn(2, f"FYI: our staging API key is {FAKE_KEY} and the API runs on port 8081.", "Noted."))
    raw = RuleClassifier().classify(ep)
    assert FAKE_KEY in repr([(c.normalized_claims, c.title, c.subject) for c in raw])  # the rules alone keep it
    for kind in ("rules", "shadow"):
        clf = build_classifier(parse_config({"classifier": {"kind": kind}}))
        if kind == "shadow":
            clf.gate.classify = lambda *a, **k: []
        cands = clf.classify(ep)
        blob = repr([(c.normalized_claims, c.title, c.retrieval_query, c.context, c.subject) for c in cands])
        assert cands and FAKE_KEY not in blob and "[secret:" in blob, kind
        assert clf.name == "rules-v2"
