"""Agent E2E (integration spec §8.1): the real Hermes agent loop with PAN, against a scripted fake model.

Runs ``hermes chat -q`` headless (subprocess, tmp ``HERMES_HOME``, ``memory.provider: pan``) against
:mod:`fake_openai`, then ``pan-memoryd`` once, then a second session. No GPU, no network.

Session 1 — the user states a preference and asks about a wiki topic; the scripted model calls
``memory_search`` and ``write_file``. Checks: PAN tools and system-prompt block offered, prefetch
``<memory-context>`` in the user message, the tool answered from the seed wiki, events in the spool.
Daemon — the preference lands in ``USER.md`` through Hermes' MemoryStore.
Session 2 — the new ``USER.md`` entry is in the frozen L1 snapshot of the next session's prompt.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from agent_harness import (hermes_unavailable_reason, last_user_message, make_home, run_hermes,  # noqa: E402
                           spool_rows, system_prompt, tool_names, tool_results)
from fake_openai import FakeOpenAI  # noqa: E402

pytestmark = pytest.mark.e2e

_REASON = hermes_unavailable_reason()
if _REASON:
    pytest.skip(f"Hermes cannot run an agent here: {_REASON}", allow_module_level=True)

TIMEOUT = 120.0
PREFERENCE = "From now on, always answer in British English spelling."
QUESTION = "Tool calls fail on vllm, is the tool call parser missing? Save a short note to notes.md."
FINAL_1 = "Yes: vLLM was started without a tool-call parser. I saved a note to notes.md."
FINAL_2 = "Understood."


def _script(work: Path) -> list:
    return [
        {"tool_calls": [{"id": "call_search", "name": "memory_search",
                         "arguments": {"query": "vllm tool call parser"}}]},
        {"tool_calls": [{"id": "call_write", "name": "write_file",
                         "arguments": {"path": str(work / "notes.md"),
                                       "content": "vLLM needs --enable-auto-tool-choice --tool-call-parser.\n"}}]},
        {"content": FINAL_1},
        # session 2
        {"content": FINAL_2},
    ]


def _check(run, label: str) -> None:
    assert run.returncode == 0, f"{label}: hermes exited {run.returncode}\n{run.output[-4000:]}"


@pytest.fixture
def agent_env(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    with FakeOpenAI(_script(work)) as server:
        home = make_home(tmp_path, base_url=server.base_url, model="fake-model", thinking=False)
        yield home, work, server


def test_agent_loop_capture_curation_and_recall(agent_env):
    home, work, server = agent_env

    # -- session 1 ---------------------------------------------------------------------------------
    run1 = run_hermes(home, f"{PREFERENCE} {QUESTION}", cwd=work, timeout=TIMEOUT)
    _check(run1, "session 1")
    assert FINAL_1 in run1.stdout
    assert run1.session_id, run1.output[-2000:]
    assert (work / "notes.md").is_file(), "write_file tool did not run"

    reqs = server.agent_requests
    assert len(reqs) == 3, [last_user_message(r)[:80] for r in reqs]
    first = reqs[0]
    # PAN provider initialized: its tools and its static system-prompt block were offered
    assert {"memory_search", "memory_read"} <= set(tool_names(first))
    assert "PAN project memory" in system_prompt(first)
    assert "Wiki pages:" in system_prompt(first)
    # prefetch: budgeted recall from the seed wiki injected into the user message
    user_msg = last_user_message(first)
    assert "<memory-context>" in user_msg and "learnings.vllm-tool-calling" in user_msg
    assert QUESTION in user_msg
    # thinking off via custom_providers.extra_body (the recipe used for Qwen3 on vLLM)
    assert first.get("chat_template_kwargs") == {"enable_thinking": False}, \
        {k: v for k, v in first.items() if k not in ("messages", "tools")}
    # memory_search was answered by PAN from the seed wiki
    search_result = json.loads(tool_results(reqs[1])["call_search"])
    assert "learnings.vllm-tool-calling" in [r["id"] for r in search_result["results"][:2]]

    # -- spool ---------------------------------------------------------------------------------------
    rows = spool_rows(home, run1.session_id)
    kinds = Counter(r["event_type"] for r in rows)
    assert kinds["turn"] == 1 and kinds["tool_call"] == 2 and kinds["file_change"] == 1, kinds
    tools = {r["content"]["tool"]: r for r in rows if r["event_type"] == "tool_call"}
    assert set(tools) == {"memory_search", "write_file"}
    assert all(r["content"]["status"] == "ok" for r in tools.values())
    (change,) = [r for r in rows if r["event_type"] == "file_change"]
    assert Path(change["content"]["path"]).name == "notes.md" and change["content"]["op"] == "write"
    (turn,) = [r for r in rows if r["event_type"] == "turn"]
    assert PREFERENCE in turn["content"]["user"] and "<memory-context>" not in turn["content"]["user"]
    assert turn["content"]["assistant"] == FINAL_1
    assert "learnings.vllm-tool-calling" in turn["content"]["recall"]  # prefetch context kept for the daemon
    assert [m["role"] for m in turn["content"]["messages"]] == ["user", "assistant", "tool", "assistant", "tool",
                                                                "assistant"]
    ends = sorted(r["content"]["kind"] for r in rows if r["event_type"] == "session_end")
    assert ends == ["close", "turn_end"], ends
    assert all(r["metadata"].get("platform") == "cli" for r in rows)

    # -- daemon ----------------------------------------------------------------------------------
    from pan.daemon.memoryd import MemoryDaemon
    from pan.paths import PanPaths

    paths = PanPaths.for_home(home)
    daemon = MemoryDaemon(paths, worker_id="e2e")
    daemon.prepare()
    try:
        reports = daemon.drain()
    finally:
        daemon.close()
    assert sum(r.claimed for r in reports) == len(rows)
    assert {r["status"] for r in spool_rows(home)} == {"done"}
    user_md = (paths.memories / "USER.md").read_text()
    assert user_md.startswith("Prefers concise answers with concrete next steps.")  # user's entry kept
    assert "British English spelling" in user_md
    log = [json.loads(line) for line in paths.curation_log.read_text().splitlines()]
    outcomes = Counter(r["outcome"] for r in log if r.get("session_id") == run1.session_id)
    assert outcomes["l1_added"] == 1, outcomes

    # -- session 2: the curated L1 entry is in the next session's frozen prompt --------------------
    run2 = run_hermes(home, "Which spelling should you use?", cwd=work, timeout=TIMEOUT)
    _check(run2, "session 2")
    assert FINAL_2 in run2.stdout
    assert run2.session_id and run2.session_id != run1.session_id
    second = server.agent_requests[3]
    assert "British English spelling" in system_prompt(second)
    assert server.steps_used == 4


def test_fake_server_and_recording_proxy_round_trip():
    """The live test's proxy passes JSON and SSE responses through unchanged and records bodies."""
    import urllib.request

    from fake_openai import RecordingProxy

    script = [{"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": {"query": "x"}}]},
              {"content": "hello there"}]
    with FakeOpenAI(script) as fake, RecordingProxy(fake.base_url) as proxy:
        def post(body):
            req = urllib.request.Request(proxy.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode()

        tools = [{"type": "function", "function": {"name": "memory_search", "parameters": {}}}]
        plain = json.loads(post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": tools}))
        call = plain["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"query": "x"}
        assert plain["choices"][0]["finish_reason"] == "tool_calls"

        sse = post({"model": "m", "stream": True, "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hi"}], "tools": tools})
        chunks = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith("data: {")]
        text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
        assert text == "hello there" and sse.rstrip().endswith("data: [DONE]")
        assert chunks[-1]["usage"]["total_tokens"] == 110

        with urllib.request.urlopen(proxy.base_url + "/models", timeout=10) as resp:
            assert json.loads(resp.read())["data"][0]["id"] == "fake-model"
        assert len(proxy.requests) == len(fake.requests) == 2 and fake.steps_used == 2
        assert len(proxy.timings) == 2
