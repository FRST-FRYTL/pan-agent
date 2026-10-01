"""Live agent E2E against a real local model (skipped unless ``PAN_LIVE_MODEL`` is set).

    PAN_LIVE_MODEL=http://localhost:8000/v1 [PAN_LIVE_MODEL_NAME=primary] pytest -m live -s

Two short headless Hermes sessions with PAN, through a recording proxy (so the test sees what
Hermes sent), with ``pan-memoryd`` run once in between:

1. the user states a preference and asks something that needs the wiki (``memory_search``);
2. a question where the curated preference (now in ``USER.md``) and the wiki fact (prefetch)
   should shape the answer.

Thinking is switched off (``chat_template_kwargs.enable_thinking: false``) to keep runs short.
Observations (timings, tool calls, answers) are printed; run with ``-s`` to see them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from agent_harness import (hermes_unavailable_reason, last_user_message, make_home, run_hermes,  # noqa: E402
                           spool_rows, system_prompt, tool_names)
from fake_openai import RecordingProxy  # noqa: E402

LIVE_URL = os.environ.get("PAN_LIVE_MODEL", "").strip()
LIVE_NAME = os.environ.get("PAN_LIVE_MODEL_NAME", "primary").strip() or "primary"
SESSION_TIMEOUT = float(os.environ.get("PAN_LIVE_TIMEOUT", "900"))

pytestmark = [
    pytest.mark.live, pytest.mark.e2e,
    pytest.mark.skipif(not LIVE_URL, reason="set PAN_LIVE_MODEL=http://localhost:8000/v1 to run live agent tests"),
]

PROMPT_1 = ("From now on, always write dates in ISO 8601 format (YYYY-MM-DD). "
            "Also: why do tool calls fail on our vLLM server? Check the knowledge wiki with memory_search "
            "first, then answer in at most three sentences.")
# "without tools": Qwen3.8 (thinking off) sometimes announces a tool call in plain text ("I'll open
# the vLLM page first") and stops; the answer must come from L1 + the prefetch context anyway.
PROMPT_2 = ("Our next maintenance window is on the 5th of March 2027. Without calling any tools, reply with "
            "one line: the date of the window, and which vLLM flags are needed so that tool calling works.")


def _observe(label: str, run, proxy: RecordingProxy, since: int, started: float) -> None:
    reqs = proxy.requests[since:]
    calls = {}  # tool_call id → name (transcripts repeat earlier calls in every later request)
    for r in proxy.agent_requests:
        for m in r.get("messages") or []:
            for c in m.get("tool_calls") or []:
                calls[c.get("id")] = (c.get("function") or {}).get("name")
    calls = list(calls.values())
    print(f"\n=== {label}: rc={run.returncode} wall={time.monotonic() - started:.1f}s "
          f"requests={len(reqs)} per-request={[round(t, 1) for t in proxy.timings[since:]]}")
    print(f"tool calls so far (from transcripts): {Counter(calls)}")
    print(f"stdout: {run.stdout.strip()[-1500:]}")
    if run.returncode != 0:
        print(f"stderr: {run.stderr[-3000:]}")


def test_live_two_sessions(tmp_path: Path):
    reason = hermes_unavailable_reason()
    if reason:
        pytest.skip(f"Hermes cannot run an agent here: {reason}")
    work = tmp_path / "work"
    work.mkdir()
    with RecordingProxy(LIVE_URL) as proxy:
        home = make_home(tmp_path, base_url=proxy.base_url, model=LIVE_NAME, thinking=False)

        # -- session 1: preference + a question that needs the wiki -------------------------------
        t0 = time.monotonic()
        run1 = run_hermes(home, PROMPT_1, cwd=work, timeout=SESSION_TIMEOUT)
        _observe("session 1", run1, proxy, 0, t0)
        assert run1.returncode == 0, run1.output[-4000:]
        assert run1.session_id

        first = proxy.agent_requests[0]
        assert {"memory_search", "memory_read"} <= set(tool_names(first))
        assert "PAN project memory" in system_prompt(first)
        assert "<memory-context>" in last_user_message(first), "prefetch did not inject wiki context"
        assert re.search(r"- \[(?:learnings|operations)\.[a-z0-9.-]+\]", last_user_message(first)), \
            "prefetch did not list a vLLM page"

        rows = spool_rows(home, run1.session_id)
        kinds = Counter(r["event_type"] for r in rows)
        pan_tools = [r["content"]["tool"] for r in rows if r["event_type"] == "tool_call"]
        print(f"session 1 events: {dict(kinds)}; tools: {pan_tools}")
        assert kinds["turn"] == 1, kinds
        assert "memory_search" in pan_tools or "memory_read" in pan_tools, (
            f"model used no PAN tool (tools: {pan_tools})")

        # -- daemon once ----------------------------------------------------------------------------
        from pan.daemon.memoryd import MemoryDaemon
        from pan.paths import PanPaths

        paths = PanPaths.for_home(home)
        daemon = MemoryDaemon(paths, worker_id="live")
        daemon.prepare()
        try:
            reports = daemon.drain()
        finally:
            daemon.close()
        log = [json.loads(line) for line in paths.curation_log.read_text().splitlines()]
        print("daemon:", [r.summary() for r in reports])
        for r in log:
            print("  curation:", r.get("outcome"), (r.get("classification") or {}).get("type"),
                  (r.get("candidate") or {}).get("title") or r.get("title") or "")
        user_md = (paths.memories / "USER.md").read_text()
        memory_md = (paths.memories / "MEMORY.md").read_text()
        print("USER.md:\n" + user_md + "\nMEMORY.md:\n" + memory_md)
        # in L1: PAN's entry in USER.md, or the agent's own memory-tool entry (USER.md or MEMORY.md)
        assert "ISO 8601" in user_md + memory_md, "preference did not reach L1"
        # PAN's classifier found the preference (added it, or saw the agent's own memory-tool entry)
        pref = [r for r in log if (r.get("classification") or {}).get("type") == "user_preference"]
        assert pref and pref[0]["outcome"] in ("l1_added", "l1_duplicate"), [r.get("outcome") for r in log]

        # -- session 2: preference (L1) + wiki fact (prefetch) should shape the answer --------------
        since = len(proxy.requests)
        n_agent = len(proxy.agent_requests)
        t1 = time.monotonic()
        run2 = run_hermes(home, PROMPT_2, cwd=work, timeout=SESSION_TIMEOUT)
        _observe("session 2", run2, proxy, since, t1)
        assert run2.returncode == 0, run2.output[-4000:]
        second = proxy.agent_requests[n_agent]
        assert "ISO 8601" in system_prompt(second), "L1 preference missing from the new session's prompt"
        assert "<memory-context>" in last_user_message(second)
        answer = run2.stdout
        assert "2027-03-05" in answer, f"preference not applied: {answer[-800:]}"
        assert "tool-call-parser" in answer or "auto-tool-choice" in answer, f"wiki fact not used: {answer[-800:]}"
