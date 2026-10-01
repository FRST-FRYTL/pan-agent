"""pan-memoryd M5: L1 reconciliation across sessions, agent L1 writes, index links, recall afterwards.

Real Hermes MemoryStore (tmp HERMES_HOME), seed wiki, fixed clock. The golden files cover the exact
output of the same fixtures; these tests pin the behaviour that matters across batches.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import apply_agent_l1_writes, fixed_clock, load_events, spool_scenarios
from pan.daemon.memoryd import MemoryDaemon
from pan.events.schema import Actor, AgentEvent, EventType
from pan.events.spool import EventSpool
from pan.memory.reader import Reader
from pan.memory.l1 import L1Writer
from pan.paths import PanPaths

pytestmark = pytest.mark.e2e


def _daemon(home: Path) -> MemoryDaemon:
    d = MemoryDaemon(PanPaths.for_home(home), clock=fixed_clock, worker_id="test")
    d.prepare()
    return d


def _log(home: Path) -> list[dict]:
    return [json.loads(x) for x in PanPaths.for_home(home).curation_log.read_text().splitlines()]


def _spool(home: Path, events) -> None:
    with EventSpool(PanPaths.for_home(home).events_db) as spool:
        for raw in events:
            spool.append(AgentEvent.from_dict(raw))


def test_fact_update_across_batches_keeps_wiki_and_l1_in_sync(pan_home):
    """A multi-session scenario as separate daemon runs (the bench runs `pan memoryd run --once` after each session)."""
    paths = PanPaths.for_home(pan_home)
    events = load_events("fact_update")
    by_session: dict[str, list] = {}
    for raw in events:
        by_session.setdefault(raw["session_id"], []).append(raw)
    apply_agent_l1_writes(pan_home, ["fact_update"])  # the agent's memory tool wrote port 8000 in session 1
    for sid in ("sess-fu1", "sess-fu2", "sess-fu3"):
        _spool(pan_home, by_session[sid])
        d = _daemon(pan_home)
        try:
            d.drain()
        finally:
            d.close()
    memory = L1Writer(pan_home).entries("memory")
    assert "vLLM server runs on port 8010." in memory and "vLLM server runs on port 8000." not in memory
    page = (paths.wiki / "operations/vllm-server.md").read_text()
    assert "~~Our vLLM server runs on port 8000.~~" in page and "- We moved the vLLM server to port 8010." in page
    assert "l1:memory" in page  # the agent's L1 write is linked in the page's provenance
    assert "port 8000." not in page.split("## Updates")[1].replace("~~", "")  # the stale answer was not written back
    index = (paths.wiki / "index.md").read_text()
    assert "[vLLM server](operations/vllm-server.md)" in index
    superseded = [x for r in _log(pan_home) for x in (r.get("l1") or []) if x["status"] == "superseded"]
    assert len(superseded) == 1 and superseded[0]["replacement"] == "vLLM server runs on port 8010."
    wiki_log = subprocess.run(["git", "-C", str(paths.wiki), "log", "--name-only", "--format=%s"],
                              capture_output=True, text=True, check=True).stdout
    first_commit = wiki_log.split("pan-memoryd:")[-1]
    assert "operations/vllm-server.md" in first_commit and "index.md" in first_commit  # same commit

    # recall in the next session: prefetch for the question finds the page with the new port
    reader = Reader(paths)
    try:
        out = reader.prefetch("Which port does our vLLM server use? Answer from memory in one sentence.")
    finally:
        reader.close()
    assert "operations.vllm-server" in out


def test_an_old_agent_write_never_reverts_a_newer_supersession(pan_home):
    """PAN superseded port 8000 → 8010 from session 2; the agent's session-1 write arrives late."""
    events = load_events("fact_update")
    l1_write = next(e for e in events if e["event_type"] == "l1_write")
    apply_agent_l1_writes(pan_home, ["fact_update"])
    _spool(pan_home, [e for e in events if e["session_id"] == "sess-fu2"])
    d = _daemon(pan_home)
    try:
        d.drain()
        assert "vLLM server runs on port 8010." in L1Writer(pan_home).entries("memory")
        _spool(pan_home, [l1_write])  # older id, processed later
        d.drain()
    finally:
        d.close()
    memory = L1Writer(pan_home).entries("memory")
    assert "vLLM server runs on port 8010." in memory and "vLLM server runs on port 8000." not in memory
    rec = _log(pan_home)[-1]
    assert rec["kind"] == "l1_write" and any(x["status"] == "kept_newer" for x in rec["l1"])


def test_gpu_fact_is_recalled_for_a_plain_question(pan_home):
    """Seed scenario: the fact must come back for "which GPU model does this machine have?"."""
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["gpu_fact"])
    d = _daemon(pan_home)
    try:
        d.drain()
    finally:
        d.close()
    assert not list((paths.wiki / "learnings").glob("*error*"))  # the empty-args retry is no learning
    reader = Reader(paths)
    try:
        out = reader.prefetch("Without running any commands: which GPU model does this machine have?")
        hits = reader.search("which GPU model does this machine have?")["results"]
    finally:
        reader.close()
    assert "GB10" in out
    assert hits[0]["id"] in ("operations.gpu", "systems.dgx-spark-host") and hits[0]["score"] >= 0.2


def test_printed_tool_calls_create_no_pages(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["toolcall_text"])
    d = _daemon(pan_home)
    try:
        d.drain()
    finally:
        d.close()
    created = sorted(p.relative_to(paths.wiki).as_posix() for p in paths.wiki.rglob("*.md")
                     if "2026-09-24" in p.read_text().split("---")[1] and p.name != "index.md")
    assert created == ["decisions/ADR-002-sqlite-not-postgres-bench-results-index.md",
                       "operations/grafana-container.md"]
    text = "\n".join(p.read_text() for p in paths.wiki.rglob("*.md"))
    for junk in ("tool_call", "<function=", "old_text=", "Memory (user)", '"arguments"', "memory_search("):
        assert junk not in text


def test_agent_edit_to_a_wiki_page_is_adopted_not_deferred_forever(pan_home):
    """Live layer-B run: the agent patched pan/wiki/operations/vllm-server.md itself (8000 → 8010),
    and its patch error + retry became a junk learning page; the user's update was lost."""
    paths = PanPaths.for_home(pan_home)
    by_session: dict[str, list] = {}
    for raw in load_events("fact_update"):
        by_session.setdefault(raw["session_id"], []).append(raw)
    _spool(pan_home, by_session["sess-fu1"])
    d = _daemon(pan_home)
    try:
        d.drain()
        page = paths.wiki / "operations/vllm-server.md"
        page.write_text(page.read_text().replace("port 8000.", "port 8010."))  # the agent's `patch`
        turn = by_session["sess-fu2"][0]
        turn = dict(turn, content=dict(turn["content"], assistant="Done — the wiki page now records port 8010."))
        edit_at = "2026-09-23T12:04:58.000Z"
        _spool(pan_home, [
            {"id": "01K0FUP0000000000000000010", "ts": edit_at, "session_id": "sess-fu2", "parent_session_id": None,
             "actor": "tool", "event_type": "tool_call", "project": None, "metadata": {}, "source_refs": [],
             "content": {"tool": "patch", "args": {"path": "operations/vllm-server.md"}, "status": "error",
                         "error_message": "Failed to read file: /work/operations/vllm-server.md"}},
            {"id": "01K0FUP0000000000000000011", "ts": edit_at, "session_id": "sess-fu2", "parent_session_id": None,
             "actor": "tool", "event_type": "tool_call", "project": None, "metadata": {}, "source_refs": [],
             "content": {"tool": "patch", "args": {"path": str(page)}, "status": "ok", "result_excerpt": "ok"}},
            {"id": "01K0FUP0000000000000000012", "ts": edit_at, "session_id": "sess-fu2", "parent_session_id": None,
             "actor": "tool", "event_type": "file_change", "project": None, "metadata": {}, "source_refs": [],
             "content": {"path": str(page), "op": "replace", "tool": "patch"}},
            dict(turn, id="01K0FUP0000000000000000013")])
        reports = d.drain()
    finally:
        d.close()
    assert sum(r.deferred for r in reports) == 0
    log = _log(pan_home)
    adopted = [r for r in log if r.get("kind") == "agent_wiki_edit"]
    assert adopted and adopted[0]["outcome"] == "adopted" and adopted[0]["path"] == "operations/vllm-server.md"
    authors = subprocess.run(["git", "-C", str(paths.wiki), "log", "--format=%an"], capture_output=True, text=True,
                             check=True).stdout.split()
    assert "hermes-agent" in authors
    assert not list((paths.wiki / "learnings").glob("patch-error*"))       # no junk learning
    user_fact = [r for r in log if r.get("session_id") == "sess-fu2" and r.get("candidate")]
    assert user_fact[-1]["candidate"]["claims"] == ["We moved the vLLM server to port 8010."]
    assert subprocess.run(["git", "-C", str(paths.wiki), "status", "--porcelain"], capture_output=True, text=True,
                          check=True).stdout == ""


def test_users_words_are_curated_before_the_agents_l1_copy_of_them(pan_home):
    """Live run: when the agent's memory write is the turn's first event, the page must still be
    built from the user's (observed) sentence; the agent's copy only links provenance."""
    paths = PanPaths.for_home(pan_home)
    events = [e for e in load_events("fact_update") if e["session_id"] == "sess-fu1"
              and e["id"] != "01K0FUP0000000000000000001"]  # no failed memory call before the write
    assert events[0]["event_type"] == "l1_write"
    _spool(pan_home, events)
    d = _daemon(pan_home)
    try:
        d.drain()
    finally:
        d.close()
    page = (paths.wiki / "operations/vllm-server.md").read_text()
    assert "- Our vLLM server runs on port 8000. (observed)" in page
    assert "Saved by the agent" not in page and "l1:memory" in page


def test_preference_writes_by_the_agent_are_only_logged(pan_home):
    with EventSpool(PanPaths.for_home(pan_home).events_db) as spool:
        spool.append(AgentEvent(event_type=EventType.L1_WRITE, session_id="s", actor=Actor.MAIN_AGENT,
                                content={"action": "add", "target": "memory",
                                         "content": "User prefers answers under 60 words."}))
    d = _daemon(pan_home)
    try:
        assert d.run_once().done == 1
    finally:
        d.close()
    assert _log(pan_home)[-1]["kind"] == "l1_write_observed"


def test_agent_notes_about_memory_tools_are_not_mirrored_to_the_wiki(pan_home):
    """M6, a benchmark run: after hunting for a save tool the agent wrote a note about
    the PAN wiki / tool names into MEMORY.md; that is meta-talk, not a project fact."""
    paths = PanPaths.for_home(pan_home)
    pages_before = sorted(p.name for p in paths.wiki.rglob("*.md"))
    with EventSpool(paths.events_db) as spool:
        spool.append(AgentEvent(event_type=EventType.L1_WRITE, session_id="s", actor=Actor.MAIN_AGENT,
                                content={"action": "add", "target": "memory",
                                         "content": "PAN knowledge wiki is maintained by the pan-memoryd daemon "
                                                    "on port 8000; tool name for saving facts unknown."}))
    d = _daemon(pan_home)
    try:
        assert d.run_once().done == 1
    finally:
        d.close()
    assert _log(pan_home)[-1]["kind"] == "l1_write_observed"
    assert sorted(p.name for p in paths.wiki.rglob("*.md")) == pages_before
