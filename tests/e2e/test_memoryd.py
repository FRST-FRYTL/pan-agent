"""pan-memoryd over the fixture scenarios in a tmp HERMES_HOME (integration spec §4.5, §9)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import M3_SCENARIOS as EVENT_SCENARIOS, fixed_clock, spool_scenarios  # noqa: N811
from pan.daemon.memoryd import (COMMITTED, DEFERRED, DaemonLock, MemoryDaemon, lock_path, main, replay,
                                running_pid)
from pan.events.spool import EventSpool
from pan.index.fts import FtsIndex
from pan.memory.l1 import L1Result
from pan.paths import PanPaths
from pan.wiki.git import GitError

pytestmark = pytest.mark.e2e

ADR = "decisions/ADR-002-run-curation-separate-background-service.md"


def _daemon(home: Path, **kw) -> MemoryDaemon:
    d = MemoryDaemon(PanPaths.for_home(home), clock=fixed_clock, worker_id="test", **kw)
    d.prepare()
    return d


def _log(home: Path) -> list[dict]:
    path = PanPaths.for_home(home).curation_log
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _git(wiki: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(wiki), *args], capture_output=True, text=True, check=True).stdout


def _wiki_files(wiki: Path) -> dict[str, str]:
    return {p.relative_to(wiki).as_posix(): p.read_text() for p in sorted(wiki.rglob("*.md"))
            if ".git" not in p.parts}


class FakeL1:
    def __init__(self, status: str = "added") -> None:
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def add(self, target: str, entry: str) -> L1Result:
        self.calls.append((target, entry))
        return L1Result(self.status, target, entry, "fake")


def test_once_over_all_fixtures(pan_home):
    paths = PanPaths.for_home(pan_home)
    n = spool_scenarios(pan_home, EVENT_SCENARIOS)
    d = _daemon(pan_home)
    try:
        reports = d.drain()
    finally:
        d.close()
    assert sum(r.claimed for r in reports) == n
    with EventSpool(paths.events_db) as spool:
        assert spool.stats() == {"new": 0, "processing": 0, "done": n, "dead": 0}

    # wiki: one ADR created, the seed learning updated, all in ONE pan-memoryd commit
    assert (paths.wiki / ADR).is_file()
    learning = (paths.wiki / "learnings/vllm-tool-calling.md").read_text()
    assert "--max-model-len 32768" in learning and "## Updates" in learning
    log = _git(paths.wiki, "log", "--format=%an <%ae>|%s").splitlines()
    assert log[0].startswith("pan-memoryd <pan-memoryd@localhost>|pan-memoryd: 2 wiki change(s)")
    assert log[-1].endswith("|wiki: initialize") and len(log) == 2
    assert _git(paths.wiki, "status", "--porcelain") == ""

    # index updated: the new ADR is searchable
    idx = FtsIndex(paths.index_db, readonly=True)
    try:
        assert idx.lookup("decisions.adr-002") == ADR
        assert idx.search("curation background service pan-memoryd")[0].id == "decisions.adr-002"
    finally:
        idx.close()

    # L1: preference appended to USER.md, user's own entry untouched
    user_md = (paths.memories / "USER.md").read_text()
    assert user_md.startswith("Prefers concise answers with concrete next steps.")
    assert "Write meeting notes as Markdown files in the repo" in user_md

    # curation log: one record per candidate, every one labelled
    records = _log(pan_home)
    outcomes = sorted((r["session_id"], r["outcome"]) for r in records)
    assert outcomes == [("sess-dec", "committed"), ("sess-dec", "dropped"), ("sess-dup", "ignored"),
                        ("sess-env", "committed"), ("sess-noise", "dropped"), ("sess-noise", "dropped"),
                        ("sess-pref", "l1_added")]
    committed = [r for r in records if r["outcome"] == COMMITTED]
    assert all(r["commit"] == _git(paths.wiki, "rev-parse", "HEAD").strip() for r in committed)
    assert all(r["classification"]["type"] and r["classifier"] == "rules-v2" for r in records)


def test_running_twice_yields_no_duplicates(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, EVENT_SCENARIOS)
    d = _daemon(pan_home)
    try:
        d.drain()
        wiki_after_first = _wiki_files(paths.wiki)
        user_after_first = (paths.memories / "USER.md").read_text()
        head = _git(paths.wiki, "rev-parse", "HEAD")
        first = len(_log(pan_home))
        assert d.drain()[0].claimed == 0  # same spool again: nothing left to do
        spool_scenarios(pan_home, EVENT_SCENARIOS, fresh_ids=True)  # the same conversations again
        d.drain()
    finally:
        d.close()
    assert _wiki_files(paths.wiki) == wiki_after_first
    assert (paths.memories / "USER.md").read_text() == user_after_first
    assert _git(paths.wiki, "rev-parse", "HEAD") == head
    second = _log(pan_home)[first:]
    assert sorted(r["outcome"] for r in second) == ["dropped"] * 3 + ["ignored"] * 3 + ["l1_duplicate"]


def test_second_instance_is_refused(pan_home):
    paths = PanPaths.for_home(pan_home)
    lock = DaemonLock(lock_path(paths))
    assert lock.acquire()
    try:
        assert running_pid(lock_path(paths)) > 0
        assert main(["--once", "--hermes-home", str(pan_home)]) == 1
    finally:
        lock.release()
    assert running_pid(lock_path(paths)) is None


def test_cli_once_processes_fixtures(pan_home, capsys):
    spool_scenarios(pan_home, ["preference", "duplicate"])
    assert main(["run", "--once", "--hermes-home", str(pan_home)]) == 0
    out = capsys.readouterr().out
    assert "l1_added=1" in out and "ignored=1" in out
    assert (PanPaths.for_home(pan_home).root / "memoryd.log").exists()


def test_manual_edit_defers_the_decision(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["env_fact"])
    d = _daemon(pan_home, l1=FakeL1())
    target = paths.wiki / "learnings/vllm-tool-calling.md"
    target.write_text(target.read_text() + "\nManual note.\n")  # uncommitted manual edit
    try:
        report = d.run_once()
        assert report.deferred == 2 and report.done == 0
        assert _log(pan_home)[-1]["outcome"] == DEFERRED
        assert d.run_once().deferred == 2
        assert len(_log(pan_home)) == 1  # deferral logged once
        _git(paths.wiki, "commit", "-qam", "manual edit")
        report = d.run_once()
        assert report.done == 2 and report.commit
    finally:
        d.close()
    text = target.read_text()
    assert "Manual note." in text and "--max-model-len 32768" in text
    with EventSpool(paths.events_db) as spool:
        assert all(spool.get(r["event_ids"][0])["attempts"] == 0 for r in _log(pan_home)[-1:])


def test_commit_failure_reverts_and_retries(pan_home, monkeypatch):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["decision"])
    d = _daemon(pan_home, l1=FakeL1())
    try:
        def boom(*a, **k):
            raise GitError("hook rejected")
        monkeypatch.setattr(d.git, "commit", boom)
        report = d.run_once()
        assert report.failed == 1 and not (paths.wiki / ADR).exists()
        assert d.index.lookup("decisions.adr-002") is None
        monkeypatch.undo()
        report = d.run_once()
        assert report.commit and (paths.wiki / ADR).exists()
    finally:
        d.close()
    outcomes = [r["outcome"] for r in _log(pan_home)]
    assert outcomes == ["dropped", "failed", "committed"]


def test_validation_failure_reverts_the_write(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["decision"])
    d = _daemon(pan_home, l1=FakeL1())
    real = d.curator.curate

    def broken(cand, **kw):
        cur = real(cand, **kw)
        cur.page.body += "\n[broken](nowhere.md)\n"
        return cur
    d.curator.curate = broken
    try:
        report = d.run_once()
    finally:
        d.close()
    assert report.failed == 1 and not (paths.wiki / ADR).exists()
    rec = _log(pan_home)[-1]
    assert rec["outcome"] == "failed" and "broken link" in rec["error"]


def test_repeated_failures_end_dead_with_a_copy(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["preference"])
    d = _daemon(pan_home, l1=FakeL1(status="error"))
    try:
        reports = [d.run_once() for _ in range(3)]
    finally:
        d.close()
    assert [r.failed for r in reports] == [1, 1, 1] and reports[-1].dead == 1
    assert (paths.dead / "01K0PREF000000000000000001.json").is_file()
    with EventSpool(paths.events_db) as spool:
        assert spool.stats()["dead"] == 1


def test_l1_full_falls_back_to_wiki(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["preference"])
    d = _daemon(pan_home, l1=FakeL1(status="l1_full"))
    try:
        d.run_once()
    finally:
        d.close()
    rec = _log(pan_home)[-1]
    assert rec["outcome"] == COMMITTED and rec["l1"][0]["status"] == "l1_full"
    assert rec["decision"]["rationale"].startswith("l1_full → wiki")
    assert "Write meeting notes as Markdown" in (paths.wiki / "preferences/user-preferences.md").read_text()


def test_open_episode_is_released_until_idle(pan_home):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["noise"])
    clock = [1790143200.0]  # 2026-09-23T06:00Z … events are later that day → trailing tool group is "fresh"
    d = MemoryDaemon(paths, clock=lambda: clock[0], worker_id="t", l1=FakeL1())
    d.prepare()
    try:
        report = d.run_once()
        assert report.pending == 1 and report.done == 1
        with EventSpool(paths.events_db) as spool:
            assert spool.stats()["new"] == 1
        clock[0] = fixed_clock()
        assert d.run_once().done == 1
    finally:
        d.close()


def test_l1_write_events_are_logged(pan_home):
    from pan.events.schema import Actor, AgentEvent, EventType

    paths = PanPaths.for_home(pan_home)
    with EventSpool(paths.events_db) as spool:
        spool.append(AgentEvent(event_type=EventType.L1_WRITE, session_id="s", actor=Actor.MAIN_AGENT,
                                content={"action": "add", "target": "user", "content": "Likes tea."}))
    d = _daemon(pan_home, l1=FakeL1())
    try:
        assert d.run_once().done == 1
    finally:
        d.close()
    rec = _log(pan_home)[-1]
    assert rec["kind"] == "l1_write_observed" and rec["l1"][0]["entry"] == "Likes tea."


def test_replay_uses_a_scratch_profile(pan_home, tmp_path):
    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, EVENT_SCENARIOS)
    user_before = (paths.memories / "USER.md").read_text()
    records = replay(paths, tmp_path / "replay", clock=fixed_clock)
    assert sorted(r["outcome"] for r in records) == sorted(
        ["committed", "dropped", "ignored", "committed", "dropped", "dropped", "l1_added"])
    assert (paths.memories / "USER.md").read_text() == user_before  # real profile untouched
    assert not paths.curation_log.exists()
    assert (tmp_path / "replay/hermes_home/pan/wiki" / ADR).exists()
    assert not (paths.wiki / ADR).exists()


def test_foreground_daemon_stops_on_sigterm(pan_home):
    import os
    import signal
    import sys
    import time

    paths = PanPaths.for_home(pan_home)
    spool_scenarios(pan_home, ["preference"])
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = dict(os.environ, HERMES_HOME=str(pan_home),
               PYTHONPATH=os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.Popen([sys.executable, "-m", "pan.daemon.memoryd", "run", "--hermes-home", str(pan_home)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not paths.curation_log.exists():
            assert proc.poll() is None, proc.communicate()
            time.sleep(0.1)
        assert paths.curation_log.exists(), "daemon did not process the queued event"
        assert running_pid(lock_path(paths)) == proc.pid
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert running_pid(lock_path(paths)) is None
    assert _log(pan_home)[0]["outcome"] == "l1_added"
