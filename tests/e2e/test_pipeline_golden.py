"""Golden e2e test (spec §8.3): fixture events → pan-memoryd → wiki / L1 / index, compared with
``tests/fixtures/expected/<scenario>.json``.

Each scenario runs alone in a fresh tmp HERMES_HOME (seed wiki, fixture memories, fixed clock).
The comparison ignores volatile values (timestamps, batch/commit ids, worker, patches); page texts
are compared in full (dates are fixed by the clock).

Regenerate after an intended behaviour change and review the diff:

    PAN_UPDATE_GOLDEN=1 python -m pytest tests/e2e/test_pipeline_golden.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import EVENT_SCENARIOS, FIXTURES, apply_agent_l1_writes, fixed_clock, spool_scenarios
from pan.daemon.memoryd import MemoryDaemon
from pan.index.fts import FtsIndex
from pan.paths import PanPaths

pytestmark = pytest.mark.e2e

EXPECTED = FIXTURES / "expected"
UPDATE = os.environ.get("PAN_UPDATE_GOLDEN") == "1"


def _changed_paths(wiki: Path) -> dict[str, list[str]]:
    """Files the daemon's commits touched, relative to the initial commit."""
    out = subprocess.run(["git", "-C", str(wiki), "diff", "--name-status", "HEAD~1", "HEAD"] if _commits(wiki) > 1
                         else ["git", "-C", str(wiki), "status", "--porcelain"], capture_output=True, text=True,
                         check=True).stdout
    created, updated = [], []
    for line in out.splitlines():
        status, path = line.split(None, 1)
        (created if status in ("A", "??") else updated).append(path)
    return {"created": sorted(created), "updated": sorted(updated)}


def _commits(wiki: Path) -> int:
    out = subprocess.run(["git", "-C", str(wiki), "rev-list", "--count", "HEAD"], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def run_scenario(home: Path, name: str) -> dict:
    paths = PanPaths.for_home(home)
    spool_scenarios(home, [name])
    apply_agent_l1_writes(home, [name])
    user_before = (paths.memories / "USER.md").read_text()
    daemon = MemoryDaemon(paths, clock=fixed_clock, worker_id="golden")
    try:
        daemon.prepare()
        daemon.drain()
    finally:
        daemon.close()
    records = [json.loads(line) for line in paths.curation_log.read_text().splitlines()]
    pages = _changed_paths(paths.wiki)
    idx = FtsIndex(paths.index_db, readonly=True)
    try:
        indexed = {p: idx.lookup(_page_id(paths.wiki / p)) == p
                   for p in pages["created"] + pages["updated"]}
    finally:
        idx.close()
    user_after = (paths.memories / "USER.md").read_text()
    return {
        "scenario": name,
        "records": [{
            "session_id": r["session_id"],
            "event_ids": r["event_ids"],
            "classification": r["classification"],
            "claims": (r["candidate"] or {}).get("claims", []),
            "claim_evidence": (r["candidate"] or {}).get("claim_evidence", []),
            "outcome": r["outcome"],
            "action": (r["decision"] or {}).get("action"),
            "target_pages": (r["decision"] or {}).get("target_pages"),
            "evidence": (r["decision"] or {}).get("evidence"),
            "confidence": (r["decision"] or {}).get("confidence"),
            "provenance": (r["decision"] or {}).get("provenance"),
            "l1": [{k: x[k] for k in ("status", "target", "entry", "replacement") if k in x}
                   for x in r["l1"] or []] or None,
        } for r in records],
        "pages": pages,
        "indexed": indexed,
        "page_texts": {p: (paths.wiki / p).read_text() for p in pages["created"] + pages["updated"]},
        "l1_added": {"user": user_after[len(user_before):].split("\n§\n")[1:] if user_after != user_before else []},
        "l1_final": {"memory": _entries(paths.memories / "MEMORY.md")},
    }


def _entries(path: Path) -> list[str]:
    text = path.read_text() if path.exists() else ""
    return [e.strip() for e in text.split("\n§\n") if e.strip()]


def _page_id(path: Path) -> str:
    from pan.wiki.frontmatter import Page
    return Page.from_text(path.read_text()).id


@pytest.mark.parametrize("scenario", EVENT_SCENARIOS)
def test_golden(scenario, pan_home):
    actual = run_scenario(pan_home, scenario)
    golden = EXPECTED / f"{scenario}.json"
    if UPDATE or not golden.exists():
        if not UPDATE:
            pytest.fail(f"missing golden file {golden}; run with PAN_UPDATE_GOLDEN=1 and review it")
        golden.write_text(json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    expected = json.loads(golden.read_text())
    assert actual == expected, f"daemon output differs from {golden.name} (PAN_UPDATE_GOLDEN=1 to regenerate)"
    assert all(actual["indexed"].values())


def test_expected_outcomes_table():
    """The README table in tests/fixtures/expected/ stays true for the golden files."""
    def load(name):
        return json.loads((EXPECTED / f"{name}.json").read_text())

    def kinds(name):
        return [(r["classification"]["type"], r["classification"]["destination"], r["action"], r["outcome"])
                for r in load(name)["records"]]

    assert kinds("preference") == [("user_preference", "user", "create", "l1_added")]
    assert load("preference")["l1_added"]["user"]
    assert kinds("env_fact") == [("environment", "wiki", "update", "committed")]
    assert load("env_fact")["pages"] == {"created": [], "updated": ["learnings/vllm-tool-calling.md"]}
    assert kinds("decision") == [("noise", "none", None, "dropped"), ("decision", "wiki", "create", "committed")]
    assert load("decision")["pages"]["created"][0].startswith("decisions/ADR-002-")
    assert kinds("duplicate") == [("environment", "wiki", "ignore", "ignored")]
    assert load("duplicate")["pages"] == {"created": [], "updated": []}
    assert kinds("noise") == [("noise", "none", None, "dropped")] * 2
    # M5
    assert kinds("fact_update") == [("environment", "wiki", "create", "committed"),
                                    ("environment", "wiki", "update", "committed"),   # l1_write → provenance
                                    ("environment", "wiki", "update", "committed"),   # the update, superseding
                                    ("noise", "none", None, "dropped")]
    fu = load("fact_update")
    assert fu["pages"] == {"created": ["operations/vllm-server.md"], "updated": ["index.md"]}
    assert "vLLM server runs on port 8010." in fu["l1_final"]["memory"]
    assert "vLLM server runs on port 8000." not in fu["l1_final"]["memory"]
    assert kinds("toolcall_text") == [("noise", "none", None, "dropped"), ("decision", "wiki", "create", "committed"),
                                      ("environment", "wiki", "create", "committed")]
    assert kinds("gpu_fact") == [("environment", "wiki", "create", "committed")]
    assert load("gpu_fact")["pages"]["created"] == ["operations/gpu.md"]
