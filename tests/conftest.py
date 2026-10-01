"""Shared fixtures: an isolated HERMES_HOME per test, seeded from tests/fixtures/hermes_home."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# Unit tests also run in a clean venv without Hermes (release CI); the few that need it are marked.
HERMES_AVAILABLE = importlib.util.find_spec("hermes_cli") is not None
requires_hermes = pytest.mark.skipif(not HERMES_AVAILABLE, reason="needs Hermes Agent in the venv")

# Isolate the whole test session from the real ~/.hermes BEFORE any Hermes module is imported:
# Hermes' bootstrap creates its home (logs, caches, state.db) and redirects TMPDIR into it at import
# time, which happens during collection — before per-test fixtures can set HERMES_HOME.
_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="pan-tests-"))
os.environ["HERMES_HOME"] = str(_SESSION_ROOT / "hermes_home")
os.environ["TMPDIR"] = str(_SESSION_ROOT / "tmp")
(_SESSION_ROOT / "tmp").mkdir()
tempfile.tempdir = None  # re-read TMPDIR


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh HERMES_HOME (config.yaml + memories/) with the env var pointing at it."""
    home = tmp_path / "hermes_home"
    shutil.copytree(FIXTURES / "hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def seed_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    shutil.copytree(FIXTURES / "wiki", wiki)
    return wiki


def load_events(name: str) -> list[dict]:
    path = FIXTURES / "events" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def events():
    return load_events


EVENT_SCENARIOS = sorted(p.stem for p in (FIXTURES / "events").glob("*.jsonl"))
# The M3 scenarios (daemon tests pin their outcomes); M5 added fact_update, toolcall_text, gpu_fact.
M3_SCENARIOS = ["decision", "duplicate", "env_fact", "noise", "preference"]


def apply_agent_l1_writes(hermes_home: Path, names) -> None:
    """Replay the side effect of the agent's own ``memory`` tool calls: every ``l1_write`` event of
    the scenarios is applied to MEMORY.md / USER.md through Hermes' MemoryStore (in the real system
    the tool wrote the file before PAN saw the event)."""
    from pan.hermes import compat

    with compat.hermes_home_scope(hermes_home):
        store = compat.load_l1_store()
        for name in names:
            for raw in load_events(name):
                if raw["event_type"] != "l1_write":
                    continue
                c = raw["content"]
                if c["action"] == "add":
                    store.add(c["target"], c["content"])
                elif c["action"] == "replace":
                    store.replace(c["target"], c["metadata"]["old_text"], c["content"])


@pytest.fixture
def indexed_home(hermes_home: Path) -> Path:
    """hermes_home with the seed wiki at pan/wiki and a freshly built pan/index.db (M2 read path)."""
    from pan.index.fts import rebuild
    from pan.paths import PanPaths

    paths = PanPaths.for_home(hermes_home)
    paths.root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURES / "wiki", paths.wiki)
    rebuild(paths.index_db, paths.wiki)
    return hermes_home


def load_messages(name: str) -> list[dict]:
    return json.loads((FIXTURES / "messages" / f"{name}.json").read_text())


@pytest.fixture
def capture_hub():
    """PAN's process-wide capture registry, emptied (spools closed) before and after the test."""
    from pan.hermes.capture import HUB

    HUB.clear()
    yield HUB
    HUB.clear()


# -- M3: daemon helpers ----------------------------------------------------------------------------

# Fixed daemon clock: 2026-09-24T12:00:00Z, a day after the fixture events (idle episodes close) and
# the same local date in every timezone (page dates in golden files are deterministic).
FIXED_NOW = 1790251200.0


def fixed_clock() -> float:
    return FIXED_NOW


@pytest.fixture
def pan_home(hermes_home: Path) -> Path:
    """hermes_home with the seed wiki (not yet a git repo, not indexed) at pan/wiki."""
    from pan.paths import PanPaths

    paths = PanPaths.for_home(hermes_home)
    paths.root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURES / "wiki", paths.wiki)
    return hermes_home


def spool_scenarios(hermes_home: Path, names, *, fresh_ids: bool = False) -> int:
    """Append the fixture events of ``names`` to the profile's spool; ``fresh_ids`` re-mints the ids
    (the same content arriving again, e.g. a repeated conversation)."""
    from pan.events.schema import AgentEvent, new_ulid
    from pan.events.spool import EventSpool
    from pan.paths import PanPaths

    n = 0
    with EventSpool(PanPaths.for_home(hermes_home).events_db) as spool:
        for name in names:
            for raw in load_events(name):
                if fresh_ids:
                    raw = dict(raw, id=new_ulid())
                spool.append(AgentEvent.from_dict(raw))
                n += 1
    return n


def scenario_episodes(name: str):
    """Fixture events of one scenario grouped into episodes (everything closed)."""
    from pan.events.schema import AgentEvent
    from pan.memory.episodes import group_events

    return group_events([AgentEvent.from_dict(e) for e in load_events(name)], now=FIXED_NOW).episodes


@pytest.fixture(autouse=True)
def pin_rules_classifier(monkeypatch: pytest.MonkeyPatch) -> str:
    """The shipped default gate is the LLM (M9); tests pin rules-v2 so no test ever reaches the live
    model endpoint. Tests that want the gate configure it explicitly (fake server). Returns the
    shipped default."""
    import pan.config
    shipped = pan.config.DEFAULT_CLASSIFIER_KIND
    monkeypatch.setattr(pan.config, "DEFAULT_CLASSIFIER_KIND", "rules")
    return shipped


@pytest.fixture(autouse=True)
def pin_fts_retrieval(monkeypatch: pytest.MonkeyPatch) -> str:
    """The shipped default retrieval is hybrid (M7, ADR-012: FTS + a local embedding sidecar); tests
    pin FTS so no test depends on a sidecar that happens to run on the machine. Tests of the dense
    path configure a fake sidecar explicitly. Returns the shipped default."""
    import pan.config
    shipped = pan.config.DEFAULT_RETRIEVAL_MODE
    monkeypatch.setattr(pan.config, "DEFAULT_RETRIEVAL_MODE", "fts")
    return shipped
