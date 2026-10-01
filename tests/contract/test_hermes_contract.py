"""Contract tests: every assumption PAN makes about the pinned Hermes (integration spec §8.2).

Run on every commit and on every upstream sync. A failure here means: fix pan/hermes/ before merging
the sync — never patch Hermes.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

import pan.hermes  # noqa: E402
from pan import HERMES_PIN  # noqa: E402
from pan.hermes import compat  # noqa: E402
from pan.hermes.provider import TOOL_NAMES, PanMemoryProvider  # noqa: E402


def test_running_hermes_matches_pin():
    assert compat.hermes_version() == HERMES_PIN


# 1. discovery ------------------------------------------------------------------------------------

def test_pan_is_discovered_via_entry_point(hermes_home):
    from plugins.memory import find_provider_dir, find_provider_entry_point, load_memory_provider

    assert find_provider_entry_point("pan") is not None, "pan-agent not installed (uv pip install -e ./pan-agent)"
    resolved = find_provider_dir("pan")
    ours = Path(pan.hermes.__file__).resolve().parent
    assert resolved is None or Path(resolved).resolve() == ours, (
        f"provider 'pan' resolves to {resolved}, not our package — a bundled/user provider shadows it")
    provider = load_memory_provider("pan", register_skills=False)
    assert provider is not None and provider.name == "pan"


# 2. provider interface ---------------------------------------------------------------------------

PAN_HOOKS = ["initialize", "system_prompt_block", "prefetch", "get_tool_schemas", "handle_tool_call",
             "sync_turn", "on_memory_write", "on_pre_compress", "on_delegation", "on_session_switch",
             "on_session_end", "shutdown"]


@pytest.mark.parametrize("hook", PAN_HOOKS)
def test_memory_provider_still_has_hook(hook):
    from agent.memory_provider import MemoryProvider

    assert callable(getattr(MemoryProvider, hook, None)), f"MemoryProvider.{hook} is gone"


def test_sync_turn_receives_messages():
    from agent.memory_provider import MemoryProvider

    assert "messages" in inspect.signature(MemoryProvider.sync_turn).parameters


def test_provider_instantiates_and_initializes(hermes_home):
    provider = PanMemoryProvider()
    provider.initialize("sess-1", hermes_home=str(hermes_home), platform="cli", agent_context="primary")
    assert (hermes_home / "pan").is_dir()
    assert provider.system_prompt_block()


# 3. general plugin hooks -------------------------------------------------------------------------

@pytest.mark.parametrize("hook", ["post_tool_call", "subagent_start", "subagent_stop", "on_session_end",
                                  "on_session_finalize", "on_skill_lifecycle"])
def test_plugin_hook_exists(hook):
    from hermes_cli.plugins import VALID_HOOKS

    assert hook in VALID_HOOKS


# 4. tool names -----------------------------------------------------------------------------------

def test_tools_do_not_collide_with_core_tools():
    assert not TOOL_NAMES & compat.core_tool_names()


def test_memory_manager_routes_pan_tools():
    from agent.memory_manager import MemoryManager

    manager = MemoryManager()
    manager.add_provider(PanMemoryProvider())
    assert TOOL_NAMES <= manager.get_all_tool_names()


# 5. L1 writes through Hermes' MemoryStore --------------------------------------------------------

def test_l1_write_round_trips_without_drift(hermes_home):
    writer = compat.load_l1_store()
    result = writer.add("user", "Writes specs as Markdown files in the repo.")
    assert result.get("success"), result

    agent_store = compat.load_l1_store()  # a second store, like a running agent's
    result = agent_store.add("memory", "Memory curator runs as pan-memoryd.")
    assert result.get("success"), f"agent store saw drift after PAN's write: {result}"
    assert not list((hermes_home / "memories").glob("*.bak.*"))
    assert "Writes specs as Markdown" in (hermes_home / "memories" / "USER.md").read_text()


def test_l1_char_limit_enforced(hermes_home):
    store = compat.load_l1_store()
    result = store.add("user", "x" * 5000)
    assert not result.get("success")


# 6. session store --------------------------------------------------------------------------------

def test_session_db_opens_read_only_with_expected_schema(hermes_home):
    from hermes_state import SessionDB

    db_path = hermes_home / "state.db"
    SessionDB(db_path).close()  # create schema like a running Hermes would
    ro = compat.open_session_db_readonly(db_path)
    ro.close()
    with sqlite3.connect(db_path) as conn:
        messages = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        sessions = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert {"session_id", "role", "content", "tool_calls", "tool_call_id", "tool_name"} <= messages
    assert "parent_session_id" in sessions


# 7. config keys PAN sets -------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["provider", "memory_enabled", "user_profile_enabled", "nudge_interval"])
def test_memory_config_key_exists(key):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert key in DEFAULT_CONFIG["memory"]
