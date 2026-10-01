"""The ONLY module that touches Hermes internals that are not documented plugin surfaces (spec §4.10).

Every assumption made here is pinned by tests/contract/. If an upstream sync breaks one of these
functions, fix it here — nowhere else in PAN imports Hermes internals.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pan import HERMES_PIN

logger = logging.getLogger(__name__)


def hermes_version() -> str:
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:  # pragma: no cover - Hermes not importable
        return "unknown"


def check_hermes_pin() -> bool:
    """Warn (never fail) when the running Hermes differs from the version PAN was tested against."""
    running = hermes_version()
    if running != HERMES_PIN:
        logger.warning("PAN is pinned to hermes-agent %s but %s is running; "
                       "run PAN's contract tests before relying on it.", HERMES_PIN, running)
        return False
    return True


def core_tool_names() -> frozenset[str]:
    """Tool names reserved by Hermes core (a provider tool with one of these names is dropped)."""
    from toolsets import _HERMES_CORE_TOOLS
    return frozenset(_HERMES_CORE_TOOLS)


def load_l1_store() -> Any:
    """Hermes' on-disk MemoryStore (MEMORY.md / USER.md) with configured limits.

    Writing through this store uses Hermes' file lock and entry format, so a running agent does
    not detect "external drift" (spec §4.9). Must be called with HERMES_HOME set for the profile.
    """
    from tools.memory_tool import load_on_disk_store
    return load_on_disk_store()


@contextmanager
def hermes_home_scope(hermes_home: str | Path) -> Iterator[None]:
    """Point Hermes at ``hermes_home`` for the duration of the block (``HERMES_HOME`` env var, plus
    Hermes' context-local override when available). pan-memoryd is single-threaded, so the env
    change is safe there; the previous values are always restored."""
    home = str(Path(hermes_home).expanduser())
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = home
    token = None
    try:
        from hermes_constants import set_hermes_home_override
        token = set_hermes_home_override(home)
    except Exception:  # older/newer Hermes without the override: the env var is enough
        token = None
    try:
        yield
    finally:
        if token is not None:
            try:
                from hermes_constants import reset_hermes_home_override
                reset_hermes_home_override(token)
            except Exception:
                pass
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


def l1_entry_delimiter() -> str:
    from tools.memory_tool import ENTRY_DELIMITER
    return ENTRY_DELIMITER


def l1_entries(store: Any, target: str) -> list[str]:
    """Current entries of ``target`` ("user" | "memory") as loaded by the store."""
    return list(store.user_entries if target == "user" else store.memory_entries)


def l1_char_limit(store: Any, target: str) -> int:
    return int(store.user_char_limit if target == "user" else store.memory_char_limit)


def l1_target_enabled(store: Any, target: str) -> bool:
    check = getattr(store, "target_enabled", None)
    return bool(check(target)) if callable(check) else True


def open_session_db_readonly(db_path: Path | None = None) -> Any:
    """Hermes' session store (state.db), read-only. None db_path = active profile's default."""
    from hermes_state import SessionDB
    return SessionDB(db_path, read_only=True) if db_path else SessionDB(read_only=True)


def read_session_messages(db_path: Path, session_id: str) -> list[dict]:
    """Active messages of one session from ``state.db`` (read-only), in insertion order. Each dict
    has at least ``id``, ``role``, ``content``, ``timestamp`` (epoch seconds), ``tool_call_id``."""
    db = open_session_db_readonly(db_path)
    try:
        return list(db.get_messages(session_id))
    finally:
        db.close()
