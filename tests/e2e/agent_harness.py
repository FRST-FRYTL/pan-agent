"""Helpers for agent E2E tests: a PAN-configured HERMES_HOME and headless Hermes runs.

Recipe (headless Hermes): ``config.yaml`` with ``model.provider: custom``
+ ``model.base_url``, ``memory.provider: pan``; then ``hermes chat -q <prompt> -Q --yolo`` with
``HERMES_HOME`` set and stdin not a TTY (answers once and exits). ``PYTHONPATH`` puts this checkout's
``pan-agent/src`` first so the entry point ``pan = "pan.hermes"`` resolves to the code under test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PAN_SRC = Path(__file__).resolve().parents[2] / "src"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SESSION_ID_RE = re.compile(r"^session_id:\s*(\S+)", re.M)


def hermes_executable() -> Optional[str]:
    """The ``hermes`` console script next to the running interpreter (else on PATH)."""
    candidate = Path(sys.executable).parent / "hermes"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("hermes")


def hermes_unavailable_reason() -> Optional[str]:
    """Why Hermes cannot run an agent here (None = it can)."""
    try:
        import run_agent  # noqa: F401
        import hermes_cli.main  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        return f"Hermes not importable: {type(exc).__name__}: {exc}"
    if hermes_executable() is None:
        return "no `hermes` executable next to the Python interpreter or on PATH"
    return None


def write_config(home: Path, *, base_url: str, model: str, api_key: str = "dummy",
                 thinking: Optional[bool] = False, extra: Optional[Dict[str, Any]] = None) -> Path:
    """``config.yaml`` for a custom OpenAI-compatible endpoint with PAN as memory provider.

    ``thinking=False`` adds a ``custom_providers`` entry for the same base_url whose ``extra_body``
    sends ``chat_template_kwargs: {enable_thinking: false}`` (Qwen3 reasoning off on vLLM).
    """
    config: Dict[str, Any] = {
        "model": {"default": model, "provider": "custom", "base_url": base_url, "api_key": api_key,
                  "context_length": 131072},
        "memory": {"provider": "pan", "memory_enabled": True, "user_profile_enabled": True,
                   "nudge_interval": 0, "memory_char_limit": 2200, "user_char_limit": 1375},
    }
    if thinking is not None:
        config["custom_providers"] = [{
            "name": "pan-e2e", "base_url": base_url, "api_key": api_key,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}]
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def make_home(root: Path, *, base_url: str, model: str, seed_wiki: bool = True, **config_kw: Any) -> Path:
    """Fresh HERMES_HOME under ``root``: fixture memories, PAN config, optional seed wiki + index."""
    home = root / "hermes_home"
    shutil.copytree(FIXTURES / "hermes_home", home)
    write_config(home, base_url=base_url, model=model, **config_kw)
    if seed_wiki:
        from pan.index.fts import rebuild
        from pan.paths import PanPaths

        paths = PanPaths.for_home(home)
        paths.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(FIXTURES / "wiki", paths.wiki)
        rebuild(paths.index_db, paths.wiki)
    return home


@dataclass
class HermesRun:
    returncode: int
    stdout: str
    stderr: str
    session_id: Optional[str]

    @property
    def output(self) -> str:
        return self.stdout + "\n" + self.stderr


def run_hermes(home: Path, prompt: str, *, cwd: Path, timeout: float = 120.0,
               extra_args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> HermesRun:
    """One headless Hermes turn: ``hermes chat -q <prompt> -Q --yolo`` (stdin = /dev/null)."""
    exe = hermes_executable()
    assert exe, "hermes executable not found"
    run_env = dict(os.environ)
    run_env.update(env or {})
    run_env["HERMES_HOME"] = str(home)
    # Private HOME as well: nothing Hermes resolves via ``~`` can reach the real ~/.hermes.
    fake_home = home.parent / "user_home"
    fake_home.mkdir(exist_ok=True)
    run_env["HOME"] = str(fake_home)
    run_env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PAN_SRC), run_env.get("PYTHONPATH", "")) if p)
    run_env.setdefault("NO_COLOR", "1")
    cmd = [exe, "chat", "-q", prompt, "-Q", "--yolo", *(extra_args or [])]
    proc = subprocess.run(cmd, cwd=cwd, env=run_env, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    match = SESSION_ID_RE.search(proc.stderr) or SESSION_ID_RE.search(proc.stdout)
    return HermesRun(proc.returncode, proc.stdout, proc.stderr, match.group(1) if match else None)


def spool_rows(home: Path, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Spool rows (any status, incl. queue fields), oldest first."""
    from pan.events.spool import EventSpool

    db = home / "pan" / "events.db"
    if not db.exists():
        return []
    with EventSpool(db) as spool:
        return spool.list_rows(session_id=session_id, limit=100_000)


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def last_user_message(request: Dict[str, Any]) -> str:
    for message in reversed(request.get("messages") or []):
        if message.get("role") == "user":
            return text_of(message.get("content"))
    return ""


def system_prompt(request: Dict[str, Any]) -> str:
    return "\n".join(text_of(m.get("content")) for m in request.get("messages") or [] if m.get("role") == "system")


def tool_names(request: Dict[str, Any]) -> List[str]:
    return [(t.get("function") or {}).get("name", "") for t in request.get("tools") or []]


def tool_results(request: Dict[str, Any]) -> Dict[str, str]:
    """tool_call_id → tool result text for every tool message in the request."""
    return {m.get("tool_call_id"): text_of(m.get("content")) for m in request.get("messages") or []
            if m.get("role") == "tool"}
