"""`pan setup` safety net (ADR-011 §2–3): the tested-Hermes check, the settings backup and `pan uninstall`.

Hermes settings are only ever changed through the documented `hermes config set|unset` CLI. Before the
first change, `pan setup` records the previous value of every key it touches (and a copy of
``config.yaml``) in ``$HERMES_HOME/pan/setup-backup.json``. The backup is written once and never
overwritten, so re-running setup keeps the pre-PAN state. `pan uninstall` restores it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from pan import HERMES_PIN, __version__
from pan.paths import PanPaths

BACKUP_NAME = "setup-backup.json"
_ABSENT = None  # recorded when a key was not set in config.yaml (Hermes used its default)


def backup_path(paths: PanPaths) -> Path:
    return paths.root / BACKUP_NAME


def tested_hermes_problem(running: str) -> str | None:
    """None if ``running`` is the Hermes version this release was tested against, else the reason."""
    if running == HERMES_PIN:
        return None
    return (f"Hermes {running} is not the version pan-agent {__version__} was tested against "
            f"(hermes-agent {HERMES_PIN}). Install the matching pan-agent release or Hermes {HERMES_PIN}, "
            "or re-run with --force to continue at your own risk.")


def _lookup(config: Mapping[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def read_current_settings(hermes_home: Path, keys) -> dict[str, Any]:
    """Current values of ``keys`` in ``config.yaml`` (None = not set there)."""
    import yaml

    cfg_file = hermes_home / "config.yaml"
    config = {}
    if cfg_file.exists():
        config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    return {key: _lookup(config, key) for key in keys}


def write_backup(paths: PanPaths, keys) -> tuple[Path, bool]:
    """Record the pre-PAN values of ``keys`` once. Returns (backup file, written now)."""
    target = backup_path(paths)
    if target.exists():
        return target, False
    paths.root.mkdir(parents=True, exist_ok=True)
    cfg_file = paths.hermes_home / "config.yaml"
    config_copy = None
    if cfg_file.exists():
        config_copy = target.with_name("config.yaml.pre-pan")
        shutil.copy2(cfg_file, config_copy)
    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pan_version": __version__,
        "settings": read_current_settings(paths.hermes_home, keys),
        "config_copy": config_copy.name if config_copy else None,
    }
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return target, True


def _restore_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def restore_commands(paths: PanPaths, hermes: str) -> list[list[str]]:
    """`hermes config` commands that restore the recorded settings (set, or unset if it was absent)."""
    record = json.loads(backup_path(paths).read_text(encoding="utf-8"))
    cmds = []
    for key, value in record["settings"].items():
        if value is _ABSENT:
            cmds.append([hermes, "config", "unset", key])
        else:
            cmds.append([hermes, "config", "set", key, _restore_value(value)])
    return cmds


def uninstall(paths: PanPaths, hermes: str, *, dry_run: bool,
              run: Callable[..., Any] = subprocess.run, echo: Callable[[str], None] = print) -> int:
    """Restore the Hermes settings from the setup backup. PAN's data under ``$HERMES_HOME/pan`` stays."""
    if not backup_path(paths).exists():
        echo(f"no setup backup at {backup_path(paths)}; nothing to restore. "
             "To switch back to Hermes' built-in memory: hermes config set memory.provider \"\"")
        return 1
    for cmd in restore_commands(paths, hermes):
        echo(("would run: " if dry_run else "running: ") + " ".join(cmd))
        if not dry_run:
            run(cmd, check=True)
    if not dry_run:
        done = backup_path(paths).with_name(BACKUP_NAME + ".restored")
        backup_path(paths).replace(done)
        echo(f"settings restored; backup kept as {done}")
    echo(f"PAN data stays in {paths.root} (delete it yourself if you no longer need it); "
         "remove the package with: uv pip uninstall --python <hermes-venv>/bin/python pan-agent")
    return 0
