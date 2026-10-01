"""`pan setup` safety net: tested-Hermes refusal, settings backup, `pan uninstall` (ADR-011 §2–3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pan import HERMES_PIN, cli, install
from pan.paths import PanPaths

KEYS = list(cli.HERMES_SETTINGS)


def _home(tmp_path: Path, config: str | None) -> PanPaths:
    home = tmp_path / "home"
    home.mkdir()
    if config is not None:
        (home / "config.yaml").write_text(config)
    return PanPaths.for_home(home)


def test_tested_hermes_problem():
    assert install.tested_hermes_problem(HERMES_PIN) is None
    msg = install.tested_hermes_problem("0.0.1")
    assert "0.0.1" in msg and HERMES_PIN in msg and "--force" in msg


def test_backup_records_previous_values_once(tmp_path):
    paths = _home(tmp_path, "memory:\n  provider: holographic\n  memory_enabled: false\n")
    target, written = install.write_backup(paths, KEYS)
    assert written
    rec = json.loads(target.read_text())
    assert rec["settings"] == {"memory.provider": "holographic", "memory.memory_enabled": False,
                               "memory.user_profile_enabled": None, "memory.nudge_interval": None}
    assert (paths.root / rec["config_copy"]).read_text().startswith("memory:")
    # a second setup (config now holds PAN's values) must not overwrite the pre-PAN record
    (paths.hermes_home / "config.yaml").write_text("memory:\n  provider: pan\n")
    _, written_again = install.write_backup(paths, KEYS)
    assert not written_again
    assert json.loads(target.read_text())["settings"]["memory.provider"] == "holographic"


def test_backup_without_config_file(tmp_path):
    paths = _home(tmp_path, None)
    target, _ = install.write_backup(paths, KEYS)
    rec = json.loads(target.read_text())
    assert rec["config_copy"] is None and set(rec["settings"].values()) == {None}


def test_uninstall_restores_set_and_unset(tmp_path):
    paths = _home(tmp_path, "memory:\n  provider: holographic\n  memory_enabled: false\n  nudge_interval: 10\n")
    install.write_backup(paths, KEYS)
    ran, out = [], []
    rc = install.uninstall(paths, "hermes", dry_run=False, run=lambda cmd, check: ran.append(cmd), echo=out.append)
    assert rc == 0
    assert ["hermes", "config", "set", "memory.provider", "holographic"] in ran
    assert ["hermes", "config", "set", "memory.memory_enabled", "false"] in ran
    assert ["hermes", "config", "set", "memory.nudge_interval", "10"] in ran
    assert ["hermes", "config", "unset", "memory.user_profile_enabled"] in ran
    assert not install.backup_path(paths).exists()
    assert install.backup_path(paths).with_name("setup-backup.json.restored").exists()


def test_uninstall_dry_run_and_missing_backup(tmp_path):
    paths = _home(tmp_path, "")
    out: list[str] = []
    assert install.uninstall(paths, "hermes", dry_run=True, echo=out.append) == 1
    install.write_backup(paths, KEYS)
    ran: list = []
    assert install.uninstall(paths, "hermes", dry_run=True, run=lambda *a, **k: ran.append(a), echo=out.append) == 0
    assert not ran and install.backup_path(paths).exists()
    assert any(line.startswith("would run: hermes config unset") for line in out)


def test_setup_refuses_untested_hermes(tmp_path, monkeypatch, capsys):
    paths = _home(tmp_path, "")
    monkeypatch.setenv("HERMES_HOME", str(paths.hermes_home))
    monkeypatch.setattr(cli, "_hermes_bin", lambda: "hermes")
    monkeypatch.setattr("pan.hermes.compat.hermes_version", lambda: "9.9.9")
    calls: list = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: calls.append(a))
    assert cli.main(["setup"]) == 2
    assert "refusing" in capsys.readouterr().err and not calls and not paths.root.exists()
    assert cli.main(["setup", "--dry-run", "--force"]) == 0
    assert "warning (--force)" in capsys.readouterr().err and not calls


@pytest.mark.parametrize("argv", [["setup", "--dry-run"], ["uninstall", "--dry-run"]])
def test_cli_parses(argv, tmp_path, monkeypatch):
    paths = _home(tmp_path, "")
    monkeypatch.setenv("HERMES_HOME", str(paths.hermes_home))
    monkeypatch.setattr(cli, "_hermes_bin", lambda: "hermes")
    monkeypatch.setattr("pan.hermes.compat.hermes_version", lambda: HERMES_PIN)
    assert cli.main(argv) in (0, 1)
