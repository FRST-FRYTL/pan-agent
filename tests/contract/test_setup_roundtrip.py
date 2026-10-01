"""Contract: `pan setup` then `pan uninstall` through the real `hermes config` CLI leaves the settings
as they were (ADR-011 §3). Runs against a scratch profile only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.contract

HERMES = Path(sys.executable).with_name("hermes")
PAN = Path(sys.executable).with_name("pan")


def _run(exe: Path, *args: str, home: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HERMES_HOME": str(home)}
    return subprocess.run([str(exe), *args], env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not (HERMES.exists() and PAN.exists()), reason="hermes / pan entry points not in this venv")
def test_setup_uninstall_roundtrip(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  provider: ''\n  memory_enabled: false\n")
    before = yaml.safe_load((home / "config.yaml").read_text())["memory"]

    r = _run(PAN, "setup", home=home)
    assert r.returncode == 0, r.stderr + r.stdout
    mem = yaml.safe_load((home / "config.yaml").read_text())["memory"]
    assert mem["provider"] == "pan" and mem["memory_enabled"] is True
    assert (home / "pan" / "setup-backup.json").exists()

    r = _run(PAN, "uninstall", home=home)
    assert r.returncode == 0, r.stderr + r.stdout
    after = yaml.safe_load((home / "config.yaml").read_text())["memory"]
    assert after.get("provider") in ("", None) and after["memory_enabled"] is False
    assert "user_profile_enabled" not in after and "nudge_interval" not in after, after
    assert before == {"provider": "", "memory_enabled": False}
