"""systemd unit rendering/installation (tmp dirs only, never enabled) and the M3 CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import requires_hermes
from pan.cli import main as pan_main
from pan.config import DEFAULT_CONFIG_YAML, PanConfig, load_config, parse_config, write_default_config
from pan.daemon import service
from pan.paths import PanPaths


def test_unit_renders_with_quoting(tmp_path):
    home = tmp_path / "Pan Agent" / "home"
    unit = service.render_unit(home, ["/opt/venv with space/bin/pan-memoryd"])
    assert '\nExecStart="/opt/venv with space/bin/pan-memoryd" run --hermes-home "' in unit
    assert f'\nEnvironment="HERMES_HOME={home.resolve()}"' in unit
    assert "${" not in unit and "[Install]\nWantedBy=default.target" in unit


def test_install_and_uninstall_in_tmp_unit_dir(tmp_path):
    unit_dir = tmp_path / "systemd" / "user"
    target, content, changed = service.install(tmp_path / "home", unit_dir=unit_dir, dry_run=True)
    assert changed and not target.exists() and "pan-memoryd" in content
    target, _, changed = service.install(tmp_path / "home", unit_dir=unit_dir)
    assert changed and target == unit_dir / service.UNIT_NAME and target.read_text() == content
    assert service.install(tmp_path / "home", unit_dir=unit_dir)[2] is False  # idempotent
    assert service.uninstall(unit_dir=unit_dir, dry_run=True) == (target, True) and target.exists()
    assert service.uninstall(unit_dir=unit_dir) == (target, True) and not target.exists()
    assert service.uninstall(unit_dir=unit_dir) == (target, False)


def test_default_unit_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert service.default_unit_dir() == tmp_path / "cfg" / "systemd" / "user"


def test_cli_daemon_install_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert pan_main(["daemon", "install", "--dry-run", "--unit-dir", str(tmp_path / "units")]) == 0
    assert "ExecStart=" in capsys.readouterr().out
    assert not (tmp_path / "units").exists()
    assert pan_main(["daemon", "install", "--unit-dir", str(tmp_path / "units")]) == 0
    out = capsys.readouterr().out
    assert "not enabled" in out and (tmp_path / "units" / service.UNIT_NAME).exists()
    assert pan_main(["daemon", "uninstall", "--unit-dir", str(tmp_path / "units")]) == 0


def test_default_config_file(tmp_path):
    paths = PanPaths.for_home(tmp_path)
    assert write_default_config(paths) and not write_default_config(paths)
    assert paths.config.read_text() == DEFAULT_CONFIG_YAML
    assert load_config(paths) == PanConfig()
    import yaml
    assert parse_config(yaml.safe_load(DEFAULT_CONFIG_YAML)) == PanConfig()


def test_status_shows_spool_and_daemon(pan_home, capsys, monkeypatch):
    from conftest import spool_scenarios

    spool_scenarios(pan_home, ["noise"])
    assert pan_main(["status"]) == 0
    out = capsys.readouterr().out
    assert "new=2, processing=0, done=0, dead=0" in out
    assert "pan-memoryd" in out and "not running" in out


@requires_hermes  # replay writes L1 through Hermes' MemoryStore
def test_memory_inspect_and_replay(pan_home, capsys, tmp_path):
    from conftest import spool_scenarios

    spool_scenarios(pan_home, ["duplicate", "preference"])
    assert pan_main(["memoryd", "run", "--once", "--hermes-home", str(pan_home)]) == 0
    capsys.readouterr()
    assert pan_main(["memory", "inspect", "--session", "sess-dup"]) == 0
    out = capsys.readouterr().out
    assert "01K0DUP0000000000000000001" in out and "environment→wiki" in out and "ignored" in out
    assert pan_main(["memory", "inspect", "--json", "--limit", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["events"]) == 2 and {r["outcome"] for r in data["decisions"]} == {"ignored", "l1_added"}
    assert pan_main(["memory", "replay", "--out", str(tmp_path / "rp"), "--json"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    # the real run already put the preference into USER.md, so the replay (copy of it) sees a duplicate
    assert sorted(r["outcome"] for r in replayed) == ["ignored", "l1_duplicate"]
    assert (Path(tmp_path / "rp" / "hermes_home" / "pan" / "events.db")).exists()
