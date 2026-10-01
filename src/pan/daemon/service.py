"""systemd user unit for pan-memoryd: render, install, uninstall (never enables or starts anything).

``pan daemon install`` writes ``~/.config/systemd/user/pan-memoryd.service`` (or ``--unit-dir``) and
prints the ``systemctl --user`` commands to run; enabling the service is left to the user.
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path
from typing import Optional

UNIT_NAME = "pan-memoryd.service"
TEMPLATE = Path(__file__).with_name("pan-memoryd.service")


def default_unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "systemd" / "user"


def _quote(arg: str) -> str:
    """systemd command-line quoting (double quotes when needed; ``\\`` and ``"`` escaped, ``%`` doubled)."""
    arg = arg.replace("%", "%%")
    if arg and not any(c in arg for c in ' \t"\'\\;$'):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


def memoryd_executable() -> list[str]:
    """The venv's ``pan-memoryd`` console script, else ``python -m pan.daemon.memoryd``."""
    script = Path(sys.executable).with_name("pan-memoryd")
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "pan.daemon.memoryd"]


def render_unit(hermes_home: Path, executable: Optional[list[str]] = None) -> str:
    home = str(Path(hermes_home).expanduser().resolve())
    cmd = [*(executable or memoryd_executable()), "run", "--hermes-home", home]
    template = string.Template(TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(exec_start=" ".join(_quote(c) for c in cmd), hermes_home=home.replace("%", "%%"),
                               hermes_home_env=_quote(f"HERMES_HOME={home}"))


def install(hermes_home: Path, *, unit_dir: Optional[Path] = None, dry_run: bool = False,
            executable: Optional[list[str]] = None) -> tuple[Path, str, bool]:
    """Write the unit. Returns (path, content, changed). ``dry_run`` writes nothing."""
    target = (unit_dir or default_unit_dir()) / UNIT_NAME
    content = render_unit(hermes_home, executable)
    changed = not target.exists() or target.read_text(encoding="utf-8") != content
    if not dry_run and changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return target, content, changed


def uninstall(*, unit_dir: Optional[Path] = None, dry_run: bool = False) -> tuple[Path, bool]:
    """Remove the unit file. Returns (path, existed)."""
    target = (unit_dir or default_unit_dir()) / UNIT_NAME
    existed = target.exists()
    if existed and not dry_run:
        target.unlink()
    return target, existed
