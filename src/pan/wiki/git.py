"""Minimal git wrapper for the wiki repo (subprocess, no GitPython).

The daemon (M3) uses it for one commit per curator decision batch, touching only its own paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_AUTHOR = "pan-memoryd <pan-memoryd@localhost>"
_AUTHOR_NAME, _AUTHOR_EMAIL = "pan-memoryd", "pan-memoryd@localhost"
# Never inherit a surrounding repo's context (e.g. when called from inside a git hook).
_SCRUB_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
              "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX", "GIT_COMMON_DIR")


class GitError(RuntimeError):
    pass


def git_available() -> bool:
    return shutil.which("git") is not None


def _split_author(author: str) -> tuple[str, str]:
    if "<" in author and author.rstrip().endswith(">"):
        name, email = author.rsplit("<", 1)
        return name.strip(), email.rstrip().rstrip(">").strip()
    return author.strip(), _AUTHOR_EMAIL


class GitRepo:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- plumbing ------------------------------------------------------------------------------

    def _run(self, *args: str, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
        full_env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
        full_env.update({"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
        if env:
            full_env.update(env)
        # gpg signing from a global config would make unattended commits hang or fail.
        cmd = ["git", "-c", "commit.gpgsign=false", "-C", str(self.root), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
        return proc

    @staticmethod
    def _pathspec(paths: Optional[Iterable[str | Path]]) -> List[str]:
        return ["--", *[str(p) for p in paths]] if paths is not None else []

    # -- API -----------------------------------------------------------------------------------

    def is_repo(self) -> bool:
        return (self.root / ".git").exists()

    def init(self, author: str = DEFAULT_AUTHOR) -> bool:
        """Create the repo with a local identity. Returns False if it already existed."""
        if self.is_repo():
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        self._run("init", "-q")
        name, email = _split_author(author)
        self._run("config", "user.name", name)
        self._run("config", "user.email", email)
        return True

    def head(self) -> Optional[str]:
        proc = self._run("rev-parse", "--verify", "-q", "HEAD", check=False)
        return (proc.stdout.strip() or None) if proc.returncode == 0 else None

    def dirty_paths(self, paths: Optional[Sequence[str | Path]] = None) -> List[str]:
        """Paths (relative to the repo) with staged, unstaged or untracked changes."""
        if paths is not None and not paths:
            return []
        proc = self._run("status", "--porcelain=v1", "-z", "--untracked-files=all",
                         *self._pathspec(paths))
        out: List[str] = []
        entries = proc.stdout.split("\0")
        i = 0
        while i < len(entries):
            entry = entries[i]
            i += 1
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            if "R" in code or "C" in code:
                i += 1  # skip the rename source
            out.append(path)
        return sorted(out)

    def is_dirty(self, paths: Optional[Sequence[str | Path]] = None) -> bool:
        return bool(self.dirty_paths(paths))

    def commit(self, paths: Sequence[str | Path], message: str,
               author: str = DEFAULT_AUTHOR) -> Optional[str]:
        """Stage and commit exactly ``paths`` (additions, edits, deletions). Returns the new SHA,
        or None if those paths had no changes. Other dirty files in the repo are left alone."""
        if not paths:
            return None
        rel = [str(p) for p in paths]
        if not self.dirty_paths(rel):
            return None
        self._run("add", "-A", "--", *rel)
        name, email = _split_author(author)
        env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
               "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
        self._run("commit", "-q", "-m", message, "--", *rel, env=env)
        return self.head()

    def diff(self, rev: Optional[str] = None, paths: Optional[Sequence[str | Path]] = None) -> str:
        """Diff of the working tree against ``rev`` (default HEAD); on an empty repo, ''."""
        if rev is None and self.head() is None:
            return ""
        return self._run("diff", rev or "HEAD", *self._pathspec(paths)).stdout

    def show(self, rev: str = "HEAD") -> str:
        return self._run("show", "--format=%H%n%an <%ae>%n%s", rev).stdout

    def log(self, n: int = 20) -> List[str]:
        if self.head() is None:
            return []
        return self._run("log", f"-{n}", "--format=%H %s").stdout.splitlines()
