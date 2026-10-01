"""Profile-scoped file locations (spec §3). Always derived from the hermes_home Hermes passes in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PanPaths:
    hermes_home: Path

    @classmethod
    def for_home(cls, hermes_home: str | Path) -> "PanPaths":
        return cls(Path(hermes_home).expanduser())

    @property
    def root(self) -> Path:
        return self.hermes_home / "pan"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def events_db(self) -> Path:
        return self.root / "events.db"

    @property
    def wiki(self) -> Path:
        return self.root / "wiki"

    @property
    def index_db(self) -> Path:
        return self.root / "index.db"

    @property
    def vectors_db(self) -> Path:
        """Derived dense-vector index (ADR-012), next to ``index_db``."""
        return self.root / "vectors.db"

    @property
    def curation_log(self) -> Path:
        return self.root / "curation-log.jsonl"

    @property
    def dead(self) -> Path:
        return self.root / "dead"

    @property
    def memories(self) -> Path:
        """Hermes L1 files (MEMORY.md, USER.md). PAN writes them only via Hermes' MemoryStore."""
        return self.hermes_home / "memories"

    @property
    def state_db(self) -> Path:
        """Hermes session store (read-only for PAN)."""
        return self.hermes_home / "state.db"

    def ensure(self) -> None:
        for d in (self.root, self.wiki, self.dead):
            d.mkdir(parents=True, exist_ok=True)
