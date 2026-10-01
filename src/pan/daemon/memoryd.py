"""pan-memoryd — background curation service (integration spec §4.5, §9).

One instance per ``HERMES_HOME`` (``fcntl`` lock on ``pan/memoryd.lock``). Each batch:

    claim events → group into episodes → classify → curate → apply → reindex → mark done

- **apply (wiki)**: write the page, validate the wiki (a write that introduces a validation issue is
  reverted and its events fail), reindex the page; one git commit per batch (author ``pan-memoryd``)
  covering only the daemon's own paths. A commit failure reverts the batch's writes.
- **apply (L1)**: adds through Hermes' MemoryStore (:mod:`pan.memory.l1`); ``l1_full`` → the
  curator writes the fact to the wiki instead.
- **L1 reconciliation (M5, D1)**: after every wiki candidate, each *observed* claim is checked
  against the L1 entries; a contradicted factual entry (same subject, different port/host/path/
  version/…) is superseded via ``MemoryStore.replace`` — unless an ``l1_write`` event shows the
  entry was written after the observation. Results are in the record's ``l1`` list.
- **agent L1 writes (M5)**: an ``l1_write`` event with a factual ``MEMORY.md`` entry (not a
  preference, not a note about memory/tools — M6) reconciles the other L1 entries against it and goes to the wiki as an ``inferred``
  claim — usually the wiki already has the fact (IGNORE), then only the page's ``sources`` get the
  event and ``l1:memory`` (provenance link). An agent ``replace`` supersedes the old wiki bullet.
- **index**: a page the daemon creates is linked from ``wiki/index.md`` (``## Pages`` → area) in
  the same commit.
- **crash backfill (M5)**: a turn-less group closed by idle/flush is completed from Hermes'
  ``state.db`` when it holds the missing turn (:mod:`pan.memory.backfill`, read-only).
- **defer**: a target page with uncommitted manual edits is not touched; its events go back to
  the spool (not counted as a failed attempt) and are retried on a later poll.
- **failures**: ``mark_failed`` (attempts++); an event that turns ``dead`` is also copied to
  ``pan/dead/<id>.json``.
- every classification and decision is appended to ``pan/curation-log.jsonl``.
- **classifier (M9)**: ``classifier.kind`` in the PAN config picks the LLM gate (``llm``, the default;
  :mod:`pan.memory.llm_gate`), rules-v2 (``rules``) or ``shadow``; the gate's per-episode info (version,
  latency, fallback reason, …) is logged as ``gate`` on the episode's first record.

Open episodes (tool calls whose ``turn`` event has not arrived) are released back to the spool and
picked up again on a later poll; they are closed after ``daemon.episode_idle_minutes``.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pan.config import PanConfig, load_config
from pan.events.schema import (Actor, AgentEvent, CuratorAction, Destination, EventType, MemoryCandidate,
                               MemoryType, new_ulid, to_jsonable)
from pan.events.spool import EventSpool, default_worker_id
from pan.index.fts import FtsIndex
from pan.memory.backfill import StateDbBackfill
from pan.memory.classifier import Classifier, RuleClassifier, classification, meta_talk
from pan.memory.curator import Curation, Curator, DeterministicCurator, explicit_olds
from pan.memory.episodes import Episode, group_events, parse_ts, previous_turn
from pan.memory.claims import normalize
from pan.memory.dates import bare
from pan.memory.facts import subject_of, subject_tags, values
from pan.memory.llm_gate import build_classifier
from pan.memory.l1 import ADDED, DUPLICATE, ERROR, L1_FULL, L1Result, L1Writer, is_preference_entry
from pan.memory.retrieval import build_retriever
from pan.paths import PanPaths
from pan.wiki.git import DEFAULT_AUTHOR, GitError, git_available
from pan.wiki.frontmatter import Page
from pan.wiki.store import WikiError, WikiStore

logger = logging.getLogger("pan.memoryd")

LOCK_NAME = "memoryd.lock"
LOG_NAME = "memoryd.log"
PRUNE_EVERY_SECONDS = 3600.0

# Outcomes in the curation log.
COMMITTED = "committed"      # wiki change written and committed
APPLIED = "applied"          # wiki change written, not committed (auto_commit off / no git)
IGNORED = "ignored"
DROPPED = "dropped"          # noise
DEFERRED = "deferred"
FAILED = "failed"
L1_ADDED = "l1_added"
L1_DUPLICATE = "l1_duplicate"
LOGGED = "logged"            # l1_write event recorded only (preference / user profile / removal)
AGENT_AUTHOR = "hermes-agent <hermes-agent@localhost>"   # commits of wiki edits the agent made itself


# -- single-instance lock -------------------------------------------------------------------------

class DaemonLock:
    """Exclusive ``flock`` on ``pan/memoryd.lock``; the file holds the owner's pid."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "DaemonLock":
        if not self.acquire():
            raise RuntimeError(f"pan-memoryd already running (lock {self.path}, pid {running_pid(self.path)})")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def running_pid(lock_path: Path) -> Optional[int]:
    """Pid of the daemon holding ``lock_path`` (0 if the pid is unreadable), None if nobody holds it."""
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            try:
                return int(os.read(fd, 32).decode().strip() or 0)
            except ValueError:
                return 0
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def lock_path(paths: PanPaths) -> Path:
    return paths.root / LOCK_NAME


# -- batch bookkeeping ----------------------------------------------------------------------------

@dataclass
class BatchReport:
    batch_id: str = ""
    claimed: int = 0
    episodes: int = 0
    pending: int = 0
    markers: int = 0
    outcomes: Counter = field(default_factory=Counter)
    commit: Optional[str] = None
    done: int = 0
    failed: int = 0
    deferred: int = 0
    dead: int = 0

    @property
    def idle(self) -> bool:
        """No progress: nothing claimed, or only open episodes / deferred decisions."""
        return self.done == 0 and self.failed == 0

    def summary(self) -> str:
        outs = ", ".join(f"{k}={v}" for k, v in sorted(self.outcomes.items())) or "-"
        return (f"batch {self.batch_id}: claimed={self.claimed} episodes={self.episodes} pending={self.pending} "
                f"done={self.done} failed={self.failed} deferred={self.deferred} outcomes[{outs}]"
                + (f" commit={self.commit[:10]}" if self.commit else ""))


@dataclass
class _Write:
    path: str
    before: Optional[str]      # None = file did not exist


@dataclass
class _Batch:
    writes: Dict[str, _Write] = field(default_factory=dict)       # path → first snapshot in this batch
    records: List[Dict[str, Any]] = field(default_factory=list)
    episode_paths: Dict[str, List[str]] = field(default_factory=dict)
    episode_events: Dict[str, List[str]] = field(default_factory=dict)  # episodes with wiki writes
    summaries: List[str] = field(default_factory=list)


class _Deferred(Exception):
    pass


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# -- the daemon -----------------------------------------------------------------------------------

class MemoryDaemon:
    """The pipeline. ``run_once`` processes one batch; ``run`` polls until ``stop`` is set."""

    def __init__(self, paths: PanPaths, config: Optional[PanConfig] = None, *,
                 classifier: Optional[Classifier] = None, curator: Optional[Curator] = None,
                 l1: Optional[L1Writer] = None, clock: Callable[[], float] = time.time,
                 worker_id: Optional[str] = None) -> None:
        self.paths = paths
        self.config = config or load_config(paths)
        self.clock = clock
        self.worker_id = worker_id or default_worker_id()
        paths.ensure()
        self.spool = EventSpool(paths.events_db)
        self.store = WikiStore(paths.wiki)
        self.index = FtsIndex(paths.index_db)
        self.classifier = classifier or build_classifier(self.config)
        self.curator = curator or DeterministicCurator(
            self.store, build_retriever(paths, self.config.retrieval, purpose="curator", fts_index=self.index),
            today=self.today)
        self._embed_client = None
        self._vectors_down_logged = False
        self.l1 = l1 or L1Writer(paths.hermes_home)
        self.git = self.store.git if git_available() else None
        self.backfill = StateDbBackfill(paths.state_db)
        self._known = ""
        self._known_entries: List[str] = []
        self._own_l1: Dict[str, List[str]] = {}
        self._pan_l1: Optional[Dict[str, str]] = None
        self._deferred_logged: set[str] = set()
        self._last_prune = 0.0

    # -- lifecycle ---------------------------------------------------------------------------------

    def today(self) -> str:
        return datetime.fromtimestamp(self.clock()).date().isoformat()

    def prepare(self) -> None:
        """Wiki skeleton + git repo (idempotent) and a fresh index (the wiki may have changed by hand)."""
        self.store.init(git=self.git is not None)
        if self.git is not None and not self.git.is_repo():
            self.git = None
        self.index.rebuild(self.paths.wiki)
        self.sync_vectors()

    # -- dense vectors (M7, ADR-012): derived like the FTS index; best effort ------------------------

    def sync_vectors(self, pages: Optional[List[Page]] = None) -> None:
        """Embed new/changed pages into ``vectors.db`` (all pages when ``pages`` is None, which also
        drops deleted ones). A missing sidecar only means FTS-only retrieval until the next sync."""
        from pan.index.embed import EmbedClient, EmbedError
        from pan.index.vectors import sync_wiki

        cfg = self.config.retrieval
        if "hybrid" not in (cfg.mode, cfg.curator_mode):
            return
        if self._embed_client is None:
            self._embed_client = EmbedClient(cfg.embed_url, timeout_s=max(cfg.timeout_s, 30.0),
                                             embed_model=cfg.embed_model, dimensions=cfg.embed_dimensions,
                                             cooldown_s=5.0)
        try:
            sync_wiki(self.paths, cfg, pages=pages, client=self._embed_client)
            self._vectors_down_logged = False
        except EmbedError as exc:
            if not self._vectors_down_logged:
                logger.warning("vectors.db not updated (retrieval sidecar unavailable: %s); FTS only", exc)
                self._vectors_down_logged = True
        except Exception:
            logger.warning("vectors.db update failed", exc_info=True)

    def close(self) -> None:
        self.index.close()
        self.spool.close()

    # -- one batch ---------------------------------------------------------------------------------

    def run_once(self, *, flush: bool = False) -> BatchReport:
        cfg = self.config.daemon
        report = BatchReport(batch_id=new_ulid())
        events = self.spool.claim(cfg.batch_size, self.worker_id, max_attempts=cfg.max_attempts)
        report.claimed = len(events)
        if not events:
            return report
        grouping = group_events(events, now=self.clock(), idle_seconds=cfg.episode_idle_minutes * 60, flush=flush)
        self.spool.release(e.id for e in grouping.pending)
        report.pending, report.markers, report.episodes = (len(grouping.pending), len(grouping.markers),
                                                           len(grouping.episodes))
        if cfg.backfill_from_state_db:
            for episode in grouping.episodes:
                self._backfill(episode)
        batch = _Batch()
        self._known = self._l1_snapshot()
        self._adopt_agent_wiki_edits(grouping.episodes, batch, report.batch_id)
        wiki_before = self.git.head() if self.git is not None else None
        baseline = {str(i) for i in self.store.validate()}
        done: List[str] = []
        failed: Dict[str, str] = {}
        deferred: List[str] = []

        # Episodes and markers in event order: an agent's L1 write is reconciled before the facts
        # observed after it (and never re-applied over them).
        # An l1_write belongs to the turn it happened in: it is handled right after that turn's
        # episode (the user's own words are curated first; the agent's copy then links to them).
        units: List[tuple[tuple[str, int], int, Any]] = [((ep.id, 0), i, ep)
                                                          for i, ep in enumerate(grouping.episodes)]
        for m in grouping.markers:
            owner = None
            if m.event_type is EventType.L1_WRITE:
                owner = next((ep for ep in grouping.episodes if ep.session_id == m.session_id
                              and ep.turn is not None and ep.turn.id > m.id), None)
                if owner is not None and m.content.get("content"):
                    self._own_l1.setdefault(owner.id, []).append(str(m.content["content"]))
            units.append(((owner.id if owner is not None else m.id, 1), -1, m))
        for _, i, unit in sorted(units, key=lambda u: u[0]):
            if i < 0:
                marker = unit
                if marker.event_type is EventType.L1_WRITE:
                    status, error = self._process_l1_write(marker, batch, baseline, report.batch_id)
                    if status == DEFERRED:
                        deferred.append(marker.id)
                        continue
                    if status == FAILED:
                        failed[marker.id] = error
                        continue
                done.append(marker.id)
                continue
            episode = unit
            previous = previous_turn(grouping.episodes, i) or self._history_turn(episode)
            status, error = self._process_episode(episode, previous, batch, baseline, report.batch_id)
            if status == "done":
                done.extend(episode.event_ids)
            elif status == DEFERRED:
                deferred.extend(episode.event_ids)
            else:
                failed.update({eid: error for eid in episode.event_ids})

        commit = self._commit(batch, report.batch_id, wiki_before, done, failed)
        report.commit = commit
        if batch.writes:
            self.sync_vectors()  # full sync: also drops pages a failed episode restored away
        for rec in batch.records:
            rec["wiki_before"] = wiki_before
            if rec.get("outcome") == APPLIED and commit:
                rec["outcome"], rec["commit"] = COMMITTED, commit
            report.outcomes[rec.get("outcome", "?")] += 1
        self._append_log(batch.records)

        report.done = self.spool.mark_done(done)
        self.spool.release(deferred)
        report.deferred = len(deferred)
        for eid, error in failed.items():
            status = self.spool.mark_failed(eid, error, max_attempts=cfg.max_attempts)
            report.failed += 1
            if status == "dead":
                report.dead += 1
                self._write_dead(eid)
        self._maybe_prune()
        if not report.idle:
            logger.info(report.summary())
        return report

    def drain(self, *, flush: bool = False, max_batches: int = 1000) -> List[BatchReport]:
        """``run_once`` until a batch processes nothing (``--once``, replay, tests)."""
        reports = []
        for _ in range(max_batches):
            report = self.run_once(flush=flush)
            reports.append(report)
            if report.idle:
                break
        return reports

    def run(self, stop: threading.Event) -> None:
        poll = max(0.1, float(self.config.daemon.poll_seconds))
        logger.info("pan-memoryd running for %s (poll %.1fs)", self.paths.hermes_home, poll)
        while not stop.is_set():
            try:
                report = self.run_once()
            except Exception:
                logger.exception("pan-memoryd batch failed")
                report = BatchReport()
            if report.idle:
                stop.wait(poll)
        logger.info("pan-memoryd stopped")

    # -- episodes ----------------------------------------------------------------------------------

    def _history_turn(self, episode: Episode) -> Optional[Episode]:
        prior = self.spool.history(episode.session_id, before_id=episode.id, limit=1,
                                   event_types=[EventType.TURN.value])
        return Episode(episode.session_id, prior, closed_by="turn") if prior else None

    def _backfill(self, episode: Episode) -> None:
        """Complete a turn-less idle/flush group from state.db (see :mod:`pan.memory.backfill`)."""
        if episode.turn is not None or episode.closed_by not in ("idle", "flush") or not self.backfill.available():
            return
        prior = self.spool.last_event(episode.session_id, EventType.TURN.value, before_id=episode.id)
        found = self.backfill.missing_turn(
            episode.session_id, first_event_ts=parse_ts(episode.events[0].ts),
            after_ts=parse_ts(prior.ts) if prior else 0.0,
            captured_user=str(prior.content.get("user") or "") if prior else "")
        if not found:
            return
        event = AgentEvent(
            event_type=EventType.TURN, session_id=episode.session_id, actor=Actor.MAIN_AGENT,
            content={"user": found["user"], "assistant": found["assistant"], "messages": found["messages"],
                     "backfill": "state.db"},
            source_refs=[f"session:{episode.session_id}", *(f"msg:{i}" for i in found["message_ids"])],
            parent_session_id=episode.events[0].parent_session_id, project=episode.project,
            metadata={"backfill": "state.db"})
        try:
            self.spool.append_claimed(event, self.worker_id)
        except Exception as exc:
            logger.warning("could not record backfilled turn for %s: %s", episode.session_id, exc)
            return
        episode.events.append(event)
        episode.closed_by += "+state.db"
        logger.info("backfilled turn of session %s from state.db (%d messages)", episode.session_id,
                    len(found["messages"]))

    def _wiki_rel(self, path: str) -> Optional[str]:
        """Wiki-relative path when ``path`` (absolute, as the agent's file tools report it) lies in
        the wiki; None otherwise."""
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                return None
            rel = p.resolve().relative_to(self.paths.wiki.resolve())
        except (ValueError, OSError):
            return None
        return rel.as_posix() if rel.suffix == ".md" and ".git" not in rel.parts else None

    def _adopt_agent_wiki_edits(self, episodes: List[Episode], batch: _Batch, batch_id: str) -> None:
        """The agent edited a wiki page with its own file tools (layer-B run: ``patch`` on
        ``pan/wiki/operations/vllm-server.md``). PAN is the single writer, but reverting would lose
        work and leaving the file dirty would defer every later decision on it forever. So the edit
        is committed as-is, author ``hermes-agent`` (auditable in git, reindexed), when the page
        still parses; otherwise it is left for a human (logged)."""
        if self.git is None:
            return
        edits: Dict[str, List[str]] = {}
        for ep in episodes:
            for ev in ep.file_changes:
                rel = self._wiki_rel(str(ev.content.get("path") or ""))
                if rel is not None:
                    edits.setdefault(rel, []).append(ev.id)
        for rel, event_ids in sorted(edits.items()):
            if not self.git.is_dirty([rel]):
                continue
            page = self.store.get_path(rel)
            rec: Dict[str, Any] = {"ts": _iso(self.clock()), "batch_id": batch_id, "worker": self.worker_id,
                                   "kind": "agent_wiki_edit", "event_ids": event_ids, "path": rel,
                                   "session_id": None, "episode_id": None}
            if page is None and (self.paths.wiki / rel).exists():
                rec.update(outcome="left", error="page no longer parses; left for manual review")
            else:
                try:
                    sha = self.git.commit([rel], f"hermes-agent: edit to {rel} (adopted by pan-memoryd)\n\n"
                                          f"Events: {', '.join(event_ids)}\nBatch: {batch_id}",
                                          author=AGENT_AUTHOR)
                except GitError as exc:
                    rec.update(outcome="left", error=f"commit failed: {exc}")
                else:
                    rec.update(outcome="adopted", commit=sha)
                    if page is not None:
                        self.index.update([page])
                    else:
                        self.index.remove_paths([rel])
            logger.warning("agent edited wiki page %s directly (%s)", rel, rec["outcome"])
            batch.records.append(rec)

    def _l1_snapshot(self) -> str:
        """Current L1 text (both files): what the agent has in its prompt anyway."""
        self._own_l1 = {}
        try:
            self._known_entries = list(self.l1.entries("memory") + self.l1.entries("user"))
        except Exception:
            self._known_entries = []
        return "\n".join(self._known_entries)

    def _known_for(self, episode: Episode) -> str:
        """L1 snapshot for ``episode``. A classifier with ``exclude_own_l1_writes`` (the LLM gate)
        does not see the entries the agent wrote *during this turn*: the agent's ``memory`` tool
        saves the user's words before PAN runs, and the gate would take the user's statement for a
        restatement of known memory (validation failure shape: notes, retractions and deadlines lost)."""
        own = [normalize(x) for x in self._own_l1.get(episode.id, [])]
        if not own or not getattr(self.classifier, "exclude_own_l1_writes", False):
            return self._known
        keep = [e for e in self._known_entries
                if not any(o and (o == normalize(e) or o in normalize(e) or normalize(e) in o) for o in own)]
        return "\n".join(keep)

    def _classify(self, episode: Episode, previous: Optional[Episode]) -> List[MemoryCandidate]:
        if getattr(self.classifier, "last_info", None) is not None:
            self.classifier.last_info = None  # type: ignore[attr-defined]
        try:
            return self.classifier.classify(episode, previous, known=self._known_for(episode))
        except TypeError:  # a classifier without the ``known`` keyword (older interface)
            return self.classifier.classify(episode, previous)

    def _gate_info(self) -> Optional[Dict[str, Any]]:
        """Per-episode info of a model gate (M9), None for classifiers that keep none."""
        info = getattr(self.classifier, "last_info", None)
        return dict(info) if isinstance(info, dict) else None

    def _process_episode(self, episode: Episode, previous: Optional[Episode], batch: _Batch,
                         baseline: set, batch_id: str) -> tuple[str, str]:
        try:
            candidates = self._classify(episode, previous)
        except Exception as exc:
            logger.exception("classifier failed on episode %s", episode.id)
            batch.records.append(self._record(batch_id, episode, None, None, FAILED, error=str(exc)))
            return FAILED, f"classifier: {exc}"
        gate = self._gate_info()
        status, error = "done", ""
        for n, cand in enumerate(candidates):
            try:
                outcome, curation, l1 = self._process_candidate(cand, episode, batch, baseline)
                rec_error = None
            except _Deferred as exc:
                outcome, curation, l1, rec_error = DEFERRED, None, None, str(exc)
                if status == "done":
                    status = DEFERRED
            except Exception as exc:
                logger.warning("curation failed for %s: %s", cand.id, exc, exc_info=True)
                outcome, curation, l1, rec_error = FAILED, None, None, f"{type(exc).__name__}: {exc}"
                status, error = FAILED, rec_error
            if outcome == DEFERRED and cand.id in self._deferred_logged:
                continue  # log a deferral once, not every poll
            if outcome == DEFERRED:
                self._deferred_logged.add(cand.id)
            rec = self._record(batch_id, episode, cand, curation, outcome, l1=l1, error=rec_error)
            if gate is not None:
                rec["classifier"] = gate.get("classifier") or rec["classifier"]
                rec["gate"] = gate if n == 0 else {k: gate[k] for k in ("version", "status") if k in gate}
            batch.records.append(rec)
        return status, error

    def _process_candidate(self, cand: MemoryCandidate, episode: Episode, batch: _Batch,
                           baseline: set) -> tuple[str, Optional[Curation], Optional[List[Dict[str, Any]]]]:
        cls = cand.classification
        if not cls.should_remember or cls.destination is Destination.NONE:
            return DROPPED, None, None
        if cls.destination in (Destination.USER, Destination.CORE):
            return self._apply_l1(cand, episode, batch, baseline)
        curation = self.curator.curate(cand)
        if curation.decision.action is CuratorAction.IGNORE:
            return IGNORED, curation, self._reconcile_l1(cand)
        self._apply_wiki(curation, episode, batch, baseline)
        return APPLIED, curation, self._reconcile_l1(cand)

    # -- L1 reconciliation (M5) ----------------------------------------------------------------------

    def _pan_l1_writes(self) -> Dict[str, str]:
        """Entries PAN itself wrote by superseding (text → id of the observation behind it), from the
        curation log (loaded once, then kept up to date in memory)."""
        if self._pan_l1 is None:
            self._pan_l1 = {}
            try:
                with open(self.paths.curation_log, encoding="utf-8") as fh:
                    for line in fh:
                        if '"superseded"' not in line:
                            continue
                        rec = json.loads(line)
                        for item in rec.get("l1") or []:
                            if item.get("status") == "superseded" and item.get("replacement"):
                                ids = [str(i) for i in rec.get("event_ids") or []]
                                self._note_pan_l1(item["replacement"], max(ids) if ids else "")
            except (OSError, ValueError):
                pass
        return self._pan_l1

    def _note_pan_l1(self, text: str, event_id: str) -> None:
        assert self._pan_l1 is not None
        key = text.strip()
        if event_id > self._pan_l1.get(key, ""):
            self._pan_l1[key] = event_id

    def _written_after(self, event_id: str) -> Callable[[str], bool]:
        """``entry → True`` when the entry was written after ``event_id``: by the agent (a newer
        ``l1_write`` event) or by PAN superseding it from a newer observation."""
        later = None
        pan = self._pan_l1_writes()

        def check(entry: str) -> bool:
            nonlocal later
            want = entry.strip()
            if pan.get(want, "") > event_id:
                return True
            if later is None:
                later = [e for e in self.spool.events_after(event_id, event_types=[EventType.L1_WRITE.value])]
            return any(str(e.content.get("content") or "").strip() == want for e in later)
        return check

    def _reconcile_l1(self, cand: MemoryCandidate, *, keep: tuple = ()) -> Optional[List[Dict[str, Any]]]:
        claims = [bare(c) for c, ev in zip(cand.normalized_claims, cand.claim_evidence) if ev == "observed"]
        if not claims or not cand.event_ids:
            return None
        reconcile = getattr(self.l1, "reconcile", None)
        if reconcile is None:
            return None
        observed_at = max(cand.event_ids)
        newer = self._written_after(observed_at)
        results: List[L1Result] = []
        olds = explicit_olds(cand)
        explicit = getattr(self.l1, "reconcile_explicit", None)
        if olds and explicit is not None:   # T3: the gate's explicit supersedes also reaches L1
            try:
                results.extend(explicit(olds, claims, keep=keep, newer=newer))
            except Exception as exc:
                logger.warning("L1 explicit reconcile failed for %s: %s", cand.id, exc)
                results.append(L1Result(ERROR, "", "; ".join(olds)[:120], f"{type(exc).__name__}: {exc}"))
        for claim in claims:
            try:
                results.extend(reconcile(claim, keep=keep, newer=newer))
            except Exception as exc:  # never fail a curated (maybe already written) wiki change
                logger.warning("L1 reconcile failed for %s: %s", cand.id, exc)
                results.append(L1Result(ERROR, "", claim, f"{type(exc).__name__}: {exc}"))
        for r in results:
            if r.replacement:
                self._note_pan_l1(r.replacement, observed_at)
        return [self._l1_json(r) for r in results] or None

    @staticmethod
    def _l1_json(r: L1Result) -> Dict[str, Any]:
        out = {"status": r.status, "target": r.target, "entry": r.entry, "message": r.message}
        if r.replacement:
            out["replacement"] = r.replacement
        return out

    def _process_l1_write(self, event: AgentEvent, batch: _Batch, baseline: set, batch_id: str) -> tuple[str, str]:
        """An agent ``memory`` tool write (see module docstring). Returns (status, error)."""
        c = event.content
        action, target = str(c.get("action") or ""), str(c.get("target") or "memory")
        entry = str(c.get("content") or "").strip()
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        factual = (action in ("add", "replace") and target == "memory" and entry
                   and not is_preference_entry(entry) and not meta_talk(entry)
                   and bool(values(entry).slots or subject_of(entry)))
        if not factual:
            batch.records.append(self._l1_write_record(event, batch_id))
            return "done", ""
        episode = Episode(event.session_id, [event], closed_by="l1_write")
        subject = subject_of(entry)
        previous = str(meta.get("previous_content") or "").strip()
        cand = MemoryCandidate(
            event_ids=[event.id], classification=classification(MemoryType.ENVIRONMENT),
            normalized_claims=[entry], retrieval_query=" ".join(x for x in (subject, entry) if x),
            id=f"{event.id}:l1_write", session_id=event.session_id, title=subject or entry[:70],
            tags=subject_tags(subject), claim_evidence=["inferred"],
            context=f"Saved by the agent to L1 {target.upper()}.md with Hermes' memory tool ({action}).",
            rationale=f"agent memory tool write ({action})", subject=subject,
            supersedes=[previous] if action == "replace" and previous else [])
        l1: List[Dict[str, Any]] = [{"status": "agent_write", "target": target, "action": action, "entry": entry}]
        try:
            # The agent's write is newer than whatever else L1 says about the same subject.
            reconciled = self.l1.reconcile(entry, keep=(entry,), newer=self._written_after(event.id)) \
                if hasattr(self.l1, "reconcile") else []
            for r in reconciled:
                if r.replacement:
                    self._note_pan_l1(r.replacement, event.id)
            l1 += [self._l1_json(r) for r in reconciled]
            curation = self.curator.curate(cand)
            if curation.decision.action is CuratorAction.IGNORE:
                linked = None
                for page_id in curation.decision.target_pages[:1]:
                    linked = getattr(self.curator, "link_provenance", None) and \
                        self.curator.link_provenance(cand, page_id, extra=[f"l1:{target}"])
                if linked:
                    self._apply_wiki(linked, episode, batch, baseline)
                    curation, outcome = linked, APPLIED
                else:
                    outcome = IGNORED
            else:
                if curation.page is not None:
                    sources = [str(x) for x in curation.page.meta.get("sources") or []]
                    if f"l1:{target}" not in sources:
                        curation.page.meta["sources"] = sources + [f"l1:{target}"]
                self._apply_wiki(curation, episode, batch, baseline)
                outcome = APPLIED
        except _Deferred as exc:
            if cand.id not in self._deferred_logged:
                self._deferred_logged.add(cand.id)
                batch.records.append(self._record(batch_id, episode, cand, None, DEFERRED, l1=l1, error=str(exc)))
            return DEFERRED, str(exc)
        except Exception as exc:
            logger.warning("l1_write processing failed for %s: %s", event.id, exc, exc_info=True)
            batch.records.append(self._record(batch_id, episode, cand, None, FAILED, l1=l1,
                                              error=f"{type(exc).__name__}: {exc}"))
            return FAILED, f"{type(exc).__name__}: {exc}"
        rec = self._record(batch_id, episode, cand, curation, outcome, l1=l1)
        rec["kind"] = "l1_write"
        batch.records.append(rec)
        return "done", ""

    def _apply_l1(self, cand: MemoryCandidate, episode: Episode, batch: _Batch,
                  baseline: set) -> tuple[str, Curation, List[Dict[str, Any]]]:
        plan = self.curator.plan_l1(cand)
        results = [self.l1.add(plan.l1_target, entry) for entry in plan.l1_entries]
        l1 = [{"status": r.status, "target": r.target, "entry": r.entry, "message": r.message} for r in results]
        errors = [r for r in results if r.status not in (ADDED, DUPLICATE, L1_FULL)]
        if errors:
            raise RuntimeError(f"L1 write failed: {errors[0].status}: {errors[0].message}")
        full = [i for i, r in enumerate(results) if r.status == L1_FULL]
        if full:
            claims = [cand.normalized_claims[i] for i in full]
            evidence = [cand.claim_evidence[i] for i in full if i < len(cand.claim_evidence)]
            wiki_cand = replace(cand, normalized_claims=claims, claim_evidence=evidence)
            curation = self.curator.curate(wiki_cand, to_wiki=True)
            curation.decision.rationale = f"l1_full → wiki; {curation.decision.rationale}"
            if curation.decision.action is CuratorAction.IGNORE:
                return IGNORED, curation, l1
            self._apply_wiki(curation, episode, batch, baseline)
            return APPLIED, curation, l1
        if all(r.status == DUPLICATE for r in results):
            plan.decision.action = CuratorAction.IGNORE
            plan.decision.rationale = "already in L1 " + plan.l1_target.upper() + ".md"
            plan.decision.patch = ""
            return L1_DUPLICATE, plan, l1
        return L1_ADDED, plan, l1

    def _apply_wiki(self, curation: Curation, episode: Episode, batch: _Batch, baseline: set) -> None:
        path, page = curation.path, curation.page
        if not path or page is None:
            raise WikiError("curation without a target page")
        full = self.paths.wiki / path
        if self.git is not None and path not in batch.writes and self.git.is_dirty([path]):
            raise _Deferred(f"{path} has uncommitted manual edits")
        before = full.read_text(encoding="utf-8") if full.exists() else None
        index_before: Optional[str] = None
        link = curation.decision.action is CuratorAction.CREATE
        if link:
            if self.git is not None and "index.md" not in batch.writes and self.git.is_dirty(["index.md"]):
                raise _Deferred("index.md has uncommitted manual edits")
            index_before = self.store.index_path.read_text(encoding="utf-8") if self.store.index_path.exists() else None
        self.store.write(page)
        linked = link and self.store.link_in_index(page, today=self.today())
        new_issues = [i for i in self.store.validate() if str(i) not in baseline]
        if new_issues:
            self._restore(path, before)
            if linked:
                self._restore("index.md", index_before)
            raise WikiError("validation failed: " + "; ".join(str(i) for i in new_issues[:5]))
        batch.writes.setdefault(path, _Write(path, before))
        if linked:
            batch.writes.setdefault("index.md", _Write("index.md", index_before))
            batch.episode_paths.setdefault(episode.id, []).append("index.md")
        batch.episode_paths.setdefault(episode.id, []).append(path)
        batch.episode_events[episode.id] = episode.event_ids
        batch.summaries.append(f"- {curation.decision.action.value} {path}")
        written = self.store.get_path(path)
        if written is not None:
            self.index.update([written])
            self.sync_vectors([written])  # later candidates of this batch must see the page
        if linked:
            index_page = self.store.get_path("index.md")
            if index_page is not None:
                self.index.update([index_page])

    def _restore(self, path: str, before: Optional[str]) -> None:
        full = self.paths.wiki / path
        if before is None:
            full.unlink(missing_ok=True)
            self.index.remove_paths([path])
        else:
            full.write_text(before, encoding="utf-8")
            page = self.store.get_path(path)
            if page is not None:
                self.index.update([page])

    def _commit(self, batch: _Batch, batch_id: str, wiki_before: Optional[str], done: List[str],
                failed: Dict[str, str]) -> Optional[str]:
        if not batch.writes or self.git is None or not self.config.wiki.auto_commit:
            return None
        paths = sorted(batch.writes)
        message = "\n".join([f"pan-memoryd: {len(batch.summaries)} wiki change(s)", "", *batch.summaries, "",
                             f"Batch: {batch_id}"])
        try:
            return self.git.commit(paths, message, author=DEFAULT_AUTHOR)
        except GitError as exc:
            logger.error("wiki commit failed, reverting batch %s: %s", batch_id, exc)
            for w in batch.writes.values():
                self._restore(w.path, w.before)
            error = f"commit failed: {exc}"
            for rec in batch.records:
                if rec.get("episode_id") in batch.episode_events and rec.get("outcome") == APPLIED:
                    rec["outcome"], rec["error"] = FAILED, error
            for event_ids in batch.episode_events.values():
                for eid in event_ids:
                    if eid in done:
                        done.remove(eid)
                    failed[eid] = error
            return None

    # -- log / dead / prune ------------------------------------------------------------------------

    def _record(self, batch_id: str, episode: Episode, cand: Optional[MemoryCandidate],
                curation: Optional[Curation], outcome: str, *, l1: Optional[List[Dict[str, Any]]] = None,
                error: Optional[str] = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "ts": _iso(self.clock()), "batch_id": batch_id, "worker": self.worker_id,
            "session_id": episode.session_id, "episode_id": episode.id, "closed_by": episode.closed_by,
            "event_ids": list(cand.event_ids) if cand else episode.event_ids,
            "classifier": getattr(self.classifier, "name", type(self.classifier).__name__),
            "curator": getattr(self.curator, "name", type(self.curator).__name__),
            "classification": to_jsonable(cand.classification) if cand else None,
            "candidate": None, "decision": None, "outcome": outcome, "l1": l1, "commit": None, "error": error,
        }
        if cand is not None:
            rec["candidate"] = {"id": cand.id, "title": cand.title, "claims": cand.normalized_claims,
                                "claim_evidence": cand.claim_evidence, "retrieval_query": cand.retrieval_query,
                                "tags": cand.tags, "rationale": cand.rationale}
        if curation is not None:
            rec["decision"] = to_jsonable(curation.decision)
            rec["decision"]["path"] = curation.path
        return rec

    def _l1_write_record(self, event: AgentEvent, batch_id: str) -> Dict[str, Any]:
        c = event.content
        return {"ts": _iso(self.clock()), "batch_id": batch_id, "worker": self.worker_id,
                "session_id": event.session_id, "episode_id": None, "event_ids": [event.id],
                "kind": "l1_write_observed", "outcome": LOGGED,
                "l1": [{"status": "agent_write", "target": c.get("target"), "action": c.get("action"),
                        "entry": c.get("content")}]}

    def _append_log(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        path = self.paths.curation_log
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    def _write_dead(self, event_id: str) -> None:
        row = self.spool.get(event_id)
        if row is None:
            return
        self.paths.dead.mkdir(parents=True, exist_ok=True)
        (self.paths.dead / f"{event_id}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
        logger.warning("event %s is dead after %s attempts: %s", event_id, row.get("attempts"), row.get("last_error"))

    def _maybe_prune(self) -> None:
        now = time.time()
        if now - self._last_prune < PRUNE_EVERY_SECONDS:
            return
        self._last_prune = now
        n = self.spool.prune(self.config.daemon.event_retention_days)
        if n:
            logger.info("pruned %d done events older than %d days", n, self.config.daemon.event_retention_days)


# -- replay (debugging / benchmarks) ---------------------------------------------------------------

def replay(paths: PanPaths, out_dir: Path, *, session_id: Optional[str] = None, limit: int = 100_000,
           wiki_seed: Optional[Path] = None, clock: Callable[[], float] = time.time) -> List[Dict[str, Any]]:
    """Re-run classification + curation over stored events (any status) in a scratch HERMES_HOME
    under ``out_dir``: copies of the wiki (or ``wiki_seed``), the Hermes config and memories. The
    real profile is never written. Returns the scratch curation-log records."""
    out_dir = Path(out_dir)
    scratch = PanPaths.for_home(out_dir / "hermes_home")
    if scratch.hermes_home.exists():
        shutil.rmtree(scratch.hermes_home)
    scratch.root.mkdir(parents=True)
    for name in ("config.yaml",):
        if (paths.hermes_home / name).is_file():
            shutil.copy2(paths.hermes_home / name, scratch.hermes_home / name)
    if paths.memories.is_dir():
        shutil.copytree(paths.memories, scratch.memories)
    if paths.config.is_file():
        shutil.copy2(paths.config, scratch.config)
    seed = Path(wiki_seed) if wiki_seed else paths.wiki
    if seed.is_dir():
        shutil.copytree(seed, scratch.wiki, ignore=shutil.ignore_patterns(".git"))
    rows: List[Dict[str, Any]] = []
    if paths.events_db.exists():
        with EventSpool(paths.events_db) as src:
            rows = src.list_rows(session_id=session_id, limit=limit)
    with EventSpool(scratch.events_db) as dst:
        for row in rows:
            fields = {k: row[k] for k in ("id", "ts", "session_id", "parent_session_id", "actor", "event_type",
                                          "content", "source_refs", "project", "metadata")}
            dst.append(AgentEvent.from_dict(fields))
    daemon = MemoryDaemon(scratch, load_config(scratch), clock=clock, worker_id="replay")
    try:
        daemon.prepare()
        daemon.drain(flush=True)
    finally:
        daemon.close()
    if not scratch.curation_log.exists():
        return []
    return [json.loads(line) for line in scratch.curation_log.read_text(encoding="utf-8").splitlines() if line]


# -- CLI --------------------------------------------------------------------------------------------

def resolve_hermes_home(arg: Optional[str]) -> Path:
    return Path(arg or os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()


def run_daemon(hermes_home: Path, *, once: bool = False, flush: bool = False, log_file: bool = True) -> int:
    paths = PanPaths.for_home(hermes_home)
    paths.ensure()
    handler: Optional[logging.Handler] = None
    if log_file:
        handler = logging.FileHandler(paths.root / LOG_NAME, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("pan").addHandler(handler)
    lock = DaemonLock(lock_path(paths))
    try:
        if not lock.acquire():
            print(f"pan-memoryd already running for {hermes_home} (pid {running_pid(lock.path)})", file=sys.stderr)
            return 1
        daemon = MemoryDaemon(paths)
        try:
            daemon.prepare()
            if once:
                reports = daemon.drain(flush=flush)
                total = Counter()
                for r in reports:
                    total.update(r.outcomes)
                print(f"pan-memoryd --once: {sum(r.claimed for r in reports)} claimed, "
                      f"{sum(r.done for r in reports)} done, {sum(r.failed for r in reports)} failed, "
                      f"{reports[-1].pending if reports else 0} pending; "
                      + (", ".join(f"{k}={v}" for k, v in sorted(total.items())) or "no decisions"))
                return 0
            stop = threading.Event()
            previous = {}
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, lambda *_: stop.set())
            try:
                daemon.run(stop)
            finally:
                for sig, old in previous.items():
                    signal.signal(sig, old)
            return 0
        finally:
            daemon.close()
    finally:
        lock.release()
        if handler is not None:
            logging.getLogger("pan").removeHandler(handler)
            handler.close()


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--once", action="store_true", help="process everything queued, then exit")
    parser.add_argument("--flush", action="store_true",
                        help="with --once: also close open episodes (do not wait for the idle timeout)")
    parser.add_argument("--hermes-home", default=None, help="profile directory (default: $HERMES_HOME or ~/.hermes)")
    parser.add_argument("-v", "--verbose", action="store_true")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pan-memoryd", description="PAN memory curation daemon")
    parser.add_argument("command", nargs="?", default="run", choices=["run"], help=argparse.SUPPRESS)
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_daemon(resolve_hermes_home(args.hermes_home), once=args.once, flush=args.flush)


if __name__ == "__main__":
    sys.exit(main())
