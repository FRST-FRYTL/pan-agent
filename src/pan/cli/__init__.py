"""`pan` command (integration spec §7): version, status, setup, uninstall, wiki, index, memoryd, memory, daemon,
doctor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pan import HERMES_PIN, __version__
from pan.paths import PanPaths

# Hermes config keys PAN needs (spec §5).
HERMES_SETTINGS = {
    "memory.provider": "pan",
    "memory.memory_enabled": "true",
    "memory.user_profile_enabled": "true",
    "memory.nudge_interval": "0",  # D1: PAN's curator owns automatic L1 writes
}


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _hermes_bin() -> str | None:
    local = Path(sys.executable).with_name("hermes")
    return str(local) if local.exists() else shutil.which("hermes")


def cmd_version(_: argparse.Namespace) -> int:
    from pan.hermes.compat import hermes_version
    print(f"pan-agent {__version__} (pinned hermes-agent {HERMES_PIN}, running {hermes_version()})")
    return 0


def _spool_summary(paths: PanPaths) -> str:
    if not paths.events_db.exists():
        return "missing"
    from pan.events.spool import EventSpool
    try:
        with EventSpool(paths.events_db) as spool:
            counts = spool.stats()
    except Exception as exc:  # locked / corrupt: status must still print
        return f"unreadable ({exc})"
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def _daemon_summary(paths: PanPaths) -> str:
    from pan.daemon.memoryd import lock_path, running_pid
    pid = running_pid(lock_path(paths))
    return "not running" if pid is None else f"running (pid {pid or '?'})"


def cmd_status(_: argparse.Namespace) -> int:
    paths = PanPaths.for_home(_hermes_home())
    log_lines = 0
    if paths.curation_log.exists():
        with open(paths.curation_log, "rb") as fh:
            log_lines = sum(1 for _ in fh)
    rows = [
        ("HERMES_HOME", str(paths.hermes_home)),
        ("PAN state", f"{paths.root} ({'exists' if paths.root.exists() else 'missing'})"),
        ("config", "exists" if paths.config.exists() else "missing (defaults)"),
        ("wiki", "exists" if paths.wiki.exists() else "missing"),
        ("event spool", _spool_summary(paths)),
        ("pan-memoryd", _daemon_summary(paths)),
        ("curation log", f"{log_lines} records" if paths.curation_log.exists() else "missing"),
        ("index", "exists" if paths.index_db.exists() else "missing"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    return cmd_version(_)


def cmd_setup(args: argparse.Namespace) -> int:
    paths = PanPaths.for_home(_hermes_home())
    hermes = _hermes_bin()
    if hermes is None:
        print("hermes CLI not found in this environment; install Hermes first.", file=sys.stderr)
        return 1
    from pan import install
    from pan.hermes.compat import hermes_version

    problem = install.tested_hermes_problem(hermes_version())
    if problem and not args.force:
        print(f"refusing: {problem}", file=sys.stderr)
        return 2
    if problem:
        print(f"warning (--force): {problem}", file=sys.stderr)
    if args.dry_run:
        print(f"would record the current values of {', '.join(HERMES_SETTINGS)} in {install.backup_path(paths)}")
    else:
        target, written = install.write_backup(paths, HERMES_SETTINGS)
        print(f"{'settings backup written' if written else 'settings backup kept'}: {target}")
    for key, value in HERMES_SETTINGS.items():
        cmd = [hermes, "config", "set", key, value]
        print(("would run: " if args.dry_run else "running: ") + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    if args.dry_run:
        if not paths.config.exists():
            print(f"would write the default PAN config to {paths.config}")
        print(f"would initialize the wiki at {paths.wiki} (if missing) and rebuild {paths.index_db}")
        return 0
    from pan.config import write_default_config

    paths.ensure()
    print(f"PAN state directory ready: {paths.root}")
    if write_default_config(paths):
        print(f"default config written: {paths.config}")
    rc = _setup_wiki_and_index(paths)
    print("run the curator with `pan memoryd run` (foreground) or `pan daemon install` (systemd user unit)")
    return rc


def _setup_wiki_and_index(paths: PanPaths) -> int:
    from pan.index.fts import FtsIndex
    from pan.wiki.store import WikiStore

    store = WikiStore(paths.wiki)
    if not store.exists():
        store.init()
        print(f"wiki initialized: {paths.wiki}")
    idx = FtsIndex(paths.index_db)
    try:
        n = idx.rebuild(paths.wiki)
    finally:
        idx.close()
    print(f"index built: {n} pages -> {paths.index_db}")
    return 0


# -- wiki / index ---------------------------------------------------------------------------------

def _paths() -> PanPaths:
    return PanPaths.for_home(_hermes_home())


def cmd_wiki_init(_: argparse.Namespace) -> int:
    from pan.wiki.store import WikiStore

    paths = _paths()
    created = WikiStore(paths.wiki).init()
    print(f"wiki {'initialized' if created else 'already initialized'}: {paths.wiki}")
    return 0


def cmd_wiki_search(args: argparse.Namespace) -> int:
    from pan.config import load_config
    from pan.memory.reader import Reader

    paths = _paths()
    reader = Reader(paths, config=load_config(paths).retrieval)
    try:
        result = reader.search(" ".join(args.query), type=args.type, limit=args.limit)
    finally:
        reader.close()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "error" not in result else 1
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    if not result["results"]:
        print(result.get("note", "no results"))
        return 0
    for hit in result["results"]:
        where = f" § {hit['section']}" if hit["section"] and hit["section"] != hit["title"] else ""
        print(f"{hit['score']:.2f}  {hit['id']}  ({hit['type']}, {hit['status']})  {hit['path']}")
        print(f"      {hit['title']}{where}")
        if hit["snippet"]:
            print(f"      {hit['snippet']}")
    return 0


def cmd_wiki_read(args: argparse.Namespace) -> int:
    from pan.memory.reader import Reader
    from pan.wiki.frontmatter import serialize

    reader = Reader(_paths())
    try:
        result = reader.read(args.page, section=args.section, raw=True)
    finally:
        reader.close()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif "error" in result:
        print(result["error"], file=sys.stderr)
        if result.get("sections"):
            print("sections: " + ", ".join(result["sections"]), file=sys.stderr)
    else:
        print(serialize(result["frontmatter"], "\n" + result["content"]), end="")
    return 1 if "error" in result else 0


def cmd_wiki_validate(_: argparse.Namespace) -> int:
    from pan.wiki.store import WikiStore

    store = WikiStore(_paths().wiki)
    issues = store.validate()
    for issue in issues:
        print(issue)
    print(f"{len(store.page_files())} pages, {len(issues)} issue(s)")
    return 1 if issues else 0


def cmd_index_rebuild(_: argparse.Namespace) -> int:
    from pan.config import load_config
    from pan.index.embed import EmbedError
    from pan.index.fts import FtsIndex
    from pan.index.vectors import sync_wiki
    from pan.wiki.store import WikiStore

    paths = _paths()
    if not WikiStore(paths.wiki).exists():
        print(f"no wiki at {paths.wiki} (run `pan wiki init`)", file=sys.stderr)
        return 1
    idx = FtsIndex(paths.index_db)
    try:
        n = idx.rebuild(paths.wiki)
    finally:
        idx.close()
    print(f"indexed {n} pages -> {paths.index_db}")
    try:
        stats = sync_wiki(paths, load_config(paths).retrieval, rebuild=True)
    except EmbedError as exc:
        print(f"vectors not rebuilt (retrieval sidecar unavailable: {exc}); FTS only until it is up",
              file=sys.stderr)
        return 0
    if stats is not None:
        print(f"embedded {stats['pages']} pages ({stats['embedded_units']} units) -> {paths.vectors_db}")
    return 0


def cmd_index_status(args: argparse.Namespace) -> int:
    from pan.index.fts import FtsIndex
    from pan.index.vectors import VectorIndex

    paths = _paths()
    idx = FtsIndex(paths.index_db, readonly=True)
    wiki = paths.wiki if paths.wiki.is_dir() else None
    try:
        info = idx.status(wiki)
    finally:
        idx.close()
    vec = VectorIndex(paths.vectors_db, readonly=True).status(wiki) if paths.vectors_db.exists() else {"exists": False}
    if args.json:
        print(json.dumps(dict(info, vectors=vec), indent=2))
    else:
        for key, value in list(info.items()) + [(f"vectors.{k}", v) for k, v in vec.items()]:
            if isinstance(value, list):
                value = ", ".join(value) or "-"
            print(f"{key:<15} {value}")
    return 0 if info.get("usable") else 1


# -- memoryd / memory / daemon --------------------------------------------------------------------

def cmd_memoryd_run(args: argparse.Namespace) -> int:
    import logging

    from pan.daemon.memoryd import resolve_hermes_home, run_daemon

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_daemon(resolve_hermes_home(args.hermes_home), once=args.once, flush=args.flush)


def _read_log(paths: PanPaths, session: str | None, limit: int) -> list[dict]:
    if not paths.curation_log.exists():
        return []
    records = []
    for line in paths.curation_log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if session is None or rec.get("session_id") == session:
            records.append(rec)
    return records[-limit:] if limit > 0 else records


def _decision_line(rec: dict) -> str:
    cls = rec.get("classification") or {}
    dec = rec.get("decision") or {}
    target = ", ".join(dec.get("target_pages") or []) or "-"
    what = f"{cls.get('type', rec.get('kind', '?'))}→{cls.get('destination', '-')}"
    action = dec.get("action", "-")
    extra = f"  error: {rec['error']}" if rec.get("error") else ""
    return f"{rec.get('ts', '?')}  {rec.get('session_id') or '-'}  {what:<28} {action:<7} {rec.get('outcome', '?'):<12} {target}{extra}"


def cmd_memory_inspect(args: argparse.Namespace) -> int:
    from pan.events.spool import EventSpool

    paths = _paths()
    rows = []
    if paths.events_db.exists():
        with EventSpool(paths.events_db) as spool:
            rows = spool.list_rows(session_id=args.session, status=args.status, limit=args.limit)
    records = _read_log(paths, args.session, args.limit)
    if args.json:
        print(json.dumps({"events": rows, "decisions": records}, indent=2, ensure_ascii=False, default=str))
        return 0
    print(f"events ({len(rows)}):")
    for r in rows:
        err = f"  [{r['last_error'][:80]}]" if r.get("last_error") else ""
        kind = r["content"].get("kind") if r["event_type"] == "session_end" else r["content"].get("tool")
        label = r["event_type"] + (f":{kind}" if kind else "")
        print(f"  {r['id']}  {r['ts']}  {r['session_id']:<20} {label:<22} {r['status']:<10} a={r['attempts']}{err}")
    print(f"decisions ({len(records)}):")
    for rec in records:
        print("  " + _decision_line(rec))
    return 0


def cmd_memory_replay(args: argparse.Namespace) -> int:
    import tempfile

    from pan.daemon.memoryd import replay

    paths = _paths()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pan-replay-"))
    records = replay(paths, out, session_id=args.session, limit=args.limit,
                     wiki_seed=Path(args.wiki) if args.wiki else None)
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False, default=str))
    else:
        for rec in records:
            print(_decision_line(rec))
        print(f"{len(records)} records; scratch profile: {out / 'hermes_home'}")
    return 0


def cmd_daemon_install(args: argparse.Namespace) -> int:
    from pan.daemon import service

    home = Path(args.hermes_home).expanduser() if args.hermes_home else _hermes_home()
    target, content, changed = service.install(home, unit_dir=Path(args.unit_dir) if args.unit_dir else None,
                                               dry_run=args.dry_run)
    if args.dry_run:
        print(f"would write {target}:\n")
        print(content)
        return 0
    print(f"{'wrote' if changed else 'unchanged'}: {target}")
    print("not enabled — to start it: systemctl --user daemon-reload && "
          f"systemctl --user enable --now {service.UNIT_NAME}")
    return 0


def cmd_daemon_uninstall(args: argparse.Namespace) -> int:
    from pan.daemon import service

    target, existed = service.uninstall(unit_dir=Path(args.unit_dir) if args.unit_dir else None,
                                        dry_run=args.dry_run)
    if not existed:
        print(f"not installed: {target}")
        return 0
    print(f"{'would remove' if args.dry_run else 'removed'}: {target}")
    print(f"if it was enabled: systemctl --user disable --now {service.UNIT_NAME} && systemctl --user daemon-reload")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    from pan import install

    hermes = _hermes_bin()
    if hermes is None:
        print("hermes CLI not found in this environment.", file=sys.stderr)
        return 1
    return install.uninstall(_paths(), hermes, dry_run=args.dry_run)


def cmd_doctor(args: argparse.Namespace) -> int:
    from pan import doctor

    return doctor.run_cli(args, _paths())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pan", description="PAN — Persistent Agent Nexus")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="show PAN and Hermes versions").set_defaults(func=cmd_version)
    sub.add_parser("status", help="show PAN state").set_defaults(func=cmd_status)
    p_setup = sub.add_parser("setup", help="configure Hermes to use PAN and create PAN state")
    p_setup.add_argument("--dry-run", action="store_true", help="print the changes without applying them")
    p_setup.add_argument("--force", action="store_true", help="continue on a Hermes version PAN was not tested with")
    p_setup.set_defaults(func=cmd_setup)
    p = sub.add_parser("uninstall", help="restore the Hermes settings `pan setup` changed (keeps PAN data)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_uninstall)

    p_wiki = sub.add_parser("wiki", help="knowledge wiki").add_subparsers(dest="wiki_command", required=True)
    p_wiki.add_parser("init", help="create the wiki skeleton + git repo").set_defaults(func=cmd_wiki_init)
    p = p_wiki.add_parser("search", help="search the wiki index")
    p.add_argument("query", nargs="+")
    p.add_argument("--type", default=None, help="page type filter (default: any)")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wiki_search)
    p = p_wiki.add_parser("read", help="print a page by id or path")
    p.add_argument("page")
    p.add_argument("--section", default=None, help="only this heading")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wiki_read)
    p_wiki.add_parser("validate", help="check schema, ids and links").set_defaults(func=cmd_wiki_validate)

    p_index = sub.add_parser("index", help="derived search index").add_subparsers(
        dest="index_command", required=True)
    p_index.add_parser("rebuild", help="rebuild index.db (and vectors.db) from the wiki").set_defaults(func=cmd_index_rebuild)
    p = p_index.add_parser("status", help="show index state and staleness")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index_status)

    from pan.daemon.memoryd import add_run_arguments

    p_memoryd = sub.add_parser("memoryd", help="memory curation daemon").add_subparsers(
        dest="memoryd_command", required=True)
    p = p_memoryd.add_parser("run", help="run pan-memoryd in the foreground")
    add_run_arguments(p)
    p.set_defaults(func=cmd_memoryd_run)

    p_memory = sub.add_parser("memory", help="events, classifications and curator decisions").add_subparsers(
        dest="memory_command", required=True)
    p = p_memory.add_parser("inspect", help="show spool events and curation-log records")
    p.add_argument("--session", default=None)
    p.add_argument("--status", default=None, choices=["new", "processing", "done", "dead"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_memory_inspect)
    p = p_memory.add_parser("replay", help="re-run classification/curation on stored events in a scratch profile")
    p.add_argument("--session", default=None)
    p.add_argument("--limit", type=int, default=100_000, help="most recent N events")
    p.add_argument("--out", default=None, help="scratch directory (default: a new temp dir)")
    p.add_argument("--wiki", default=None, help="seed wiki to start from (default: a copy of the current wiki)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_memory_replay)

    p_daemon = sub.add_parser("daemon", help="systemd user unit for pan-memoryd").add_subparsers(
        dest="daemon_command", required=True)
    for name, func, text in (("install", cmd_daemon_install, "write the unit (does not enable it)"),
                             ("uninstall", cmd_daemon_uninstall, "remove the unit file")):
        p = p_daemon.add_parser(name, help=text)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--unit-dir", default=None, help="default: ~/.config/systemd/user")
        if name == "install":
            p.add_argument("--hermes-home", default=None, help="profile the unit serves (default: $HERMES_HOME)")
        p.set_defaults(func=func)

    from pan.doctor import add_arguments as add_doctor_arguments

    p = sub.add_parser("doctor", help="check the model endpoint (tool calls, json_schema) for PAN")
    add_doctor_arguments(p)
    p.set_defaults(func=cmd_doctor)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
