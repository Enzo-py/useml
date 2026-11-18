"""The `useml` command.

    useml init [--root DIR] [--yes]   set up this machine and this project (adds, never overwrites)
    useml check [--measure]           check that everything is in place
    useml paths                       where every cache goes (and where weights already are)
    useml hub ls                      what has been downloaded, how big, last used
    useml hub get <uri> [--extract]   download, print the local path
    useml hub status                  which layers of the network policy hold
    useml hub scan [--move [--apply]] caches left in the home; move them under the root
    useml hub watch [--pid N | -- <command>]   the external network monitor

    useml store info                  root, total size, budget, free disk
    useml store spaces                per space: entries, size, cost, last access
    useml store usage [--days N]      hits / misses / writes per day
    useml store check [--deep] [--repair]
    useml store gc [--max-size 450GB] [--older-than DAYS] [--space S] [--apply]
    useml store move <destination> [--delete-source]
    useml store unpin <owner>

    useml data ls                     named processed datasets, rows, table size
    useml data tables                 every table, its size, the names using it
    useml data info <name>            one processed dataset in detail
    useml data jobs                   detached jobs and their progress
    useml data follow <job>           a bar fed by a running job
    useml data rm <name> [--table]    forget a name (and delete its table if unused)

    useml config ls                   every config, by kind
    useml config show <name> [key=value ...]   the resolved YAML and its hash
    useml config describe <name>      what the file says, without building
    useml config find <hash>          the git versions that produce this hash

    useml exp ls                      every experience, its status, steps done
    useml exp show <name>             its steps: done, duration, output kind
    useml exp follow <name>           a bar over the steps, the log as it comes
    useml exp log <name> [-n 80]      the end of its log
    useml exp run <name>              run the steps not done yet (detached)
    useml exp stop <name> [--now]     stop after the current step (or kill now)
    useml exp why <name>              where it is; if it crashed, why (signal, OOM, trace)
    useml exp attach <name>           its tmux terminal
    useml exp stack <name>            the Python stack of a running run (is it stuck?)
    useml exp bundle <name> [-o DIR] [--data auto|rows|none] [--archive] [--checkpoints]
    useml exp verify <bundle>         check a bundle's files against its manifest
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys


def human(size) -> str:
    if size is None:
        return "?"
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _date(timestamp) -> str:
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _ls(args) -> int:
    from . import hub

    entries = hub.ls(refresh_sizes=args.refresh)
    if args.json:
        print(json.dumps(entries, indent=1))
        return 0
    if not entries:
        print("nothing downloaded through useml yet")
        return 0
    for entry in entries:
        flag = "  MISSING" if entry["missing"] else ""
        print(f"{human(entry['size']):>10}  {_date(entry['last_used'])}  {entry['uses']:>4}× "
              f" {entry['uri']}{flag}\n{'':>12}{entry['path']}")
    print(f"\n{human(sum(entry['size'] for entry in entries))} in {len(entries)} downloads")
    return 0


def _scan(args) -> int:
    from . import hub, paths

    found = hub.scan()
    if not found:
        print("no cache left in the home")
        return 0
    for entry in found:
        target = entry["target"] or "—"
        use = "  (in use)" if entry["in_use"] else ""
        note = f"  [{entry['note']}]" if entry["note"] else ""
        print(f"{human(entry['bytes']):>10}  {entry['path']}{use}{note}\n{'':>12}→ {target}")
    print(f"\n{human(sum(entry['bytes'] or 0 for entry in found))} in the home")
    if not args.move:
        return 0
    if paths.root() is None:
        print("no root configured: nothing to move to", file=sys.stderr)
        return 1
    for entry in found:
        if entry["target"] is None:
            continue
        report = hub.move(entry["path"], apply=args.apply)
        verb = "moved" if args.apply else "would move"
        print(f"{entry['path']}: {verb} {len(report['moved'])} entries to {report['target']}"
              + (f", {len(report['conflicts'])} conflicts left in place"
                 if report["conflicts"] else ""))
        for conflict in report["conflicts"][:10]:
            print(f"    conflict: {conflict}")
    if not args.apply:
        print("\ndry run: add --apply to move")
    return 0


def _store(args) -> int:
    from . import store as stores

    store = stores.open(args.root) if args.root else stores.open()
    command = args.store_command
    if command == "info":
        total, limit = store.total_bytes(), stores.budget()
        print(f"store:   {store.root}")
        print(f"size:    {human(total)}"
              + (f" of {human(limit)} budget ({100 * total / limit:.0f} %)" if limit else
                 "   (no budget: [store] max_size in the config)"))
        print(f"free:    {human(stores.free_disk_bytes(store.root))} on its disk")
        print(f"spaces:  {len(store.declared())} declared")
        return 0
    if command == "spaces":
        rows = store.spaces()
        if not rows:
            print("empty store, no space declared")
        for row in rows:
            last = _date(row["last_access"]) if row["last_access"] else "—"
            managed = f"  (+managed {human(row['managed_bytes'])})" \
                if row["managed_bytes"] else ""
            print(f"{human(row['bytes']):>10}  {row['cost'] or '?':<9} {row['entries']:>8} "
                  f"entries  {row['pinned']:>5} pinned  {last}  {row['space']}{managed}")
        return 0
    if command == "usage":
        for row in store.usage(args.days):
            print(f"{row['day']}  {row['space']:<36} hits {row['hits']:>8}  "
                  f"misses {row['misses']:>8}  writes {row['writes']:>8}")
        return 0
    if command == "check":
        report = stores.check(store, deep=args.deep)
        for name, items in report.items():
            print(f"  {name:<12} {len(items)}")
        if args.repair:
            print(stores.repair(store, report))
        return 0
    if command == "gc":
        limit = stores.parse_size(args.max_size) if args.max_size else (
            stores.budget() if args.older_than is None else None)
        if limit is None and args.older_than is None:
            print("nothing to do: give --max-size or --older-than, or set [store] max_size",
                  file=sys.stderr)
            return 1
        plan = stores.garbage_plan(store, spaces=args.space or None,
                                   older_than_days=args.older_than, max_bytes=limit,
                                   include_pinned=args.include_pinned,
                                   include_precious=args.include_precious)
        size = sum(entry["size"] for entry in plan)
        by_space: dict = {}
        for entry in plan:
            by_space[entry["space"]] = by_space.get(entry["space"], 0) + entry["size"]
        for space, freed in sorted(by_space.items(), key=lambda item: -item[1]):
            print(f"{human(freed):>10}  {space}")
        if limit is not None and store.total_bytes() - size > limit:
            print(f"⚠ still over budget after this plan: pinned, precious or managed data "
                  f"holds {human(store.total_bytes() - size)}")
        if not args.apply:
            print(f"dry run: {len(plan)} entries, {human(size)}; add --apply to delete")
            return 0
        print(f"freed {human(stores.collect_garbage(store, plan))}")
        return 0
    if command == "move":
        print(stores.move(store, args.destination, delete_source=args.delete_source))
        print(f"set it for the next runs: export {stores.ROOT_VARIABLE}="
              f"{args.destination}  (or [store] root in the config)")
        return 0
    if command == "unpin":
        print(f"{store.unpin_owner(args.owner)} pins removed")
        return 0
    return 1


def _data(args) -> int:
    from . import data

    command = args.data_command
    if command == "ls":
        entries = data.ls()
        if not entries:
            print(f"no processed dataset yet (root: {data.data_root()})")
        for entry in entries:
            print(f"{human(entry['table_bytes']):>10}  {entry['rows']:>10} rows  "
                  f"{_date(entry['last_used'])}  {entry['name']:<24} {entry['table']}")
        return 0
    if command == "tables":
        for table in data.tables():
            names = ", ".join(table["names"]) or "(unused)"
            outputs = ", ".join(f"{name} {tuple(spec['shape'])} {spec['stored']}"
                                for name, spec in table["outputs"].items())
            print(f"{human(table['bytes']):>10}  {table['table']:<40} {outputs}  ← {names}")
        return 0
    if command == "info":
        entry = data.record(args.name)
        view = data.load(args.name)
        print(json.dumps(entry, indent=1, default=str))
        print(view)
        print(f"missing rows: {view.missing}   in memory: {human(view.nbytes())}")
        return 0
    if command == "jobs":
        for job in data.jobs():
            state = job.state
            progress = f"{state.get('done', 0)}/{state['total']}" if state.get("total") else ""
            print(f"{state['status']:<8} {progress:>16}  {job.id}  {state.get('phase', '')}")
        return 0
    if command == "follow":
        status = data.job(args.job).follow()
        return 0 if status in ("done", "running") else 1
    if command == "rm":
        freed = data.rm(args.name, table=args.table)
        print(f"removed {args.name}" + (f", freed {human(freed)}" if freed else ""))
        return 0
    return 1


def _config(args) -> int:
    from . import config

    if args.config_command == "ls":
        library = config.library()
        print(f"library: {library.root}")
        for kind, names in library.ls().items():
            print(f"  {kind} ({len(names)}): {', '.join(names)}")
        return 0
    if args.config_command == "show":
        overrides = dict(config.parse_override(text) for text in args.overrides)
        loaded = config.load_raw(args.name, args.kind, overrides=overrides)
        print(f"# {loaded.cited_name}  cfg:{loaded.hash}  ({loaded.kind})")
        print(loaded.to_yaml(), end="")
        return 0
    if args.config_command == "describe":
        print(config.describe(args.name, args.kind))
        return 0
    if args.config_command == "find":
        found = config.find(args.hash)
        if not found:
            print("no committed version produces this hash")
        for entry in found:
            print(f"{entry['commit']}  {entry['date']}  {entry['path']}")
        return 0
    return 1


def _exp(args) -> int:
    from . import exp

    command = args.exp_command
    if command == "ls":
        experiences = exp.ls()
        if not experiences:
            print(f"no experience under {exp.root()}")
        for entry in experiences:
            steps = entry.steps()
            done = sum(step["done"] for step in steps)
            print(f"{entry.status:<8} {done:>3}/{len(steps):<3} {entry.name}")
        return 0
    if command == "verify":
        problems = exp.verify(args.name)
        print("\n".join(problems) if problems else "intact: every file matches the manifest")
        return 1 if problems else 0
    if command == "bundle":
        exp.bundle(args.name, args.output, data=args.data, archive=args.archive,
                   checkpoints=args.checkpoints, partial=args.partial)
        return 0
    entry = exp.open(args.name)
    if command == "why":
        print(entry.why())
        return 0
    if command == "attach":
        entry.attach()
        return 0
    if command == "stack":
        print(entry.stack())
        return 0
    if command == "show":
        print(entry)
        for step in entry.steps():
            seconds = f"{step['seconds']:.0f}s" if step["seconds"] else ""
            mark = "✓" if step["done"] else "·"
            print(f"  {mark} {step['name']:<24} {seconds:>8}  {step['output'] or '':<14} "
                  f"{step['function']}")
        error = entry.state.get("error")
        if error and entry.status != "done":
            print(f"\n{error}")
        return 0
    if command == "follow":
        entry.follow()
        return 0
    if command == "log":
        print(entry.log(args.n))
        return 0
    if command == "run":
        entry.run(follow=not args.no_follow)
        return 0
    if command == "stop":
        entry.stop(now=args.now)
        return 0
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="useml", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("paths", help="where every cache goes")
    init_parser = commands.add_parser("init", help="set up this machine and this project")
    init_parser.add_argument("project", nargs="?", default=None)
    init_parser.add_argument("--root", help="the cache root of this machine (a big disk)")
    init_parser.add_argument("--store", help="an existing cache store to use in place")
    init_parser.add_argument("--yes", action="store_true", help="write without asking")
    init_parser.add_argument("--no-machine", action="store_true")
    init_parser.add_argument("--no-project", action="store_true")
    check_parser = commands.add_parser("check", help="check that everything is in place")
    check_parser.add_argument("--measure", action="store_true",
                               help="also measure the caches left in the home (slow)")

    hub_parser = commands.add_parser("hub", help="downloads")
    hub_commands = hub_parser.add_subparsers(dest="hub_command", required=True)
    ls_parser = hub_commands.add_parser("ls", help="recorded downloads")
    ls_parser.add_argument("--refresh", action="store_true", help="measure sizes again")
    ls_parser.add_argument("--json", action="store_true")
    get_parser = hub_commands.add_parser("get", help="download and print the path")
    get_parser.add_argument("uri")
    get_parser.add_argument("--dest")
    get_parser.add_argument("--extract", action="store_true")
    get_parser.add_argument("--refresh", action="store_true")
    hub_commands.add_parser("status", help="which layers of the network policy hold")
    scan_parser = hub_commands.add_parser("scan", help="caches left in the home")
    scan_parser.add_argument("--move", action="store_true", help="plan the move")
    scan_parser.add_argument("--apply", action="store_true", help="really move")
    watch_parser = hub_commands.add_parser("watch", help="external network monitor")
    watch_parser.add_argument("--pid", type=int)
    watch_parser.add_argument("--max-connections", type=int, default=3)
    watch_parser.add_argument("--interval", type=float, default=1.0)
    watch_parser.add_argument("--sustained", type=int, default=3)
    watch_parser.add_argument("--no-kill", action="store_true")
    watch_parser.add_argument("--log")
    watch_parser.add_argument("cmd", nargs=argparse.REMAINDER)

    store_parser = commands.add_parser("store", help="the key → value cache")
    store_parser.add_argument("--root", help="another store than the default one")
    store_commands = store_parser.add_subparsers(dest="store_command", required=True)
    store_commands.add_parser("info")
    store_commands.add_parser("spaces")
    usage_parser = store_commands.add_parser("usage")
    usage_parser.add_argument("--days", type=int, default=30)
    check_parser = store_commands.add_parser("check")
    check_parser.add_argument("--deep", action="store_true")
    check_parser.add_argument("--repair", action="store_true")
    gc_parser = store_commands.add_parser("gc")
    gc_parser.add_argument("--max-size")
    gc_parser.add_argument("--older-than", type=float, metavar="DAYS")
    gc_parser.add_argument("--space", action="append")
    gc_parser.add_argument("--include-pinned", action="store_true")
    gc_parser.add_argument("--include-precious", action="store_true")
    gc_parser.add_argument("--apply", action="store_true")
    move_parser = store_commands.add_parser("move")
    move_parser.add_argument("destination")
    move_parser.add_argument("--delete-source", action="store_true")
    unpin_parser = store_commands.add_parser("unpin")
    unpin_parser.add_argument("owner")

    data_parser = commands.add_parser("data", help="processed datasets")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("ls")
    data_commands.add_parser("tables")
    info_parser = data_commands.add_parser("info")
    info_parser.add_argument("name")
    data_commands.add_parser("jobs")
    follow_parser = data_commands.add_parser("follow")
    follow_parser.add_argument("job")
    rm_parser = data_commands.add_parser("rm")
    rm_parser.add_argument("name")
    rm_parser.add_argument("--table", action="store_true")

    config_parser = commands.add_parser("config", help="YAML configs")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("ls")
    show_parser = config_commands.add_parser("show")
    show_parser.add_argument("name")
    show_parser.add_argument("overrides", nargs="*", help="key.sub=value")
    show_parser.add_argument("--kind")
    describe_parser = config_commands.add_parser("describe")
    describe_parser.add_argument("name")
    describe_parser.add_argument("--kind")
    find_parser = config_commands.add_parser("find")
    find_parser.add_argument("hash")

    exp_parser = commands.add_parser("exp", help="experiences")
    exp_commands = exp_parser.add_subparsers(dest="exp_command", required=True)
    exp_commands.add_parser("ls")
    for command in ("show", "follow", "log", "run", "stop", "why", "attach", "stack",
                    "bundle", "verify"):
        sub = exp_commands.add_parser(command)
        sub.add_argument("name")
        if command == "bundle":
            sub.add_argument("-o", "--output")
            sub.add_argument("--data", default="auto", choices=("auto", "rows", "none"))
            sub.add_argument("--archive", action="store_true")
            sub.add_argument("--checkpoints", action="store_true")
            sub.add_argument("--partial", action="store_true")
        if command == "log":
            sub.add_argument("-n", type=int, default=40)
        if command == "run":
            sub.add_argument("--no-follow", action="store_true")
        if command == "stop":
            sub.add_argument("--now", action="store_true")

    args = parser.parse_args(argv)
    from . import hub, paths

    if args.command == "init":
        from ._setup import init

        init(args.project, root=args.root, store=args.store, yes=args.yes, machine=not args.no_machine,
             project_setup=not args.no_project)
        return 0
    if args.command == "check":
        from ._setup import check

        checks = check(measure=args.measure)
        for level, message in checks:
            print(f"{level} {message}")
        return 1 if any(level == "✗" for level, _ in checks) else 0
    if args.command == "exp":
        return _exp(args)
    if args.command == "config":
        return _config(args)
    if args.command == "data":
        return _data(args)
    if args.command == "store":
        return _store(args)
    if args.command == "paths":
        print(paths.diagnose())
        return 0
    if args.hub_command == "ls":
        return _ls(args)
    if args.hub_command == "get":
        print(hub.get(args.uri, dest=args.dest, extract_archive=args.extract,
                      refresh=args.refresh))
        return 0
    if args.hub_command == "status":
        for name, value in hub.status().items():
            print(f"  {name:<24} {value}")
        return 0
    if args.hub_command == "scan":
        return _scan(args)
    if args.hub_command == "watch":
        command = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
        return hub.watch(command or None, pid=args.pid, max_connections=args.max_connections,
                         interval=args.interval, sustained=args.sustained,
                         kill=not args.no_kill, log_path=args.log)
    return 1


if __name__ == "__main__":
    sys.exit(main())
