"""The ``loki-survey`` command line."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from importlib import resources
from pathlib import Path

from loki_survey.collect import IncompleteRunError
from loki_survey.config import (
    ConfigError,
    Resolved,
    load_resolved,
    resolve,
    resolved_dict,
)
from loki_survey.executors.batch import SchedulerError, SubmitError
from loki_survey.rundir import DriverBusyError, RunDir, atomic_write_json, read_json


def cmd_new(args: argparse.Namespace) -> int:
    run = RunDir.at(args.rundir)
    run.root.mkdir(parents=True, exist_ok=True)
    for name in ("survey.toml", "site.toml"):
        target = run.root / name
        if target.exists():
            print(f"{target} exists; leaving it")
            continue
        text = resources.files("loki_survey.templates").joinpath(name).read_text()
        target.write_text(text)
        print(f"wrote {target}")
    print("Edit both files, then run: loki-survey check", run.root)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from loki_survey.plan import candidate_options  # noqa: PLC0415
    from loki_survey.unit import versions  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = resolve(run)
    buffer_gb = candidate_options().memory_gb
    if resolved.site.loki_memory_gb <= buffer_gb:
        msg = (
            f"{run.site_toml} [unit]: memory_gb - overhead_gb leaves loki "
            f"{resolved.site.loki_memory_gb:g} GB, which must be above its "
            f"{buffer_gb:g} GB candidate buffer; raise memory_gb or lower overhead_gb"
        )
        raise ConfigError(msg)
    if run.resolved_json.is_file():
        old = read_json(run.resolved_json)
        if old["science_sha256"] != resolved.science_sha256 and any(
            run.out_dir.glob("*.json"),
        ):
            print(
                "survey.toml changed after units finished; "
                "start a new run directory instead",
                file=sys.stderr,
            )
            return 1
    run.ensure()
    atomic_write_json(run.resolved_json, resolved_dict(resolved, versions()))
    survey = resolved.survey
    print(
        f"survey '{survey.name}': science {resolved.science_sha256[:12]}, "
        f"analysis {resolved.analysis_sha256[:12]}",
    )
    print(f"  {len(survey.regions)} regions, {len(survey.boxes)} boxes")
    print(f"  wrote {run.resolved_json}")
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    from loki_survey.prep import run_prep  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    run_prep(run, resolved, nthreads=args.threads or resolved.site.max_threads)
    return 0


def _manifests(run: RunDir) -> list[dict]:
    paths = sorted(run.prep_dir.glob("*/manifest.json"))
    if not paths:
        msg = "no prepared series; run 'loki-survey prep' first"
        raise FileNotFoundError(msg)
    return [read_json(p) for p in paths]


def cmd_plan(args: argparse.Namespace) -> int:
    from loki_survey.plan import run_plan  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    run_plan(run, resolved, _manifests(run))
    return 0


def _selected_units(run: RunDir, wanted: list[str] | None) -> list:
    from loki_survey.plan import load_units  # noqa: PLC0415

    units = load_units(run)
    if wanted:
        unknown = sorted(set(wanted) - {u.id for u in units})
        if unknown:
            msg = f"unknown units: {unknown}"
            raise ConfigError(msg)
        units = [u for u in units if u.id in set(wanted)]
    return units


def cmd_submit(args: argparse.Namespace) -> int:
    from loki_survey.executors import batch_executor  # noqa: PLC0415
    from loki_survey.executors.local import LocalExecutor  # noqa: PLC0415
    from loki_survey.status import StatusDB  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    run.ensure()
    units = _selected_units(run, args.units)
    db = StatusDB(run)
    try:
        with run.driving():
            db.sync(units, resolved.site.unit_memory_gb)
            if resolved.site.executor == "local":
                db.reset_running()
                LocalExecutor(run, resolved, db).submit(units)
            else:
                executor = batch_executor(run, resolved, db)
                executor.reconcile()
                if not getattr(args, "watch_only", False):
                    retried = db.retry_failed([u.id for u in units])
                    if retried:
                        print(f"retrying {retried} failed units")
                    print(f"submitted {executor.submit(units)} units")
                if getattr(args, "wait", False):
                    executor.watch(units)
                else:
                    print(f"follow them with: loki-survey watch {run.root}")
    except KeyboardInterrupt:
        if resolved.site.executor == "local":
            raise
        print(
            f"\nstopped watching; the jobs keep running: loki-survey watch {run.root}",
        )
        return 130
    finally:
        db.close()
    return _report(run, resolved)


def cmd_watch(args: argparse.Namespace) -> int:
    """Poll and resubmit; unlike submit, leave failed units failed."""
    args.wait = True
    args.watch_only = True
    return cmd_submit(args)


def cmd_cancel(args: argparse.Namespace) -> int:
    from loki_survey.executors import batch_executor  # noqa: PLC0415
    from loki_survey.status import StatusDB  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    if resolved.site.executor == "local":
        print("local units stop when their 'submit' is interrupted (Ctrl-C)")
        return 0
    db = StatusDB(run)
    try:
        with run.driving():
            count = batch_executor(run, resolved, db).cancel()
    finally:
        db.close()
    print(f"cancelled {count} attempts; 'submit' runs their units again")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from loki_survey.executors import batch_executor  # noqa: PLC0415
    from loki_survey.status import StatusDB  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    if resolved.site.executor != "local":
        db = StatusDB(run)
        try:
            with run.driving():
                db.sync(_selected_units(run, None), resolved.site.unit_memory_gb)
                batch_executor(run, resolved, db).reconcile()
        except DriverBusyError:
            print("a watcher is driving this run; showing its latest state")
        finally:
            db.close()
    return _report(run, resolved)


def _report(run: RunDir, resolved: Resolved) -> int:
    """Print unit counts, failures and excluded hosts; 1 if any unit failed."""
    from loki_survey.status import StatusDB  # noqa: PLC0415

    db = StatusDB(run)
    try:
        db.sync(_selected_units(run, None), resolved.site.unit_memory_gb)
        counts = db.counts()
        split = counts.pop("split", 0)
        total = sum(counts.values())
        print(
            "  ".join(
                f"{s} {counts.get(s, 0)}"
                for s in ("done", "running", "queued", "pending", "failed")
            )
            + f"  of {total}",
        )
        if split:
            print(f"{split} units hit their time limit and run as half-band parts")
        for uid, failure, attempts, detail in db.failures():
            print(f"FAILED {uid}: {failure} after {attempts} attempts")
            if detail:
                print("    " + detail.replace("\n", "\n    "))
        excluded = db.excluded_hosts()
        if excluded:
            print(f"excluded hosts: {', '.join(excluded)}")
        quarantined = list(run.quarantine_dir.glob("*"))
        if quarantined:
            print(
                f"{len(quarantined)} temporary directories of failed attempts "
                f"are in {run.quarantine_dir}",
            )
        return 0 if counts.get("failed", 0) == 0 else 1
    finally:
        db.close()


def cmd_collect(args: argparse.Namespace) -> int:
    from loki_survey.collect import collect  # noqa: PLC0415
    from loki_survey.plan import load_units  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    summary = collect(run, resolved, load_units(run), partial=args.partial)
    removed = sum(summary["removed"].values())
    print(
        f"collect: {summary['members']} unit candidates from "
        f"{summary['units_collected']} of {summary['units']} units in "
        f"{summary['groups']} groups; {summary['selected']} selected, {removed} "
        f"removed (ledger {run.collect_dir / 'ledger.jsonl'})",
    )
    if summary["units_missing"]:
        print(f"  {len(summary['units_missing'])} units had not finished")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from loki_survey.evaluate import evaluate  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    summary = evaluate(run, load_resolved(run))
    catalogue = summary.get("catalogue")
    if catalogue:
        pulsars = catalogue["pulsars"]
        found = sum(p["found"] for p in pulsars)
        in_band = sum(p["in_band"] for p in pulsars)
        print(f"evaluate: {found} of {in_band} catalogued pulsars in band found")
        for p in pulsars:
            best = p["best"]
            if best is not None and not best["selected"]:
                print(f"  {p['name']}: its group was removed by {best['removed_by']}")
            elif p["in_band"] and not p["found"]:
                print(f"  {p['name']}: not found at its own frequency")
    lines = summary.get("lines")
    if lines:
        print(
            f"evaluate: {lines['n_lines']} lines in {lines['series']}; "
            f"{lines['groups_on_lines']} groups lie on one",
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from loki_survey.plan import load_units  # noqa: PLC0415
    from loki_survey.report import REPORT_HTML, write_report  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    write_report(run, load_resolved(run), load_units(run))
    print(f"report: {run.report_dir / REPORT_HTML}")
    return 0


def cmd_run_unit(args: argparse.Namespace) -> int:
    run = RunDir.at(args.rundir)
    # Written before loki is imported: a job that dies before this file
    # exists points at its node, not at the unit.
    atomic_write_json(
        run.unit_started(args.unit, args.attempt),
        {
            "host": platform.node().split(".")[0],
            "pid": os.getpid(),
            "time": time.time(),
            "lsf_job": os.environ.get("LSB_JOBID"),
            "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
        },
    )
    from loki_survey.unit import run_unit  # noqa: PLC0415

    summary = run_unit(run, args.unit, args.attempt)
    found, seconds = summary["n_candidates"], summary["total_s"]
    print(f"{args.unit}: {found} candidates in {seconds:.1f} s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    for step in (cmd_check, cmd_prep, cmd_plan, cmd_submit):
        code = step(args)
        if code:
            return code
    if not _all_done(RunDir.at(args.rundir)):
        print("units are still running; collect, evaluate and report when they end")
        return 0
    args.partial = False
    for step in (cmd_collect, cmd_evaluate, cmd_report):
        code = step(args)
        if code:
            return code
    return 0


def _all_done(run: RunDir) -> bool:
    from loki_survey.plan import load_units  # noqa: PLC0415

    return all(run.unit_summary(u.id).is_file() for u in load_units(run))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loki-survey", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:  # noqa: ANN001
        p = sub.add_parser(name, help=help_text)
        p.add_argument("rundir", type=Path, help="the run directory")
        p.set_defaults(func=func)
        return p

    add("new", cmd_new, "write template survey.toml and site.toml")
    add("check", cmd_check, "validate the config and write resolved.json")
    p = add("prep", cmd_prep, "read, normalise, mask and downsample each series")
    p.add_argument("--threads", type=int, default=None)
    add("plan", cmd_plan, "write the unit list and build EP plans")
    p = add("submit", cmd_submit, "run every unit that is not done")
    p.add_argument("--units", nargs="*", help="only these unit ids")
    p.add_argument(
        "--wait",
        action="store_true",
        help="on a cluster, keep polling and resubmitting until all units end",
    )
    p = add("watch", cmd_watch, "poll a cluster run and resubmit until all units end")
    p.add_argument("--units", nargs="*", help="only these unit ids")
    add("cancel", cmd_cancel, "cancel a cluster run's queued and running jobs")
    add("status", cmd_status, "show unit states and failures")
    p = add("collect", cmd_collect, "group every unit's candidates; write the ledger")
    p.add_argument(
        "--partial",
        action="store_true",
        help="collect although some units have not finished",
    )
    add("evaluate", cmd_evaluate, "match candidates to the catalogue and to lines")
    add("report", cmd_report, "write report/index.html")
    p = add(
        "run",
        cmd_run,
        "check, prep, plan and submit; then collect, evaluate and report",
    )
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--units", nargs="*", help="only these unit ids")
    p.add_argument("--wait", action="store_true", help="as for submit")
    p = add("run-unit", cmd_run_unit, "run one unit (used by executors)")
    p.add_argument("unit")
    p.add_argument("--attempt", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    # A watcher's output often goes to a file; show each line as it happens.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        FileNotFoundError,
        PermissionError,
        DriverBusyError,
        SchedulerError,
        SubmitError,
        IncompleteRunError,
    ) as exc:
        print(f"loki-survey: {exc}", file=sys.stderr)
        return 2
