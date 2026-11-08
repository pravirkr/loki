"""The ``loki-survey`` command line."""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from loki_survey.config import ConfigError, load_resolved, resolve, resolved_dict
from loki_survey.rundir import RunDir, atomic_write_json, read_json


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
    from loki_survey.unit import versions  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = resolve(run)
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
    print(f"survey '{survey.name}': science {resolved.science_sha256[:12]}")
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


def cmd_submit(args: argparse.Namespace) -> int:
    from loki_survey.executors.local import LocalExecutor  # noqa: PLC0415
    from loki_survey.plan import load_units  # noqa: PLC0415
    from loki_survey.status import StatusDB  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    run.ensure()
    units = load_units(run)
    if args.units:
        wanted = set(args.units)
        units = [u for u in units if u.id in wanted]
    db = StatusDB(run)
    try:
        db.reset_running()
        db.sync(units, resolved.site.unit_memory_gb)
        LocalExecutor(run, resolved, db).submit(units)
    finally:
        db.close()
    return cmd_status(args)


def cmd_status(args: argparse.Namespace) -> int:
    from loki_survey.plan import load_units  # noqa: PLC0415
    from loki_survey.status import StatusDB  # noqa: PLC0415

    run = RunDir.at(args.rundir)
    resolved = load_resolved(run)
    db = StatusDB(run)
    try:
        db.sync(load_units(run), resolved.site.unit_memory_gb)
        counts = db.counts()
        total = sum(counts.values())
        print(
            "  ".join(
                f"{s} {counts.get(s, 0)}"
                for s in ("done", "running", "pending", "failed")
            )
            + f"  of {total}",
        )
        for uid, failure, attempts, detail in db.failures():
            print(f"FAILED {uid}: {failure} after {attempts} attempts")
            if detail:
                print("    " + detail.replace("\n", "\n    "))
        return 0 if counts.get("failed", 0) == 0 else 1
    finally:
        db.close()


def cmd_run_unit(args: argparse.Namespace) -> int:
    from loki_survey.unit import run_unit  # noqa: PLC0415

    summary = run_unit(RunDir.at(args.rundir), args.unit, args.attempt)
    found, seconds = summary["n_candidates"], summary["total_s"]
    print(f"{args.unit}: {found} candidates in {seconds:.1f} s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    for step in (cmd_check, cmd_prep, cmd_plan, cmd_submit):
        code = step(args)
        if code:
            return code
    return 0


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
    add("status", cmd_status, "show unit states and failures")
    p = add("run", cmd_run, "check, prep, plan and submit")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--units", nargs="*", help="only these unit ids")
    p = add("run-unit", cmd_run_unit, "run one unit (used by executors)")
    p.add_argument("unit")
    p.add_argument("--attempt", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, PermissionError) as exc:
        print(f"loki-survey: {exc}", file=sys.stderr)
        return 2
