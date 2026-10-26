"""Bit-exact CUDA regression golden for DynamicThresholdScheme.solve().

``record`` stores the thresholds, Pd and cost of modes "shared" and "robust"
on the cases of solve_suite.py (two seeds each) in the gitignored
``golden/solve_golden.json``; ``check`` reruns them and compares bit for bit.
Use it to show that a refactor leaves the schedules unchanged. The golden is
only valid for the nvcc, flags and GPU architecture that recorded it (its
provenance says which); across toolchains compare statistically instead
(solve_suite.py).

(The earlier manifest ``golden/solve_manifest.json`` held the research
workspace's E18/E20/E22 solvers. The guided beam changed the search, so it no
longer applies.)

    python golden_solve.py record
    python golden_solve.py check [--mode shared|robust]...
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys

import click
import numpy as np
from solve_suite import QUICK, make

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "golden" / "solve_golden.json"
SEEDS = (1, 2)


def run_case(mode: str, case: tuple[str, float, float], seed: int) -> dict:
    scheme = make(case, seed, "cuda", mode, 1)
    scheme.run(10)
    return scheme.solve()


def provenance() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {"git_head": head, "host": platform.node(), "backend": "cuda"}


@click.command()
@click.argument("cmd", type=click.Choice(["record", "check"]))
@click.option("--mode", "modes", multiple=True, default=("shared", "robust"))
def main(cmd: str, modes: tuple[str, ...]) -> None:
    if cmd == "record":
        cases = []
        for mode in ("shared", "robust"):
            for case in QUICK:
                for seed in SEEDS:
                    sol = run_case(mode, case, seed)
                    cases.append(
                        {
                            "mode": mode,
                            "case": list(case),
                            "seed": seed,
                            "thresholds": np.asarray(sol["thresholds"], np.float32)
                            .view(np.uint32)
                            .tolist(),
                            "pd": sol["pd"],
                            "c": sol["c"],
                        },
                    )
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"provenance": provenance(), "cases": cases}))
        click.echo(f"recorded {len(cases)} cases in {GOLDEN}")
        return
    golden = json.loads(GOLDEN.read_text())
    bad = n = 0
    for ref in golden["cases"]:
        if ref["mode"] not in modes:
            continue
        n += 1
        case = tuple(ref["case"])
        sol = run_case(ref["mode"], case, ref["seed"])
        got = np.asarray(sol["thresholds"], np.float32).view(np.uint32).tolist()
        tag = f"{ref['mode']:7s} {case[0]:6s} snr={case[1]:<5} pd={case[2]:<5} seed={ref['seed']}"
        if got == ref["thresholds"] and sol["pd"] == ref["pd"] and sol["c"] == ref["c"]:
            click.echo(f"ok    {tag}")
        else:
            bad += 1
            click.echo(f"DIFF  {tag}  pd={sol['pd']:.4f} (ref {ref['pd']:.4f})")
    click.echo("bit-identical" if bad == 0 else f"{bad} of {n} cases differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
