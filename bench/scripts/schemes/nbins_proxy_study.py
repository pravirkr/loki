"""Does a threshold schedule found at a small nbins hold at the true nbins?

The EP planner searches the schedule at ``nbins_sim = min(nbins, max(64,
4 / ref_ducy))`` (the memory of the search grows with nbins) and evaluates
chunk populations at the true nbins. The score is a maximum over boxcar
templates, so the noise tail depends on nbins and wtsp, the signal much less.
This study measures what that costs: for each (pattern, ref_ducy, wtsp, true
nbins) it solves

  native: at the true nbins, and
  proxy:  at nbins_sim,

then rescores both schedules at the true nbins with fresh samples (Pd on
2^18 signal trials, cost from an evaluation on 2^17 noise trials).

Reported per case: Pd of each schedule at the true nbins, L = C / Pd of each,
and L(proxy) / L(native). Gate: L ratio <= 1.10 and proxy Pd >= 0.9 pd_min.

    PYTHONPATH=src python nbins_proxy_study.py --out study.json
"""

from __future__ import annotations

import json
import math
import pathlib

import click
import numpy as np
from _branching import BRANCHING_ISSUE15, BRANCHING_PROD

from loki import libloki

lt = libloki.thresholds

NSIM_MIN = 64
MIN_PULSE_BINS = 4.0


def nbins_sim(nbins: int, ref_ducy: float) -> int:
    resolved = math.ceil(MIN_PULSE_BINS / ref_ducy)
    sim = (max(NSIM_MIN, resolved) + 3) // 4 * 4
    return min(nbins, sim)


def make(bp, nbins, ref_ducy, ducy_max, wtsp, snr, pd_min, backend, mode, seed):
    return lt.DynamicThresholdScheme(
        np.asarray(bp, dtype=np.float32),
        ref_ducy,
        nbins=nbins,
        ntrials=1024,
        nprobs=40,
        snr_final=snr,
        nthresholds=100,
        ducy_max=ducy_max,
        wtsp=wtsp,
        mode=mode,
        seed=seed,
        backend=backend,
        nthreads=16,
        guess="window",
        min_pd=pd_min,
        w_terminal=1.0,
    )


def solve(bp, nbins, ref_ducy, ducy_max, wtsp, snr, pd_min, backend, mode, seed):
    s = make(bp, nbins, ref_ducy, ducy_max, wtsp, snr, pd_min, backend, mode, seed)
    s.run(10)
    return s.solve()


def rescore(bp, thresholds, nbins, ref_ducy, ducy_max, wtsp, snr, pd_min, backend):
    """Pd and L of a schedule at the true nbins, on fresh samples."""
    s = make(bp, nbins, ref_ducy, ducy_max, wtsp, snr, pd_min, backend, "none", 1)
    pd_states = s.evaluate(thresholds, 1 << 18, 0xA11CE)
    pd = float(pd_states[-1]["success_h1_cumul"])
    cs = s.evaluate(thresholds, 1 << 17, 0xB0B)
    last = cs[-1]
    c = float(last["complexity_cumul"]) + float(last["complexity"])
    return pd, c, c / pd


PATTERNS = {
    "prod": (BRANCHING_PROD[:127], [64, 128, 256]),
    "seg8": (BRANCHING_ISSUE15[8], [64, 128, 256, 512, 1024]),
    "seg32": (BRANCHING_ISSUE15[32], [64, 128, 256, 512, 1024]),
}


@click.command()
@click.option("--backend", default="cuda")
@click.option("--mode", default="robust")
@click.option("--patterns", default="seg8,seg32,prod")
@click.option("--ref-ducy", "ref_ducys", default="0.05,0.1,0.2")
@click.option("--wtsp", "wtsps", default="1.0,1.2")
@click.option("--pd-min", "pd_mins", default="0.1,0.5")
@click.option("--snr", default=8.0)
@click.option("--seed", default=1)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=None)
def main(backend, mode, patterns, ref_ducys, wtsps, pd_mins, snr, seed, out):
    rows = []
    for name in patterns.split(","):
        bp, nbins_list = PATTERNS[name]
        for ref_ducy in map(float, ref_ducys.split(",")):
            for wtsp in map(float, wtsps.split(",")):
                for pd_min in map(float, pd_mins.split(",")):
                    for nbins in nbins_list:
                        sim = nbins_sim(nbins, ref_ducy)
                        if sim == nbins:
                            continue  # the proxy is the native search
                        ducy_max = 0.2
                        try:
                            nat = solve(
                                bp,
                                nbins,
                                ref_ducy,
                                ducy_max,
                                wtsp,
                                snr,
                                pd_min,
                                backend,
                                mode,
                                seed,
                            )
                            prx = solve(
                                bp,
                                sim,
                                ref_ducy,
                                ducy_max,
                                wtsp,
                                snr,
                                pd_min,
                                backend,
                                mode,
                                seed,
                            )
                            pd_n, c_n, l_n = rescore(
                                bp,
                                nat["thresholds"],
                                nbins,
                                ref_ducy,
                                ducy_max,
                                wtsp,
                                snr,
                                pd_min,
                                backend,
                            )
                            pd_p, c_p, l_p = rescore(
                                bp,
                                prx["thresholds"],
                                nbins,
                                ref_ducy,
                                ducy_max,
                                wtsp,
                                snr,
                                pd_min,
                                backend,
                            )
                        except Exception as exc:
                            click.echo(f"SKIP {name} nbins={nbins}: {exc}")
                            continue
                        row = {
                            "pattern": name,
                            "ref_ducy": ref_ducy,
                            "wtsp": wtsp,
                            "pd_min": pd_min,
                            "nbins": nbins,
                            "nbins_sim": sim,
                            "pd_native": pd_n,
                            "pd_proxy": pd_p,
                            "c_native": c_n,
                            "c_proxy": c_p,
                            "L_native": l_n,
                            "L_proxy": l_p,
                            "L_ratio": l_p / l_n,
                            "pass": bool(l_p / l_n <= 1.10 and pd_p >= 0.9 * pd_min),
                        }
                        rows.append(row)
                        click.echo(
                            f"{name:6s} ducy={ref_ducy:<5} wtsp={wtsp:<4} pd={pd_min:<4} "
                            f"nbins={nbins:<5} sim={sim:<3} pd nat/prx="
                            f"{pd_n:.3f}/{pd_p:.3f}  L ratio={row['L_ratio']:.3f} "
                            f"{'ok' if row['pass'] else 'FAIL'}",
                        )
    if out is not None:
        out.write_text(json.dumps(rows, indent=1))
    n_fail = sum(not r["pass"] for r in rows)
    click.echo(f"{len(rows)} cases, {n_fail} outside the gate")


if __name__ == "__main__":
    main()
