"""Reduced acceptance suite for DynamicThresholdScheme.solve() (mode "shared").

A smaller version of the research workspace's quick tier: the production
pattern and the issue #15 patterns at several S/N, targets and solver seeds.
Every returned schedule is judged on samples the solver never saw: Pd by a
direct count of fresh unpruned signal trials (2^18, Wilson-free: the binomial
error is reported), cost C by an evaluation on fresh noise (2^19 trials).

Gates per case (the workspace's):
  - every seed returns a schedule;
  - every seed's judged Pd >= 0.90 pd_min and the median >= 0.98 pd_min;
  - every seed's reported Pd within 10 % of the judged one.
The judged cost per detection L = C / Pd is printed per case (geometric mean
over seeds); compare backends or refine levels with --refine and --backend.

    PYTHONPATH=src python solve_suite.py --backend cuda --refine full
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import time

import click
import numpy as np
from _branching import BRANCHING_ISSUE15, BRANCHING_PROD

from loki import libloki

lt = libloki.thresholds

FAMILIES = {
    "prod": (BRANCHING_PROD[:127], {"ref_ducy": 0.1, "ducy_max": 0.3, "wtsp": 1.2}),
    **{
        f"seg{n}": (
            BRANCHING_ISSUE15[n],
            {"ref_ducy": 0.1, "ducy_max": 0.2, "wtsp": 1.5},
        )
        for n in (4, 8, 16, 32)
    },
}
# (family, snr, pd_min)
QUICK = [
    ("prod", 10.0, 0.1),
    ("prod", 10.0, 0.05),
    ("seg8", 8.1, 0.5),
    ("seg32", 6.0, 0.1),
    ("seg16", 8.1, 0.1),
    ("seg4", 10.0, 0.9),
]
JUDGE_H1 = 1 << 18
JUDGE_H0 = 1 << 19


def judge_seed(case: tuple[str, float, float], what: str) -> int:
    key = f"{case}-{what}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") >> 1


def make(case, seed, backend, refine, nthreads, nbins=64):
    fam, snr, pd_min = case
    bp, kw = FAMILIES[fam]
    return lt.DynamicThresholdScheme(
        np.asarray(bp, dtype=np.float32),
        kw["ref_ducy"],
        nbins=nbins,
        ntrials=1024,
        nprobs=40,
        snr_final=snr,
        nthresholds=100,
        ducy_max=kw["ducy_max"],
        wtsp=kw["wtsp"],
        mode="shared",
        seed=seed,
        backend=backend,
        nthreads=nthreads,
        guess="window",
        min_pd=pd_min,
        w_terminal=1.0,
        refine=refine,
    )


def judge(case, thresholds, backend, nthreads):
    """Pd and cost of a schedule on fresh samples (a scheme only to simulate)."""
    probe = make(case, 1, backend, "none", nthreads)
    scores = probe.simulate_h1_scores(JUDGE_H1, judge_seed(case, "h1"))
    alive = np.all(scores > np.asarray(thresholds, dtype=np.float32)[:, None], axis=0)
    pd = float(alive.mean())
    states = probe.evaluate(thresholds, JUDGE_H0, judge_seed(case, "h0"))
    last = states[-1]
    c = float(last["complexity_cumul"]) + float(last["complexity"])
    return pd, c, c / pd if pd > 0 else float("inf")


@click.command()
@click.option("--backend", default="cuda")
@click.option(
    "--refine",
    default="full",
    type=click.Choice(["none", "shortlist", "full"]),
)
@click.option("--seeds", default=8)
@click.option("--nthreads", default=16)
@click.option(
    "--cases",
    default="all",
    help="'all' or comma-separated indices of QUICK",
)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=None)
def main(backend, refine, seeds, nthreads, cases, out):
    sel = QUICK if cases == "all" else [QUICK[int(i)] for i in cases.split(",")]
    rows, failures = [], 0
    for case in sel:
        fam, snr, pd_min = case
        pds, ls, claims, times = [], [], [], []
        ok = True
        for seed in range(1, seeds + 1):
            t0 = time.perf_counter()
            scheme = make(case, seed, backend, refine, nthreads)
            scheme.run(10)
            sol = scheme.solve()
            times.append(time.perf_counter() - t0)
            if len(sol["thresholds"]) == 0:
                ok = False
                continue
            pd, _c, ell = judge(case, sol["thresholds"], backend, nthreads)
            pds.append(pd)
            ls.append(ell)
            claims.append(sol["pd"])
            if pd < 0.9 * pd_min or abs(sol["pd"] - pd) > 0.1 * pd:
                ok = False
        if len(pds) < seeds or statistics.median(pds) < 0.98 * pd_min:
            ok = False
        failures += 0 if ok else 1
        geo = float(np.exp(np.mean(np.log(ls)))) if ls else float("nan")
        row = {
            "family": fam,
            "snr": snr,
            "pd_min": pd_min,
            "backend": backend,
            "refine": refine,
            "pass": ok,
            "pd_min_judged": min(pds) if pds else None,
            "pd_median_judged": statistics.median(pds) if pds else None,
            "max_claim_error": max(abs(c - p) / p for c, p in zip(claims, pds))
            if pds
            else None,
            "L_geomean": geo,
            "t_median_s": statistics.median(times),
        }
        rows.append(row)
        click.echo(
            f"{'ok  ' if ok else 'FAIL'} {fam:6s} snr={snr:<5} pd={pd_min:<5} "
            f"pd judged min/med={row['pd_min_judged']:.3f}/{row['pd_median_judged']:.3f} "
            f"claim err={row['max_claim_error']:.3f}  L={geo:.4g}  t={row['t_median_s']:.2f}s",
        )
    click.echo(f"{len(rows)} cases, {failures} failing")
    if out is not None:
        out.write_text(json.dumps(rows, indent=1))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
