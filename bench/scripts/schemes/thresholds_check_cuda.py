"""Production driver for DynamicThresholdScheme (CUDA by default).

Runs one configuration in one or more modes and re-scores every returned
schedule on the same fresh samples, so the modes can be compared directly.

    python thresholds_check_cuda.py --mode improved,shared --min-pd 0.1

Per mode:
  search     run() then get_best_path_thresholds(min_pd): what a live search
             of legacy or improved uses. Its cost and Pd are the search's own
             Monte Carlo estimates and are optimistic.
  solve      mode shared only: the schedule of solve() (fresh-sample
             selection, refinement, repair), with the Pd and cost it measured
             on samples that chose nothing.
  re-scored  evaluate() of the schedule on one fresh seed shared by all modes
             (--report-trials, --report-seed). L0 = C / Pd with C the scored
             nodes (the historical cost); L1 adds the survivors of the last
             stage (w = 1), which is what mode shared optimises.

The modes use different grids by design: shared anchors the Pd grid at
--min-pd (--nprobs bins from min_pd to 1) and needs the window beam, while
legacy and improved use --nprobs bins from --prob-min to 1.
"""

from __future__ import annotations

import time

import click
import numpy as np
from _branching import BRANCHING_PROD

from loki import libloki

BP = BRANCHING_PROD[:127]


def terminal(
    states: np.ndarray, nstages: int, nthr: int, nprobs: int, min_pd: float
) -> dict | None:
    grid = states.reshape(nstages, nthr, nprobs)[-1]
    mask = ~grid["is_empty"] & (grid["success_h1_cumul"] >= min_pd)
    if not np.any(mask):
        return None
    costs = np.where(mask, grid["cost"], np.inf)
    return grid.ravel()[int(np.argmin(costs))]


def rescore(scheme, path: np.ndarray, trials: int, seed: int) -> dict:  # noqa: ANN001
    last = scheme.evaluate(path, trials, seed)[-1]
    if bool(last["is_empty"]):
        return {"pd": 0.0, "c0": float("nan"), "c1": float("nan")}
    pd = float(last["success_h1_cumul"])
    c0 = float(last["complexity_cumul"])
    return {"pd": pd, "c0": c0, "c1": c0 + float(last["complexity"])}


def run_mode(mode: str, o: dict) -> dict:
    kw = {
        "ref_ducy": o["ref_ducy"],
        "nbins": o["nbins"],
        "ntrials": o["ntrials"],
        "nprobs": o["nprobs"] if o["nprobs"] else (40 if mode == "shared" else 30),
        "prob_min": o["prob_min"],
        "snr_final": o["snr"],
        "nthresholds": o["nthresholds"],
        "ducy_max": o["ducy_max"],
        "beam_width": o["beam_width"],
        "wtsp": o["wtsp"],
        "mode": mode,
        "guess": "window" if mode == "shared" else o["guess"],
    }
    extra = {"nthreads": o["nthreads"]} if o["backend"] == "cpu" else {}
    if mode == "shared":
        extra |= {"min_pd": o["min_pd"], "w_terminal": o["w"], "refine": o["refine"]}
    scheme = libloki.thresholds.DynamicThresholdScheme(
        BP, seed=o["seed"], backend=o["backend"], batch_size=256, **kw, **extra
    )
    t0 = time.perf_counter()
    scheme.run(thres_neigh=o["thres_neigh"])
    t_run = time.perf_counter() - t0
    out = {"mode": mode, "t_run": t_run}

    path = np.asarray(scheme.get_best_path_thresholds(o["min_pd"]), dtype=np.float32)
    cell = terminal(
        scheme.get_states(), len(BP), kw["nthresholds"], scheme.nprobs, o["min_pd"]
    )
    out["search"] = (
        None
        if cell is None or path.size != len(BP)
        else {
            "L": float(cell["cost"]),
            "pd": float(cell["success_h1_cumul"]),
            "c": float(cell["complexity_cumul"]),
        }
    )
    final = path
    if mode == "shared":
        t0 = time.perf_counter()
        sol = scheme.solve()
        out["t_solve"] = time.perf_counter() - t0
        out["solve"] = {
            k: sol[k] for k in ("pd", "pd_se", "c", "cost", "delivery", "refine")
        }
        out["diag"] = sol["diagnostics"]
        final = np.asarray(sol["thresholds"], dtype=np.float32)
    if o["save"]:
        out["saved"] = scheme.save(outdir=o["outdir"])
    if final.size == len(BP):
        out["searched_path"] = (
            rescore(scheme, path, o["report_trials"], o["report_seed"])
            if path.size == len(BP)
            else None
        )
        out["final"] = rescore(scheme, final, o["report_trials"], o["report_seed"])
    return out


def show(out: dict, o: dict) -> None:
    print(f"\n== mode {out['mode']} ==")
    t = f"run {out['t_run']:.2f}s" + (
        f", solve {out['t_solve']:.2f}s" if "t_solve" in out else ""
    )
    print(f"  time: {t}")
    s = out["search"]
    print(
        "  search (in-run, optimistic): "
        + (
            "no path met min_pd"
            if s is None
            else f"L {s['L']:.6g}  Pd {s['pd']:.4f}  C {s['c']:.6g}"
        )
    )
    if "solve" in out:
        v = out["solve"]
        print(
            f"  solve ({v['refine']}): Pd {v['pd']:.4f} +/- {v['pd_se']:.4f}  C(w=1) {v['c']:.6g}"
            f"  L1 {v['cost']:.6g}  delivery {v['delivery']}"
        )
        keys = (
            "initial_shift",
            "tightened_stages",
            "repair_shift",
            "shortlist_selected",
            "block_moves",
            "search_missed_target",
        )
        print(
            "    diagnostics: "
            + ", ".join(f"{k}={out['diag'][k]:.4g}" for k in keys if k in out["diag"])
        )
    if "saved" in out:
        print(f"  saved {out['saved']}")
    for key, label in (
        ("searched_path", "re-scored search path"),
        ("final", "re-scored returned schedule"),
    ):
        r = out.get(key)
        if r:
            print(
                f"  {label} ({o['report_trials']} trials, seed {o['report_seed']}): "
                f"Pd {r['pd']:.4f}  L0 {r['c0'] / r['pd']:.6g}  L1 {r['c1'] / r['pd']:.6g}"
            )


@click.command()
@click.option(
    "--mode",
    "modes",
    default="improved",
    show_default=True,
    help="legacy, improved, shared, or a comma-separated list to compare",
)
@click.option("--backend", default="cuda", show_default=True)
@click.option("--min-pd", default=0.1, show_default=True)
@click.option(
    "--refine",
    default="full",
    type=click.Choice(["none", "shortlist", "full"]),
    show_default=True,
    help="shared only",
)
@click.option(
    "--w",
    default=1.0,
    show_default=True,
    help="shared only: weight of the last stage's survivors",
)
@click.option("--seed", default=1, show_default=True)
@click.option("--ntrials", default=1024, show_default=True)
@click.option("--nprobs", default=0, help="0: 40 for shared, 30 otherwise")
@click.option(
    "--prob-min", default=0.05, show_default=True, help="legacy/improved grid"
)
@click.option(
    "--guess",
    default="analytic",
    type=click.Choice(["analytic", "pilot", "window"]),
    show_default=True,
    help="legacy/improved beam (shared always uses window)",
)
@click.option("--beam-width", default=2.5, show_default=True)
@click.option("--thres-neigh", default=11, show_default=True)
@click.option("--nbins", default=64, show_default=True)
@click.option("--nthresholds", default=100, show_default=True)
@click.option("--snr", default=10.0, show_default=True)
@click.option("--ref-ducy", default=0.1, show_default=True)
@click.option("--ducy-max", default=0.3, show_default=True)
@click.option("--wtsp", default=1.2, show_default=True)
@click.option("--nthreads", default=16, show_default=True, help="CPU backend")
@click.option("--report-trials", default=1 << 16, show_default=True)
@click.option("--report-seed", default=10_001, show_default=True)
@click.option("--save/--no-save", default=True, show_default=True)
@click.option("--outdir", default="scheme_results/", show_default=True)
def main(modes: str, **o) -> None:  # noqa: ANN003
    print(
        f"stages {len(BP)}  backend {o['backend']}  min_pd {o['min_pd']}  seed {o['seed']}"
        f"  search ntrials {o['ntrials']}  nbins {o['nbins']}"
    )
    results = []
    for mode in modes.split(","):
        out = run_mode(mode.strip(), o)
        show(out, o)
        results.append(out)
    if len(results) > 1:
        print("\ncomparison (re-scored returned schedules, same fresh samples)")
        print(f"  {'mode':9s} {'Pd':>7s} {'L0':>12s} {'L1':>12s} {'time(s)':>8s}")
        for r in results:
            f = r.get("final")
            if f:
                print(
                    f"  {r['mode']:9s} {f['pd']:7.4f} {f['c0'] / f['pd']:12.6g}"
                    f" {f['c1'] / f['pd']:12.6g} {r['t_run'] + r.get('t_solve', 0.0):8.2f}"
                )
            else:
                print(f"  {r['mode']:9s} no schedule")


if __name__ == "__main__":
    main()
