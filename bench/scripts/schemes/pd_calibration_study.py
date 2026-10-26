"""Study why the selected path delivers less than ``min_pd``.

Subcommands (all CUDA by default; pass ``--backend cpu`` for a check):

``cells``
    Re-score every non-empty terminal cell of a search, not only the selected
    one. If the in-run estimator is sound, unselected cells are unbiased on
    average and only the selected cell is low (selection, the winner's curse).
``ntrials``
    Bias and spread of the selected path against the search trial count.
``modes``
    Legacy against improved mode at the same trial count.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
from _branching import BRANCHING_ISSUE15, BRANCHING_PROD, ISSUE15_KW, PROD_KW

from loki import libloki

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_OFFSET = 10_000


def config(name: str) -> tuple[np.ndarray, dict, int]:
    """Branching pattern, scheme kwargs and thres_neigh of a named config."""
    if name == "prod":
        return BRANCHING_PROD[:127], dict(PROD_KW), 11
    if name == "planner":
        # Production pattern with EPRegionPlanner's own scheme settings
        kw = dict(PROD_KW)
        kw.update(nprobs=10, beam_width=0.7, wtsp=1.2)
        return BRANCHING_PROD[:127], kw, 10
    if name.startswith("seg"):
        # seg<nseg>[-<snr>], e.g. seg8 or seg16-6: the issue #15 patterns
        nseg, _, snr = name[3:].partition("-")
        kw = dict(ISSUE15_KW)
        if snr:
            kw["snr_final"] = float(snr)
        return BRANCHING_ISSUE15[int(nseg)], kw, 10
    msg = f"unknown config {name}"
    raise click.BadParameter(msg)


def make_scheme(
    bp: np.ndarray,
    kw: dict,
    *,
    seed: int,
    mode: str,
    backend: str,
    ntrials: int | None = None,
    guess: str = "analytic",
) -> libloki.thresholds.DynamicThresholdScheme:
    args = dict(kw)
    if ntrials is not None:
        args["ntrials"] = ntrials
    extra = {"nthreads": 8} if backend == "cpu" else {"batch_size": 256}
    return libloki.thresholds.DynamicThresholdScheme(
        bp,
        mode=mode,
        seed=seed,
        backend=backend,
        guess=guess,
        **extra,
        **args,
    )


def lower_bin(probs: np.ndarray, value: float) -> int:
    """utils::find_lower_bin_index: -1 below probs[0], last bin at the top."""
    idx = int(np.searchsorted(probs, value, side="right")) - 1
    return min(idx, probs.size - 1)


def backtrack(
    grid: np.ndarray,
    thresholds: np.ndarray,
    probs: np.ndarray,
    ithres: int,
    iprob: int,
) -> np.ndarray:
    """Path of terminal cell (ithres, iprob), as get_best_path_thresholds."""
    nstages = grid.shape[0]
    state = grid[-1, ithres, iprob]
    path = [float(state["threshold"])]
    for istage in range(nstages - 2, -1, -1):
        jthr = int(np.argmin(np.abs(thresholds - state["threshold_prev"])))
        jprob = lower_bin(probs, float(state["success_h1_cumul_prev"]))
        state = grid[istage, jthr, jprob]
        if bool(state["is_empty"]):
            msg = f"backtracking hit an empty state at stage {istage}"
            raise RuntimeError(msg)
        path.append(float(state["threshold"]))
    return np.asarray(path[::-1], dtype=np.float32)


def rescore(
    scheme: libloki.thresholds.DynamicThresholdScheme,
    path: np.ndarray,
    ntrials: int,
    seed: int,
) -> tuple[float, float]:
    """(P_d, L) of a fixed path on fresh trials; NaN when the path dies."""
    last = scheme.evaluate(path, ntrials, seed)[-1]
    if bool(last["is_empty"]):
        return float("nan"), float("nan")
    return float(last["success_h1_cumul"]), float(last["cost"])


def search(
    bp: np.ndarray,
    kw: dict,
    neigh: int,
    *,
    seed: int,
    mode: str,
    backend: str,
    ntrials: int | None = None,
    guess: str = "analytic",
) -> tuple[libloki.thresholds.DynamicThresholdScheme, float]:
    scheme = make_scheme(
        bp,
        kw,
        seed=seed,
        mode=mode,
        backend=backend,
        ntrials=ntrials,
        guess=guess,
    )
    t0 = time.perf_counter()
    scheme.run(neigh)
    return scheme, time.perf_counter() - t0


def selected(
    scheme: libloki.thresholds.DynamicThresholdScheme,
    min_pd: float,
) -> tuple[np.ndarray, float, float] | None:
    """Return the selected path with its in-run (P_d, L), or None."""
    path = np.asarray(scheme.get_best_path_thresholds(min_pd), np.float32)
    if path.size == 0:
        return None
    grid = scheme.get_states().reshape(
        scheme.nstages,
        scheme.nthresholds,
        scheme.nprobs,
    )[-1]
    mask = ~grid["is_empty"] & (grid["success_h1_cumul"] >= min_pd)
    cell = grid.ravel()[int(np.argmin(np.where(mask, grid["cost"], np.inf)))]
    return path, float(cell["success_h1_cumul"]), float(cell["cost"])


def stats(values: list[float] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "no finite samples"
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = std / np.sqrt(arr.size)
    return f"mean {arr.mean():+.5f} ± {sem:.5f}  std {std:.5f}  n={arr.size}"


@click.group()
@click.option("--config", "cfg", default="prod", show_default=True)
@click.option("--backend", default="cuda", show_default=True)
@click.option("--min-pd", default=0.1, show_default=True)
@click.option("--eval-trials", default=16384, show_default=True)
@click.option("--seed-start", default=100, show_default=True)
@click.option("--num-seeds", default=16, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=RESULTS / "pd_calibration",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, **opts: object) -> None:
    ctx.obj = opts


@cli.command()
@click.option("--mode", default="improved", show_default=True)
@click.option(
    "--pd-floor",
    default=0.05,
    show_default=True,
    help="re-score only cells with in-run P_d at or above this",
)
@click.pass_obj
def cells(opts: dict, mode: str, pd_floor: float) -> None:
    """E1: in-run against re-scored P_d for every terminal cell."""
    bp, kw, neigh = config(opts["cfg"])
    seeds = range(opts["seed_start"], opts["seed_start"] + opts["num_seeds"])
    rows = []
    for seed in seeds:
        scheme, _ = search(
            bp,
            kw,
            neigh,
            seed=seed,
            mode=mode,
            backend=opts["backend"],
        )
        states = scheme.get_states().reshape(
            scheme.nstages,
            scheme.nthresholds,
            scheme.nprobs,
        )
        thresholds = np.asarray(scheme.thresholds)
        probs = np.asarray(scheme.probs)
        last = states[-1]
        sel = selected(scheme, opts["min_pd"])
        sel_path = None if sel is None else sel[0]
        live = np.argwhere(~last["is_empty"] & (last["success_h1_cumul"] >= pd_floor))
        for ithr, iprob in live:
            cell = last[ithr, iprob]
            path = backtrack(states, thresholds, probs, int(ithr), int(iprob))
            re_pd, re_cost = rescore(
                scheme,
                path,
                opts["eval_trials"],
                seed + EVAL_OFFSET,
            )
            is_sel = sel_path is not None and np.array_equal(path, sel_path)
            rows.append(
                (
                    seed,
                    float(cell["success_h1_cumul"]),
                    re_pd,
                    float(cell["cost"]),
                    re_cost,
                    is_sel,
                ),
            )
        n_seed = sum(r[0] == seed for r in rows)
        print(f"seed {seed}: {n_seed} cells re-scored", flush=True)
    arr = np.array(
        rows,
        dtype=[
            ("seed", "i8"),
            ("in_pd", "f8"),
            ("re_pd", "f8"),
            ("in_cost", "f8"),
            ("re_cost", "f8"),
            ("selected", "?"),
        ],
    )
    out = opts["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"cells_{opts['cfg']}_{mode}_{opts['backend']}.npy", arr)
    sigma = np.sqrt(arr["re_pd"] * (1 - arr["re_pd"]) / opts["eval_trials"])
    diff = arr["re_pd"] - arr["in_pd"]
    sel = arr["selected"]
    ratio = arr["re_cost"] / arr["in_cost"]
    print(f"=== {opts['cfg']} {mode} {opts['backend']}: re-scored - in-run P_d ===")
    print(f"all cells        {stats(diff)}")
    print(f"unselected       {stats(diff[~sel])}")
    print(f"selected         {stats(diff[sel])}")
    print(f"relative, all    {stats(diff / arr['in_pd'])}")
    print(f"eval sigma (mean) {np.nanmean(sigma):.5f}")
    for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)):
        m = (arr["in_pd"] >= lo) & (arr["in_pd"] < hi)
        print(f"in-run P_d [{lo:.2f},{hi:.2f})  {stats(diff[m])}")
    print(f"re-scored / in-run L, all {stats(ratio)}")
    print(f"re-scored / in-run L, sel {stats(ratio[sel])}")


@cli.command()
@click.option("--mode", default="improved", show_default=True)
@click.option("--trials", default="512,1024,2048,4096,8192", show_default=True)
@click.pass_obj
def ntrials(opts: dict, mode: str, trials: str) -> None:
    """E2: selected-path bias and spread against the search trial count."""
    bp, kw, neigh = config(opts["cfg"])
    seeds = range(opts["seed_start"], opts["seed_start"] + opts["num_seeds"])
    for nt in (int(t) for t in trials.split(",")):
        in_pd, re_pd, ratio_l, secs = [], [], [], []
        for seed in seeds:
            scheme, dt = search(
                bp,
                kw,
                neigh,
                seed=seed,
                mode=mode,
                backend=opts["backend"],
                ntrials=nt,
            )
            secs.append(dt)
            sel = selected(scheme, opts["min_pd"])
            if sel is None:
                continue
            path, pd_in, cost_in = sel
            pd_re, cost_re = rescore(
                scheme,
                path,
                opts["eval_trials"],
                seed + EVAL_OFFSET,
            )
            in_pd.append(pd_in)
            re_pd.append(pd_re)
            ratio_l.append(cost_re / cost_in)
        re = np.asarray(re_pd)
        print(
            f"=== ntrials {nt} ({mode}, {opts['backend']}), "
            f"run {np.median(secs):.2f} s ==="
        )
        print(f"in-run P_d     {stats(in_pd)}")
        print(f"re-scored P_d  {stats(re_pd)}")
        print(f"bias (re-in)   {stats(re - np.asarray(in_pd))}")
        print(
            f"feasible       {np.mean(re >= opts['min_pd']):.3f}   min {re.min():.4f}"
        )
        print(f"L re/in        {stats(ratio_l)}", flush=True)


@cli.command()
@click.pass_obj
def modes(opts: dict) -> None:
    """E3: legacy against improved, selected path."""
    bp, kw, neigh = config(opts["cfg"])
    seeds = range(opts["seed_start"], opts["seed_start"] + opts["num_seeds"])
    for mode in ("legacy", "improved"):
        in_pd, re_pd, re_cost, secs = [], [], [], []
        for seed in seeds:
            scheme, dt = search(
                bp,
                kw,
                neigh,
                seed=seed,
                mode=mode,
                backend=opts["backend"],
            )
            secs.append(dt)
            sel = selected(scheme, opts["min_pd"])
            if sel is None:
                continue
            path, pd_in, _ = sel
            pd_re, cost_re = rescore(
                scheme,
                path,
                opts["eval_trials"],
                seed + EVAL_OFFSET,
            )
            in_pd.append(pd_in)
            re_pd.append(pd_re)
            re_cost.append(cost_re)
        re = np.asarray(re_pd)
        print(f"=== {mode} ({opts['backend']}), run {np.median(secs):.2f} s ===")
        print(f"in-run P_d     {stats(in_pd)}")
        print(f"re-scored P_d  {stats(re_pd)}")
        print(f"bias (re-in)   {stats(re - np.asarray(in_pd))}")
        print(f"feasible       {np.mean(re >= opts['min_pd']):.3f}")
        print(f"re-scored L    {stats(re_cost)}", flush=True)


@cli.command()
@click.option("--mode", default="improved", show_default=True)
@click.option("--ks", default="1,4,8", show_default=True, help="ncandidates")
@click.option("--ntrials-cal", default=1 << 15, show_default=True)
@click.option("--ntrials-cost", default=4096, show_default=True)
@click.option("--check-trials", default=1 << 16, show_default=True)
@click.option("--guess", default="analytic", show_default=True)
@click.pass_obj
def calib(
    opts: dict,
    guess: str,
    mode: str,
    ks: str,
    ntrials_cal: int,
    ntrials_cost: int,
    check_trials: int,
) -> None:
    """E4: delivered P_d and cost of calibrated paths (independent rescore)."""
    bp, kw, neigh = config(opts["cfg"])
    seeds = range(opts["seed_start"], opts["seed_start"] + opts["num_seeds"])
    klist = [int(k) for k in ks.split(",")]
    rows: dict[str, list[tuple[float, float, float]]] = {"uncal": []}
    rows.update({f"K={k}": [] for k in klist})
    for seed in seeds:
        scheme, t_run = search(
            bp,
            kw,
            neigh,
            seed=seed,
            mode=mode,
            backend=opts["backend"],
            guess=guess,
        )
        check_seed = seed + 2 * EVAL_OFFSET
        sel = selected(scheme, opts["min_pd"])
        if sel is not None:
            pd, cost = rescore(scheme, sel[0], check_trials, check_seed)
            rows["uncal"].append((pd, cost, t_run))
        for k in klist:
            t0 = time.perf_counter()
            res = scheme.calibrate(
                opts["min_pd"],
                ncandidates=k,
                ntrials_cal=ntrials_cal,
                ntrials_cost=ntrials_cost,
                seed=seed + EVAL_OFFSET,
            )
            dt = time.perf_counter() - t0
            if res["thresholds"].size == 0:
                continue
            pd, cost = rescore(scheme, res["thresholds"], check_trials, check_seed)
            rows[f"K={k}"].append((pd, cost, dt))
        print(f"seed {seed} done", flush=True)
    sigma = np.sqrt(opts["min_pd"] * (1 - opts["min_pd"]) / check_trials)
    print(
        f"=== {opts['cfg']} {mode} {opts['backend']} guess {guess}: "
        "independent rescore "
        f"({check_trials} trials, sigma {sigma:.4f}); ntrials_cal {ntrials_cal}",
    )
    for name, vals in rows.items():
        if not vals:
            print(f"{name:6s} no paths")
            continue
        arr = np.asarray(vals)
        pd = arr[:, 0]
        print(
            f"{name:6s} P_d {stats(pd)}  "
            f"feasible {np.mean(pd >= opts['min_pd'] - 2 * sigma):.2f}"
            f"  min {pd.min():.4f}",
        )
        print(f"       L   {stats(arr[:, 1])}   time {np.median(arr[:, 2]):.3f} s")


@cli.command()
@click.option("--modes", default="improved,shared", show_default=True)
@click.option("--targets", default="0.1,0.05", show_default=True)
@click.option("--check-trials", default=1 << 15, show_default=True)
@click.option("--prob-min", type=float, default=None, help="override prob_min")
@click.option("--guess", default="analytic", show_default=True)
@click.option(
    "--anchor",
    is_flag=True,
    help="the planner's grid: min_pd a bin edge, bins down to min_pd / 2 "
    "(needs a single target)",
)
@click.pass_obj
def frontier(
    opts: dict,
    guess: str,
    anchor: bool,
    modes: str,
    targets: str,
    check_trials: int,
    prob_min: float | None,
) -> None:
    """E5: honest cost frontier and selected paths, per mode, same seeds.

    The frontier is the minimum-cost terminal state of every probability bin
    (as in the paper's frontier plot), each re-simulated on independent trials.
    """
    bp, kw, neigh = config(opts["cfg"])
    if prob_min is not None:
        kw["prob_min"] = prob_min
    if anchor:
        target = float(targets.split(",")[0])
        step = -np.log10(target) / (kw["nprobs"] - 1)
        below = int(np.ceil(np.log10(2.0) / step))
        kw["prob_min"] = float(target * 10 ** (-below * step))
        kw["nprobs"] = kw["nprobs"] + below
    seeds = range(opts["seed_start"], opts["seed_start"] + opts["num_seeds"])
    tlist = [float(t) for t in targets.split(",")]
    out = opts["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for mode in modes.split(","):
        at_target = {t: [] for t in tlist}
        sel = {t: [] for t in tlist}
        secs = []
        points = []
        for seed in seeds:
            scheme, dt = search(
                bp,
                kw,
                neigh,
                seed=seed,
                mode=mode,
                backend=opts["backend"],
                guess=guess,
            )
            secs.append(dt)
            states = scheme.get_states().reshape(
                scheme.nstages,
                scheme.nthresholds,
                scheme.nprobs,
            )
            thresholds = np.asarray(scheme.thresholds)
            probs = np.asarray(scheme.probs)
            last = states[-1]
            pd_f, l_f = [], []
            for iprob in range(scheme.nprobs):
                col = last[:, iprob]
                ok = np.flatnonzero(~col["is_empty"])
                if ok.size == 0:
                    continue
                ithr = int(ok[np.argmin(col["cost"][ok])])
                path = backtrack(states, thresholds, probs, ithr, iprob)
                pd, cost = rescore(scheme, path, check_trials, seed + EVAL_OFFSET)
                if np.isfinite(pd) and pd > 0:
                    pd_f.append(pd)
                    l_f.append(cost)
                    points.append(
                        (
                            seed,
                            float(col["success_h1_cumul"][ithr]),
                            float(col["cost"][ithr]),
                            pd,
                            cost,
                        ),
                    )
            order = np.argsort(pd_f)
            lp = np.log10(np.asarray(pd_f)[order])
            ll = np.log10(np.asarray(l_f)[order])
            for t in tlist:
                if lp.size >= 2 and lp[0] <= np.log10(t) <= lp[-1]:
                    at_target[t].append(10 ** np.interp(np.log10(t), lp, ll))
                chosen = selected(scheme, t)
                if chosen is not None:
                    pd, cost = rescore(
                        scheme,
                        chosen[0],
                        check_trials,
                        seed + 2 * EVAL_OFFSET,
                    )
                    sel[t].append((chosen[1], pd, cost))
            print(f"{mode} seed {seed} done ({dt:.2f} s)", flush=True)
        np.save(
            out / f"frontier_{opts['cfg']}_{mode}.npy",
            np.array(
                points,
                dtype=[
                    ("seed", "i8"),
                    ("in_pd", "f8"),
                    ("in_cost", "f8"),
                    ("re_pd", "f8"),
                    ("re_cost", "f8"),
                ],
            ),
        )
        print(
            f"=== {opts['cfg']} {mode} prob_min {kw['prob_min']} guess {guess}: "
            f"run {np.median(secs):.2f} s ===",
        )
        for t in tlist:
            print(f"honest frontier L at P_d {t:g}: {stats(at_target[t])}")
            if sel[t]:
                arr = np.asarray(sel[t])
                print(
                    f"  selected (min_pd {t:g}): in-run P_d {stats(arr[:, 0])}",
                )
                print(
                    f"    delivered P_d {stats(arr[:, 1])}  min {arr[:, 1].min():.4f}"
                )
                print(
                    f"    within 10%: {np.mean(np.abs(arr[:, 1] / t - 1) <= 0.1):.2f}"
                    f"   delivered L {stats(arr[:, 2])}",
                )


if __name__ == "__main__":
    cli()
