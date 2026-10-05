"""Bias study: Monte Carlo bias at 127 production stages.

Compares in-run search estimates (what the live pipeline uses) to independent
``evaluate()`` re-scores (reporting only).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
import h5py
import numpy as np
from _branching import BRANCHING_PROD, PROD_KW
from matplotlib import pyplot as plt

from loki import libculoki, libloki

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

BP = BRANCHING_PROD[:127]
EVAL_TRIALS = 4096
EVAL_OFFSET = 10_000
NEIGH = 11
MIN_PD = 0.1
SEARCH_TRIALS = (1024, 2048, 4096)

KW = {k: v for k, v in PROD_KW.items() if k != "ntrials"}


def summarise(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "no finite samples"
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return f"mean {arr.mean():.6g}  std {std:.6g}  n={arr.size}"


def chosen_cell(states: np.ndarray, nthr: int, nprobs: int) -> dict | None:
    grid = states.reshape(len(BP), nthr, nprobs)[-1]
    mask = ~grid["is_empty"] & (grid["success_h1_cumul"] >= MIN_PD)
    if not np.any(mask):
        return None
    costs = np.where(mask, grid["cost"], np.inf)
    return grid.ravel()[int(np.argmin(costs))]


def rescore(
    scheme: libculoki.thresholds.DynamicThresholdSchemeCUDA,
    path: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    ev = scheme.evaluate(np.asarray(path, np.float32), EVAL_TRIALS, seed)
    last = ev[-1]
    if bool(last["is_empty"]):
        return float("nan"), float("nan"), float("nan")
    return (
        float(last["cost"]),
        float(last["success_h1_cumul"]),
        float(last["complexity_cumul"]),
    )


def median_run(
    factory: Callable[[], libculoki.thresholds.DynamicThresholdSchemeCUDA],
    repeat: int = 3,
) -> float:
    factory()
    samples = []
    for _ in range(repeat):
        scheme = factory()
        t0 = time.perf_counter()
        scheme.run(NEIGH)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def one_search(
    factory: Callable[[int], libculoki.thresholds.DynamicThresholdSchemeCUDA],
    scorer: libculoki.thresholds.DynamicThresholdSchemeCUDA,
    seed: int,
    nthr: int,
    nprobs: int,
) -> dict:
    scheme = factory(seed)
    scheme.run(NEIGH)
    cell = chosen_cell(scheme.get_states(), nthr, nprobs)
    path = np.asarray(scheme.get_best_path_thresholds(MIN_PD), dtype=np.float32)
    cost, pd, comp = rescore(scorer, path, seed + EVAL_OFFSET)
    row = {
        "seed": seed,
        "in_cost": float("nan") if cell is None else float(cell["cost"]),
        "in_pd": float("nan") if cell is None else float(cell["success_h1_cumul"]),
        "in_comp": float("nan") if cell is None else float(cell["complexity_cumul"]),
        "re_cost": cost,
        "re_pd": pd,
        "re_comp": comp,
        "feasible": float(pd >= MIN_PD),
        "path": path,
    }
    print(
        f"  seed {seed}  in L {row['in_cost']:.6g} Pd {row['in_pd']:.4f}"
        f"  re L {cost:.6g} Pd {pd:.4f}  feasible {bool(row['feasible'])}",
        flush=True,
    )
    return row


def path_stats(rows: list[dict]) -> str:
    paths = [row["path"] for row in rows if row["path"].size == len(BP)]
    if not paths:
        return "no paths"
    keys = [row["path"].tobytes() for row in rows if row["path"].size == len(BP)]
    _uniq, counts = np.unique(keys, return_counts=True)
    modal = int(counts.max())
    stacked = np.stack(paths)
    total = 0.0
    pairs = 0
    for i in range(len(stacked)):
        for j in range(i + 1, len(stacked)):
            total += float(np.abs(stacked[i] - stacked[j]).sum())
            pairs += 1
    mean_l1 = total / pairs if pairs else float("nan")
    return (
        f"unique {len(_uniq)}/{len(keys)}  "
        f"modal fraction {modal / len(keys):.3f}  "
        f"mean pairwise L1 {mean_l1:.4g}"
    )


def write_plots(cuda_rows: dict[int, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    markers = {1024: "o", 2048: "s", 4096: "^"}
    ax = axes[0]
    for ntrials, rows in cuda_rows.items():
        x = np.array([r["in_cost"] for r in rows])
        y = np.array([r["re_cost"] for r in rows])
        ax.scatter(x, y, marker=markers[ntrials], label=f"search N={ntrials}")
    finite = np.concatenate(
        [
            np.array([r["in_cost"] for r in rows] + [r["re_cost"] for r in rows])
            for rows in cuda_rows.values()
        ],
    )
    finite = finite[np.isfinite(finite)]
    lo, hi = float(finite.min()), float(finite.max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1)
    ax.set_xlabel("in-run L (operational estimate)")
    ax.set_ylabel("re-scored L (evaluate, 4096 trials)")
    ax.set_title("operational estimate vs reporting")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for ntrials, rows in cuda_rows.items():
        y = np.array([r["re_pd"] for r in rows])
        x = np.full(y.shape, ntrials) + np.linspace(-40, 40, y.size)
        ax.scatter(x, y, marker=markers[ntrials], label=f"N={ntrials}")
    ax.axhline(MIN_PD, color="0.3", lw=1, label="target 0.1")
    ax.set_xlabel("search ntrials")
    ax.set_ylabel("re-scored detection probability")
    ax.set_title("reporting Pd vs target")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bias_panels.pdf")
    fig.savefig(out_dir / "bias_panels.png", dpi=140)
    plt.close(fig)

    rows = cuda_rows[1024]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(np.full(len(rows), 0), [r["in_cost"] for r in rows], label="in-run L")
    ax.scatter(np.full(len(rows), 1), [r["re_cost"] for r in rows], label="re-scored L")
    for r in rows:
        ax.plot([0, 1], [r["in_cost"], r["re_cost"]], color="0.75", lw=0.6, zorder=0)
    ax.set_xticks([0, 1], ["in-run", "re-scored"])
    ax.set_ylabel("L")
    ax.set_title("seed scatter at ntrials=1024")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "path_scatter.pdf")
    fig.savefig(out_dir / "path_scatter.png", dpi=140)
    plt.close(fig)


@click.command()
@click.option(
    "--out-dir",
    type=Path,
    default=REPO_ROOT / "results",
    help="write optional plots/npz here (results is gitignored)",
)
@click.option(
    "--seed-start",
    type=int,
    default=100,
    help="first search seed (inclusive)",
)
@click.option("--num-seeds", type=int, default=16, help="number of consecutive seeds")
@click.option(
    "--plots",
    action="store_true",
    help="write bias_panels and path_scatter figures to out-dir",
)
@click.option(
    "--save-npz",
    action="store_true",
    help="write bias_study_127.npz arrays to out-dir (local analysis only)",
)
def main(
    out_dir: Path,
    seed_start: int,
    num_seeds: int,
    plots: bool,
    save_npz: bool,
) -> None:
    seeds = list(range(seed_start, seed_start + num_seeds))
    nthr = KW["nthresholds"]
    nprobs = KW["nprobs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"stages {len(BP)}  seeds {seeds[0]}..{seeds[-1]} ({len(seeds)})"
        f"  evaluate {EVAL_TRIALS}  min_pd {MIN_PD}",
    )
    print("evaluate() is reporting only. The live path is the in-run path.")

    print("=== CUDA improved run() timing (median of 3) ===")
    for ntrials in SEARCH_TRIALS:
        seconds = median_run(
            lambda ntrials=ntrials: libculoki.thresholds.DynamicThresholdSchemeCUDA(
                BP,
                mode="improved",
                seed=1,
                batch_size=256,
                ntrials=ntrials,
                **KW,
            ),
        )
        print(f"cuda ntrials {ntrials:4d}  {seconds:.3f} s")

    def cuda_factory(
        ntrials: int,
    ) -> Callable[[int], libculoki.thresholds.DynamicThresholdSchemeCUDA]:
        def make(seed: int) -> libculoki.thresholds.DynamicThresholdSchemeCUDA:
            return libculoki.thresholds.DynamicThresholdSchemeCUDA(
                BP,
                mode="improved",
                seed=seed,
                batch_size=256,
                ntrials=ntrials,
                **KW,
            )

        return make

    scorer = libculoki.thresholds.DynamicThresholdSchemeCUDA(
        BP,
        mode="improved",
        seed=1,
        batch_size=256,
        ntrials=1024,
        **KW,
    )
    cuda_rows: dict[int, list[dict]] = {}
    for ntrials in SEARCH_TRIALS:
        print(f"=== CUDA improved search ntrials {ntrials} ===")
        cuda_rows[ntrials] = [
            one_search(cuda_factory(ntrials), scorer, seed, nthr, nprobs)
            for seed in seeds
        ]
        rows = cuda_rows[ntrials]
        print(f"in L   {summarise([r['in_cost'] for r in rows])}")
        print(f"re L   {summarise([r['re_cost'] for r in rows])}")
        print(f"in Pd  {summarise([r['in_pd'] for r in rows])}")
        print(f"re Pd  {summarise([r['re_pd'] for r in rows])}")
        print(
            f"bias L (re-in) {summarise([r['re_cost'] - r['in_cost'] for r in rows])}",
        )
        print(f"bias Pd (re-in) {summarise([r['re_pd'] - r['in_pd'] for r in rows])}")
        feas = np.mean([r["feasible"] for r in rows])
        print(f"fraction with re-scored Pd >= {MIN_PD}: {feas:.3f}")
        print(f"paths {path_stats(rows)}")

    print("=== CPU improved, 8 threads, search ntrials 1024 ===")

    def cpu_make(seed: int) -> libloki.thresholds.DynamicThresholdScheme:
        return libloki.thresholds.DynamicThresholdScheme(
            BP,
            mode="improved",
            seed=seed,
            nthreads=8,
            ntrials=1024,
            **KW,
        )

    cpu_rows = [one_search(cpu_make, scorer, seed, nthr, nprobs) for seed in seeds]
    print(f"cpu in L  {summarise([r['in_cost'] for r in cpu_rows])}")
    print(f"cpu re L  {summarise([r['re_cost'] for r in cpu_rows])}")
    print(f"cpu re Pd {summarise([r['re_pd'] for r in cpu_rows])}")
    print(
        f"cpu fraction re-scored Pd >= {MIN_PD}: "
        f"{np.mean([r['feasible'] for r in cpu_rows]):.3f}",
    )
    print(f"cpu paths {path_stats(cpu_rows)}")

    print("=== guess path vs optimised path (reporting evaluate at 4096) ===")
    probe = cuda_factory(1024)(seeds[0])
    probe.run(NEIGH)
    with tempfile.TemporaryDirectory() as tmp:
        saved = probe.save(outdir=tmp)
        with h5py.File(saved, "r") as handle:
            guess = np.asarray(handle["guess_path"], dtype=np.float32)
    guess_costs = []
    guess_pds = []
    for seed in seeds:
        cost, pd, _comp = rescore(scorer, guess, seed + EVAL_OFFSET)
        guess_costs.append(cost)
        guess_pds.append(pd)
    opt = cuda_rows[1024]
    opt_mean = float(np.mean([r["re_cost"] for r in opt]))
    guess_mean = float(np.mean(guess_costs))
    ratio = guess_mean / opt_mean if opt_mean else float("nan")
    print(f"optimised re L  {summarise([r['re_cost'] for r in opt])}")
    print(f"guess re L      {summarise(guess_costs)}")
    print(f"optimised re Pd {summarise([r['re_pd'] for r in opt])}")
    print(f"guess re Pd     {summarise(guess_pds)}")
    print(
        f"guess / optimised re-scored L = {ratio:.3g}  "
        "(reporting; live search uses the in-run optimised path)",
    )

    if save_npz:
        arrays = {}
        for ntrials, rows in cuda_rows.items():
            for key in (
                "seed",
                "in_cost",
                "in_pd",
                "in_comp",
                "re_cost",
                "re_pd",
                "re_comp",
                "feasible",
            ):
                arrays[f"cuda_{ntrials}_{key}"] = np.asarray([r[key] for r in rows])
            arrays[f"cuda_{ntrials}_path"] = np.stack([r["path"] for r in rows])
        for key in ("seed", "in_cost", "in_pd", "re_cost", "re_pd", "feasible"):
            arrays[f"cpu1024_{key}"] = np.asarray([r[key] for r in cpu_rows])
        arrays["guess_re_cost"] = np.asarray(guess_costs)
        arrays["guess_re_pd"] = np.asarray(guess_pds)
        arrays["guess_path"] = guess
        np.savez(out_dir / "bias_study_127.npz", **arrays)
        print(f"wrote {out_dir / 'bias_study_127.npz'}")

    if plots:
        write_plots(cuda_rows, out_dir)
        print("wrote bias_panels.* and path_scatter.*")


if __name__ == "__main__":
    main()
