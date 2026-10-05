"""Production driver for DynamicThresholdSchemeCUDA.

Operational row: run() then get_best_path_thresholds(). That path is what a
live search uses immediately. Its cost and detection probability are the
optimiser's own Monte Carlo estimates, and they are optimistic.

"""

from __future__ import annotations

import numpy as np
from _branching import BRANCHING_PROD

from loki import libculoki

BP = BRANCHING_PROD[:127]
SEARCH_SEED = 1
REPORT_SEED = 10_001
REPORT_TRIALS = 4096
MIN_PD = 0.1

KW = {
    "ref_ducy": 0.1,
    "nbins": 64,
    "ntrials": 1024,
    "nprobs": 30,
    "prob_min": 0.05,
    "snr_final": 10.0,
    "nthresholds": 100,
    "ducy_max": 0.3,
    "beam_width": 2.5,
    "wtsp": 1.2,
    "mode": "improved",
}


def terminal(states: np.ndarray, nstages: int, nthr: int, nprobs: int) -> dict | None:
    grid = states.reshape(nstages, nthr, nprobs)[-1]
    mask = ~grid["is_empty"] & (grid["success_h1_cumul"] >= MIN_PD)
    if not np.any(mask):
        return None
    costs = np.where(mask, grid["cost"], np.inf)
    return grid.ravel()[int(np.argmin(costs))]


def main() -> None:
    scheme = libculoki.thresholds.DynamicThresholdSchemeCUDA(
        BP,
        seed=SEARCH_SEED,
        batch_size=256,
        **KW,
    )
    scheme.run(thres_neigh=11)
    path = np.asarray(scheme.get_best_path_thresholds(MIN_PD), dtype=np.float32)
    cell = terminal(scheme.get_states(), len(BP), KW["nthresholds"], KW["nprobs"])
    print(
        f"stages {len(BP)}  mode improved  search ntrials {KW['ntrials']}"
        f"  seed {SEARCH_SEED}",
    )
    print("operational (path used by the live search)")
    if cell is None or path.size != len(BP):
        print("  no path met min_pd")
    else:
        print(
            f"  L {float(cell['cost']):.6g}  Pd {float(cell['success_h1_cumul']):.4f}"
            f"  complexity {float(cell['complexity_cumul']):.6g}",
        )
    saved = scheme.save(outdir="scheme_results/")
    print(f"  saved {saved}")

    report = scheme.evaluate(path, REPORT_TRIALS, REPORT_SEED)
    last = report[-1]
    print(
        f"reporting (evaluate, {REPORT_TRIALS} trials, seed {REPORT_SEED};"
        " not used by the live search)",
    )
    if bool(last["is_empty"]):
        print("  empty")
    else:
        print(
            f"  L {float(last['cost']):.6g}  Pd {float(last['success_h1_cumul']):.4f}"
            f"  complexity {float(last['complexity_cumul']):.6g}",
        )


if __name__ == "__main__":
    main()
