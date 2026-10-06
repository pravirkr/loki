"""Stage-0 CPU vs CUDA survival parity (statistical, not bit-identical).

Stage 0 has a single parent cell, so this isolates noise injection, profile
scaling, and boxcar scoring without deeper DP mixing. CPU and CUDA use
different RNGs; expect agreement within binomial sampling noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        "cuda" not in libloki.available_backends(),
        reason="CUDA backend not built",
    ),
]

BP = np.array(
    [4.0, 9.0, 1.0, 2.25575101, 3.98980204, 3.0, 2.80514208, 3.20839363],
    dtype=np.float32,
)
NTHR = 60
NPROBS = 12
NTRIALS = 1024
THRES_NEIGH = 6
N_SEEDS = 3
TOL = 3.0 / np.sqrt(NTRIALS)

KW = {
    "ref_ducy": 0.1,
    "nbins": 64,
    "ntrials": NTRIALS,
    "nprobs": NPROBS,
    "prob_min": 0.05,
    "snr_final": 8.0,
    "nthresholds": NTHR,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.2,
}


def _stage0_survival_by_threshold(states: np.ndarray, nstages: int) -> dict[int, tuple[float, float]]:
    grid = states.reshape(nstages, NTHR, NPROBS)[0]
    out: dict[int, tuple[float, float]] = {}
    for ithr, iprob in np.argwhere(~grid["is_empty"]):
        out[int(ithr)] = (
            float(grid["success_h0"][ithr, iprob]),
            float(grid["success_h1"][ithr, iprob]),
        )
    return out


@pytest.mark.parametrize("mode", ["improved", "legacy"])
def test_cpu_cuda_stage0_survival_parity(mode: str) -> None:
    cpu_by_seed: list[dict[int, tuple[float, float]]] = []
    cuda_by_seed: list[dict[int, tuple[float, float]]] = []
    for seed in range(N_SEEDS):
        cpu = libloki.thresholds.DynamicThresholdScheme(
            BP, mode=mode, seed=seed, nthreads=8, **KW
        )
        gpu = libloki.thresholds.DynamicThresholdScheme(
            BP, mode=mode, seed=seed, backend="cuda", **KW
        )
        cpu.run(thres_neigh=THRES_NEIGH)
        gpu.run(thres_neigh=THRES_NEIGH)
        cpu_by_seed.append(_stage0_survival_by_threshold(cpu.get_states(), len(BP)))
        cuda_by_seed.append(_stage0_survival_by_threshold(gpu.get_states(), len(BP)))

    common = sorted(
        set.intersection(*[set(d) for d in cpu_by_seed + cuda_by_seed])
    )
    assert common, "no common nonempty stage-0 cells between CPU and CUDA"

    for ithr in common:
        cpu_h0 = np.mean([d[ithr][0] for d in cpu_by_seed])
        cpu_h1 = np.mean([d[ithr][1] for d in cpu_by_seed])
        cuda_h0 = np.mean([d[ithr][0] for d in cuda_by_seed])
        cuda_h1 = np.mean([d[ithr][1] for d in cuda_by_seed])
        assert abs(cpu_h0 - cuda_h0) <= TOL, (
            f"mode={mode} thr idx {ithr} H0 cpu={cpu_h0:.4f} cuda={cuda_h0:.4f}"
        )
        assert abs(cpu_h1 - cuda_h1) <= TOL, (
            f"mode={mode} thr idx {ithr} H1 cpu={cpu_h1:.4f} cuda={cuda_h1:.4f}"
        )
