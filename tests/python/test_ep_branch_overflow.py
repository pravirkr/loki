"""EP on a synthetic series whose branching overflows the per-batch workspace.

One short, bright impulse in white noise, at the anchor segment, makes nearly
every leaf survive the first stages. Some of them branch past the planned
pattern (the product of their per-parameter branch counts exceeds
batch_size * branch_max). Before the fix, the branch write loops ran past the
workspace: the run aborted with "Branch factor exceeded workspace size"
followed by heap corruption (SIGABRT), or segfaulted. Now an oversize batch is
retried in smaller pieces and the run completes.

The configuration is one 687 s span at 163.84 us, 60-70 Hz, 64 bins, order 3,
64 segments, with the parameter limits of a circular-orbit prior of
p_orb_min = 687 s, m_c <= 1, m_p >= 1.1 (pyloki ParamLimits.from_circular).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from loki import libloki

if TYPE_CHECKING:
    from pathlib import Path

NSAMPS = 2**22
TSAMP = 2 * 8.192e-05
NSEG = 64
ANCHOR = 59
# from_circular((60, 70), 687.19476736, m_c_max=1.0, m_p_min=1.1, poly_order=3)
PARAM_LIMITS = [
    [-54.373973588540686, 54.373973588540686],  # jerk (m/s^3)
    [-5946.90563843783, 5946.90563843783],  # accel (m/s^2)
    [59.869826803184516, 70.15186872961807],  # freq (Hz)
]


def _series() -> np.ndarray:
    rng = np.random.default_rng(20261007)
    ts = rng.standard_normal(NSAMPS)
    start = int((ANCHOR + 0.5) * NSAMPS / NSEG)
    width = int(0.005 / TSAMP)  # 5 ms
    ts[start : start + width] += 200.0 / np.sqrt(width)  # matched S/N 200
    return ts.astype(np.float32)


def _config() -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=TSAMP,
        nbins=64,
        eta=1.0,
        param_limits=np.array(PARAM_LIMITS, dtype=np.float64),
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        nthreads=1,
        bseg_brute=NSAMPS // NSEG // 16,
        bseg_ffa=NSAMPS // NSEG,
        prune_poly_order=3,
        p_orb_min=687.19476736,
        m_c_max=1.0,
        m_p_min=1.1,
    )


@pytest.mark.parametrize(
    ("thresholds", "batch_size"),
    [
        # Before the fix: SIGABRT (heap corruption after the size check).
        ([3.0] * (NSEG - 1), 1024),
        # Before the fix: segfault. A small batch cannot average a few
        # heavily branching leaves against the rest.
        (np.linspace(2.9, 4.5, NSEG - 1).tolist(), 64),
    ],
)
def test_ep_survives_branching_past_the_workspace(
    tmp_path: Path,
    thresholds: list[float],
    batch_size: int,
) -> None:
    ts_e = _series()
    ep = libloki.prune.EPMultiPassFourier(
        _config(),
        thresholds,
        ref_segs=[ANCHOR],
        max_sugg=2**18,
        batch_size=batch_size,
        show_progress=False,
    )
    ep.execute(ts_e, np.ones_like(ts_e), str(tmp_path), "overflow")
    result = tmp_path / f"overflow_pruning_nstages_{NSEG}_results.h5"
    with h5py.File(result) as f:
        run = f[f"runs/{ANCHOR:03d}_00"]
        level_stats = run["level_stats"][()]
        n_cands = run["scores"].shape[0]
    # The search reached the last stage and kept candidates.
    assert level_stats["level"].max() == NSEG - 1
    assert n_cands > 0
