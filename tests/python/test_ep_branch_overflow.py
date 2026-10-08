"""EP on a synthetic series whose branching overflows the per-batch workspace.

One short, bright impulse in white noise, at the anchor segment, makes nearly
every leaf survive the first stages. Some batches then need more children
than the workspace holds (batch_size * branch_max): branch counts are integer
and step up across the band, and the product over parameters is not capped.
Before the fix, the branch write loops ran past the workspace and the run
aborted with heap corruption (SIGABRT, e.g. "free(): invalid size") or
segfaulted. Now an oversize batch is retried in smaller pieces and the run
completes.

Configuration: 2^22 samples at 100 us (419 s), 64 segments, 64 bins, order 3,
over a rectangular (jerk, accel, freq) box.
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
TSAMP = 1.0e-4
NSEG = 64
ANCHOR = 59
PARAM_LIMITS = [
    [-146.0, 146.0],  # jerk (m/s^3)
    [-9743.0, 9743.0],  # accel (m/s^2)
    [98.09, 114.94],  # freq (Hz)
]


def _series() -> np.ndarray:
    rng = np.random.default_rng(1)
    ts = rng.standard_normal(NSAMPS)
    start = int((ANCHOR + 0.5) * NSAMPS / NSEG)
    width = 30  # 3 ms
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
    )


@pytest.mark.parametrize(
    ("thresholds", "batch_size"),
    [
        # Before the fix: SIGABRT (heap corruption after the size check).
        ([3.0] * (NSEG - 1), 1024),
        # Before the fix: SIGABRT as well, through a smaller batch.
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
