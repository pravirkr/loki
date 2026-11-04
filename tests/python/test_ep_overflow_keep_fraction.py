"""EP with a world tree too small for its survivors: overflow_keep_fraction.

One bright impulse in white noise, at the anchor segment, lets thousands of
leaves survive every stage. With a tree that holds them all, nothing
overflows and the fraction must not change a bit of the output. With a tree
of 4096 leaves the stages overflow; level_stats counts the overflowing
batches, and a larger fraction keeps more survivors at the price of more
trims.

Configuration: 2^20 samples at 100 us, 32 segments, 32 bins, order 3, over a
rectangular (jerk, accel, freq) box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from loki import libloki

if TYPE_CHECKING:
    from pathlib import Path

NSAMPS = 2**20
TSAMP = 1.0e-4
NSEG = 32
ANCHOR = NSEG // 2
PARAM_LIMITS = [
    [-50.0, 50.0],  # jerk (m/s^3)
    [-3000.0, 3000.0],  # accel (m/s^2)
    [98.0, 99.0],  # freq (Hz)
]
THRESHOLDS = [3.0] * (NSEG - 1)
FIELDS = ("scores", "param_sets", "level_stats")
BACKENDS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            "cuda" not in libloki.available_backends(),
            reason="CUDA backend not built",
        ),
    ),
]


def _series() -> np.ndarray:
    rng = np.random.default_rng(1)
    ts = rng.standard_normal(NSAMPS)
    start = int((ANCHOR + 0.5) * NSAMPS / NSEG)
    width = 30  # 3 ms
    ts[start : start + width] += 100.0 / np.sqrt(width)  # matched S/N 100
    return ts.astype(np.float32)


def _config(keep_fraction: float) -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=TSAMP,
        nbins=32,
        eta=1.0,
        param_limits=np.array(PARAM_LIMITS, dtype=np.float64),
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        nthreads=1,
        bseg_brute=NSAMPS // NSEG // 16,
        nsegments=NSEG,
        prune_poly_order=3,
        overflow_keep_fraction=keep_fraction,
    )


def _run(
    tmp_path: Path,
    keep_fraction: float,
    max_sugg: int,
    backend: str,
) -> dict[str, np.ndarray]:
    ts_e = _series()
    tag = f"keep{keep_fraction}_sugg{max_sugg}"
    ep = libloki.prune.EPMultiPassFourier(
        _config(keep_fraction),
        THRESHOLDS,
        ref_segs=[ANCHOR],
        max_sugg=max_sugg,
        batch_size=64,
        show_progress=False,
        backend=backend,
    )
    ep.execute(ts_e, np.ones_like(ts_e), str(tmp_path), tag)
    result = tmp_path / f"{tag}_pruning_nstages_{NSEG}_results.h5"
    with h5py.File(result) as f:
        run = f[f"runs/{ANCHOR:03d}_00"]
        return {field: run[field][()] for field in FIELDS}


def test_overflow_keep_fraction_is_checked() -> None:
    assert _config(0.5).overflow_keep_fraction == 0.5
    assert _config(1.0).overflow_keep_fraction == 1.0
    for value in (0.0, -0.5, 1.5, np.nan):
        with pytest.raises(ValueError, match="overflow_keep_fraction"):
            _config(value)


@pytest.mark.parametrize("backend", BACKENDS)
def test_keep_fraction_is_neutral_without_overflow(
    tmp_path: Path,
    backend: str,
) -> None:
    half = _run(tmp_path, 0.5, 2**20, backend)
    assert half["level_stats"]["n_overloads"].sum() == 0
    for keep_fraction in (0.75, 1.0):
        other = _run(tmp_path, keep_fraction, 2**20, backend)
        for field in FIELDS:
            np.testing.assert_array_equal(other[field], half[field])


@pytest.mark.parametrize("backend", BACKENDS)
def test_keep_fraction_on_overflow(tmp_path: Path, backend: str) -> None:
    max_sugg = 4096
    half = _run(tmp_path, 0.5, max_sugg, backend)
    most = _run(tmp_path, 0.9, max_sugg, backend)
    for out in (half, most):
        assert out["level_stats"]["n_overloads"].sum() > 0
        assert out["level_stats"]["n_leaves_surv"].max() <= max_sugg
        assert out["level_stats"]["level"].max() == NSEG - 1
    # Keeping more of each overflowing pool trims more often and ends with
    # more candidates.
    assert (
        most["level_stats"]["n_overloads"].sum()
        > half["level_stats"]["n_overloads"].sum()
    )
    assert most["scores"].shape[0] > half["scores"].shape[0]
