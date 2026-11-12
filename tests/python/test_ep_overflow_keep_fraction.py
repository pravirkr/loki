"""EP with a world tree too small for its survivors: overflow_keep_fraction.

One bright impulse in white noise, at the anchor segment, lets thousands of
leaves survive every stage. With a tree that holds them all, nothing
overflows and the fraction must not change a bit of the output. With a tree
of 4096 leaves the stages overflow; level_stats counts the overflowing
batches, and a larger fraction keeps more survivors at the price of more
trims.

A bright pulsar fills the tree too, at every stage. One overflow then decides
whether a weak pulsar elsewhere in the band survives: the median keeps half
the pool and holds the raised threshold for the rest of the stage, while a
larger fraction keeps what fits.

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
# Stage i folds i + 2 segments; a signal's S/N grows as the square root of
# the segments folded, and so does this cut, to 4 at the last stage.
RISING_THRESHOLDS = [4.0 * np.sqrt((i + 2) / NSEG) for i in range(NSEG - 1)]
TOBS = NSAMPS * TSAMP
BRIGHT = (98.2, 50.0)  # (freq Hz, matched S/N)
WEAK = (98.5, 10.0)
DUTY = 0.1
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


def _two_pulsars(seed: int) -> np.ndarray:
    """Top-hat pulsars in white noise: the bright one and the weak one."""
    rng = np.random.default_rng(seed)
    ts = rng.standard_normal(NSAMPS)
    t = np.arange(NSAMPS) * TSAMP
    for freq, snr in (BRIGHT, WEAK):
        on = (t * freq) % 1.0 < DUTY
        # A top hat of duty d and height a folds to S/N a * sqrt(d * NSAMPS).
        ts[on] += snr / np.sqrt(DUTY * NSAMPS)
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
    *,
    ts_e: np.ndarray | None = None,
    thresholds: list[float] = THRESHOLDS,
) -> dict[str, np.ndarray]:
    if ts_e is None:
        ts_e = _series()
    tag = f"keep{keep_fraction}_sugg{max_sugg}"
    ep = libloki.prune.EPMultiPassFourier(
        _config(keep_fraction),
        thresholds,
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


def _weak_best(out: dict[str, np.ndarray]) -> float:
    """Best final score within two resolutions 1 / T of the weak pulsar, or 0."""
    freqs = out["param_sets"][:, -3, 0]
    near = np.abs(freqs - WEAK[0]) <= 2.0 / TOBS
    return float(out["scores"][near].max()) if near.any() else 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_median_trim_can_lose_a_weak_pulsar(tmp_path: Path, backend: str) -> None:
    max_sugg = 16384
    kept = {0.5: 0, 0.9: 0}
    for seed in range(1, 6):
        ts_e = _two_pulsars(seed)
        full = _run(
            tmp_path, 0.5, 2**20, backend, ts_e=ts_e, thresholds=RISING_THRESHOLDS
        )
        assert full["level_stats"]["n_overloads"].sum() == 0
        best = _weak_best(full)
        assert best > 0
        found = {}
        for keep_fraction in kept:
            out = _run(
                tmp_path,
                keep_fraction,
                max_sugg,
                backend,
                ts_e=ts_e,
                thresholds=RISING_THRESHOLDS,
            )
            assert out["level_stats"]["n_overloads"].sum() > 0
            found[keep_fraction] = _weak_best(out) == best
            kept[keep_fraction] += found[keep_fraction]
        # Keeping more of the pool never loses the leaf the median keeps.
        assert found[0.9] or not found[0.5]
    # The median loses the weak pulsar's best leaf in more seeds.
    assert kept[0.9] > kept[0.5]
