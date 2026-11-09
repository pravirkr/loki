"""An EP leaf's phase model and the per-sample fold, from Python."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

C = 299792458.0


def taylor_leaf(accel: float, jerk: float, freq: float) -> np.ndarray:
    """A reported jerk-search leaf: (jerk, accel, freq, d0, f0) rows."""
    return np.array(
        [[jerk, 1.0], [accel, 1.0], [freq, 1e-6], [0.0, 0.0], [freq, 0.0]],
    )


def test_leaf_phase_is_the_taylor_polynomial() -> None:
    accel, jerk, freq, t_ref = 30.0, -0.01, 142.3, 50.0
    times = np.linspace(0.0, 100.0, 11)
    phase, model = libloki.psr_utils.leaf_phase(
        taylor_leaf(accel, jerk, freq),
        times,
        t_ref,
    )
    dt = times - t_ref
    delay = accel * dt**2 / 2 + jerk * dt**3 / 6
    assert model == "taylor"
    np.testing.assert_allclose(phase, freq * (dt - delay / C), rtol=1e-12)


def test_leaf_phase_rejects_a_flat_leaf() -> None:
    with pytest.raises(ValueError, match="n_params"):
        libloki.psr_utils.leaf_phase(np.zeros(8), np.zeros(2), 0.0)


def test_fold_phase_follows_an_accelerated_pulse() -> None:
    # A pulse train whose arrival times follow the leaf: folded with the
    # leaf's own phase, every pulse lands in one bin; at constant frequency
    # it smears.
    rng = np.random.default_rng(1)
    nsamps, tsamp, nbins = 1 << 16, 1e-3, 32
    times = np.arange(nsamps) * tsamp
    t_ref = times[-1] / 2
    leaf = taylor_leaf(2000.0, 0.0, 37.0)
    phase, _ = libloki.psr_utils.leaf_phase(leaf, times, t_ref)
    frac = phase - np.floor(phase)
    on = np.abs(frac - 0.5) < 0.5 / nbins
    ts_e = (rng.standard_normal(nsamps) + 3.0 * on).astype(np.float32)
    ts_v = np.ones(nsamps, dtype=np.float32)
    fold = libloki.fold.fold_phase(ts_e, ts_v, phase, nbins)
    assert fold.shape == (2, nbins)
    assert fold[1].sum() == pytest.approx(nsamps)
    profile = fold[0] / fold[1]
    assert np.argmax(profile) == nbins // 2
    constant = libloki.fold.fold_phase(ts_e, ts_v, 37.0 * (times - t_ref), nbins)
    assert profile.max() > 2 * (constant[0] / constant[1]).max()


def test_get_phase_idx_takes_a_frequency() -> None:
    assert libloki.psr_utils.get_phase_idx(
        delta_t=0.25,
        freq=2.0,
        nbins=8,
        delay=0.0,
    ) == pytest.approx(4.0)
