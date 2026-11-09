"""An EP leaf's phase model, from Python."""

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


def test_get_phase_idx_takes_a_frequency() -> None:
    assert libloki.psr_utils.get_phase_idx(
        delta_t=0.25,
        freq=2.0,
        nbins=8,
        delay=0.0,
    ) == pytest.approx(4.0)
