"""Search parameter limits from physical circular-orbit limits."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

TOBS = (1 << 25) * 64e-6


@pytest.mark.parametrize("poly_order", [2, 3, 4])
def test_circular_param_limits_matches_pyloki(poly_order: int) -> None:
    pyloki_config = pytest.importorskip("pyloki.config")
    expected = pyloki_config.ParamLimits.from_circular(
        (25.0, 29.0), 5 * TOBS, 10.0, 1.2, poly_order=poly_order
    ).limits
    limits = libloki.configs.circular_param_limits(
        25.0, 29.0, poly_order=poly_order, p_orb_min=5 * TOBS, m_c_max=10.0, m_p_min=1.2
    )
    assert limits.shape == (poly_order, 2)
    np.testing.assert_allclose(limits, expected, rtol=1e-6)


def test_implied_circular_orbit_inverts_the_limits() -> None:
    limits = libloki.configs.circular_param_limits(
        25.0, 29.0, poly_order=3, p_orb_min=5 * TOBS, m_c_max=10.0, m_p_min=1.2
    )
    p_orb, m_c_min = libloki.configs.implied_circular_orbit(limits, m_p=1.2)
    assert p_orb == pytest.approx(5 * TOBS, rel=1e-9)
    assert m_c_min == pytest.approx(10.0, rel=1e-6)
    # An acceleration-only box implies no orbit.
    assert libloki.configs.implied_circular_orbit(limits[1:]) is None


def test_circular_limits_drive_a_search_config() -> None:
    limits = libloki.configs.circular_param_limits(
        140.0, 145.0, poly_order=3, p_orb_min=5 * TOBS
    )
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=1 << 25,
        tsamp=64e-6,
        nbins=64,
        eta=1.0,
        param_limits=limits,
        bseg_brute=4096,
        bseg_ffa=1 << 18,
        prune_poly_order=3,
    )
    assert cfg.nparams == 3
    stored = cfg.param_limits
    np.testing.assert_array_equal(stored["min"], limits[:, 0])
    np.testing.assert_array_equal(stored["max"], limits[:, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poly_order": 1, "p_orb_min": 1e4},
        {"poly_order": 5, "p_orb_min": 1e4},
        {"poly_order": 3, "p_orb_min": 0.0},
        {"poly_order": 3, "p_orb_min": 1e4, "m_c_max": -1.0},
        # A microsecond orbit needs a velocity above c.
        {"poly_order": 3, "p_orb_min": 1e-6},
    ],
)
def test_circular_param_limits_rejects_unphysical_input(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="circular orbit limits"):
        libloki.configs.circular_param_limits(25.0, 29.0, **kwargs)
