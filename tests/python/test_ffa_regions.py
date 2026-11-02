"""FFA octave regions from Python."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki


def test_generate_ffa_regions_covers_band_by_octaves() -> None:
    regions = libloki.plans.generate_ffa_regions(1 / 200, 1 / 25, 64e-6, 64, 1.0)
    assert regions.dtype.names == ("f_start", "f_end", "nbins", "eta")
    order = np.argsort(regions["f_start"])
    f_start, f_end = regions["f_start"][order], regions["f_end"][order]
    assert f_start[0] == pytest.approx(25.0)
    assert f_end[-1] == pytest.approx(200.0)
    np.testing.assert_allclose(f_start[1:], f_end[:-1])
    # Each octave down doubles the bins and the tolerance in bins.
    np.testing.assert_array_equal(regions["nbins"][order], [256, 128, 64])
    np.testing.assert_allclose(regions["eta"][order], [4.0, 2.0, 1.0])
