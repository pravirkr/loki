from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki_survey.config import Box, Orbit, Region
from loki_survey.plan import Unit, band_edges, check_nyquist, param_limits
from loki_survey.prep import downsample, inf_off_ranges, truncated_length
from loki_survey.unit import keeps_raw

if TYPE_CHECKING:
    from pathlib import Path


def region(**kw: object) -> Region:
    base = {
        "name": "r",
        "f_min": 10.0,
        "f_max": 1000.0,
        "nbins": 32,
        "downsample": 1,
        "ducy_max": 0.2,
        "bands": 1,
    }
    return Region(**{**base, **kw})


def box(**kw: object) -> Box:
    base = {
        "name": "b",
        "kind": "ep",
        "order": "jerk",
        "limits": {"jerk": [-1.0, 1.0], "accel": [-5.0, 5.0]},
        "orbit": None,
        "regions": ("r",),
        "use_fourier": None,
        "segments": None,
        "bseg_brute": None,
        "ref_segs": None,
    }
    return Box(**{**base, **kw})


def test_bands_are_log_uniform_and_cover_the_region() -> None:
    edges = band_edges(region(bands=2))
    assert edges[0][0] == 10.0
    assert edges[-1][1] == 1000.0
    assert edges[0][1] == pytest.approx(100.0)
    assert math.log(edges[0][1] / edges[0][0]) == pytest.approx(
        math.log(edges[1][1] / edges[1][0]),
    )


def test_explicit_limits_in_loki_row_order() -> None:
    assert param_limits(box(), 100.0, 200.0) == [
        [-1.0, 1.0],
        [-5.0, 5.0],
        [100.0, 200.0],
    ]


def test_orbit_limits_come_from_loki() -> None:
    rows = param_limits(
        box(limits={}, orbit=Orbit(p_orb_min=3600.0, m_c_max=0.5, m_p_min=1.2)),
        100.0,
        200.0,
    )
    assert len(rows) == 3
    # loki widens the band by the orbit's largest Doppler shift.
    assert rows[-1][0] < 100.0
    assert rows[-1][1] > 200.0
    assert rows[0][0] == -rows[0][1]


def test_nyquist_check() -> None:
    unit = Unit(
        id="u",
        series="s",
        region="r",
        band=0,
        box="b",
        kind="ffa",
        order="freq",
        downsample=1,
        nsamps=1024,
        tsamp=1e-3,
        f_min=10.0,
        f_max=100.0,
        nbins=32,
        ducy_max=0.2,
        param_limits=[[10.0, 100.0]],
        use_fourier=None,
        segments=None,
        bseg_brute=None,
        ref_segs=None,
        orbit=None,
        plan_key=None,
    )
    with pytest.raises(ValueError, match="sampling interval"):
        check_nyquist(unit)


def test_downsample_keeps_the_sufficient_statistics() -> None:
    rng = np.random.default_rng(1)
    e = rng.normal(size=16).astype(np.float32)
    v = rng.uniform(0.5, 2.0, size=16).astype(np.float32)
    de, dv = downsample(e, v, 4)
    assert de.shape == (4,)
    assert de.sum() == pytest.approx(e.sum(), rel=1e-6)
    assert dv.sum() == pytest.approx(v.sum(), rel=1e-6)


def test_truncation() -> None:
    assert truncated_length(35_840_000, "pow2") == 1 << 25
    assert truncated_length(1 << 20, "pow2") == 1 << 20
    assert truncated_length(1000, "none") == 1000


def test_inf_off_ranges(tmp_path: Path) -> None:
    dat = tmp_path / "s.dat"
    dat.write_bytes(b"")
    (tmp_path / "s.inf").write_text(
        " On/Off bin pair #  1      =  0         , 999\n"
        " On/Off bin pair #  2      =  1500      , 1899\n",
    )
    assert inf_off_ranges(dat, 2000) == [(1000, 1500), (1900, 2000)]


def test_raw_sample_is_fixed_and_near_the_fraction() -> None:
    ids = [f"unit{i}" for i in range(4000)]
    kept = [i for i in ids if keeps_raw(i, 0.25)]
    assert kept == [i for i in ids if keeps_raw(i, 0.25)]
    assert 0.2 < len(kept) / len(ids) < 0.3
    assert not any(keeps_raw(i, 0.0) for i in ids)
