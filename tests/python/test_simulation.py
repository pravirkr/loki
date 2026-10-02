"""Bindings for simulated timeseries and the arrays built on load."""

from __future__ import annotations

import numpy as np

from loki import libloki


def test_generate_shape_and_seed() -> None:
    params = libloki.simulation.ModulatorParams()
    first = libloki.simulation.PulseSignalConfig(
        0.05,
        0.001,
        2048,
        15.0,
        0.1,
        "derivative",
        params,
        None,
        7,
    )
    series = first.generate("gaussian", 0.5)
    assert series.nsamps == 2048
    assert series.dt == np.float64(0.001)
    assert series.ts_e.shape == (2048,)
    assert series.ts_v.shape == (2048,)

    second = libloki.simulation.PulseSignalConfig(
        0.05,
        0.001,
        2048,
        15.0,
        0.1,
        "derivative",
        params,
        None,
        7,
    )
    repeat = second.generate("gaussian", 0.5)
    np.testing.assert_array_equal(series.ts_e, repeat.ts_e)
    nxt = second.generate("gaussian", 0.5)
    assert not np.array_equal(repeat.ts_e, nxt.ts_e)


def test_timeseries_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    intensity = rng.normal(size=128).astype(np.float32)
    variance = np.ones(128, dtype=np.float32)
    series = libloki.io.TimeSeries(intensity, variance, 6.4e-5)
    path = tmp_path / "toy.tim"
    series.write(str(path))

    options = libloki.io.ReadOptions()
    options.preprocess = False
    loaded = libloki.io.TimeSeries.read(str(path), options)
    np.testing.assert_array_equal(loaded.ts_e, intensity)
    assert loaded.nsamps == 128
    assert loaded.dt == np.float64(6.4e-5)
    assert np.all(loaded.ts_v == 1.0)
