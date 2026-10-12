import numpy as np
import pytest

from loki import libloki

io = libloki.io
TSAMP = 1e-3


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n).astype(np.float32)


def test_robust_white_noise() -> None:
    raw = _noise(1 << 16) * 3.0 + 10.0
    ts_e, ts_v, rep = io.preprocess(raw, TSAMP, io.PreprocessOptions(), nthreads=2)
    assert ts_e.dtype == np.float32
    assert ts_e.shape == raw.shape
    assert ts_v.shape == raw.shape
    assert np.all(np.isfinite(ts_e))
    assert np.all(ts_v >= 0.0)
    assert rep.method == io.PreprocessMethod.Robust
    assert rep.n_masked == 0
    assert np.median(ts_v[ts_v > 0]) == pytest.approx(1.0, rel=0.05)
    assert rep.global_scale == pytest.approx(3.0, rel=0.05)
    assert rep.block_mu.size == rep.block_sigma.size
    assert rep.zero_weight_fraction < 1e-3


def test_variance_step_weights() -> None:
    raw = _noise(1 << 16)
    raw[1 << 15 :] *= 3.0
    _, ts_v, _ = io.preprocess(raw, TSAMP)
    lo = np.median(ts_v[: 1 << 14])
    hi = np.median(ts_v[-(1 << 14) :])
    assert hi / lo == pytest.approx(1.0 / 9.0, rel=0.15)


def test_burst_is_masked() -> None:
    raw = _noise(1 << 16)
    raw[20000:20500] += 5.0
    ts_e, ts_v, rep = io.preprocess(raw, TSAMP)
    assert rep.n_masked >= 500
    assert np.all(ts_v[20000:20500] == 0.0)
    assert np.all(ts_e[ts_v == 0.0] == 0.0)


def test_zscore_matches_legacy_read(tmp_path) -> None:
    raw = _noise(4096) + 2.0
    series = io.TimeSeries(raw, np.ones_like(raw), TSAMP)
    path = tmp_path / "toy.tim"
    series.write(str(path))

    opts = io.PreprocessOptions()
    opts.method = io.PreprocessMethod.ZScore
    opts.filter_window = 0.5
    read_opts = io.ReadOptions()
    read_opts.preprocessing = opts
    loaded = io.TimeSeries.read(str(path), read_opts)
    ts_e, ts_v, rep = io.preprocess(raw, TSAMP, opts)
    np.testing.assert_array_equal(loaded.ts_e, ts_e)
    assert np.all(ts_v == 1.0)
    assert rep.method == io.PreprocessMethod.ZScore


def test_timeseries_preprocess_in_place() -> None:
    raw = _noise(1 << 15)
    series = io.TimeSeries(raw, np.ones_like(raw), TSAMP)
    rep = series.preprocess(io.PreprocessOptions(), nthreads=2)
    ts_e, ts_v, _ = io.preprocess(raw, TSAMP, nthreads=2)
    np.testing.assert_array_equal(series.ts_e, ts_e)
    np.testing.assert_array_equal(series.ts_v, ts_v)
    assert rep.nsamps == raw.size


def test_options_and_birdies() -> None:
    opts = io.PreprocessOptions()
    opts.birdies = [io.Birdie(50.0, 0.1)]
    opts.gain_model = io.GainModel.Multiplicative
    assert opts.birdies[0].freq == 50.0
    opts.validate()
    opts.clip_sigma = -1.0
    with pytest.raises(ValueError):
        opts.validate()
    with pytest.raises(ValueError):
        io.preprocess(_noise(1024), TSAMP, opts)
