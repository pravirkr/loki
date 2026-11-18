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


def test_read_info_observation(tmp_path) -> None:
    raw = _noise(4096)
    path = tmp_path / "info.tim"
    io.TimeSeries(raw, np.ones_like(raw), TSAMP).write(str(path))
    info = io.TimeSeries.read_info(str(path))
    assert info.nsamps == raw.size
    read_opts = io.ReadOptions()
    read_opts.preprocess = False
    obs = io.TimeSeries.read(str(path), read_opts).observation
    assert info.observation.input_path == obs.input_path == str(path)
    assert info.observation.original_nsamps == raw.size
    assert info.observation.dm == obs.dm


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


def test_zap_modes_and_self_check() -> None:
    raw = _noise(1 << 18, 3)
    opts = io.PreprocessOptions()
    assert opts.zap_mode == io.ZapMode.Noise
    assert opts.zap_seed is None
    opts.birdies = [io.Birdie(37.0, 2.0)]
    ts_e, _, rep = io.preprocess(raw, TSAMP, opts, nthreads=2)
    assert rep.n_zapped > 0
    (band,) = rep.zap_bands
    assert (band.f_lo, band.f_hi) == (20.0, 50.0)
    assert band.n_zapped > 0
    assert abs(band.power_ratio - 1.0) < 0.25
    opts.zap_seed = 5
    seeded, _, _ = io.preprocess(raw, TSAMP, opts, nthreads=2)
    assert not np.array_equal(seeded, ts_e)
    opts.zap_mode = io.ZapMode.Zero
    _, _, rep = io.preprocess(raw, TSAMP, opts, nthreads=2)
    assert rep.zap_bands[0].power_ratio < 0.1


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(), reason="CUDA backend not built"
)
def test_cuda_matches_cpu() -> None:
    raw = _noise((1 << 17) + 333, seed=5)
    raw[40000:40400] += 4.0
    raw[90000:90200] *= 6.0
    cpu = io.preprocess(raw, TSAMP)
    gpu = io.preprocess(raw, TSAMP, backend="cuda")
    np.testing.assert_array_equal(cpu[1] == 0.0, gpu[1] == 0.0)
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-5, atol=1e-6)
    assert gpu[2].n_masked == cpu[2].n_masked > 0
    np.testing.assert_array_equal(gpu[2].block_mu, cpu[2].block_mu)

    series = io.TimeSeries(raw, np.ones_like(raw), TSAMP)
    series.preprocess(io.PreprocessOptions(), backend="cuda")
    np.testing.assert_array_equal(series.ts_e, gpu[0])


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(), reason="CUDA backend not built"
)
def test_cuda_zap_matches_cpu() -> None:
    n = 1 << 17
    t = np.arange(n) * TSAMP
    raw = _noise(n, seed=6) + 0.3 * np.sin(2 * np.pi * 123.4 * t).astype(np.float32)
    raw[50000:53000] *= 6.0
    opts = io.PreprocessOptions()
    opts.zap_periodic = True
    opts.birdies = [io.Birdie(321.0, 0.5)]
    cpu = io.preprocess(raw, TSAMP, opts, nthreads=2)
    gpu = io.preprocess(raw, TSAMP, opts, backend="cuda")
    # cuFFT and pocketfft round differently: float-level agreement.
    assert gpu[2].n_zapped == cpu[2].n_zapped > 0
    assert gpu[2].n_masked == cpu[2].n_masked > 0
    np.testing.assert_array_equal(cpu[1] == 0.0, gpu[1] == 0.0)
    err = np.sqrt(np.mean((gpu[0] - cpu[0]) ** 2) / np.mean(cpu[0] ** 2))
    assert err < 1e-4
    assert len(gpu[2].zap_bands) == len(cpu[2].zap_bands)
    for b_cpu, b_gpu in zip(cpu[2].zap_bands, gpu[2].zap_bands, strict=True):
        assert b_gpu.n_zapped == b_cpu.n_zapped
        assert b_gpu.power_ratio == pytest.approx(b_cpu.power_ratio, abs=2e-3)
