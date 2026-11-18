from pathlib import Path

import h5py
import numpy as np
import pytest

from loki import libloki

io = libloki.io
TSAMP = 64e-6
BACKENDS = ["cpu"] + (["cuda"] if "cuda" in libloki.available_backends() else [])


def _pulsar(n: int, f0: float, amp: float, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    phase = np.mod(np.arange(n) * TSAMP * f0, 1.0)
    x[phase < 0.05] += amp
    return x


@pytest.mark.parametrize("backend", BACKENDS)
def test_preprocess_matches_io_preprocess(backend: str) -> None:
    raw = _pulsar(1 << 17, 37.0, 0.3, seed=1)
    opts = io.PreprocessOptions()
    opts.zap_periodic = True
    ts_e, ts_v, rep = io.preprocess(raw, TSAMP, opts, nthreads=2, backend=backend)
    series, srep = io.SearchSeries.preprocess(
        raw, TSAMP, opts, nthreads=2, backend=backend
    )
    assert len(series) == raw.size
    assert series.tsamp == TSAMP
    assert series.backend == backend
    assert srep.n_zapped == rep.n_zapped
    assert srep.n_masked == rep.n_masked
    e, v = series.to_host()
    # The same kernels on the same backend: bit for bit.
    np.testing.assert_array_equal(e, ts_e)
    np.testing.assert_array_equal(v, ts_v)


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_host_round_trip(backend: str) -> None:
    e = _pulsar(4096, 50.0, 0.0, seed=2)
    v = np.linspace(0.5, 2.0, e.size, dtype=np.float32)
    series = io.SearchSeries.from_host(e, v, TSAMP, backend=backend)
    he, hv = series.to_host()
    np.testing.assert_array_equal(he, e)
    np.testing.assert_array_equal(hv, v)
    with pytest.raises(ValueError):
        io.SearchSeries.from_host(e, v[:-1], TSAMP, backend=backend)


def test_ffa_sweep_searches_a_series(tmp_path: Path) -> None:
    n = 1 << 15
    raw = _pulsar(n, 123.0, 1.0, seed=3)
    series, _ = io.SearchSeries.preprocess(raw, TSAMP)
    ts_e, ts_v = series.to_host()
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=n,
        tsamp=TSAMP,
        nbins=64,
        eta=1.0,
        param_limits=np.array([[100.0, 150.0]], dtype=np.float64),
        ducy_max=0.2,
        wtsp=1.5,
        use_fourier=True,
        nthreads=2,
        max_process_memory_gb=4.0,
        snr_min=3.0,
    )
    sweep = libloki.ffa.FFAFreqSweep(cfg, show_progress=False)
    sweep.execute(ts_e, ts_v, str(tmp_path), "spans")
    sweep.execute(series, str(tmp_path), "series")
    with (
        h5py.File(tmp_path / "spans_ffa_results.h5", "r") as a,
        h5py.File(tmp_path / "series_ffa_results.h5", "r") as b,
    ):
        snr_a = a["detections/snr"][()]
        assert snr_a.size > 0
        np.testing.assert_array_equal(snr_a, b["detections/snr"][()])
        np.testing.assert_array_equal(
            a["detections/param_sets"][()], b["detections/param_sets"][()]
        )
