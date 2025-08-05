from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pyloki.config import PulsarSearchConfig
from pyloki.core import common
from pyloki.ffa import compute_ffa
from pyloki.io.timeseries import TimeSeries

import loki

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def mock_data() -> tuple[np.ndarray, np.ndarray]:
    nsamps = 2**23
    rng = np.random.default_rng()
    ts_e = rng.normal(size=nsamps).astype(np.float32)
    ts_v = rng.normal(size=nsamps).astype(np.float32)
    return ts_e, ts_v


@pytest.fixture
def default_params() -> dict[str, Any]:
    nsamps = 2**23
    return {
        "segment_len": nsamps // 2**13,
        "nbins": 64,
        "tsamp": 0.000064,
        "t_ref": 0.0,
        "nthreads": 8,
        "freqs_arr": np.linspace(140, 145, 50),
        "nsamps": nsamps,
    }


@pytest.mark.parametrize(
    ("py_fn", "cpp_fn", "decimal"),
    [
        (common.brutefold_start, loki.fold.compute_brute_fold, 5),
        (common.brutefold_start_complex, loki.fold.compute_brute_fold_complex, 2),
    ],
)
def test_brute_fold_variants(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    py_fn: Callable,
    cpp_fn: Callable,
    decimal: int,
) -> None:
    ts_e, ts_v = mock_data
    out_py = py_fn(
        ts_e,
        ts_v,
        default_params["freqs_arr"],
        segment_len=default_params["segment_len"],
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=default_params["t_ref"],
    )
    out_cpp = cpp_fn(
        ts_e,
        ts_v,
        default_params["freqs_arr"],
        segment_len=default_params["segment_len"],
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=default_params["t_ref"],
        nthreads=default_params["nthreads"],
    )
    np.testing.assert_array_almost_equal(
        out_cpp.reshape(out_py.shape),
        out_py,
        decimal=decimal,
    )


@pytest.mark.parametrize(("use_fft_shifts", "decimal"), [(False, 3), (True, 1)])
def test_ffa_freq(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fft_shifts: bool,
    decimal: int,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]

    cfg_py = PulsarSearchConfig(
        nsamps=default_params["nsamps"],
        tsamp=default_params["tsamp"],
        nbins=default_params["nbins"],
        tol_bins=1,
        param_limits=param_limits,
        bseg_brute=1024,
        bseg_ffa=default_params["nsamps"],
        use_fft_shifts=use_fft_shifts,
    )

    cfg_cpp = loki.configs.PulsarSearchConfig(
        nsamps=default_params["nsamps"],
        tsamp=default_params["tsamp"],
        nbins=default_params["nbins"],
        tol_bins=1,
        param_limits=param_limits,
        bseg_brute=1024,
        bseg_ffa=default_params["nsamps"],
        use_fft_shifts=use_fft_shifts,
        nthreads=default_params["nthreads"],
    )

    cpp_fn = (
        loki.ffa.compute_ffa if not use_fft_shifts else loki.ffa.compute_ffa_complex
    )

    out_cpp, _ = cpp_fn(ts_e, ts_v, cfg_cpp, quiet=True)
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)

    if use_fft_shifts:
        shape = (*out_py.shape[:-1], (out_py.shape[-1] - 1) * 2)
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(shape),
            np.fft.irfft(out_py),
            decimal=decimal,
        )
    else:
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(out_py.shape),
            out_py,
            decimal=decimal,
        )


@pytest.mark.parametrize(("use_fft_shifts", "decimal"), [(False, 3), (True, 1)])
def test_ffa_accel(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fft_shifts: bool,
    decimal: int,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]

    cfg_py = PulsarSearchConfig(
        nsamps=default_params["nsamps"],
        tsamp=default_params["tsamp"],
        nbins=default_params["nbins"],
        tol_bins=2,
        param_limits=param_limits,
        bseg_brute=1024,
        bseg_ffa=default_params["nsamps"],
        use_fft_shifts=use_fft_shifts,
    )

    cfg_cpp = loki.configs.PulsarSearchConfig(
        nsamps=default_params["nsamps"],
        tsamp=default_params["tsamp"],
        nbins=default_params["nbins"],
        tol_bins=2,
        param_limits=param_limits,
        bseg_brute=1024,
        bseg_ffa=default_params["nsamps"],
        use_fft_shifts=use_fft_shifts,
        nthreads=default_params["nthreads"],
    )

    cpp_fn = (
        loki.ffa.compute_ffa if not use_fft_shifts else loki.ffa.compute_ffa_complex
    )

    out_cpp, _ = cpp_fn(ts_e, ts_v, cfg_cpp, quiet=True)
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)

    if use_fft_shifts:
        shape = (*out_py.shape[:-1], (out_py.shape[-1] - 1) * 2)
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(shape),
            np.fft.irfft(out_py),
            decimal=decimal,
        )
    else:
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(out_py.shape),
            out_py,
            decimal=decimal,
        )
