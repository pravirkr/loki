"""Test CPU implementations match Python implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.core import fold
from pyloki.ffa import compute_ffa
from pyloki.io.timeseries import TimeSeries

import loki

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("py_fn", "cpp_fn", "decimal"),
    [
        (fold.brutefold_start, loki.fold.compute_brute_fold_time, 5),
        (fold.brutefold_start_complex, loki.fold.compute_brute_fold_fourier, 2),
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
    freqs_arr = np.linspace(140, 145, 50)
    segment_len = default_params["nsamps"] // 2**13
    t_ref = 0.0
    out_py = py_fn(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=t_ref,
    )
    out_cpp = cpp_fn(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=t_ref,
        nthreads=8,
    )
    np.testing.assert_array_almost_equal(
        out_cpp.reshape(out_py.shape),
        out_py,
        decimal=decimal,
    )


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_freq(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]
    cfg_py = PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=use_fourier,
        **default_params,
    )
    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)

    if use_fourier:
        out_cpp, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_allclose(
            out_cpp.reshape(out_py.shape),
            out_py,
            rtol=0.05,
            atol=2.0,
        )
    else:
        out_cpp, _ = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(out_py.shape),
            out_py,
            decimal=3,
        )


def test_ffa_freq_fourier_return_to_time(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]
    cfg_py = PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=True,
        **default_params,
    )
    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)
    out_py_time = np.fft.irfft(out_py).astype(np.float32)
    out_cpp, _ = loki.ffa.compute_ffa_fourier_return_to_time(
        ts_e, ts_v, cfg_cpp, quiet=True
    )
    np.testing.assert_array_almost_equal(
        out_cpp.reshape(out_py_time.shape),
        out_py_time,
        decimal=1,
    )


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_accel(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]

    cfg_py = PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=use_fourier,
        **default_params,
    )
    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)

    if use_fourier:
        out_cpp, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_allclose(
            out_cpp.reshape(out_py.shape),
            out_py,
            rtol=0.05,
            atol=2.0,
        )
    else:
        out_cpp, _ = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(out_py.shape),
            out_py,
            decimal=3,
        )


def test_ffa_accel_fourier_return_to_time(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]

    cfg_py = PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=True,
        **default_params,
    )
    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)
    out_py_time = np.fft.irfft(out_py).astype(np.float32)
    out_cpp, _ = loki.ffa.compute_ffa_fourier_return_to_time(
        ts_e, ts_v, cfg_cpp, quiet=True
    )
    np.testing.assert_array_almost_equal(
        out_cpp.reshape(out_py_time.shape),
        out_py_time,
        decimal=1,
    )


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_jerk(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = ParamLimits.from_upper(
        [-0.5, -50.0, 143.5],
        (-1, 1),
        default_params["nsamps"] * default_params["tsamp"],
    )

    cfg_py = PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=use_fourier,
        **default_params,
    )

    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)

    if use_fourier:
        out_cpp, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_allclose(
            out_cpp.reshape(out_py.shape),
            out_py,
            rtol=0.05,
            atol=2.0,
        )
    else:
        out_cpp, _ = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg_cpp, quiet=True)
        np.testing.assert_array_almost_equal(
            out_cpp.reshape(out_py.shape),
            out_py,
            decimal=3,
        )


def test_ffa_jerk_fourier_return_to_time(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = ParamLimits.from_upper(
        [-0.5, -50.0, 143.5],
        (-1, 1),
        default_params["nsamps"] * default_params["tsamp"],
    )

    cfg_py = PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=True,
        **default_params,
    )

    cfg_cpp = loki.configs.PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_py = compute_ffa(TimeSeries(ts_e, ts_v, cfg_py.tsamp), cfg_py, quiet=True)
    out_py_time = np.fft.irfft(out_py).astype(np.float32)
    out_cpp, _ = loki.ffa.compute_ffa_fourier_return_to_time(
        ts_e, ts_v, cfg_cpp, quiet=True
    )
    np.testing.assert_array_almost_equal(
        out_cpp.reshape(out_py_time.shape),
        out_py_time,
        decimal=1,
    )
