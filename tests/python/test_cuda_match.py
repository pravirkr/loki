"""Test CUDA implementations match CPU implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pyloki.config import ParamLimits

import loki
from loki import libculoki

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("cpp_fn", "cuda_fn", "decimal"),
    [
        (
            loki.fold.compute_brute_fold_time,
            libculoki.fold.compute_brute_fold_time_cuda,
            5,
        ),
        (
            loki.fold.compute_brute_fold_fourier,
            libculoki.fold.compute_brute_fold_fourier_cuda,
            2,
        ),
    ],
)
def test_brute_fold_variants_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    cpp_fn: Callable,
    cuda_fn: Callable,
    decimal: int,
) -> None:
    """Test CUDA brute fold matches CPU version."""
    ts_e, ts_v = mock_data
    freqs_arr = np.linspace(140, 145, 50)
    segment_len = default_params["nsamps"] // 2**13
    t_ref = 0.0
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
    out_cuda = cuda_fn(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=t_ref,
        device_id=0,
    )
    np.testing.assert_array_almost_equal(out_cuda, out_cpp, decimal=decimal)


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_freq_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]
    cfg = loki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libculoki.ffa.compute_ffa_fourier_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )
    else:
        out_cpu, plan_cpu = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, plan_cuda = libculoki.ffa.compute_ffa_time_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )

    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_freq_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]
    cfg = loki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = loki.ffa.compute_ffa_fourier_return_to_time(
        ts_e, ts_v, cfg, quiet=True
    )
    out_cuda, _ = libculoki.ffa.compute_ffa_fourier_return_to_time_cuda(
        ts_e, ts_v, cfg, device_id=0, quiet=True
    )
    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=1)


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_accel_cuda_vs_cpu(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    """Test CUDA FFA matches CPU for acceleration search."""
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]
    cfg = loki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libculoki.ffa.compute_ffa_fourier_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )
    else:
        out_cpu, _ = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libculoki.ffa.compute_ffa_time_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )
    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_accel_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]
    cfg = loki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = loki.ffa.compute_ffa_fourier_return_to_time(
        ts_e, ts_v, cfg, quiet=True
    )
    out_cuda, _ = libculoki.ffa.compute_ffa_fourier_return_to_time_cuda(
        ts_e, ts_v, cfg, device_id=0, quiet=True
    )
    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=1)


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_jerk_cuda(
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
    cfg = loki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits.limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = loki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libculoki.ffa.compute_ffa_fourier_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )
    else:
        out_cpu, _ = loki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libculoki.ffa.compute_ffa_time_cuda(
            ts_e, ts_v, cfg, device_id=0, quiet=True
        )
    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_jerk_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = ParamLimits.from_upper(
        [-0.5, -50.0, 143.5],
        (-1, 1),
        default_params["nsamps"] * default_params["tsamp"],
    )
    cfg = loki.configs.PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = loki.ffa.compute_ffa_jerk_fourier_return_to_time(
        ts_e, ts_v, cfg, quiet=True
    )
    out_cuda, _ = libculoki.ffa.compute_ffa_jerk_fourier_return_to_time_cuda(
        ts_e, ts_v, cfg, device_id=0, quiet=True
    )
    np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=1)
