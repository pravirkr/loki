"""Test CUDA implementations match CPU implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from loki import libloki
from pyloki.config import ParamLimits

if "cuda" not in libloki.available_backends():
    pytest.skip("CUDA backend not built", allow_module_level=True)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("kind", "decimal"),
    [("time", 5), ("fourier", 1)],
)
def test_brute_fold_variants_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    kind: str,
    decimal: int,
) -> None:
    """Test CUDA brute fold matches CPU version."""
    ts_e, ts_v = mock_data
    freqs_arr = np.linspace(140, 145, 50)
    segment_len = default_params["nsamps"] // 2**13
    t_ref = 0.0
    fold_fn: Callable = (
        libloki.fold.compute_brute_fold_time
        if kind == "time"
        else libloki.fold.compute_brute_fold_fourier
    )
    out_cpp = fold_fn(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=t_ref,
        nthreads=8,
    )
    out_cuda = fold_fn(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=default_params["nbins"],
        tsamp=default_params["tsamp"],
        t_ref=t_ref,
        backend="cuda",
        device=0,
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
    cfg = libloki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_fourier(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_allclose(out_cuda, out_cpu, atol=5.0)
    else:
        out_cpu, _ = libloki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_time(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_freq_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(143.0, 144.0)]
    cfg = libloki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
    )
    out_cuda, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
        backend="cuda",
        device=0,
    )
    np.testing.assert_allclose(out_cuda, out_cpu, rtol=0.05, atol=1.0)


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
    cfg = libloki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_fourier(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_allclose(out_cuda, out_cpu, atol=5.0)
    else:
        out_cpu, _ = libloki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_time(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_accel_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = [(-50.0, 50.0), (143.5, 144.0)]
    cfg = libloki.configs.PulsarSearchConfig(
        eta=2,
        param_limits=param_limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
    )
    out_cuda, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
        backend="cuda",
        device=0,
    )
    np.testing.assert_allclose(out_cuda, out_cpu, rtol=0.05, atol=1.0)


@pytest.mark.parametrize(("use_fourier"), [False, True])
def test_ffa_jerk_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    ts_e, ts_v = mock_data
    param_limits = ParamLimits.from_upper(
        143.5,
        [-0.5, -50.0],
        (-1, 1),
        default_params["nsamps"] * default_params["tsamp"],
    )
    cfg = libloki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits.limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    if use_fourier:
        out_cpu, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_fourier(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_allclose(out_cuda, out_cpu, atol=5.0)
    else:
        out_cpu, _ = libloki.ffa.compute_ffa_time(ts_e, ts_v, cfg, quiet=True)
        out_cuda, _ = libloki.ffa.compute_ffa_time(
            ts_e,
            ts_v,
            cfg,
            quiet=True,
            backend="cuda",
            device=0,
        )
        np.testing.assert_array_almost_equal(out_cuda, out_cpu, decimal=3)


def test_ffa_jerk_fourier_return_to_time_cuda(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
) -> None:
    ts_e, ts_v = mock_data
    param_limits = ParamLimits.from_upper(
        143.5,
        [-0.5, -50.0],
        (-1, 1),
        default_params["nsamps"] * default_params["tsamp"],
    )
    cfg = libloki.configs.PulsarSearchConfig(
        eta=4,
        param_limits=param_limits.limits,
        use_fourier=True,
        nthreads=8,
        **default_params,
    )
    out_cpu, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
    )
    out_cuda, _ = libloki.ffa.compute_ffa_fourier_return_to_time(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
        backend="cuda",
        device=0,
    )
    np.testing.assert_allclose(out_cuda, out_cpu, atol=1.0)


@pytest.mark.parametrize("use_fourier", [False, True])
def test_ffa_scores_cuda_vs_cpu(
    mock_data: tuple[np.ndarray, np.ndarray],
    default_params: dict[str, Any],
    *,
    use_fourier: bool,
) -> None:
    """Test CUDA compute_ffa_scores matches CPU."""
    ts_e, _ = mock_data
    ts_v = np.ones_like(ts_e)
    param_limits = [(143.0, 144.0)]
    cfg = libloki.configs.PulsarSearchConfig(
        eta=1,
        param_limits=param_limits,
        use_fourier=use_fourier,
        nthreads=8,
        **default_params,
    )
    scores_cpu, _ = libloki.ffa.compute_ffa_scores(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
        backend="cpu",
    )
    scores_cuda, _ = libloki.ffa.compute_ffa_scores(
        ts_e,
        ts_v,
        cfg,
        quiet=True,
        backend="cuda",
        device=0,
    )
    # The GPU validation measured < 1e-4 absolute difference (time and Fourier).
    np.testing.assert_allclose(scores_cuda, scores_cpu, rtol=1e-3, atol=1e-2)


def test_snr_boxcar_2d_cuda_vs_cpu() -> None:
    """Test 2D boxcar scoring CPU vs CUDA parity."""
    np.random.seed(42)
    folds = np.random.randn(32, 64).astype(np.float32)
    widths = np.array([1, 2, 4, 8], dtype=np.uint64)

    # Per-width
    scores_cpu = libloki.scores.snr_boxcar_2d(folds, widths, backend="cpu")
    scores_cuda = libloki.scores.snr_boxcar_2d(folds, widths, backend="cuda", device=0)
    np.testing.assert_allclose(scores_cuda, scores_cpu, rtol=1e-4, atol=1e-4)

    # Max
    max_cpu = libloki.scores.snr_boxcar_2d_max(folds, widths, backend="cpu")
    max_cuda = libloki.scores.snr_boxcar_2d_max(folds, widths, backend="cuda", device=0)
    np.testing.assert_allclose(max_cuda, max_cpu, rtol=1e-4, atol=1e-4)


def test_snr_boxcar_3d_cuda_vs_cpu() -> None:
    """Test 3D boxcar scoring CPU vs CUDA parity."""
    np.random.seed(42)
    # 3D folds shape: (nprofiles, 2, nbins) where channel 0 is signal, channel 1 is positive variance
    signal = np.random.randn(16, 64).astype(np.float32)
    variance = np.ones((16, 64), dtype=np.float32)
    folds = np.stack([signal, variance], axis=1)
    widths = np.array([1, 2, 4, 8], dtype=np.uint64)

    # Per-width
    scores_cpu = libloki.scores.snr_boxcar_3d(folds, widths, backend="cpu")
    scores_cuda = libloki.scores.snr_boxcar_3d(folds, widths, backend="cuda", device=0)
    np.testing.assert_allclose(scores_cuda, scores_cpu, rtol=1e-4, atol=1e-4)

    # Max
    max_cpu = libloki.scores.snr_boxcar_3d_max(folds, widths, backend="cpu")
    max_cuda = libloki.scores.snr_boxcar_3d_max(folds, widths, backend="cuda", device=0)
    np.testing.assert_allclose(max_cuda, max_cpu, rtol=1e-4, atol=1e-4)
