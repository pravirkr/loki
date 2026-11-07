"""End-to-end EP recovery: an injected pulsar survives the planned thresholds."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from loki import libloki

NSAMPS = 1 << 16
DT = 64e-6
TRUE_FREQ = 142.3
TRUE_ACCEL = 5.0
INJECTED_SNR = 15.0
# The final EP score of the injected pulsar must reach this floor.
RECOVER_SNR_MIN = 10.0
# Two frequency resolutions 1 / T_obs.
FREQ_TOL = 2.0 / (NSAMPS * DT)
SEED = 1


def _cuda_built() -> bool:
    return "cuda" in libloki.available_backends()


def _make_injected() -> tuple[np.ndarray, np.ndarray]:
    mod = libloki.simulation.ModulatorParams()
    mod.acc = TRUE_ACCEL
    cfg = libloki.simulation.PulseSignalConfig(
        1.0 / TRUE_FREQ,
        DT,
        NSAMPS,
        INJECTED_SNR,
        0.1,
        "derivative",
        mod,
        None,
        SEED,
    )
    series = cfg.generate("gaussian", 0.5)
    noise_std = float(np.sqrt(series.ts_v[0]))
    ts_e = np.asarray(series.ts_e, dtype=np.float32) / noise_std
    ts_v = np.ones_like(ts_e, dtype=np.float32)
    return ts_e, ts_v


def _search_config(
    param_limits: np.ndarray | None = None,
    **orbit: float,
) -> libloki.configs.PulsarSearchConfig:
    if param_limits is None:
        param_limits = np.array([[-10.0, 10.0], [140.0, 145.0]], dtype=np.float64)
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=DT,
        nbins=32,
        eta=1.0,
        param_limits=param_limits,
        ducy_max=0.3,
        use_fourier=False,
        nthreads=1,
        max_process_memory_gb=4.0,
        bseg_brute=1024,
        nsegments=8,
        prune_poly_order=2,
        **orbit,
    )


def _best_score_near_truth(result_h5: Path, nparams: int = 2) -> float:
    best = 0.0
    with h5py.File(result_h5, "r") as handle:
        for chunk in handle["chunks"].values():
            for run in chunk["runs"].values():
                scores = np.asarray(run["scores"][:], dtype=np.float64)
                # Leaves are (nparams + 2) rows of (value, error); row
                # nparams - 1 is the frequency.
                freqs = np.asarray(
                    run["param_sets"][:, nparams - 1, 0], dtype=np.float64
                )
                near = np.abs(freqs - TRUE_FREQ) <= FREQ_TOL
                if np.any(near):
                    best = max(best, float(scores[near].max()))
    return best


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ep_freq_sweep_recovers_injected_pulsar(tmp_path: Path, backend: str) -> None:
    """The plan's own thresholds keep a pulsar above target_snr to the end."""
    if backend == "cuda" and not _cuda_built():
        pytest.skip("CUDA backend not built")
    ts_e, ts_v = _make_injected()
    cfg = _search_config()
    cache = tmp_path / "plan.h5"
    sweep = libloki.prune.EPFreqSweep(
        cfg,
        min_pd=0.5,
        plan_cache_file=str(cache),
        show_progress=False,
        ref_segs=[4],
        backend=backend,
    )
    sweep.execute(ts_e, ts_v, str(tmp_path), "recovery")
    best = _best_score_near_truth(tmp_path / "recovery_ep_results.h5")
    assert best >= RECOVER_SNR_MIN, (
        f"{backend}: best score near the pulsar {best:.2f} is below {RECOVER_SNR_MIN}"
    )
    # The frozen plan (loaded from the cache) recovers it again.
    again = libloki.prune.EPFreqSweep(
        cfg,
        min_pd=0.5,
        plan_cache_file=str(cache),
        show_progress=False,
        ref_segs=[4],
        backend=backend,
    )
    again.execute(ts_e, ts_v, str(tmp_path), "recovery_cached")
    assert _best_score_near_truth(tmp_path / "recovery_cached_ep_results.h5") >= (
        RECOVER_SNR_MIN
    )


def test_ep_freq_sweep_recovers_with_five_parameters(tmp_path: Path) -> None:
    """A circular search plans and prunes by its orbit limits."""
    ts_e, ts_v = _make_injected()
    # Orbits of at least 1000 observations with a companion of at most
    # 0.05 M_sun: |accel| <= 38 m/s^2 covers the injected pulsar, and the
    # higher derivatives keep the tree small.
    orbit = {"p_orb_min": 1000.0 * NSAMPS * DT, "m_c_max": 0.05, "m_p_min": 1.2}
    limits = libloki.configs.circular_param_limits(140.0, 145.0, poly_order=5, **orbit)
    cfg = _search_config(limits, **orbit)
    sweep = libloki.prune.EPFreqSweep(
        cfg, min_pd=0.5, show_progress=False, ref_segs=[4]
    )
    sweep.execute(ts_e, ts_v, str(tmp_path), "five")
    best = _best_score_near_truth(tmp_path / "five_ep_results.h5", nparams=5)
    assert best >= RECOVER_SNR_MIN, (
        f"best score near the pulsar {best:.2f} is below {RECOVER_SNR_MIN}"
    )


@pytest.mark.parametrize("poly_basis", ["taylor", "chebyshev"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ep_freq_sweep_recovers_non_power_of_two_nsamps(
    tmp_path: Path, backend: str, poly_basis: str
) -> None:
    """Non-power-of-2 nsamps: EP takes the full series and recovers the pulsar.

    EP keeps whole segments only: 100000 samples are 64 brute segments of
    1562, so it analyses the first 99968 (8 segments of 12496) and reports
    at their midpoint. Both polynomial bases must recover the injected
    parameters there.
    """
    if backend == "cuda" and not _cuda_built():
        pytest.skip("CUDA backend not built")
    nsamps_np2 = 100_000
    true_accel = 400.0
    mod = libloki.simulation.ModulatorParams()
    mod.acc = true_accel
    sig_cfg = libloki.simulation.PulseSignalConfig(
        1.0 / TRUE_FREQ,
        DT,
        nsamps_np2,
        INJECTED_SNR,
        0.1,
        "derivative",
        mod,
        None,
        SEED,
    )
    series = sig_cfg.generate("gaussian", 0.5)
    noise_std = float(np.sqrt(series.ts_v[0]))
    ts_e = np.asarray(series.ts_e, dtype=np.float32) / noise_std
    ts_v = np.ones_like(ts_e, dtype=np.float32)

    param_limits = np.array([[385.0, 415.0], [142.0, 142.6]], dtype=np.float64)
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=nsamps_np2,
        tsamp=DT,
        nbins=32,
        eta=1.0,
        param_limits=param_limits,
        ducy_max=0.3,
        use_fourier=False,
        nthreads=1,
        max_process_memory_gb=4.0,
        bseg_brute=1024,
        nsegments=8,
        prune_poly_order=2,
    )
    sweep = libloki.prune.EPFreqSweep(
        cfg,
        min_pd=0.5,
        poly_basis=poly_basis,
        show_progress=False,
        ref_segs=[4],
        backend=backend,
    )
    prefix = f"np2_{backend}_{poly_basis}"
    assert cfg.nsamps_input == nsamps_np2
    assert cfg.nsamps == 99_968
    assert cfg.bseg_ffa == 12_496
    sweep.execute(ts_e, ts_v, str(tmp_path), prefix)
    result_h5 = tmp_path / f"{prefix}_ep_results.h5"

    # The analysed midpoint is 16 samples (~1 ms) before the input midpoint,
    # a frequency offset of ~2e-7 Hz at a=400 m/s^2, far below 1 / T_obs.
    freq_tol_np2 = 1.0 / (nsamps_np2 * DT)
    best = 0.0
    with h5py.File(result_h5, "r") as handle:
        for chunk in handle["chunks"].values():
            for run in chunk["runs"].values():
                scores = np.asarray(run["scores"][:], dtype=np.float64)
                accels = np.asarray(run["param_sets"][:, 0, 0], dtype=np.float64)
                freqs = np.asarray(run["param_sets"][:, 1, 0], dtype=np.float64)
                near = (np.abs(freqs - TRUE_FREQ) <= freq_tol_np2) & (
                    np.abs(accels - true_accel) <= 10.0
                )
                if np.any(near):
                    best = max(best, float(scores[near].max()))
    assert best >= RECOVER_SNR_MIN, (
        f"{backend}/{poly_basis}: best score near the injection {best:.2f} is below "
        f"{RECOVER_SNR_MIN}"
    )
