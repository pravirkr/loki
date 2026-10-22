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


def _search_config() -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=DT,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [140.0, 145.0]], dtype=np.float64),
        ducy_max=0.3,
        use_fourier=False,
        nthreads=1,
        max_process_memory_gb=4.0,
        bseg_brute=1024,
        bseg_ffa=NSAMPS // 8,
        prune_poly_order=2,
    )


def _best_score_near_truth(result_h5: Path) -> float:
    best = 0.0
    with h5py.File(result_h5, "r") as handle:
        for chunk in handle["chunks"].values():
            for run in chunk["runs"].values():
                scores = np.asarray(run["scores"][:], dtype=np.float64)
                # Leaves are (nparams + 2) rows of (value, error); row 1 is
                # the frequency of a two-parameter search.
                freqs = np.asarray(run["param_sets"][:, 1, 0], dtype=np.float64)
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
