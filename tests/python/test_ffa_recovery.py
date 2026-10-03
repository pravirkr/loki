"""End-to-end FFA recovery on injected pulsars (narrow search window)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from loki import libloki

NSAMPS = 2**23
PERIOD = 0.01
DT = 64e-6
TRUE_FREQ = 100.0
TRUE_ACCEL = 200.0
SEARCH_SNR_MIN = 6.0
# Injected folded S/N is 10. Recovery must stay at or above this floor.
RECOVER_SNR_MIN = 8.0
# One frequency step at this length is ~3e-5 Hz.
# One acceleration step is ~2.6 m/s^2.
FREQ_TOL = 1.0e-4
ACCEL_TOL = 5.0
SEED = 1


def _make_injected(accel_mps2: float) -> tuple[np.ndarray, np.ndarray]:
    mod = libloki.simulation.ModulatorParams()
    mod.acc = accel_mps2
    cfg = libloki.simulation.PulseSignalConfig(
        PERIOD,
        DT,
        NSAMPS,
        10.0,
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
    *,
    use_fourier: bool,
    search_accel: bool,
) -> libloki.configs.PulsarSearchConfig:
    if search_accel:
        param_limits = np.array([[180.0, 220.0], [99.95, 100.05]], dtype=np.float64)
    else:
        param_limits = np.array([[99.95, 100.05]], dtype=np.float64)
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=DT,
        nbins=64,
        eta=1.0,
        param_limits=param_limits,
        ducy_max=0.2,
        wtsp=1.5,
        use_fourier=use_fourier,
        nthreads=4,
        max_process_memory_gb=4.0,
        snr_min=SEARCH_SNR_MIN,
    )


def _best_recovered_snr(result_h5: Path, *, expect_accel: bool) -> float:
    if not result_h5.is_file():
        return 0.0
    with h5py.File(result_h5, "r") as handle:
        if "snr" not in handle or "param_sets" not in handle:
            return 0.0
        snrs = np.asarray(handle["snr"][:], dtype=np.float64)
        params = np.asarray(handle["param_sets"][:], dtype=np.float64)
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in handle.attrs.get("param_names", [])
        ]
        freq_col = names.index("freq") if "freq" in names else params.shape[1] - 1
        accel_col = names.index("accel") if "accel" in names else None
        best = 0.0
        for snr, row in zip(snrs, params, strict=True):
            if abs(row[freq_col] - TRUE_FREQ) > FREQ_TOL:
                continue
            if expect_accel:
                if accel_col is None or abs(row[accel_col] - TRUE_ACCEL) > ACCEL_TOL:
                    continue
            best = max(best, float(snr))
        return best


def test_ffa_freq_sweep_recovers_injected_pulsar(tmp_path: Path) -> None:
    """Reuse each injected series for both folding modes, one search at a time."""
    cases = (
        (0.0, False),
        (0.0, True),
        (TRUE_ACCEL, False),
        (TRUE_ACCEL, True),
    )
    series = {
        0.0: _make_injected(0.0),
        TRUE_ACCEL: _make_injected(TRUE_ACCEL),
    }
    for accel_mps2, use_fourier in cases:
        ts_e, ts_v = series[accel_mps2]
        search_accel = accel_mps2 != 0.0
        cfg = _search_config(use_fourier=use_fourier, search_accel=search_accel)
        prefix = f"recovery_{accel_mps2:g}_{'fourier' if use_fourier else 'time'}"
        sweep = libloki.ffa.FFAFreqSweep(cfg, show_progress=False)
        sweep.execute(ts_e, ts_v, str(tmp_path), prefix)
        best = _best_recovered_snr(
            tmp_path / f"{prefix}_ffa_results.h5",
            expect_accel=search_accel,
        )
        assert best >= RECOVER_SNR_MIN, (
            f"{prefix}: best recovered S/N {best:.2f} is below {RECOVER_SNR_MIN}"
        )
