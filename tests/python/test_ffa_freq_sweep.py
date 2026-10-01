"""Test the FFA frequency sweep matches the FFA scores of its chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from loki import libloki

if TYPE_CHECKING:
    from pathlib import Path

TSAMP = 0.000064
NSAMPS = 2**20
F_INJ = 31.7  # In a chunk with more bins and widths than the first
SNR_MIN = 4.0  # No score within 1e-4, so rounding can't move rows across


@pytest.fixture
def pulsar_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    phase = (np.arange(NSAMPS) * TSAMP * F_INJ) % 1.0
    ts_e = rng.normal(size=NSAMPS) + 0.1 * (phase < 0.05)
    return ts_e.astype(np.float32), np.ones(NSAMPS, dtype=np.float32)


@pytest.fixture
def sweep_params() -> dict[str, Any]:
    return {
        "nsamps": NSAMPS,
        "tsamp": TSAMP,
        "nbins": 16,
        "eta": 1,
        "param_limits": np.array([[20.0, 400.0]], dtype=np.float64),
        "octave_scale": 2.0,
        "nbins_max": 32,  # The last chunk is capped and holds the most scores
        "nthreads": 8,
    }


def chunk_rows(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    cfgs: list[libloki.configs.PulsarSearchConfig],
) -> np.ndarray:
    """Rows (freq, width, snr) of compute_ffa_scores on each chunk, >= SNR_MIN."""
    rows = []
    for c in cfgs:
        scores, plan = libloki.ffa.compute_ffa_scores(ts_e, ts_v, c, quiet=True)
        widths = np.asarray(c.score_widths)
        scores = scores.reshape(-1, len(widths))
        freqs = np.asarray(plan.params_dict["freq"])
        i_freq, i_width = np.nonzero(scores >= SNR_MIN)
        rows.append(
            np.column_stack([freqs[i_freq], widths[i_width], scores[i_freq, i_width]]),
        )
    return np.concatenate(rows)


def sweep_rows(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    cfg: libloki.configs.PulsarSearchConfig,
    outdir: Path,
) -> np.ndarray:
    """Rows (freq, width, snr) written by FFAFreqSweep."""
    sweep = libloki.ffa.FFAFreqSweep(cfg, show_progress=False)
    sweep.execute(ts_e, ts_v, outdir=str(outdir), file_prefix="test")
    with h5py.File(outdir / "test_ffa_results.h5", "r") as f:
        return np.column_stack([f["param_sets"][:], f["snr"][:]])


def test_ffa_freq_sweep(
    pulsar_data: tuple[np.ndarray, np.ndarray],
    sweep_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    ts_e, ts_v = pulsar_data
    cfg = libloki.configs.PulsarSearchConfig(snr_min=SNR_MIN, **sweep_params)
    cfgs = libloki.plans.FFARegionPlannerFourier(cfg).cfgs
    n_widths = [len(c.score_widths) for c in cfgs]
    np.testing.assert_array_less(n_widths[0], max(n_widths))
    expected = chunk_rows(ts_e, ts_v, cfgs)
    out = sweep_rows(ts_e, ts_v, cfg, tmp_path)
    np.testing.assert_allclose(out[:, 0], expected[:, 0], rtol=1e-9)
    np.testing.assert_array_equal(out[:, 1], expected[:, 1])
    # Folded in place by the sweep, returned to time by compute_ffa_scores
    np.testing.assert_allclose(out[:, 2], expected[:, 2], rtol=1e-5)
    best_freq = out[np.argmax(out[:, 2]), 0]
    np.testing.assert_allclose(best_freq, F_INJ, atol=1 / (NSAMPS * TSAMP))


def test_ffa_freq_sweep_buffer(
    pulsar_data: tuple[np.ndarray, np.ndarray],
    sweep_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    ts_e, ts_v = pulsar_data
    cfg = libloki.configs.PulsarSearchConfig(snr_min=SNR_MIN, **sweep_params)
    planner = libloki.plans.FFARegionPlannerFourier(cfg)
    expected = chunk_rows(ts_e, ts_v, planner.cfgs)
    # Room for exactly the passing candidates on top of the largest chunk
    cfg = libloki.configs.PulsarSearchConfig(
        snr_min=SNR_MIN,
        max_passing_candidates=len(expected),
        **sweep_params,
    )
    out = sweep_rows(ts_e, ts_v, cfg, tmp_path)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    nscores = [
        libloki.ffa.compute_ffa_scores(ts_e, ts_v, c, quiet=True)[0].size
        for c in planner.cfgs
    ]
    np.testing.assert_equal(planner.stats.max_nscores, max(nscores))


def test_ffa_freq_sweep_too_many_candidates(
    pulsar_data: tuple[np.ndarray, np.ndarray],
    sweep_params: dict[str, Any],
    tmp_path: Path,
) -> None:
    ts_e, ts_v = pulsar_data
    cfg = libloki.configs.PulsarSearchConfig(
        snr_min=0.0,
        max_passing_candidates=1000,
        **sweep_params,
    )
    sweep = libloki.ffa.FFAFreqSweep(cfg, show_progress=False)
    with pytest.raises(RuntimeError, match="max_passing_candidates"):
        sweep.execute(ts_e, ts_v, outdir=str(tmp_path), file_prefix="test")
