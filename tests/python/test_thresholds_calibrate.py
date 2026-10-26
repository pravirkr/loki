"""Tests for DynamicThresholdScheme.calibrate (CPU and CUDA)."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

BACKENDS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            "cuda" not in libloki.available_backends(),
            reason="CUDA backend not built",
        ),
    ),
]
BRANCHING = np.full(8, 2.0, dtype=np.float32)
SCHEME_KW = {
    "ref_ducy": 0.1,
    "nbins": 32,
    "ntrials": 512,
    "nprobs": 10,
    "prob_min": 0.05,
    "snr_final": 8.0,
    "nthresholds": 40,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.0,
    "mode": "improved",
}


def _scheme(backend: str, seed: int):  # noqa: ANN202
    extra = {"nthreads": 4} if backend == "cpu" else {}
    return libloki.thresholds.DynamicThresholdScheme(
        BRANCHING, seed=seed, backend=backend, **extra, **SCHEME_KW
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_calibrate_hits_min_pd(backend: str) -> None:
    min_pd, ntrials_cal, ncheck = 0.3, 16384, 65536
    scheme = _scheme(backend, seed=7)
    scheme.run(6)
    path = scheme.get_best_path_thresholds(min_pd)
    res = scheme.calibrate(min_pd, ntrials_cal=ntrials_cal, seed=101)
    assert res["thresholds"].shape == (scheme.nstages,)
    # At least ceil(min_pd * N) survivors, exactly that many after a lowering
    kept = res["pd_certified"] * ntrials_cal
    assert kept >= np.ceil(min_pd * ntrials_cal)
    if res["delta"] < 0:
        assert kept == np.ceil(min_pd * ntrials_cal)
    assert res["states"].shape == (scheme.nstages,)
    # The search is untouched
    np.testing.assert_array_equal(scheme.get_best_path_thresholds(min_pd), path)
    # Independent rescore within 5 sigma of both samples
    sigma = np.sqrt(min_pd * (1 - min_pd) * (1 / ntrials_cal + 1 / ncheck))
    last = scheme.evaluate(res["thresholds"], ncheck, 202)[-1]
    assert abs(float(last["success_h1_cumul"]) - res["pd_certified"]) < 5 * sigma
    assert float(last["success_h1_cumul"]) > min_pd - 5 * sigma
    # Deterministic for a fixed seed
    again = scheme.calibrate(min_pd, ntrials_cal=ntrials_cal, seed=101)
    np.testing.assert_array_equal(again["thresholds"], res["thresholds"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_simulate_h1_scores_shape(backend: str) -> None:
    scheme = _scheme(backend, seed=5)
    scores = scheme.simulate_h1_scores(1024, 9)
    assert scores.shape == (scheme.nstages, 1024)
    assert np.all(np.diff(scores.mean(axis=1)) > 0)


def test_calibrate_rejects_bad_input() -> None:
    scheme = _scheme("cpu", seed=5)
    scheme.run(6)
    with pytest.raises(ValueError):  # noqa: PT011
        scheme.calibrate(0.0)
    with pytest.raises(ValueError):  # noqa: PT011
        scheme.calibrate(0.1, ntrials_cal=0)


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(),
    reason="CUDA backend not built",
)
def test_shared_mode_pd_is_honest() -> None:
    kw = {**SCHEME_KW, "mode": "shared"}
    scheme = libloki.thresholds.DynamicThresholdScheme(
        BRANCHING, seed=7, backend="cuda", **kw
    )
    scheme.run(6)
    min_pd = 0.3
    path = scheme.get_best_path_thresholds(min_pd)
    final = scheme.get_states().reshape(
        scheme.nstages, scheme.nthresholds, scheme.nprobs
    )[-1]
    ok = ~final["is_empty"] & (final["success_h1_cumul"] >= min_pd)
    pd_inrun = float(final["success_h1_cumul"][ok][np.argmin(final["cost"][ok])])
    last = scheme.evaluate(path, 65536, 99)[-1]
    sigma = np.sqrt(pd_inrun * (1 - pd_inrun) * (1 / (16 * 512) + 1 / 65536))
    assert abs(float(last["success_h1_cumul"]) - pd_inrun) < 5 * sigma
