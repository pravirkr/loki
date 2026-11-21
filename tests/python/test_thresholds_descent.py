"""Tests for DynamicThresholdScheme mode "descent" (CPU and CUDA)."""

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
BRANCHING = np.array([2, 3, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2], dtype=np.float32)
MIN_PD = 0.3
KW = {"ref_ducy": 0.1, "nbins": 32, "snr_final": 8.0, "ducy_max": 0.3, "wtsp": 1.5}
# Small samples keep the tests fast; the defaults are larger.
PASSES = [
    (1024, 0.4, 0.2, 1 << 20, 1),
    (2048, 0.3, 0.1, 1 << 20, 1),
    (2048, 0.2, 0.05, 4, 1),
]


def _scheme(backend: str, seed: int = 7, mode: str = "descent", **opts):
    extra = (
        {"descent_passes": PASSES, "descent_nsignal": 8192} if mode == "descent" else {}
    )
    return libloki.thresholds.DynamicThresholdScheme(
        BRANCHING,
        mode=mode,
        seed=seed,
        backend=backend,
        nthreads=2,
        min_pd=MIN_PD,
        **{**extra, **opts},
        **KW,
    )


def _judge(backend: str, thresholds: np.ndarray, seed: int = 991):
    """Pd by a direct count of fresh signal trials, cost on fresh noise."""
    judge = _scheme(backend, seed=seed)
    scores = judge.simulate_h1_scores(1 << 15, seed)
    pd = float(np.mean(np.all(scores > thresholds[:, None], axis=0)))
    last = judge.evaluate(thresholds, 1 << 14, seed + 1)[-1]
    return pd, float(last["complexity_cumul"]) + float(last["complexity"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_delivers_the_target_pd(backend: str) -> None:
    scheme = _scheme(backend)
    scheme.run()  # nothing to search in this mode
    sol = scheme.solve()
    thresholds = sol["thresholds"]
    assert thresholds.shape == (scheme.nstages,)
    assert sol["delivery"] in ("met", "within_tolerance")
    assert sol["pd_trials"] == 1 << 17
    # pinned at the target: no overshoot beyond the sampling error
    assert 0.97 * MIN_PD <= sol["pd"] <= 1.05 * MIN_PD
    pd, cost = _judge(backend, thresholds)
    assert pd == pytest.approx(sol["pd"], rel=0.05)
    assert cost == pytest.approx(sol["c"], rel=0.1)
    assert sol["diagnostics"]["descent_moves"] > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_is_deterministic_per_seed(backend: str) -> None:
    first = _scheme(backend, seed=11).solve()
    again = _scheme(backend, seed=11).solve()
    other = _scheme(backend, seed=12).solve()
    np.testing.assert_array_equal(first["thresholds"], again["thresholds"])
    assert first["pd"] == again["pd"]
    assert not np.array_equal(first["thresholds"], other["thresholds"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_is_not_costlier_than_shared(backend: str) -> None:
    descent = _scheme(backend).solve()
    shared = _scheme(
        backend, mode="shared", ntrials=512, nprobs=20, nthresholds=40, guess="window"
    )
    shared.run(10, False)
    reference = shared.solve()
    pd_d, cost_d = _judge(backend, descent["thresholds"])
    pd_s, cost_s = _judge(backend, reference["thresholds"])
    assert pd_d >= 0.95 * MIN_PD
    # at most the cost of the grid search at no less than the target Pd
    assert cost_d <= 1.05 * cost_s or pd_s < 0.95 * MIN_PD


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_keeps_the_last_threshold_above_a_floor(backend: str) -> None:
    free = _scheme(backend).solve()
    # a floor below the free last threshold changes nothing
    low = _scheme(backend, final_threshold=1.0).solve()
    np.testing.assert_array_equal(low["thresholds"], free["thresholds"])
    floor = float(free["thresholds"][-1]) + 0.5
    fixed = _scheme(backend, final_threshold=floor).solve()
    assert fixed["thresholds"][-1] >= np.float32(floor)
    assert 0.0 < fixed["diagnostics"]["pd_ceiling"] < 1.0
    assert fixed["pd"] >= 0.95 * MIN_PD
    # a floor cannot make the schedule cheaper
    assert fixed["c"] >= 0.9 * free["c"]
    # fap: Q^-1(fap / (nbins x widths)), one-sided Gaussian tail
    by_fap = _scheme(backend, fap=1e-10).solve()
    assert by_fap["thresholds"][-1] > 6.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_default_plan_chooses_its_start(backend: str) -> None:
    sol = libloki.thresholds.DynamicThresholdScheme(
        BRANCHING,
        mode="descent",
        seed=7,
        backend=backend,
        nthreads=2,
        min_pd=MIN_PD,
        descent_nsignal=8192,
        **KW,
    ).solve()
    assert sol["delivery"] in ("met", "within_tolerance")
    assert 0.97 * MIN_PD <= sol["pd"] <= 1.05 * MIN_PD
    diag = sol["diagnostics"]
    # three levels, each run once or twice
    assert 3 <= diag["descent_passes"] <= 6
    # the cost of the smooth start over the flat one, on one noise sample
    assert diag["start_cost_ratio"] > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_has_no_search_states(backend: str, tmp_path) -> None:
    scheme = _scheme(backend)
    scheme.solve()
    for call in (
        scheme.get_states,
        lambda: scheme.get_best_path_thresholds(MIN_PD),
        lambda: scheme.save(str(tmp_path)),
    ):
        with pytest.raises(RuntimeError, match="no search states"):
            call()
    # evaluate() works on the solved schedule
    thresholds = scheme.solve()["thresholds"]
    assert len(scheme.evaluate(thresholds, 1024, 3)) == len(thresholds)


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_starts_from_a_given_schedule(backend: str) -> None:
    cold = _scheme(backend).solve()
    # another seed's solution, shifted: a start that is near but not on target
    donor = _scheme(backend, seed=3).solve()["thresholds"] + np.float32(0.2)
    warm = _scheme(backend, descent_start=donor, descent_passes=PASSES[1:]).solve()
    assert warm["delivery"] in ("met", "within_tolerance")
    pd_w, cost_w = _judge(backend, warm["thresholds"])
    _, cost_c = _judge(backend, cold["thresholds"])
    assert pd_w >= 0.95 * MIN_PD
    assert cost_w <= 1.1 * cost_c
    with pytest.raises(Exception, match="one starting threshold per stage"):
        _scheme(backend, descent_start=donor[:-1]).solve()


def test_descent_starts_from_zscores_of_another_bin_count() -> None:
    donor = _scheme("cpu", seed=3).solve()
    assert donor["zscores"].shape == donor["thresholds"].shape
    kw = {**KW, "nbins": 64}
    args = {"mode": "descent", "backend": "cpu", "nthreads": 2, "min_pd": MIN_PD}
    args |= {"descent_nsignal": 8192, "seed": 7}
    dts = libloki.thresholds.DynamicThresholdScheme
    cold = dts(BRANCHING, descent_passes=PASSES, **args, **kw).solve()
    warm = dts(
        BRANCHING,
        descent_passes=PASSES[1:],
        descent_start=donor["zscores"],
        descent_start_zscores=True,
        **args,
        **kw,
    ).solve()
    assert warm["delivery"] in ("met", "within_tolerance")
    assert 0.97 * MIN_PD <= warm["pd"] <= 1.05 * MIN_PD
    assert warm["c"] <= 1.15 * cold["c"]


def test_descent_reports_an_unreachable_target() -> None:
    sol = _scheme("cpu", final_threshold=9.0).solve()
    assert sol["delivery"] == "short"
    assert len(sol["thresholds"]) == 0
    assert sol["pd"] == pytest.approx(sol["diagnostics"]["pd_ceiling"])
    assert sol["pd"] < MIN_PD
