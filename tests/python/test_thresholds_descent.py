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
        **{"min_pd": MIN_PD, **KW, **extra, **opts},
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
    assert 0.97 * MIN_PD <= sol["pd"] <= 1.05 * MIN_PD
    pd, cost = _judge(backend, thresholds)
    assert pd == pytest.approx(sol["pd"], rel=0.05)
    assert cost == pytest.approx(sol["c"], rel=0.1)
    assert sol["diagnostics"]["descent_moves"] > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_pd_policy_pin_vs_free(backend: str) -> None:
    pinned = _scheme(backend, pd_policy="pin").solve()
    assert 0.97 * MIN_PD <= pinned["pd"] <= 1.05 * MIN_PD
    # Short low-cost tree with high S/N: free policy settles at an interior
    # Pd > min_pd and achieves lower L = C / Pd than pinning at min_pd.
    short_bp = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float32)
    kw = dict(
        ref_ducy=0.1,
        nbins=32,
        snr_final=8.5,
        ducy_max=0.2,
        wtsp=1.5,
        mode="descent",
        seed=7,
        backend=backend,
        nthreads=2,
        min_pd=0.2,
        descent_nsignal=8192,
    )
    pin_s = libloki.thresholds.DynamicThresholdScheme(
        short_bp, pd_policy="pin", **kw
    ).solve()
    free_s = libloki.thresholds.DynamicThresholdScheme(
        short_bp, pd_policy="free", **kw
    ).solve()
    assert 0.97 * 0.2 <= pin_s["pd"] <= 1.05 * 0.2
    assert free_s["pd"] > 0.35
    assert free_s["cost"] < pin_s["cost"]
    with pytest.raises(ValueError, match="unknown pd_policy"):
        _scheme(backend, pd_policy="bogus")


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
    descent = _scheme(backend, pd_policy="free").solve()
    shared = _scheme(
        backend, mode="shared", ntrials=512, nprobs=20, nthresholds=40, guess="window"
    )
    shared.run(10, False)
    reference = shared.solve()
    pd_d, cost_d = _judge(backend, descent["thresholds"])
    pd_s, cost_s = _judge(backend, reference["thresholds"])
    assert pd_d >= 0.95 * MIN_PD
    # Under pd_policy="free", L = C / Pd is no worse than the grid search's
    assert (cost_d / pd_d) <= 1.05 * (cost_s / max(pd_s, 1e-9)) or pd_s < 0.95 * MIN_PD
    # Under pd_policy="pin", C at Pd = min_pd is no costlier than shared's C
    pinned = _scheme(backend, pd_policy="pin").solve()
    pd_p, cost_p = _judge(backend, pinned["thresholds"])
    assert 0.95 * MIN_PD <= pd_p <= 1.05 * MIN_PD
    assert cost_p <= 1.05 * cost_s


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_keeps_the_last_threshold_above_a_floor(backend: str) -> None:
    free = _scheme(backend).solve()
    # a floor below the free last threshold changes nothing
    low = _scheme(backend, final_threshold=1.0).solve()
    np.testing.assert_array_equal(low["thresholds"], free["thresholds"])
    floor = float(free["thresholds"][-1]) + 0.5
    fixed = _scheme(backend, final_threshold=floor).solve()
    assert fixed["thresholds"][-1] >= np.float32(floor) - 1e-5
    assert fixed["thresholds"][-1] == pytest.approx(np.float32(floor), abs=1e-4)
    assert 0.0 < fixed["diagnostics"]["pd_ceiling"] < 1.0
    assert 0.95 * MIN_PD <= fixed["pd"] <= 1.05 * MIN_PD
    # an active floor cannot improve L = C / Pd beyond sampling noise
    assert fixed["cost"] >= 0.9 * free["cost"]
    # fap: Q^-1(fap / (nbins x widths)), one-sided Gaussian tail
    by_fap = _scheme(backend, fap=1e-10).solve()
    assert by_fap["thresholds"][-1] > 6.5


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(), reason="CUDA backend not built"
)
def test_descent_fap_floor_counts_the_true_bins() -> None:
    """At 121 bins the CUDA profile is padded to 124: the fap floor counts 121
    start bins per width on both backends, so it binds at the same value."""
    free = _scheme("cpu", nbins=121).solve()
    last = {
        backend: float(_scheme(backend, nbins=121, fap=1e-9).solve()["thresholds"][-1])
        for backend in ("cpu", "cuda")
    }
    assert last["cpu"] > float(free["thresholds"][-1])  # the floor binds
    assert last["cuda"] == pytest.approx(last["cpu"], abs=1e-4)


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
    assert diag["descent_moves"] > 0
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
        lambda: scheme.calibrate(MIN_PD),
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
    pd_c, cost_c = _judge(backend, cold["thresholds"])
    assert pd_w >= 0.95 * MIN_PD
    assert (cost_w / pd_w) <= 1.15 * (cost_c / pd_c)
    with pytest.raises(ValueError, match="one starting threshold per stage"):
        _scheme(backend, descent_start=donor[:-1])


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
    assert warm["cost"] <= 1.15 * cold["cost"]


def _deep_pattern() -> np.ndarray:
    """A 127-stage pattern shaped like the planner's: early 3s, a tail near 1."""
    tail = 1.0 + 0.6 * np.exp(-np.arange(118) / 10.0)
    bp = np.r_[[1, 3, 1, 1, 1, 3, 1, 1, 1], tail].astype(np.float32)
    bp[[17, 40, 77]] = [4.0, 2.0, 2.0]
    return bp


@pytest.mark.parametrize("backend", BACKENDS)
def test_descent_at_production_depth(backend: str) -> None:
    """127 stages with the default passes and sample (the planner's shape)."""
    bp = _deep_pattern()
    scheme = libloki.thresholds.DynamicThresholdScheme(
        bp,
        mode="descent",
        seed=7,
        backend=backend,
        nthreads=2,
        min_pd=0.1,
        **{**KW, "nbins": 16},
    )
    sol = scheme.solve()
    assert sol["thresholds"].shape == (127,)
    assert sol["delivery"] in ("met", "within_tolerance")
    assert 0.97 * 0.1 <= sol["pd"] <= 1.05 * 0.1
    assert sol["diagnostics"]["descent_moves"] > 0


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(), reason="CUDA backend not built"
)
def test_descent_at_256_bins_on_cuda() -> None:
    """Above 128 bins the CUDA design runs the warp scorer and pruned H1."""
    scheme = libloki.thresholds.DynamicThresholdScheme(
        BRANCHING,
        mode="descent",
        seed=7,
        backend="cuda",
        min_pd=MIN_PD,
        **{**KW, "nbins": 256},
    )
    sol = scheme.solve()
    assert sol["delivery"] in ("met", "within_tolerance")
    assert 0.97 * MIN_PD <= sol["pd"] <= 1.05 * MIN_PD
    scores = scheme.simulate_h1_scores(1 << 15, 991)
    pd = float(np.mean(np.all(scores > sol["thresholds"][:, None], axis=0)))
    assert pd == pytest.approx(sol["pd"], rel=0.05)


def test_descent_with_one_stage() -> None:
    scheme = libloki.thresholds.DynamicThresholdScheme(
        np.array([2.0], dtype=np.float32),
        mode="descent",
        seed=7,
        backend="cpu",
        nthreads=2,
        min_pd=MIN_PD,
        descent_passes=PASSES,
        descent_nsignal=8192,
        **KW,
    )
    sol = scheme.solve()
    assert sol["thresholds"].shape == (1,)
    assert np.isfinite(sol["thresholds"][0])
    assert sol["delivery"] in ("met", "within_tolerance")
    assert 0.97 * MIN_PD <= sol["pd"] <= 1.05 * MIN_PD


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("min_pd", [1e-3, 0.999])
def test_descent_at_the_edges_of_min_pd(backend: str, min_pd: float) -> None:
    """The pin aims at min_pd + 2 SE on one 2^17-trial bank and the report
    counts Pd on another: Pd within 2 SE +/- 4 sqrt(2) SE of min_pd."""
    sol = _scheme(backend, min_pd=min_pd).solve()
    assert sol["delivery"] in ("met", "within_tolerance")
    se = np.sqrt(min_pd * (1.0 - min_pd) / sol["pd_trials"])
    assert min_pd - 4.0 * se <= sol["pd"] <= min_pd + 8.0 * se
    assert np.all(np.isfinite(sol["thresholds"]))


def test_descent_reports_an_unreachable_target() -> None:
    sol = _scheme("cpu", final_threshold=9.0).solve()
    assert sol["delivery"] == "short"
    assert len(sol["thresholds"]) == 0
    assert sol["pd"] == pytest.approx(sol["diagnostics"]["pd_ceiling"])
    assert sol["pd"] < MIN_PD


BAD_SETTINGS = [
    ({"min_pd": 0.0}, "min_pd"),
    ({"min_pd": 1.0}, "min_pd"),
    ({"fap": -0.1}, "fap"),
    ({"fap": 1.0}, "fap"),
    ({"fap": float("nan")}, "fap"),
    ({"final_threshold": float("nan")}, "final_threshold"),
    ({"final_threshold": -1.0}, "final_threshold"),
    ({"descent_nsignal": 0}, "descent_nsignal"),
    ({"descent_passes": [(0, 0.3, 0.1, 8, 1)]}, "ntrials"),
    ({"descent_passes": [(1 << 21, 0.3, 0.1, 8, 1)]}, "ntrials"),
    ({"descent_passes": [(1024, 0.0, 0.1, 8, 1)]}, "margin"),
    ({"descent_passes": [(1024, float("nan"), 0.1, 8, 1)]}, "margin"),
    ({"descent_passes": [(1024, 0.3, 0.0, 8, 1)]}, "step"),
    ({"descent_passes": [(1024, 0.1, 0.3, 8, 1)]}, "below margin"),
    ({"descent_passes": [(1024, 0.3, 0.1, 1, 8)]}, "min_block"),
    ({"descent_passes": [(1024, 0.3, 0.1, 8, 0)]}, "min_block"),
    ({"descent_passes": [(1024, 0.025, 0.0125, 8, 1)]}, "margin - step"),
    ({"w_terminal": float("nan")}, "w_terminal"),
    ({"w_terminal": -1.0}, "w_terminal"),
    ({"nbins": 2000}, "nbins"),
    ({"ref_ducy": 0.0}, "ref_ducy"),
    ({"wtsp": 0.0}, "wtsp"),
    ({"snr_final": float("nan")}, "finite"),
    ({"descent_start": [1.0] * (len(BRANCHING) - 1)}, "one starting threshold"),
    ({"descent_start": [float("nan")] * len(BRANCHING)}, "finite"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("opts", "match"), BAD_SETTINGS)
def test_descent_rejects_bad_settings_at_construction(
    backend: str, opts: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _scheme(backend, **opts)


def test_descent_settings_are_not_read_by_other_modes() -> None:
    # The search modes ignore the descent's settings.
    _scheme("cpu", mode="shared", descent_passes=[(0, 0.0, 0.0, 1, 8)], fap=-1.0)
