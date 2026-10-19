"""Tests for DynamicThresholdScheme.solve (mode "shared", CPU and CUDA)."""

from __future__ import annotations

from pathlib import Path

import h5py
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
MIN_PD = 0.3
KW = {
    "ref_ducy": 0.1,
    "nbins": 32,
    "ntrials": 512,
    "nprobs": 20,
    "snr_final": 8.0,
    "nthresholds": 40,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.0,
    "guess": "window",
}


def _scheme(backend: str, seed: int = 7, mode: str = "shared", refine: str = "full"):
    extra = {"nthreads": 2} if backend == "cpu" else {}
    return libloki.thresholds.DynamicThresholdScheme(
        BRANCHING,
        mode=mode,
        seed=seed,
        backend=backend,
        min_pd=MIN_PD,
        refine=refine,
        **extra,
        **KW,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("refine", ["none", "shortlist", "full"])
def test_solve_reports_the_pd_of_its_thresholds(backend: str, refine: str) -> None:
    scheme = _scheme(backend, refine=refine)
    scheme.run(10)
    sol = scheme.solve()
    thresholds = sol["thresholds"]
    assert thresholds.shape == (scheme.nstages,)
    assert sol["delivery"] != "no_path"
    assert sol["refine"] == refine
    assert sol["pd"] >= 0.9 * MIN_PD
    assert sol["pd_se"] > 0
    assert sol["pd_trials"] == 1 << 17
    assert len(sol["eval_states"]) == scheme.nstages
    last = sol["eval_states"][-1]
    assert sol["c"] == pytest.approx(
        float(last["complexity_cumul"]) + float(last["complexity"]),
    )
    assert sol["cost"] == pytest.approx(sol["c"] / sol["pd"])
    # An independent evaluation agrees within the errors of the two samples
    check = scheme.evaluate(thresholds, 65536, 424242)[-1]
    se = np.sqrt(sol["pd"] * (1 - sol["pd"]) * (1 / sol["pd_trials"] + 1 / 65536))
    assert abs(float(check["success_h1_cumul"]) - sol["pd"]) < 5 * se


@pytest.mark.parametrize("mode", ["legacy", "improved"])
def test_solve_needs_mode_shared(mode: str) -> None:
    scheme = _scheme("cpu", mode=mode)
    scheme.run(6)
    with pytest.raises(ValueError, match="shared"):
        scheme.solve()


def test_shared_mode_on_the_cpu_needs_the_window_beam() -> None:
    kw = {**KW, "guess": "analytic"}
    with pytest.raises(ValueError, match="window"):
        libloki.thresholds.DynamicThresholdScheme(
            BRANCHING,
            mode="shared",
            seed=7,
            backend="cpu",
            nthreads=2,
            **kw,
        )


@pytest.mark.parametrize("mode", ["legacy", "improved", "shared"])
def test_save_in_every_mode_and_the_solution_after_solve(
    mode: str,
    tmp_path: Path,
) -> None:
    scheme = _scheme("cpu", mode=mode, refine="shortlist")
    scheme.run(6)
    plain = Path(scheme.save(outdir=str(tmp_path / "plain")))
    with h5py.File(plain) as f:
        assert {"states", "thresholds", "probs", "beam_lo", "beam_hi"} <= set(f)
        assert f["states"].shape == (
            scheme.nstages,
            scheme.nthresholds,
            scheme.nprobs,
        )
        assert "solution" not in f
    if mode != "shared":
        return
    sol = scheme.solve()
    solved = Path(scheme.save(outdir=str(tmp_path / "solved")))
    with h5py.File(solved) as f:
        grp = f["solution"]
        np.testing.assert_array_equal(grp["thresholds"][...], sol["thresholds"])
        np.testing.assert_array_equal(
            grp["search_thresholds"][...],
            sol["search_thresholds"],
        )
        assert grp.attrs["pd"] == pytest.approx(sol["pd"])
        assert grp.attrs["delivery"] == sol["delivery"]
        assert grp.attrs["refine"] == "shortlist"
        assert grp["eval_complexity"].shape == (scheme.nstages,)
        # the search itself is still there for plotting the paths
        assert f["states"].shape[0] == scheme.nstages


def test_shared_states_have_the_pd_grid_anchored_at_min_pd() -> None:
    scheme = _scheme("cpu")
    probs = scheme.probs
    assert np.all(np.diff(probs) > 0)
    assert probs[-1] == pytest.approx(1.0, abs=1e-5)
    # min_pd is a bin edge, with nprobs edges from it to 1
    above = probs[probs >= MIN_PD * 0.9999]
    assert above[0] == pytest.approx(MIN_PD, rel=1e-4)
    assert len(above) == KW["nprobs"]
    assert len(probs) - len(above) <= 16
