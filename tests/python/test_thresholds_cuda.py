"""Tests for the CUDA dynamic threshold scheme."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

pytestmark = pytest.mark.skipif(
    "cuda" not in libloki.available_backends(),
    reason="CUDA backend not built",
)

BRANCHING = np.array(
    [4.0, 9.0, 1.0, 2.25575101, 3.98980204, 3.0, 2.80514208, 3.20839363,
     1.0, 1.0, 3.0, 1.0, 1.0, 2.25575101, 1.99490102, 3.0,
     1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 2.80514208, 1.06946454],
    dtype=np.float32,
)  # fmt: skip
SCHEME_KW = {
    "ref_ducy": 0.1,
    "nbins": 64,
    "ntrials": 256,
    "nprobs": 12,
    "prob_min": 0.05,
    "snr_final": 8.0,
    "nthresholds": 60,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.2,
}
THRES_NEIGH = 6


def _scheme(mode: str, seed: int | None, **kwargs: object):  # noqa: ANN202
    return libloki.thresholds.DynamicThresholdScheme(
        BRANCHING, mode=mode, seed=seed, backend="cuda", **{**SCHEME_KW, **kwargs}
    )


def _field_bytes(states: np.ndarray) -> np.ndarray:
    """Raw bytes of every field (the struct padding is not compared)."""
    return np.concatenate(
        [
            np.ascontiguousarray(states[name]).view(np.uint8).reshape(len(states), -1)
            for name in states.dtype.names
        ],
        axis=1,
    )


@pytest.mark.parametrize("mode", ["legacy", "improved"])
def test_seeded_runs_are_bit_identical(mode: str) -> None:
    scheme_a = _scheme(mode, seed=11)
    scheme_a.run(thres_neigh=THRES_NEIGH)
    states_a = _field_bytes(scheme_a.get_states())

    scheme_b = _scheme(mode, seed=11)
    scheme_b.run(thres_neigh=THRES_NEIGH)
    np.testing.assert_array_equal(states_a, _field_bytes(scheme_b.get_states()))

    # Re-running the same object starts from a clean grid.
    scheme_a.run(thres_neigh=THRES_NEIGH)
    np.testing.assert_array_equal(states_a, _field_bytes(scheme_a.get_states()))

    scheme_c = _scheme(mode, seed=12)
    scheme_c.run(thres_neigh=THRES_NEIGH)
    assert not np.array_equal(states_a, _field_bytes(scheme_c.get_states()))


@pytest.mark.parametrize("mode", ["legacy", "improved"])
def test_states_layout_matches_cpu(mode: str) -> None:
    scheme = _scheme(mode, seed=3)
    scheme.run(thres_neigh=THRES_NEIGH)
    states = scheme.get_states()
    cpu = libloki.thresholds.DynamicThresholdScheme(
        BRANCHING, mode=mode, seed=3, **SCHEME_KW
    )
    assert states.dtype == cpu.get_states().dtype
    assert states.shape == (
        len(BRANCHING) * len(scheme.thresholds) * len(scheme.probs),
    )
    np.testing.assert_array_equal(scheme.thresholds, cpu.thresholds)
    np.testing.assert_array_equal(scheme.probs, cpu.probs)
    assert len(scheme.get_best_path_thresholds()) == len(BRANCHING)


def test_save_records_run_metadata(tmp_path) -> None:  # noqa: ANN001
    h5py = pytest.importorskip("h5py")
    scheme = _scheme("improved", seed=5, batch_size=128)
    scheme.run(thres_neigh=THRES_NEIGH)
    path = scheme.save(outdir=str(tmp_path))
    with h5py.File(path, "r") as f:
        assert f.attrs["seed"] == 5
        assert f.attrs["batch_size"] == 128
        assert f.attrs["thres_neigh"] == THRES_NEIGH
        assert f.attrs["mode"] == "improved"
        np.testing.assert_array_equal(
            _field_bytes(f["states"][...].ravel()), _field_bytes(scheme.get_states())
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nbins": 2048},
        {"prob_min": 0.0},
        {"ducy_max": 1.5},
        {"beam_width": 1e-4},
        {"batch_size": 0},
        {"mode": "bogus"},
    ],
)
def test_rejects_invalid_input(kwargs: dict) -> None:
    mode = kwargs.pop("mode", "improved")
    with pytest.raises(ValueError):  # noqa: PT011
        _scheme(mode, seed=1, **kwargs)


@pytest.mark.parametrize("mode", ["improved", "legacy"])
def test_evaluate_is_reproducible_and_leaves_the_grid_alone(mode: str) -> None:
    scheme = _scheme(mode, seed=9)
    scheme.run(thres_neigh=THRES_NEIGH)
    before = _field_bytes(scheme.get_states())
    path = np.array(scheme.get_best_path_thresholds(), dtype=np.float32)
    assert path.shape == BRANCHING.shape
    first = scheme.evaluate(path, ntrials=64, seed=21)
    second = scheme.evaluate(path, ntrials=64, seed=21)
    np.testing.assert_array_equal(_field_bytes(first), _field_bytes(second))
    np.testing.assert_array_equal(before, _field_bytes(scheme.get_states()))
    other = scheme.evaluate(path, ntrials=64, seed=22)
    assert not np.array_equal(_field_bytes(first), _field_bytes(other))


def test_rejects_invalid_branching() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        libloki.thresholds.DynamicThresholdScheme(
            np.array([2.0, np.nan, 2.0], dtype=np.float32), 0.1, backend="cuda"
        )
    with pytest.raises(ValueError):  # noqa: PT011
        libloki.thresholds.DynamicThresholdScheme(
            np.array([2.0], dtype=np.float32), 0.1, backend="cuda"
        )
