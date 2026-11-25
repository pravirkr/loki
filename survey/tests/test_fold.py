"""``fold``'s leaves: kept from an EP result, built from FFA Taylor parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from loki_survey.fold import candidate_leaf, taylor_leaf

if TYPE_CHECKING:
    from pathlib import Path


def test_taylor_leaf_rows() -> None:
    leaf = taylor_leaf(np.array([-0.5, 30.0, 142.3]))
    assert leaf.shape == (5, 2)
    np.testing.assert_array_equal(leaf[:3, 0], [-0.5, 30.0, 142.3])
    assert leaf[3, 0] == 0.0  # no delay
    assert leaf[4, 0] == 142.3  # f0
    assert not leaf[:, 1].any()  # no errors


def write_output(path: Path, *, leaves: bool) -> None:
    with h5py.File(path, "w") as handle:
        handle["candidates/id"] = np.array([7, 9], dtype=np.uint64)
        handle["candidates/param_sets"] = np.array([[1.0, 50.0], [2.0, 60.0]])
        if leaves:
            handle["survey/leaves"] = np.arange(16, dtype=np.float64).reshape(2, 4, 2)


def test_candidate_leaf_prefers_the_kept_leaf(tmp_path: Path) -> None:
    write_output(tmp_path / "ep.h5", leaves=True)
    write_output(tmp_path / "ffa.h5", leaves=False)
    with h5py.File(tmp_path / "ep.h5") as handle:
        np.testing.assert_array_equal(
            candidate_leaf(handle, 9),
            np.arange(8, 16).reshape(4, 2),
        )
    with h5py.File(tmp_path / "ffa.h5") as handle:
        np.testing.assert_array_equal(candidate_leaf(handle, 9)[:2, 0], [2.0, 60.0])
        with pytest.raises(KeyError, match="not found once"):
            candidate_leaf(handle, 8)
