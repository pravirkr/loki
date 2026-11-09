"""``collect``: grouping across units, labels, removal rules and the ledger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki_survey.collect import (
    IncompleteRunError,
    collect,
    dm_call,
    group_by_frequency,
    load_groups,
)
from loki_survey.config import Analysis, Collect

from .synthetic import (
    HARMONIC,
    INTERFERENCE,
    LONE,
    PULSAR,
    SOURCES,
    make_run,
    resolved,
)

if TYPE_CHECKING:
    from pathlib import Path


def group_of(groups: dict, f: float) -> int:
    return int(np.argmin(np.abs(groups["f0"] - f)))


def ledger(tmp_path: Path) -> list[dict]:
    text = (tmp_path / "run" / "collect" / "ledger.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_groups_span_boxes_and_dms_and_remove_nothing(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    summary = collect(run, resolved(), units)
    groups = load_groups(run)
    assert summary["groups"] == len(SOURCES)
    assert summary["members"] == sum(len(snrs) for _, _, snrs in SOURCES)
    assert summary["selected"] == len(SOURCES)
    assert ledger(tmp_path) == []

    # Groups are numbered by falling S/N: the interference (S/N 20) first.
    assert group_of(groups, INTERFERENCE) == 0
    pulsar = group_of(groups, PULSAR)
    assert pulsar == 1
    assert groups["n_members"][pulsar] == 4
    assert groups["boxes"][pulsar] == "ep,ffa"
    assert groups["dm"][pulsar] == 10.0
    assert groups["dm_call"][pulsar] == "dispersed"
    assert groups["snr"][pulsar] == pytest.approx(15.0)
    assert groups["lowest_dm_snr_ratio"][pulsar] == pytest.approx(8.0 / 15.0)
    assert groups["dm_call"][group_of(groups, INTERFERENCE)] == "undispersed"
    assert groups["dm_call"][group_of(groups, LONE)] == "highest_dm"
    harmonic = group_of(groups, HARMONIC)
    assert groups["harmonic_of"][harmonic] == pulsar
    assert (groups["harmonic_p"][harmonic], groups["harmonic_q"][harmonic]) == (2, 1)
    assert groups["harmonic_of"][pulsar] == -1


def test_each_rule_removes_with_a_ledger_entry(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    rules = Collect(snr_min=7.5, drop_undispersed=True, remove_harmonics=True)
    summary = collect(run, resolved(Analysis(collect=rules)), units)
    groups = load_groups(run)
    removed = {e["rule"]: e["f0"] for e in ledger(tmp_path)}
    assert set(removed) == {"snr_min", "drop_undispersed", "remove_harmonics"}
    assert removed["snr_min"] == pytest.approx(LONE, abs=0.01)
    assert removed["drop_undispersed"] == pytest.approx(INTERFERENCE, abs=0.01)
    assert removed["remove_harmonics"] == pytest.approx(HARMONIC, abs=0.01)
    # The pulsar must come through every rule.
    assert groups["selected"][group_of(groups, PULSAR)]
    assert summary["selected"] == 1
    assert sum(summary["removed"].values()) == 3


def test_max_candidates_keeps_the_strongest(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    collect(run, resolved(Analysis(collect=Collect(max_candidates=2))), units)
    groups = load_groups(run)
    kept = sorted(groups["snr"][groups["selected"]].tolist(), reverse=True)
    assert kept == [20.0, 15.0]
    assert [e["rule"] for e in ledger(tmp_path)] == ["max_candidates"] * 2


def test_an_unfinished_unit_needs_partial(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    run.unit_summary(units[0].id).unlink()
    with pytest.raises(IncompleteRunError, match="--partial"):
        collect(run, resolved(), units)
    summary = collect(run, resolved(), units, partial=True)
    assert summary["units_missing"] == [units[0].id]


def test_a_changed_output_is_refused(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    with run.unit_output(units[1].id).open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="sha256"):
        collect(run, resolved(), units)


def test_grouping_links_neighbours_within_the_radius() -> None:
    tobs = np.full(5, 10.0)  # one resolution is 0.1 Hz
    f0 = np.array([1.00, 1.15, 1.32, 2.0, 1.48])
    labels = group_by_frequency(f0, tobs, radius_bins=2.0)
    # 1.00-1.15-1.32-1.48 chain in steps under 0.2 Hz; 2.0 stands alone.
    assert labels.tolist() == [0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    ("peak", "call"),
    [(0.0, "undispersed"), (10.0, "dispersed"), (20.0, "highest_dm")],
)
def test_dm_call(peak: float, call: str) -> None:
    assert dm_call(peak, np.array([0.0, 10.0, 20.0]), dm_zero_max=0.0) == call


def test_dm_call_edges() -> None:
    dms = np.array([5.0, 10.0])
    assert dm_call(5.0, dms, dm_zero_max=0.0) == "lowest_dm"
    assert dm_call(5.0, np.array([5.0]), dm_zero_max=0.0) == "single_dm"
    assert dm_call(float("nan"), dms, dm_zero_max=0.0) == "unknown_dm"
