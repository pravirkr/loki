"""``evaluate``: catalogue matches (harmonics, Doppler, DM) and spectral lines."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki_survey import evaluate as evaluate_module
from loki_survey.collect import collect
from loki_survey.config import Analysis, Evaluate
from loki_survey.evaluate import (
    _recall,
    evaluate,
    find_lines,
    match_lines,
    observation_of,
    read_catalogue,
    recall_state,
)
from loki_survey.rundir import RunDir

from .synthetic import (
    HARMONIC,
    LONE,
    PULSAR,
    TOBS,
    make_run,
    resolved,
    unit_for,
    write_output,
)

if TYPE_CHECKING:
    from pathlib import Path

CATALOGUE = f"""# a comment line
name,f0,p0,dm,pb,a1
PSR_A,{PULSAR * (1 + 5e-5)},,10.0,,
PSR_B,,{1 / 33.3},40.0,0.5,2.0
PSR_C,{LONE * (1 + 4e-4)},,20.0,,
"""


def test_read_catalogue(tmp_path: Path) -> None:
    path = tmp_path / "cat.csv"
    path.write_text(CATALOGUE)
    pulsars = read_catalogue(path)
    assert [p["name"] for p in pulsars] == ["PSR_A", "PSR_B", "PSR_C"]
    assert pulsars[1]["f0"] == pytest.approx(33.3)
    # 2 pi a1 / pb: 2 light seconds over half a day.
    assert pulsars[1]["orbit_vc"] == pytest.approx(2 * np.pi * 2.0 / 43200.0)
    path.write_text("name,f0\nPSR_X,\n")
    with pytest.raises(ValueError, match="positive f0 or p0"):
        read_catalogue(path)


def test_catalogue_recall_harmonics_and_doppler(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    (run.root / "cat.csv").write_text(CATALOGUE)
    res = resolved(Analysis(evaluate=Evaluate(catalogue="cat.csv")))
    collect(run, res, units)
    summary = evaluate(run, res)
    recall = {p["name"]: p for p in summary["catalogue"]["pulsars"]}
    # 5e-5 off: inside the Earth's Doppler window of 1e-4.
    assert recall["PSR_A"]["found"]
    assert recall["PSR_A"]["harmonics_found"] == 1  # the group at 2 f
    assert recall["PSR_A"]["best"]["selected"]
    # 4e-4 off and no orbit given: outside the window, so not found.
    assert recall["PSR_C"]["in_band"]
    assert not recall["PSR_C"]["found"]
    assert not recall["PSR_B"]["found"]
    entries = [
        json.loads(line)
        for line in (run.evaluate_dir / "matches.jsonl").read_text().splitlines()
    ]
    harmonic = next(e for e in entries if abs(e["f0"] - HARMONIC) < 0.1)
    first = harmonic["catalogue"][0]
    assert (first["name"], first["p"], first["q"]) == ("PSR_A", 2, 1)
    # No prepared series in this run: lines are skipped, not guessed.
    assert summary["lines"] is None


def recall_of(groups: list[tuple[float, bool, str, list[tuple[int, int]]]]) -> dict:
    """_recall of one pulsar at 100 Hz (in band) over hand-made groups.

    Each group: (S/N, kept, removed_by, [(p, q) it matches the pulsar at]).
    """
    snr, kept, removed, ratios = zip(*groups, strict=True)
    table = {
        "snr": np.array(snr),
        "selected": np.array(kept),
        "removed_by": np.array(removed),
    }
    matches = [[{"name": "P", "p": p, "q": q} for p, q in rs] for rs in ratios]
    units = {"f_min": np.array([50.0]), "f_max": np.array([200.0])}
    pulsar = {"name": "P", "f0": 100.0, "dm": None}
    (row,) = _recall([pulsar], matches, table, units)
    return row


def test_recall_reports_a_kept_fundamental_over_a_stronger_removed_harmonic() -> None:
    row = recall_of(
        [(20.0, True, "", [(1, 1)]), (30.0, False, "remove_harmonics", [(7, 11)])],
    )
    assert row["found"]
    assert row["fundamental_selected"]
    assert (row["best"]["p"], row["best"]["q"]) == (1, 1)
    assert recall_state(row) == "selected"


def test_recall_says_found_as_a_harmonic_when_only_that_group_is_kept() -> None:
    row = recall_of([(9.0, True, "", [(2, 1)])])
    assert not row["found"]
    assert recall_state(row) == "found as 2/1"


def test_recall_prefers_a_low_order_harmonic_to_a_stronger_high_order_one() -> None:
    # A bright unrelated group near 7/3 must not hide the pulsar's 2/1 group.
    row = recall_of([(40.0, True, "", [(7, 3)]), (9.0, True, "", [(2, 1)])])
    assert recall_state(row) == "found as 2/1"
    # Between equal orders, the stronger group.
    row = recall_of([(9.0, True, "", [(3, 1)]), (12.0, True, "", [(1, 3)])])
    assert recall_state(row) == "found as 1/3"


def test_recall_says_removed_when_the_fundamental_was_removed() -> None:
    row = recall_of(
        [(10.0, False, "remove_harmonics", [(1, 1)]), (15.0, True, "", [(1, 2)])],
    )
    assert row["found"]
    assert not row["fundamental_selected"]
    assert recall_state(row) == "removed by remove_harmonics"


def test_recall_state_without_a_match() -> None:
    assert recall_state({"best": None, "in_band": True}) == "not found"
    assert recall_state({"best": None, "in_band": False}) == ""


def test_dm_tolerance_rejects_a_match_at_another_dm(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    (run.root / "cat.csv").write_text(f"name,f0,dm\nPSR_A,{PULSAR},30.0\n")
    res = resolved(Analysis(evaluate=Evaluate(catalogue="cat.csv", dm_tol=5.0)))
    collect(run, res, units)
    summary = evaluate(run, res)
    assert not summary["catalogue"]["pulsars"][0]["found"]


def test_find_lines_finds_a_sinusoid_and_nothing_in_noise() -> None:
    rng = np.random.default_rng(3)
    n, tsamp = 1 << 16, 1e-3
    t = np.arange(n) * tsamp
    weights = np.ones(n, dtype=np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    quiet = find_lines(noise, weights, tsamp, fap=1e-3)
    assert quiet["lines"] == []
    hum = noise + 0.3 * np.sin(2 * np.pi * 60.0 * t).astype(np.float32)
    found = find_lines(hum, weights, tsamp, fap=1e-3)
    assert len(found["lines"]) == 1
    assert found["lines"][0]["f"] == pytest.approx(60.0, abs=found["df"])
    hits = match_lines(np.array([60.0, 61.0]), np.full(2, found["df"]), found["lines"])
    assert hits[0] is not None
    assert hits[1] is None


def test_observation_of_a_series() -> None:
    # By default, the DM trials of one observation share its name.
    assert observation_of("obsA_DM10.00") == "obsA"
    assert observation_of("obsA_DM-2.5_beam3") == "obsA_beam3"
    assert observation_of("obsA") == "obsA"
    # A pattern names it by its group "obs", its one group, or its match.
    assert observation_of("ter5_b3_dm12.0", r"^(?P<obs>.*)_dm") == "ter5_b3"
    assert observation_of("ter5_b3_dm12.0", r"_(b[0-9]+)_") == "b3"
    assert observation_of("ter5_b3_dm12.0", r"b[0-9]+") == "b3"
    # A series the pattern does not match is its own observation.
    assert observation_of("other", r"_(b[0-9]+)_") == "other"


def test_lines_of_one_observation_label_only_its_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hum, other_hum = 50.0, 60.0
    run = RunDir.at(tmp_path / "run")
    run.root.mkdir(parents=True)
    # Observation a has lines at both frequencies, observation b at neither:
    # the 50 Hz candidate is a's, the 60 Hz one only b's.
    rows = {
        ("a", 0.0): [(hum, 20.0)],
        ("a", 10.0): [(hum, 12.0)],
        ("b", 0.0): [(other_hum, 15.0)],
    }
    units = []
    for (obs, dm), cands in rows.items():
        unit = unit_for("ffa", dm)
        series = f"{obs}_DM{dm:05.1f}"
        unit = replace(unit, id=unit.id.replace(unit.series, series), series=series)
        units.append(unit)
        write_output(run, unit, dm, cands)
        (run.prep_dir / series).mkdir(parents=True)
    df = 1 / TOBS
    lines_of = {
        "a_DM000.0": [hum, other_hum],
        "b_DM000.0": [],
    }
    searched = []

    def fake_series_lines(_run: RunDir, series: str, fap: float) -> dict:
        searched.append(series)
        lines = [
            {"f": f, "f_lo": f - df, "f_hi": f + df, "power": 99.0}
            for f in lines_of[series]
        ]
        return {"series": series, "df": df, "threshold": fap, "lines": lines}

    monkeypatch.setattr(evaluate_module, "series_lines", fake_series_lines)
    res = resolved()
    collect(run, res, units)
    summary = evaluate(run, res)

    # Each observation's lowest-DM series is searched, once.
    assert sorted(searched) == ["a_DM000.0", "b_DM000.0"]
    lines = summary["lines"]
    searched_in = [
        (o["observation"], o["series"], o["n_lines"]) for o in lines["observations"]
    ]
    assert searched_in == [("a", "a_DM000.0", 2), ("b", "b_DM000.0", 0)]
    assert lines["n_lines"] == 2
    assert lines["groups_on_lines"] == 1
    entries = [
        json.loads(line)
        for line in (run.evaluate_dir / "matches.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["f0"] == pytest.approx(hum)
    assert entries[0]["line"]["observation"] == "a"
