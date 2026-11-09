"""``evaluate``: catalogue matches (harmonics, Doppler, DM) and spectral lines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki_survey.collect import collect
from loki_survey.config import Analysis, Evaluate
from loki_survey.evaluate import evaluate, find_lines, match_lines, read_catalogue

from .synthetic import HARMONIC, LONE, PULSAR, make_run, resolved

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
