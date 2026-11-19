"""The tutorial survey runs end to end and finds each source as it should."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki_survey.cli import main
from loki_survey.collect import load_groups
from loki_survey.rundir import RunDir
from loki_survey.tutorial import NSAMPS, SOURCES, TSAMP, Source, signal

from .conftest import write_site

if TYPE_CHECKING:
    from pathlib import Path

# Two frequency resolutions 1 / T_obs.
FREQ_TOL = 2.0 / (NSAMPS * TSAMP)


def test_tutorial_finds_its_sources(tmp_path: Path, unit_python: str) -> None:
    run = tmp_path / "tutorial"
    assert main(["tutorial", str(run)]) == 0
    # The units run on this test's Python, not the one that wrote site.toml.
    write_site(run, unit_python)

    assert main(["run", str(run)]) == 0

    groups = load_groups(RunDir.at(run))
    calls = {}
    for source in SOURCES:
        near = abs(groups["f0"] - source.freq) <= FREQ_TOL
        assert near.sum() == 1, f"{source.name}: {near.sum()} groups"
        g = int(near.argmax())
        assert groups["selected"][g]
        assert groups["dm"][g] == source.dm
        calls[source.name] = groups["dm_call"][g]
    assert calls == {
        "PSR_A": "dispersed",
        "PSR_B": "dispersed",
        "RFI_150Hz": "undispersed",
    }
    recall = json.loads((run / "evaluate" / "summary.json").read_text())
    (psr_a,) = recall["catalogue"]["pulsars"]
    assert psr_a["name"] == "PSR_A"
    assert psr_a["found"]
    on_lines = [
        json.loads(line)
        for line in (run / "evaluate" / "matches.jsonl").read_text().splitlines()
    ]
    assert [round(m["f0"]) for m in on_lines if m["line"]] == [150]
    assert (run / "report" / "index.html").is_file()


def test_tutorial_keeps_an_existing_run(tmp_path: Path) -> None:
    run = tmp_path / "tutorial"
    run.mkdir()
    (run / "survey.toml").write_text("mine\n")
    assert main(["tutorial", str(run)]) == 2
    assert (run / "survey.toml").read_text() == "mine\n"


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name)
def test_tutorial_signal_has_its_snr(source: Source) -> None:
    t = np.arange(NSAMPS) * TSAMP
    wave = signal(source, 10.0, t)
    # Matched filter on unit-variance noise: S/N = sqrt(sum of squares).
    assert np.sqrt(np.sum(wave**2)) == pytest.approx(10.0)


def test_a_missing_loki_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PyPI's 'loki' is another project; say which loki loki-survey needs."""
    monkeypatch.setitem(sys.modules, "loki", None)
    assert main(["tutorial", str(tmp_path / "tutorial")]) == 2
    assert "github.com/pravirkr/loki" in capsys.readouterr().err


def test_a_loki_too_old_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from loki import libloki  # noqa: PLC0415

    monkeypatch.delattr(libloki.psr_utils, "leaf_phase")
    assert main(["tutorial", str(tmp_path / "tutorial")]) == 2
    err = capsys.readouterr().err
    assert "too old for loki-survey" in err
    assert "libloki.psr_utils.leaf_phase" in err
