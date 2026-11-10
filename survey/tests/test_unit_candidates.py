"""loki's candidate step in a unit: its options, a retry from loki's spool."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import h5py
import pytest

from loki_survey import unit as unit_module
from loki_survey.cli import main
from loki_survey.rundir import RunDir, read_json

from .conftest import write_site
from .test_end_to_end import RECOVER_SNR_MIN, SURVEY, best_near_truth, simulate

if TYPE_CHECKING:
    from pathlib import Path

FFA_UNIT = "sim_DM10.00__mid__b0__ffa_accel"
EP_UNIT = "sim_DM10.00__mid__b0__ep_accel"
# The FFA box alone, down to S/N 5: the noise leaves far more distinct tracks
# than loki's smallest candidate memory can group.
FFA_SURVEY = SURVEY.split('[[search.box]]\nname = "ep_accel"')[0].replace(
    "snr_min = 7.0",
    "snr_min = 5.0",
)
TINY_CANDIDATE_GB = 1e-4


def make_run(tmp_path: Path, unit_python: str, *, survey: str = SURVEY) -> Path:
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(survey)
    write_site(run, unit_python)
    return run


def set_candidate_memory(run: Path, gb: float) -> None:
    site = (run / "site.toml").read_text()
    (run / "site.toml").write_text(
        site.replace("threads = 2", f"threads = 2\ncandidate_memory_gb = {gb}"),
    )


def test_cell_cap_keeps_the_strongest_trial_of_each_cell(
    tmp_path: Path,
    unit_python: str,
) -> None:
    survey = SURVEY + "[candidates]\nmax_per_cell = 1\ncell_log_width = 2e-4\n"
    run = make_run(tmp_path, unit_python, survey=survey)

    assert main(["run", str(run)]) == 0

    for uid in (FFA_UNIT, EP_UNIT):
        output = run / "out" / f"{uid}.h5"
        assert best_near_truth(output) >= RECOVER_SNR_MIN
        with h5py.File(output, "r") as handle:
            assert handle.attrs["n_cell_dropped"] > 0
            assert handle["processing"].attrs["max_per_cell"] == 1


def test_a_failed_candidate_step_runs_again_from_loki_spool(
    tmp_path: Path,
    unit_python: str,
) -> None:
    run = make_run(tmp_path, unit_python, survey=FFA_SURVEY)
    set_candidate_memory(run, TINY_CANDIDATE_GB)

    assert main(["run", str(run)]) == 0

    summary = read_json(RunDir.at(run).unit_summary(FFA_UNIT))
    assert summary["candidates_republished"]
    assert summary["resumed_from"] is None
    assert best_near_truth(run / "out" / f"{FFA_UNIT}.h5") >= RECOVER_SNR_MIN


def test_a_later_attempt_publishes_a_kept_spool_without_searching(
    tmp_path: Path,
    unit_python: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = make_run(tmp_path, unit_python, survey=FFA_SURVEY)
    set_candidate_memory(run, TINY_CANDIDATE_GB)
    for step in ("check", "prep", "plan"):
        assert main([step, str(run)]) == 0
    rundir = RunDir.at(run)

    # Attempt 1: the search ends, and its candidate step fails twice.
    def fail(*_args: object) -> None:
        msg = "out of candidate memory"
        raise RuntimeError(msg)

    monkeypatch.setattr(unit_module, "publish_spool", fail)
    with pytest.raises(RuntimeError, match="out of candidate memory"):
        unit_module.run_unit(rundir, FFA_UNIT, 1)
    monkeypatch.undo()
    rundir.quarantine_dir.mkdir(exist_ok=True)
    shutil.move(rundir.unit_tmp(FFA_UNIT, 1), rundir.quarantine_dir / f"{FFA_UNIT}.1")

    # Attempt 2 publishes the kept spool and never searches.
    def no_search(*_args: object) -> None:
        msg = "searched again"
        raise AssertionError(msg)

    monkeypatch.setattr(unit_module, "search", no_search)
    summary = unit_module.run_unit(rundir, FFA_UNIT, 2)
    assert summary["resumed_from"].endswith(f"{FFA_UNIT}.1/{FFA_UNIT}_ffa_spool.h5")
    assert best_near_truth(rundir.unit_output(FFA_UNIT)) >= RECOVER_SNR_MIN


def test_a_spool_of_another_search_is_not_resumed(
    tmp_path: Path,
    unit_python: str,
) -> None:
    run = make_run(tmp_path, unit_python, survey=FFA_SURVEY)
    for step in ("check", "prep", "plan"):
        assert main([step, str(run)]) == 0
    rundir = RunDir.at(run)
    kept = rundir.quarantine_dir / f"{FFA_UNIT}.1"
    kept.mkdir(parents=True)
    with h5py.File(kept / f"{FFA_UNIT}_ffa_spool.h5", "w") as handle:
        handle.attrs["complete"] = 1
    (kept / unit_module.SEARCH_JSON).write_text(json.dumps({"search_sha256": "x"}))
    unit = next(u for u in unit_module.load_units(rundir) if u.id == FFA_UNIT)
    assert unit_module.kept_spool(rundir, unit, "y") is None
    assert (
        unit_module.kept_spool(rundir, unit, "x") == kept / f"{FFA_UNIT}_ffa_spool.h5"
    )
