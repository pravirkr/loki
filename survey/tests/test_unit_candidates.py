"""loki's candidate step in a unit, as survey.toml's [candidates] says."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py

from loki_survey.cli import main

from .conftest import write_site
from .test_end_to_end import RECOVER_SNR_MIN, SURVEY, best_near_truth, simulate

if TYPE_CHECKING:
    from pathlib import Path

FFA_UNIT = "sim_DM10.00__mid__b0__ffa_accel"
EP_UNIT = "sim_DM10.00__mid__b0__ep_accel"


def make_run(tmp_path: Path, unit_python: str, *, survey: str = SURVEY) -> Path:
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(survey)
    write_site(run, unit_python)
    return run


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
