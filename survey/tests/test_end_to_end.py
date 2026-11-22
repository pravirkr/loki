"""Milestone 1's check: a simulated pulsar is found end to end, FFA and EP."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np

from loki import libloki
from loki_survey.cli import main
from loki_survey.plan import loki_candidate_defaults, options_record

from .conftest import write_site

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from .conftest import Fake

NSAMPS = 1 << 18
DT = 64e-6
TRUE_FREQ = 142.3
TRUE_ACCEL = 50.0
INJECTED_SNR = 15.0
# The injected pulsar must come back at this folded S/N or more.
RECOVER_SNR_MIN = 10.0
# Two frequency resolutions 1 / T_obs.
FREQ_TOL = 2.0 / (NSAMPS * DT)
SEED = 1

SURVEY = """
[survey]
name = "end-to-end"
[data]
series = ["data/*.dat"]
[search]
snr_min = 7.0
min_pd = 0.5
[[search.region]]
name = "mid"
f_hz = [100.0, 200.0]
nbins = 32
[[search.box]]
name = "ffa_accel"
kind = "ffa"
order = "accel"
limits = { accel = [-100.0, 100.0] }
[[search.box]]
name = "ep_accel"
kind = "ep"
order = "accel"
limits = { accel = [-100.0, 100.0] }
segments = 8
bseg_brute = 1024
ref_segs = [4]
[output]
raw_fraction = 1.0
"""


def simulate(path: Path) -> None:
    mod = libloki.simulation.ModulatorParams()
    mod.acc = TRUE_ACCEL
    cfg = libloki.simulation.PulseSignalConfig(
        1.0 / TRUE_FREQ,
        DT,
        NSAMPS,
        INJECTED_SNR,
        0.1,
        "derivative",
        mod,
        None,
        SEED,
    )
    series = cfg.generate("gaussian", 0.5)
    libloki.io.TimeSeries(
        np.asarray(series.ts_e, dtype=np.float32),
        np.asarray(series.ts_v, dtype=np.float32),
        DT,
    ).write(str(path))


def best_near_truth(output: Path) -> float:
    with h5py.File(output, "r") as handle:
        f0 = np.asarray(handle["candidates/f0"][:])
        snr = np.asarray(handle["candidates/snr"][:])
    near = np.abs(f0 - TRUE_FREQ) <= FREQ_TOL
    return float(snr[near].max()) if near.any() else 0.0


def test_simulated_pulsar_is_found(tmp_path: Path, unit_python: str) -> None:
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    survey = SURVEY + '[evaluate]\ncatalogue = "catalogue.csv"\n'
    (run / "survey.toml").write_text(survey)
    (run / "catalogue.csv").write_text(f"name,f0\nSIM,{TRUE_FREQ}\n")
    write_site(run, unit_python)

    assert main(["run", str(run)]) == 0

    for box in ("ffa_accel", "ep_accel"):
        uid = f"sim_DM10.00__mid__b0__{box}"
        best = best_near_truth(run / "out" / f"{uid}.h5")
        assert best >= RECOVER_SNR_MIN, f"{box}: best S/N {best:.2f}"
        with h5py.File(run / "out" / f"{uid}.h5", "r") as handle:
            assert "science_sha256" in handle["survey"].attrs
        assert (run / "out" / "raw").is_dir()

    # Each unit ran loki's candidate options for its kind, as resolved.json
    # records them: the EP unit reduces its leaves per run, the FFA unit keeps
    # every trial.
    recorded = json.loads((run / "resolved.json").read_text())["candidate_options"]
    assert recorded["ep"] == options_record(loki_candidate_defaults("ep"))
    assert recorded["ffa"] == options_record(loki_candidate_defaults("ffa"))
    for box, kind in (("ffa_accel", "ffa"), ("ep_accel", "ep")):
        uid = f"sim_DM10.00__mid__b0__{box}"
        with h5py.File(run / "out" / f"{uid}.h5", "r") as handle:
            ran = handle["processing"].attrs
            for name in ("max_per_cell", "cell_bins", "dominance_kappa"):
                assert ran[name] == recorded[kind][name], (box, name)

    # The EP unit keeps loki's per-level table of each run, overloads included;
    # the FFA unit has no pruning runs.
    ep_uid = "sim_DM10.00__mid__b0__ep_accel"
    ep = json.loads((run / "out" / f"{ep_uid}.json").read_text())
    assert ep["n_runs"] >= 1
    assert 0 <= ep["n_runs_overloaded"] <= ep["n_runs"]
    with h5py.File(run / "out" / f"{ep_uid}.h5", "r") as handle:
        tables: list[str] = []
        handle["level_stats"].visititems(
            lambda name, node: (
                tables.append(name) if isinstance(node, h5py.Dataset) else None
            ),
        )
        assert len(tables) == ep["n_runs"]
        assert "n_overloads" in handle["level_stats"][tables[0]].dtype.names
    ffa = json.loads((run / "out" / "sim_DM10.00__mid__b0__ffa_accel.json").read_text())
    assert ffa["n_runs"] == 0

    # Collected, matched and reported: FFA and EP found the pulsar, one group.
    summary = json.loads((run / "collect" / "summary.json").read_text())
    assert summary["groups"] >= 1
    recall = json.loads((run / "evaluate" / "summary.json").read_text())
    (pulsar,) = recall["catalogue"]["pulsars"]
    assert pulsar["found"]
    assert pulsar["best"]["selected"]
    assert "SIM" in (run / "report" / "index.html").read_text()
    # Folded with loki's own model: the pulsar keeps its S/N, the 1.002 f
    # control loses it, and both halves carry it.
    (folded, *_) = json.loads((run / "folds" / "summary.json").read_text())["groups"]
    assert folded["snr"] >= RECOVER_SNR_MIN
    assert folded["snr_control"] < folded["snr"] / 2
    assert min(folded["snr_halves"]) > folded["snr"] / 2
    assert "Folds" in (run / "report" / "index.html").read_text()

    # A second run finds nothing to do and changes no output.
    before = {p.name: p.stat().st_mtime_ns for p in (run / "out").glob("*.h5")}
    assert main(["run", str(run)]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in (run / "out").glob("*.h5")}
    assert before == after

    # The analysis may change after units finished; the science may not.
    (run / "survey.toml").write_text(survey + "[collect]\nmax_candidates = 5\n")
    assert main(["check", str(run)]) == 0
    (run / "survey.toml").write_text(survey.replace("snr_min = 7.0", "snr_min = 6.0"))
    assert main(["check", str(run)]) == 1


def test_simulated_pulsar_is_found_through_lsf(
    tmp_path: Path,
    unit_python: str,
    fake: Fake,
) -> None:
    """The same run through the fake LSF, with one unit out of memory once."""
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(SURVEY)
    write_site(run, unit_python, kind="lsf")
    ep_unit = "sim_DM10.00__mid__b0__ep_accel"
    fake.control(hosts=["h1", "h2"], need_gb={ep_unit: 5.0})

    assert main(["run", str(run), "--wait"]) == 0

    for box in ("ffa_accel", "ep_accel"):
        uid = f"sim_DM10.00__mid__b0__{box}"
        best = best_near_truth(run / "out" / f"{uid}.h5")
        assert best >= RECOVER_SNR_MIN, f"{box}: best S/N {best:.2f}"
    # The EP unit asked for 4 GB, was killed for memory, and ran at 6 GB.
    arrays = fake.jobs().values()
    assert [round(j["mem_gb"], 1) for j in arrays] == [4.0, 6.0]


def test_check_refuses_a_unit_too_small_for_loki(
    tmp_path: Path,
    unit_python: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse a unit whose loki budget leaves no room past the candidate buffer."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "survey.toml").write_text(SURVEY)
    write_site(run, unit_python, memory_gb=2.0, overhead_gb=1.0)
    assert main(["check", str(run)]) == 2
    assert "candidate buffer" in capsys.readouterr().err


def test_plan_explains_a_unit_too_small_for_ep(
    tmp_path: Path,
    unit_python: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report loki's planning refusal in site.toml's terms, not as a traceback."""
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(SURVEY)
    write_site(run, unit_python, memory_gb=2.0, overhead_gb=0.5)
    assert main(["run", str(run)]) == 2
    assert "could not plan this EP search with 1.5 GB" in capsys.readouterr().err


def test_cuda_units_build_their_own_ep_plans(
    tmp_path: Path,
    unit_python: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CUDA plan is sized by a GPU, so plan leaves it to the units."""
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(SURVEY)
    write_site(run, unit_python)
    site = (run / "site.toml").read_text()
    (run / "site.toml").write_text(
        site.replace("threads = 2", 'threads = 2\nbackend = "cuda"'),
    )
    for step in ("check", "prep", "plan"):
        assert main([step, str(run)]) == 0
    assert "built by the CUDA units on their GPU" in capsys.readouterr().out
    assert not list((run / "plans").glob("*.h5"))
