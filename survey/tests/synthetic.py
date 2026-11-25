"""Unit outputs written by hand, for the analysis tests: no search runs.

Each output has the layout ``run-unit`` writes (loki's common candidate tables
with the run's provenance) and a summary JSON, so ``collect`` reads it as it
reads a searched unit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import h5py
import numpy as np

from loki_survey.config import Analysis, Resolved, Site
from loki_survey.rundir import RunDir, atomic_write_json, sha256_file

from .conftest import make_unit

if TYPE_CHECKING:
    from pathlib import Path

    from loki_survey.plan import Unit

SCIENCE = "synthetic-science"
TOBS = 100.0
SPEED_OF_LIGHT = 299792458.0
# A pulsar, interference, a harmonic of the pulsar and a lone weak candidate:
# (name, frequency in Hz, {(box, dm): S/N}).
PULSAR = 142.3
INTERFERENCE = 50.0
HARMONIC = 2 * PULSAR
LONE = 77.7
SOURCES = (
    ("pulsar", PULSAR, {("ffa", 0.0): 8.0, ("ffa", 10.0): 15.0, ("ep", 10.0): 14.0,
                        ("ffa", 20.0): 9.0}),
    ("interference", INTERFERENCE, {("ffa", 0.0): 20.0, ("ffa", 10.0): 12.0,
                                    ("ffa", 20.0): 8.0}),
    ("harmonic", HARMONIC, {("ffa", 10.0): 9.0}),
    ("lone", LONE, {("ffa", 20.0): 7.2}),
)  # fmt: skip
DMS = (0.0, 10.0, 20.0)
BOXES = ("ffa", "ep")


def site() -> Site:
    return Site(
        executor="local",
        max_memory_gb=2.0,
        max_threads=1,
        unit_memory_gb=1.0,
        unit_overhead_gb=0.1,
        unit_threads=2,
        backend="cpu",
        device=0,
        python="python",
        poll_s=0.1,
        oom_retries=1,
        oom_factor=1.5,
    )


def resolved(analysis: Analysis | None = None) -> Resolved:
    survey = type("Survey", (), {"name": "synthetic"})()
    return Resolved(
        survey=survey,  # type: ignore[arg-type]
        site=site(),
        science_sha256=SCIENCE,
        analysis=analysis or Analysis(),
    )


def unit_for(box: str, dm: float) -> Unit:
    return replace(
        make_unit(f"s_DM{dm:05.1f}__r__b0__{box}"),
        series=f"s_DM{dm:05.1f}",
        box=box,
        kind="ffa" if box == "ffa" else "ep",
        f_min=10.0,
        f_max=1000.0,
    )


def write_output(
    run: RunDir,
    unit: Unit,
    dm: float,
    rows: list[tuple[float, float]],
) -> None:
    """Write a unit's output and summary; ``rows`` are (frequency, S/N)."""
    run.out_dir.mkdir(parents=True, exist_ok=True)
    path = run.unit_output(unit.id)
    n = len(rows)
    f0 = np.array([r[0] for r in rows], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["complete"] = 1
        handle.attrs["param_names"] = np.array(["accel", "freq"], dtype=object)
        obs = handle.create_group("observation")
        obs.attrs["dm"] = dm
        obs.attrs["tobs"] = TOBS
        cands = handle.create_group("candidates")
        cands["id"] = np.arange(n, dtype=np.uint64)
        cands["cluster_id"] = np.arange(n, dtype=np.uint64)
        cands["f0"] = f0
        cands["f1"] = np.zeros(n)
        cands["f2"] = np.zeros(n)
        cands["snr"] = np.array([r[1] for r in rows], dtype=np.float32)
        cands["width"] = np.full(n, 2, dtype=np.uint32)
        cands["nbins"] = np.full(n, 32, dtype=np.uint32)
        cands["duty_cycle"] = np.full(n, 2 / 32)
        cands["param_sets"] = np.stack((np.zeros(n), f0), axis=1)
        handle.create_group("survey").attrs["science_sha256"] = SCIENCE
    atomic_write_json(
        run.unit_summary(unit.id),
        {
            "unit": unit.id,
            "total_s": 10.0,
            "peak_rss_gb": 0.5,
            "output_sha256": sha256_file(path),
            "science_sha256": SCIENCE,
        },
    )


def make_run(tmp_path: Path) -> tuple[RunDir, list[Unit]]:
    """Write one unit per box and DM with the sources' candidates."""
    run = RunDir.at(tmp_path / "run")
    run.root.mkdir(parents=True, exist_ok=True)
    units = []
    rng = np.random.default_rng(1)
    for box in BOXES:
        for dm in DMS:
            unit = unit_for(box, dm)
            units.append(unit)
            rows = [
                # Each unit measures the frequency within a tenth of a bin.
                (f + rng.uniform(-0.1, 0.1) / TOBS, snrs[(box, dm)])
                for _, f, snrs in SOURCES
                if (box, dm) in snrs
            ]
            write_output(run, unit, dm, rows)
    return run, units
