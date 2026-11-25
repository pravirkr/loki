"""``tutorial``: a small simulated survey, ready to run end to end.

Four dedispersed series, at DM 0, 20, 40 and 60, hold white noise and three
sources, each at the S/N it would have at that trial DM:

- PSR_A, an isolated 142.3 Hz pulsar at DM 40, listed in ``catalogue.csv``;
- PSR_B, a 117.1 Hz pulsar at DM 20 under a 30 m/s^2 acceleration, not
  listed: the find;
- a 150 Hz sinusoid, interference that is strongest at DM 0, where the line
  finder sees it too.

The series are 2^18 samples of 64 us (16.8 s), short enough to search on a
laptop in about a minute. ``survey.toml`` searches 100-200 Hz with an FFA and
an EP acceleration box, and ``site.toml`` runs the units on this machine.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

NSAMPS = 1 << 18
TSAMP = 64e-6
DMS = (0.0, 20.0, 40.0, 60.0)
# A toy dedispersion loss: a source's S/N falls by sqrt(2) at this many
# pc/cm^3 from its own DM.
DM_WIDTH = 15.0
C_M_S = 299792458.0


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "pulsar" or "sine"
    freq: float  # Hz, at the middle of the series
    dm: float
    snr: float  # matched-filter S/N at its own DM
    accel: float = 0.0  # m/s^2
    duty: float = 0.05  # pulse FWHM as a fraction of the period
    catalogued: bool = False


SOURCES = (
    Source("PSR_A", "pulsar", freq=142.3, dm=40.0, snr=20.0, catalogued=True),
    Source("PSR_B", "pulsar", freq=117.1, dm=20.0, snr=18.0, accel=30.0),
    Source("RFI_150Hz", "sine", freq=150.0, dm=0.0, snr=25.0),
)

SURVEY = """\
# The tutorial survey: see docs/tutorial.md in loki-survey.
[survey]
name = "tutorial"

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

[evaluate]
catalogue = "catalogue.csv"
"""

SITE = """\
# This machine: up to {parallel} units at once.
[executor]
kind = "local"
max_memory_gb = {max_memory_gb}
max_threads = {max_threads}
python = "{python}"

[unit]
memory_gb = {unit_memory_gb}
overhead_gb = {unit_overhead_gb}
threads = {unit_threads}
"""

# A tutorial unit fits in this memory and thread count; the tutorial runs up
# to MAX_PARALLEL_UNITS of them at once, as this machine's threads allow.
UNIT_MEMORY_GB = 4.0
UNIT_OVERHEAD_GB = 1.0
UNIT_THREADS = 2
MAX_PARALLEL_UNITS = 2


def snr_at(source: Source, dm: float) -> float:
    """Return the S/N a source has in the series dedispersed at ``dm``."""
    return source.snr / np.sqrt(1.0 + ((dm - source.dm) / DM_WIDTH) ** 2)


def signal(source: Source, snr: float, t: np.ndarray) -> np.ndarray:
    """Return a source's samples at matched-filter S/N ``snr`` in unit noise."""
    # Times from the middle of the series, where loki reports frequencies.
    dt = t - NSAMPS * TSAMP / 2
    if source.kind == "sine":
        wave = np.sin(2 * np.pi * source.freq * dt)
    else:
        # A constant acceleration a lowers the frequency by f a dt / c.
        phase = source.freq * (dt - source.accel * dt**2 / (2 * C_M_S))
        offset = (phase + 0.5) % 1.0 - 0.5
        sigma = source.duty / (2 * np.sqrt(2 * np.log(2)))
        wave = np.exp(-0.5 * (offset / sigma) ** 2)
    return snr * wave / np.sqrt(np.sum(wave**2))


def series_name(dm: float) -> str:
    return f"tutorial_DM{dm:05.2f}"


def write_series(path: Path, ts_e: np.ndarray, dm: float) -> None:
    """Write a PRESTO .dat and .inf, with the series' DM in the .inf."""
    from loki import libloki  # noqa: PLC0415

    libloki.io.TimeSeries(ts_e, np.ones_like(ts_e), TSAMP).write(str(path))
    inf = path.with_suffix(".inf")
    text, count = re.subn(
        r"(Dispersion measure \(cm-3 pc\)\s*=\s*)\S+",
        rf"\g<1>{dm:.5f}",
        inf.read_text(),
    )
    if count != 1:
        msg = f"{inf}: no dispersion measure line to set"
        raise RuntimeError(msg)
    inf.write_text(text)


def write_tutorial(root: Path, seed: int = 1) -> list[Path]:
    """Write the tutorial's series, survey.toml, site.toml and catalogue.csv."""
    if (root / "survey.toml").exists():
        msg = f"{root / 'survey.toml'} exists; give the tutorial a new directory"
        raise FileExistsError(msg)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(NSAMPS) * TSAMP
    written = []
    for dm in DMS:
        ts = rng.standard_normal(NSAMPS)
        for source in SOURCES:
            ts += signal(source, snr_at(source, dm), t)
        path = data / f"{series_name(dm)}.dat"
        write_series(path, ts.astype(np.float32), dm)
        written.append(path)
    (root / "survey.toml").write_text(SURVEY)
    threads = os.cpu_count() or UNIT_THREADS
    parallel = max(1, min(MAX_PARALLEL_UNITS, threads // UNIT_THREADS))
    (root / "site.toml").write_text(
        SITE.format(
            parallel=parallel,
            max_memory_gb=parallel * UNIT_MEMORY_GB,
            max_threads=parallel * UNIT_THREADS,
            python=sys.executable,
            unit_memory_gb=UNIT_MEMORY_GB,
            unit_overhead_gb=UNIT_OVERHEAD_GB,
            unit_threads=UNIT_THREADS,
        ),
    )
    rows = [f"{s.name},{s.freq},{s.dm}" for s in SOURCES if s.catalogued]
    (root / "catalogue.csv").write_text("name,f0,dm\n" + "\n".join(rows) + "\n")
    return written
