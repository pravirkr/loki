from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from loki_survey.plan import Unit

# Units run as subprocesses; this is the Python they use. It defaults to the
# Python running the tests, which must import the same loki.
UNIT_PYTHON = os.environ.get("LOKI_SURVEY_TEST_PYTHON", sys.executable)


FAKE_SCHEDULER = Path(__file__).with_name("fake_scheduler.py")
SCHEDULER_COMMANDS = (
    "bsub",
    "bjobs",
    "bkill",
    "sbatch",
    "squeue",
    "scontrol",
    "sacct",
    "scancel",
)


@pytest.fixture
def unit_python() -> str:
    return UNIT_PYTHON


def write_site(
    run: Path,
    python: str,
    *,
    kind: str = "local",
    max_memory_gb: float = 8.0,
    memory_gb: float = 4.0,
    overhead_gb: float = 1.0,
    oom_retries: int = 1,
) -> None:
    """Write a site.toml for this machine, or for the fake LSF."""
    machine = "max_threads = 4" if kind == "local" else ""
    scheduler = (
        '[lsf]\nmem_unit = "MB"\nreserve_per_task = true\n' if kind == "lsf" else ""
    )
    (run / "site.toml").write_text(
        f"""
[executor]
kind = "{kind}"
max_memory_gb = {max_memory_gb}
{machine}
python = "{python}"
poll_s = 0.2

[unit]
memory_gb = {memory_gb}
overhead_gb = {overhead_gb}
threads = 2

[retry]
oom_retries = {oom_retries}
oom_factor = 1.5

{scheduler}""",
    )


def make_unit(uid: str) -> Unit:
    """Return a small FFA unit; the stub tests never search it."""
    return Unit(
        id=uid,
        series="s",
        region="r",
        band=0,
        box="b",
        kind="ffa",
        order="freq",
        downsample=1,
        nsamps=1024,
        tsamp=1e-3,
        f_min=1.0,
        f_max=2.0,
        nbins=32,
        ducy_max=0.2,
        param_limits=[[1.0, 2.0]],
        use_fourier=None,
        segments=None,
        bseg_brute=None,
        ref_segs=None,
        orbit=None,
        plan_key=None,
    )


def stub_python(tmp_path: Path, body: str) -> str:
    """Write a 'python' that takes the run-unit arguments and runs ``body``."""
    script = tmp_path / "stub.py"
    script.write_text(body)
    wrapper = tmp_path / "stub-python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


class Fake:
    """The fake scheduler's control file and job table."""

    def __init__(self, state: Path) -> None:
        self.state = state

    def control(self, **values: object) -> None:
        (self.state / "control.json").write_text(json.dumps(values))

    def jobs(self) -> dict:
        path = self.state / "jobs.json"
        return json.loads(path.read_text()) if path.is_file() else {}


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in SCHEDULER_COMMANDS:
        wrapper = bindir / name
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SCHEDULER}" {name} "$@"\n',
        )
        wrapper.chmod(0o755)
    state = tmp_path / "sched"
    state.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SCHED_DIR", str(state))
    return Fake(state)
