"""The local executor's failure handling, with stub units in place of loki."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loki_survey.config import Resolved, Site
from loki_survey.executors.local import LocalExecutor
from loki_survey.rundir import RunDir
from loki_survey.status import CRASHED, OOM, StatusDB

from .conftest import make_unit, stub_python

if TYPE_CHECKING:
    from pathlib import Path

GB = 1_000_000_000


def run_stub(tmp_path: Path, body: str, *, memory_gb: float = 0.5) -> StatusDB:
    run = RunDir.at(tmp_path / "run")
    run.ensure()
    site = Site(
        executor="local",
        max_memory_gb=2.0,
        max_threads=2,
        unit_memory_gb=memory_gb,
        unit_overhead_gb=0.1,
        unit_threads=1,
        backend="cpu",
        device=0,
        python=stub_python(tmp_path, body),
        poll_s=0.05,
        oom_retries=1,
        oom_factor=1.5,
    )
    resolved = Resolved(survey=None, site=site, science_sha256="test")  # type: ignore[arg-type]
    units = [make_unit("u0")]
    db = StatusDB(run)
    db.sync(units, site.unit_memory_gb)
    LocalExecutor(run, resolved, db).submit(units)
    return db


def test_success_needs_the_summary(tmp_path: Path) -> None:
    body = (
        "import sys, pathlib\n"
        "run, uid = sys.argv[4], sys.argv[5]  # after -m loki_survey run-unit\n"
        "pathlib.Path(run, 'out', uid + '.json').write_text('{}')\n"
    )
    db = run_stub(tmp_path, body)
    assert db.get("u0").state == "done"


def test_crash_is_reported_not_retried(tmp_path: Path) -> None:
    db = run_stub(tmp_path, "print('loki: something broke')\nraise SystemExit(3)\n")
    state = db.get("u0")
    assert (state.state, state.failure, state.attempts) == ("failed", CRASHED, 1)
    _, _, _, detail = db.failures()[0]
    assert "something broke" in detail


def test_out_of_memory_retries_with_more_then_fails(tmp_path: Path) -> None:
    allocate = (
        "import time\n"
        f"b = bytearray({int(1.2 * GB)})\n"
        "for i in range(0, len(b), 4096):\n"
        "    b[i] = 1\n"
        "time.sleep(60)\n"
    )
    db = run_stub(tmp_path, allocate, memory_gb=0.5)
    state = db.get("u0")
    assert (state.state, state.failure, state.attempts) == ("failed", OOM, 2)
    rows = db.conn.execute(
        "SELECT attempt, memory_gb, failure FROM attempts ORDER BY attempt",
    ).fetchall()
    assert rows == [(1, 0.5, OOM), (2, 0.75, OOM)]
    assert not any((tmp_path / "run" / "tmp").iterdir())
