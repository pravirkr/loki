"""The LSF and Slurm executors against a fake scheduler, with stub units.

The parsers are checked on output in the formats wexac's LSF 10.1 prints;
each failure class is then driven end to end through ``fake_scheduler.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from loki_survey.config import Batch, Resolved, Site
from loki_survey.executors.batch import (
    ENDED,
    KILL,
    MEMLIMIT,
    NODE,
    PENDING,
    RUNLIMIT,
    RUNNING,
    SUSPENDED,
    ArrayRequest,
    BatchExecutor,
    JobRecord,
    classify,
)
from loki_survey.executors.lsf import LsfScheduler, parse_bjobs, parse_footer
from loki_survey.executors.slurm import (
    SlurmScheduler,
    parse_sacct,
    parse_scontrol,
    parse_squeue,
)
from loki_survey.rundir import RunDir
from loki_survey.status import (
    CANCELLED,
    CRASHED,
    HOST,
    KILLED,
    LOST,
    OOM,
    PREEMPTED,
    STALLED,
    TIMEOUT,
    StatusDB,
)

from .conftest import make_unit, stub_python

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import Fake

# Writes the started marker as run-unit does, then succeeds unless the run's
# behaviour.json says this unit crashes.
STUB = """
import json, os, pathlib, sys
run, uid, attempt = pathlib.Path(sys.argv[4]), sys.argv[5], sys.argv[7]
marker = run / "logs" / f"{uid}.{attempt}.started"
marker.write_text(json.dumps({"host": os.environ.get("FAKE_HOST")}))
behaviour = run / "behaviour.json"
how = json.loads(behaviour.read_text()).get(uid) if behaviour.is_file() else None
if how == "crash":
    print("loki: something broke")
    sys.exit(3)
(run / "out" / f"{uid}.json").write_text("{}")
"""

BJOBS = {
    "COMMAND": "bjobs",
    "JOBS": 4,
    "RECORDS": [
        {
            "JOBID": "7",
            "JOBINDEX": "3",
            "STAT": "RUN",
            "EXIT_CODE": "",
            "EXIT_REASON": "",
            "EXEC_HOST": "2*hgn57",
            "MAX_MEM": "159.3 Gbytes",
        },
        {
            "JOBID": "7",
            "JOBINDEX": "4",
            "STAT": "EXIT",
            "EXIT_CODE": "137",
            "EXIT_REASON": "TERM_MEMLIMIT: job killed after reaching LSF memory "
            "usage limit",
            "EXEC_HOST": "1*cn111:2*cn817:1*cn820",
            "MAX_MEM": "1.9 Gbytes",
        },
        {
            "JOBID": "7",
            "JOBINDEX": "5",
            "STAT": "SSUSP",
            "EXIT_CODE": "",
            "EXIT_REASON": "",
            "EXEC_HOST": "hgn45",
            "MAX_MEM": "321 Mbytes",
        },
        {"JOBID": "8", "ERROR": "Job <8> is not found"},
    ],
}
FOOTER_OK = """
Sender: LSF System <DoNotReply>
Subject: Job 392579: <crackle_test2> in cluster <wexac> Done

Job was executed on host(s) <4*hgn19>, in queue <short-gpu>, as user <me>.
Successfully completed.

Resource usage summary:

    CPU time :                                   826.00 sec.
    Max Memory :                                 12786 MB
"""
FOOTER_OOM = """
Sender: LSF System <DoNotReply>
Job was executed on host(s) <cn409>, in queue <physics-short>, as user <me>.
TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.
Exited with exit code 130.

    Max Memory :                                 182272 MB
"""
SACCT = """\
41_1|COMPLETED|0:0|n1|
41_1.batch|COMPLETED|0:0|n1|2.5G
41_2|FAILED|0:9|n2|
41_2.batch|OUT_OF_MEMORY|0:9|n2|1048576K
41_3|CANCELLED by 1000|0:15|n1|
41_[4-9%2]|PENDING|0:0|None assigned|
"""


def test_bjobs_records() -> None:
    records = parse_bjobs(json.dumps(BJOBS))
    assert set(records) == {("7", 3), ("7", 4), ("7", 5)}
    running = records["7", 3]
    assert (running.state, running.host) == (RUNNING, "hgn57")
    assert running.max_mem_gb == pytest.approx(159.3 * (1 << 30) / 1e9)
    oom = records["7", 4]
    assert (oom.state, oom.exit_code, oom.reason, oom.host) == (
        ENDED,
        137,
        MEMLIMIT,
        "cn111",
    )
    assert records["7", 5].state == SUSPENDED


def test_lsf_footers() -> None:
    ok = parse_footer(FOOTER_OK)
    assert (ok.state, ok.exit_code, ok.reason, ok.host) == (ENDED, 0, None, "hgn19")
    assert ok.max_mem_gb == pytest.approx(12786 * (1 << 20) / 1e9)
    oom = parse_footer(FOOTER_OOM)
    assert (oom.exit_code, oom.reason, oom.host) == (130, MEMLIMIT, "cn409")
    assert parse_footer("still running, no footer yet") is None
    # A requeued element appends a footer per run; the last one counts.
    assert parse_footer(FOOTER_OOM + FOOTER_OK).exit_code == 0


def test_slurm_records() -> None:
    ended = parse_sacct(SACCT)
    assert set(ended) == {("41", 1), ("41", 2), ("41", 3)}
    assert (ended["41", 1].exit_code, ended["41", 1].reason) == (0, None)
    assert ended["41", 1].max_mem_gb == pytest.approx(2.5 * (1 << 30) / 1e9)
    # The out-of-memory kill is on the batch step only.
    assert (ended["41", 2].reason, ended["41", 2].exit_code) == (MEMLIMIT, 137)
    assert ended["41", 3].reason == KILL
    live = parse_squeue("41_5|RUNNING|n3\n41_6|PENDING|\n41_[7-9]|PENDING|\n")
    assert set(live) == {("41", 5), ("41", 6)}
    assert live["41", 5].state == RUNNING
    assert live["41", 5].host == "n3"


def test_scontrol_records() -> None:
    """Slurm 20.02's format, as the container check printed it."""
    text = (
        "JobId=17 ArrayJobId=17 ArrayTaskId=3 JobName=r.0004 JobState=TIMEOUT "
        "Reason=TimeLimit ExitCode=0:15 NodeList=node-3 BatchHost=node-3\n"
        "JobId=19 ArrayJobId=17 ArrayTaskId=4-6%2 JobState=PENDING "
        "Reason=JobArrayTaskLimit ExitCode=0:0 NodeList=(null)\n"
    )
    records = parse_scontrol(text)
    assert set(records) == {("17", i) for i in (3, 4, 5, 6)}
    ended = records["17", 3]
    assert (ended.state, ended.reason, ended.exit_code, ended.host) == (
        ENDED,
        RUNLIMIT,
        128 + 15,
        "node-3",
    )
    assert records["17", 5].state == PENDING
    assert records["17", 5].host is None


@pytest.mark.parametrize(
    ("record", "began", "cancel", "expected"),
    [
        (JobRecord(ENDED, 130, MEMLIMIT), True, False, OOM),
        (JobRecord(ENDED, 140, RUNLIMIT), True, False, TIMEOUT),
        (JobRecord(ENDED, 130, KILL), True, False, KILLED),
        (JobRecord(ENDED, 130, KILL), True, True, CANCELLED),
        (JobRecord(ENDED, 1, NODE), False, False, HOST),
        (JobRecord(ENDED, 255), False, False, HOST),
        (JobRecord(ENDED, 132), True, False, HOST),
        (JobRecord(ENDED, 3), True, False, CRASHED),
        (JobRecord(ENDED, 0), True, False, CRASHED),
        (JobRecord(ENDED, None), False, False, LOST),
    ],
)
def test_classes(record: JobRecord, began: bool, cancel: bool, expected: str) -> None:
    assert classify(record, began=began, cancel_requested=cancel)[0] == expected


def batch(**changes: object) -> Batch:
    fields = {
        "queue": None,
        "account": None,
        "time_min": None,
        "select": None,
        "constraint": None,
        "hosts": (),
        "gpu": None,
        "mem_unit": "MB",
        "reserve_per_task": True,
        "max_array": 1000,
        "extra": (),
    }
    return Batch(**{**fields, **changes})


def site(kind: str, python: str, **changes: object) -> Site:
    fields = {
        "executor": kind,
        "max_memory_gb": 2.0,
        "max_threads": 1,
        "unit_memory_gb": 0.5,
        "unit_overhead_gb": 0.1,
        "unit_threads": 1,
        "backend": "cpu",
        "device": 0,
        "python": python,
        "poll_s": 0.05,
        "oom_retries": 1,
        "oom_factor": 1.5,
        "submit_timeout_s": 5.0,
        "requeue_retries": 2,
        "host_failures": 2,
        "stall_min": 0.0,
        "batch": batch(),
    }
    return Site(**{**fields, **changes})


def test_bsub_and_sbatch_arguments(tmp_path: Path) -> None:
    run = RunDir.at(tmp_path)
    request = ArrayRequest(
        name="r.0001",
        script=tmp_path / "r.0001.sh",
        n=3,
        memory_gb=24.0,
        threads=4,
        gpu=None,
        exclude_hosts=("cn800", "cn801"),
        max_running=50,
    )
    lsf = LsfScheduler(
        run,
        site("lsf", "python", batch=batch(queue="short", select="model==X")),
    )
    argv = lsf.bsub_args(request)
    total_mb = 22889  # 24e9 bytes in MiB, rounded up
    assert argv[argv.index("-J") + 1] == "r.0001[1-3]%50"
    assert argv[argv.index("-M") + 1] == str(total_mb)
    assert argv[argv.index("-R") + 1] == (
        f"span[hosts=1] rusage[mem={-(-total_mb // 4)}] "
        "select[(model==X) && hname!=cn800 && hname!=cn801]"
    )
    assert argv[argv.index("-q") + 1] == "short"
    slurm = SlurmScheduler(run, site("slurm", "python", batch=batch(queue="gen")))
    argv = slurm.sbatch_args(request)
    assert f"--mem={total_mb}M" in argv
    assert "--array=1-3%50" in argv
    assert "--exclude=cn800,cn801" in argv
    assert "--partition=gen" in argv


def drive(
    tmp_path: Path,
    kind: str = "lsf",
    *,
    n: int = 1,
    crash: tuple[str, ...] = (),
    **changes: object,
) -> tuple[StatusDB, BatchExecutor]:
    """Run ``n`` stub units to the end through the fake scheduler."""
    run = RunDir.at(tmp_path / "run")
    run.ensure()
    (run.root / "behaviour.json").write_text(json.dumps(dict.fromkeys(crash, "crash")))
    resolved = Resolved(
        survey=None,  # type: ignore[arg-type]
        site=site(kind, stub_python(tmp_path, STUB), **changes),
        science_sha256="test",
    )
    units = [make_unit(f"u{i}") for i in range(n)]
    db = StatusDB(run)
    db.sync(units, resolved.site.unit_memory_gb)
    kinds = {"lsf": LsfScheduler, "slurm": SlurmScheduler}
    executor = BatchExecutor(run, resolved, db, kinds[kind](run, resolved.site))
    executor.watch(units)
    return db, executor


def history(db: StatusDB, uid: str) -> list[tuple]:
    return db.conn.execute(
        "SELECT attempt, memory_gb, failure, host FROM attempts WHERE unit = ? "
        "ORDER BY attempt",
        (uid,),
    ).fetchall()


@pytest.mark.parametrize("kind", ["lsf", "slurm"])
def test_units_run_to_the_end(tmp_path: Path, fake: Fake, kind: str) -> None:
    fake.control(hosts=["h1", "h2"])
    db, _ = drive(tmp_path, kind, n=3)
    assert db.counts() == {"done": 3}
    assert len(fake.jobs()) == 1  # one array for units of one size


@pytest.mark.parametrize("kind", ["lsf", "slurm"])
def test_out_of_memory_is_resubmitted_with_more(
    tmp_path: Path,
    fake: Fake,
    kind: str,
) -> None:
    fake.control(need_gb={"u0": 0.7})
    db, _ = drive(tmp_path, kind)
    assert db.get("u0").state == "done"
    assert [(a, m, f) for a, m, f, _ in history(db, "u0")] == [
        (1, 0.5, OOM),
        (2, 0.75, None),
    ]


def test_out_of_memory_read_from_the_footer(tmp_path: Path, fake: Fake) -> None:
    """Bjobs has forgotten the job (CLEAN_PERIOD); its footer still says why."""
    fake.control(need_gb={"u0": 0.7}, forget=["u0:1"])
    db, _ = drive(tmp_path)
    assert [f for _, _, f, _ in history(db, "u0")] == [OOM, None]


def test_out_of_memory_stops_at_the_retry_limit(tmp_path: Path, fake: Fake) -> None:
    fake.control(need_gb={"u0": 5.0})
    db, _ = drive(tmp_path)
    state = db.get("u0")
    assert (state.state, state.failure, state.attempts) == ("failed", OOM, 2)


@pytest.mark.parametrize("how", ["kill", "preempt"])
def test_killed_or_preempted_job_is_resubmitted(
    tmp_path: Path,
    fake: Fake,
    how: str,
) -> None:
    fake.control(**{how: ["u0:1"]})
    db, _ = drive(tmp_path)
    expected = KILLED if how == "kill" else PREEMPTED
    assert [f for _, _, f, _ in history(db, "u0")] == [expected, None]


@pytest.mark.parametrize("kind", ["lsf", "slurm"])
def test_bad_host_is_excluded(tmp_path: Path, fake: Fake, kind: str) -> None:
    fake.control(hosts=["h1", "h2"], bad_hosts=["h1"])
    db, _ = drive(tmp_path, kind, n=4)
    assert db.counts() == {"done": 4}
    assert db.excluded_hosts() == ["h1"]
    failed_on = {h for u in ("u0", "u2") for _, _, f, h in history(db, u) if f}
    assert failed_on == {"h1"}


def test_failure_on_every_host_excludes_none(tmp_path: Path, fake: Fake) -> None:
    """A broken prologue is not a host fault: no unit ever starts anywhere."""
    fake.control(hosts=["h1", "h2"])
    db, _ = drive(tmp_path, n=2, prologue="exit 3")
    assert db.excluded_hosts() == []
    state = db.get("u0")
    assert (state.state, state.failure, state.attempts) == ("failed", HOST, 3)
    assert "before run-unit started" in db.failures()[0][3]


def test_crash_is_reported_not_retried(tmp_path: Path, fake: Fake) -> None:
    fake.control()
    db, executor = drive(tmp_path, n=2, crash=("u1",))
    assert db.get("u0").state == "done"
    state = db.get("u1")
    assert (state.state, state.failure, state.attempts) == ("failed", CRASHED, 1)
    assert "something broke" in db.failures()[0][3]
    # Once the cause is fixed, an explicit submit runs it again.
    (tmp_path / "run" / "behaviour.json").write_text("{}")
    assert db.retry_failed(["u0", "u1"]) == 1
    executor.watch([make_unit("u0"), make_unit("u1")])
    assert db.counts() == {"done": 2}


def test_time_limit_is_reported(tmp_path: Path, fake: Fake) -> None:
    fake.control(timeout=["u0:1"])
    db, _ = drive(tmp_path)
    state = db.get("u0")
    assert (state.state, state.failure) == ("failed", TIMEOUT)


def test_stalled_job_is_killed_and_resubmitted(tmp_path: Path, fake: Fake) -> None:
    fake.control(hang=["u0:1"])
    db, _ = drive(tmp_path, stall_min=0.005)
    assert [f for _, _, f, _ in history(db, "u0")] == [STALLED, None]
    element = fake.jobs()["101"]["elements"]["1"]
    assert element["how"] == "kill"  # the stalled job was killed


def test_lost_job_is_resubmitted(tmp_path: Path, fake: Fake) -> None:
    fake.control(vanish=["u0:1"])
    db, _ = drive(tmp_path)
    assert [f for _, _, f, _ in history(db, "u0")] == [LOST, None]


def test_busy_scheduler_makes_no_duplicate(tmp_path: Path, fake: Fake) -> None:
    """The first bsub reports a failure yet queues the array (mbatchd busy)."""
    fake.control(busy=1)
    db, _ = drive(tmp_path, n=2)
    assert db.counts() == {"done": 2}
    assert len(fake.jobs()) == 1


def test_cancel_returns_units_to_pending(tmp_path: Path, fake: Fake) -> None:
    fake.control(hang=["u0:1"])
    run = RunDir.at(tmp_path / "run")
    run.ensure()
    resolved = Resolved(
        survey=None,  # type: ignore[arg-type]
        site=site("lsf", stub_python(tmp_path, STUB)),
        science_sha256="test",
    )
    units = [make_unit("u0")]
    db = StatusDB(run)
    db.sync(units, resolved.site.unit_memory_gb)
    executor = BatchExecutor(run, resolved, db, LsfScheduler(run, resolved.site))
    executor.submit(units)
    executor.reconcile()
    assert db.get("u0").state == "running"
    assert executor.cancel() == 1
    executor.reconcile()
    state = db.get("u0")
    assert (state.state, state.failure) == ("pending", CANCELLED)
