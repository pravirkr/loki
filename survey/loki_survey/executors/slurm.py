"""Slurm: submit arrays with ``sbatch``; read them with squeue, scontrol, sacct.

``squeue`` lists live elements. ``scontrol`` still holds an element for the
cluster's MinJobAge after it ends; ``sacct`` (Slurm's accounting) holds it
after that, with its peak memory. ``slurmstepd`` also writes why it killed a
job (memory, time limit, preemption) into the job's output, which settles a
record that only says the job was cancelled. An element none of them knows
has ended without a trace and is called lost.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from loki_survey.executors.batch import (
    ENDED,
    KILL,
    MEMLIMIT,
    NODE,
    PENDING,
    PREEMPT,
    RUNLIMIT,
    RUNNING,
    SUSPENDED,
    ArrayRequest,
    JobRecord,
    SchedulerError,
    gb_to_units,
    run_command,
)

if TYPE_CHECKING:
    from loki_survey.config import Site
    from loki_survey.rundir import RunDir
    from loki_survey.status import Attempt

MIB = 1 << 20
RSS_BYTES = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
STATES = {
    "PENDING": (PENDING, None),
    "REQUEUED": (PENDING, None),
    "REQUEUE_HOLD": (PENDING, None),
    "REQUEUE_FED": (PENDING, None),
    "RESV_DEL_HOLD": (PENDING, None),
    "CONFIGURING": (RUNNING, None),
    "RUNNING": (RUNNING, None),
    "COMPLETING": (RUNNING, None),
    "STAGE_OUT": (RUNNING, None),
    "SIGNALING": (RUNNING, None),
    "SUSPENDED": (SUSPENDED, None),
    "STOPPED": (SUSPENDED, None),
    "COMPLETED": (ENDED, None),
    "FAILED": (ENDED, None),
    "OUT_OF_MEMORY": (ENDED, MEMLIMIT),
    "TIMEOUT": (ENDED, RUNLIMIT),
    "DEADLINE": (ENDED, RUNLIMIT),
    "PREEMPTED": (ENDED, PREEMPT),
    "CANCELLED": (ENDED, KILL),
    "NODE_FAIL": (ENDED, NODE),
    "BOOT_FAIL": (ENDED, NODE),
}
ELEMENT = re.compile(r"^(\d+)_(\d+)(?:\.(\S+))?$")
# What slurmstepd writes into a job's output when it kills the job.
STEPD_REASONS = (
    (re.compile(r"oom-kill|Exceeded job memory limit|exceeded memory limit"), MEMLIMIT),
    (re.compile(r"DUE TO TIME LIMIT"), RUNLIMIT),
    (re.compile(r"DUE TO PREEMPTION"), PREEMPT),
    (re.compile(r"DUE TO NODE FAILURE"), NODE),
)
# How long sacct looks back when finding an array by name.
LOOKUP_WINDOW_S = 2 * 24 * 3600


def parse_rss_gb(text: str) -> float | None:
    """Sacct MaxRSS such as '123456K' or '2.5G' in GB (10^9 bytes)."""
    match = re.fullmatch(r"([\d.]+)([KMGT]?)", text.strip())
    if not match:
        return None
    return float(match.group(1)) * RSS_BYTES.get(match.group(2), 1) / 1e9


def parse_exit(text: str) -> int | None:
    """Sacct ExitCode 'code:signal' as a shell exit code."""
    match = re.fullmatch(r"(\d+):(\d+)", text.strip())
    if not match:
        return None
    code, signal = int(match.group(1)), int(match.group(2))
    return 128 + signal if signal else code


def parse_squeue(text: str) -> dict[tuple[str, int], JobRecord]:
    """Read ``squeue -h -r -o '%i|%T|%N'`` into records."""
    records = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        match = ELEMENT.match(parts[0]) if len(parts) == 3 else None
        if match is None:
            continue
        state, reason = STATES.get(parts[1], (RUNNING, None))
        records[(match.group(1), int(match.group(2)))] = JobRecord(
            state=state,
            reason=reason,
            host=parts[2] or None,
            text=parts[1],
        )
    return records


def _indices(text: str) -> list[int]:
    """Expand an ArrayTaskId such as '3', '4-7%2' or '1,3-5'."""
    out = []
    for part in text.split("%", maxsplit=1)[0].split(","):
        lo, _, hi = part.partition("-")
        if lo.isdigit():
            out += range(int(lo), int(hi or lo) + 1)
    return out


def parse_scontrol(text: str) -> dict[tuple[str, int], JobRecord]:
    """Read ``scontrol -o show job`` (one job per line) into records."""
    records = {}
    for line in text.splitlines():
        fields = dict(re.findall(r"(\w+)=(\S*)", line))
        base, tasks = fields.get("ArrayJobId"), fields.get("ArrayTaskId")
        if not base or not tasks:
            continue
        state, reason = STATES.get(fields.get("JobState", ""), (ENDED, None))
        host = fields.get("NodeList", "")
        for index in _indices(tasks):
            records[(base, index)] = JobRecord(
                state=state,
                exit_code=parse_exit(fields.get("ExitCode", "")),
                reason=reason,
                host=host if host and host != "(null)" else None,
                text=f"{fields.get('JobState')} ({fields.get('Reason', '')})",
            )
    return records


def parse_sacct(text: str) -> dict[tuple[str, int], JobRecord]:
    """Read ``sacct -n -P -o JobID,State,ExitCode,NodeList,MaxRSS`` into records.

    An element's steps (``.batch``, ``.extern``) are merged into it: Slurm may
    report the out-of-memory kill on the batch step only.
    """
    rows: dict[tuple[str, int], list[list[str]]] = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5:
            continue
        match = ELEMENT.match(parts[0])
        if not match:
            continue
        rows.setdefault((match.group(1), int(match.group(2))), []).append(
            [match.group(3) or "", *parts[1:]],
        )
    records = {}
    for key, steps in rows.items():
        main = next((s for s in steps if not s[0]), steps[0])
        word = main[1].split()[0] if main[1] else ""
        state, reason = STATES.get(word, (ENDED, None))
        if any(s[1].startswith("OUT_OF_MEMORY") for s in steps):
            state, reason = ENDED, MEMLIMIT
        peaks = [p for s in steps if (p := parse_rss_gb(s[4])) is not None]
        records[key] = JobRecord(
            state=state,
            exit_code=parse_exit(main[2]),
            reason=reason,
            host=main[3] if main[3] not in {"", "None assigned"} else None,
            max_mem_gb=max(peaks) if peaks else None,
            text=main[1],
        )
    return records


class SlurmScheduler:
    def __init__(self, run: RunDir, site: Site) -> None:
        if site.batch is None:
            msg = "Slurm needs a [slurm] table in site.toml"
            raise ValueError(msg)
        self.run = run
        self.batch = site.batch

    def sbatch_args(self, request: ArrayRequest) -> list[str]:
        batch = self.batch
        cap = f"%{request.max_running}" if request.max_running else ""
        out = self.run.sched_logs_dir / f"{request.name}.%A.%a.out"
        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={request.name}",
            f"--array=1-{request.n}{cap}",
            "--ntasks=1",
            f"--cpus-per-task={request.threads}",
            f"--mem={gb_to_units(request.memory_gb, MIB)}M",
            f"--output={out}",
        ]
        if batch.queue:
            argv.append(f"--partition={batch.queue}")
        if batch.account:
            argv.append(f"--account={batch.account}")
        if batch.time_min:
            argv.append(f"--time={batch.time_min}")
        if batch.constraint:
            argv.append(f"--constraint={batch.constraint}")
        if request.exclude_hosts:
            argv.append(f"--exclude={','.join(request.exclude_hosts)}")
        if request.gpu:
            argv.append(f"--gres={request.gpu}")
        return [*argv, *batch.extra, str(request.script)]

    def submit_once(
        self,
        request: ArrayRequest,
        timeout_s: float,
    ) -> tuple[str | None, str]:
        argv = self.sbatch_args(request)
        done = run_command(argv, timeout_s)
        if done is None:
            return None, f"{argv[0]}: no answer within {timeout_s:.0f} s"
        text = (done.stdout + done.stderr).strip()
        match = re.match(r"(\d+)", done.stdout.strip())
        if done.returncode != 0 or not match:
            return None, text
        return match.group(1), text

    def lookup(self, name: str) -> list[str]:
        found = set()
        done = run_command(["squeue", "-h", f"--name={name}", "-o", "%A"])
        if done is not None and done.returncode == 0:
            found |= {w for w in done.stdout.split() if w.isdigit()}
        since = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime(time.time() - LOOKUP_WINDOW_S),
        )
        done = run_command(
            ["sacct", "-n", "-P", "-X", f"--name={name}", "-S", since, "-o", "JobID"],
        )
        if done is not None and done.returncode == 0:
            found |= {
                m.group(1) for m in re.finditer(r"^(\d+)", done.stdout, re.MULTILINE)
            }
        return sorted(found)

    def poll(self, jobs: list[str]) -> dict[tuple[str, int], JobRecord]:
        ids = ",".join(jobs)
        done = run_command(["squeue", "-h", "-r", "-j", ids, "-o", "%i|%T|%N"])
        if done is None:
            msg = "squeue is missing or did not answer"
            raise SchedulerError(msg)
        if done.returncode != 0 and "Invalid job id" not in done.stderr:
            msg = f"squeue failed: {done.stderr.strip()[:200]}"
            raise SchedulerError(msg)
        live = parse_squeue(done.stdout) if done.returncode == 0 else {}
        recent: dict[tuple[str, int], JobRecord] = {}
        for job in jobs:
            done = run_command(["scontrol", "-o", "show", "job", job])
            if done is not None and done.returncode == 0:
                recent.update(parse_scontrol(done.stdout))
        done = run_command(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                ids,
                "-o",
                "JobID,State,ExitCode,NodeList,MaxRSS",
            ],
        )
        accounted = parse_sacct(done.stdout) if done and done.returncode == 0 else {}
        records = {}
        for key in {*accounted, *recent, *live}:
            record = live.get(key) or recent.get(key) or accounted[key]
            if record.state == ENDED:
                record = self._settle(key, record, accounted.get(key))
            records[key] = record
        return records

    def _settle(
        self,
        key: tuple[str, int],
        record: JobRecord,
        accounted: JobRecord | None,
    ) -> JobRecord:
        """Add accounting's peak memory, and slurmstepd's word on a kill."""
        peak = accounted.max_mem_gb if accounted else record.max_mem_gb
        reason = record.reason
        if accounted is not None and accounted.reason == MEMLIMIT:
            reason = MEMLIMIT
        if reason in {None, KILL}:
            for out in self.run.sched_logs_dir.glob(f"*.{key[0]}.{key[1]}.out"):
                text = out.read_text(errors="replace")
                for pattern, said in STEPD_REASONS:
                    if pattern.search(text):
                        reason = said
                        break
        return JobRecord(
            state=record.state,
            exit_code=record.exit_code,
            reason=reason,
            host=record.host,
            max_mem_gb=peak,
            text=record.text,
        )

    def ended_record(self, attempt: Attempt) -> JobRecord | None:  # noqa: ARG002
        return None

    def cancel(self, job: str, index: int | None, *, force: bool = False) -> None:  # noqa: ARG002
        run_command(["scancel", job if index is None else f"{job}_{index}"])
