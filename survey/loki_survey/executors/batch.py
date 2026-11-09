"""Cluster executors: units as job arrays on LSF or Slurm, driven by polling.

The driver never chains jobs on one another (a dependency on a job that
failed pends forever). It submits arrays, then polls: each poll reads the
scheduler's record of every live attempt, classes each failure and decides
what to do with it. Out-of-memory units are resubmitted with more memory;
preempted, killed, lost and stalled units, and units on a failing host, are
resubmitted as they were; a unit that hits its time limit is split into two
half bands (split.py), as many times over as the site allows; a crash is
reported, not retried.

A job runs ``jobs/<array>.sh``, which picks its unit from ``jobs/<array>.txt``
by array index, touches a heartbeat file while it runs, and calls
``run-unit``. ``run-unit`` writes a started marker first, so a job that died
before it (a broken node, a failing prologue) is told apart from a unit that
crashed.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from loki_survey.config import ConfigError
from loki_survey.executors.local import log_tail
from loki_survey.plan import build_ep_plan, load_units
from loki_survey.rundir import read_json, sha256_text
from loki_survey.split import expand, load_splits, record_split, split_depth, split_unit
from loki_survey.status import (
    CANCELLED,
    CRASHED,
    HOST,
    KILLED,
    LOST,
    OOM,
    PREEMPTED,
    REQUEUED,
    SPLIT,
    STALLED,
    TIMEOUT,
    Attempt,
    StatusDB,
)

if TYPE_CHECKING:
    from pathlib import Path

    from loki_survey.config import Resolved
    from loki_survey.plan import Unit
    from loki_survey.rundir import RunDir

# Normalised job states.
PENDING = "pending"
RUNNING = "running"
SUSPENDED = "suspended"
UNKNOWN = "unknown"  # the scheduler lost touch with the host; the job may live
ENDED = "ended"
# Normalised end reasons from the scheduler's own record.
MEMLIMIT = "memlimit"
RUNLIMIT = "runlimit"
PREEMPT = "preempt"
KILL = "kill"
NODE = "node"
# What each scheduler end reason means for the unit.
REASON_CLASSES = {
    MEMLIMIT: OOM,
    RUNLIMIT: TIMEOUT,
    PREEMPT: PREEMPTED,
    KILL: KILLED,
    NODE: HOST,
}
# Signals that point at the node rather than the unit: SIGILL (an instruction
# the CPU lacks) and SIGBUS (a hardware or file-system fault), as shell exit
# codes 128 + signal.
HOST_EXIT_CODES = (128 + 4, 128 + 7)
# A stall is this many heartbeats missed in a row.
HEARTBEATS_PER_STALL = 6
# A scheduler may not list a job it accepted moments ago, and a network file
# system may show a finished job's files only up to a minute after it ends
# (NFS caches a directory's attributes for up to 60 s). Wait this many polls
# before deciding from what is missing.
SETTLE_POLLS = 3
ARRAY_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
BYTES_PER_GB = 1e9


class SchedulerError(RuntimeError):
    """The scheduler could not be reached; nothing is decided this poll."""


class SubmitError(RuntimeError):
    """An array could not be submitted before the site's submit timeout."""


@dataclass(frozen=True)
class JobRecord:
    """One array element as the scheduler reports it."""

    state: str
    exit_code: int | None = None
    reason: str | None = None
    host: str | None = None
    max_mem_gb: float | None = None
    text: str = ""


@dataclass(frozen=True)
class ArrayRequest:
    name: str
    script: Path
    n: int
    memory_gb: float
    threads: int
    gpu: str | None
    exclude_hosts: tuple[str, ...]
    max_running: int


class Scheduler(Protocol):
    def submit_once(
        self,
        request: ArrayRequest,
        timeout_s: float,
    ) -> tuple[str | None, str]: ...

    def lookup(self, name: str) -> list[str]: ...

    def poll(self, jobs: list[str]) -> dict[tuple[str, int], JobRecord]: ...

    def ended_record(self, attempt: Attempt) -> JobRecord | None: ...

    def cancel(self, job: str, index: int | None, *, force: bool = False) -> None: ...


def run_command(
    argv: list[str],
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run a scheduler command; None if it is missing or does not answer in time."""
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def gb_to_units(gb: float, unit_bytes: int) -> int:
    """Express ``gb`` (10^9 bytes) in a scheduler's memory unit, rounded up."""
    return math.ceil(gb * BYTES_PER_GB / unit_bytes)


def run_tag(run: RunDir) -> str:
    """Return a job-name prefix that tells this run's arrays from any other's."""
    slug = ARRAY_NAME.sub("-", run.root.name)[:16] or "run"
    return f"{slug}-{sha256_text(str(run.root))[:6]}"


def job_script(run: RunDir, resolved: Resolved, name: str) -> str:
    """Write out the shell script every element of array ``name`` runs."""
    site = resolved.site
    q = shlex.quote
    units = q(str(run.jobs_dir / f"{name}.txt"))
    logs = q(str(run.logs_dir))
    threads = site.unit_threads
    lines = [
        "#!/bin/bash",
        f"# loki-survey array {name}: element i runs line i of {name}.txt.",
        'idx="${LSB_JOBINDEX:-${SLURM_ARRAY_TASK_ID:-}}"',
        'if [ -z "$idx" ]; then echo "no array index" >&2; exit 2; fi',
        f'read -r uid attempt < <(sed -n "${{idx}}p" {units})',
    ]
    if site.stall_min > 0:
        beat_s = max(1, int(site.stall_min * 60 / HEARTBEATS_PER_STALL))
        lines += [
            "# Heartbeat: touch a file while the job lives.",
            f'alive={logs}/"$uid.$attempt.alive"',
            "( trap 'kill $s 2>/dev/null; exit 0' TERM",
            f'  while :; do touch "$alive"; sleep {beat_s} & s=$!; wait $s; done ) &',
            "beat=$!",
            "trap 'kill $beat 2>/dev/null' EXIT",
        ]
    if site.prologue.strip():
        lines += ["# Site prologue.", site.prologue.rstrip()]
    run_unit = (
        f"{q(site.python)} -m loki_survey run-unit {q(str(run.root))} "
        f'"$uid" --attempt "$attempt" >> {logs}/"$uid.$attempt.log" 2>&1'
    )
    threads_env = (
        f"export OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads}"
    )
    lines += [threads_env, run_unit]
    return "\n".join(lines) + "\n"


def classify(
    record: JobRecord,
    *,
    began: bool,
    cancel_requested: bool,
) -> tuple[str, str]:
    """Class an ended attempt that left no summary, with the words to report."""
    code = record.exit_code
    words = record.text or (f"exit {code}" if code is not None else "no exit code")
    failure = REASON_CLASSES.get(record.reason or "")
    if cancel_requested and failure in {None, KILLED}:
        failure = CANCELLED
    elif failure is None:
        if code in HOST_EXIT_CODES:
            failure, words = HOST, f"{words} (signal {code - 128})"
        elif code not in {None, 0} and not began:
            failure, words = HOST, f"{words} before run-unit started"
        elif code is None:
            failure = LOST
        else:
            failure = CRASHED
            if code == 0:
                words = "exited 0 without writing its summary"
    return failure, words


class BatchExecutor:
    def __init__(
        self,
        run: RunDir,
        resolved: Resolved,
        db: StatusDB,
        scheduler: Scheduler,
    ) -> None:
        self.run = run
        self.resolved = resolved
        self.site = resolved.site
        self.batch = resolved.site.batch
        self.db = db
        self.scheduler = scheduler
        self.tag = run_tag(run)
        # The units this driver has submitted or been asked to watch.
        self.units: dict[str, Unit] = {}

    # -- polling -----------------------------------------------------------

    def reconcile(self) -> None:
        """Bring every live attempt up to date with the scheduler."""
        self._adopt_unsubmitted()
        live = [a for a in self.db.live() if a.job]
        if not live:
            return
        records = self.scheduler.poll(sorted({a.job for a in live if a.job}))
        for attempt in live:
            self._update(attempt, records.get((attempt.job, attempt.index)))

    def _adopt_unsubmitted(self) -> None:
        """Settle arrays a driver recorded but may not have submitted."""
        for name in self.db.unsubmitted_arrays():
            found = sorted(self.scheduler.lookup(name), key=int)
            if found:
                self._keep_first(found)
                self.db.set_array_job(name, found[0])
                print(f"found {name} as job {found[0]}")
            else:
                self.db.drop_array(name, "never reached the scheduler")

    def _keep_first(self, jobs: list[str]) -> None:
        for duplicate in jobs[1:]:
            print(f"cancel duplicate job {duplicate}")
            self.scheduler.cancel(duplicate, None, force=False)

    def _update(self, a: Attempt, record: JobRecord | None) -> None:
        started = self.run.unit_started(a.unit, a.attempt)
        if not a.began and started.is_file():
            a = _with(a, began=True)
            self.db.mark_began(a.unit, a.attempt, _marker_host(started))
        if self.run.unit_summary(a.unit).is_file():
            self.db.finish(
                a.unit,
                a.attempt,
                state="done",
                exit_code=record.exit_code if record else None,
                peak_rss_gb=record.max_mem_gb if record else None,
                host=record.host if record else None,
            )
            print(f"done  {a.unit}")
            return
        if record is None:
            record = self.scheduler.ended_record(a)
        settle_s = SETTLE_POLLS * self.site.poll_s
        if record is None:
            if time.time() - a.started < settle_s:
                return
            record = JobRecord(ENDED, text="the scheduler has no record of the job")
        if record.state == PENDING:
            return
        if record.state in {RUNNING, UNKNOWN, SUSPENDED}:
            self.db.mark_running(a.unit, a.attempt, record.host)
            if record.state == SUSPENDED:
                # A suspended job cannot beat; its stall clock starts again
                # when it resumes.
                self.db.restart_stall_clock(a.unit, a.attempt)
            elif self._stalled(a):
                self._stall(a, record)
            return
        if record.reason is None:
            # No scheduler reason: the job's own files (summary, started
            # marker) decide its class, so give them time to appear.
            if a.ended_seen is None:
                self.db.mark_ended_seen(a.unit, a.attempt)
                return
            if time.time() - a.ended_seen < settle_s:
                return
        self._ended(a, record)

    def _stalled(self, a: Attempt) -> bool:
        if self.site.stall_min <= 0 or a.running_since is None:
            return False
        alive = self.run.unit_alive(a.unit, a.attempt)
        beat = alive.stat().st_mtime if alive.is_file() else 0.0
        return time.time() - max(beat, a.running_since) > self.site.stall_min * 60

    def _stall(self, a: Attempt, record: JobRecord) -> None:
        if a.job is not None:
            self.scheduler.cancel(a.job, a.index, force=record.state == UNKNOWN)
        detail = (
            f"no heartbeat for {self.site.stall_min:g} min "
            f"(scheduler state {record.state})"
        )
        self._decide(a, record, STALLED, detail)

    def _ended(self, a: Attempt, record: JobRecord) -> None:
        failure, detail = classify(
            record,
            began=a.began,
            cancel_requested=a.cancel_requested,
        )
        if failure == CRASHED:
            tail = log_tail(self.run.unit_log(a.unit, a.attempt))
            detail = f"{detail}\n{tail}" if tail else detail
        if failure in {HOST, LOST}:
            tail = log_tail(self._sched_log(a))
            detail = f"{detail}\n{tail}" if tail else detail
        self._decide(a, record, failure, detail)

    def _sched_log(self, a: Attempt) -> Path:
        return self.run.sched_logs_dir / f"{a.array}.{a.job}.{a.index}.out"

    def _decide(self, a: Attempt, record: JobRecord, failure: str, detail: str) -> None:
        """Record a failed attempt and choose: resubmit, or report."""
        self._quarantine(a)
        site = self.site
        state, memory = "failed", None
        if failure == OOM:
            if self.db.count_failures(a.unit, (OOM,)) >= site.oom_retries:
                detail = f"{detail}; out of memory {site.oom_retries + 1} times"
            elif a.memory_gb >= site.max_memory_gb:
                detail = (
                    f"{detail}; already at the site's largest reservation "
                    f"({site.max_memory_gb:g} GB)"
                )
            else:
                state = "pending"
                memory = min(a.memory_gb * site.oom_factor, site.max_memory_gb)
        elif failure in REQUEUED:
            if failure == HOST and record.host:
                self._count_host_failure(record.host, detail)
            budget_left = (
                self.db.count_failures(a.unit, REQUEUED) < site.requeue_retries
            )
            stalled_before = (
                failure == STALLED and self.db.count_failures(a.unit, (STALLED,)) > 0
            )
            if budget_left and not stalled_before:
                state = "pending"
        elif failure == CANCELLED:
            state = "pending"
        elif failure == TIMEOUT:
            parts = self._split(a)
            if parts:
                state = SPLIT
                detail = f"{detail}; split into {', '.join(p.id for p in parts)}"
            else:
                limit = self.batch.time_min if self.batch else None
                detail = (
                    f"{detail}; split {split_depth(a.unit)} times already "
                    f"([retry] timeout_splits = {site.timeout_splits}); give its "
                    f"region more bands in survey.toml, or raise time_min in "
                    f"site.toml (now {limit or 'the queue default'})"
                )
        self.db.finish(
            a.unit,
            a.attempt,
            state=state,
            exit_code=record.exit_code,
            failure=failure,
            detail=detail,
            peak_rss_gb=record.max_mem_gb,
            memory_gb=memory,
            host=record.host,
        )
        where = f" on {record.host}" if record.host else ""
        if state == SPLIT:
            print(f"{failure:9s} {a.unit}{where}: {detail.splitlines()[0]}")
        elif state == "pending" and memory is not None:
            print(f"{failure:9s} {a.unit}{where}: resubmit with {memory:.3g} GB")
        elif state == "pending":
            print(f"{failure:9s} {a.unit}{where}: resubmit")
        else:
            print(f"FAIL {failure} {a.unit}{where}: {detail.splitlines()[0]}")

    def _split(self, a: Attempt) -> list[Unit] | None:
        """Split a unit that hit its time limit; return its parts, or None."""
        if split_depth(a.unit) >= self.site.timeout_splits:
            return None
        unit = self.units.get(a.unit)
        if unit is None and self.run.units_jsonl.is_file():
            unit = next((u for u in load_units(self.run) if u.id == a.unit), None)
        if unit is None:
            return None
        try:
            parts = split_unit(self.resolved, unit)
            for part in parts:
                if part.plan_key is not None:
                    build_ep_plan(self.run, self.resolved, part)
        except (ConfigError, ValueError) as exc:
            print(f"could not split {a.unit}: {exc}")
            return None
        record_split(self.run, unit, parts, TIMEOUT)
        # The parts start from the parent's reservation.
        self.db.sync(parts, a.memory_gb)
        return parts

    def _count_host_failure(self, host: str, detail: str) -> None:
        """Exclude a host once enough failures point at it and not elsewhere.

        Failures on several hosts while no unit starts anywhere point at the
        setup (a broken prologue, a wrong Python), not at a host, so none is
        excluded. Failures on one host are its own, even when every job lands
        there (a host that fails fast drains the queue into itself).
        """
        failures = self.db.host_failed(host, detail.splitlines()[0])
        if host in self.db.excluded_hosts() or failures < self.site.host_failures:
            return
        ran_elsewhere = self.db.hosts_that_ran_units() - {host}
        failed_elsewhere = self.db.hosts_with_failures() - {host}
        if ran_elsewhere or not failed_elsewhere:
            self.db.exclude_host(host)
            print(f"exclude host {host} after {failures} failures")

    def _quarantine(self, a: Attempt) -> None:
        tmp = self.run.unit_tmp(a.unit, a.attempt)
        if tmp.exists():
            target = self.run.quarantine_dir / tmp.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(tmp), str(target))

    # -- submitting ---------------------------------------------------------

    def submit(self, units: list[Unit]) -> int:
        """Submit every pending unit as arrays grouped by memory; return how many."""
        self.units.update({u.id: u for u in units})
        groups: dict[float, list[Unit]] = {}
        for unit in units:
            state = self.db.get(unit.id)
            if state.state == "pending":
                # A reservation raised in site.toml applies to every unit; one
                # raised after running out of memory is kept.
                memory = max(state.memory_gb or 0.0, self.site.unit_memory_gb)
                groups.setdefault(round(memory, 3), []).append(unit)
        submitted = 0
        excluded = tuple(self.db.excluded_hosts())
        max_array = self.batch.max_array if self.batch else len(units)
        for memory, group in sorted(groups.items()):
            for start in range(0, len(group), max_array):
                chunk = group[start : start + max_array]
                self._submit_array(chunk, memory, excluded)
                submitted += len(chunk)
        return submitted

    def _submit_array(
        self,
        units: list[Unit],
        memory_gb: float,
        excluded: tuple[str, ...],
    ) -> None:
        name = f"{self.tag}.{self.db.next_array_seq():04d}"
        self.db.add_array(name, len(units))
        lines = []
        for index, unit in enumerate(units, start=1):
            log = self.run.unit_log(unit.id, self.db.get(unit.id).attempts + 1)
            attempt = self.db.queue(
                unit.id,
                memory_gb,
                str(log),
                array=name,
                index=index,
            )
            lines.append(f"{unit.id} {attempt}\n")
        (self.run.jobs_dir / f"{name}.txt").write_text("".join(lines))
        script = self.run.jobs_dir / f"{name}.sh"
        script.write_text(job_script(self.run, self.resolved, name))
        script.chmod(0o755)
        request = ArrayRequest(
            name=name,
            script=script,
            n=len(units),
            memory_gb=memory_gb,
            threads=self.site.unit_threads,
            gpu=self.batch.gpu if self.batch else None,
            exclude_hosts=excluded,
            max_running=self.site.max_running,
        )
        job = self._submit_with_retry(request)
        if job is None:
            self.db.drop_array(name, "submission failed")
            msg = (
                f"could not submit {name} within {self.site.submit_timeout_s:g} s; "
                f"see {self.run.jobs_dir / name}.submit.log"
            )
            raise SubmitError(msg)
        self.db.set_array_job(name, job)
        print(
            f"submitted {name}: {len(units)} units at {memory_gb:.1f} GB as job {job}",
        )

    def _submit_with_retry(self, request: ArrayRequest) -> str | None:
        """Submit once per try; look the array up before trying again.

        A scheduler that is slow to answer (LSF's "mbatchd busy") may accept
        a job and still report a failure, so a blind retry would run every
        unit twice.
        """
        deadline = time.time() + self.site.submit_timeout_s
        log = self.run.jobs_dir / f"{request.name}.submit.log"
        while True:
            remaining = max(deadline - time.time(), 1.0)
            job, text = self.scheduler.submit_once(request, remaining)
            with log.open("a") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
            if job is not None:
                return job
            found = sorted(self.scheduler.lookup(request.name), key=int)
            if found:
                self._keep_first(found)
                return found[0]
            if time.time() + self.site.poll_s > deadline:
                return None
            time.sleep(self.site.poll_s)

    # -- driving ------------------------------------------------------------

    def watch(self, units: list[Unit]) -> None:
        """Poll and resubmit until no unit is pending, queued or running."""
        last = None
        while True:
            try:
                self.reconcile()
            except SchedulerError as exc:
                # A busy or restarting scheduler: try again next poll.
                print(time.strftime("%H:%M:%S"), f"poll skipped: {exc}", flush=True)
                time.sleep(self.site.poll_s)
                continue
            # A unit split since the last poll is run as its parts.
            units = expand(units, load_splits(self.run))
            self.submit(units)
            counts = self.db.counts()
            line = "  ".join(
                f"{s} {counts.get(s, 0)}"
                for s in ("done", "running", "queued", "pending", "failed")
            )
            if counts.get(SPLIT):
                line += f"  split {counts[SPLIT]}"
            if line != last:
                print(time.strftime("%H:%M:%S"), line, flush=True)
                last = line
            if not any(counts.get(s, 0) for s in ("pending", "queued", "running")):
                return
            time.sleep(self.site.poll_s)

    def cancel(self) -> int:
        """Cancel every live attempt; the units go back to pending."""
        live = [a for a in self.db.live() if a.job]
        for a in live:
            self.db.request_cancel(a.unit, a.attempt)
        for job in sorted({a.job for a in live if a.job}, key=int):
            self.scheduler.cancel(job, None, force=False)
        return len(live)


def _with(a: Attempt, **changes: object) -> Attempt:
    return Attempt(**{**a.__dict__, **changes})


def _marker_host(path: Path) -> str | None:
    try:
        return read_json(path).get("host")
    except (OSError, json.JSONDecodeError):
        return None
