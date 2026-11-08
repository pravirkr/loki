"""The local executor: run units as processes on this machine.

Units are packed by memory and threads within the site's limits. Each unit's
resident memory is watched; a unit that passes its reservation is killed and,
up to the site's retry count, run again with ``oom_factor`` more memory. A
crash is reported with the end of its log and not retried.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from loki_survey.status import CRASHED, INTERRUPTED, OOM, TOO_LARGE, StatusDB
from loki_survey.unit import unit_env

if TYPE_CHECKING:
    from loki_survey.config import Resolved
    from loki_survey.plan import Unit
    from loki_survey.rundir import RunDir

LOG_TAIL_LINES = 20


@dataclass
class Running:
    unit: Unit
    attempt: int
    memory_gb: float
    proc: subprocess.Popen[bytes]
    log: IO[bytes]
    peak_rss_gb: float = 0.0
    killed_for_memory: bool = False


def rss_gb(pid: int) -> float:
    """Resident memory of ``pid`` in GB, or 0 once it has exited."""
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / 1e9
        except OSError:
            return 0.0
        return 0.0
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return int(out) * 1024 / 1e9 if out else 0.0
    return 0.0


def log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class LocalExecutor:
    def __init__(self, run: RunDir, resolved: Resolved, db: StatusDB) -> None:
        self.run = run
        self.resolved = resolved
        self.site = resolved.site
        self.db = db

    def _launch(self, unit: Unit, memory_gb: float) -> Running:
        attempt = self.db.get(unit.id).attempts + 1
        log_path = self.run.unit_log(unit.id, attempt)
        attempt = self.db.start(unit.id, memory_gb, str(log_path))
        log = log_path.open("wb")
        cmd = [
            self.site.python,
            "-m",
            "loki_survey",
            "run-unit",
            str(self.run.root),
            unit.id,
            "--attempt",
            str(attempt),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=unit_env(self.site.unit_threads),
            start_new_session=True,
        )
        print(f"start {unit.id} (attempt {attempt}, {memory_gb:.1f} GB)")
        return Running(unit, attempt, memory_gb, proc, log)

    def _quarantine(self, unit_id: str, attempt: int) -> None:
        tmp = self.run.unit_tmp(unit_id, attempt)
        if tmp.exists():
            self.run.quarantine_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(self.run.quarantine_dir / tmp.name))

    def _finish(self, job: Running, pending: deque[tuple[Unit, float]]) -> None:
        job.log.close()
        code = job.proc.returncode
        uid = job.unit.id
        if code == 0 and self.run.unit_summary(uid).is_file():
            self.db.finish(
                uid,
                job.attempt,
                state="done",
                exit_code=code,
                peak_rss_gb=job.peak_rss_gb,
            )
            print(f"done  {uid}")
            return
        self._quarantine(uid, job.attempt)
        log = self.run.unit_log(uid, job.attempt)
        if job.killed_for_memory or code == -signal.SIGKILL:
            if job.attempt <= self.site.oom_retries:
                memory = job.memory_gb * self.site.oom_factor
                self.db.finish(
                    uid,
                    job.attempt,
                    state="pending",
                    exit_code=code,
                    failure=OOM,
                    detail=f"peak {job.peak_rss_gb:.2f} GB of {job.memory_gb:.2f} GB",
                    peak_rss_gb=job.peak_rss_gb,
                    memory_gb=memory,
                )
                pending.append((job.unit, memory))
                print(f"oom   {uid}: retry with {memory:.1f} GB")
                return
            self.db.finish(
                uid,
                job.attempt,
                state="failed",
                exit_code=code,
                failure=OOM,
                detail=f"peak {job.peak_rss_gb:.2f} GB of {job.memory_gb:.2f} GB",
                peak_rss_gb=job.peak_rss_gb,
            )
            print(f"FAIL  {uid}: out of memory")
            return
        self.db.finish(
            uid,
            job.attempt,
            state="failed",
            exit_code=code,
            failure=CRASHED,
            detail=log_tail(log),
            peak_rss_gb=job.peak_rss_gb,
        )
        print(f"FAIL  {uid}: exit {code}; see {log}")

    def submit(self, units: list[Unit]) -> None:
        """Run every unit that is not done; return when none is left running."""
        site = self.site
        pending: deque[tuple[Unit, float]] = deque()
        for unit in units:
            state = self.db.get(unit.id)
            if state.state in {"pending", "failed"}:
                memory = state.memory_gb or site.unit_memory_gb
                if state.failure != OOM:
                    memory = max(memory, site.unit_memory_gb)
                pending.append((unit, memory))
        running: list[Running] = []
        try:
            while pending or running:
                used_mem = sum(j.memory_gb for j in running)
                used_threads = len(running) * site.unit_threads
                deferred: deque[tuple[Unit, float]] = deque()
                while pending:
                    unit, memory = pending.popleft()
                    if memory > site.max_memory_gb:
                        attempt = self.db.start(unit.id, memory, "")
                        self.db.finish(
                            unit.id,
                            attempt,
                            state="failed",
                            exit_code=None,
                            failure=TOO_LARGE,
                            detail=(
                                f"needs {memory:.1f} GB; the site allows "
                                f"{site.max_memory_gb:.1f} GB"
                            ),
                        )
                        print(f"FAIL  {unit.id}: needs more memory than the site has")
                        continue
                    if (
                        used_mem + memory <= site.max_memory_gb
                        and used_threads + site.unit_threads <= site.max_threads
                    ):
                        running.append(self._launch(unit, memory))
                        used_mem += memory
                        used_threads += site.unit_threads
                    else:
                        deferred.append((unit, memory))
                pending = deferred
                time.sleep(site.poll_s)
                still = []
                for job in running:
                    rss = rss_gb(job.proc.pid)
                    job.peak_rss_gb = max(job.peak_rss_gb, rss)
                    if rss > job.memory_gb and job.proc.poll() is None:
                        job.killed_for_memory = True
                        os.killpg(job.proc.pid, signal.SIGKILL)
                    if job.proc.poll() is None:
                        still.append(job)
                    else:
                        self._finish(job, pending)
                running = still
        except KeyboardInterrupt:
            for job in running:
                if job.proc.poll() is None:
                    os.killpg(job.proc.pid, signal.SIGTERM)
                job.proc.wait()
                job.log.close()
                self._quarantine(job.unit.id, job.attempt)
                self.db.finish(
                    job.unit.id,
                    job.attempt,
                    state="pending",
                    exit_code=job.proc.returncode,
                    failure=INTERRUPTED,
                )
            raise
