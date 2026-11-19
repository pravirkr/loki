"""LSF: submit arrays with ``bsub``, read them with ``bjobs``, stop them with ``bkill``.

``bjobs`` forgets a finished job after the cluster's CLEAN_PERIOD (an hour on
many sites), so an element it no longer lists is read from the footer LSF
appends to the element's output file. An element with neither is lost.

``bstop`` is not used: it suspends a running job, which keeps its memory, and
does not hold it.
"""

from __future__ import annotations

import json
import re
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
    UNKNOWN,
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

# Bytes in each LSF_UNIT_FOR_LIMITS unit; LSF counts memory in binary units.
MEM_UNIT_BYTES = {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}
SIZE_BYTES = {
    "Kbytes": 1 << 10,
    "Mbytes": 1 << 20,
    "Gbytes": 1 << 30,
    "Tbytes": 1 << 40,
}
STATES = {
    "PEND": PENDING,
    "PSUSP": PENDING,
    "WAIT": PENDING,
    "RUN": RUNNING,
    "PROV": RUNNING,
    "SSUSP": SUSPENDED,
    "USUSP": SUSPENDED,
    "UNKWN": UNKNOWN,
    "ZOMBI": UNKNOWN,
    "DONE": ENDED,
    "EXIT": ENDED,
}
# LSF's TERM_* exit reasons, by what they mean for the unit.
REASONS = {
    "TERM_MEMLIMIT": MEMLIMIT,
    "TERM_SWAP": MEMLIMIT,
    "TERM_RUNLIMIT": RUNLIMIT,
    "TERM_CPULIMIT": RUNLIMIT,
    "TERM_DEADLINE": RUNLIMIT,
    "TERM_PREEMPT": PREEMPT,
    "TERM_WINDOW": PREEMPT,
    "TERM_REQUEUE_ADMIN": PREEMPT,
    "TERM_REQUEUE_OWNER": PREEMPT,
    "TERM_OWNER": KILL,
    "TERM_ADMIN": KILL,
    "TERM_FORCE_OWNER": KILL,
    "TERM_FORCE_ADMIN": KILL,
    "TERM_EXTERNAL_SIGNAL": KILL,
    "TERM_LOAD": NODE,
    "TERM_ZOMBIE": NODE,
    "TERM_UNKNOWN": NODE,
    "TERM_REMOVE_HUNG_JOB": NODE,
    "TERM_CWD_NOTEXIST": NODE,
    "TERM_ORPHAN_SYSTEM": NODE,
}
FIELDS = "jobid jobindex stat exit_code exit_reason exec_host max_mem"
SUBMITTED = re.compile(r"Job <(\d+)> is submitted")
TERM = re.compile(r"\bTERM_[A-Z_]+")
SIZE = re.compile(r"([\d.]+)\s*([KMGT]bytes)")
FOOTER_START = "Sender: LSF System"
# How many job ids one bjobs call asks about.
BJOBS_CHUNK = 200


def parse_host(exec_host: str) -> str | None:
    """First host of an EXEC_HOST such as '4*hgn19' or '1*cn111:2*cn817'."""
    first = exec_host.split(":", maxsplit=1)[0].strip()
    host = first.split("*")[-1]
    return host or None


def parse_size_gb(text: str) -> float | None:
    """'159.3 Gbytes' or 'Max Memory : 12786 MB' in GB (10^9 bytes)."""
    match = SIZE.search(text)
    if match:
        return float(match.group(1)) * SIZE_BYTES[match.group(2)] / 1e9
    match = re.search(r"([\d.]+)\s*MB\b", text)
    if match:
        return float(match.group(1)) * MEM_UNIT_BYTES["MB"] / 1e9
    return None


def parse_bjobs(text: str) -> dict[tuple[str, int], JobRecord]:
    """Read ``bjobs -json -o FIELDS`` into records keyed by (job id, index)."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"bjobs gave no JSON: {text[:200]!r}"
        raise SchedulerError(msg) from exc
    records = {}
    for row in doc.get("RECORDS", []):
        if "ERROR" in row or "STAT" not in row:
            continue
        reason_text = row.get("EXIT_REASON", "")
        term = TERM.search(reason_text)
        code = row.get("EXIT_CODE", "")
        state = STATES.get(row["STAT"], UNKNOWN)
        exit_code = int(code) if code.strip().isdigit() else None
        if row["STAT"] == "DONE" and exit_code is None:
            exit_code = 0
        records[(row["JOBID"], int(row.get("JOBINDEX") or 0))] = JobRecord(
            state=state,
            exit_code=exit_code,
            reason=REASONS.get(term.group(0)) if term else None,
            host=parse_host(row.get("EXEC_HOST", "")),
            max_mem_gb=parse_size_gb(row.get("MAX_MEM", "")),
            text=reason_text or f"{row['STAT']} exit {code or 0}",
        )
    return records


def parse_footer(text: str) -> JobRecord | None:
    """Read the footer LSF appends to a job's output, if it has one.

    A requeued element appends one footer per run; the last one counts.
    """
    last = text.rfind(FOOTER_START)
    if last > 0:
        text = text[last:]
    if "Successfully completed." in text:
        exit_code = 0
    else:
        match = re.search(r"Exited with exit code (\d+)\.", text)
        if match is None:
            return None
        exit_code = int(match.group(1))
    term = TERM.search(text)
    host = re.search(r"executed on host\(s\) <([^>]+)>", text)
    memory = re.search(r"Max Memory :\s*(.+)", text)
    reason_line = re.search(r"TERM_[A-Z_]+:[^\n]*", text)
    return JobRecord(
        state=ENDED,
        exit_code=exit_code,
        reason=REASONS.get(term.group(0)) if term else None,
        host=parse_host(host.group(1)) if host else None,
        max_mem_gb=parse_size_gb(memory.group(1)) if memory else None,
        text=reason_line.group(0) if reason_line else f"exit {exit_code}",
    )


class LsfScheduler:
    def __init__(self, run: RunDir, site: Site) -> None:
        if site.batch is None:
            msg = "LSF needs an [lsf] table in site.toml"
            raise ValueError(msg)
        self.run = run
        self.batch = site.batch

    def bsub_args(self, request: ArrayRequest) -> list[str]:
        batch = self.batch
        unit_bytes = MEM_UNIT_BYTES[batch.mem_unit]
        total = gb_to_units(request.memory_gb, unit_bytes)
        per_task = -(-total // request.threads) if batch.reserve_per_task else total
        select = [f"({batch.select})"] if batch.select else []
        select += [f"hname!={host}" for host in request.exclude_hosts]
        resources = f"span[hosts=1] rusage[mem={per_task}]"
        if select:
            resources += f" select[{' && '.join(select)}]"
        cap = f"%{request.max_running}" if request.max_running else ""
        out = self.run.sched_logs_dir / f"{request.name}.%J.%I.out"
        argv = [
            "bsub",
            "-J",
            f"{request.name}[1-{request.n}]{cap}",
            "-n",
            str(request.threads),
            "-R",
            resources,
            "-o",
            str(out),
        ]
        if batch.mem_limit:
            argv += ["-M", str(total)]
        if batch.queue:
            argv += ["-q", batch.queue]
        if batch.time_min:
            argv += ["-W", str(batch.time_min)]
        if batch.hosts:
            argv += ["-m", " ".join(batch.hosts)]
        if request.gpu:
            argv += ["-gpu", request.gpu]
        return [*argv, *batch.extra, str(request.script)]

    def submit_once(
        self,
        request: ArrayRequest,
        timeout_s: float,
    ) -> tuple[str | None, str]:
        argv = self.bsub_args(request)
        done = run_command(argv, timeout_s)
        if done is None:
            return None, f"{argv[0]}: no answer within {timeout_s:.0f} s"
        text = (done.stdout + done.stderr).strip()
        match = SUBMITTED.search(done.stdout)
        return (match.group(1) if match else None), text

    def lookup(self, name: str) -> list[str]:
        done = run_command(["bjobs", "-a", "-json", "-o", "jobid", "-J", name])
        if done is None:
            return []
        try:
            rows = json.loads(done.stdout).get("RECORDS", [])
        except json.JSONDecodeError:
            return []
        return sorted({r["JOBID"] for r in rows if "JOBID" in r and "ERROR" not in r})

    def poll(self, jobs: list[str]) -> dict[tuple[str, int], JobRecord]:
        records: dict[tuple[str, int], JobRecord] = {}
        for start in range(0, len(jobs), BJOBS_CHUNK):
            chunk = jobs[start : start + BJOBS_CHUNK]
            done = run_command(["bjobs", "-a", "-json", "-o", FIELDS, *chunk])
            if done is None:
                msg = "bjobs is missing or did not answer"
                raise SchedulerError(msg)
            records.update(parse_bjobs(done.stdout))
        return records

    def ended_record(self, attempt: Attempt) -> JobRecord | None:
        out = (
            self.run.sched_logs_dir
            / f"{attempt.array}.{attempt.job}.{attempt.index}.out"
        )
        try:
            return parse_footer(out.read_text(errors="replace"))
        except OSError:
            return None

    def cancel(self, job: str, index: int | None, *, force: bool = False) -> None:
        target = job if index is None else f"{job}[{index}]"
        run_command(["bkill", *(["-r"] if force else []), target])
