"""A fake LSF and Slurm for the executor tests.

Run as ``bsub``, ``bjobs``, ``bkill``, ``sbatch``, ``squeue``, ``scontrol``,
``sacct`` or ``scancel`` (the name is the first argument). Elements run synchronously at
submission, one after another, on hosts taken in turn from the control file.
The control file (``$FAKE_SCHED_DIR/control.json``) makes chosen elements
fail; an element is named by ``"<unit>:<attempt>"``, read from the array's
unit list next to its script:

- ``hosts``: host names, taken in turn; ``prefer``: a host that takes every
  element while it is not excluded; ``bad_hosts``: hosts where a job dies
  before its script runs (exit 255);
- ``need_gb``: {unit: GB}: an element given less memory is killed for it;
- ``kill``, ``preempt``, ``timeout``: elements ended that way, unrun;
- ``hang``: elements left running forever without running their script;
- ``vanish``: elements that leave no record and no footer;
- ``forget``: elements bjobs no longer lists, as after LSF's CLEAN_PERIOD;
- ``busy``: how many submissions to accept yet report as failed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_SCHED_DIR"])
FOOTER = """
------------------------------------------------------------
Sender: LSF System <lsfadmin@{host}>
Subject: Job {job}[{index}]: <{name}[{index}]> in cluster <fake> {word}

Job <{name}[{index}]> was submitted from host <login> by user <me>.
Job was executed on host(s) <{host}>, in queue <q>, as user <me>.
{reason}
{result}

Resource usage summary:

    Max Memory :                                 {mem_mb} MB
"""
LSF_REASONS = {
    "oom": "TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.",
    "kill": "TERM_OWNER: job killed by owner.",
    "preempt": "TERM_PREEMPT: job killed after preemption.",
    "timeout": "TERM_RUNLIMIT: job killed after reaching LSF run time limit.",
}
SLURM_STATES = {
    "oom": "OUT_OF_MEMORY",
    "kill": "CANCELLED by 1000",
    "preempt": "PREEMPTED",
    "timeout": "TIMEOUT",
}
EXIT_CODES = {"oom": 137, "kill": 130, "preempt": 130, "timeout": 140, "bad": 255}


def load(name: str, default: object) -> object:
    path = STATE / name
    return json.loads(path.read_text()) if path.is_file() else default


def save(name: str, value: object) -> None:
    (STATE / name).write_text(json.dumps(value, indent=1))


def element_key(script: str, index: int) -> str:
    units = Path(script).with_suffix(".txt").read_text().splitlines()
    uid, attempt = units[index - 1].split()
    return f"{uid}:{attempt}"


def run_element(job: dict, index: int, control: dict, host: str) -> dict:
    """Run one element (or fail it as the control file says); return its record."""
    key = element_key(job["script"], index)
    uid = key.split(":")[0]
    out = job["out"].replace("%J", job["id"]).replace("%A", job["id"])
    out = out.replace("%I", str(index)).replace("%a", str(index))
    record = {"host": host, "how": "ok", "code": 0, "state": "ended", "mem_mb": 10}
    # A forgotten element is one bjobs no longer lists (past CLEAN_PERIOD).
    record["forgotten"] = key in control.get("forget", [])
    for how in ("kill", "preempt", "timeout", "hang", "vanish"):
        if key in control.get(how, []):
            record["how"] = how
    if host in control.get("bad_hosts", []):
        record["how"] = "bad"
    elif control.get("need_gb", {}).get(uid, 0) > job["mem_gb"]:
        record["how"] = "oom"
        record["mem_mb"] = int(job["mem_gb"] * 1000)
    how = record["how"]
    if how == "hang":
        record["state"] = "running"
        return record
    if how == "ok":
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("LSB_", "SLURM_"))
        }
        env["FAKE_HOST"] = host
        if job["kind"] == "lsf":
            env.update(LSB_JOBID=job["id"], LSB_JOBINDEX=str(index))
        else:
            env.update(SLURM_ARRAY_JOB_ID=job["id"], SLURM_ARRAY_TASK_ID=str(index))
        with open(out, "a") as handle:  # noqa: PTH123
            done = subprocess.run(
                ["/bin/bash", job["script"]],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        record["code"] = done.returncode
    else:
        record["code"] = EXIT_CODES.get(how)
    if how != "vanish" and job["kind"] == "lsf":
        result = (
            "Successfully completed."
            if record["code"] == 0
            else f"Exited with exit code {record['code']}."
        )
        with open(out, "a") as handle:  # noqa: PTH123
            handle.write(
                FOOTER.format(
                    host=host,
                    job=job["id"],
                    index=index,
                    name=job["name"],
                    word="Done" if record["code"] == 0 else "Exited",
                    reason=LSF_REASONS.get(how, ""),
                    result=result,
                    mem_mb=record["mem_mb"],
                ),
            )
    return record


def submit(
    kind: str,
    name: str,
    n: int,
    mem_gb: float,
    exclude: list[str],
    out: str,
    script: str,
) -> int:
    control = load("control.json", {})
    jobs = load("jobs.json", {})
    counter = load("counter.json", {"job": 100, "host": 0})
    counter["job"] += 1
    job = {
        "id": str(counter["job"]),
        "kind": kind,
        "name": name,
        "mem_gb": mem_gb,
        "out": out,
        "script": script,
        "elements": {},
    }
    hosts = [h for h in control.get("hosts", ["h1"]) if h not in exclude]
    for index in range(1, n + 1):
        host = hosts[counter["host"] % len(hosts)]
        counter["host"] += 1
        if control.get("prefer") in hosts:
            host = control["prefer"]
        job["elements"][str(index)] = run_element(job, index, control, host)
    jobs[job["id"]] = job
    save("jobs.json", jobs)
    save("counter.json", counter)
    if control.get("busy", 0) > 0:
        control["busy"] -= 1
        save("control.json", control)
        print("mbatchd: batch system not responding", file=sys.stderr)
        return 1
    if kind == "lsf":
        print(f"Job <{job['id']}> is submitted to queue <q>.")
    else:
        print(job["id"])
    return 0


def bsub(args: list[str]) -> int:
    opts, i = {}, 0
    while i < len(args) - 1:
        opts[args[i]] = args[i + 1]
        i += 2
    script = args[-1]
    match = re.fullmatch(r"(.+)\[1-(\d+)\](?:%\d+)?", opts["-J"])
    exclude = re.findall(r"hname!=(\S+?)(?:\s|\]|$)", opts["-R"])
    if "-M" in opts:
        mem_mb = int(opts["-M"])
    else:  # the limit follows the reservation, per slot
        per_slot = re.search(r"rusage\[mem=(\d+)\]", opts["-R"]).group(1)
        mem_mb = int(per_slot) * int(opts["-n"])
    mem_gb = mem_mb * (1 << 20) / 1e9
    return submit(
        "lsf",
        match.group(1),
        int(match.group(2)),
        mem_gb,
        exclude,
        opts["-o"],
        script,
    )


def sbatch(args: list[str]) -> int:
    opts = dict(a[2:].split("=", 1) for a in args if a.startswith("--") and "=" in a)
    match = re.fullmatch(r"1-(\d+)(?:%\d+)?", opts["array"])
    mem_gb = int(opts["mem"].rstrip("M")) * (1 << 20) / 1e9
    exclude = opts["exclude"].split(",") if "exclude" in opts else []
    return submit(
        "slurm",
        opts["job-name"],
        int(match.group(1)),
        mem_gb,
        exclude,
        opts["output"],
        args[-1],
    )


def lsf_row(job: dict, index: str, element: dict) -> dict:
    how, code = element["how"], element["code"]
    stat = "DONE" if code == 0 else "EXIT"
    if element["state"] == "running":
        stat = "RUN"
    return {
        "JOBID": job["id"],
        "JOBINDEX": index,
        "STAT": stat,
        "EXIT_CODE": "" if stat != "EXIT" else str(code),
        "EXIT_REASON": LSF_REASONS.get(how, "").rstrip("."),
        "EXEC_HOST": f"1*{element['host']}",
        "MAX_MEM": f"{element['mem_mb']} Mbytes",
    }


def bjobs(args: list[str]) -> int:
    jobs = load("jobs.json", {})
    if "-J" in args:
        name = args[args.index("-J") + 1]
        rows = [{"JOBID": j["id"]} for j in jobs.values() if j["name"] == name]
    else:
        ids = [a for a in args if a.isdigit()]
        rows = []
        for jid in ids:
            if jid not in jobs:
                rows.append({"JOBID": jid, "ERROR": f"Job <{jid}> is not found"})
                continue
            for index, element in jobs[jid]["elements"].items():
                if element["how"] != "vanish" and not element["forgotten"]:
                    rows.append(lsf_row(jobs[jid], index, element))
    print(json.dumps({"COMMAND": "bjobs", "JOBS": len(rows), "RECORDS": rows}))
    return 0


def kill(target: str, how: str) -> None:
    jobs = load("jobs.json", {})
    match = re.fullmatch(r"(\d+)(?:[\[_](\d+)\]?)?", target)
    job = jobs.get(match.group(1))
    if job is None:
        return
    for index, element in job["elements"].items():
        if match.group(2) in {None, index} and element["state"] == "running":
            element.update(state="ended", how=how, code=EXIT_CODES[how])
    save("jobs.json", jobs)


def bkill(args: list[str]) -> int:
    kill(args[-1], "kill")
    return 0


def scancel(args: list[str]) -> int:
    kill(args[-1], "kill")
    return 0


def squeue(args: list[str]) -> int:
    jobs = load("jobs.json", {})
    names = [a.split("=", 1)[1] for a in args if a.startswith("--name=")]
    if names:
        for job in jobs.values():
            if job["name"] == names[0]:
                print(job["id"])
        return 0
    ids = args[args.index("-j") + 1].split(",")
    for jid in ids:
        for index, element in jobs.get(jid, {}).get("elements", {}).items():
            if element["state"] == "running":
                print(f"{jid}_{index}|RUNNING|{element['host']}")
    return 0


def sacct(args: list[str]) -> int:
    jobs = load("jobs.json", {})
    names = [a.split("=", 1)[1] for a in args if a.startswith("--name=")]
    if names:
        for job in jobs.values():
            if job["name"] == names[0]:
                print(f"{job['id']}_1")
        return 0
    ids = args[args.index("-j") + 1].split(",")
    for jid in ids:
        for index, element in jobs.get(jid, {}).get("elements", {}).items():
            how, code = element["how"], element["code"]
            if element["state"] == "running" or how == "vanish":
                continue
            state = SLURM_STATES.get(how, "COMPLETED" if code == 0 else "FAILED")
            host = element["host"]
            if how == "oom":
                # Slurm marks the batch step, not always the job.
                print(f"{jid}_{index}|FAILED|0:9|{host}|")
                print(
                    f"{jid}_{index}.batch|OUT_OF_MEMORY|0:9|{host}|{element['mem_mb']}M",
                )
            else:
                print(f"{jid}_{index}|{state}|{code}:0|{host}|")
                print(
                    f"{jid}_{index}.batch|{state}|{code}:0|{host}|{element['mem_mb']}M",
                )
    return 0


def scontrol(args: list[str]) -> int:  # noqa: ARG001
    """Know no ended jobs, as after MinJobAge: records come from sacct."""
    return 0


COMMANDS = {
    "bsub": bsub,
    "bjobs": bjobs,
    "bkill": bkill,
    "sbatch": sbatch,
    "squeue": squeue,
    "sacct": sacct,
    "scancel": scancel,
    "scontrol": scontrol,
}

if __name__ == "__main__":
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2:]))
