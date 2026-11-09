# loki-survey

Run a loki pulsar survey end to end: from dedispersed time series to a
candidate list, on a laptop, a single server or an LSF or Slurm cluster.

```bash
pip install ./survey            # needs loki installed
loki-survey new myrun           # writes myrun/survey.toml and myrun/site.toml
# edit both files, put the series where survey.toml points
loki-survey run myrun           # check, prep, plan and submit
loki-survey status myrun
```

## Stages

| Command | Reads | Writes |
| --- | --- | --- |
| `check` | `survey.toml`, `site.toml` | `resolved.json`: every value used, the science hash |
| `prep` | the series (`.dat`/`.inf` or `.tim`) | `prep/<series>/`: normalised, masked, downsampled arrays and a manifest |
| `plan` | the manifests | `units.jsonl`, and the EP plans in `plans/` |
| `submit` | `units.jsonl` | `out/<unit>.h5` and `out/<unit>.json` per unit |
| `watch` | the cluster's job records | resubmissions, `status.sqlite` |
| `status` | `out/`, `status.sqlite` | nothing |

Each stage does only what is missing, so rerunning `loki-survey run` after an
interruption picks up where it stopped.

- `survey.toml` holds the science: the series, the frequency regions and the
  search boxes. `check` refuses to change it once units have finished; start a
  new run directory instead.
- `site.toml` holds the machine: memory, threads, and the executor. It can
  change between runs without invalidating results.

## Units

A unit is one series × frequency band × search box. It runs in its own
process, writes in `tmp/` and renames its output into `out/` when it
finishes. The output keeps loki's common candidate tables (`/candidates`,
`/clusters`, `/search`, `/observation`) and adds `/survey`, the provenance:
the unit, the science hash, the input and array hashes and the code versions.
loki's full result file is kept in `out/raw/` for a fixed sample of units
(`[output] raw_fraction`).

The local executor packs units within the site's memory and threads. A unit
above its memory reservation is stopped and run again with more memory, up
to `[retry] oom_retries` times. A unit that crashes is reported with the end
of its log and not retried.

## Clusters

With `kind = "lsf"` or `kind = "slurm"` in `site.toml`, `submit` sends the
units as job arrays, one per memory size, and returns. `watch` then polls the
scheduler and resubmits until every unit has ended (`submit --wait` and
`run --wait` do both). Stop `watch` at any time and start it again: the jobs
keep running, and only one `submit` or `watch` can drive a run at once.

Every failure is classed from the scheduler's own record:

| Class | From | What happens |
| --- | --- | --- |
| `oom` | LSF TERM_MEMLIMIT, Slurm OUT_OF_MEMORY | resubmitted with `oom_factor` more memory, up to `oom_retries` times |
| `preempted`, `killed`, `lost` | the scheduler's state; no record and no footer | resubmitted, up to `requeue_retries` times |
| `host` | the job died before `run-unit` started, or SIGILL/SIGBUS | resubmitted; the host is excluded after `host_failures` failures, unless failures on several hosts with no unit started anywhere point at the setup instead |
| `stalled` | no heartbeat for `stall_min` minutes | the job is killed and resubmitted once |
| `timeout` | LSF TERM_RUNLIMIT, Slurm TIMEOUT | reported: give the region more bands, or raise `time_min` |
| `crashed` | any other non-zero exit | reported with the end of its log; not retried |

A plain `submit` also retries units that failed, for after their cause is
fixed; `watch` leaves them failed. `cancel` stops a run's jobs and returns
their units to pending.

LSF forgets a finished job after its CLEAN_PERIOD, so an element `bjobs` no
longer lists is read from the footer LSF appends to its output in
`logs/sched/`. Slurm needs its accounting (`sacct`) for the failure classes;
without it, a failed element can only be called lost.

## Tests

```bash
cd survey && pytest tests
```

`test_end_to_end.py` simulates a pulsar, runs every stage with FFA and EP
boxes and checks that both find it, on this machine and through a fake LSF.
`test_batch_executor.py` drives each failure class through
`fake_scheduler.py`, a stand-in for LSF and Slurm. Units run in subprocesses
with `LOKI_SURVEY_TEST_PYTHON` (default: the Python running the tests).
