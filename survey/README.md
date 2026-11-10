# loki-survey

Run a loki pulsar survey end to end: from dedispersed time series to a
candidate list, on a laptop, a single server or an LSF or Slurm cluster.

```bash
pip install ./survey            # needs loki installed
loki-survey new myrun           # writes myrun/survey.toml and myrun/site.toml
# edit both files, put the series where survey.toml points
loki-survey run myrun           # check, prep, plan, submit; then collect,
                                # evaluate and report once every unit is done
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
| `collect` | `out/` | `collect/candidates.h5`, `collect/ledger.jsonl`, `collect/summary.json` |
| `evaluate` | `collect/`, the catalogue, the lowest-DM series | `evaluate/matches.jsonl`, `evaluate/summary.json` |
| `fold` | `collect/`, the units' leaves, `prep/` | `folds/folds.h5` and `folds/summary.json` |
| `report` | everything above | `report/report.json` and `report/index.html` made from it |

Each stage does only what is missing, so rerunning `loki-survey run` after an
interruption picks up where it stopped.

- `survey.toml` holds the science: the series, the frequency regions and the
  search boxes. `check` refuses to change it once units have finished; start a
  new run directory instead. Its `[collect]` and `[evaluate]` tables change no
  search and have their own hash, so they may change at any time; every
  analysis output records the hash it was made with.
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

Within a unit, loki groups the trials above `snr_min` into candidates, as
`[candidates]` in `survey.toml` says. On real data a search can leave
millions of trials, more than loki can group in its candidate memory
(`[unit] candidate_memory_gb`, reserved within the unit's loki budget). Set
`max_per_cell` and `cell_log_width` to keep only the strongest trials of
each frequency cell.

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
| `timeout` | LSF TERM_RUNLIMIT, Slurm TIMEOUT | split into two half bands, `timeout_splits` times over; then reported: give the region more bands, or raise `time_min` |
| `crashed` | any other non-zero exit | reported with the end of its log; not retried |

A split unit is replaced by its two parts, each half its band in log
frequency (EP parts get plans of their own); `splits.jsonl` records every
split, and `units.jsonl` keeps the plan as it was. A plain `submit` also
retries units that failed, for after their cause is fixed; `watch` leaves them failed. `cancel` stops a run's jobs and returns
their units to pending.

On a GPU cluster (`[unit] backend = "cuda"`), the driver and the units need
different builds of loki. The driver (`check`, `prep`, `plan`, `collect`,
`evaluate`, `fold`) runs where you start it, often a login node without the
NVIDIA driver, where a CUDA build of loki cannot be imported; run it with a
CPU-only build. Point `[executor] python` at a Python with the CUDA build for
the units. CUDA units build their EP plans on their own GPU, since a plan is
sized by the GPU's free memory.

LSF forgets a finished job after its CLEAN_PERIOD, so an element `bjobs` no
longer lists is read from the footer LSF appends to its output in
`logs/sched/`. Slurm needs its accounting (`sacct`) for the failure classes;
without it, a failed element can only be called lost.

## Candidates

`collect` reads every unit's candidates (loki has already grouped them within
the unit) and groups them across units: candidates from any box, band or DM
whose frequencies lie within `radius_bins` resolutions 1/T are one group, and
its strongest member represents it. Grouping removes nothing; every unit
candidate stays in `/members` with its group. Each group gets labels:

- `dm_call`: `undispersed` (strongest at the lowest DM searched, at most
  `dm_zero_max`), `dispersed` (strongest inside the DMs searched), `lowest_dm`
  or `highest_dm` (strongest at an edge, so the DM may lie outside the
  search), `single_dm` or `unknown_dm`; the S/N at each DM is kept;
- `harmonic_of`: a stronger group, p/q times this one's frequency, with p and
  q up to `harmonic_max`.

A group leaves the list only through a removal rule, each off by default:
`snr_min`, `drop_undispersed`, `remove_harmonics`, `max_candidates`. Every
group a rule removes is in `collect/ledger.jsonl` with the rule and why.
`collect` refuses a run with unfinished units unless given `--partial`.

`evaluate` labels each group with the catalogued pulsars it matches (a CSV of
`name`, `f0` or `p0`, and optionally `dm`, `pb` in days and `a1` in light
seconds; `[evaluate] catalogue`), at p/q times their frequency within the
Doppler window (`doppler`, 1e-4 by default for the Earth's orbit, plus the
orbit's v/c), and with the spectral lines of the lowest-DM series. Lines and
matches are labels; they remove nothing. Its summary lists every catalogued
pulsar: searched or not, found or not, and whether a rule removed its group.

`fold` folds the strongest selected groups (`[fold] candidates`) on their own
series with loki's model of their leaf (`libloki.psr_utils.leaf_phase`, the
model the search used, circular orbits included) and scores the profile with
loki's boxcar scorer. With it come a control fold at 1.002 f, where a pulsar
smears, the two halves of the span folded apart, and, with `dm_curve`, the
same model folded on every DM. `run-unit` keeps each EP candidate's leaf for
this; an FFA candidate's leaf is built from its Taylor parameters.

`report` writes `report/report.json`, everything the page shows, and
`report/index.html`, made from that file alone: units and failures, cost,
candidates with their S/N against DM, catalogue recall, folds with their
profiles, and the ledger.

## Tests

```bash
cd survey && pytest tests
```

`test_end_to_end.py` simulates a pulsar, runs every stage with FFA and EP
boxes and checks that both find it and that `evaluate` matches it to a
catalogue, on this machine and through a fake LSF. `test_collect.py`,
`test_evaluate.py` and `test_report.py` use unit outputs written by hand
(`synthetic.py`): a pulsar over three DMs and two boxes, interference, a
harmonic and a lone candidate.
`test_batch_executor.py` drives each failure class through
`fake_scheduler.py`, a stand-in for LSF and Slurm. Units run in subprocesses
with `LOKI_SURVEY_TEST_PYTHON` (default: the Python running the tests).
