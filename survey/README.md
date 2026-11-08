# loki-survey

Run a loki pulsar survey end to end: from dedispersed time series to a
candidate list, on a laptop, a single server or (later) a cluster.

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

## Tests

```bash
cd survey && pytest tests
```

`test_end_to_end.py` simulates a pulsar, runs every stage with FFA and EP
boxes and checks that both find it. Units run in subprocesses with
`LOKI_SURVEY_TEST_PYTHON` (default: the Python running the tests).
