# What a run writes

Everything a run makes lives in its directory, next to `survey.toml` and
`site.toml`. Results are written under a temporary name and then renamed, so
a result file that exists is complete. Each stage records the
hashes of what it read, so any output can be traced back to the
configuration and the inputs that made it.

```text
run/
  survey.toml, site.toml   what to search, and where to run it
  resolved.json            check: every value used, the science and analysis hashes
  prep/<series>/           prep: normalised, masked, downsampled arrays and a manifest
  units.jsonl              plan: one line per unit
  splits.jsonl             units split after hitting their time limit, if any
  plans/                   plan: the EP plans, shared by units of the same geometry
  out/<unit>.h5            a unit's candidates and provenance
  out/<unit>.json          a unit's summary; its presence means the unit is done
  out/raw/                 loki's full result files, for [output] raw_fraction of units
  collect/                 candidates grouped across units, and the ledger
  evaluate/                catalogue matches and spectral lines
  folds/                   the folds of the strongest groups
  report/                  report.json and index.html
  status.sqlite            every attempt of every unit: state, host, failure class
  logs/                    one log per unit attempt; logs/sched/ for the scheduler's
  jobs/                    cluster only: the job scripts and their unit lists
  tmp/                     units at work; empty when nothing runs
  quarantine/              work directories of failed attempts, kept for a look
```

## Unit outputs: `out/<unit>.h5` and `out/<unit>.json`

A unit is one series × frequency band × search box. Its id says which, for
example `tutorial_DM40.00__mid__b0__ep_accel`. Its HDF5 output keeps loki's
common candidate tables, described in loki's `docs/candidates.md`:

- `/candidates`: the candidates loki selected, sorted by S/N. Columns:
  - `snr`, `width`, `nbins`, `duty_cycle`;
  - `f0`, `f1`, `f2`, the frequency and its derivatives at the middle of the
    series;
  - `param_sets`, loki's native parameters;
  - `id`, `cluster_id` and `member_count`;
- `/clusters`: every group loki made within the unit, selected or not;
- `/search`, `/observation`, `/processing`: loki's search settings, the
  series' metadata (DM, `tobs`, `reference_time_seconds`) and the candidate
  options;
- root attributes: `n_detections`, `n_clusters`, `n_candidates`, and
  `n_cell_dropped` when `[candidates] max_per_cell` is set;
- `/survey`, added by loki-survey:
  - attributes: the unit, `science_sha256`, `inputs_sha256`, `arrays_sha256`,
    the code versions, host, attempt, backend, threads and loki's memory
    budget;
  - `candidates_republished` and `resumed_from`, which say whether the
    candidate step ran again from loki's spool;
  - `/survey/leaves` (EP only), each candidate's final leaf, which `fold`
    uses;
- `/level_stats/<chunk>/<run>` (EP only): loki's per-level table of each
  pruning run (survivors, harvests, `n_overloads`). An overload means the run's
  buffer filled and loki trimmed its survivors.

`out/<unit>.json` is the summary:
- `n_candidates` and `n_clusters`;
- `n_runs`, `n_runs_overloaded` and `n_overloads`: EP pruning runs, those whose
  buffer overflowed, and the overloads in all (0 for FFA units). The report
  lists the units with overflowed runs;
- `search_s`, `total_s` and `peak_rss_gb`;
- `output_sha256`, the hash of the HDF5 file;
- `science_sha256`;
- `candidates_republished` and `resumed_from`.

## `collect/`

`candidates.h5` holds four tables, each a group of equal-length datasets:

| Table | One row per | Columns |
| --- | --- | --- |
| `/units` | unit | `id`, `series`, `region`, `band`, `box`, `kind`, `dm`, `f_min`, `f_max`, `tobs`, `n_candidates`, `output_sha256` |
| `/members` | unit candidate | `group`, `unit` (row in `/units`), `unit_candidate_id`, `dm`, `snr`, `f0`, `f1`, `f2`, `width`, `nbins`, `duty_cycle` |
| `/groups` | group (one source) | see below |
| `/dm_curves` | group × DM | `group`, `dm`, `snr`: the best S/N of the group at each DM |

Groups are numbered by falling S/N. The columns of `/groups`:
- `id`, and `rep_member`, the strongest member (a row in `/members`);
- the representative's `f0`, `f1`, `f2`, `snr`, `width`, `nbins`,
  `duty_cycle`, `dm` and `unit`;
- `n_members`, `n_units`, `n_dms`, `boxes`, and `f_extent_bins`, the span of
  the members' frequencies in resolutions 1/T;
- `dm_call`, and `lowest_dm_snr_ratio`, the S/N at the lowest DM over the
  best;
- `harmonic_of`, `harmonic_p` and `harmonic_q`, a stronger group at p/q times
  this one's frequency (-1 if none);
- `selected`, and `removed_by`, the rule that removed the group, if any.

The file's attributes are the DMs searched (`dms`), `science_sha256`,
`analysis_sha256` and `inputs_sha256`.

`ledger.jsonl` has one line per removed group. Each line gives:
- `rule` and `reason`;
- `group`, `f0`, `snr`, `dm`, `dm_call` and `n_members`.

Nothing leaves the candidate list without a line here. `summary.json` gives:
- the counts;
- the DM calls and harmonics;
- the units collected or missing;
- the hashes of `candidates.h5` and of the ledger.

## `evaluate/`

`matches.jsonl` has one line per group that matched something. Each line
gives:
- `group`, `f0`, `snr` and `dm`;
- `catalogue`: each matching pulsar, with its `name`, the harmonic `p`/`q`,
  `df_hz` and `dm_offset`;
- `line`: the spectral line the group lies on, if any.

`summary.json` lists:
- every catalogued pulsar: in band or not, `found` (a group at its own
  frequency, 1/1), `fundamental_selected` (such a group was kept), and its
  `best` group with whether a rule removed it. `best` is the group that tells
  most: the strongest kept fundamental group, else the strongest removed one,
  else the strongest kept harmonic group, else the strongest removed one. The
  page and the command line state it as selected, found as p/q, removed by a
  rule, or not found;
- the lines found.

A match is a coincidence in frequency within the window, not a check that the
group is that pulsar: a sub- or high-order harmonic match can be chance,
especially with many groups and `harmonic_max` 16.

`lines_<series>.json` caches the line search of the lowest-DM series: the
frequency resolution `df`, the power threshold, and each line's frequency
range and power.

## `folds/`

`folds.h5` has a group `group_NNNNNN` per folded group, named by the group's
row in `/groups`. Its attributes:
- `unit`, the unit whose series was folded;
- `model`, the phase model loki used: `taylor`, `circular_snap` or
  `circular_crackle`;
- `snr` and `width`, the folded S/N and its boxcar width;
- `snr_control`, the S/N at the control frequency.

Its datasets:
- `profile` and `profile_control`, shape (2, nbins): the sums of the series
  and of its variances per bin, from `libloki.fold.fold_phase`;
- `leaf`, the leaf folded;
- `dm_curve`, the (DM, S/N) pairs of the same model folded on every series.

`summary.json` repeats these numbers per group, with the S/N of the two
halves of the observation, and the hashes of the analysis and of
`collect/candidates.h5`.

## `report/`

`report.json` holds everything the page shows, and `index.html` is made from
it alone:
- the run and its units, with failures and cost;
- the candidates, with their DM curves;
- catalogue recall, lines and folds;
- the ledger.

evaluate and fold each record the sha256 of the collect candidates they read.
collect renumbers its groups, so the report leaves out an evaluation or folds
made from an earlier collect. `stale` lists the stages left out, and the page
names the commands to run again.

To compare runs or build your own plots, read `report.json` or the tables
above, not the HTML.
