# Tutorial: a simulated survey, end to end

`loki-survey tutorial` writes a small survey of simulated series that runs on
a laptop in under a minute. It walks through every stage on data whose answer
is known, so you can see what each stage does before pointing the tool at
your own data.

## Install

loki-survey runs on loki's Python module. Build loki from a clone of its
repository, then install loki-survey from the same clone:

```bash
git clone https://github.com/pravirkr/loki && cd loki
pip install .
pip install ./survey
```

The package named `loki` on PyPI is a different project. If loki-survey finds
no loki, or one older than it needs, it stops and says so.

## Write the survey

```bash
loki-survey tutorial demo
```

`demo/` now holds:

- `data/`: four dedispersed series, `tutorial_DM00.00` to `tutorial_DM60.00`
  (DM 0, 20, 40 and 60), each 2^18 samples of 64 us (16.8 s) as PRESTO
  `.dat` and `.inf`;
- `survey.toml`: search 100 to 200 Hz with 32 bins, in two boxes. An FFA box
  and an EP box both search accelerations up to 100 m/s^2;
- `site.toml`: run the units on this machine, two at a time;
- `catalogue.csv`: the one known pulsar.

The series hold white noise and three sources. Each source appears at the
S/N it would have at that trial DM, so it is strongest at its own DM:

| Source | Frequency | DM | Notes |
| --- | --- | --- | --- |
| PSR_A | 142.3 Hz | 40 | isolated, in `catalogue.csv` |
| PSR_B | 117.1 Hz | 20 | accelerated at 30 m/s^2, not catalogued: the find |
| RFI_150Hz | 150 Hz | 0 | a sinusoid, strongest undispersed: interference |

## Run it

```bash
loki-survey run demo
```

`run` goes through every stage. Each stage does only what is missing, so
running it again after an interruption carries on. The output, shortened:

```text
survey 'tutorial': science 8e22424931b6, analysis 81f58c1a8571
  1 regions, 2 boxes
  wrote demo/resolved.json
prep: tutorial_DM00.00.dat
...
plan: EP plan 98bfbf27c655f59a (for tutorial_DM00.00__mid__b0__ep_accel)
plan: 8 units, 1 EP plans
          ep_accel x mid               4 units
         ffa_accel x mid               4 units
start tutorial_DM00.00__mid__b0__ffa_accel (attempt 1, 4.0 GB)
start tutorial_DM00.00__mid__b0__ep_accel (attempt 1, 4.0 GB)
...
done 8  running 0  queued 0  pending 0  failed 0  of 8
collect: 16 unit candidates from 8 of 8 units in 3 groups; 3 selected, 0 removed (ledger demo/collect/ledger.jsonl)
evaluate: 1 of 1 catalogued pulsars in band found
evaluate: 1 lines in tutorial_DM00.00; 1 groups lie on one
fold: group 0 at 142.300764 Hz: S/N 16.26 (search 16.95, control 2.48, halves 11.54/11.46; taylor)
fold: group 1 at 150.005728 Hz: S/N 15.46 (search 15.65, control 2.79, halves 11.21/10.91; taylor)
fold: group 2 at 117.100519 Hz: S/N 14.09 (search 14.46, control 2.64, halves 10.51/10.93; taylor)
report: demo/report/index.html
```

On another machine the last digits may differ. What each line means:

- **check** validates both files and writes `resolved.json`, every value
  used. Its science hash covers everything that changes what a search finds.
- **prep** reads each series, normalises it with loki's robust preprocessing
  and writes `prep/`.
- **plan** splits the survey into units. Here that is 4 series × 1 band × 2
  boxes = 8 units. It also builds the plan the EP box needs, once, shared by
  its four units.
- **submit** runs each unit in its own process. A unit writes `out/<unit>.h5`
  (loki's candidates with the run's provenance) and `out/<unit>.json`.
- **collect** groups the 16 unit candidates across units: candidates from any
  box or DM within two frequency resolutions 1/T are one source. Three groups
  remain, one per source. Each group is labelled by where it is strongest:
  PSR_A and PSR_B are `dispersed` (strongest at DM 40 and 20), and the
  sinusoid is `undispersed` (strongest at DM 0).
- **evaluate** matches the groups with `catalogue.csv` and finds PSR_A. It
  also looks for spectral lines in the DM 0 series and finds the sinusoid's
  line at 150 Hz. Matches and lines are labels: nothing is removed.
- **fold** folds each group's strongest candidate with loki's own model of
  its leaf, and checks it three ways:
  - a control fold at 1.002 times the frequency, where a real signal smears
    (S/N 2.5 to 2.8 here);
  - the two halves of the observation, which a steady signal fills about
    equally;
  - its S/N at every DM.
- **report** writes `report/index.html`.

## Read the report

Open `demo/report/index.html` in a browser. It shows:
- the units and what they cost;
- the candidates, each with its S/N against DM, and the catalogue recall;
- the lines;
- the folded profiles;
- the ledger of anything a rule removed (nothing here).

PSR_B is the candidate to follow up. It is dispersed, matches nothing in the
catalogue, and folds to S/N 14 with its control at the noise. The sinusoid
folds well too: it is undispersed and on a line, which is how interference
shows itself.

## Change things

- **The analysis** (`[collect]`, `[evaluate]`, `[fold]`) can change at any
  time without searching again. For example, drop undispersed groups:

  ```bash
  printf '\n[collect]\ndrop_undispersed = true\n' >> demo/survey.toml
  loki-survey check demo && loki-survey collect demo && loki-survey evaluate demo
  loki-survey fold demo && loki-survey report demo
  ```

  The sinusoid's group is now in `collect/ledger.jsonl`, with the rule that
  removed it and why ("strongest at DM 0, the lowest searched"); collect says
  "2 selected, 1 removed", and fold skips it.
- **The science** (the series, regions, boxes, `[search]`, `[candidates]`) is
  fixed once units have run; `check` refuses a change. Start a new directory:
  `loki-survey tutorial demo2`, edit its `survey.toml`, run it.
- **The machine** (`site.toml`) can change at any time. To run the same
  survey on a cluster, set `[executor] kind = "lsf"` or `"slurm"` and the
  scheduler's table. `loki-survey new` writes a template with every key
  explained. Then `loki-survey run demo --wait`.

[outputs.md](outputs.md) describes every file a run writes.
