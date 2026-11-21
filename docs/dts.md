# Dynamic threshold scheme

`loki::detection::DynamicThresholdScheme` (Python
`libloki.thresholds.DynamicThresholdScheme`) designs the pruning thresholds of
the hierarchical (EP) search: one threshold per stage. This page describes the
problem, the five modes, the default pipeline (`shared`), the bank descent
(`descent`) and why they are built the way they are. Code comments only describe the implementation.

## Problem

The search splits the data into M + 1 segments and runs M stages. Stage s
branches every surviving node by B_s, scores each child (the maximum boxcar
S/N of its fold) and keeps the children above the threshold γ_s. A schedule
γ_0 .. γ_{M-1} fixes

- the cost, every scored node plus `w_terminal` times the survivors of the
  last stage (they go on to the next processing step):

  ```text
  C = 1 + Σ_s n_{s-1} B_s + w n_{M-1},   n_s = n_{s-1} B_s α_s(γ),   n_{-1} = 1
  ```

  with α_s the conditional survival of noise at stage s (`w_terminal = 0` is
  the historical definition);
- the detection probability Pd: the probability that a signal of the design
  S/N (`snr_final`) survives every stage.

The scheme always simulates the boxcar score over the configured widths
(`ducy_max`, `wtsp`). With `use_boxcar_kadane` the search prunes on the Kadane
score instead, a faster approximation of it; the thresholds are still designed
for the boxcar, and the scores the search reports are the boxcar's (see
[candidates](candidates.md)).

The objective is to minimise the cost per detection L = C / Pd subject to
Pd ≥ `min_pd`. A Pd above `min_pd` is allowed when it lowers L (short patterns
often have their optimum there).

Neither C nor Pd has a closed form (correlated, maximum-over-templates random
walks over up to ~130 stages), so both are Monte Carlo estimates, and an
exhaustive search over the schedules is intractable. Four modes are a dynamic
programme over cells (stage, threshold, Pd bin) that keeps one candidate per
cell: an approximate, compressed-state search with no optimality guarantee.
The cell is not a sufficient state (two paths into one cell leave different
noise populations behind), its edge weights are Monte Carlo estimates, and
the grid and the beam limit what it can reach. Mode `descent` does not search
cells: it improves one schedule on fixed samples (see
[the bank descent](#the-bank-descent)).

## Modes

| mode | search | after the search | use |
|---|---|---|---|
| `legacy` | one simulation per transition, analytic beam | none | reference, paper figures |
| `improved` | one simulation per parent cell, analytic beam (or `guess`) | none (`calibrate()` optional) | reference, paper figures |
| `shared` (default) | shared signal trials, guided beam, one pass | `solve()`: tighten, repair, report | production |
| `robust` | as `shared`, plus a second pass | `solve()`: shortlist, block refinement, repair, report | when ~2.5x the run time buys a lower L |
| `descent` | none: constrained descent of one schedule on a signal bank and a noise genealogy | `solve()` is the whole mode: descent, pin, report | lowest C at Pd = `min_pd`; a floor on the last threshold (`fap`) |

The EP planner uses `descent` on both backends, once per region, with
`snr_min` as the floor of the last threshold and each region started from the
previous one's z-scores; every chunk of the region runs that schedule (see
[memory.md](memory.md#planning)).

## The winner's curse, and what the default mode does about it

Picking the best of many noisy candidates also picks the one whose estimate
came out luckiest (the optimiser's or winner's curse). The scheme suffers
from it twice.

1. **Pd side.** In `legacy` and `improved` every candidate's Pd comes from its
   own resampled signal trials, and each cell keeps the cheapest candidate
   whose Pd cleared its bin, i.e. the one whose Pd estimate was high by chance.
   This acts at every stage: on the 127-stage production pattern the in-run Pd
   of `improved` is 10-25 % optimistic, and the delivered Pd scatters by about
   7 % between seeds.
2. **Cost side.** At a stage with B_s = 1 the threshold barely matters, because
   the next stage scores nearly the same fold. If every transition is allowed,
   the search follows noise in its 1024 noise trials there and returns paths
   with sudden drops that cost about 10 % in L on fresh samples. The fixed
   `thres_neigh` of `improved` hides this, but it is absolute (in S/N): it is
   too tight where the score distributions move fast (early stages, short
   patterns, issue #15).

No estimate that was used to choose a schedule is unbiased for the chosen
schedule, so the curse cannot be removed inside the search. It can be reduced
there (common random numbers, fewer effective choices), and it is removed by
measuring the chosen schedule on samples that took no part in choosing it
(sample splitting). Mode `shared` does exactly that:

- **Common random numbers.** Every path's Pd is counted on the same unpruned
  signal trials (16 × `ntrials`, at most 2^20; a cell keeps a bitmask of the
  trials still above its thresholds). Candidates are compared on the same
  trials, so their differences are not noise. Noise trials that survive keep
  their slot; failed slots are refilled stratified over the survivors.
- **Guided beam (fewer effective choices).** An unpruned pilot simulation
  (2^15 trials, sorted on the device) gives per stage the signal-score
  standard deviation σ_s, a feasibility window [Q_s(0.02 p / M),
  Q_s(1 - p / 2)] of signal quantiles (p = the lowest Pd bin), and the guess
  g_s = min(median signal score, trials-aware S/N), clamped into the window.
  Stage s searches the grid thresholds within `beam_sigma` σ_s of g_s (inside
  the window), and a transition may change the offset γ_s - g_s by at most
  `reach_sigma` σ_s. Unclamped, a bright pulsar's guess sits at the trials
  term, below the later stages' windows, and the reach forces the first
  thresholds up to climb into them.
  The threshold may still drop (heavily branched stages), but only as the
  score distributions move. Measured on 5 branching patterns × 2 targets, the
  paths of an unconstrained search lie within 2.5 σ of the guess and change
  their offset by at most 0.9 σ per stage; hence the defaults 3 and 1. A reach
  of 0.5 σ costs 9 % on a 4-segment pattern.
- **Ranking.** A cell keeps the candidate of lowest cost / Pd (cost so far,
  plus `w_terminal` times the survivors at the last stage). In the search's
  states, a noise survivor count below 8 per 1024 trials is treated as 8 (a
  near-zero count is mostly luck, and a deep search would exploit it): the
  floored pass rate enters the cost so far, every descendant's cost and the
  `robust` cost to go. The survivor maps keep the actual trials, and `solve()`
  measures C on fresh noise, so the reported C is not floored. The Pd grid has `nprobs` bins from `min_pd`
  to 1 (`min_pd` is an edge) and at most 16 below it.
- **`solve()` on fresh samples** (2^17 signal trials each, 2^15 noise trials):
  1. *tighten* on bank A: lower every threshold by one common shift if fewer
     than the target (`min_pd` + 2 binomial standard errors) of the trials
     survive, then raise each threshold to just below the lowest score of the
     surviving trials (no sampled signal is lost);
  2. *repair* on bank B: lower all thresholds by one common shift if the Pd
     there is below the target;
  3. *report* on bank C, which chose nothing: Pd is a direct count with its
     binomial error; C is an evaluation on fresh noise.

Mode `robust` adds a second search pass (candidates ranked by cost so far plus
the first pass's cost to go of their noise survivors, the beam recentred on
the first pass's path), compares up to four candidate paths on fresh noise, and
moves blocks of stages by ±2 grid steps when that lowers L by 2 % on one fresh
noise sample and by 0.5 % on a second. These constants are heuristics chosen
once in the research workspace (`dts_work`), not tuned per pattern.

## The bank descent

Mode `descent` solves a narrower problem than the other modes, the one the
long patterns of a production search actually pose: **minimise C subject to
Pd = `min_pd`**. On those patterns L rises steeply with Pd, so the constraint
binds and every bit of Pd above the target is paid for (a schedule delivered
at Pd 0.11-0.17 for a target of 0.1 costs 6-41 % more than the same schedule
shifted to 0.1). It has no cells, no Pd bins, no threshold grid and no beam,
and it runs at the search's own `nbins` whatever its size.

Two samples make both quantities exact functions of the schedule:

- **Signal bank.** The unpruned scores of `descent_nsignal` (2^15) signal
  trials at every stage. Pd(γ) is the fraction of trials above every
  threshold, for any γ.
- **Noise genealogy.** One noise simulation pruned at a reference schedule
  γ - margin that keeps, per stage, every trial's score and the slot of the
  trial it was resampled from. A noise node of the search tree survives a
  schedule γ' ≥ reference when its whole ancestry is above γ', so C(γ') is a
  count along the ancestry, on the same noise for every γ'. One backward pass
  gives every trial the cost of the subtree below it.

From these the change of C and of Pd for moving one threshold, or one block of
consecutive thresholds, by ±step is counted exactly (no derivative, no
noise between the candidates). A multiplier λ, the cost paid per unit of Pd,
selects the moves: raise a block where the cost it saves exceeds λ times the
Pd it loses, lower it where λ times the Pd gained exceeds the cost added. λ is
bisected until the selected moves leave Pd unchanged; a common shift of all
thresholds then pins Pd to the target on the signal bank, and the move is
kept when the genealogy's cost fell. Blocks matter: consecutive late stages
score nearly the same fold, so one threshold alone cannot move there (single
stage moves stall 5-8 % above the block version at a low target). A sweep
goes through disjoint blocks of 2^k, ..., 2, 1 stages, with the block
boundaries offset by half a block on alternate sweeps.

A genealogy is valid only above its reference, so the descent rebuilds it a
few times around the current schedule with a shrinking margin and step. The
default plan has three levels: 4096 trials at margin 0.4 and step 0.2 with
every block size, 8192 trials at 0.3/0.1 with blocks of at most 32 stages,
and 8192 trials at 0.2/0.05 with blocks of at most 8. A level runs a second
time, on a new genealogy, when its first pass lowered the cost by more than
25 %; `descent_passes` replaces the plan by a list of passes that each run
once.

**The start.** The solutions of long branching patterns share a shape in
units of the unpruned signal score, z_s = (mean_s - γ_s) / σ_s: loose at the
very first stages (z about 2), at the mean signal score within the first few
percent of the stages (z about 0), then rising steadily to z about 1.7 at the
end. The descent builds two candidates, that shape and the same z at every
stage, pins both to the target, evaluates both on one noise sample (4096
trials, the same for both) and starts from the cheaper one. On 127-stage
patterns the shape is 1.1 to 6.8 times cheaper than the flat start at
targets up to 0.5 and is chosen there; at a target of 0.9 and on patterns of
4-32 segments the flat one usually is. From the flat start the same plan
ends about 2 % above six fixed passes (worst case seen: 16 %), which is
what a pattern unlike these would get. The
shape is only a start: the moves are free, and a start that is far off makes
the levels repeat.

After the descent the schedule is measured on samples that took no part in
it, as in `shared`: a common shift of either sign pins Pd to `min_pd` + 2
binomial standard errors on a fresh bank of 2^17 signal trials, a second
fresh bank reports Pd, and a fresh noise evaluation reports C.

**The last threshold.** `final_threshold` is a floor on γ_{M-1}; `fap`
derives one as the S/N whose one-sided normal tail is `fap` / (nbins × number
of boxcar widths), the false-alarm probability of one fold with the trials of
its templates (`fap` 0: no floor; when both are set the higher one holds).
This is a per-fold probability: for a false-alarm budget of the whole search,
divide it by the number of folds that reach the last stage (`fap` 0.01 is
only S/N 3.6-4.8 at 16-1024 bins, and a floor of 7 needs about 10^-9). The
descent is free above the floor. A block that holds the last stage moves
down with the last threshold stopping at the floor, and when a pin would
take it below, it rests on the floor and the other stages shift to keep Pd;
a floor below the free solution changes nothing. If the signal cannot reach
the target through the floor alone, the status is `short` and `pd_ceiling`
in the diagnostics is the Pd the floor allows. On 170 patterns of a
production plan (16-128 bins, target 0.1) the free last thresholds were
5.6-8.2, below 7 on 45 of them. With a floor of 7 those 45 end at 7.00-7.13
with Pd 0.100-0.103 and cost 3.8 % more than the free schedule (geometric
mean, judged on independent samples); raising the last threshold of the
free schedule and shifting the others costs 2.9 %, so the descent with a
floor is within 1 % of that baseline but not better than it.

**A given start.** `descent_start` replaces the two candidates, e.g. by the
schedule solved for a neighbouring pattern. Measured with fixed passes on 43
pairs of neighbouring patterns (same parameter count and bin count),
relative to a six-pass solve from the flat start: C 0.980 with one pass per
level, 0.986 with the last two levels only (0.11 s against 0.24 s).

The solution also reports the schedule as `zscores`, (mean - threshold) /
standard deviation of the unpruned signal score per stage, and
`descent_start_zscores` reads the start in that form. Schedules carry over
to other bin counts, S/N and pulse widths in z-scores: a chain of ten
patterns through 16, 32, 64 and 128 bins, each started from the previous
warm solution with two passes, stays at 0.95-1.04 of six-pass solves from
the flat start (mean 0.98), and chains of thresholds at one bin count do not drift either (0.988
over 37 links, worst link 1.085).

What it guarantees and what it does not: at the end no block move of the last
step size lowers C at equal Pd on its samples, which is a first-order
(Kuhn-Tucker like) condition, not a proof of a global minimum. The evidence
that it is close is empirical: in the research prototype, starts as
different as an equal-quantile schedule and a pinned `robust` solution ended
at the same cost, the cost of the library mode scatters by 2-5 % between
seeds, and chains of warm starts end about 2 % lower still.

## Measured behaviour

Production pattern (127 stages), nbins 64, S/N 10, `min_pd` 0.1, CUDA on an
L40S. Each schedule is evaluated with `evaluate_scheme` (65536 fresh trials);
mean and coefficient of variation over 8 search seeds:

| mode | Pd | cv | C | cv | L | cv | time |
|---|---|---|---|---|---|---|---|
| `improved` | 0.088 | 0.068 | 4.9e4 | 0.18 | 5.5e5 | 0.12 | 0.16 s |
| `shared` | 0.104 | 0.043 | 5.6e4 | 0.11 | 5.4e5 | 0.074 | 0.22 s |
| `robust` | 0.103 | 0.023 | 5.1e4 | 0.041 | 4.9e5 | 0.039 | 0.60 s |

The same suite with `descent` (2026-11, L40, 8 seeds, 4 host threads; Pd is
the lowest and highest seed, C the geometric mean):

| case | mode | Pd | C | cv of C | time |
|---|---|---|---|---|---|
| production, `min_pd` 0.1 | `shared` | 0.102-0.114 | 5.75e4 | 0.099 | 0.23 s |
| | `robust` | 0.102-0.108 | 5.26e4 | 0.024 | 0.56 s |
| | `descent` | 0.101-0.104 | 4.68e4 | 0.020 | 0.18 s |
| production, `min_pd` 0.05 | `shared` | 0.051-0.056 | 1.29e4 | 0.129 | 0.23 s |
| | `robust` | 0.052-0.052 | 1.14e4 | 0.029 | 0.54 s |
| | `descent` | 0.051-0.053 | 1.07e4 | 0.030 | 0.19 s |

On the short patterns of issue #15 the shared modes deliver a Pd well above
the target because L is lowest there (8 segments, target 0.5: 0.75-0.78;
16 segments, target 0.1: 0.50-0.55; 32 segments at S/N 6, target 0.1:
0.11-0.19); `descent` delivers the target (0.502-0.505, 0.100-0.102 and
0.100-0.102) in 0.02-0.05 s, at a higher L (8 segments: 4.0 against 3.0).
That is the difference of the two
objectives, not a defect of either: use a shared mode when the lowest cost
per detection is wanted, `descent` when the target Pd is the requirement.

On 127-stage patterns the descent takes 0.14-0.16 / 0.16 / 0.16-0.18 /
0.23 s at 16 / 32 / 64 / 128 bins on the GPU with 4 host threads,
constructor included (`shared` with its constructor, `run()` and `solve()`:
0.11 / 0.14 / 0.23 / 0.43 s): slower than `shared` at 16 and 32 bins,
faster from 64. The host passes over the genealogy are about 0.08 s of it
whatever the bin count. Above 128 bins the unpruned signal samples dominate (the descent's
2^15 trials and the two final banks of 2^17): 0.68 s at 256 bins, 2.0 s at
512 and 7.4 s at 1024, with no refit and no proxy bin count. On the CPU it
took 2.6 s at 16 bins with 4 threads.

On 230 branching patterns of a production plan (2-5 parameters, 16-1024
bins, target 0.1, one seed each) every solve returned `met` or
`within_tolerance`, and the Pd counted on 2^16 independent trials was
0.0974-0.106 (the count itself has an error of 1.2 %). Over 8 seeds on 22 of
them, counted on 2^17 trials, Pd was 0.0990-0.1044 in all 176 solves, with a
standard deviation per pattern of 0.0004-0.0011 (the binomial error of the
pin is 0.0008), and C scattered by 1.5 % (median; 4.8 % on the worst
pattern).

## Settings

`SharedOptions` holds the settings of the shared modes: `min_pd`,
`w_terminal` (1), `beam_sigma` (3) and `reach_sigma` (1). The constructor's
`nprobs` is the number of Pd bins from `min_pd` to 1; `ntrials` (1024) is the
noise population per cell. `prob_min`, `guess`, `beam_width` and
`thres_neigh` apply to `legacy` and `improved` only.

Mode `descent` reads `min_pd`, `w_terminal`, `fap`, `final_threshold`,
`descent_passes` (builds of the noise genealogy: trials, margin below the
current schedule, step, largest and smallest block), `descent_nsignal`
(2^15; 2^14 costs 2-7 % in C), `descent_start` and `descent_start_zscores`. `run()` does nothing in this mode and the
constructor's `ntrials`, `nprobs`, `nthresholds` and beam settings are not
used.

`ref_ducy` is the reference pulse's full width at a tenth of maximum (FWTM),
in turns: the simulated pulse is a Gaussian of rms width
σ = `ref_ducy` / (2 sqrt(2 ln 10)). For a pulse of measured FWHM duty d, set
`ref_ducy` = sqrt(ln 10 / ln 2) d ≈ 1.82 d.

The bin width is the resolution at which the search trades Pd against cost,
and it matters most for L. Run time and run() memory grow linearly with
`nprobs`. Measured before the guided beam (production pattern, nbins 64,
`ntrials` 1024, grid anchored at `min_pd` 0.1, CUDA on an L40S, 16 seeds; L of
the selected path rescored on 2^20 trials):

| bins per decade | 19 | 29 | 39 | 49 | 59 | 79 |
|---|---|---|---|---|---|---|
| L / L(79) | 1.53 | 1.34 | 1.20 | 1.13 | 1.05 | 1.00 |
| scatter of L | 0.23 | 0.20 | 0.15 | 0.14 | 0.12 | 0.16 |
| run time (s) | 0.09 | 0.13 | 0.16 | 0.20 | 0.24 | 0.32 |
| memory (GiB) | 1.3 | 2.0 | 2.7 | 3.3 | 4.0 | 5.3 |

The default (40 bins above `min_pd`) is the compromise. More thresholds do not
help (`nthresholds` 50-200 were equivalent).

`legacy` and `improved` accept `guess = "analytic"` (the lower of the expected
signal S/N and the trials-aware S/N, the beam of the paper),
`"pilot"` (the median unpruned signal score instead of the expected S/N) or
`"window"` (every grid threshold in the feasibility window and every
transition, two passes ranked with a cost to go). On short patterns the
analytic beam misses the cheap paths (issue #15); `window` is 2.4x slower at
the production pattern.

## Results and files

`solve()` returns a `SchemeSolution`: the thresholds, their reported Pd (with
standard error and trial count), C and L, the noise evaluation per stage, the
raw search selection with its in-run (optimistic) Pd and C, the shortlist
(`robust`), named diagnostics, and a delivery status (`met`,
`within_tolerance`, `short`, `no_path`).

`save()` writes the search states (stage × threshold × Pd bin), the grids,
`beam_lo`/`beam_hi`, `guess_path` (the guess searched around: in `robust` the
second pass's recentred guess, which follows the first pass's path by
construction) and, in the shared modes, `guess_path_pilot`; after `solve()`
also a group `solution`. In the shared modes the `cost` of the last stage's
cells is the terminal rank (C + w n_{M-1}) / Pd, so ranking the saved terminal
cells by `cost` selects the same path as the solver; in the other modes and
stages it is `complexity_cumul / success_h1_cumul`.

## Evaluation

`DynamicThresholdScheme::evaluate(thresholds, ntrials, seed)` measures a
schedule on the engine's backend with a given seed (on CUDA with a
device-wide compaction). The free function `evaluate_scheme(...)` is the same
evaluation on the CPU; like `determine_scheme(...)`, it takes an optional
`seed` and draws a fresh random one when it is unset. The EP planner
evaluates the chunk branching patterns with `evaluate()` on its own backend.

## Reproducibility

Results depend on the seed, `ntrials`, `nprobs`, the sample sizes above and,
on the GPU, `batch_size`; on the CPU they do not depend on the thread count.
`bench/scripts/schemes/golden_solve.py record|check` keeps a bit-exact CUDA
regression golden of both shared modes (valid for one toolchain);
`golden_evaluate.py` does the same for `evaluate()`.

## Limitations

- No optimality guarantee: the search is approximate and the constants above
  are heuristics. The descent stops at a schedule no block move improves,
  which need not be the global minimum.
- `descent` does not look for a Pd above `min_pd` that lowers L (short
  patterns). It has no search states: `save()`, `get_states()` and
  `get_best_path_thresholds()` raise in this mode, and there is no
  shortlist.
- The samples of `descent` do not shrink with the bin count (2^15 signal
  trials for the moves, 2^17 each for the pin and the report, as `solve()`
  of the shared modes): above 128 bins they are its run time and its memory.
- The EP planner floors the last threshold at `snr_min`; `fap` is available
  from the scheme's API only.
- The Pd guarantee is statistical (target `min_pd` + 2 standard errors of a
  2^17-trial count) and holds for the simulated signal model: a pulse of the
  reference shape at the design S/N, folded without loss. Mismatch loss of the
  EP search and multi-leaf detection are outside it.
- Pd is survival to the final candidate list, not completeness at a
  false-alarm rate. In the four search modes the last threshold is chosen by
  cost alone (`w_terminal` weighs the final survivors) and nothing bounds the
  false-alarm probability per final trial; `descent` with `fap` or
  `final_threshold` bounds it from below (issue #42). At low S/N a high Pd can need a final list that keeps about
  10^-5 of the noise, so the next processing step must be cheap enough for it.
- The reported C is an independent but noisy Monte Carlo estimate; on very
  large irregular trees it can be off by a large factor.
