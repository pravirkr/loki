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
| `descent` | none: constrained descent of one schedule on a signal bank and a noise genealogy | `solve()` is the whole mode: descent, pin/repair, report | near-lowest C at Pd = `min_pd` (`pd_policy = "pin"`, default) or low L = C / Pd at Pd ≥ `min_pd` (`pd_policy = "free"`) across all bin counts; optional floor on the last threshold (`fap`) |

The EP planner uses `descent` with `pd_policy = "pin"` on both backends, once
per region at the region's true `nbins`, with each region started from the
previous one's z-scores; every chunk of the region runs that schedule, and
`snr_min` is applied to the reintegrated post-`ascend` scores when writing
candidates rather than flooring the pruning schedule (see
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

Mode `descent` works directly in continuous threshold space without a cell
grid, Pd bins or a beam, and runs natively at the search's own `nbins` up to
1024. Its `pd_policy` selects the objective:

- **`pd_policy = "pin"` (default, used by `EPRegionPlanner`):** minimises C
  subject to Pd = `min_pd`, always pinning two-sided to the target on a fresh
  bank. On long production patterns L = C / Pd rises steeply with Pd, so
  pinning keeps delivered Pd unimodal and tightly centred on the target across
  seeds (a schedule delivered at Pd 0.11-0.17 for a target of 0.1 costs 6-41 %
  more than the same schedule shifted to 0.1).
- **`pd_policy = "free"`:** minimises L = C / Pd subject to Pd ≥ `min_pd` via
  Dinkelbach fractional programming. On short or low-cost trees where fixed
  early-stage work dominates, L falls with Pd and the descent settles at an
  interior Pd > `min_pd`.

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
noise between the candidates; per pass only signal trials above the reference
schedule need to be scanned). A multiplier λ, the cost paid per unit of Pd,
selects the moves: raise a block where the cost it saves exceeds λ times the Pd
it loses, lower it where λ times the Pd gained exceeds the cost added. Under
`"pin"`, λ ≥ 0 is bisected to hold Pd at the target and a common shift pins Pd
on the signal bank; under `"free"`, a Dinkelbach lower bound λ ≥ C / Pd lets
moves that lower C / Pd while keeping Pd ≥ target apply directly (including a
full-schedule shift block to adjust Pd globally: one block over every stage on
every sweep, whatever a pass's block sizes), bisecting above C / Pd only
when the constraint binds. Blocks matter: consecutive late stages score nearly
the same fold, so one threshold alone cannot move there (single-stage moves
stall 5-8 % above the block version at a low target). A sweep goes through
disjoint blocks of 2^k, ..., 2, 1 stages, with the block boundaries offset by
half a block on alternate sweeps (except the `"free"` full-schedule block).

A genealogy is valid only above its reference, so the descent rebuilds it a
few times around the current schedule with a shrinking margin and step. The
default plan has three levels: 4096 trials at margin 0.4 and step 0.2 with
every block size, 8192 trials at 0.3/0.1 with blocks of at most 32 stages,
and 8192 trials at 0.2/0.05 with blocks of at most 8 (in `"free"` mode when a
flat or interior start is chosen, one wider coarse level at 0.8/0.4 is
prepended so early stages can drop below the flat start without hitting the
margin wall). A level runs a second time, on a new genealogy, when its first
pass lowered the objective by more than 25 % (5 % for flat/interior starts in
`"free"`); `descent_passes` replaces the plan by a list of passes that each run
once.

**The start.** The schedules of long branching patterns share a shape in
units of the unpruned signal score, z_s = (mean_s - γ_s) / σ_s: loose at the
very first stages (z about 2), at the mean signal score within the first few
percent of the stages (z about 0), then rising steadily to z about 1.7 at the
end. The descent builds that shape and a flat-z candidate at the target,
evaluates both on one noise sample (4096 trials, the same for all) and starts
from the cheaper one. In `"free"` mode it also tests one upward step above the
target; if L drops there, a bracketing and bisection search over Pd up to 0.995
locates a lower-L starting Pd (deep high-cost trees reject the first upward
probe immediately). On 127-stage patterns the smooth shape is 1.1 to 6.8 times
cheaper than the flat start at targets up to 0.5 and is chosen there; on
patterns of 4-32 segments a higher-Pd start is chosen under `"free"` when L
falls above `min_pd`. The shape is only a start: the moves are free, and a
start that is far off makes the levels repeat.

After the descent the schedule is measured on samples that took no part in
it, as in `shared`: under `"pin"`, a common shift of either sign pins Pd to
`min_pd` + 2 binomial standard errors on a fresh bank of 2^17 signal trials;
under `"free"`, if the search Pd exceeds the target by more than
`max(0.15 * min(target, 1 - target), 6 * SE_search)`, a one-sided repair shifts
down only if Pd fell below the target (otherwise two-sided pinning applies). A
second fresh bank reports Pd (evaluated with stage-wise pruning of dead signal
trials above 128 bins on GPU and L2-tiled trajectories on CPU), and a fresh
noise evaluation reports C.

**The last threshold.** `final_threshold` is an optional floor on γ_{M-1};
`fap` derives one as the S/N whose one-sided normal tail is `fap` / (nbins ×
number of boxcar widths), the false-alarm probability of one fold with the
trials of its templates (`fap` 0: no floor; when both are set the higher one
holds). This is a per-fold probability: for a false-alarm budget of the whole
search, divide it by the number of folds that reach the last stage (`fap` 0.01
is only S/N 3.6-4.8 at 16-1024 bins, and a floor of 7 needs about 10^-9). The
descent is free above the floor. A block that holds the last stage moves down
with the last threshold stopping at the floor, and while γ_{M-1} rests on the
active floor (or a downward pin would take it below), γ_{M-1} stays locked on
the floor and the earlier stages [0, M-1) shift to keep Pd (KKT complementary
slackness); a floor below the free solution changes nothing. If the signal
cannot reach the target through the floor alone, the status is `short` and
`pd_ceiling` in the diagnostics is the Pd the floor allows.

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
step size lowers the objective on its samples, which is a first-order
(Kuhn-Tucker like) condition, not a proof of a global minimum. Empirically,
restarting from a solve's own z-scores four times on fresh samples across 12
patterns (16-512 bins, 2-5 parameters) reaches a best L 0.990 of the cold solve
(geometric mean; 0.971 worst, single restarts 0.971-1.074), and C scatters by
1.5 % median (4.5 % worst) across seeds at `min_pd` 0.1. A stronger reference
(2^17 signal and 2^15 noise trials, finer steps, nine solves, best judged C)
puts a cold `pin` solve at about 1.05 of one such restart and 1.07 of the best
of nine (1.04-1.13 over 14 patterns), and at equal Pd the grid modes are
2-8.5 % cheaper on cheap and mid-length patterns and 11-35 % costlier on
costly ones (see [performance.md](performance.md)). The descent's
C is near the lowest found, not the lowest.

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
| | `descent` (`pin`) | 0.101-0.104 | 4.68e4 | 0.020 | 0.16 s |
| | `descent` (`free`) | 0.101-0.104 | 4.66e4 | 0.023 | 0.21 s |
| production, `min_pd` 0.05 | `shared` | 0.051-0.056 | 1.29e4 | 0.129 | 0.23 s |
| | `robust` | 0.052-0.052 | 1.14e4 | 0.029 | 0.54 s |
| | `descent` (`pin`) | 0.051-0.053 | 1.07e4 | 0.030 | 0.17 s |
| | `descent` (`free`) | 0.051-0.053 | 1.07e4 | 0.018 | 0.20 s |

On short patterns (4-64 stages, 64 bins, 8 seeds) `descent` with
`pd_policy = "free"` delivers interior Pd > `min_pd` where L = C / Pd is lower,
matching or beating both `shared` (0.904-0.998 of its L) and `robust`
(0.911-0.998 of its L) in 0.027-0.100 s (see [performance.md](performance.md)).

On 127-stage patterns a cold `descent` (`pin`) solve takes 0.130 / 0.138 /
0.172 / 0.198 / 0.249 / 0.363 / 0.593 s at 16 / 32 / 64 / 128 / 256 / 512 /
1024 bins on the GPU with 4 host threads (faster than `shared` from 32 bins
upward; 0.10-0.15 / 0.18 / 0.28 / 0.50 s when chained across regions in the EP
planner), with no refit and no proxy bin count: warp-collaborative
shared-memory boxcar scoring eliminates local-memory prefix-sum spilling above
64 bins, periodic compaction drops dead signal trials in the pin and report
banks above 128 bins, and cache-blocked transposition + active-trial pinning
keep host moves at 71-92 ms. On the CPU with 4 threads it takes 1.8 / 1.7 /
3.2 / 5.7 / 10.3 / 20.2 / 36.4 s across 16 .. 1024 bins using L2-tiled signal
trajectories (the H1 working set is a tile of about 3 MB per thread; a whole
CPU design grows host memory by 41 / 115 / 437 MB at 64 / 256 / 1024 bins).

Across 1,344 `pin` solves spanning `min_pd` from 0.01 to 0.99 (21 patterns from
8 to 127 stages and 16 to 1024 bins × 8 seeds, judged on 2^17 fresh signal
trials), zero solves fell more than 3 binomial standard errors below `min_pd`
and zero delivered `short`; every judged Pd stayed inside `target + 2 SE ± 5 SE`
(for `min_pd = 0.1`, 0.0976-0.1038 across all 168 solves).

## Settings

`SharedOptions` holds the settings of the shared modes: `min_pd`,
`w_terminal` (1), `beam_sigma` (3) and `reach_sigma` (1). The constructor's
`nprobs` is the number of Pd bins from `min_pd` to 1; `ntrials` (1024) is the
noise population per cell. `prob_min`, `guess`, `beam_width` and
`thres_neigh` apply to `legacy` and `improved` only.

Mode `descent` reads `min_pd`, `w_terminal`, `pd_policy` (`"pin"` default, or
`"free"`), `fap`, `final_threshold`, `descent_passes` (builds of the noise
genealogy: trials, margin below the current schedule, step, largest and
smallest block), `descent_nsignal` (2^15; 2^14 costs 2-7 % in C),
`descent_start` and `descent_start_zscores`. `run()` does nothing in this mode
and the constructor's `ntrials`, `nprobs`, `nthresholds` and beam settings are
not used. The constructor checks these settings and throws
`std::invalid_argument` (`ValueError` in Python) for any that describe no
descent: `min_pd` outside (0, 1); `fap` outside [0, 1); a negative or
non-finite `final_threshold`; `descent_nsignal` outside [1, 2^20]; a pass
whose `ntrials` is outside [1, 2^20], whose margin or step is not finite and
positive, whose step is not below its margin (a down move would have no room)
or whose block sizes are not 1 <= smallest <= largest; a `descent_start`
without exactly one finite value per stage. Other modes do not read the
descent settings and do not check them. A given start must be a plausible
schedule: one far below the noise sits on a plateau where no block move
changes the cost (measured: a start at z-score +50, i.e. 50 standard
deviations below the mean signal score, made no move and cost 32 times the
default solve, reported `met`).

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
`within_tolerance`, `short`, `no_path`). `within_tolerance` means the reported
Pd is below `min_pd` but at least 0.9 `min_pd`. In `descent` the pin aims 2
standard errors above `min_pd` on one bank and the report counts on another,
so about 10 % of solves report `within_tolerance` (169 of 1,584 measured, all
with an independently judged Pd in the normal range): it is sampling error,
not a failure.

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
`seed` and draws a fresh random one when it is unset. The EP planner does not
evaluate the chunk patterns separately: a chunk's Pd is its region design's
report (the branching changes neither Pd nor the pass rates), and the chunk is
sized from that design's noise evaluation.

## Reproducibility

Results depend on the seed, `ntrials`, `nprobs`, the sample sizes above and,
on the GPU, `batch_size`; on the CPU they do not depend on the thread count,
and `descent` gives bit-identical results with 1, 2, 4 or 8 host threads on
either backend. Each sample of a solve draws from `derive_seed(seed ^ salt)`
with a fixed salt per role (signal sample, pin bank, report bank, noise
genealogies, report noise), and the salts differ in their low bits, so two
seeds s and s ^ (salt_a ^ salt_b) share samples across roles (for `descent`,
e.g. seeds 4 and 7 swap the signal sample and the pin bank). Use widely
spaced seeds, not consecutive ones, for independent replications.
`bench/scripts/schemes/golden_solve.py record|check` keeps a bit-exact CUDA
regression golden of both shared modes (valid for one toolchain);
`golden_evaluate.py` does the same for `evaluate()`.

## Limitations

- No optimality guarantee: the search is approximate and the constants above
  are heuristics. The descent stops at a schedule no block move improves,
  which need not be the global minimum.
- `descent` has no search states: `save()`, `get_states()` and
  `get_best_path_thresholds()` raise in this mode, and there is no shortlist.
- `fap` and `final_threshold` are available from `DynamicThresholdScheme`'s API
  (`SharedOptions`); the EP planner leaves the last pruning threshold free and
  applies `snr_min` strictly to the post-`ascend` candidate scores.
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
