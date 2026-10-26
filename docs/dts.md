# Dynamic threshold scheme

`loki::detection::DynamicThresholdScheme` (Python
`libloki.thresholds.DynamicThresholdScheme`) designs the pruning thresholds of
the hierarchical (EP) search: one threshold per stage. This page describes the
problem, the four modes, the default pipeline (`shared`) and why it is built
the way it is. Code comments only describe the implementation.

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

The objective is to minimise the cost per detection L = C / Pd subject to
Pd ≥ `min_pd`. A Pd above `min_pd` is allowed when it lowers L (short patterns
often have their optimum there).

Neither C nor Pd has a closed form (correlated, maximum-over-templates random
walks over up to ~130 stages), so both are Monte Carlo estimates, and an
exhaustive search over the schedules is intractable. Every mode is a dynamic
programme over cells (stage, threshold, Pd bin) that keeps one candidate per
cell: an approximate, compressed-state search with no optimality guarantee.

## Modes

| mode | search | after the search | use |
|---|---|---|---|
| `legacy` | one simulation per transition, analytic beam | none | reference, paper figures |
| `improved` | one simulation per parent cell, analytic beam (or `guess`) | none (`calibrate()` optional) | reference, paper figures |
| `shared` (default) | shared signal trials, guided beam, one pass | `solve()`: tighten, repair, report | production |
| `robust` | as `shared`, plus a second pass | `solve()`: shortlist, block refinement, repair, report | when ~2.5x the run time buys a lower L |

The EP planner uses `shared` on both backends, once per region; every chunk
of the region runs that schedule (see
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
  g_s = min(median signal score, trials-aware S/N). Stage s searches the grid
  thresholds within `beam_sigma` σ_s of g_s (inside the window), and a
  transition may change the offset γ_s - g_s by at most `reach_sigma` σ_s.
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

## Measured behaviour

Production pattern (127 stages), nbins 64, S/N 10, `min_pd` 0.1, CUDA on an
L40S. Each schedule is evaluated with `evaluate_scheme` (65536 fresh trials);
mean and coefficient of variation over 8 search seeds:

| mode | Pd | cv | C | cv | L | cv | time |
|---|---|---|---|---|---|---|---|
| `improved` | 0.088 | 0.068 | 4.9e4 | 0.18 | 5.5e5 | 0.12 | 0.16 s |
| `shared` | 0.104 | 0.043 | 5.6e4 | 0.11 | 5.4e5 | 0.074 | 0.22 s |
| `robust` | 0.103 | 0.023 | 5.1e4 | 0.041 | 4.9e5 | 0.039 | 0.60 s |

On the reduced acceptance suite (`bench/scripts/schemes/solve_suite.py`:
production pattern and the 4-32 segment patterns of issue #15, several S/N and
targets) both shared modes pass every gate: every seed's measured Pd
≥ 0.9 `min_pd`, the median ≥ 0.98 `min_pd`, and the reported Pd within 10 % of
an independent count (observed: within 2 %). The CPU agrees with CUDA within
the seed scatter and takes 15 s (`shared`) and 69 s (`robust`) on the
production pattern with 16 threads.

Evaluating with 1024 trials (the default of `evaluate_scheme`) adds Monte
Carlo noise of the same order as the differences between the modes (about
30 % on C at this pattern): compare modes with ≥ 2^16 trials.

## Settings

`SharedOptions` holds the settings of the shared modes: `min_pd`,
`w_terminal` (1), `beam_sigma` (3) and `reach_sigma` (1). The constructor's
`nprobs` is the number of Pd bins from `min_pd` to 1; `ntrials` (1024) is the
noise population per cell. `prob_min`, `guess`, `beam_width` and
`thres_neigh` apply to `legacy` and `improved` only.

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
evaluation on the CPU with a fresh random seed per call. The EP planner
evaluates the chunk branching patterns with `evaluate()` on its own backend.

## Reproducibility

Results depend on the seed, `ntrials`, `nprobs`, the sample sizes above and,
on the GPU, `batch_size`; on the CPU they do not depend on the thread count.
`bench/scripts/schemes/golden_solve.py record|check` keeps a bit-exact CUDA
regression golden of both shared modes (valid for one toolchain);
`golden_evaluate.py` does the same for `evaluate()`.

## Limitations

- No optimality guarantee: the search is approximate and the constants above
  are heuristics.
- The Pd guarantee is statistical (target `min_pd` + 2 standard errors of a
  2^17-trial count) and holds for the simulated signal model: a pulse of the
  reference shape at the design S/N, folded without loss. Mismatch loss of the
  EP search and multi-leaf detection are outside it.
- Pd is survival to the final candidate list, not completeness at a
  false-alarm rate. The last threshold is chosen by cost alone (`w_terminal`
  weighs the final survivors); nothing bounds the false-alarm probability per
  final trial. At low S/N a high Pd can need a final list that keeps about
  10^-5 of the noise, so the next processing step must be cheap enough for it.
- The reported C is an independent but noisy Monte Carlo estimate; on very
  large irregular trees it can be off by a large factor.
