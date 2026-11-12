# Performance work: rules, what was tried, what not to retry

This page is the memory of Loki's optimisation cycles. Read it before
starting one. It records the contract every performance change must meet,
the engineering rules that came out of past cycles, the ideas that were
measured and rejected (with the number that rejected them), and the items
that were deferred with a reason. An idea listed under "do not retry" needs a
new argument or a new measurement before it is tried again, not a new
implementation.

Benchmark and validation scripts live in `bench/scripts/` (see its README for
the FFA and EP workloads and the comparison tools).

## Numerics contract

- Freeze golden runs with injected pulses before a cycle. The harness
  (`bench/scripts/golden_runs/`: `run_golden.sh` runs seeded FFA and EP
  searches on cpu and cuda, `compare_h5.py` compares two output trees file
  by file) does not overflow the EP world tree, so a change to the trimming
  path needs its own check.
- A change that keeps the arithmetic (layout, fusion with identical
  operations, host work, memory model) **must stay bit-exact**, proven by a
  unit test.
- A change that moves floats (phases, FFTs, summation order) is acceptable
  when: the top-N candidates are identical; the Jaccard overlap at the
  production threshold is at least 99.9 %; matched scores agree within 1e-5
  relative (Fourier: within the CPU–GPU spread); an injected pulse's S/N moves
  by under 0.1 %.
- A change that moves chunk widths or parameter trials needs an injection
  study at 2^25 samples.
- Time-domain GPU outputs jitter run to run (float atomics in the brute
  fold): compare against the old-vs-old jitter, not against zero.
- Kernel math, RNG streams and output formats never change as a side effect
  of a layout or API change.

## Engineering rules

- Portable gains only: work reduction and algorithmic or design fixes. No
  per-device tuning of block sizes, unrolls or carveouts.
- Prefer explicit kernel loops over abstraction unless a gain is measured.
  The CPU Chebyshev transform structs and the fused circular
  resolve-and-transform were reverted for this reason.
- SIMD intrinsics shared across kernels live in `lib/cpu/simd_utils.hpp`.
- GPU runs get one CPU core (`nthreads = 1`). Fix host bottlenecks by doing
  less work or removing synchronisations, not by adding host threads.
- Measure in `Release` only (`cmake --preset release`, `ctest --preset
  release`, `PYTHONPATH=src pytest -n auto tests/python`). Debug builds are
  far too slow for the test suites.
- CUDA and x86 SIMD checks run on the cluster, never on GitHub Actions.
- Every time budget needs a split: when the production bench reports one
  total, a 50 s host stage can hide behind a 0.4 s GPU sweep (see "Candidate
  clustering" below).

## Do not retry (measured, no gain, or rejected by design)

### EP planner and segment layout

| Idea | Why not |
|---|---|
| Sub-octave chunk splitting, cost-model driven | Each EP chunk pays a fixed FFA initialisation, threshold and tree setup, and `2 · max_drift · f` of frequency overlap at every boundary. Filling one chunk to `max_process_memory_gb` beats splitting octaves. |
| Padding a non-power-of-two `nsamps` to a power-of-two segment count | Up to 2× FFA cost (2^23 + 1 samples became 2^24). Replaced by equal-length segments (`docs/segments.md`). |
| An uneven last EP segment (longer or shorter tail) | Breaks the equal-signal assumption the calibrated thresholds rest on. EP truncates to whole segments instead (fewer than 1 / `bseg_brute` of the series; S/N cost about half that). |

### EP on CUDA

| Idea | Result |
|---|---|
| Larger cuFFT batches | No gain |
| Shift-add coefficient table | DRAM-bound, no gain |
| Dual write of folds | No gain |
| Register accumulation in the ascend step | No gain |
| Thrust → CUB in the world tree | 13 ms, not worth the code |
| Batch size 2^17 | No gain |
| One-batch branch lookahead | No gain |
| FP64 removal alone | No gain |
| Ping-pong world tree | No gain |
| One warp per parent's children | No gain |
| Occupancy tuning | Rejected by the portability rule |
| Writing 3-D `folds_out` for non-direct batches | At production scale (2^25, jerk, 16 runs) all 2219 batches were direct and none non-direct |
| Hoisting `PruneCUDAImpl` across chunks | Constructor 0.4–1.4 ms per chunk against 90–330 ms of pruning; it also depends on the chunk's grid and steps, so a hoist needs a re-initialise API |

### EP on CPU

| Idea | Result |
|---|---|
| Non-temporal survivor stores | No gain |
| Software prefetch | No gain |
| AVX-512 16-column transpose in `BoxcarScorer3D::score_group` | 1–4 % **slower** than 8×8 blocks on Zen 4 (nbins 64–256, bit-identical, repeatable). Removed. Intel AVX-512 was not measured. `simd_utils::transpose16x16` stays for the lane FFT. |
| Reusing member index vectors in `PruneCircTaylorDPFuncts` | Not worth it |

### Host FFT

| Idea | Result |
|---|---|
| pocketfft → ducc0 | Port worked (427/427 tests with `ducc0_0_42_0`) but measured 0.6× on 16384 × 16–64 rows and about 0.5× on repeated prime lengths on ARM. Reverted. Only an x86 benchmark (Zen 4 AVX-512 and AVX2) can reopen it, and only if a profile shows pocketfft as a measurable share of CPU time. If retried: never define `DUCC0_NO_LOWLEVEL_THREADING`; call `ducc0::resize_thread_pool(1)` once, outside OpenMP. |

## Kept after measurement

- `BranchOffsetOutIter` (CUDA): an end-to-end A/B is below noise (2^25 jerk
  sweep 0.37–0.38 s), but by count it removes two device-to-host copies per
  branch call, about 9–13 ms (2–3 %) of the sweep.
- Pre-sized CUDA brute-fold tables: the tables grew lazily inside the chunk
  loop (a `cudaMalloc`, an implicit sync, and thrust holding old plus new
  storage). `ep_cuda_brute_table_sizes` is now the single source for the
  memory model and the allocation.

## Deferred, with the reason

| Item | Why deferred |
|---|---|
| Device-driven GPU batches (persistent kernel or CUDA graphs) | Large change for 0.2–0.3 s of a 1.5 s prune window; the whole GPU sweep is 0.38 s at 2^25 |
| Exact K-th trimming on world-tree overflow | Science change on overflow; needs a recovery study |
| FP16 / BF16 tree folds | Science change; needs a recovery study |
| CPU intra-run threading | Workers are `min(nthreads, n_runs)`; with 16 runs a large node idles in the pruning stage. Medium effort, exact numerics required. Measure the pruning share of a CPU production run first. |
| S/N lost to grid snapping in the FFA tree (issue #9, issue #39, "long brute segments bias accelerated signals by about 2.5 bins") | One mechanism, three reports. Needs an injection and refold harness before any kernel-math change; the 2.5-bin figure has no recorded reproduction. |

## Measured production profile (2026-11, L40, 2^25 samples, jerk, 16 runs)

| Stage | `main` (`b6ca7ef`) | `ep_pipeline` with the fix (`fdbb20e`) |
|---|---|---|
| GPU EP sweep | 0.37 s | 0.38 s |
| Candidate post-processing (14,685 detections, 197 clusters) | about 50.7 s | 2.21 s |
| Whole run (`loki search ep`, `ep_production_cuda.toml`) | 51.1 s | 3.2 s |

Measured 2026-11-09 on an EDGAR L40 node (AMD host, `nthreads = 1`), CLI
pair from `bench/scripts/ep/ep_production_cuda.py`'s docstring. The
candidate stage is the `Candidate postprocessing complete in ... s` log
line; before the fix it is the wall time less the sweep. Detections,
clusters and candidates are bit-identical between the two builds. The
candidate stage is still about 70 % of the run; clustering steps that move
floats (a 2-D index, closed-form roots) are the next option if that
matters. On the stress fixture (4965 trials of one source,
`tests/cpp/api/candidates_stress_t.cpp`) the clustering went from 1.40 s to
0.043 s on an M-series laptop, bit-identical.

`ep_production_cuda.py` reports the split (`sweep_s`, `post_sweep_s`) and
runs the candidate stage with the CLI's default `[candidates]`. It feeds
raw samples where the CLI preprocesses (robust normalisation), so its
detection counts differ from the CLI's (15,142 detections, 225 clusters;
candidate stage 2.29 s of a 2.8 s run); the best leaf has the same
parameters (score 11.51 against the CLI's 11.48).

### Candidate clustering

Clustering is host work after the search. With one bright source seen in
14,685 trials, the frequency-window prefilter of the transitive clusterer
selects nearly every node, and every pair not yet in one component ran the
full phase-span computation: a recursive root finder with a heap allocation
at every recursion level and a 64-step bisection per sign change. The fix
(bit-exact, see `lib/search/candidates.cpp`): fixed-size stack storage for
the polynomials, and a lower bound on the span from the endpoint values that
rejects most pairs before any root is found. The lower bound never changes a
link decision. `tests/cpp/api/candidates_stress_t.cpp` checks the result
against references computed in the same process (pairwise closure, anchored
assignment, phase extents). The bit-identity to the previous implementation
was verified by fingerprint when the fix was made; that pattern is platform-
specific (it differs on a no-FMA clang build), so it is not asserted in CI.

If cluster counts grow into the thousands, the harmonic flagging loop in
`CandidateClusterer::finish` (clusters² × `harmonic_denominator_max`, with a
span computation inside) is the next cliff; the same two fixes apply.

### Per-run reduction of EP final leaves

A bright source, or interference, leaves a large EP search with millions
of final leaves above `snr_min`, nearly all of them responses to one pulse.
Before the reduction every leaf was written to the spool, read back, sorted
and grouped, and the run groups were copied into the result, so the
candidate stage could write as much as the search. The reduction
(`docs/candidates.md`, "Keeping the strongest trials per frequency cell")
keeps the strongest leaves of each frequency cell of each run, absorbs
leaves a stronger leaf of the run explains, and runs as each run is written.
Measured 2026-11-12 on an EDGAR L40 node (`--threads 4`, CUDA):

| Search | Final leaves | Before | After |
|---|---|---|---|
| Circular orbit, bright pulsar, 2^26 samples, 32 runs | 83.5 M | 10 GB spool; candidate stage killed after more than 10 min (reading and sorting 15 of 32 runs took 393 s) | 1.5 MB spool and result; candidate stage 0.26 s; whole run 67 s; peak RSS 1.13 GB |
| Acceleration, 2^27 samples, 128 runs (publish only) | 0.73 M above `snr_min` | did not finish in 30 min | 1.3 s, 0.10 GB, 6,823 detections |
| Jerk, 2^26 samples, 224 runs (publish only) | 0.15 M above `snr_min` | | 0.4 s, 2,874 detections |

Every known pulsar in these observations kept its best S/N and every run
that detected it; the reduction with `max_per_cell = 0` is bit-identical
to the unreduced product, and a search that reduces as it writes gives the
same product as reducing its full spool at publication (CPU, exact).

Measured and rejected: pooling every run's leaves into the same frequency
cells (as first proposed). The strongest runs of a source fill its cell, so
its detections in weaker runs are dropped: with 32 leaves per cell, sources
lost many of the runs that detected them (for example 21 to 12, and
16 to 7), although every best S/N survived. Cells are per run.

### Spectral zap modes

`zap_mode = "noise"` (`docs/preprocessing.md`) replaces every birdie-window
bin with Gaussian noise. With masked samples (gaps), one removal leaves
about the masked fraction of a line in the valid samples: a bright binary
pulsar at S/N above 100 still folded to S/N 9.5 after a single pass,
2.5 % of the series masked. The hard modes therefore repeat the removal
three times (re-zeroing the gaps in between) when at least 1e-4 of the
samples are gaps, and take the fill level from the cleaned spectrum.
Measured 2026-11-12 on an EDGAR node, 4 threads, 1,219 birdie windows
(2.2 M bins), whole preprocessing:

| Series | No zap | `phase_kept` | `noise` / `zero` |
|---|---|---|---|
| 2^27 samples, 40 masked bursts (3 removal passes) | 3.9 s | 13.5 s | 24.0 s |
| 2^27 samples, no gaps (1 pass) | 4.0 s | 13.0 s | 12.8 s |

Each further pass costs one FFT pair, about 2 s at 2^26 and 5.6 s at
2^27. On the bright-binary series, 1, 2, 3, 5 and 8 passes left the
pulsar's best S/N at 9.5, 7.2, 7.1, 7.0 and 7.0, against 6.7 for the same
search on an externally hard-zapped copy; the strongest unrelated
candidate in that search is 7.4. Three passes are kept. The removal is
alternating projection: it converges fast while gap length times window
width is well below 1, and slowly for long gaps in wide windows (a
least-squares solve would converge faster, and is deferred until a series
needs it).

Zeroing the gaps after the fill also removes part of the fill, and the
removal passes strip the leakage that genuine gapped noise would put into
a window. Uncorrected, the zapped bins came out at about (1 - f)^2 of
their neighbours' power for short gaps (0.72 at f = 0.15, gaps of 0.2 s);
a flat 1 / (1 - f) correction gave 0.85. The fill is now raised by the
retention that the mask's spectrum predicts for each window: zapped over
other power is 0.95-1.02 for f up to 0.15, and 0.995-1.011 above 10 Hz
on the bright-binary series. `phase_kept` (the previous zap) is
bit-identical to before.
