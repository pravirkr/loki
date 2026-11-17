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
| Sub-octave chunk splitting, cost-model driven | Each EP chunk pays a fixed FFA initialisation, threshold and tree setup, and `2 · max_drift · f` of frequency overlap at every boundary. Filling one chunk to `max_process_memory_gb` beats splitting octaves. Seed slices are a different lever: they split a chunk's tree, not its FFA, and add no overlap ("Seed slices of a run"). |
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
| Per-chunk CUDA batch ladder (halve only the batch of a chunk whose minimum width does not fit; the planner halves one batch cap for every chunk) | Plans that fit were unchanged, and a circular search that the global ladder took to batch 8 kept 51 of 54 chunks at their default batch. But a lowered chunk stops at the first batch at which its minimum width fits, so it is barely wider than the minimum: a full-band jerk search (2^26 samples, 1.5 GB) planned 191 chunks at batches 4096-32768 against 98 at 4096, and pruned in 37.6 s against 30.0 s (wall 46.1 against 35.8 s); a tighter circular plan had 531 chunks against 110 and planned in 1297 s against 297 s. Narrow chunks repeat the per-chunk setup and the `2 · max_drift · f` overlap. A fix needs the batch and the width chosen together from the search cost (deferred) |

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

- Compressed per-run tables in result files: on an EP acceleration search
  of 2^24 samples (8 chunks of 32 runs, CUDA), `level_stats` (one row per
  pruning level, mostly small integers) was 2.9 MB of a 5.4 MB result, and
  the HDF5 headers of about 2,100 small objects another 1.2 MB. Deflate with
  byte shuffle shrinks `level_stats` and `snail_scheme` about sevenfold. A
  compressed dataset needs a chunk index, which in the oldest HDF5 file
  format cost more metadata than compression saved; written in the 1.10
  format (one chunk, no index B-tree) the result is 2.3 MB (-58 %), and the
  `--no-candidates` file 26.2 instead of 29.1 MB. Writing the 512 tables
  costs about 66 ms more (104 to 170 ms); the search took 34 s. Values are
  identical.

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

Harmonic flagging (`CandidateClusterer::flag_harmonic_families`) keeps the
roots in a frequency-ordered index and looks up, for each representative and
each reduced ratio `p/q` (simplest first), only the roots whose frequency
can pass: the Doppler window, or the track window from Markov's inequality (a
degree-n phase polynomial with p(0) = 0 spans at least |p'(0)| / n^2), so
the phase-track span is computed for few pairs. Measured 2026-11-14 on
synthetic unrelated representatives over 5-720 Hz (T = 5000 s, orders up to
16, CPU): the pairwise loop over every root and every q took 10.5 s for 10^4
(6.2 s with the span's early rejection, 3.9 s with the Markov bound before
it); the index takes 0.19 s for 10^4 and 4.2 s for 10^5, with the same
flags.

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

`loki publish` runs the same candidate stage as the search. Measured
2026-11-14 on an EDGAR L40 node (bright pulsar, acceleration, 2^23 samples,
32 runs, CUDA, 0.34 M final leaves, 0.30 M above `snr_min`): publishing the
full `--no-candidates` file (26 MB, written by a 2.2 s search) took 0.18 s
with the EP defaults, the 112 candidates of the search that reduced as it
wrote. With `max_per_cell = 0` it took 79 s at `memory_gb = 4` against
65 s for the same stage inside the search (465 candidates both ways), and
both fail at `memory_gb = 1`. The cost is in grouping every leaf, not in
publishing a spool.

Measured and rejected: pooling every run's leaves into the same frequency
cells (as first proposed). The strongest runs of a source fill its cell, so
its detections in weaker runs are dropped: with 32 leaves per cell, sources
lost many of the runs that detected them (for example 21 to 12, and
16 to 7), although every best S/N survived. Cells are per run.

### Region growth factor (`region_growth_factor`)

The planners split the period range into regions whose period grows by
`region_growth_factor`; nbins and eta grow with the period, so the grid's
phase tolerance, `eta / nbins` turns at f_max, is the same in every region
(the duty-cycle resolution of the search, written as `phase_tolerance`). The
factor only sets how far a region's bins widen in time from its fast end to
its slow end. Full-band jerk search (10-720 Hz, 2^26 samples, nbins 16,
eta 1, 32 runs, CUDA), 28 injected jerk pulsars per series:

| Factor | Regions | S/N vs 2.0, duty 1/16 (48 below 600 Hz) | Search | Plan CUDA / CPU |
|---|---|---|---|---|
| 2.0 | 7 | - | 5.6 s | 18 s / 313 s |
| 1.5 | 11 | +2.5 +- 1.9 % | 8.3 s | 26 s / 452 s |
| 1.2 | 24 | +4.2 +- 1.8 % | 11.7 s | 56 s / 1080 s |

A single source's S/N moves by about 12 % (rms) between factors, even
where nbins is the same: the chunk extent changes the seed grid and the
thresholds. Pulses narrower than the phase tolerance are lost at every
factor (a 0.25 ms pulse below 100 Hz); lower eta, or split the band into
searches with their own nbins, to search narrower duty cycles. Default 1.5.

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

### RFI control of the EP pruning (`[ep.rfi]`, CPU)

A 2^24-sample series with a bright pulsar (S/N about 275) and four
injected accelerated pulsars (final S/N 12-16), 60-130 Hz,
|a| <= 400 m/s^2, 16 runs, 4 threads, 16 GB (max_sugg 2^18.7, 2 chunks in
every case):

| Input | Overloads | Runs (of 16) keeping each injection | Final leaves | Search |
|---|---|---|---|---|
| Raw, no RFI control | 130, every run | 0 / 0 / 0 / 0 | 5.2 M | - |
| Raw, `harvest = true`, `harvest_min_snr = 15` | 0 | 14 / 11 / 12 / 12 | 78 k | 13.5 s |
| Bright pulsar hard-zapped, no RFI control | 0 | 13 / 10 / 12 / 13 | 75 k | - |

Harvesting with a floor of 15 removed the bright pulsar (388
harvests in 32 runs, at most 22 per run) and recovered the injections as
well as an external hard zap did. Planning (the CPU threshold design)
dominated wall time in these runs (40-60 s).

The harvest threshold `max(15, threshold + 10)` is about 16.5
at the late levels of a 128-segment jerk search, so an RFI line of final
S/N 12 is never harvested; `harvest_offset = 3` with
`harvest_min_snr = 9` harvested it once per run in 15 of 32 runs and
removed all of its detections, as a frequency mask did. Such a low
threshold also harvests real pulsars. On a survey observation with
several known binary pulsars, `harvest_offset = 6` harvested one of them
in 30 runs (final S/N 18.7 to 7.0), offsets 3-4 removed three others
completely, and even offset 10 harvested a bright one in one
run, while the observation's strongest line needed offset 3. Brightness
alone does not separate lines from pulsars, so the default offset stays
high. A harvest threshold below a source's partial S/N removes the
source: in a test, a weak injected pulsar was harvested along with the
bright ones.

Harvest windows and fractional harmonics, on a survey segment (2^24
samples, 60-130 Hz, |a| <= 35 m/s^2, 32 runs) with a bright binary pulsar
(final S/N 111) whose 3/2 harmonic reached S/N 26:

| Harvest setting | Harvests | Bright pulsar left (runs) | 3/2 left (runs) | Detections |
|---|---|---|---|---|
| none | 0 | 30 | 30 | 463 |
| floor 15, window sweep from the leaf's acceleration | 66 | 1 | 0 (harvested itself) | 307 |
| floor 15, window sweep from the window's largest acceleration | 63 | 0 | 0 (harvested itself) | 306 |
| as above, `n_harmonics = 4`, `fractional_harmonics` | 33 | 0 | 0 (masked) | 208 |

The pulsar was harvested at level 10, where its acceleration was not yet
resolved (a = 0 with a tile of 70 m/s^2), so a sweep from the leaf's own
acceleration left the window 6 grid cells wide. Masks are per chunk: the
pulsar's 3/4 harmonic (S/N 15) lay in another chunk and stayed.

#### On CUDA

The pulsar mask and the early harvest run on CUDA (the impulsive veto is
CPU only). On the bright-pulsar series above (2^24 samples, 60-130 Hz,
|a| <= 400 m/s^2, 16 runs per chunk, harvest floor 40):

| Backend | Overloads | Runs (of 16) keeping each injection | Harvests | Harvested sources | Search |
|---|---|---|---|---|---|
| CUDA, no harvest | 23 in 15 of 32 runs | 2 / 0 / 1 / 1 | 0 | - | 7.3 s |
| CUDA, harvest | 0 | 13 / 10 / 9 / 11 | 98 | 7 | 1.4 s |
| CPU, harvest | 0 | 14 / 11 / 12 / 12 | 98 | 7 | 26.4 s |

With the bright pulsar hard-zapped instead, CUDA keeps the injections in
10 / 10 / 9 / 11 runs. The CUDA and CPU sources agree in frequency, S/N,
level and runs. The CUDA
harvest keeps the fused direct write of the passing leaves: only a batch
whose highest score reaches the harvest threshold does harvest work.
Rejected: a post-score pass on every batch from the first harvest level
(the CPU's structure) disables the direct write and cost 10 % of a
full-band jerk search (10.9 to 12.0 s) with nothing to harvest.

#### A pulsar mask over a zapped line

On a survey segment (2^26 samples, circular-orbit search at eta 2,
10-500 Hz, CUDA, 8 chunks at a 20 GB limit) with a strong instrumental
line and its harmonics zapped as `birdies` (0.15 Hz wide), 9 detections
(S/N <= 7.5) remained 0.01-0.07 Hz below the line at |a| = 585-720 m/s^2:
the birdie notches the unaccelerated spectrum, not the line's Doppler-drifted
copies. A `pulsar_mask` window of 0.15 Hz over the line (harvest off)
removed all 9 (6181 leaves masked) at the same search time (270 s). On
the same plan with the window moved out of band (0 leaves masked), the
search gave exactly those 9 detections more and every other detection
with the same frequency and S/N: the mask removes the line and nothing
else.

The mask is in the CUDA plan's memory (the bitset and seed list, about
0.3 MB here), so it moves the chunk edges of a plan at its limit. Across
four layouts of this search (20 or 30 GB, with and without the mask), a binary
pulsar in an unmasked chunk scored 8.5, 12.6, 13.9 and 16.6: at eta 2 its
S/N depends on where the chunk edges put the grid. Compare a mask's
effect within the chunks it touches, not by sources elsewhere. A mask-only
configuration no longer counts the harvest staging, which only harvesting
allocates (memory model 14).

#### CUDA batch and branch_max

A CUDA chunk's world tree holds its forecast live leaves plus one branched
batch, `batch * branch_max` slots, each `nbins` wide. With a fixed batch of
2^16 and `branch_max` floored at 32, that headroom (2^21 slots) dominated the
low-frequency chunks of a full-band jerk search (2^26 samples, 10-720 Hz,
30 GB, one chunk per region): the 10-11 Hz chunk (nbins 1024) needed
18.1 GB for a forecast of 3206 live leaves and set the plan's 19.8 GB peak.

`branch_max` is now the chunk's worst leaf, with no floor: 15 at the top,
9 in most regions and 3 at nbins 1024 on that search. The plan peak fell to
5.6 GB (that chunk 3.1 GB) with the same detections and search time, and no
run on any test series branched past it.

The batch is the work of one kernel round trip. Search time per chunk of the
same search against the batch (s):

| batch \ nbins | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 65536 | 1.76 | 0.95 | 0.52 | 0.40 | 0.36 | 0.43 | 1.16 |
| 16384 | 3.34 | 1.59 | 0.64 | 0.43 | 0.37 | 0.44 | 1.19 |
| 4096 | 9.04 | 3.74 | 1.18 | 0.57 | 0.43 | 0.48 | 1.14 |
| 1024 | 32.8 | 13.2 | 3.40 | 1.21 | 0.66 | 0.68 | 1.24 |
| 256 | 122 | 48.8 | 12.2 | 3.73 | 1.66 | 1.54 | 1.89 |

The cost follows `batch * nbins`: above about 2^22 a chunk runs at full
speed, below it the per-batch overhead dominates. A chunk is planned with
`2^22 / nbins` parents (at most 2^16): 4096 at nbins 1024, with no loss of
speed (1.12 s). The plan peak of the search above is 5.5 GB.

The headroom no longer absorbs a tree far above the noise forecast. On the
same search the 10-11 Hz chunk, which holds an instrumental line at about
ten times its forecast population, now trims (25 overloads, 101 of the
line's 2094 detections lost); every other chunk is unchanged. With the
harvest on, a series with a bright pulsar gave the same result as before
(0 overloads, the same injected pulsars recovered, a tree 2.8 times
smaller). Without it, bright sources overload more (a raw full-band jerk
search: 1332 to 1795 overloads). Bright sources and lines belong to the
preprocessing (zap, `birdies`), the pulsar mask and the harvest, not to
spare tree capacity.

Superseded: a batch from the forecast live tree (`bit_ceil(live)`, at
least 1024, with `branch_max` still 32) cut the same peak only to 11.7 GB:
the 11-22 Hz chunk's forecast (34130 leaves) kept it at 2^16.

Without the floor, the circular (crackle) forecast undercounted: it never
carried the crackle cell width past the first level, so it never counted the
crackle split that hole expansion applies. A circular search (2^26 samples,
eta 2, nbins 16, CUDA, batch 65536) of a 494.6-504 Hz slice planned
`branch_max` 9 and aborted at its first crackle split ("Branching a batch
needs 596502 leaves but the branch workspace holds 589824"). The forecast
now replays all five widths, as the kernels refine them (memory.md): the
slice plans `branch_max` 27 and 3.81 instead of 3.31 GB, and runs in 13.7 s.
In the kernels' replay of a test band (every frequency, every child in the
hole) the worst parent has exactly the forecast, 27 children against 9 for
the product without crackle, on the CPU and on CUDA. Full circular plans of
2^26 samples: 10-500 Hz at 30 GB (64 us samples) keeps its 13.9 GB peak (one
of six chunks 15 -> 27); the same search at 4 GB (81.92 us, the slice's
series) plans 42 chunks instead of 39; a 10-415 Hz search at eta 1 and 30 GB
(81.92 us) keeps its 29.5 GB peak (26 chunks instead of 27). The global batch ladder
went from 8 to 16 in both tight plans; the ladder is not monotone in the
chunk layout. The forecast carries the crackle width in its own loop, and
planning takes as long as before (the 4 GB plan: 202 against 198 s, run one
after the other). A first version replayed all five widths in a second pass:
it doubled the forecast and took 341 s. With `use_conservative_tile` the
forecast still runs twice, because the threshold pattern has always been
simulated without that tile and only the bound follows it.

#### Default harvest floor (`harvest_min_snr = 40`)

Harvesting is a failsafe that keeps a dominating source out of the tree,
not a detection step, so the floor must leave moderate sources alone. A
source of final S/N S has partial S/N of about S * sqrt(level / nsegments)
(measured: S/N 77 crossed a floor of 40 at levels 35-36 and a floor of 50
at 54-62, S/N 153 crossed 50 at 15-16, with 128 segments), so a source
whose final S/N is below the floor is never harvested. The pruning term
`threshold + harvest_offset` stays below 20, so the floor alone sets the
harvest threshold at every level.

Single injected pulsar on the zapped series above (16 runs per chunk):

| Floor | S/N 24 | S/N 39 | S/N 77 | S/N 153 (32 overloads without harvest) |
|---|---|---|---|---|
| 15 | removed | removed | removed | harvested early, 0 overloads |
| 30 | kept | cut to S/N 9 | harvested from level 20 | harvested from level 10, 0 overloads |
| 40 | kept | kept | harvested from level 35 | 0 overloads |
| 50 | kept | kept | harvested from level 54 | harvested from level 14, 0 overloads |
| 75 | kept | kept | kept | harvested from level 31, 0 overloads |

The raw series (bright pulsar, final S/N about 275) is harder: its
fractional harmonics (3/2 to 7/5, partial S/N 30-55) overload the tree
on their own.

| Floor | Overloads (runs) | With `n_harmonics = 5`, `fractional_harmonics` | Harvested sources | Search |
|---|---|---|---|---|
| none | 130 (32) | - | - | 44.8 s |
| 15 | 0 | 0 | 22 | 7.5 s |
| 30 | 0 | 0 | 11 | 15.3 s |
| 40 | 0 | 0 | 7 | 28.0 s |
| 45 | 9 (8) | 4 (3) | 6 | 31.5 s |
| 50 | 15 (13) | 5 (3) | 6 | 36.8 s |
| 75 | 73 (32) | 30 (16) | 4 | 47.9 s |

The weak injections kept 14 / 11 / 12 / 12 runs at every floor up to 50
(14 / 9 / 11 / 11 at 75). Every harvested source at floors 30-50 was the
bright pulsar or one of its harmonics. The overloads left at 45-50 with
fractional masking are in the other chunk (60-65 Hz), which holds the 3/4
and 5/7 harmonics but not the pulsar; per-chunk masks cannot reach it.
Floor 40 is the highest floor without overloads here and keeps a S/N-39
source whole, so it is the default; lower it only after measuring what a
lower floor removes. Each harvest shrinks the tree, so a higher floor costs
search time (28 s at 40 against 7.5 s at 15; 21 s at 40 with fractional
masking).

#### Seed slices of a run

A run can prune its seeds in K sequential frequency slices over the same
FFA output (`EPMultiPassEngine::set_seed_slices`): the tree then holds one
slice's subtrees at a time. This is not the rejected sub-octave splitting:
the chunk, its FFA output and its Doppler padding stay as they are, so no
frequency overlap is added. On the CPU the result is exact: on a real
2^26-sample circular Fourier series (4 runs, 128 levels, up to 4.0 M
survivors per level) K = 8 and K = 1 gave the same final leaves and level
stats bit for bit, in 27.7 s against 27.2 s.

One chunk of a circular Fourier search on CUDA (L40, 2^26 samples, nbins 16,
32 runs, 3062 frequency x 3 acceleration cells, 127 levels), K even slices:

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 3062 |
|---|---|---|---|---|---|---|---|---|
| Sweep (s) | 13.68 | 13.80 | 14.12 | 14.69 | 16.11 | 19.26 | 26.77 | 583.9 |
| Peak live leaves, worst run | 5.45 M | 2.72 M | 1.47 M | 0.91 M | 0.62 M | 0.45 M | 0.33 M | 23 k |

- Each slice costs about 5.8 ms per run, nearly all of it one round of
  kernel launches and synchronisations per level, so the overhead grows as
  K times the level count. Use the fewest slices that fit.
- The tree shrinks about K-fold up to K = 8 (worst run 5.9-fold, median
  6.7-fold). Beyond that a few seeds grow most of the tree: at K = 64 the
  median run shrinks 36-fold, the worst only 16-fold.
- The final leaves were bit-identical for K = 1, 2 and 8. Intermediate
  level counts differed by single leaves at a threshold (one survivor in
  1.3 M), which then died: the CUDA Fourier pruning is not bit-reproducible
  near thresholds (see the numerics in docs/architecture.md), and slicing,
  which changes how leaves share batches, makes that more visible.

The planner now chooses slices (docs/memory.md, "Seed slices"): a chunk that
does not fit takes the fewest slices that fit, before the width search
narrows it; on CUDA a smaller batch replaces slices only where the search-time
model says it is cheaper. On the same chunk (CUDA, limit set with
`--memory-gb`, search with `--no-candidates`):

| Limit | Before: chunks, CUDA batch | After: chunks, slices, CUDA batch | Search after |
|---|---|---|---|
| 30 GB | 1, 65536 | 1, 1, 65536 | 14.7 s |
| 3 GB | 3, 32768 | 1, 3, 65536 | 14.7 s |
| 2 GB | 9, 64 | 1, 6, 32768 | 15.5 s |
| 1.5 GB | does not fit | 2, up to 13, 16384 | 32.2 s |

At 3 GB and 2 GB the 1263 final leaves of the 32 runs equal those of the
30 GB plan (no slices) exactly; no stage of any run overflowed its tree.
On the CPU (4 threads, 4 runs of the same chunk) the plan at 2.4 GB took 13
slices and at 2.2 GB 17; both gave the final leaves and every level count of
the 30 GB plan exactly, with no overflow, at a peak RSS of 1.83 and 1.62 GB
(12.15 GB with one slice).

The batch costs search time only once it is small. The same chunk with one
slice took 13.6, 13.3, 19.6 and 32.2 s at batches 65536, 32768, 16384 and
8192: about 75 us per batch iteration of one level once launches dominate,
on top of about 1.5 ns per leaf. A slice costs about 46 us per level per run
(its iterations are small). The planner's model (`kCudaIterationSeconds`,
`kCudaLeafSeconds` in ep_chunking.cpp) uses 50 us per iteration and 1 ns per
forecast leaf, as the forecasts run about 1.65 times the measured
populations. It only ranks plans: at 2 GB it chose 1 chunk with 6 slices at
batch 32768 over 2 chunks with up to 64 slices at 65536.

Whole searches, plan only (2^26 samples, CUDA, 32 runs):

| Search | Before | After |
|---|---|---|
| circular, 10-720 Hz, 30 GB | does not fit (2 s) | 20 chunks, up to 32 slices, batch 32768; 179 s |
| circular, 10-415 Hz, 30 GB | 26 chunks, batch 16; 292 s | 9 chunks, up to 8 slices, batch 65536; 12 s |
| circular, 4 GB | 42 chunks, batch 16; 222 s | 24 chunks, up to 5 slices, batch 65536; 57 s |
| jerk, 10-720 Hz, 30 GB | 7 chunks | the same plan |

Chunk width against slices: capping the nominal chunk width (a probe build)
on the same 495 Hz search at 2 GB gave 1, 2, 4, 8 and 16 chunks with up to 6,
4, 3, 2 and 2 slices, searched in 14.5, 19.3, 28.5, 46.9 and 84.9 s. On a
700 Hz band (2 runs, 8 GB) 2 chunks of up to 48 slices took 75.6 s and 4 of
up to 41 slices 137.2 s: a narrow chunk still seeds its whole Doppler
padding, so it saves few slices and prunes the padding again. The widest
chunk that fits was the fastest in every case, and the planner's search-time
forecast ranked all seven plans in the measured order (the measured times are
1.7 to 1.9 times the forecast, the same factor for every plan of a search).

The CUDA batch costs more at high frequency: the 700 Hz plan above (2 chunks,
up to 48 slices) took 87.7 s at batch 32768 against 75.3 s at 65536, with the
same final leaves (65536 to 32768 cost nothing at 495 Hz).

Group `branch_max` replan (docs/memory.md, Width search): the 10-720 Hz
circular search at 30 GB planned 16 chunks at the full batch in 38 s, where
one minimum chunk had halved the batch of 14 of 20 chunks (179 s). The
10-500 Hz circular search at 4 GB went from 24 to 15 chunks (planning 57 s
to 13 s); the 32-run search took 310 s against 320 s. The same search on the
CPU now plans at 2.2 and 2.4 GB, where a minimum chunk failed by 0.01 GB.

Region leftovers: the 720 Hz search planned an 82.9 Hz chunk and a 1.44 Hz
sliver at the bottom of one region (nbins 16); split evenly they are two
42.2 Hz chunks (12 + 1 slices became 9 + 5), with the same peak. Searching
just those two chunks (2 runs) took 87.6 s as a chunk plus sliver and 88.7 s
split: the split removes a near-empty chunk, not search time.

Leftover split with a lower half that does not fit: the split placed the
upper half before fitting the lower one and then went on from the midpoint,
so a circular 10-500 Hz search on the CPU at 2.2 GB planned its last region
(nbins 512) as 6 chunks of halving width. Fitting the lower half before
placing either keeps the chunk the width search found: 3 chunks there, 51
instead of 54 in all, peak 1.35 against 1.34 GB. The other plans checked
(circular 10-720 Hz at 30 GB, 10-500 Hz at 4 and 30 GB on CUDA, jerk 10-720
Hz) did not change.

Seeding per run: each seed slice recomputed every seed of its chunk. Seeding
once per run gave the same final leaves in all 32 runs of the 495 Hz chunk
above (CUDA, 6 and 13 slices; only the leaf order within a run differs, as it
does between two runs of the same binary) and the same search time within
noise (32.4 and 32.5 s against 32.7 and 32.4 s at 13 slices): seeding is
negligible at a few slices and matters only near one slice per cell.

Plan cache load: loading a cached plan replayed every chunk's branching
forecast to recheck its `branch_max` (0.38 of 0.43 s per load on a 7-chunk
jerk plan; the CLI loads it twice per file). The cache key and memory model
version already cover every input of the forecast, so loading trusts the
cached value: 2 ms per load.

The 400-720 Hz part of that 720 Hz search (2^26 samples, 30 GB, CUDA) runs:
6 chunks at batch 65536 with 16 to 54 seed slices, planned in 11 s; 2 runs
searched in 994 s (about 8 minutes per run, so about 4.4 hours for 32 runs),
no overloads, device peak 29.9 GB (nvidia-smi, CUDA context included) and
1.1 GB of host memory.

Chunk bounds and a weak signal: on that 10-500 Hz search a weak real signal
at 280.14 Hz (score 8 or more in about a third of the 32 runs) was recovered
in 11 runs by the old 4 GB plan and in 5 by the new one. Searching single
chunks around it, without overloads: the old chunk's exact bounds gave 11,
the same chunk sliced or not gave identical final leaves, and the same chunk
with its bounds rounded by 4e-5 Hz gave 6. Other bounds gave 1, 5 and 11, and
the widest chunk of all (249-504 Hz, from a 30 GB plan) 19. A chunk's FFA
grid is laid out from its limits, so any change of the bounds moves every
grid point; a signal near the thresholds is caught or not by where the
points fall. Recovery has no trend with width or f_max, so the chunk width
is not a sensitivity lever. That search ran circular at eta 2, where the
loss is largest (next paragraph); at eta 1 the same signal is found in 31 of
32 runs.

Tolerance eta and a real signal (S/N about 20 when folded): on one chunk
around it, 32 runs, the runs scoring 8 or more and the median best score
were, for a jerk search, 29 / 18.2 at eta 1, 26 / 16.7 at 1.25, 22 / 14.6
at 1.5, 20 / 10.1 at 1.75 and 18 / 11.1 at 2; for a circular search 31 /
18.1, 31 / 18.0, 28 / 14.3, 20 / 10.5 and 2 / 0 (wall time of the whole
command, planning included: 127, 58, 39, 21 and 15 s). The jerk search
loses smoothly; the circular one keeps eta 1 sensitivity to 1.25 and loses
the signal at 2. Pick eta from such a curve, measured on the search's own
configuration.

Reserve of a slice (docs/memory.md): on the 495 Hz chunk above, which holds a
bright pulsar, the busiest slice held 65-100 cells' worth of leaves beyond
its noise forecast at 32 and 64 slices (the room for one CUDA batch covered
it). At 700 Hz (2 runs, 30 slices) the busiest slice peaked at 13.2 M live
leaves against a noise forecast of 28.7 M, the mean slice at 9.6 M. A
reserve of 128 cells (23 M leaves there) turned a 0.5 Hz band at 8 GB into
537 minimum-width chunks of about 1500 slices; 32 cells plans it as 2 chunks
of up to 48 slices. A reserve of 1/16 of the whole chunk's forecast made every
slice of the 720 Hz search mostly reserve (64 slices at batch 32). Wide
sliced chunks without the FFA share cap pushed the shared FFA buffers to
about 23 GB and the 720 Hz search no longer fitted.
