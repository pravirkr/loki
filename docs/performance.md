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

### Spectral zap on the GPU

A CUDA search used to run the whole preprocessing on the CPU as soon as the
zap was on (the zap had no GPU code). The zap now runs on the device, with
the whole-series transforms on cuFFT (`docs/fft.md`). Measured 2026-11-18,
L40, `loki.libloki.io.preprocess` with 4 CPU threads, synthetic series with
lines, a pulsar and bursts (0.04-0.08 % masked, so three removal passes),
`zap_periodic` on and 1,200 narrow birdie windows, `noise` mode; times
include the host-device copies. A job preprocesses once, so the first CUDA
call of the process (which also plans cuFFT) is the figure that counts; a
repeated call is given in brackets:

| Series | CPU, no zap | CPU, zap | CUDA, no zap | CUDA, zap |
|---|---|---|---|---|
| 2^26 samples | 1.8 s | 8.1 s | 0.24 s | 0.72 s (0.58 s) |
| 2^27 samples | 3.5 s | 23.6 s | 0.44-0.48 s | 1.11 s (0.95 s) |

Both backends zapped the same bins (10,419 and 20,771) and masked the same
samples; the outputs differ by 3.7e-6 of the noise RMS. On smaller cases
(2^20 samples, every mode, with and without gaps, even and odd lengths) the
zapped bins and masks were identical and the self-check ratios within 1e-4.

On a real 2^26-sample series with a bright binary pulsar, 1,219 birdie
windows and a 7.5 s masked run (CUDA EP search, 80-95 Hz, acc +-40), the
GPU zap against the CPU zap of the previous build: the same zapped bins in
all three modes (1,108,234 for `noise` and `zero`, 410,947 for
`phase_kept`), self-check ratios equal to three decimals, and the same
candidates (117, 634 and 147) with S/N differing by less than 0.001.
Preprocessing took 0.6-0.7 s instead of 5.1 s (`phase_kept`) and 8.3 s
(`noise`), the whole run 1.5 s instead of 11.6 s (`noise`), and the peak
host RSS fell from 2.5 to 1.2 GB. On a real 2^27-sample series (CUDA EP
acceleration search, 1.7 % masked, 2,539,819 zapped bins, warm plan cache)
preprocessing took 1.1 s instead of 24.5 s and the whole job 9.8 s instead
of 33.4 s (peak host RSS 2.2 instead of 4.9 GB); the zapped bins, the
self-check and the 117 candidates matched, S/N within 0.001.

### Device-resident search input (`io::SearchSeries`)

With the zap on the GPU, a CUDA search still copied the preprocessed
`ts_e`, `ts_v` back to the host and the sweep uploaded them again. The CLI
now keeps them on the device (`io::SearchSeries`) and drops the raw host
series after preprocessing. Measured 2026-11-18, L40, CUDA EP searches,
against the previous build: on the real 2^26-sample series above (all three
zap modes) preprocessing took 0.5-0.6 s instead of 0.6-0.8 s and the peak
host RSS fell from 1.15-1.19 to 0.77 GB, with bit-identical result files.
On the real 2^27-sample series, preprocessing took 1.0 s instead of 1.2 s,
the job 9.6-9.7 s instead of 9.9 s, and the peak host RSS fell from 2.23 to
1.52 GB; preprocessing and the 117 candidates (top S/N 16.76) matched, and
the result files differed only as much as two runs of the previous build
differ from each other (Fourier-domain EP jitter, `docs/architecture.md`).
The saving is host memory more than time: the round trip of 1 GB costs
about 0.2 s.

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

**Final leaves on disk (slice scratch).** A run pruned in seed slices kept
every finished slice's final leaves on the host until it wrote them. A bright
line that is not zapped fills each slice's tree, so the host memory grew with
the number of slices. Measured 2026-11-18, L40, a real 2^26-sample series
with a sinusoid at 420 Hz added (amplitude 0.02, 0.05 and 0.2 of the noise
RMS), circular Taylor search (crackle orbit, nbins 16, eta 1), 400-445 Hz,
one run, one chunk of 10 seed slices, CUDA, 30 GB; outputs on local disk:

| Series | Before: peak host RSS | After: peak host RSS | Final leaves | Scratch | Search after |
|---|---|---|---|---|---|
| noise | 0.77 GB | 0.77 GB | 6,425 | 0.8 MB | 33.6 s (33.8 before) |
| line, 0.02 | 12.1 GB (search 83.5 s) | 0.77 GB | 50-78 M | 5.8-9 GB | 81.9-93.1 s |
| line, 0.05 | killed above 14.4 GB | 0.77 GB | 110 M | 12.8 GB | 106 s |
| line, 0.2 | killed above 15.1 GB | 0.77 GB | 312 M | 36.0 GB | 235 s |

- The host memory no longer depends on the leaves; the disk takes them. Noise
  runs write almost nothing and run as before.
- The writes stall the pruning: the 36 GB took 70.9 s of the 235 s search
  (buffered writes at about 0.5 GB/s on that disk, against 2.3 GB/s with direct
  I/O), and reading them back for the per-run reduction made the
  postprocessing 64-83 s.
- Final-leaf counts of an overloaded CUDA run are not reproducible: two runs
  of the same 0.02 series gave 50.5 M and 78.0 M final leaves (23 overload
  trims each). The candidates above threshold matched (top S/N 82.63).
- A 0.02 run failed with a full disk quota when its scratch went to a home
  directory with a quota; the scratch was removed and the run stopped with an
  error. Put the output directory on a disk with room for such runs, or zap,
  mask or harvest the line.

**Final leaves in memory up to a cap.** The scratch above is a fail-safe:
real runs end with tens of final leaves (about 45,000 the most seen), and
writing them through a file only adds work. A sliced run now holds its final
leaves on the host up to 2^20 rows (`slice_spill_rows()`) and moves them to
the scratch only when a slice would pass that. The reader also reads the
series straight into its buffer and closes the file before the variance is
allocated. The same setup as above:

| Series | Peak host RSS | Scratch | Search | Postprocess |
|---|---|---|---|---|
| noise | 0.64 GB (0.77 before) | none | 33.5 s | 0.0 s |
| line, 0.02 | 0.64 GB | 8.8 GB, 76.5 M leaves, 15.9 s to write | 89.0 s | 5.4 s |

- The 0.77 GB peak was at the end of loading: the samples, the file pages the
  reader had mapped and the variance, 12 bytes per sample. The peak is now
  during preprocessing on the GPU: the samples and the variance (8 bytes per
  sample) and the CUDA context (about 0.12 GB). The search holds 0.15 GB.
- A run below the cap writes from memory, bit for bit as from the scratch.

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

Counting circular leaves by basis (`level_stats_circular`): on that chunk,
CUDA, 4 runs of 127 stages, a pass over each stage's survivors after the
stage cost 0.27 s of 15.4 s (1.8 %), whether it used Thrust counts or one
kernel with warp ballots: the cost is the extra strided read of every leaf.
Counting in the transform kernel, which already holds each leaf and its
class, costs nothing measurable (15.43 vs 15.46 s, three pairs, run-to-run
spread 0.1 s), and the per-level counts copy back once per prune instead of
syncing every stage. Writing the basis code into the leaf costs nothing
measurable either (0.3 %, inside the spread).

`ep.p_orb_max` (circular searches; docs/memory.md) kills a leaf only once its
cell, widened by `validation_significance` (5) widths, is wholly a slower
orbit, so it bites only when the snap is measured to a fraction of
`omega_min^2 |accel|`, that is for a limit close to the observation length.
Injections (2^24 samples, T = 1074 s, 100 Hz, S/N 25, circular orbits with
m_c = 0.3 at four orbital phases, `p_orb_min` = T, 8 runs, CUDA):

- limit 2000 s (1.9 T): leaves -0.6 to -4.6 % for orbits inside it and -0.2
  to -1.1 % for orbits of 3000 and 6000 s, which keep their best S/N;
- limit 1300 s (1.2 T): leaves -7 to -26 % inside it, -3 to -13 % for orbits
  of 2000 s, whose best S/N falls by 4.7-6.6 at three of four phases, and
  -2 to -4 % for orbits of 4000 s (snap never resolved), which keep theirs;
- orbits inside the limit (0.9 and 1.0 times it, 16 cases) keep their best
  S/N in every case. The best score of 7 of their 128 runs changed, and 3 of
  the 16 cases lost one run scoring 8 or more. In each changed run the leaf
  removed was a partial match scoring at most 11.4, 6.8-41 snap widths from
  the true orbit (`snap = -omega^2 accel`); in the 43 unchanged runs scoring
  15 or more the best leaf lies within 0.7 widths of it and passes the limit
  with a margin of at least 17 %, also at 1.0 times the limit.

A CPU search at the 1300 s limit behaves alike (leaves -13 %, best S/N kept,
one run fewer at 8 or more). On a real 280 Hz binary pulsar whose detections
all sit in the Taylor channel (snap unresolved; 5500 s observation, 32 runs,
272-318 Hz, eta 1), limits of 80000, 40000 and 20000 s left the recovery
unchanged (31 of 32 runs, median 18.06) and cut the leaves by 0.003, 0.15
and 2.0 % (wall time 128, 128, 126 s against 128 s). At such settings the
limit is nearly inert; restricting the children at branching with the same
margin would remove the same leaves earlier, so it would be as inert.

The validation's orbit checks prune hard but lose orbits at the period
limit. Injections (2^24 samples, T = 1074 s = `p_orb_min`, 100 Hz, S/N 25,
crackle order, m_c_max 5, eta 1, 16 runs, CUDA) at P_orb = 1.0, 1.1, 1.3
and 2.0 T, four orbital phases, with a light (0.3 M_sun) companion and with
m_c = m_c_max, m_p = m_p_min (the mass limit), were searched with the
`p_orb_min` and mass checks as they are (the point estimate
`omega^2 = -snap / accel`), with both widened by `validation_significance`
cell widths (kill only if `|snap| - k dsnap > omega_max^2 (|accel| +
k daccel)`, or if `|accel| - k daccel` exceeds the mass limit at the cell's
largest `omega^2`), and with both removed (a scratch build). Removing them
leaves 1.3-2.8 times as many leaves; widening them, 1.09-1.31 times. At
P_orb = `p_orb_min` the checks as they are lower the best score in 6 of the 8
cases (by 2.1-5.5) and lose runs scoring 8 or more in 5; widened, they recover
as much as removing them does (same best score in all 32 cases). From
1.1 `p_orb_min` the widening changes no best score. The widened checks are
the default since 2026-11-18. Rerun with the resolve fix below (runs scoring
8 or more, of 512): 350 as they were, 361 widened, 370 removed; at P_orb = T
the old checks lowered the best score in all 8 cases (by 0.7-4.4), the
widened ones in none. Each of the 9 runs the widened checks lose against
none had its best leaf off the orbit (a CPU trace of one: every leaf only
the unchecked search kept is 10 or more of 16 bins rms from the orbit and
implies a period of 0.76-0.89 T). On the real 280 Hz binary pulsar (one
chunk, 32 runs, eta 1) the widened checks keep the recovery (32 of 32,
median 18.06) with 6.6 % more leaves and 8.5 % more wall time (139 against
128 s, two pairs). Most of the loss of
compact orbits is elsewhere: without an orbit the same search finds the
signal in 14 of 16 runs (median best score 18.9); with the heavy companion
at P_orb = T, even without the checks, in 4-9 runs (median 0-8.5). Traces of
lost runs (CPU, no checks, every leaf of every stage against the true orbit,
frequency included) show two losses, the same with and without the checks:

- a run referenced at the series edge moves its reference epoch by a
  segment (4.2 s) every stage; while the snap cell is unsplit its leaves
  carry the box-centre snap, the transforms shift the jerk by the snap
  error each stage and the aggressive tile does not widen cells, so the
  orbit leaves its cells from about stage 15 and the run loses it by
  stage 41 (traced before the resolve fix below). Without the checks, runs
  referenced at segments 0-16 or 110-127 find this orbit in 5 of 24, the
  others in 25 of 40 (after the fix). Accepted as the cost of the
  aggressive tile: the remedies (mid-only references, holding the
  reference epoch, more snap cells, widening) cost run independence, a
  start-independent volume or more volume.
- a mid-series run (reference segment 42 of 128, after the resolve fix
  below) has no drift: a leaf whose phase matches the orbit to 0.2-0.9 of
  16 bins rms over the span survives to stage 33, but scores only 0.1-0.8
  above the thresholds (5.22 at stage 33 against 5.09) and falls below at
  stage 34, a stage without branching. The no-orbit control's leaf, matching
  to 0.3-0.5 bins, scores 9.0 there. Already at stages 2-5 the matched
  orbit leaf scores 4.0 falling to 2.6 against the control's 4.0-4.6 at the
  same match. Not explained yet: such losses come in contiguous blocks of
  reference segments that move with the orbital phase (all of 0-59 for one
  phase), like the known loss of runs referenced near the orbit's nodes.

Frequency of the moving-grid resolve (2026-11-18): on the moving grid a leaf
keeps its seed's f0 and carries its own frequency offset in its velocity,
but the pruning-stage resolves took the added segment's frequency from the
velocity change since t_init, dropping the leaf's offset there, so a fold
could use a neighbouring frequency cell; its phase was right. Resolving from
the leaf's absolute velocity at the added segment (every basis, CPU and
CUDA; the ascend resolves were right) changes no FFA output, and recovery
only within run-to-run jitter. On the 32 injections above, with the checks
as they are: runs scoring 8 or more 341 -> 350 of 512 (16 cases up, 9 down;
best score 10 up, 8 down), leaves +8 %. On the real 280 Hz signal at eta 1
(one chunk, 32 runs): jerk 29 of 32 before and after (median 18.21; 1 run
gained, 1 lost, 3 lower), circular 31 -> 32 (median 18.06), wall time 128 s
both. Results earlier in this file were measured before this fix, except the
mid-series trace and the edge counts above. Leaves on the real chunk:
unchanged (jerk +0.9 %, circular -0.02 %); on the injections they rise or
fall per case.

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

### Threshold scheme: Pd pinning and the bank descent (`mode = "descent"`)

Measured 2026-11 on an L40 with 4 host threads, on the acceptance suite of
`bench/scripts/schemes/solve_suite.py` and on 230 branching patterns of a
production search plan (2, 3 and 5 parameters, 16-1024 bins, mostly 127
stages, `min_pd` 0.1). Schedules were judged on samples that chose nothing
(2^17 signal trials, 3 × 2^16 noise trials). See `docs/dts.md` for the
method.

What the measurements say about the existing modes:

- On the long patterns L = C / Pd rises steeply with Pd, so `min_pd` binds
  and any Pd above it is paid for. `shared` delivered Pd 0.102-0.114 for a
  target of 0.1 (its repair only lowers thresholds); the planner's refit
  above 128 bins (a 0.1 S/N upward shift grid ranked by L) delivered
  0.11-0.17. Shifting those schedules to the target saves 6-41 % of C.
- A common shift of either sign to `min_pd` + 2 standard errors on a fresh
  bank (a two-sided pin) brings `shared` to Pd 0.099-0.104. It does not
  remove the cost scatter between seeds (cv 0.10-0.13).
- `improved`: the in-run Pd is 4-19 % optimistic on every pattern, and the
  cost at equal Pd is up to 2× that of `shared`; it is 3-4× faster
  (24 / 67 / 142 ms at 16 / 64 / 128 bins against 109 / 234 / 434 ms for
  `shared` and 281 / 573 / 1021 ms for `robust`).
- Last threshold (issue #42): at `min_pd` 0.1 the free last thresholds were
  6.0-8.4 with the descent, below a candidate cut of 7 on 51 of 230
  patterns (35 of the 170 up to 128 bins). On those 35, judged on
  independent samples at equal Pd: raising the last threshold of the free
  schedule to 7 and shifting the others costs 3.0 % of C (geometric mean,
  -3 % to +14 %); a floor of 7 inside the descent 3.4 %, i.e. 1.004 of that
  baseline (0.97-1.03), with the last threshold at 7.00-7.04 and Pd
  0.099-0.105. Of the other 135 patterns 131 are bit-identical with and
  without the floor; 4 whose free last threshold is within 0.1 of it differ
  by -5 % to 0 % in C. Two earlier versions were worse: holding the last
  threshold at the given value pulled it down when the free one was higher
  (`fap` 0.01 is S/N 3.6-4.8), and not letting a block that holds the last
  stage move down once the last threshold sat on the floor ended 1.1 %
  above the baseline (29 of 35 patterns worse).

Two plans of the descent appear below. The first version ran six passes
from the flat start (the same signal z-score at every stage): 4096 trials at
margin 0.4 and step 0.2 twice, then 8192 trials at 0.3/0.1 and at 0.2/0.05
twice each. The current default chooses its start by a measurement and runs
three levels, repeating a level while it gains ("Start by measurement,
levels that repeat" below). The numbers are for the six-pass plan unless
they are marked as the current default.

The bank descent (six-pass plan) against the search modes, same suite, 8
seeds:

| case | mode | Pd (min-max) | C (geomean) | cv of C | time |
|---|---|---|---|---|---|
| production, 0.1 | `shared` | 0.102-0.114 | 5.75e4 | 0.099 | 0.23 s |
| | `robust` | 0.102-0.108 | 5.26e4 | 0.024 | 0.56 s |
| | `descent` | 0.101-0.103 | 4.69e4 | 0.018 | 0.25 s |
| production, 0.05 | `shared` | 0.051-0.056 | 1.29e4 | 0.129 | 0.23 s |
| | `robust` | 0.052-0.052 | 1.14e4 | 0.029 | 0.54 s |
| | `descent` | 0.051-0.053 | 1.09e4 | 0.036 | 0.26 s |

On the plan's patterns the descent's cost was 0.78-1.01 of `robust` pinned
to the same Pd (research prototype, 13 patterns). The library mode on all
230 patterns, one seed each: every solve `met` (215) or `within_tolerance`
(15), Pd on 2^16 independent trials 0.0985-0.106 for the target 0.1, free
last thresholds 6.0-8.4.

Seed scatter, 22 patterns (2-5 parameters, 16-128 bins) × 8 seeds, each
schedule judged on the same independent samples: Pd 0.0991-0.1044 over the
176 solves (median 0.1014; standard deviation per pattern 0.0005-0.0012,
which is the binomial error of the 2^17-trial pin); cv of C 0.018 (median)
and 0.045 (worst pattern, 5 parameters); the reported Pd within 2.8 % of the
judged one.

Time of one solve on 127 stages (3 parameters), by bin count:

| nbins | 16 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| solve (s) | 0.22 | 0.24 | 0.28 | 0.73 | 2.1 | 7.7 |
| genealogy builds (s) | 0.04 | 0.04 | 0.05 | 0.15 | 0.30 | 0.66 |
| host moves (s) | 0.15 | 0.16 | 0.16 | 0.15 | 0.15 | 0.16 |
| engine construction (s) | 0.02 | 0.02 | 0.02 | 0.05 | 0.18 | 0.72 |

Up to 128 bins the host passes dominate: the forward pass that re-marks the
genealogy after an accepted move (70-90 ms in total) and the block deltas on
the genealogy and on the signal bank. Above 128 bins the rest is the three
unpruned signal samples (2^15 trials for the descent, 2 × 2^17 for the pin
and the report). It is slower than `shared` at 16 bins, equal at 64, faster
at 128, and 0.02-0.08 s on 4-32 segments. The sample sizes are the engine's
solve budget (2^17 per signal bank), the same at every bin count. Open:
fewer final trials at large bin counts.

Where the host time goes (16-128 bins, 127 stages, ms per solve): forward
re-mark 65 (32 candidate evaluations of about 10^6 genealogy entries each,
2 ns per entry: memory traffic of scores, parents, masks and the survivor
list), pins on the signal sample 22, backward pass 19, genealogy block
deltas 15, signal block deltas 13; 25-31 of the 32 candidates are accepted.

Warm starts (`descent_start`): the solutions of similar patterns are close,
so a solved schedule is a better start than the equal z-score one. C relative
to the full six-pass solve from the default start, judged on independent
samples:

| start | passes | C / full (geomean) | worst | time |
|---|---|---|---|---|
| default | 6 (full) | 1 | | 0.24 s |
| default | last 3 | 1.068 | 1.16 | 0.14 s |
| default | last 2 | 1.20 | 1.61 | 0.11 s |
| neighbouring pattern (43 pairs, same parameters and bins) | last 3 | 0.980 | 1.04 | 0.14 s |
| neighbouring pattern | last 2 | 0.986 | 1.02 | 0.11 s |
| one donor (3 parameters, 64 bins) for 21 patterns of 2-5 parameters, 16-128 bins, thresholds as they are | last 3 | 1.000 | 1.08 | |
| the same donor, thresholds | last 2 | 1.086 | 1.70 | |
| the same donor, as signal z-scores per stage | last 3 | 0.996 | 1.12 | |
| the same donor, as signal z-scores | last 2 | 1.024 | 1.22 | |

The warm solves below 1 show that the full solve is about 2 % from
converged. Two passes from a neighbour take 0.10 / 0.12 / 0.16 s at 16 / 64 /
128 bins, within the time of `shared` (0.11 / 0.23 / 0.43 s). The converged
schedules share a shape in signal z-scores (mean minus threshold over the
standard deviation of the unpruned signal score) on the 127-stage patterns:
0.6-2.6 at the first stage, low a quarter of the way in (-0.3 to 0.8 at
stage 31), 0.4-1.4 half way and 1.3-1.8 at the last stage.

Chains, as a planner would run its regions in order (each pattern started
from the previous warm solution, two passes, C relative to that pattern's
full solve):

| chain | links | C / full (geomean) | first half | second half | worst |
|---|---|---|---|---|---|
| thresholds, one bin count and parameter count, 5 chains | 37 | 0.988 | 0.99-1.00 | 0.96-1.01 | 1.085 |
| z-scores through 16-32-64-128-64-32-16-32-64-128 bins, 2 parameters | 9 | 0.987 | 1.006 | 0.972 | 1.023 |
| the same, 3 parameters | 9 | 0.982 | 0.988 | 0.978 | 0.997 |
| the same, 5 parameters | 9 | 0.982 | 0.969 | 0.993 | 1.040 |

No drift: the chains stay below the full solves. Three passes instead of
two gave 0.975-0.99 on the z-score chains. Used in `EPRegionPlanner` (below).

A shorter plan from the default start (the first two passes, then one each
at 0.3/0.1 and 0.2/0.05) costs 2.0 % more C (worst case 5.7 %) in 160 ms
instead of 230 ms; with hat moves it was 5.2 % more. It can be passed as
`descent_passes` where the time matters more.

Start by measurement, levels that repeat (the current default). Because the
solutions share a shape, a fixed smooth z-profile (2.0 at the first stage,
0 after 4 % of the stages, -0.1 after 16 %, 1.7 at the last one) is a
second candidate start next to the flat one; both are pinned and evaluated
on the same 4096 noise trials and the cheaper one is taken. On the 127-stage
patterns the shape costs 1.1-6.8 times less than the flat start and is
chosen in 25 of 26 cases (targets of 0.05, 0.1 and 0.5; at a target of 0.9
the flat start is 1.4 times cheaper); on 4-32 segments the flat start is
cheaper in 4 of 5 cases (the shape, forced there with three passes, ends up
to 49 % above, at 8 segments). The plan is one pass per level (0.4/0.2 all blocks, 0.3/0.1
blocks up to 32, 0.2/0.05 blocks up to 8), a level running a second time
when its first pass lowered the genealogy's cost by more than a fraction r.
C relative to the six-pass plan, 12 cases (3 short, 9 of 127 stages) × 3
seeds, judged on independent samples:

| plan | C / six-pass (geomean) | worst case | median time | passes on 127 stages |
|---|---|---|---|---|
| six passes, flat start | 1 | | 234 ms | 6 |
| measured start, r = 0.10 | 0.988 | 1.008 | 189 ms | 4-5.3 |
| measured start, r = 0.15 | 0.993 | 1.008 | 178 ms | 4-4.7 |
| measured start, r = 0.25 (default) | 0.996 | 1.022 | 156 ms | 3-4.3 |
| measured start, never repeat | 0.995 | 1.022 | 144 ms | 3 |
| flat start only, r = 0.15 | 1.010 | 1.043 | 178 ms | 4-5.3 |

What a pattern outside this family gets is the flat start with the same
levels. Forced on the 26 cases of 127 stages (one seed each): C 1.023 of the
six-pass plan (worst 1.16, a 5-parameter pattern at 32 bins; 4.3 passes on
average), against 0.995 (worst 1.05, the target of 0.9) for the default.
With r = 0.15 the flat start alone was 1.010 (worst 1.04) in the table
above: lower r if such patterns matter more than 20 ms.

The current default (`pd_policy = "pin"`) and `"free"` on the checks above:

- Suite, 8 seeds (`solve_suite.py --pd-policy free`, Dinkelbach L = C / Pd):
  production pattern at 0.1: Pd 0.101-0.104, C 4.66e4 (cv 0.023), L 4.55e5; at
  0.05: Pd 0.051-0.053, C 1.07e4 (cv 0.018), L 2.05e5 (`"pin"`: C 4.68e4 /
  1.07e4, cv 0.020 / 0.030). On short patterns (64 bins, target 0.1, 8 seeds;
  `shared` and `robust` with `ntrials` 1024, `nprobs` 40 and the `window`
  guess; judged on 2^17 signal and 3 × 2^16 noise trials), `"free"` uses a bracketing + bisection start search over Pd (exiting
  on the first upward probe when L rises on deep trees), a wider coarse level
  (margin 0.8, step 0.4) for flat/interior starts, a global-shift block (one
  block over every stage, at every pass), and
  scaled one-sided repair (`search_pd > target + max(0.15 min(target, 1 - target), 6 SE)`):

  | Pattern | `descent` (`free`) L (Pd) | `shared` L | `descent / shared` | `descent / robust` | time |
  |---|---|---|---|---|---|
  | 4 segments, S/N 10 | 4.20 (0.992-0.994) | 4.21 | 0.998 | 0.998 | 0.027 s |
  | 8 segments, S/N 8.1 | 3.01 (0.760-0.780) | 3.02 | 0.996 | 0.998 | 0.037 s |
  | 16 segments, S/N 8.1 | 5.71 (0.490-0.523) | 5.75 | 0.994 | 0.996 | 0.055 s |
  | 32 segments, S/N 8.1 | 14.05 (0.277-0.324) | 14.28 | 0.984 | 0.995 | 0.084 s |
  | 32 segments, S/N 6.0 | 39.63 (0.119-0.177) | 40.93 | 0.968 | 0.984 | 0.077 s |
  | production, first 16 stages | 15.13 (0.432-0.459) | 15.45 | 0.980 | 0.996 | 0.054 s |
  | production, first 32 stages | 65.80 (0.142-0.152) | 68.13 | 0.966 | 0.987 | 0.078 s |
  | production, first 64 stages | 1656 (0.099-0.103) | 1832 | 0.904 | 0.911 | 0.100 s |

  On 14 short patterns it was not tuned on (real patterns cut to 6-48 stages
  at 16-1024 bins, synthetic ones at other S/N, bin counts and targets
  0.02-0.3; 4 seeds) `"free"` is 0.954-1.000 of `shared`'s L and
  0.980-1.002 of `robust`'s. Its design time at 1024 bins is 0.59-0.89 s on
  127 stages and 0.09-0.39 s on 7-64 stages (`"pin"`: 0.55-0.64 s and
  0.05-0.36 s).
- The global-shift block of `"free"` was cut into two half-blocks on every
  second pass (the half-block offset of the other block sizes applied to it
  too). Kept as one block over every stage, measured against the half-block
  version on the same seeds: L(new) / L(old), paired by pattern, target and
  seed, is 1.0003 +/- 0.0006 (geometric mean over 132 cells of 33 patterns x
  targets 0.01 / 0.1 / 0.5 / 0.9, 6 seeds; cells 0.973-1.020, none worse
  beyond 3 SE and 1 %); 2-parameter 1024-bin pattern 1.004 (16 seeds); the
  14 unseen short patterns 0.9995; no solve judged below target - 3 SE. Neutral, and
  kept because it does what the code describes.

- Pd jitter across targets, bins and patterns (`pd_policy = "pin"`, 21
  patterns: 2/3/5-parameter 127-stage at 16/64/128/512/1024 bins, 64-stage, and
  five 8-32 stage patterns × 8 seeds = 168 solves per target, judged on 2^17
  independent signal trials and 2 × 2^16 noise trials):

  | target | SE (2^17) | judged Pd range | Pd / target | (Pd - target) / SE | < target - 3 SE | `short` | C cv (med / max) |
  |---|---|---|---|---|---|---|---|
  | 0.01 | 0.00027 | 0.0094-0.0113 | 0.939-1.134 | -2.2 to +4.9 | 0 / 168 | 0 | 0.038 / 0.091 |
  | 0.02 | 0.00039 | 0.0193-0.0219 | 0.965-1.097 | -1.8 to +5.0 | 0 / 168 | 0 | 0.030 / 0.074 |
  | 0.05 | 0.00060 | 0.0490-0.0529 | 0.979-1.058 | -1.7 to +4.8 | 0 / 168 | 0 | 0.019 / 0.067 |
  | 0.10 | 0.00083 | 0.0976-0.1038 | 0.976-1.038 | -2.9 to +4.6 | 0 / 168 | 0 | 0.015 / 0.045 |
  | 0.20 | 0.00110 | 0.1980-0.2051 | 0.990-1.026 | -1.8 to +4.6 | 0 / 168 | 0 | 0.010 / 0.034 |
  | 0.50 | 0.00138 | 0.4966-0.5064 | 0.993-1.013 | -2.4 to +4.7 | 0 / 168 | 0 | 0.008 / 0.040 |
  | 0.90 | 0.00083 | 0.8983-0.9047 | 0.998-1.005 | -2.1 to +5.7 | 0 / 168 | 0 | 0.008 / 0.027 |
  | 0.99 | 0.00027 | 0.9900-0.9916 | 1.000-1.002 | -0.2 to +5.9 | 0 / 168 | 0 | 0.020 / 0.055 |

  Every one of the 1,344 solves lands inside `target + 2 SE ± 5 SE` with zero
  `short` deliveries. High-bin low-cost trees (2-parameter, 1024 bins, 16
  seeds at target 0.1) stay strictly unimodal at Pd 0.1006-0.1029 (sd 0.0006,
  C cv 0.026).
- Distance to repeated restarts (no global-minimum guarantee): restarting a
  cold `pin` solve four times from its own z-scores on fresh samples across 12
  patterns (16/64/128/512 bins × 2/3/5 parameters) gives best-of-4 / cold L =
  0.990 (geometric mean, 0.971 worst; single restarts scatter 0.971-1.074).
  A stronger reference finds more: the best judged C of three cold solves
  and six chained restarts with 2^17 signal trials, 2^15 noise trials per
  genealogy and finer steps (0.1 / 0.05 / 0.025 / 0.0125), every schedule
  re-pinned to Pd 0.1 + 2 SE and judged on fresh samples (14 patterns:
  127 stages at 16 / 64 / 256 / 1024 bins x 2 / 3 / 5 parameters, a 64- and
  a 32-stage truncation). Cold over best of nine: 1.071 geometric mean
  (1.036-1.127; 1.04-1.10 on 127 stages, 1.12-1.13 on the truncations); the
  first restart alone (no selection) is 1.016 of the best, so a cold `pin`
  solve costs about 1.05 of what one such restart reaches. This is an upper
  bound on the optimum, not a lower one; seed scatter of C is within
  0.95-1.05 (mostly +/-1.5 %).
- Against the grid modes at equal Pd (same re-pinning and judge; descent 4
  seeds, `shared` / `robust` 2): on costly patterns the descent is cheaper
  (`shared` 1.14-1.35 and `robust` 1.11-1.16 of its C on 5 parameters,
  `shared` 1.01-1.07 on most 3-parameter ones); on cheap and mid-length ones
  the grid modes win by 2-8.5 % (`robust` 0.939 on a 2-parameter 64-bin
  pattern and 0.976 on a 3-parameter 256-bin one, `shared` 0.915 / 0.938 on
  the 64- and 32-stage truncations).
- Cold time on 127 stages, 3 parameters, constructor + `run()` + `solve()`
  across 16 to 1024 bins (medians of 3 warmed runs, L40 GPU + EPYC CPU). Above
  64 bins CUDA uses a warp-collaborative shared-memory scorer
  (`GenericScorer::run_warp`, 8 warps/block, up to 32 KB shared memory at 1024
  bins, bit-identical to `GenericScorer::run`), and above 128 bins evaluates the
  2^17-trial pin/report signal banks with stage-pruned kernels
  (`h1_margins_above` / `count_h1_survivors`, CUB compaction every 8 stages,
  one H1 fold buffer instead of two). On the host, a 64 × 32 cache-blocked
  `SignalSample` transpose eliminates power-of-2 L1 set-aliasing at 2^15
  trials, `pin_shift` runs vectorized on the active trials above `g_ref`
  (exact fallback otherwise), and OpenMP is limited to multi-block `block_deltas`
  (bit-identical across 1/2/4/8 threads). CPU tiles H1 trajectories in L2
  (`kTile = 64` trials/thread: an H1 tile of about 3 MB per thread instead
  of 3.2 GB of H1 folds at 1024 bins; the whole CPU design grows host RSS by
  41 / 115 / 437 MB at 64 / 256 / 1024 bins, and a CUDA design peaks at
  0.45 / 0.52 / 0.69 GB of device memory including the context):

  | nbins | `descent` GPU (4 thr / 1 thr) | `descent` GPU (floor 7) | `shared` GPU | `descent` host moves / noise (passes) | `descent` CPU (4 thr) |
  |---|---|---|---|---|---|
  | 16 | 0.130 / 0.131 s | 0.134 s | 0.108 s | 71 / 18 ms (3) | 1.76 s |
  | 32 | 0.138 / 0.136 s | 0.136 s | 0.144 s | 75 / 19 ms (3) | 1.73 s |
  | 64 | 0.172 / 0.183 s | 0.158 s | 0.231 s | 83 / 26 ms (4) | 3.16 s |
  | 128 | 0.198 / 0.206 s | 0.197 s | 0.424 s | 73 / 26 ms (3) | 5.67 s |
  | 256 | 0.249 / 0.264 s | 0.279 s | - | 92 / 37 ms (4) | 10.3 s |
  | 512 | 0.363 / 0.380 s | 0.416 s | - | 90 / 57 ms (4) | 20.2 s |
  | 1024 | 0.593 / 0.608 s | 0.669 s | - | 77 / 76 ms (3) | 36.4 s |

  `descent` is faster than `shared` from 32 bins upward and within 22 ms at 16
  bins; at 256 / 512 / 1024 bins it is down from 0.67 / 2.0 / 7.5 s before warp
  scoring and stage-pruned H1. Its engine skips the pilot simulation of the
  data-driven beams (constructor 6-10 ms at every bin count instead of 19 ms at
  16 bins and 720 ms at 1024).
- Floor of 7 (`final_threshold`), KKT active-floor pinning (holding
  $\gamma_{M-1} = \gamma_{\text{floor}}$ and shifting `[0, M-1)` whenever
  $\gamma_{M-1} \le \gamma_{\text{floor}}$): on 170 patterns up to 128 bins, C
  is 0.999 of the post-hoc floor baseline (against 1.009 before KKT pinning),
  and bit-identical to the free solve whenever the floor is inactive.

The planner on the descent (three production configurations of 2, 3 and 5
parameters — acceleration, jerk and circular orbit — 127 stages, regions of 16
to 1024 bins, target 0.1; `EPRegionPlanner` uses `pd_policy = "pin"` and leaves
`final_threshold` unset so DTS chooses $\gamma_{M-1}$ freely by cost and
`RunLeafReducer::add` enforces `snr_min` strictly after `ascend`; every
region's thresholds judged on 2^17 fresh signal trials and 2 × 2^16 noise
trials at the true bin count):

| | previous design (`shared`, re-fit above 128 bins) | descent, chained (`pin`, warp + pruned H1, free $\gamma_{M-1}$) |
|---|---|---|
| Pd of a region | 0.0997-0.167 | 0.0991-0.1041 |
| C per plan (geometric mean over the regions) | 1 | 0.721 / 0.814 / 0.790 |
| C per region | 1 | 0.538-0.986 |
| Last threshold | 5.5-7.8 | 5.84-7.83 |
| Design, 16 / 32 / 64 / 128 bins (GPU) | 0.11-0.14 / 0.15-0.22 / 0.24-0.27 / 0.44-0.47 s | first region 0.14-0.16 s; chained 0.10-0.15 s |
| Design, 256 / 512 / 1024 bins (GPU) | 1.0 / 3.2 / 12.3 s | 0.18-0.19 / 0.27-0.29 / 0.47-0.54 s |
| Whole plan (GPU, `--plan-only`, warmed) | 30.6 / 18.1 / 29.0 s | 2.6 / 2.5 / 13.2 s |
| Whole plan, 2 parameters, CPU, 4 threads | 444 s | 99.4 s |

- Every region across all three plans is cheaper than the previous design
  (0.538-0.986), with judged Pd 0.0991-0.1041 up to 1024 bins (where the
  previous re-fit drifted to 0.11-0.17). An independent judge (own seeds,
  2^17 signal and 2 x 2^16 noise trials) gives Pd 0.0995-0.1048 and C
  0.721 / 0.810 / 0.790 per plan (per region 0.540-0.968). On the 5-parameter circular plan the
  7 designs of its 10 chunks take 1.4-1.8 s in total; the rest of
  `--plan-only` is seed-grid generation.
- Chained start (previous region's z-scores, two warm passes) against cold
  default solves of every region on the same three plans (16-1024 bins): L
  chain / cold geometric mean 0.969 / 1.002 / 1.001 (per-region 0.938-1.061),
  reducing total DTS time across regions by 17 % / 17 % / 24 % (2.53 -> 2.09 s,
  1.78 -> 1.49 s, 2.32 -> 1.76 s). An independent re-measurement (same seeds
  and samples, own judge) gives 0.968 / 0.993 / 1.015 (per region
  0.933-1.072).
- Other targets (2-parameter plan, every region judged): `min_pd` 0.02 /
  0.12 / 0.3 / 0.9 / 1.0 give Pd 0.0200-0.0219 / 0.1194-0.1244 /
  0.3007-0.3075 / 0.9001-0.9030 / 0.9990-0.9992 (`min_pd` 1 cannot be met and
  is designed at 0.999, with a warning; the peak chunk memory rises from 1.44
  to 21.3 GB).
- Small memory limits shrink the design samples as `sample_budget` sets them
  (2-parameter plan, GPU): at 4 GB, 2^15 signal trials at 1024 bins, Pd
  0.100-0.105, C 0.998 of the 30 GB plan; at 2 GB, 2^14, Pd 0.100-0.106, C
  1.008; at 1.6 GB (2^13 at 1024 bins, 2^14 at 512, 15 chunks), Pd 0.100-0.103
  up to 256 bins but 0.105-0.111 at 512-1024 bins: the pin aims 2 SE above
  `min_pd`, and the SE of 2^13 trials is 0.0033. On the GPU the limit is the
  smaller of the user limit and the free device memory, so a plan made on a
  GPU with little free memory can differ.
- The replan pass (5-parameter plan at a 20 GB limit: 12 chunks, replanned
  with fixed shared FFA buffers; at 12 GB, 16 chunks) reuses the same 7
  region designs, bit-identical to the 30 GB plan (judged Pd 0.0995-0.1030).
- Plan cache: a current cache is loaded without redesign; one of the previous
  version (1.22.0) is rejected with a request to re-plan, and a different
  `min_pd` does not match. `search ep --plan-only` always re-plans.
- Two plans of one configuration hold bit-identical thresholds.
- Peak chunk population (`max_sugg`) relative to the previous design: +4.4 %
  (2-parameter), +34.2 % (3-parameter) and +0.6 % (5-parameter; predicted live
  leaves +28.9 %, +32.2 % and -6.9 %), with the same chunk counts.
- Real-data end-to-end searches of survey observations (`2^26` samples,
  32 runs, L40 GPU, 4 CPU threads, like-for-like growth factors against the
  previous `shared` + refit planner):
  - 3-parameter jerk (`10-720 Hz`, `16-1024` bins, growth factor 2.0, two
    observations): plan `18.6-18.7 -> 2.3 s` (8×), search `8.9-9.2 -> 6.6-6.9 s`
    (-24 %), total `30-31 -> 11.5-11.9 s`, peak RSS `1.13 -> 0.64 GB`. On the
    first observation all 8 known pulsars (including one at 512 bins)
    match to 0.01 S/N; on the second, 6 of 7 match (one drops 0.06 S/N, and one
    marginal off-DM detection at S/N 7.05 is missed). Across 6 neighbouring
    DM/segment files with the same jerk configuration, total known
    pulsar-file detections are 40 (new) against 43 (old): all 40 consistent
    detections match across both planners, while 3 single-file marginal off-DM
    detections at S/N 7.01-7.66 (ranks 1584-2285) are recovered only by the old
    planner.
  - 3-parameter jerk at growth factor 1.5 (11 regions up to 923 bins,
    like-for-like baseline at growth factor 1.5): plan `26.1 -> 3.0 s` (8.7×),
    search `11.5 -> 11.0 s`, total `41.2 -> 17.4 s`. Seven known pulsars
    match closely; the new planner recovers 1 additional pulsar at 923 bins
    (S/N 8.78) that the old planner missed, while missing 2 marginal detections
    (S/N 7.21 and 8.34) that the old planner kept.
  - 5-parameter circular crackle (`10-500 Hz`, `16-512` bins, `eta` 2, growth
    factor 2.0): plan `5.8 -> 1.4 s` (4.1×), search `295.2 -> 181.7 s` (-38 %,
    saving 113 s), total `302.3 -> 185.8 s`; all 3 known pulsars recovered
    (strongest improves by +0.78 S/N, other two identical).
  - What the designs predict for the marginal losses: Pd per run of each
    plan's own thresholds on the search's signal model, region by region (the
    jerk plans of growth factor 2.0 and 1.5). At S/N 7.0 the previous design
    held 0.0128-0.0215 and the descent holds 0.0100-0.0171 (0.64-0.92 of the
    previous up to 615 bins, 1.14 at 923); at S/N 8.34, 0.052-0.072 against
    0.050-0.060; at S/N 10 the descent is higher up to 128 bins (0.155-0.166
    against 0.149-0.158). The 32 runs of a search are correlated, so per
    search an S/N-7 pulsar is kept roughly 20-30 % of the time and an
    S/N-8.3 one 60-80 %, under either design. The S/N 7.0-7.7 losses (16, 32
    and 128 bins) fit the 10-25 % lower per-run Pd at S/N 7 there; the 8.34
    loss sits in a 54-bin region where both designs hold the same Pd
    (0.056 / 0.053), so it and the new gain at 923 bins are chance outcomes
    of two different schedules. Matching the previous Pd at S/N 7 needs
    `min_pd` 0.12-0.15: at 0.15 the descent holds Pd(7) 0.018-0.028 at 1.2-1.9
    times the previous plan's C (1.5-1.9 up to 128 bins); 0.2 costs 1.8-3.2
    times.
- Searches of an injected pulse with each planner's own thresholds (the
  jerk search of `bench/scripts/ep/ep_production_cuda.py`: 2^25 samples, 64
  bins, 128 segments, 16 runs, `target_snr` 9.2, `snr_min` 5; the same 100
  noise seeds per S/N for both planners):

  | Injected S/N | Searches that recover it, previous / descent | Runs holding it, previous / descent | Final leaves, previous / descent | Mean sweep, previous / descent |
  |---|---|---|---|---|
  | 9.2 | 76 / 77 of 100 | 0.125 / 0.119 | 453 / 316 | 0.68 / 0.63 s |
  | 10 | 97 / 96 | 0.204 / 0.193 | 1011 / 751 | 0.70 / 0.64 s |
  | 11 | 100 / 100 | 0.344 / 0.326 | 2938 / 2441 | 0.87 / 0.77 s |

  The fraction of runs that keep the pulse at the design S/N is above
  `min_pd` = 0.1 with both, and 5 % lower with the descent (its design Pd
  is 0.099 against 0.101 here), with 17-30 % fewer final leaves and 7-11 %
  faster sweeps. This plan has one region of 64 bins.
- The same harness at 512 bins (20 Hz pulse, jerk search 19-21 Hz, 2^25
  samples, 16 runs, EP default candidate reduction, the same 50 noise seeds
  per S/N; plan Pd 0.1216 previous, 0.1005 descent):

  | Injected S/N | Searches that recover it, previous / descent | Runs holding it, previous / descent | Paired difference | Final leaves, previous / descent |
  |---|---|---|---|---|
  | 9.2 | 50 / 49 of 50 | 0.320 / 0.253 | -0.068 +/- 0.011 | 98 / 91 |
  | 10 | 50 / 50 | 0.481 / 0.406 | -0.075 +/- 0.011 | 101 / 94 |

  The fraction of runs that hold the pulse falls by 16-21 %, as the ratio of
  the design Pd predicts (0.1005 / 0.1216 = 0.83): above 128 bins the
  previous design delivered Pd well above `min_pd` by accident, and the
  descent removes that margin. The injected S/N of this simulator is not the
  design S/N at 512 bins, so compare the two planners only.
- Price of forcing a floor of 7 on the three production plans (cold floored
  against free, judged at equal Pd on the same regions): L 1.032 (1.002-1.091;
  free last thresholds 5.82-6.89), 1.016 (1.000-1.037; free 6.55-7.07) and
  1.002 (0.983-1.040; free 7.08-7.88, inactive on 8 of 10 regions).

Kept:

- Multi-scale block moves. Single-stage moves stall 5-8 % above `robust` at
  `min_pd` 0.02: consecutive late stages score nearly the same fold and one
  threshold cannot move alone.
- 2^15 signal trials for the descent (2^14 costs 2-7 % of C); 4096-8192
  noise trials per genealogy with the block sizes limited in the late
  builds: 0.5 % above builds of 2^14-2^15 trials with every block size
  (geometric mean over 9 cases, within the seed scatter) at 270 ms instead
  of 720 ms.
- Restricting the backward subtree pass to the trials above their own
  threshold whose ancestry passes the lowest threshold a move can reach.
  Restricting it to the currently alive trials is wrong: a trial a downward
  move revives loses its subtree.

The converged schedules are smooth in z-scores, with no structure following
the branching of single stages (9 patterns, each solved and then warm
started twice): a fall from 1.2-2.6 at the first stage to about 0 within 4-8
stages, a flat minimum of -0.5 to 0.2 up to stage 16-24 (the threshold sits
at the mean signal score there), then a near-linear rise to 1.6-1.8 at the
last stage.

Do not retry:

| Idea | Why not |
|---|---|
| Hat-shaped block moves (the step largest at the middle of a block, falling to its edges) because the solutions are smooth | In place of the flat moves for blocks of 4 stages or more: C 1.011 of the flat moves' (9 cases × 3 seeds, 1.09 at a target of 0.02). In addition to them: 0.997 for 30 % more time. |
| More fine passes from the default start (four passes at margin 0.2, step 0.05 instead of two) | C 1.003 at 290 ms instead of 230 ms. What the warm chains gain (2-4 %) is not reached by repeating the fine passes on one start. |
| Importance-sampled noise bank (a mixture of boxcar-template tilts, weights instead of resampling) | Agrees with the resampling simulation within 1 % on C, but the effective sample size is about 10^-3 of the trials and falls as nbins^-1.5; a tilt along the pulse profile was worse (10^-4, biased late populations). The genealogy gives the same reuse of one noise sample without weights. |
| `improved` as the initialiser of the descent | Saves about two genealogy builds (6-22 ms) and costs a search of 24-142 ms. The start a fixed number of signal standard deviations below the mean signal score needs no search. |
| Sorting the genealogy by parent, or a branch-free scatter, for the backward pass | The scatter-add costs about 4 ns per element either way. |
| A candidate list for the backward pass when the reference is far below the schedule | With the reference a full margin below, nearly every trial is listed; no gain. |
| Less memory traffic in the genealogy passes: 16-bit parents and survivor lists, no separate "alive before the stage" buffer (14 MB to 9 MB per forward pass) | Bit-identical schedules, `solve()` 124 ms against 125 ms at 16 bins, 145 against 149 at 64, 215 against 227 at 128: the passes are not bound by memory. |
| More host threads for the descent | Only the block deltas run in parallel; the forward and backward passes are sequential over stages, and idle threads spinning between the parallel loops slow them. |
| Smooth-min differentiable genealogy (log-sum-exp survival margin on the noise genealogy and signal sample, optimised with L-BFGS-B and augmented Lagrangian) | C 1.108 of the block coordinate descent on the 127-stage production pattern (6.8 s on CPU against 0.08 s for the host moves): per-element exp/log evaluations and flat gradients on sparse late-stage survivor trees lose both time and cost to exact finite-step block counts. |
| Strided outer-trial OpenMP transpose and OpenMP `pin_shift` / `prepare_pass` with early break in `SignalSample` | Reading `stage_major[s * n + t]` with `n = 2^15` (128 KB stride) caused L1/TLB set-aliasing (+35-45 ms), the inner `if (m <= -1.0F) break` prevented AVX vectorisation, and short OpenMP regions slowed 16-64 bin solves by 40-50 % (0.21-0.27 s against 0.13-0.17 s with a 64 × 32 cache-blocked serial transpose and vectorised active-trial `pin_shift`). |
