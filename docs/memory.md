# Memory planning

`FFAFreqSweep` and `EPFreqSweep` split the frequency band so that the process
stays under `PulsarSearchConfig::max_process_memory_gb`. Both planners count the
large buffers of the sweep exactly, from the same sizes the engines allocate,
and compare them with an effective limit:

```text
effective limit = max_process_memory_gb - kUnmodelledReserveGB   (0.5 GB)
```

`kUnmodelledReserveGB` lives in `lib/algorithms/planner_memory.hpp` and is shared
by both planners. Candidate-enabled CPU sweeps first reserve
`candidates.memory_gb` from the configured process cap, then apply this planner
limit to the remaining search budget. CUDA uses a separate host candidate
budget. See [Candidate postprocessing](#candidate-postprocessing).

**Units.** Every memory limit, getter and log in Loki counts a GB as 2^30
bytes (`utils::kBytesPerGB`, `lib/detail/utils.hpp`), the convention of
Slurm's `--mem`, `top` and `htop`, and labels it "GB". `max_process_memory_gb
= 16` (`--memory-gb 16`) is the same amount as a Slurm allocation of
`--mem=16G`. That allocation must also hold what the reserve covers, so ask
Slurm for at least `--memory-gb`; the plan never exceeds `--memory-gb` minus
the reserve.

This page describes the EP model (`lib/algorithms/ep_memory.hpp`)
and how `EPRegionPlanner` uses it. The model has two policies, for the CPU and
the CUDA sweep (see [CUDA backend](#cuda-backend)); everything below is the CPU
policy unless it says otherwise.

`snr_min` is the minimum candidate S/N for the common HDF5 output of
FFA and candidate-enabled Taylor EP searches, default 7.0. It is defined in
`FFASearchConfig` and inherited by `EPSearchConfig` (TOML
`[search].snr_min`, CLI `--snr-min`, Python `snr_min`). FFA applies it to
candidate extraction. Taylor EP applies it to the final reintegrated scores
after independent runs finish and merge;
diagnostic-only EP output retains all final leaves.

`EPSearchConfig::target_snr` is an entropic information limit used only for EP
multipass and pipeline threshold design, independent of `snr_min` (TOML
`[ep].target_snr`, CLI `--target-snr`, Python `target_snr`). An explicit positive finite number overrides the default.

EP rejects frequency-only searches (use FFA) and `nbins_max > 1024` at
every entry point (`EPRegionPlanner`, `EPMultiPass`, `EPFreqSweep`, the EP
TOML), before any storage is allocated. `PulsarSearchConfig` itself accepts
both, because it also configures FFA searches.

Chunking is automatic. The sweep first forms regions for the required bin
counts (never merged), then subdivides each region in frequency until every
chunk and its shared workspace group fit the effective memory budget. Each
region's thresholds are designed once, over its whole band, by
`DynamicThresholdScheme` in mode `shared` on both backends (signal trials
common to every path, probability bins anchored at `min_pd`, the survivors of
the last stage weighted into the cost): the search, then `solve()`, which
selects, tightens and repairs the schedule on fresh samples so that its
measured Pd is `min_pd` (see `include/loki/detection/thresholds.hpp`). Every
chunk of the region runs that schedule. The chunk's `pd` and `pd_se` are the
design's measurement at the true `nbins`: the branching only multiplies the
leaves of each stage (`State::gen_next`), so neither a schedule's Pd nor its
pass rates depend on it. A design below 0.9 `min_pd` is kept with a warning.
One design per region bounds planning time: a design at 1024 bins costs
about 20 s on CUDA, mostly the refit's evaluations. Chunks below the top of a
region run a schedule designed for the region's larger branching, which is
valid (same Pd) but not the cheapest for them.

Memory and runtime of a design are bounded. Up to 128 bins the search runs at
the true `nbins` when its pools fit. Above that, or when they do not fit, it
runs at 64 bins (`kDtsNbinsSim`, raised so that the reference pulse spans at
least four bins) and `refit_scheme` re-fits the schedule at the true `nbins`:
a common upward threshold shift, scanned on fresh samples, then tightening,
repair and an independent report. The reason is that a maximum over more
boxcar templates has a heavier noise tail (about `nbins` times more
independent phase positions), so a schedule found at 64 bins is too
permissive at 256 to 1024 bins (cost per detection 1.1 to 3.3 times a native
search's; `bench/scripts/schemes/nbins_proxy_study.py`); the shift brings that
to about 1.0 to 1.05 in the same tests and the refit always reports the Pd at
the true `nbins`. The refit's sample sizes follow the memory limit (a quarter
of it, at most 4 GB: 24 B per trial and bin for a signal bank, 12 B for a noise
evaluation), and the search budget steps down (1024 trials / 40 probability
bins, 512/40, 512/20, 256/20, 128/16) until its pools fit; only a limit below
the last step fails, with a message. The search pools are checked against the
limit on their own, because they are freed before the EP search. The designs
run once per plan and are kept in the plan cache.

A chunk is sized from its own design's final noise evaluation (2^15 trials
natively, the refit's noise sample otherwise): see [Planning](#planning). It
first narrows chunks; only when the minimum chunk cannot fit does it reduce
the parent batch (CUDA). If even that cannot fit, planning fails with a
required-memory diagnostic.

## What the model counts

The sweep processes chunks in order. Contiguous chunks with the same `nbins`
form a group, and each group allocates its per-worker buffers once, sized from
the group's maxima, and frees them before the next group. The peak of the sweep
is the worst group:

```text
peak = n_workers * thread_bytes(group) + shared_bytes + input_bytes
```

| Term | Contents | Lifetime |
| --- | --- | --- |
| `thread_bytes` | `ep_workspace_bytes` (world tree, prune and branch scratch, FFA seeds) + `ep_harvest_bytes`. The pruning functors score in tiles of 16 leaves and hold no batch-sized scoring scratch | one group |
| `shared_bytes` | FFA workspace fold buffer and coordinates + output fold buffer, sized from the global maxima of `buffer_size` and `coord_size` | whole sweep |
| `input_bytes` | `ts_e` and `ts_v`, `2 * nsamps * sizeof(float)` | whole sweep |

The formulas are in `ep_memory.hpp`; every term mirrors an allocation, and the
tests in `tests/cpp/internal/ep_memory_t.cpp` compare them with the real
`EPWorkspaceCPU`, FFA buffers and `HarvestBuffer` byte for byte.

- **Workers.** `n_workers = min(nthreads, number of runs)`, where the runs are
  `n_runs` or `ref_segs.size()`. Fewer runs than threads need fewer workspaces,
  so the plan gets wider chunks. Pass the same value to `EPRegionPlanner`
  (`n_workers`) when planning outside the sweep.
- **Inputs.** The inputs are counted whoever owns them (a numpy array, a
  `TimeSeries`), because the process holds them for the whole sweep. From
  Python, float64 arrays are converted to a float32 copy; the original array is
  the caller's.
- **Harvest.** With harvesting enabled (`PruneRFIConfig::has_harvest()`), each
  run reserves `max_harvests` records on its first harvest, so its store has a
  fixed size: `max_harvests * (leaf + [fold] + score + level + seg_idx + t_ref)`.
  The term is zero when harvesting is off, so default plans do not change.
- **FFA engine.** On the CPU each chunk's FFA engine (plan, brute-fold tables)
  is freed before pruning starts; it is small and not modelled. On CUDA the
  brute-fold tables are live while the group workspace is allocated, so they
  are modelled (see below).

## CUDA backend

`EPFreqSweep` with `Exec::cuda(device)` prunes the runs of a chunk one after
another on one stream, so the model has one worker whatever `nthreads` is
(`EPMemoryContext::kind == EPMemoryKind::kCuda`, `n_workers == 1`).

**Which memory.** On CUDA, `max_process_memory_gb` means device memory only.
Host memory is never checked. Result snapshots are bounded to one active write
and one waiting payload, in addition to the producer snapshot and inputs.
The model fits execution buffers. The threshold designs run before the sweep
allocates, and their search pools are checked against the limit on their
own; tail buffers and temporary peaks are covered by fixed reserves. As for `FFAFreqSweep`, the plan is fitted to

```text
limit = min(max_process_memory_gb, free device memory - kDeviceReserveGB)   (1 GB)
effective limit = limit - reserve       (0.5 GB)
```

`kDeviceReserveGB` (`planner_memory.hpp`) covers the CUDA context, kernel
images and what other processes allocate meanwhile. `EPRegionStats::
get_memory_limit_gb()` (`memory_limit_gb` in Python) is the limit the plan was
fitted to.

| Term | Contents | Lifetime |
| --- | --- | --- |
| group workspace | `ep_cuda_ep_workspace_bytes` (`WorldTreeCUDA`, `PruneWorkspaceCUDA`, `BranchingWorkspaceCUDA`, `CUBScratchArena`, seeds, segment coordinates) + `ep_cuda_irfft_scratch_bytes` (the fused pruning kernel's statistics and tile states, one 8-byte word per branched leaf at most; no IRFFT fold scratch) | one group |
| shared | `ep_cuda_shared_bytes`: logical FFA internal fold size, coordinates, and seed-output fold buffer; internal/tree overlap is subtracted once from the group total | coordinates/output: whole sweep; shared fold arena: one group |
| inputs | the device copy of `ts_e` and `ts_v`, `2 * nsamps * sizeof(float)` | whole sweep |
| FFA transient | scratch the FFA of one chunk allocates while it runs, `ep_ffa_transient_bytes`; the largest of any chunk counts | whole sweep |

The CUB temporary storage is not a formula: `ep_cuda_cub_scratch_bytes`
(`lib/cuda/ep_memory_cuda.cu`) asks CUB for the size, so the model follows the
library version. `lib/algorithms` stays free of CUDA by taking it as a function
pointer in `EPMemoryContext`. `tests/cpp/internal/ep_memory_cuda_t.cpp` builds
the real `EPWorkspaceCUDA` and `FFAWorkspaceCUDA` (time and Fourier folds) and
compares their byte counts with the model exactly, and the device free-memory
delta within allocator granularity.

Differences from the CPU policy:

- The world tree needs a capacity above `batch_size * branch_max`, so a CUDA
  chunk's `max_sugg` is raised to `ep_cuda_effective_max_sugg` (the planner and
  the engine both apply it).
- A workspace group is allocated once and its shape must equal every chunk's
  (`validate` uses `check_equal`), so the engine builds the workspace from the
  group maxima and checks each chunk fits within it.
- The threshold scheme is `DynamicThresholdScheme` in mode `shared` on both
  backends. The CUDA and CPU engines differ statistically. A cache can be
  reused on the other backend, like FFT wisdom: its batch is adapted and its
  capacity and peak are rechecked under that run's own memory limit, worker
  count and threads (which are part of the key only for a plan from the same
  backend). Plan once on the GPU, then run on the CPU with the same cache.
- The Bluestein tables (under 64 KiB per size), CUB temporaries and the few
  small FFA arrays (counts, offsets, limits) are not in the exact model.
  `kDeviceReserveGB` (1 GB) is what the GPU planners leave for the CUDA
  context, kernel images and those arrays, and the unmodelled reserve
  (0.5 GB) is a second margin.
- The preferred batch is 2^16 parents on the GPU (`kEPBatchSizeCuda`) and 1024 on the
  CPU (`kEPBatchSize`). A batch is one kernel round trip and one world-tree
  synchronisation, so the GPU needs a large one: at 1024, one run of 2^25
  samples took about 63 s, at 2^16 about 15 s. The batch is part of the memory
  model (the world tree and the prune scratch scale with `batch * branch_max`).
  If a minimum-width CUDA chunk cannot fit, planning retries with successively
  halved batches, down to one parent. It never reduces forecast capacity
  to force a fit. The chosen batch is stored per chunk
  and of the cache, which records it. On the GPU, a run that overflows the
  world tree trims at batch boundaries, so results can depend on the batch.
- Neither backend keeps FFT plans or scratch for pruning: CUDA inverts on
  chip, the CPU with the lane FFT inside its scoring tiles. The FFA stage's
  GPU transforms keep only the small Bluestein tables; the host FFT tables
  are small, immutable and shared by all threads.
- `PruneRFIConfig` (pulsar mask, harvesting, impulsive veto) is not implemented
  on CUDA: an active one is rejected with `std::invalid_argument`.

The FFA internal fold and EP tree folds use the same allocation in a sweep:
the arena is `max(FFA internal bytes, tree fold bytes)`, rather than their sum.
The FFA output folds remain separate throughout pruning. Allocation checks
count the views logically; the sweep peak subtracts the shared overlap once.
Standalone workspaces continue to own their buffers.

At the start of `execute`, before the group workspace is allocated, the engine
checks that the device still has room for that workspace and the transient FFA
scratch (the shared buffers are already allocated).

### FFA frequency sweep on CUDA

`FFAFreqSweep` on CUDA never stores the final fold. The last two levels
(one when the plan has only two) are merged on chip and scored in the same
kernel (`lib/cuda/ffa_score_cuda.cu`; Fourier folds add an on-chip inverse
FFT, `ffa_score_fourier_impl.cuh`: a direct transform for a power-of-two
`nbins` from 32, Bluestein's algorithm through power-of-two transforms for
any other `nbins` up to 1024, both on the portable block FFT
`ffa_portable_fft.cuh`; see `docs/fft.md`). Accel searches also resolve the
final level's coordinates on the fly instead of storing them. The plan is
fitted to `min(max_process_memory_gb, free device memory - kDeviceReserveGB)`
(`FFARegionPlanner(cfg, Exec)`, which `loki search ffa --dry-run` also uses),
and the planner models what the fused path allocates (`planner_memory.hpp`):

| Term | Contents |
| --- | --- |
| fold buffers | two ping-pong buffers of `gpu_sweep_buffer_size(plan)`: the largest level the sweep stores, i.e. every level but the last `gpu_fused_score_depth` ones |
| coordinates | `gpu_sweep_coord_size(plan)`: every level's resolved coordinates, minus the final level for accel searches |
| score scratch | two slots (the GPU fills one while the host copies the other) of `min(ncoords * n_widths, kGpuScoreScratchCap)` scores and indices; `kGpuScoreScratchCap = 2^24` |
| brute fold | the phase-bin map, `nfreqs * segment_len` `uint32_t` |
| inputs | `ts_e`, `ts_v` |

A chunk the fused path cannot take (more than `kGpuFusedMaxWidths = 32` widths,
a single level, or a Fourier `nbins` above 1024) is planned and run unfused,
with full buffers and an uncapped scratch. The Bluestein tables (one per
non-power-of-two `nbins`, at most 3072 complex values each) are uploaded once
per sweep and covered by the reserve. A fused chunk whose passing (profile, width) pairs exceed the scratch
slot drops the excess, counts it, and is scored again after the sweep in
profile ranges that fit. The signed fold-stride index limit no longer applies
to fused chunks (the final fold is never indexed). The small per-chunk tables
(frequencies, accel resolve tables, pinned staging) grow on demand and are
covered by the reserves.

The chunks run as a two-slot pipeline: chunk i's kernels are queued while the
host copies chunk i - 1's candidates and appends them to the result file. Every
per-chunk upload goes through pinned staging, and no per-chunk device
allocation or legacy-stream operation is left on the fused path, so the queue
never drains between chunks.

### FFA frequency sweep on the CPU

The CPU `FFAFreqSweep` stores neither the final level nor, for accel (jerk)
searches, the level before it: `FFA::execute_scored` merges both tile by tile
in per-thread scratch and scores them there (`cpu_scored_fused_levels`,
`planner_memory.hpp`). Frequency-only searches run cone bands (temporal
tiling, time and Fourier folds) and score the top band in cache, but keep the
full buffer for the bands below.

The rule is that every allocation whose size grows with the problem (samples,
trials, bins, coordinates, widths, threads) is either part of a workspace the
planner sizes or bounded by a term the planner adds; the reserve covers only
size-independent objects. The planner counts, per chunk, and sizes the sweep
by the maximum of each term over the chunks:

| Term | Contents |
| --- | --- |
| fold buffers | two ping-pong buffers of `cpu_scored_buffer_size(plan)`: the largest stored level (every level but the last two for accel searches; the full buffer for frequency-only searches). The sweep allocates both once (`FFAWorkspaceCPU::fold_internal` and the sweep's second buffer, passed as `scratch`); pages a chunk never uses are never touched |
| coordinates | `cpu_scored_coord_size(plan)`: every level's resolved coordinates, except that accel searches keep one batch of final-level rows (`kCpuFinalCoordBatchBytes`, 32 MiB) in place of the final level, which `run_final_scored` resolves batch by batch |
| brute fold | phase-run table (time domain, and Fourier plans above `nbins_min_lossy_bf`, which fold in time and transform), three times the runs while it is built; or phasor table (Fourier), see `estimate_brutefold_table_bytes` |
| scratch | `nthreads * cpu_ffa_thread_scratch_bytes(plan)` + `cpu_hit_buffer_bytes(plan)` + the phase-step table (`cpu_phase_step_table_bytes`, the largest chunk's) |
| extra | write staging and the `max_passing_candidates` accumulator (12 B per candidate) |
| inputs | `ts_e`, `ts_v` |

`FFARegionStats::get_scratch_memory_usage()` (`scratch_memory_usage` in
Python) is the scratch term; the sweep logs it on its "peak memory" line.

- **Phase steps.** A complex (Fourier) merge rotates a profile by
  `exp(-2 pi i shift / nbins)`. The steps depend only on the coordinate, so
  they are computed once per coordinate (vectorised, in blocks of 32) and
  reused by every segment and tile. The kernels compute them for one batch of
  coordinates at a time into one table, `FFAWorkspaceCPU::phase_steps`, at
  most `kPhaseStepTableBytes` (32 MiB, about 170 k steps), or less when the
  plan needs less. Cone bands run their top coordinates in batches of whole
  tiles whose steps fit the table; segment-major merges run batches of whole
  step blocks. The step blocks are those of a single pass, so the results are
  bit-identical. Before this table the steps of every level of a band were
  held at once, which exceeded the plan by up to 7.8 GB.
- **Per-thread scratch.** `cpu_ffa_thread_scratch_bytes` is an explicit upper
  bound on what one thread allocates: the largest of the cone bands (the cone
  budget, `kCpuConeScratchBytes` = 2 MiB or `LOKI_CONE_SCRATCH_KB`; three
  budgets for time-domain bands, whose segment prefix sums may be shared), the
  fused or plain brute fold (prefix sums of `bseg_brute` samples), and the
  fused final level (tile and parent rows), plus the boxcar scorer and the
  lane FFT. Its terms mirror the allocations in `kernels_cpu.cpp`,
  `ffa_cpu.cpp`, `boxcar_kernels.hpp` and `lane_fft_cpu.cpp`; change them
  together.
- **Hits.** Hits are data dependent: one per passing (profile, width). The
  engine hands them to the sweep (`detail::HitSink`) after every batch of
  final-level coordinates, and the sweep pushes them straight into the
  candidate buffer, which is flushed to disk when full. A batch is
  `kCpuHitBatchCoords` (2^16) coordinates of whole tiles, or, for accel
  searches, whole row groups (rows that share parent rows,
  `cpu_final_row_group_rows`) up to that size. The per-thread hit vectors are
  freed after each batch, so `cpu_hit_buffer_bytes` = 2 x batch x widths x
  8 B bounds them even if every profile passes. The hits keep their
  score-index order.

The CPU sweep path allocates nothing else that grows with the problem. The
audit of everything reachable from `FFAFreqSweep::execute`:

| Allocation | Where | Class |
| --- | --- | --- |
| `fold_internal`, coordinates, `phase_steps` | `FFAWorkspaceCPU` | workspace |
| second ping-pong buffer (`m_fold_time`) | sweep | workspace |
| candidate accumulator, write staging | sweep | workspace ("extra") |
| phase-run / phasor tables, run lists while built | `BruteFold`, `brute_fold_ts_complex` | brute-fold term |
| cone tiles, prefix sums (per thread or shared) | `ffa_cone_band_freq`, `ffa_complex_cone_band_freq` | per-thread bound |
| brute-fold prefix sums and rows | `brute_fold_ts`, `brute_fold_ffa_fused_freq`, `brute_fold_ts_complex` | per-thread bound |
| final tiles and parent rows | `ffa_final_scored`, `ffa_final2_scored`, `complex_final_tiles`, `ffa_complex_final2_tiles` | per-thread bound |
| `BoxcarScorer3D`, `EVTransform` (scoring, forward / inverse batches) | `ConeScoreCtx`, `FourierScoreCtx`, `lane_fft_cpu.cpp` | per-thread bound |
| hits | `ConeScoreCtx` per-thread vectors | hit term |
| tile lists, batch lists, per-chunk plan and decode tables | kernels, `FFAPlan`, `RegionDecode` | small: at most one entry per tile of a batch, or per level |
| FFT plans and Bluestein tables (one per `nbins`) | `EVTransform::get_plan` | size-independent (reserve) |

Standalone `FFA::execute_scored` (public API) collects every hit in the
caller's vector and allocates its own second fold buffer; it is not part of a
memory plan. The benchmark hook `LOKI_FUSE_LEVELS=0` stores the final level
and allocates an unplanned buffer (it logs a warning).

The workspace is allocated once, sized by the maximum of each term over all
chunks. Chunks are chosen by cost, with the memory limit as a hard cap
(`regions.cpp`, `planner_cost.hpp`):

- **Cost.** A chunk's modelled run time sums, over its own plan's levels,
  `nsegments * ncoords * 2 * nbins` times a per-stage weight (brute fold,
  stored merges, the fused level L-1, the fused final level), plus a fixed
  cost per chunk. The weights come from one-core level timing
  (`LOKI_FFA_LEVEL_TIMING`). An accel grid is designed at its chunk's highest
  frequency, so a wide chunk pays for trials its low end does not need.
- **Partition.** Each region is split on a geometric frequency grid (64 steps,
  refined when the limit admits no chunk one step wide) by dynamic
  programming: the cheapest set of chunks that each fit the limit. A
  frequency-only search minimises the chunk count instead: its grid does not
  depend on the width, and splitting would only move the grid origin.
- **Cap.** If the workspace (the per-term maxima, possibly set by different
  chunks) exceeds the limit, the per-chunk limit is lowered by the excess and
  the regions are partitioned again. The plan does not depend on the order in
  which chunks are accepted, more memory never costs more, and once memory
  stops binding the plan no longer depends on the limit.

The GPU planner keeps the widest chunk that fits (bisection on the width),
because a GPU chunk has real fixed costs.

Test hooks shrink the batches to exercise their edges (`ffa_memory_t.cpp`):
`LOKI_PHASE_TABLE_KB`, `LOKI_HIT_BATCH_COORDS`, `LOKI_FINAL_COORD_BATCH_KB`.
`tests/cpp/internal/ffa_memory_t.cpp` also checks that the planner's terms
cover the real workspace and the per-thread bounds of every chunk.

## Reserve

The reserve covers what cannot be counted deterministically: heap
fragmentation and allocator arenas, thread stacks, small per-task objects (FFA
plans, FFT tables, logging), and the interpreter or application around the
library. It is a fixed amount rather than a fraction of the limit, so large
limits stay usable and small limits stay feasible.

`bench/scripts/measure_ep_peak_rss.py` runs sweeps in fresh subprocesses and
compares the peak RSS with the model. `ru_maxrss` is bytes on macOS and KiB
on Linux; the script converts both. On macOS (Apple silicon), for 1 to 8
threads, 32 bins, time and Fourier folds and `nsamps` from 2^16 to 2^18, the
residual (`peak RSS - RSS after import - model`) was a flat 0.145-0.153 GB,
growing by under 1 MB per thread. The model explains the rest, so the reserve
stays a fixed 0.5 GB with no per-thread term. On this Linux host, one thread,
32 bins, time-domain folds and `nsamps` of 2^16 (140-142 Hz) left a residual
of 0.129 GB, inside the same reserve. Re-run the script when the allocations
change, and before relying on very large thread counts.
With `--backend cuda` the script also samples device memory
(`cudaMemGetInfo`) for the whole sweep and reports that high-water mark
against the model. On CUDA the host residual is not the quantity the limit
checks: `max_process_memory_gb` means device memory, and host memory is not
checked.

## Planning

### Parameter limits

The world tree grows with the parameter box, so the box should hold only
physical orbits. The branching of a stage is a phase drift in bins, so the
tree depends on `f * nbins` and on the derivative limits; the jerk limit
dominates over a long observation. Measured at `T_obs` = 2147 s and nbins 512
(25-29 Hz): a box of accel +/- 1000 and jerk +/- 4 needs 132 GB for a 0.01 Hz
chunk, while the circular-orbit limits of a coherent jerk search (P_orb >= 5
`T_obs`, m_c <= 10, m_p >= 1.2: accel +/- 499, jerk +/- 0.29) need 2.4 GB, and
the whole 4 Hz band plans into 15 GB. Each chunk's FFA band is also padded by
the drift these limits allow (`calculate_max_drift`).

Give either explicit limits (`search.acc_*`, `search.jerk_*`) or a
`[search.orbit]` table, whose limits follow from `search::circular_param_limits`
(pyloki's `from_circular`; `libloki.configs.circular_param_limits` in Python):

```toml
[search.orbit]
order = "jerk"          # "accel" | "jerk" | "snap"
p_orb_min_tobs = 5.0    # or p_orb_min in seconds
m_c_max = 10.0
m_p_min = 1.2
```

`f_min` and `f_max` are then intrinsic spin frequencies, widened by `v/c`.
Explicit limits with a jerk are reported as the fastest circular orbit they
reach (`search::implied_circular_orbit`), with a warning when it is shorter
than the observation. `[ep]` `p_orb_min`, `m_c_max` and `m_p_min` are a
different setting: the leaf validation of the circular pruning basis.

### Ring capacity

The population of a run is forecast from its region's design: `L_0 = n_seeds`
(the FFA coordinates, exact) and, after stage `s`,
`L_s = ceil(n_seeds * c_s * m_s)`, where `c_s` is the noise survivors per seed
(`State::complexity`) and `m_s` a margin for the Monte Carlo error of the
forecast (issue #15, section 7). `c_s` is a product of measured pass rates
`s_j`, each with binomial relative error `sqrt((1 - s_j) / (N s_j))` on `N`
noise trials per stage, so

```text
eps_s = sqrt(sum_{j <= s} (1 - s_j) / (N s_j)),    m_s = 1 + z eps_s,    z = 2.33
```

and a ring sized with it falls short of the true mean about 1 % of the time
(`ep_ring_margin`, `kEPRingMarginZ`). With `N = 2^15` and production schedules
`m_s` is about 1.1 to 1.2. A stage whose evaluation prunes every noise trial
is counted at the floor `s_j = 1 / N` (one trial kept): it adds about 1 to
`eps_s^2`, so `m_s >= 1 + z` from that stage on. That over-allocates, on the
safe side, and only where the forecast population is already tiny. During
stage `s` the ring holds the unconsumed
parents and the survivors written so far, so it is sized from the largest
adjacent sum, plus one branched batch, rounded up to a multiple of 1024
leaves (`kEPRingGranularity`; neither ring needs a power of two):

```text
live     = max_s (L_{s-1} + L_s)
max_sugg = round_up(live + batch_size * branch_max + 1, 1024)
```

The margin covers simulation error only. Data that departs from the noise
model (RFI, a bright source) can exceed it; the world tree then trims
survivors at batch boundaries (a warning, and a loss of sensitivity). It
never allocates more: memory stays within the plan. Logs and the CLI print
`max_sugg` and the forecast populations as powers of two (`2^21.3`); the
cache and the Python stats keep the exact counts.
`max_passing_candidates` is an FFA output-buffer setting and does not control
EP.

### Width search

The minimum nominal frequency chunk width is currently 0.01 Hz (or the
remaining band if smaller), plus drift padding. A minimum-chunk failure refers
to this permitted width; it does not establish that every finer grid-aligned
chunk would be infeasible.

`EPRegionPlanner` designs each FFA region once over its whole band, then
splits the region into chunks from its top frequency down, growing each chunk
until the sweep would exceed the effective limit with it (bisection on the
width, with probes beyond the bracket since branching changes discretely with
the band). A probe simulates nothing. Its population is the region design's
on the probe's own branching pattern `bp'`:

```text
c'_s = c_s * prod_{j <= s} bp'_j / bp_j
```

This is exact, not an approximation. A schedule's pass rates `s_j` are
survivor fractions of the simulated trials, which never see the branching, so
the region design's evaluation is also the chunk's. The ring margin carries
over with them. The design is memoised by branching pattern, across replans
and batch retries.
Because the shared FFA buffers are sized from the global maxima, a chunk placed
late can enlarge them after earlier groups were already fitted:

1. **First pass.** Plan every region with the shared size growing as chunks are
   placed. If the final peak fits, this is the plan.
2. **Replan.** Otherwise, plan again with the shared size fixed at the first
   pass's maximum, so every group is fitted next to the final shared buffers.
3. **Cap search.** If a minimum-width chunk does not fit next to that size,
   bisect a scale `s` in (0, 1) on the fixed shared size (8 passes) and keep the
   largest `s` whose pass succeeds. A smaller size leaves earlier groups more
   room and makes the chunks that set the size narrower. If no scale works,
   the planner throws the original "Cannot fit minimum viable chunk" error.

After a replan the planner checks that the final peak is within the limit.
`ChunkPlan::replanned` and `shared_cap_scale` record which path ran. Chunk
evaluations (each builds an `FFAPlan`) are memoised across passes.

`EPChunkConfig::chunk_memory_gb` and `EPChunkStats::memory_gb` are the memory of
one chunk run alone. `EPRegionStats::get_max_memory_gb()` is the sweep peak
above, the value compared with the limit.

The CUDA sweep writes owned host snapshots directly into the final HDF5
chunk/run layout using one writer thread. Successful completion waits for all
writes and propagates errors; interrupted files are not resumable. The CLI
caches plans in `<outdir>/<prefix>_ep_plan.h5` by default. Set
`output.use_plan_cache = false` or pass `--no-plan-cache` to disable automatic
caching. Explicit cache paths retain strict validation; obsolete implicit
caches are regenerated.

## Candidate postprocessing

EP candidate publication also reserves `candidates.memory_gb` before CPU
planning when the candidate-enabled sweep overload is used. CLI dry runs,
plan-only runs, cache preparation and execution use this same reservation.
The legacy diagnostic-only API does not reserve a candidate index. Its CPU
memory model now additionally covers a uint32 winning-width result per final
leaf, up to `4 * max_sugg` bytes per worker. Memory-model version 12 invalidates
older caches so this term cannot be omitted.

CUDA recovers final widths in batches using existing device index/score
scratch, so it requires no additional problem-sized device allocation. The
owned host result snapshot has one additional uint32 per leaf. At most three
snapshots can coexist (the current producer, a pending payload and the active
write), bounding this addition by `3 * 4 * capacity` bytes. As before, CUDA's
search cap bounds device memory, and native host snapshots are outside it.
The candidate graph has its separate host budget and starts after the writer
has drained. EP source/provenance indexing is charged against that budget;
candidate records are read from each merged run in bounded batches.

FFA sweeps reserve `candidates.memory_gb` for host-side candidate processing
before fitting CPU chunks; set it below `max_process_memory_gb`. CUDA's search
cap remains a device limit, and this candidate budget applies separately to
host memory. The raw writer also stages uint64 detection identities (8 bytes
per write-batch row), included in the host sweep memory estimate.

Postprocessing sorts bounded candidate runs on disk and merges two runs at a
time. Transitive grouping keeps a union-find node and ordered neighbor indexes
for each distinct full Taylor track, including nonrepresentatives that may
bridge components. Exact-track width duplicates share a node; its widest
recorded duty cycle preserves all possible graph edges. Nodes, both index
trees, final membership remapping and representative vectors have a conservative
memory charge including allocation growth. No dense pairwise matrix is built.
Anchored mode charges its representative vector and frequency index instead.
The raw membership column is remapped in bounded batches after transitive
components are complete, so later bridges cannot leave stale cluster IDs. Exhaustion is an
explicit error, never a silent candidate cap. HDF5/runtime/allocator overhead
is additional to accounted buffers. Temporary disk use can exceed the final
HDF5 size even when raw retention is disabled: the sort keeps 184 bytes per
detection in two runs at its peak, about 37 GB for 10^8 detections (see
[Candidate products](candidates.md) for the estimate).
