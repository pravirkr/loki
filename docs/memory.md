# Memory planning

`FFAFreqSweep` and `EPFreqSweep` split the frequency band so that the process
stays under `PulsarSearchConfig::max_process_memory_gb`. Both planners count the
large buffers of the sweep exactly, from the same sizes the engines allocate,
and compare them with an effective limit:

```text
effective limit = max_process_memory_gb - kUnmodelledReserveGB   (0.5 GB)
```

`kUnmodelledReserveGB` lives in `lib/algorithms/planner_memory.hpp` and is shared
by both planners. This page describes the EP model (`lib/algorithms/ep_memory.hpp`)
and how `EPRegionPlanner` uses it. The model has two policies, for the CPU and
the CUDA sweep (see [CUDA backend](#cuda-backend)); everything below is the CPU
policy unless it says otherwise.

## What the model counts

The sweep processes chunks in order. Contiguous chunks with the same `nbins`
form a group, and each group allocates its per-worker buffers once, sized from
the group's maxima, and frees them before the next group. The peak of the sweep
is the worst group:

```text
peak = n_workers * thread_bytes(group) + shared_bytes + input_bytes
```

| Term | Contents | Lifetime |
|---|---|---|
| `thread_bytes` | `ep_workspace_bytes` (world tree, prune and branch scratch, FFA seeds) + `ep_irfft_scratch_bytes` (Fourier folds only) + `ep_harvest_bytes` | one group |
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
Host memory is never checked (it is small: the inputs and the HDF5 buffers).
As for `FFAFreqSweep`, the plan is fitted to

```text
limit = min(max_process_memory_gb, free device memory - kDeviceReserveGB)   (1 GB)
effective limit = limit - kUnmodelledReserveGB                              (0.5 GB)
```

`kDeviceReserveGB` (`planner_memory.hpp`) covers the CUDA context, kernel
images and what other processes allocate meanwhile. `EPRegionStats::
get_memory_limit_gb()` (`memory_limit_gb` in Python) is the limit the plan was
fitted to.

| Term | Contents | Lifetime |
|---|---|---|
| group workspace | `ep_cuda_ep_workspace_bytes` (`WorldTreeCUDA`, `PruneWorkspaceCUDA`, `BranchingWorkspaceCUDA`, `CUBScratchArena`, seeds, segment coordinates) + `ep_cuda_irfft_scratch_bytes` (the Fourier scratch; a two-element stand-in for float folds) | one group |
| shared | `ep_cuda_shared_bytes`: FFA workspace fold buffer and coordinates, device output fold buffer | whole sweep |
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
- The threshold scheme is the CUDA `DynamicThresholdScheme`. It is
  reproducible for a given seed and batch size, and differs statistically from
  the CPU scheme, so plans are not interchangeable between backends.
- cuFFT work areas and the few small FFA arrays (counts, offsets, limits) are
  not in the exact model. `kDeviceReserveGB` (1 GB) is what the GPU planners
  leave for the CUDA context, kernel images and those work areas, and the
  0.5 GB unmodelled reserve is a second margin.
- `PruneRFIConfig` (pulsar mask, harvesting, impulsive veto) is not implemented
  on CUDA: an active one is rejected with `std::invalid_argument`.

At the start of `execute`, before the group workspace is allocated, the engine
checks that the device still has room for that workspace and the transient FFA
scratch (the shared buffers are already allocated).

## Reserve

The reserve covers what cannot be counted deterministically: heap
fragmentation and allocator arenas, thread stacks, small per-task objects (FFA
and FFTW plans, logging), and the interpreter or application around the
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

`EPRegionPlanner` splits each FFA region into chunks, growing each chunk until
the sweep would exceed the effective limit with it (bisection on the width).
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

## Runtime checks

`EPFreqSweep` checks the plan as it runs:

- after allocating the shared FFA buffers, their size must equal the model;
- before each group, the model's total must fit the effective limit (a
  `std::runtime_error` otherwise, e.g. for a hand-edited plan);
- after allocating a group's workspaces, their size must not exceed the model
  (a `std::logic_error`: the model has drifted from the allocations).

## Plan cache

The plan cache (`plan_cache_file`) is version 1.4.0. It stores the model inputs
next to the chunks: `ep_memory_model_version` (`kEPMemoryModelVersion`,
currently 3), `batch_size`, `n_workers`, `harvest_enabled`, `max_harvests` and
`harvest_store_folds`, and also `ep_backend` (`cpu` or `cuda`),
`ep_memory_model_kind` and `dts_batch_size` (the batch size of the threshold
simulation). `device_arch` (`sm_89`, or `host` for a CPU plan) is recorded and
not checked. A cache written for the other backend is rejected: the model and
the threshold scheme differ (a plan is cheap next to the sweep, so re-plan).
Loading rejects a cache whose inputs differ, skips the `nthreads` check for a
CUDA plan, and recomputes the peak with the current policy against the current
limit. Bump `kEPMemoryModelVersion` with every change to the formulas, and the
cache version in `ep_regions.cpp` with every change to the file layout.
