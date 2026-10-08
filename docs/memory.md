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
and how `EPRegionPlanner` uses it. It covers the CPU sweep only: the GPU EP sweep
does not plan its memory this way yet.

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
- **FFA engine.** Each chunk's FFA engine (plan, brute-fold tables) is freed
  before pruning starts; it is small and not modelled.

## Reserve

The reserve covers what cannot be counted deterministically: heap
fragmentation and allocator arenas, thread stacks, small per-task objects (FFA
and FFTW plans, logging), and the interpreter or application around the
library. It is a fixed amount rather than a fraction of the limit, so large
limits stay usable and small limits stay feasible.

`bench/scripts/measure_ep_peak_rss.py` runs sweeps in fresh subprocesses and
compares the peak RSS with the model. On macOS (Apple silicon), for 1 to 8
threads, 32 bins, time and Fourier folds and `nsamps` from 2^16 to 2^18, the
residual (`peak RSS - RSS after import - model`) was a flat 0.145-0.153 GB,
growing by under 1 MB per thread. The model explains the rest, so the reserve
stays a fixed 0.5 GB with no per-thread term. Re-run the script when the
allocations change, and on Linux before relying on very large thread counts.

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

The plan cache (`plan_cache_file`) stores the model inputs next to the chunks:
`ep_memory_model_version` (`kEPMemoryModelVersion`), `batch_size`, `n_workers`,
`harvest_enabled`, `max_harvests` and `harvest_store_folds`. Loading rejects a
cache whose inputs differ, and recomputes the peak and rejects a plan that no
longer fits. Bump `kEPMemoryModelVersion` with every change to the formulas, and
the cache version in `ep_regions.cpp` with every change to the file layout.
