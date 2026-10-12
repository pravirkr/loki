# FFTs in loki

This page covers where loki transforms profiles between the time and Fourier
domains, which library does each transform, and how the fused on-chip inverse
FFT of the CUDA FFA sweep works. Read it before changing
`lib/cuda/fft_cuda.cu`, `lib/cuda/ffa_portable_fft.cuh` or
`lib/cuda/ffa_score_fourier_*`.

## Conventions

- A Fourier fold stores `nbins_f = nbins / 2 + 1` harmonics per profile for
  each of `e` (signal) and `v` (variance). A profile's row is `2 * nbins_f`
  complex values: `e` first, then `v`.
- Inverse transforms are unnormalized C2R. The `1 / nbins` is applied later:
  the scorers take it as `fold_scale`.
- C2R treats the input as Hermitian, so the imaginary parts of the DC
  harmonic and (for even `nbins`) the Nyquist harmonic are ignored. cuFFT and
  FFTW both behave this way. The on-chip transforms below zero those parts
  explicitly.
- `nbins` is any integer `>= 2`. Nothing in loki requires a power of two. The
  FFT code must keep it that way.

## Who does which transform

| Where | Library | Notes |
| --- | --- | --- |
| CPU (`lib/cpu/fft_cpu.cpp`) | FFTW (`FFTWManager`) | Plans cached per size. |
| GPU, batched (`lib/cuda/fft_cuda.cu`) | cuFFT (`CUFFTManager`) | Lossy brute fold (rFFT), unfused FFA levels, EP. |
| GPU, fused FFA Fourier scorer | `PortableFFT` (`ffa_portable_fft.cuh`) | On chip, inside the final-level kernel. |

### cuFFT plan cache

A cuFFT plan is tied to its batch size, and every FFA chunk has a different
batch size. Building a plan for each chunk cost seconds on large sweeps.
`CUFFTManager` therefore has two modes.

- **Prepared sizes** (`prepare_plans`, used by the FFA sweep). Each size gets
  two plans per direction: one for `max_batch` rows and one for a granule of
  1024 rows. `execute_chunked` runs a call as full `max_batch` blocks, then
  full granules. The rows that remain are copied into a granule-sized scratch
  buffer and transformed there. Rows are independent transforms, so this
  split does not change the result.
- **Exact sizes** (EP). One plan per distinct batch size, as before.

## Fused Fourier final level (CUDA FFA sweep)

`FFAFreqSweep` on CUDA never stores the final fold (see `docs/memory.md`).
The kernel `kernel_final_score_c2c` (`ffa_score_fourier_impl.cuh`) does four
steps:

1. merges the last one or two complex FFA levels from their parents in global
   memory;
2. inverts each final profile on chip;
3. writes the real `e` and `v` rows to shared memory;
4. scores them with the same boxcar scorers as the time-domain path
   (`ffa_score_device.cuh`).

The host entry points are in `ffa_score_fourier_{,computed_,freq_}cuda.cu`:
one translation unit per coordinate source, so the template instantiations
compile in parallel.

### One complex transform for e and v

`e` and `v` are real, so their spectra are Hermitian. The kernel extends both
half spectra to all `N` harmonics, dropping the imaginary parts of DC and
Nyquist. It then forms `X_e + i X_v` and runs one complex inverse DFT. The
result is `z = e + i v`, so a single transform yields both rows. The inverse
uses the forward FFT: `IDFT(x) = conj(FFT(conj(x)))`.

### Direct transform and Bluestein

- **Power-of-two `nbins` in [32, 1024]:** the inverse is a direct `M = N`
  point complex FFT.
- **Any other `nbins` in [2, 1024]:** the inverse uses Bluestein's algorithm.
  - Since `k n = (k^2 + n^2 - (n - k)^2) / 2`:

    ```python

    x[n] = c[n] * sum_k (X[k] c[k]) conj(c[n - k]),   c[j] = exp(i pi j^2 / N)
    ```

    This is a circular convolution. It is exact when its length
    `M >= 2N - 1`. The kernel uses `M = max(32, bit_ceil(2N - 1))`, a power of
    two up to 2048.
  - Per profile it runs:
    1. `a = (X_e + i X_v) * c`;
    2. `A = FFT_M(a)`;
    3. `B = conj(A * K)`, where `K` is the table below;
    4. `D = FFT_M(B)`;
    5. `z = conj(D) * c`.
  - The table comes from `ffa_fourier_bluestein_table(nbins)`, built on the
    host in double precision. It holds `c[0..N)`, then
    `K = FFT_M(conj(c[|j|]) wrapped circularly) / M`, at most 3072 complex
    floats.
  - The workspace (`FFAWorkspaceCUDA::bluestein_table`) keeps one table per
    `nbins`. The sweep uploads every chunk's table before its pipeline
    starts. The kernel reads the table through the L1 cache.
  - The chirp phase is computed from `j^2 mod 2N` in integers, so it is exact
    for large `j`.
  - Bluestein does roughly 2.5-5x the arithmetic of a direct transform of a
    similar size. It is still far faster than the unfused cuFFT route, which
    writes the whole complex fold to global memory and reads it back.
- **`nbins > 1024`:** the chunk is planned unfused and goes through cuFFT
  (`gpu_fused_score_applies` in `planner_memory.hpp`).

### Kernel shape and occupancy

- Each block handles `kPB` profiles, in rounds of `F` (one transform per
  profile). Every round:
  1. merges the harmonics into per-transform half-spectrum buffers;
  2. builds the FFT input from them;
  3. runs the transform(s);
  4. writes the rows.
- After the last round, warps score the stored rows. The thread scorer is
  used when `nbins <= 64` with at least 2^16 profiles, matching
  `score_and_filter_cuda_d`.
- Barriers inside a round cover only the threads of one transform
  (`FFT::sync`). A transform of 32 threads or fewer uses a warp barrier. The
  block barrier comes after the rounds, before scoring.
- The per-block row budget is 12 KiB (`RowLayout::kMaxRowBytes`). These
  kernels are latency bound, and occupancy is limited by shared memory per
  thread. Cutting the budget from 40 KiB to 12 KiB doubled the achieved
  occupancy of the dominant Bluestein kernel (17% to 33%).
- Layouts: `PortableC2C<M>` uses 128 threads per block, with 4 elements per
  thread for `M <= 128`, 8 for `M` up to 1024, and 16 for 2048.
  - 64-thread blocks were 20% slower, because they doubled the shared memory
    per thread.
  - 256-thread blocks were mixed.
- The largest configuration uses about 33 KiB of shared memory, so every
  kernel fits the 48 KiB per-block limit of pre-Volta GPUs.
  `cuda_utils::set_kernel_dynamic_smem` fails with a clear message if a
  future configuration does not fit the device.

## Fused pruning step (CUDA EP)

The EP prunes in batches of branched leaves. One kernel per batch replaces
the shift-add kernel, (for Fourier folds) the copy that cuFFT C2R
overwrites and the C2R itself, and the scorer:
`kernel_ep_shift_add_score_c2c` (`ep_score_fourier_cuda.cu`) for complex
folds, `kernel_ep_shift_add_score_time` (`ep_score_time_cuda.cu`) for
time-domain folds. Their block bookkeeping is shared
(`ep_score_device.cuh`). Per leaf they:

1. add the parent fold (world tree) and the FFA segment fold shifted by the
   leaf's phase. Fourier folds are rotated harmonic by harmonic
   (`ep_shift_add_harmonic`, `ep_fourier_device.cuh`, which
   `kernel_shift_add_linear_complex` also uses). Time-domain folds use the
   whole-bin shift of `kernel_shift_add_linear`. Either way the merged
   folds are bit-identical to the unfused path;
2. (Fourier) invert `X_e + i X_v` on chip, direct or Bluestein, as the FFA
   kernel does;
3. score the profile with the same scorers and the same regime rule as
   `score_and_filter_max_cuda_d` (the maximum over the widths), and
   compare it with the threshold. Any number of widths is supported: the
   width constants live in dynamic shared memory, and the widths are scored
   in register groups of 32 after one prefix sum (the maximum is
   order-free; `kEPWidthGroup` = 8 registers per group, which was the fastest
   of 4, 8, 16 and 32 at every nbins). With `use_boxcar_kadane` the Kadane scorer
   (`kadane_device.cuh`) replaces the boxcar: see below;
4. count the passing leaves and the min/max score of the valid ones
   (block atomics; the min/max use an order-preserving key, so they are
   exact).

The Fourier kernel keeps the FFA kernel's block shape, rounds and row
budget, and loads the next round's harmonics into registers before the
current round's transform. The time-domain kernel has 128 threads; it
forms the e and v rows of 32 leaves (thread scorer) or of up to 16 leaves
in 12 KiB (warp scorer) in shared memory.

**Kadane scorer.** `ep_score_kadane_rows` (`ep_score_device.cuh`) runs
`kadane_group_max_snr` (`kadane_device.cuh`), the scan of
`score_and_filter_max_cuda_kadane_d`, on the block's rows:

- both fused kernels keep one normalized row per leaf, `e * rsqrt(v)` as the
  Kadane kernels compute it (Fourier: `(z.re / n) * rsqrt(z.im / n)` after
  the on-chip inverse, the normalized C2R's arithmetic), so a block holds up
  to 32 leaves in 12 KiB (`RowLayout`'s Kadane layout);
- a warp scores `32 / nbiases` leaves at once, one lane per bias, each lane
  running one sequential scan (maximum and minimum subarray, for the wrapped
  candidate) and the biases combined by shuffles. `nbiases` is 2 or 3 and is
  read from the device pointer `EPScoreArgs::kadane_biases`, so a future
  table of biases per nbins is just another pointer;
- the standalone kernels use the same device functions
  (`kernel_kadane_warp_multiprofile`, and `kernel_kadane_segment_parallel`
  for nbins above 256), and their outputs are unchanged bit for bit.

The scan is not the cost of the fused kernel: skipping it entirely saves 16%
of the kernel at nbins 64 and nothing measurable from nbins 256. The
standalone kernels are at the DRAM bandwidth of the GPU for nbins from about
100 (the 134 MB of 2^17 profiles of 128 bins takes 212 us), so the gain of
the fused path is the removed round trip through the irfft scratch (2.4x on
the 2^25 jerk benchmark).

**Direct writes.** When even an all-passing batch fits the tree's free
slots (`add_batch_scattered`'s fast path), the kernel also writes the
survivors into the tree. Every valid leaf is resolved and transformed
first (one kernel), and the survivors are ranked in leaf order (a decoupled
look-back over ticketed tiles), the order of the `DeviceSelect::Flagged`
compaction it replaces. Each survivor's leaf, merged fold (merged again
from the L2-resident rows) and score go to slot `write_head + rank`. Other
batches use the staged path (the folds in `branched_folds_d`, then
`add_batch_scattered`, which may raise the threshold and trim the tree).

**Pipelining.** A direct-mode batch only enqueues work. Its count and min/max
are collected after the next batch's branch is enqueued, so the stream sync
the branch needs for its count also covers the fused kernel.

**Seeds and ascend (Fourier).** The seed scores and the ascend scores (the
final S/N in the result file) come from the same kernel in rows mode
(`ep_score_rows_fourier_cuda`): each row is inverted and scored as it is,
with no FFA term, mask or statistics.

**When it applies** (`ep_cuda_fused_batch`, `ep_cuda_fused_rows`,
`ep_memory.hpp`, shared by the functors and the memory model):

- batches: the boxcar or the Kadane scorer, with Fourier nbins up to 1024
  and time-domain nbins up to 5119;
- seeds and ascend: Fourier nbins up to 1024 (always the boxcar).

Only then do the prune functors skip the irfft scratch and the cuFFT plans
(`ep_cuda_needs_irfft_scratch`): they keep them only for Fourier nbins above
1024.

**Numerics.** The merged folds, the tree contents and the survivor order are
as before. Time-domain boxcar scores are bit-identical, and so are the
Kadane scores up to nbins 256. Above that, the unfused Kadane kernel merges
10 segments of each profile, which sums in another order than the fused
sequential scan: the S/N differs by up to 1e-4. For Fourier folds only the
inverse FFT differs from cuFFT: the S/N moves by about 1e-6, so a leaf whose
score is within that of the threshold can flip. On the 2^25 jerk benchmark
the final survivors and the top candidates are unchanged, the final S/N
moves by at most 4e-6, and the per-level survivor counts move by under
1e-5. C2R ignores the imaginary part of the Nyquist harmonic, which a
fractional shift makes nonzero. The on-chip transform always drops it, as
`numpy.fft.irfft` does. cuFFT's result then depends on the algorithm it
picks, which changes with the batch size.

## PortableFFT

`PortableFFT<M, E, F>` is a forward, natural-order, power-of-two complex FFT
for one block. Its interface follows the cuFFTDx block-FFT model:

- It runs `F` transforms per block. Transform `threadIdx.y` uses `stride =
  M / E` threads, and thread `threadIdx.x` holds elements `threadIdx.x + i *
  stride` (`i < E`) on input and output.
- `execute(v, smem)` transforms the thread's `E` values in place. It uses
  `shared_memory_size` bytes of `smem`.
- Every thread of the block must call `execute`, even threads whose
  transform is inactive.

The algorithm is a Stockham FFT with register radix-E passes.

- The input layout is exactly the read pattern of a first radix-E pass. That
  pass needs no twiddles, so it runs in registers, followed by one write to
  shared memory.
- Each later pass reads `R` elements, synchronizes, applies twiddles, runs a
  radix-R DFT in registers and writes back in place. `R = E`, except the last
  pass may be shorter when `M` is not a power of `E`. `M = 2048` with
  `E = 16` takes passes 16, 16 and 8, with three exchanges through shared
  memory.
- The register DFTs (`pfft_dft<R>`, R up to 16) are unrolled radix-2
  butterflies with constant twiddles.
- Pass twiddles: `sincospif(-2 k / (Ns R))` of the exact fraction, raised to
  powers `r < R` by recurrence. No tables are needed, and the arithmetic is
  FP32.
- Accuracy: across every `nbins` from 2 to 1024, the fused scores differ from
  cuFFT C2R plus `score_and_filter_cuda_d` by at most 3.7e-6 in S/N (median
  7e-7). End to end, the GPU sweep is within 2e-4 S/N of the CPU (FFTW) sweep
  at matched chunking, about 4x closer than the old unfused GPU path.

## Why not cuFFTDx, or cuFFT callbacks

- **cuFFT** is a host library. It cannot run inside a kernel, so it cannot be
  fused with the merge and the scorer.
  - Load callbacks could fold the last merge into the transform. Scoring
    needs whole profiles, though, so the time series would still be written
    out and read back.
  - Legacy callbacks need static cuFFT and are deprecated. LTO callbacks need
    nvJitLink.
- **cuFFTDx** (MathDX) was the first implementation and was dropped.
  - It requires sm_70+, and loki supports sm_50+.
  - Its sizes are compile-time, so arbitrary `nbins` still needed Bluestein
    on top.
  - It is built for a single `SM<>`, so a fat binary had to guard its device
    code per architecture and fall back elsewhere.
  - Measured on an L40S, it was only 3% (power of two) to 7% (Bluestein)
    faster on the fused kernels. Dropping it costs about 2-7% of wall time on
    Fourier accel sweeps.
- MathDX is still used for cuRANDDx (EP device RNG, sm_70+).

## Portability

- **CUDA requirements:** the fused path needs only CUDA intrinsics that exist
  on sm_50: `__syncthreads`, `__syncwarp`, `__shfl_sync`, `sincospif`,
  `atomicAdd` and opt-in shared memory below 48 KiB. CUDA 13 removed sm_50 to
  sm_72, so pre-Volta builds need a CUDA 12 toolkit.
- **Testing on a newer GPU:** build with
  `-DLOKI_CUDA_ARCHITECTURES=60` (or `50`). The binary carries compute_60
  PTX, which a newer GPU JIT-compiles. The first run is slow; set
  `CUDA_CACHE_MAXSIZE` to keep the JIT cache.
- **HIP:** `PortableFFT` needs no vendor FFT library, so it carries over.
  `sincospif`, `__syncthreads` and the register DFTs exist in HIP as they
  are. Two things need care:
  - the warp-barrier assumption: a transform of 32 threads or fewer lies in
    one 32-lane warp. That also holds in a 64-lane wavefront, but needs a
    wavefront barrier or `__syncthreads`;
  - the scorers' 32-lane shuffles and masks (`ffa_score_device.cuh`).

## Tests and benchmarks

- `tests/cpp/internal/ffa_final_score_cuda_t.cpp`:
  - the fused Fourier scorer against the unfused merge kernels, cuFFT C2R and
    `score_and_filter_cuda_d`, at a tolerance of 1e-5;
  - power-of-two and other `nbins`;
  - depth 1 and 2, freq-only and accel, both scorer regimes;
  - a sweep over every `nbins` from 2 to 1024.
- `tests/cpp/internal/ep_score_cuda_t.cpp`: the fused EP step against the
  unfused chain (shift-add, cuFFT C2R, `score_and_filter_max_cuda_d`):
  - Fourier: every `nbins` from 2 to 1024; both scorer regimes; more than
    32 widths;
  - time domain: bit for bit;
  - row scoring (seeds, ascend) against cuFFT + `snr_boxcar_3d_max_cuda_d`.

  It checks the merged folds bit for bit, the S/N within 1e-5, the masks,
  counts and min/max, and, in direct mode, the survivors' slots, leaves,
  folds and scores. Its spectra keep DC and Nyquist real, where cuFFT C2R
  is well defined.
- Validate a change end to end with an injection study at 2^25, not only
  2^23. Chunk widths change the search grid (`docs/memory.md`).

Measured on an L40S (`nthreads = 1`, 10 GB budget, wall time, 2^25
samples, accel search, Fourier folds). The previous release ran these
unfused (cuFFT inverse plus a separate scorer), and every other stage of the
sweep was also slower then. A cuFFTDx version of the fused kernel was 2-7%
faster than PortableFFT.

| `nbins` | Previous release | Fused, PortableFFT |
| --- | --- | --- |
| 64-1024 (power of two, direct) | 60.0 s | 9.4 s |
| 50-800 (Bluestein) | 29.4 s | 8.5 s |
