# Loki architecture

This page records the frozen layout of the library and the rules that keep it
that way. New algorithms, helpers and backends are added by following the
recipes at the end; they should not need a new directory or a new pattern.

## One library, one module, backend chosen at run time

- `loki::loki` is a single library. A CPU-only build compiles `lib/**/*.cpp`.
  A GPU build adds `lib/cuda/*.cu` to the same target.
- Python gets a single extension, `libloki`. Every class or function that can
  run on more than one backend takes `backend="cpu" | "cuda"` and `device=`,
  both keyword-only.
- C++ callers pass a `loki::Exec` (`Exec::cpu()`, `Exec::cuda(device)`).
  `loki::available_backends()` reports what this build contains. Asking for a
  backend the build lacks throws `std::invalid_argument`.

```text
public class (include/loki/<domain>/x.hpp)          one type per concept, pIMPL
  └─ facade (lib/<domain>/x.cpp)                    make_x_engine(): the only #ifdef
       ├─ engine interface (lib/<domain>/x_engine.hpp)   abstract class + factories
       ├─ CPU engine  (lib/cpu/x_cpu.cpp)           make_x_cpu()
       └─ GPU engine  (lib/cuda/x_cuda.cu)          make_x_gpu()   (GPU builds only)
```

The backend is chosen once, at construction. Each `execute` costs one virtual
call, which is nanoseconds against kernels that run for milliseconds or
longer. Inner loops never cross the facade. Internal pipelines such as
`FFAFreqSweep` and `EPFreqSweep` build the concrete engines directly with
`make_*_cpu` / `make_*_gpu` and share buffers between them, so device data
stays on the device.

## Layout

```text
include/loki/<domain>/*.hpp   public API: pure C++20, identical in every build
lib/
  <domain>/*.cpp              facades and backend-neutral code, mirroring include/loki/<domain>/
  <domain>/*_engine.hpp       engine interface + make_*_cpu / make_*_gpu declarations
  <domain>/*.hpp              other private headers of that domain
  common/                     backend.cpp, dispatch.hpp (dispatch helpers), plans.cpp
  core/                       kinematics and kernel interfaces shared by CPU and GPU code
  detail/                     internal utilities: errors, math, timing, progress, psr_utils
  cpu/*_cpu.cpp               CPU engines and kernels (OpenMP, SIMD); CPU-only headers
  cuda/*_cuda.cu, *.cuh       GPU engines, kernels, device types and helpers
src/
  loki_python.cpp             PYBIND11_MODULE: calls one bind_<submodule>() per submodule
  bindings/bind_<sub>.cpp     one file per Python submodule
tests/
  cpp/api/*_t.cpp             black-box Catch2 tests: public headers only (loki_api_tests)
  cpp/internal/*_t.cpp        white-box Catch2 tests of private code (loki_internal_tests)
  consumer/                   find_package(loki) project built against an install (CI)
  python/                     pytest suite for libloki
scripts/check_architecture.sh the mechanical checks for these rules (pre-commit and CI)
```

Domains are `algorithms`, `common`, `detection`, `io`, `pipelines`, `search`,
`simulation`, `utils`, plus the private-only `core` and `detail`.

## Rules

### 1. Public API (`include/loki/`)

- Standard C++20 only. No CUDA, Thrust, CCCL or backend macro, and no
  `#ifdef` that changes declarations between builds.
- One type per concept. Never add a `FooCUDA` / `FooCPU` public type; add a
  backend to `Foo` instead.
- Host memory is passed as `std::span`. Device memory is passed through a
  separate overload that takes `DeviceSpan<T>` and an optional `Stream`. Host
  and device memory are never told apart by inspecting a pointer.
- Public headers include only `loki/...` public headers and the standard
  library.
- Shared resources are opaque handles built with an `Exec`:
  `memory::FFAWorkspace<T>`, `memory::EPWorkspace<T>` and `math::FFTManager`.
  An algorithm that receives one checks that it was built for the same
  backend and device.
- A handle's storage is private: `class Impl; Impl& impl();` sit under
  `private:`, with `friend struct loki::detail::HandleAccess;`. Library code
  reaches the concrete buffers only through the accessors built on
  `HandleAccess` (`memory::detail::cpu_workspace`, `cuda_workspace`,
  `math::detail::cpu_fft`, `cuda_fft`). A new handle follows the same
  pattern.

### 2. Signatures

- `Exec exec = {}` is the last parameter of every constructor and free
  function that dispatches. A convenience overload `(required..., Exec)` that
  drops the tuning knobs is allowed.
- No `int nthreads` overloads. Callers write `Exec::cpu(n)`.
- Classes built from a search config (`FFA`, `EPMultiPass`, `FFAFreqSweep`,
  `EPFreqSweep`) take the CPU thread count from the config. For them `Exec`
  selects only the backend and device, and a non-default `Exec.nthreads` logs
  a warning that it is ignored.

### 3. Dispatch

- Each facade `.cpp` has a `make_<algo>_engine(...)` in an anonymous
  namespace. It is the only place in that file that tests `LOKI_ENABLE_GPU`:

  ```cpp
  if (exec.backend == Backend::kCPU) {
      return detail::make_foo_cpu(...);
  }
  #ifdef LOKI_ENABLE_GPU
  if (exec.backend == loki::detail::kGPUBackend) {
      return detail::make_foo_gpu(..., exec.device);
  }
  #endif
  loki::detail::throw_unavailable("Foo", exec.backend);
  ```

- Engines are named `<Algo>CpuEngine` / `<Algo>CudaEngine`. Factories are
  named `make_<algo>_cpu` / `make_<algo>_gpu`, in `loki::<domain>::detail`.
  They are declared unconditionally in `lib/<domain>/<algo>_engine.hpp`. The
  `_gpu` names leave room for a HIP build of the same sources.
- A backend that is compiled in but has no engine for an algorithm throws
  `throw_unimplemented`. A backend that is not compiled in throws
  `throw_unavailable`. Both are `[[noreturn]]` and live in
  `lib/common/dispatch.hpp`.
- Every `DeviceSpan` overload of a facade calls `check_device` on its views.
- Free functions that dispatch (e.g. `snr_boxcar_*` in
  `lib/detection/score.cpp`) have no engine object. Each one follows the same
  three branches inline: CPU, then the GPU branch under `#ifdef
  LOKI_ENABLE_GPU`, then `throw_unavailable`. Their `DeviceSpan` overloads
  first call `common_device(...)` so that all views agree on one device. In a
  build without a GPU backend they throw `throw_unavailable(..., kCUDA)`.
- A public free function that is CPU-only by design (no `Exec`, e.g.
  `MatchedFilter`, `append_snr_boxcar_3d_hits`) may be defined in its facade
  `.cpp`. Kernels it shares with a CPU engine go in a `lib/cpu/` header (e.g.
  `cpu/boxcar_kernels.hpp`).
- Name GPU-only checks after `kGPUBackend` or the caller's `exec.backend`,
  never a hard-coded `Backend::kCUDA`, so a HIP build reuses them.
- A CPU engine whose `execute(DeviceSpan...)` cannot work throws
  `throw_no_device_memory`.

### 4. Backend macros

- `LOKI_ENABLE_GPU` (any GPU backend) and `LOKI_ENABLE_CUDA` (which backend)
  are PRIVATE compile definitions of `loki`. They never reach consumers.
- `LOKI_ENABLE_GPU` appears only in the dispatch code of facade `.cpp` files
  in `lib/<domain>/` (`make_*_engine` functions and dispatching free
  functions), in the handle constructors in `lib/utils/`, in
  `lib/common/backend.cpp` and `lib/common/dispatch.hpp`, and in
  `tests/cpp/internal/`. Never in a header, in `lib/cpu/`, in `include/`, or
  in API tests.
- Code in `lib/cuda/` is compiled only for GPU builds and needs no guard.
- `LOKI_HD` in `types.hpp` is keyed on `__CUDACC__`, not on the backend, so
  the public header is still identical in every build.

### 5. Private headers

- Every repository header is included by its path from the repository root,
  in one of exactly two forms:
  - public: `#include "loki/algorithms/ffa.hpp"` (resolved in `include/`);
  - private: `#include "lib/detail/math.hpp"`, `"lib/cuda/cuda_utils.cuh"`,
    `"lib/algorithms/ffa_engine.hpp"`.

  The prefix alone tells a reader whether a header is part of the installed
  API. The `lib/` prefix cannot collide with public headers or with system
  directories such as CUDA's `cuda/`. Private headers stay next to the `.cpp`
  that implements them. A bare `"file.hpp"` is allowed only for a header in
  the same directory (e.g. a test helper).
- Include order is enforced by `.clang-format` (`IncludeBlocks: Regroup`).
  Each group is its own block, separated by a blank line and sorted
  alphabetically:
  1. the file's own header: `foo.hpp` for `foo.cpp`, `foo_cpu.cpp`,
     `foo_cuda.cu` or `foo_t.cpp`;
  2. the C++ standard library (`<vector>`, `<cstdint>`);
  3. system and third-party headers (`<omp.h>`, `<fmt/format.h>`,
     `<cuda_runtime.h>`, `<catch2/...>`);
  4. public Loki headers (`"loki/..."`);
  5. private Loki headers (`"lib/..."`);
  6. other same-directory headers.

  "Own header" means the header that declares what the file defines. An
  engine file (`lib/cpu/ffa_cpu.cpp`, `lib/cuda/ffa_cuda.cu`) implements
  `make_*_cpu` / `make_*_gpu` from `lib/<area>/<algo>_engine.hpp`, not the
  public facade, so it has no own header in group 1. It includes the facade
  header only if it uses a facade type (as `thresholds_cuda.cu` uses
  `State`). Public headers are already compiled standalone by
  `check_architecture.sh`, so nothing is lost.

  Because each group is a separate block, clang-tidy's
  `llvm-include-order` (which sorts within a block) agrees with
  clang-format. Do not reorder includes by hand; run clang-format.
- `.hpp` files are host C++. Device declarations, Thrust containers and CUDA
  types go in `lib/cuda/*.cuh`. For example `core/taylor.hpp` declares the
  host functions, and `cuda/taylor_cuda.cuh` declares the device ones.
- Device-only fold types live in `cuda/types_cuda.cuh` (`ComplexTypeCUDA`,
  `CudaFoldType<T>`).

### 6. Consumers inside the repository

- `applications/` and `src/` (Python) use public headers only.
- C++ tests come in two executables, and the build enforces the boundary:
  - `tests/cpp/api/` (`loki_api_tests`) links only `loki::loki`, exactly like
    a downstream project. A `"lib/..."` include there fails to compile.
    Prefer an API test: it exercises behaviour through the public API and
    survives internal refactors.
  - `tests/cpp/internal/` (`loki_internal_tests`) and `bench/` link
    `loki::internal`, an in-tree INTERFACE target that is never installed.
    It adds the repository root (for `"lib/..."`), the backend macros and
    the CUDA headers. Use it only for private code the public API cannot
    reach or cannot test precisely: math helpers, HDF5 writers, FFT plan
    caches, masks, kernels.
  - Do not add `lib/` or the repository root to an include path by hand.
- `tests/consumer/` is a separate CMake project that builds against an
  installed Loki with `find_package(loki)`. CI runs it, so the installed
  package keeps working for downstream users.
- GPU tests are gated at run time, so a CPU-only build reports them as
  skipped instead of compiling them away:
  `if (!loki::is_available(Backend::kCUDA)) { SKIP("..."); }`. Use
  `#ifdef LOKI_ENABLE_CUDA` only when a test needs internal CUDA types
  (today only `fft_t.cpp`).
- GPU results are compared to the CPU or to an owning instance bit for bit
  where the kernel is deterministic. The time-domain brute fold accumulates
  with float `atomicAdd`, so its outputs and anything derived from them
  (time-domain FFA folds, scores, `level_stats.score_max`) vary from run to
  run at the float32 rounding level. Compare those with a tolerance.

### 7. Namespaces

The namespace follows the directory. Public headers:

| Directory | Namespace |
| ----------- | ----------- |
| `include/loki/algorithms/` | `loki::algorithms` |
| `include/loki/pipelines/` | `loki::pipelines` |
| `include/loki/detection/`, `io/`, `search/`, `simulation/` | `loki::<directory>` |
| `include/loki/common/` | `loki`, except `plans.hpp` (`loki::plans`) and `coord.hpp` (`loki::coord`) |
| `include/loki/utils/` | one namespace per header: `fft.hpp` uses `loki::math`, `workspace.hpp` uses `loki::memory`, `psr_utils.hpp` uses `loki::psr_utils` |

The `common/` and `utils/` exceptions are part of the frozen API. Do not add
more: a new public header takes the namespace of its directory.

Private code:

- `lib/<domain>/` uses the public namespace of that domain. So does the
  `lib/cpu/` and `lib/cuda/` code that implements it. For example,
  `lib/cuda/prune_cuda.cu` is `loki::algorithms`, and
  `lib/cpu/ffa_freq_sweep_cpu.cpp` is `loki::pipelines`. Code in
  `lib/utils/` uses the namespace of the handle it backs (`loki::math` or
  `loki::memory`).
- Engine interfaces and the `make_*_cpu` / `make_*_gpu` factories live in
  `loki::<domain>::detail`. `loki::detail` holds only the cross-domain
  helpers in `lib/common/dispatch.hpp` (`throw_unavailable`,
  `HandleAccess`, `DeviceStorage`, ...).
- `lib/core/` (kinematics, kernel entry points, transforms) and its device
  halves in `lib/cuda/*_cuda.cuh` use `loki::core`.
- Low-level helper headers use a namespace named after the header:
  `detail/utils.hpp` uses `loki::utils`, `detail/error_check.hpp` uses
  `loki::error_check`, `detail/timing.hpp` uses `loki::timing`,
  `cuda/cuda_utils.cuh` uses `loki::cuda_utils`, `cuda/cub_helpers.cuh` uses
  `loki::cub_helpers`, `cuda/device_rng.cuh` uses `loki::device_rng`,
  `cpu/simd_utils.hpp` uses `loki::simd_utils`,
  `cpu/brute_fold_intrinsics.hpp` uses `loki::brute_fold_intrinsics`,
  `detail/progress.hpp` uses `loki::progress`. The
  device companion of a host header uses the host header's namespace:
  `cuda/kernel_utils.cuh` uses `loki::utils`, `cuda/types_cuda.cuh` uses
  `loki`, `cuda/coord_cuda.cuh` uses `loki::coord`.
- Code private to one translation unit goes in an anonymous namespace.
  Implementation helpers shared inside a namespace `N` go in `N::detail`.
  Never use `details`, and never put a name in the global namespace.

## Recipes

### Adding an algorithm `Foo`

1. `include/loki/<domain>/foo.hpp`: the class (pIMPL) or free function, with
   `Exec exec = {}` last, and `DeviceSpan` overloads if it accepts device
   data.
2. `lib/<domain>/foo_engine.hpp`: `class FooEngine` with pure virtual host
   entry points, device entry points whose default throws, and the
   declarations of `make_foo_cpu` / `make_foo_gpu`.
3. `lib/<domain>/foo.cpp`: `make_foo_engine` (rule 3) and the forwarding
   methods.
4. `lib/cpu/foo_cpu.cpp`: `FooCpuEngine` and `make_foo_cpu`, with explicit
   instantiations for templated engines.
5. Optional: `lib/cuda/foo_cuda.cu` (plus `foo_cuda.cuh` if other `.cu` files
   need it), with `FooCudaEngine` and `make_foo_gpu`. Until it exists, the
   facade calls `throw_unimplemented` for the GPU backend.
6. `src/bindings/bind_<domain>.cpp`: the binding. Use `py::kw_only()` before
   `backend` / `device`, and build the `Exec` with `make_exec(backend, device)`.
7. `tests/cpp/api/foo_t.cpp`: a CPU test through the public API. Gate CUDA
   checks with `loki::is_available(Backend::kCUDA)` and `SKIP`. Add a
   `tests/cpp/internal/` test only for private helpers the API cannot reach.

No CMake edit is needed: the globs pick up the new files.

### Adding a GPU backend (e.g. AMD / HIP)

1. Add `Backend::kHIP` and `Exec::hip(device)` to `common/backend.hpp`, and
   their names to `to_string` / `parse_backend`.
2. In CMake, add a `LOKI_GPU=CUDA|HIP` choice. Compile `lib/cuda/*.cu` with
   `LANGUAGE HIP` and define `LOKI_ENABLE_HIP` and `LOKI_ENABLE_GPU`.
3. Point `kGPUBackend` in `lib/common/dispatch.hpp` at `Backend::kHIP` under
   `LOKI_ENABLE_HIP`.
4. Add `lib/cuda/gpu_compat.cuh`, which maps the CUDA, cuFFT, Thrust and CUB
   names onto their HIP equivalents.

The facades do not change: they already dispatch on `kGPUBackend` through
`make_*_gpu`. One GPU backend is compiled per build.

## Numerics

- Release builds compile with `-ffast-math` (see `loki_compile_options` in
  `CMakeLists.txt`). Library and test code must not rely on IEEE
  infinities, NaN propagation or `std::isfinite`:
  - use `utils::is_finite` / `utils::is_nan` (`lib/detail/utils.hpp`), which
    inspect the bits;
  - use sentinels such as `std::numeric_limits<T>::max()` / `lowest()` for
    open bounds (as `ParamWindow` does), never `infinity()`.
- Because of `-ffast-math`, OpenMP reductions and vectorisation, results may
  differ in the last bits between compilers, thread counts and CPUs. The
  thread count is part of the reproducibility key.
- Random draws never depend on threads or on scheduling. Give every
  independent work item its own stream keyed on (seed, purpose, item), e.g.
  `make_pcg` in `thresholds_cpu.cpp` with `math::NormalSampler`, and never
  keep generator state in `thread_local` or in per-thread slots. A fixed seed
  then gives the same result for any thread count and any run order
  (`DynamicThresholdScheme::run` and `evaluate` on the CPU are bit-identical
  across thread counts).
- On the GPU, the time-domain brute fold accumulates with float `atomicAdd`
  and is not bit-reproducible run to run (rule 6). Every other path is
  deterministic for a fixed seed.
- Thread counts. The caller's `nthreads` is never capped from above:
  library code clamps only from below, `nthreads = std::max(nthreads, 1)`,
  before using it in `num_threads(...)`. `omp_get_max_threads()` would let
  `OMP_NUM_THREADS` or an enclosing region override an explicit request,
  and the thread count is part of the reproducibility key. Only the
  configuration and CLI entry points (`configs.cpp` FFA sweep setup,
  `applications/loki.cpp`) use `omp_get_max_threads()`, to resolve
  `nthreads <= 0` to "all threads".

## Checks

All rule checks are scripted:

```bash
scripts/check_architecture.sh             # CXX=<compiler> to choose the compiler
SKIP_COMPILE=1 scripts/check_architecture.sh   # skip the header compile
```

The script checks that:

- public headers carry no GPU code and no backend macro;
- repository includes are `"loki/..."` or `"lib/..."` (or same-directory);
- `"lib/..."` never appears in `include/`, `src/`, `applications/` or
  `tests/cpp/api/`;
- `LOKI_ENABLE_*` appears only where rule 4 allows it;
- public headers open only the namespaces rule 7 allows;
- every public header compiles on its own.

A new rule should come with a new check in the script.

## Portability

CI builds with the oldest compilers `CMakeLists.txt` accepts (GCC 13,
Clang 18). Code that newer compilers accept can still fail there:

- Never call `error_check::*` (or anything else with a `std::source_location`
  default argument) directly inside an OpenMP region. GCC < 14 then asks for
  hidden `source_location` statics in the data clauses, and an exception
  cannot leave a region anyway. Validate before the region, or design the
  loop so the invariant holds by construction. A check inside a called
  function is fine.
- OpenMP regions in free functions use `default(none)` and list what they
  use: `firstprivate` for read-only scalars, pointers and spans, `shared` for
  containers and objects; `constexpr` constants need no listing
  (clang-tidy `openmp-use-default-none`).
- Regions in member functions omit the `default` clause. Data-sharing
  clauses cannot name class members, so `default(none)` would force a local
  copy of every member. They carry
  `// NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members`.
  Keep such regions small; work that needs many members belongs in a member
  helper called from the loop.
- A lambda must not capture a structured binding (Clang 18 with OpenMP
  rejects it). Bind a named variable first.
- Code under `#if defined(__AVX2__)` / `__AVX512F__` is not compiled on
  arm64. Do not add `const` to a variable that such a branch writes, and do
  not trust a local arm64 build for those branches.

## Static analysis

`.clang-tidy` (root) and `tests/.clang-tidy` (inherits it, relaxes a few
test-only checks) define the checks; each file lists why a check is off.

```bash
run-clang-tidy -p build-dev "$PWD/(lib|src|tests|applications)/"
```

- A suppression names its check and gives a reason, on its own line:
  `// NOLINTNEXTLINE(<check>): <reason>`. `.clang-format` never reflows
  NOLINT comments (`CommentPragmas`).
