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
tests/cpp/*_t.cpp             Catch2 tests (may use private headers via loki::internal)
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
- A CPU engine whose `execute(DeviceSpan...)` cannot work throws
  `throw_no_device_memory`.

### 4. Backend macros

- `LOKI_ENABLE_GPU` (any GPU backend) and `LOKI_ENABLE_CUDA` (which backend)
  are PRIVATE compile definitions of `loki`. They never reach consumers.
- `LOKI_ENABLE_GPU` appears only in `make_*_engine` functions, in the handle
  constructors in `lib/utils/`, in `lib/common/backend.cpp` and
  `lib/common/dispatch.hpp`, and in tests.
- Code in `lib/cuda/` is compiled only for GPU builds and needs no guard.
- `LOKI_HD` in `types.hpp` is keyed on `__CUDACC__`, not on the backend, so
  the public header is still identical in every build.

### 5. Private headers

- Private headers are included without the `loki/` prefix, relative to `lib/`:
  `"detail/math.hpp"`, `"cuda/cuda_utils.cuh"`, `"algorithms/ffa_engine.hpp"`.
  So `#include "loki/..."` always means public and anything else means
  private.
- `.hpp` files are host C++. Device declarations, Thrust containers and CUDA
  types go in `lib/cuda/*.cuh`. For example `core/taylor.hpp` declares the
  host functions, and `cuda/taylor_cuda.cuh` declares the device ones.
- Device-only fold types live in `cuda/types_cuda.cuh` (`ComplexTypeCUDA`,
  `CudaFoldType<T>`).

### 6. Consumers inside the repository

- `applications/` and `src/` (Python) use public headers only.
- `tests/cpp` and `bench` link `loki::internal`, an in-tree INTERFACE target
  that is never installed. It adds `lib/` to the include path, along with the
  backend macros and the CUDA headers, for white-box tests. Do not add `lib/`
  to an include path by hand.

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
7. `tests/cpp/foo_t.cpp`: a CPU test. Gate CUDA checks with
   `loki::is_available(Backend::kCUDA)` and `SKIP`.

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

## Checks

```bash
# Public headers carry no GPU code
grep -rnE "cuda_runtime|thrust/|cuda/std|LOKI_ENABLE" include/   # expect nothing

# Private headers are never included through the public prefix
grep -rnE '#include "loki/(core|detail|cuda)/' lib src tests bench applications   # expect nothing

# Each public header compiles on its own, without lib/ or backend macros
for h in $(cd include && find loki -name '*.hpp'); do
  echo "#include \"$h\"" | c++ -std=c++20 -fsyntax-only -Iinclude -x c++ -
done
```
