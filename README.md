<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/loki-logo-dark.svg">
    <img alt="loki" src="docs/_static/logo/loki-logo.svg" width="360">
  </picture>
</p>

**loki** searches for short-period binary pulsars with coherent signal detection and pruning.

A high-performance **C++20** pulsar searching library with **Python bindings**. The same core library supports **CPU-only**, **GPU-accelerated**, and **CPU+GPU** builds: CPU paths always compile; CUDA paths are added when the CUDA toolchain is enabled at configure time.

Worked examples on simulated data are in [`notebooks/`](notebooks/README.md).

---

## Requirements

### Compilers

- **GCC >= 11** or **Clang >= 17** (C++20 support required)
- **CUDA >= 12.2** *(optional, for GPU acceleration)*
- **CMake >= 3.25**
- **Python >= 3.12** *(for Python bindings)*

MSVC is not supported.

### System libraries (always required)

These are **never** downloaded by loki and must be discoverable by CMake on your system — install them before building. The recommended way is via [conda-forge](https://conda-forge.org) or [mamba](https://mamba.readthedocs.io/en/latest/):

| Library | Minimum Version | conda-forge install |
| --------- | ----------------- | --------------------- |
| HDF5 | 1.10 (1.14 tested; 2.x not yet supported) | `mamba install "hdf5=1.14"` |
| OpenMP | - | `mamba install libomp` *(macOS)* / `libgomp` *(Linux)* |
| CMake | 3.25 | `mamba install cmake>=3.25` |
| Ninja | - | `mamba install ninja` |
| GCC | 11 | system GCC 11+, or `mamba install gxx_linux-64` *(Linux)* |
| Python | 3.12 | `mamba install python>=3.12` |

Header-only C++ dependencies (fmt, spdlog, HighFive, CLI11, xsimd, etc.) are fetched automatically via [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) unless `LOKI_USE_SYSTEM_DEPS=ON`.

### GPU and toolchain support policy

Frozen minimum requirements for CUDA builds:

| Component | Minimum |
| --------- | ------- |
| CUDA Toolkit | 12.2 |
| GCC (when used as nvcc host compiler) | 11 |
| NVIDIA driver | R525 (CUDA 12.x builds), R580 (CUDA 13.x builds) |
| CMake | 3.25 |
| GPU compute capability | **sm_50** (Maxwell) with CUDA 12.x; sm_75 with CUDA 13 |

- CUDA Toolkit 12.2+, with a host compiler that toolkit accepts: nvcc 12.2–12.3 take GCC <= 12
  (Clang <= 16), nvcc 12.4–12.6 GCC <= 13.2 (Clang <= 17/18), nvcc 12.8+ GCC 14.
- An older driver runs a newer 12.x build through CUDA minor-version compatibility, as long as
  the binary holds SASS for the GPU (the default `native` architecture does). Check the driver
  with `nvidia-smi` on a compute node, not the login node.

---

## Installation modes

### Mode A — Python (`uv pip install`) *(recommended for pipelines)*

Best for using loki from Python in a conda environment. Builds one extension, `libloki`. CPU and CUDA are selected at call time with `backend="cpu"` or `backend="cuda"`.

```bash
mamba create -n loki_env python=3.12
mamba activate loki_env
mamba install -c conda-forge cmake>=3.25 ninja hdf5 libomp  # macOS
# Linux: system GCC >= 11 works; otherwise also install gxx_linux-64

export CPM_SOURCE_CACHE="$HOME/.cache/CPM"   # optional; avoids re-downloading CPM deps

uv pip install git+https://github.com/pravirkr/loki.git
```

#### CUDA modes (`LOKI_CUDA`)

| Value | Behaviour |
| ------- | ----------- |
| **`AUTO`** *(default)* | Build GPU support if `nvcc` is found; otherwise CPU-only. |
| **`ON`** | **Require** the CUDA toolkit and MathDX. **Hard fail** if either is missing. |
| **`OFF`** | CPU-only even if CUDA is installed. |

```bash
# Default: AUTO (GPU if nvcc is available, else CPU)
uv pip install git+https://github.com/pravirkr/loki.git

# Force GPU — fails with a clear error if CUDA/GPU is unavailable
uv pip install git+https://github.com/pravirkr/loki.git -C cmake.define.LOKI_CUDA=ON

# Force CPU-only
uv pip install git+https://github.com/pravirkr/loki.git -C cmake.define.LOKI_CUDA=OFF
```

#### Native CPU tuning (`-march=native`)

Local `uv pip install` defaults to **`LOKI_ENABLE_NATIVE_ARCH=ON`**, so Release builds use `-march=native` for best performance on **your** machine.

To disable (e.g. when cross-compiling or building a generic binary):

```bash
uv pip install git+https://github.com/pravirkr/loki.git -C cmake.define.LOKI_ENABLE_NATIVE_ARCH=OFF
```

An image or binary that runs on other machines needs it off; the
[container](#containers) sets an explicit CPU baseline instead.

---

### Mode B — C++ library (`cmake` + `find_package`)

Best for linking loki directly from C++20 code.

```bash
git clone https://github.com/pravirkr/loki.git
cd loki

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOKI_BUILD_PYTHON=OFF \
  -DLOKI_CUDA=AUTO

cmake --build build -j
cmake --install build --prefix "$HOME/.local"
```

Use from your `CMakeLists.txt`:

```cmake
find_package(loki CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE loki::loki)
```

When loki was built **with CUDA**, the same `loki::loki` target contains the GPU engines. Callers pass `loki::Exec::cpu()` or `loki::Exec::cuda(device)` (Python: `backend="cpu"` / `backend="cuda"`, `device=0`). `loki::available_backends()` reports which backends this build contains. The installed headers are identical in every build and include no CUDA header; the backend macros are private to the library.

Pipelines that run many searches can allocate buffers once and share them:

```cpp
loki::memory::FFAWorkspace<float> ws(plan, loki::Exec::cuda(0));
loki::math::FFTManager fft(loki::Exec::cuda(0));
loki::algorithms::FFA<float> ffa(ws, fft, cfg, loki::Exec::cuda(0));
```

The library layout and the rules for adding algorithms or backends are in [`docs/architecture.md`](docs/architecture.md).

---

## CMake options reference

Project-specific options use the **`LOKI_`** prefix. Standard CMake options keep their usual names.

| Option | Default | Description |
| -------- | ----------------- | --------------------- |
| **`LOKI_CUDA`** | `AUTO` | CUDA build mode: `AUTO`, `ON`, or `OFF` |
| **`LOKI_CUDA_ARCHITECTURES`** | `native` | Passed to `CMAKE_CUDA_ARCHITECTURES` (`native`, `61;80`, `all-major`, …). Minimum supported GPU: **sm_50**. |
| **`LOKI_ENABLE_NATIVE_ARCH`** | `ON` | Add `-march=native` in Release builds |
| **`LOKI_CPU_ARCH`** | *(empty)* | Fixed `-march` value instead of native (`x86-64-v2`, `x86-64-v3`, …), for binaries that run on other machines. The `portable` preset sets `x86-64-v2` and SASS `75;80;86;89;90`. |
| **`LOKI_CURANDDX_SM`** | *(empty)* | Pin cuRANDDx's `SM<>` value for every architecture slice; empty: each slice uses its own |
| **`LOKI_USE_SYSTEM_DEPS`** | `OFF` | Prefer system CPM packages over pinned downloads |
| **`LOKI_BUILD_PYTHON`** | `ON` | Build Python extension modules |
| **`LOKI_PYTHON_INPLACE`** | `OFF` | Copy the built module into `src/loki/` for `PYTHONPATH=src` (the `release` preset turns it on; ignored under pip) |
| **`LOKI_BUILD_TESTING`** | `OFF` | Build C++ Catch2 tests |
| **`LOKI_BUILD_BENCHMARKS`** | `OFF` | Build Google Benchmark executables |
| **`LOKI_BUILD_DOCS`** | `OFF` | Build documentation |
| **`LOKI_ENABLE_COVERAGE`** | `OFF` | Compile with `--coverage` |
| **`LOKI_ENABLE_IPO`** | `OFF` | Link-time optimization (Release) |
| **`LOKI_WARNINGS_AS_ERRORS`** | `OFF` | `-Werror` on Loki's own C++ sources (the `ci` preset turns it on) |
| **`LOKI_ENABLE_NVTX`** | `OFF` | NVTX ranges for `nsys` profiling (sweeps, chunks, FFA stages, preprocessing, EP). Needs the CUDA toolkit headers, also for CPU-only builds. |
| **`BUILD_SHARED_LIBS`** | `OFF` | `ON` = shared `libloki`; `OFF` = static `libloki.a`. Must be `OFF` when `LOKI_BUILD_PYTHON=ON`. |

### CPU vs GPU vs CPU+GPU

- **CPU-only** (`LOKI_CUDA=OFF`): all `.cpp` sources; no `.cu`; Python module `libloki` only. `available_backends()` is `["cpu"]`.
- **CPU+GPU** (`LOKI_CUDA=AUTO` or `ON` with a working toolchain): `lib/**/*.cpp` and `lib/cuda/*.cu` in one library and one Python module, `libloki`. `available_backends()` is `["cpu", "cuda"]`.
- The library is always usable on CPU. GPU code paths are compiled only when CUDA is enabled at configure time, and selected per call with `Exec` / `backend`.

### GPU architecture

- **Minimum GPU:** sm_50 (Maxwell). CMake rejects lower values in `LOKI_CUDA_ARCHITECTURES`,
  and sm_50–sm_72 with a CUDA 13 toolkit (which supports sm_75+; use CUDA 12.x for older GPUs).
- **Fat binaries:** pass multiple SMs, e.g. `-DLOKI_CUDA_ARCHITECTURES='61;80'` or `all-major`.
- **MathDX / cuRANDDx** is downloaded and linked only when any target arch is sm_70+ and `LOKI_FORCE_CURAND_RNG=OFF`.

---

## Python bindings and `BUILD_SHARED_LIBS`

Pip/scikit-build sets **`BUILD_SHARED_LIBS=OFF`**. The C++ core is linked **statically** into the Python extension `libloki.cpython-*.so`. The wheel ships that one importable module — no separate `libloki.so` and no custom loader paths. CUDA support, when built, lives in the same module.

For **C++-only** builds (`LOKI_BUILD_PYTHON=OFF`), either shared or static `loki` is fine:

- **`BUILD_SHARED_LIBS=ON`**: shared `libloki.so` for `find_package(loki)`.
- **`BUILD_SHARED_LIBS=OFF`** (default): static `libloki.a`.

Do not combine **`LOKI_BUILD_PYTHON=ON`** with **`BUILD_SHARED_LIBS=ON`**; CMake will fail with an explicit error.

---

## Optional: CPM source cache

```bash
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"
```

Add to `~/.bashrc` or `~/.zshrc` to cache downloaded dependencies across rebuilds.

---

## Development builds

`CMakePresets.json` keeps every build in Release. `release` is the default and
builds into `build/` (CUDA if found, tests, Python, `-march=native`):

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release -j 8
PYTHONPATH=src pytest -n auto tests/python
```

The `release` build copies the Python module into `src/loki/`
(`LOKI_PYTHON_INPLACE`), so `PYTHONPATH=src` imports it without an install.

`cuda` requires CUDA and adds the benchmarks (`build-cuda/`), `asan` is the
ASan/UBSan build (`build-asan/`, skips the `[slow]` tests), and `ci` and
`ci-tidy` are the CI builds. `pip install` builds into `build-wheel/`.

---

## Command-line output

`loki --help` lists the subcommands; `loki search ffa --help` and the
others list their options. Output is controlled by three options, accepted
before or after the subcommand:

- `-v` adds per-chunk and per-run detail, `-vv` everything;
- `-q` keeps warnings and errors and turns off the progress bars, `-qq`
  keeps errors only;
- `--log-file PATH` also appends the log to a file, as plain text, with at
  least the info lines whatever `-q` hides on the console.

Colour and live progress bars appear only on an interactive terminal; a
pipe or a batch job's log gets plain lines. `NO_COLOR=1` turns colour off,
`CLICOLOR_FORCE=1` forces it on, and `LOKI_PROGRESS` (`rich`, `plain` or
`off`) overrides the progress bars.

Exit status: 0 success, 1 failure, 130 stopped by Ctrl-C, 143 stopped by
SIGTERM (e.g. Slurm's `scancel`); any other value is an invalid command
line.

---

## Containers

[`docker/Dockerfile`](docker/Dockerfile) builds a Loki-only image: the Python
module (the C++ library is linked into it), the `loki` command, and the
runtime libraries those need (HDF5, NumPy, pyloki). It does not include
dedispersion, filterbank tools, or injection codes. A Nextflow process, or a
larger pipeline image, can use this container instead of compiling Loki on
the cluster.

The image installs Loki with `uv`, the same installer as the Python
instructions above. It does not use the `release` CMake preset. That preset
is for a machine you are building on: tests stay on, and both `-march=native`
and the CUDA architecture list stay at their defaults (`native`). An image
has to run on other nodes, so the Dockerfile turns native tuning off, sets
an explicit CPU baseline and GPU list, and leaves the tests out.

The image is portable: the default CPU baseline is `x86-64-v3`, and the CUDA
image lists GPU architectures explicitly (`80;86;90` by default). Both are
build arguments. There is no conda package. On a machine where you can
compile, use mamba only for the compilers and libraries (HDF5), then
`uv pip install`. On a cluster, build the image once (or a `.sif` from it)
instead of adding a third packaging channel. There is no required container
registry: clone this repo and `docker build` on a machine with Docker and
network access (the build fetches pyloki and CPM dependencies).

```bash
docker build -f docker/Dockerfile --target cpu  -t loki:cpu  .
docker build -f docker/Dockerfile --target cuda -t loki:cuda .
```

```bash
# narrower CPU baseline, or a different set of GPUs
docker build -f docker/Dockerfile --target cuda -t loki:cuda \
  --build-arg LOKI_MARCH=x86-64 \
  --build-arg LOKI_CUDA_ARCHITECTURES="80;90" .
```

Compiling CUDA needs no GPU or driver, so the image builds with `LOKI_CUDA=ON`
and an explicit `LOKI_CUDA_ARCHITECTURES`; only the default `native` needs a
GPU at configure time. The Dockerfile also checks that the finished module
contains the CUDA backend.

On Docker, smoke-test the module after build:

```bash
docker run --rm loki:cpu python -c "import loki; print(loki.libloki.available_backends())"
docker run --rm --gpus all loki:cuda python -c "import loki; print(loki.libloki.available_backends())"
```

On a cluster that runs Apptainer (Singularity), convert the **local** Docker
tag to a `.sif` (no registry).

```bash
# after docker build ... -t loki:cuda .
apptainer build loki.sif docker-daemon://loki:cuda
apptainer exec --nv loki.sif python -c "import loki; print(loki.libloki.available_backends())"
```

Copy the `.sif` to shared storage if compute nodes cannot build images. There
is no separate Singularity definition file.

**Nextflow (typical HPC, Singularity executor):** point `container` at the
`.sif` path. Use `containerOptions '--nv'` on GPU processes.

```groovy
process LOKI_SEARCH {
    container "${projectDir}/containers/loki.sif"
    containerOptions '--nv'

    script:
    """
    python -c "import loki; print(loki.libloki.available_backends())"
    """
}
```

**Nextflow (Docker executor):** use the local tag, e.g. `container 'loki:cuda'`,
and configure GPUs in your site `nextflow.config`.

A pipeline that already has its own image can copy Loki in instead of
rebuilding it:

```dockerfile
COPY --from=loki:cuda /opt/venv /opt/loki-venv
ENV PATH=/opt/loki-venv/bin:$PATH
```

### HPC clusters (older drivers and compilers)

Loki builds with the system compiler of Debian 12, Ubuntu 22.04 and RHEL 9
(GCC >= 11) and CUDA 12.2+. Check the driver on a **compute node**
(`nvidia-smi`): a CUDA 12.x build runs on driver R525 or newer.

- Conda: [`environment.yml`](environment.yml) is the floor toolchain (CUDA 12.4,
  GCC 12, HDF5 1.14); its header lists the exports and the `pip install`
  line. [`environment-modern.yml`](environment-modern.yml) is the newest one.
  Build on the node type that runs loki: `-march=native` from a login node
  can crash (SIGILL) on older compute nodes. For one install shared by
  several node types, build with `-Ccmake.define.LOKI_ENABLE_NATIVE_ARCH=OFF
  -Ccmake.define.LOKI_CPU_ARCH=x86-64-v2`.
- Containers: [`docker/Dockerfile.floor`](docker/Dockerfile.floor) (docker or
  podman) and [`docker/loki.def`](docker/loki.def) (Apptainer) build the same
  floor image: Ubuntu 22.04, CUDA 12.4, x86-64-v2, SASS `75;80;86;89;90`.

```bash
apptainer build loki.sif docker/loki.def
apptainer exec --nv loki.sif loki --version
```

---

## License

MIT — see [LICENSE](LICENSE).

Candidate clustering for FFA and Taylor EP, the versioned HDF5 schema, and
PulsarX folding export are documented in [Candidate products](docs/candidates.md).
The common EP candidate path supports Taylor searches with RFI control disabled.
