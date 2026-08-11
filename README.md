# loki

Leverage Optimal significance to unveil Keplerian orbIt pulsars

A high-performance **C++20** pulsar searching library with **Python bindings**. The same core library supports **CPU-only**, **GPU-accelerated**, and **CPU+GPU** builds: CPU paths always compile; CUDA paths are added when the CUDA toolchain is enabled at configure time.

---

## Requirements

### Compilers

- **GCC >= 13.2** or **Clang >= 18** (C++20 support required)
- **CUDA >= 12.6** *(optional, for GPU acceleration)*
- **CMake >= 3.18**
- **Python >= 3.12** *(for Python bindings)*

MSVC is not supported.

### System libraries (always required)

These are **never** downloaded by loki and must be discoverable by CMake on your system — install them before building. The recommended way is via [conda-forge](https://conda-forge.org) or [mamba](https://mamba.readthedocs.io/en/latest/):

| Library | Minimum Version | conda-forge install |
| --------- | ----------------- | --------------------- |
| HDF5 | - | `mamba install hdf5` |
| FFTW (float + OpenMP) | - | `mamba install fftw` |
| OpenMP | - | `mamba install libomp` *(macOS)* / `libgomp` *(Linux)* |
| CMake | 3.18 | `mamba install cmake>=3.18` |
| Ninja | - | `mamba install ninja` |
| GCC | 13.2 | `mamba install gcc>=13.2 gxx>=13.2` *(Linux)* |
| Python | 3.12 | `mamba install python>=3.12` |

Header-only C++ dependencies (fmt, spdlog, HighFive, CLI11, xsimd, etc.) are fetched automatically via [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) unless `LOKI_USE_SYSTEM_DEPS=ON`.

### GPU and toolchain support policy

Frozen minimum requirements for CUDA builds:

| Component | Minimum |
| --------- | ------- |
| CUDA Toolkit | 12.6 |
| GCC (when used as nvcc host compiler) | 13.2 |
| CMake | 3.18 |
| GPU compute capability | **sm_50** (Maxwell) |

- CUDA Toolkit 12.6+, with a host compiler supported by that toolkit.
- For CUDA 12.6 specifically: GCC 7.3–13.2, or Clang 7–18.

---

## Installation modes

### Mode A — Python (`uv pip install`) *(recommended for pipelines)*

Best for using loki from Python in a conda environment. Builds `libloki` (CPU) and, when CUDA is available, `libculoki` (GPU).

```bash
mamba create -n loki_env python=3.12
mamba activate loki_env
mamba install -c conda-forge cmake>=3.18 ninja hdf5 fftw libomp  # macOS
# Linux: also install gcc>=13.2 gxx>=13.2

export CPM_SOURCE_CACHE="$HOME/.cache/CPM"   # optional; avoids re-downloading CPM deps

uv pip install git+https://github.com/pravirkr/loki.git
```

#### CUDA modes (`LOKI_CUDA`)

| Value | Behaviour |
| ------- | ----------- |
| **`AUTO`** *(default)* | Build GPU support if `nvcc` is found; otherwise CPU-only. |
| **`ON`** | **Require** CUDA toolkit, NVIDIA driver/GPU (`nvidia-smi`), and MathDX. **Hard fail** if any are missing. |
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

Pre-built wheels (when published via cibuildwheel) use `LOKI_ENABLE_NATIVE_ARCH=OFF` so they run on a wide range of CPUs.

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

When loki was built **with CUDA**, the installed package also requires `CUDAToolkit` and defines the preprocessor macro **`LOKI_ENABLE_CUDA`** on the `loki::loki` target so GPU declarations in public headers are visible.

---

## CMake options reference

Project-specific options use the **`LOKI_`** prefix. Standard CMake options keep their usual names.

| Option | Default | Description |
| -------- | ----------------- | --------------------- |
| **`LOKI_CUDA`** | `AUTO` | CUDA build mode: `AUTO`, `ON`, or `OFF` |
| **`LOKI_CUDA_ARCHITECTURES`** | `native` | Passed to `CMAKE_CUDA_ARCHITECTURES` (`native`, `61;80`, `all-major`, …). Minimum supported GPU: **sm_50**. |
| **`LOKI_ENABLE_NATIVE_ARCH`** | `ON` | Add `-march=native` in Release builds |
| **`LOKI_USE_SYSTEM_DEPS`** | `OFF` | Prefer system CPM packages over pinned downloads |
| **`LOKI_BUILD_PYTHON`** | `ON` | Build Python extension modules |
| **`LOKI_BUILD_TESTING`** | `OFF` | Build C++ Catch2 tests |
| **`LOKI_BUILD_BENCHMARKS`** | `OFF` | Build Google Benchmark executables |
| **`LOKI_BUILD_DOCS`** | `OFF` | Build documentation |
| **`LOKI_ENABLE_COVERAGE`** | `OFF` | Compile with `--coverage` |
| **`LOKI_ENABLE_IPO`** | `OFF` | Link-time optimization (Release) |
| **`BUILD_SHARED_LIBS`** | `ON` | `ON` = shared `libloki`, `OFF` = static `libloki.a` |

### CPU vs GPU vs CPU+GPU

- **CPU-only** (`LOKI_CUDA=OFF`): all `.cpp` sources; no `.cu`; no `LOKI_ENABLE_CUDA` macro; Python module `libloki` only.
- **CPU+GPU** (`LOKI_CUDA=AUTO` or `ON` with working toolchain): `.cpp` **and** `.cu` in one library; `LOKI_ENABLE_CUDA` defined; Python gets `libloki` + `libculoki`.
- The library is always usable on CPU; GPU code paths are compiled only when CUDA is enabled at configure time.

### GPU architecture

- **Minimum GPU:** sm_50 (Maxwell). CMake rejects lower values in `LOKI_CUDA_ARCHITECTURES`.
- **Fat binaries:** pass multiple SMs, e.g. `-DLOKI_CUDA_ARCHITECTURES='61;80'` or `all-major`.
- **MathDX / cuRANDDx** is downloaded and linked only when any target arch is sm_70+ and `LOKI_FORCE_CURAND_RNG=OFF`.

---

## Python vs static libraries on macOS

Python modules are built as **shared bundles** (`.so`). With **`BUILD_SHARED_LIBS=ON`** (default for pip installs), `libloki` is a shared library the extension links against cleanly.

With **`BUILD_SHARED_LIBS=OFF`**, loki becomes a static archive (`.a`). Embedding static libraries into macOS Python extensions often causes **duplicate or unresolved symbols** at link time because the linker treats bundles differently from normal executables. For Python bindings, keep **`BUILD_SHARED_LIBS=ON`**.

For pure C++ static linking (`LOKI_BUILD_PYTHON=OFF`), `BUILD_SHARED_LIBS=OFF` is fine.

---

## Optional: CPM source cache

```bash
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"
```

Add to `~/.bashrc` or `~/.zshrc` to cache downloaded dependencies across rebuilds.

---

## Development builds

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DLOKI_BUILD_TESTING=ON \
  -DLOKI_CUDA=AUTO

cmake --build build -j
ctest --test-dir build
```

---

## License

MIT — see [LICENSE](LICENSE).
