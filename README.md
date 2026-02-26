# loki

Leverage Optimal significance to unveil Keplerian orbIt pulsars

A high-performance C++20 pulsar searching library with Python bindings.

---

## Requirements

Before installing, ensure your system meets the following requirements.

### Compilers

- **GCC >= 14** or **Clang >= 18** (C++20 support required)
- **CUDA >= 12.6** *(optional, for GPU acceleration)*

### Build Tools

- **CMake >= 3.30**
- **Ninja** *(recommended)* or GNU Make

### System Libraries

The following must be installed and discoverable by CMake on your system.
The recommended way is via [conda-forge](https://conda-forge.org) or [mamba](https://mamba.readthedocs.io/en/latest/):

| Library | Minimum Version | conda-forge install |
|---------|-----------------|---------------------|
| Python | 3.12 | `conda install python>=3.12` |
| HDF5 | 2.0 | `conda install hdf5>=2.0` |
| FFTW | 3.3 | `conda install fftw>=3.3` |
| OpenMP / libomp | 5.0 | `conda install libomp` *(macOS)* |
| CMake | 3.30 | `conda install cmake>=3.30` |
| Ninja | any | `conda install ninja` |
| GCC 14 | 14.0 | `conda install gcc>=14 gxx>=14` *(Linux)* |

---

## Installation

### 1. Set Up a Conda Environment *(recommended)*

```bash
mamba create -n loki python>=3.12
mamba activate loki

# Install system C++ libraries
mamba install -c conda-forge cmake>=3.30 ninja hdf5>=2.0 fftw>=3.3

# Linux: install GCC 14 and OpenMP
mamba install -c conda-forge gcc>=14 gxx>=14

# macOS: install libomp (Clang via Xcode is used automatically)
mamba install -c conda-forge libomp
```

### 2. (Optional) Set a Persistent CPM Download Cache

CMake downloads header-only C++ libraries on first build. Set this environment variable once to cache them permanently and avoid re-downloads across builds:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"
```

### 3. Install LOKI

```bash
uv pip install git+https://github.com/pravirkr/loki.git
```

### 4. GPU Build (Optional)

If you have a CUDA-capable GPU with CUDA Toolkit >= 12.6 installed:

```bash
# Auto-detect GPU (default behaviour — uses GPU if found, CPU otherwise)
uv pip install git+https://github.com/pravirkr/loki.git

# Explicitly require GPU — fails with a clear error if CUDA is not found
uv pip install git+https://github.com/pravirkr/loki.git -C cmake.define.ENABLE_CUDA=ON

# Explicitly disable GPU even if CUDA is installed
uv pip install git+https://github.com/pravirkr/loki.git -C cmake.define.ENABLE_CUDA=OFF
```
