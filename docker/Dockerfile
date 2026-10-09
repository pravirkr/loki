# Loki image: the library, its Python module, and the loki command.
# It does not include dedispersion, filterbank tools, or injection codes.
# Downstream pipelines pull this image, or copy /opt/venv into a larger one.
#
#   docker build -f docker/Dockerfile --target cpu  -t loki:cpu  .
#   docker build -f docker/Dockerfile --target cuda -t loki:cuda .
#
# Defaults: portable x86-64-v3 (override with --build-arg LOKI_MARCH=x86-64)
# and CUDA architectures 80;86;90 (override with LOKI_CUDA_ARCHITECTURES).

ARG UV_VERSION=0.12.18
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.6.3
ARG LOKI_MARCH=x86-64-v3
ARG JOBS=8

# uv binary only. The image does not use the "release" CMake preset: that
# preset builds tests and leaves -march=native and CUDA architectures at
# "native", which is the wrong binary for a machine other than the builder.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

# ---- shared build packages (CPU image and CUDA image) ----
FROM ubuntu:${UBUNTU_VERSION} AS cpu-build

ARG LOKI_MARCH
ARG JOBS
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    CMAKE_BUILD_PARALLEL_LEVEL=${JOBS} \
    CFLAGS=-march=${LOKI_MARCH} \
    CXXFLAGS=-march=${LOKI_MARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libfftw3-dev \
        libgomp1 \
        libhdf5-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /usr/local/bin/uv
RUN uv venv /opt/venv --python /usr/bin/python3 \
    && uv pip install "scipy>=1.14"

WORKDIR /src
COPY . /src

# pyloki is a runtime dependency and is fetched while the package is built.
# These -C flags are the portable image settings. They are not the release preset.
RUN uv pip install . \
        -C cmake.define.LOKI_CUDA=OFF \
        -C cmake.define.LOKI_ENABLE_NATIVE_ARCH=OFF \
        -C cmake.define.LOKI_BUILD_TESTING=OFF \
        -C cmake.define.LOKI_BUILD_BENCHMARKS=OFF \
        -C cmake.define.LOKI_BUILD_DOCS=OFF \
    && python -c "import loki; backends = loki.libloki.available_backends(); assert backends == ['cpu'], backends" \
    && command -v loki

# ---- CPU runtime ----
FROM ubuntu:${UBUNTU_VERSION} AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HDF5_USE_FILE_LOCKING=FALSE \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        libfftw3-double3 \
        libfftw3-single3 \
        libgomp1 \
        libhdf5-310 \
        libhdf5-hl-310 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cpu-build /opt/venv /opt/venv

LABEL org.opencontainers.image.title="loki" \
      org.opencontainers.image.description="Loki pulsar-search library (CPU)" \
      org.opencontainers.image.source="https://github.com/pravirkr/loki"

WORKDIR /data
CMD ["python3", "-c", "import loki; print(loki.libloki.available_backends())"]

# ---- CUDA build ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cuda-build

ARG LOKI_MARCH
ARG LOKI_CUDA_ARCHITECTURES="80;86;90"
ARG JOBS
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    CMAKE_BUILD_PARALLEL_LEVEL=${JOBS} \
    CFLAGS=-march=${LOKI_MARCH} \
    CXXFLAGS=-march=${LOKI_MARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libfftw3-dev \
        libgomp1 \
        libhdf5-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /usr/local/bin/uv
RUN uv venv /opt/venv --python /usr/bin/python3 \
    && uv pip install "scipy>=1.14"

WORKDIR /src
COPY . /src

# LOKI_CUDA=ON refuses to configure unless nvidia-smi sees a GPU, which a
# normal image build does not have. AUTO compiles CUDA when nvcc is present.
RUN uv pip install . \
        -C cmake.define.LOKI_CUDA=AUTO \
        -C cmake.define.LOKI_ENABLE_NATIVE_ARCH=OFF \
        -C cmake.define.LOKI_CUDA_ARCHITECTURES="${LOKI_CUDA_ARCHITECTURES}" \
        -C cmake.define.CMAKE_CUDA_FLAGS="-Xcompiler=-march=${LOKI_MARCH}" \
        -C cmake.define.LOKI_BUILD_TESTING=OFF \
        -C cmake.define.LOKI_BUILD_BENCHMARKS=OFF \
        -C cmake.define.LOKI_BUILD_DOCS=OFF \
    && python -c "import loki; backends = loki.libloki.available_backends(); assert 'cuda' in backends, backends" \
    && command -v loki

# ---- CUDA runtime ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS cuda

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HDF5_USE_FILE_LOCKING=FALSE \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        libfftw3-double3 \
        libfftw3-single3 \
        libgomp1 \
        libhdf5-310 \
        libhdf5-hl-310 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda-build /opt/venv /opt/venv

LABEL org.opencontainers.image.title="loki" \
      org.opencontainers.image.description="Loki pulsar-search library (CPU and CUDA)" \
      org.opencontainers.image.source="https://github.com/pravirkr/loki"

WORKDIR /data
CMD ["python3", "-c", "import loki; print(loki.libloki.available_backends())"]
