#!/usr/bin/env bash
# Configure is done; run ci-tidy build in parallel with prebuild lint, then clang-tidy.
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

FAILURES_FILE="${repo_root}/.ci-static-failures"
export FAILURES_FILE
export CI_AGGREGATE=1
export LLVM_VERSION="${LLVM_VERSION:-23}"
export BUILD_DIR="${BUILD_DIR:-build-ci-tidy}"

: >"$FAILURES_FILE"

cmake --build --preset ci-tidy --parallel "$(nproc)" &
build_pid=$!

PHASE=prebuild scripts/ci_static_analysis.sh

if ! wait "$build_pid"; then
    printf 'cmake build (ci-tidy)\n' >>"$FAILURES_FILE"
fi

PHASE=postbuild scripts/ci_static_analysis.sh
exit $?
