#!/usr/bin/env bash
# Architecture, pre-commit lint/format hooks, and clang-tidy — all failures reported at end.
#
# Expects a finished CMake build at BUILD_DIR (for run-clang-tidy).
#
# Usage:
#   LLVM_VERSION=23 BUILD_DIR=build-ci-tidy scripts/ci_static_analysis.sh

set -uo pipefail

LLVM_VERSION=${LLVM_VERSION:-23}
BUILD_DIR=${BUILD_DIR:-build-ci-tidy}
CLANG_TIDY="clang-tidy-${LLVM_VERSION}"
RUN_CLANG_TIDY="run-clang-tidy-${LLVM_VERSION}"
ARCH_CXX="clang++-${LLVM_VERSION}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

failures=()

run_step() {
    local name=$1
    shift
    printf '\n======== %s ========\n' "$name"
    if "$@"; then
        printf 'OK: %s\n' "$name"
    else
        printf 'FAILED: %s\n' "$name"
        failures+=("$name")
    fi
}

run_step "architecture (docs/architecture.md, standalone header compile)" \
    env CXX="$ARCH_CXX" scripts/check_architecture.sh

# Same hooks as .pre-commit-config.yaml except check-architecture (full compile above).
pre_commit_hooks=(
    trailing-whitespace
    end-of-file-fixer
    mixed-line-ending
    check-yaml
    check-toml
    check-merge-conflict
    check-added-large-files
    check-case-conflict
    clang-format
    cmake-format
    cmake-lint
    ruff-check
    ruff-format
)

for hook in "${pre_commit_hooks[@]}"; do
    run_step "pre-commit:${hook}" pre-commit run "$hook" --all-files
done

run_step "clang-tidy" \
    "$RUN_CLANG_TIDY" -quiet -p "$BUILD_DIR" \
    -clang-tidy-binary "$CLANG_TIDY" \
    -warnings-as-errors='*' -j "$(nproc)" \
    "$repo_root/(lib|src|tests|applications)/"

printf '\n======== summary ========\n'
if ((${#failures[@]} == 0)); then
    echo "All static analysis checks passed."
    exit 0
fi

echo "The following checks failed:"
for name in "${failures[@]}"; do
    echo "  - $name"
done
exit 1
