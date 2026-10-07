#!/usr/bin/env bash
# Architecture, pre-commit lint/format hooks, and clang-tidy.
#
# PHASE:
#   prebuild  — architecture + pre-commit (no compile DB)
#   postbuild — clang-tidy (needs BUILD_DIR)
#   all       — default; run everything sequentially
#
# CI_AGGREGATE=1: prebuild/postbuild record failures and only exit non-zero on
# postbuild or all after printing the combined summary (for parallel lint job).
#
# Usage:
#   LLVM_VERSION=23 BUILD_DIR=build-ci-tidy scripts/ci_static_analysis.sh

set -uo pipefail

LLVM_VERSION=${LLVM_VERSION:-23}
BUILD_DIR=${BUILD_DIR:-build-ci-tidy}
PHASE=${PHASE:-all}
CI_AGGREGATE=${CI_AGGREGATE:-0}
FAILURES_FILE=${FAILURES_FILE:-}

CLANG_TIDY="clang-tidy-${LLVM_VERSION}"
RUN_CLANG_TIDY="run-clang-tidy-${LLVM_VERSION}"
ARCH_CXX="clang++-${LLVM_VERSION}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

failures=()

record_failure() {
    local name=$1
    failures+=("$name")
    if [[ -n $FAILURES_FILE ]]; then
        printf '%s\n' "$name" >>"$FAILURES_FILE"
    fi
}

run_step() {
    local name=$1
    shift
    printf '\n======== %s ========\n' "$name"
    if "$@"; then
        printf 'OK: %s\n' "$name"
    else
        printf 'FAILED: %s\n' "$name"
        record_failure "$name"
    fi
}

run_prebuild() {
    run_step "architecture (docs/architecture.md, standalone header compile)" \
        env CXX="$ARCH_CXX" scripts/check_architecture.sh

    local pre_commit_hooks=(
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
}

run_postbuild() {
    run_step "clang-tidy" \
        "$RUN_CLANG_TIDY" -quiet -p "$BUILD_DIR" \
        -clang-tidy-binary "$CLANG_TIDY" \
        -warnings-as-errors='*' -j "$(nproc)" \
        "$repo_root/(lib|src|tests|applications)/"
}

print_summary() {
    local -a all_failures=()
    if [[ -n $FAILURES_FILE && -f $FAILURES_FILE ]]; then
        mapfile -t all_failures <"$FAILURES_FILE"
    else
        all_failures=("${failures[@]}")
    fi

    printf '\n======== summary ========\n'
    if ((${#all_failures[@]} == 0)); then
        echo "All static analysis checks passed."
        return 0
    fi

    echo "The following checks failed:"
    local seen=""
    local name
    for name in "${all_failures[@]}"; do
        if [[ "$seen" != *"|$name|"* ]]; then
            echo "  - $name"
            seen+="|$name|"
        fi
    done
    return 1
}

if [[ -n $FAILURES_FILE && $PHASE == prebuild && $CI_AGGREGATE == 1 ]]; then
    : >"$FAILURES_FILE"
fi

case "$PHASE" in
    prebuild)
        run_prebuild
        if [[ $CI_AGGREGATE == 1 ]]; then
            exit 0
        fi
        print_summary
        ;;
    postbuild)
        run_postbuild
        print_summary
        ;;
    all)
        run_prebuild
        run_postbuild
        print_summary
        ;;
    *)
        echo "Unknown PHASE=$PHASE (expected prebuild, postbuild, or all)" >&2
        exit 2
        ;;
esac
