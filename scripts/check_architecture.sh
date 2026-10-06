#!/usr/bin/env bash
# Mechanical checks for the rules in docs/architecture.md.
#
# Usage: scripts/check_architecture.sh
#   CXX=<compiler>   compiler for the standalone public-header check (default c++)
#   SKIP_COMPILE=1   skip the standalone public-header check
#
# Exits non-zero and prints every violation found.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

status=0
fail() {
    printf '  %s\n' "$@"
    status=1
}
section() { printf '%s\n' "$1"; }

# All C++/CUDA sources under the given directories.
sources() { find "$@" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null; }

# 1. Public headers carry no GPU code and no backend macro (rule 1, rule 4).
section "public headers are GPU-free"
while IFS= read -r line; do fail "$line"; done < <(
    grep -rnE 'cuda_runtime|thrust/|cuda/std|cub/|LOKI_ENABLE_' include/)

# 2. Include prefixes (rule 5). Repo headers are "loki/..." (public) or
#    "lib/..." (private). A bare "file.hpp" is a same-directory include.
section "include prefixes are loki/ or lib/"
while IFS= read -r line; do fail "$line"; done < <(
    sources lib tests/cpp bench applications |
        xargs grep -nE '#include "[^"]*/' |
        grep -vE '#include "(loki|lib)/')

section "private lib/ headers stay out of public code"
while IFS= read -r line; do fail "$line"; done < <(
    sources include src applications tests/cpp/api |
        xargs grep -nE '#include "lib/')

# 3. Backend macros only where rule 4 allows them.
section "LOKI_ENABLE_* only in dispatch code, lib/cuda and internal tests"
while IFS= read -r line; do fail "$line"; done < <(
    sources include lib src applications bench tests/cpp |
        grep -vE '^lib/cuda/|^tests/cpp/internal/|^lib/common/(backend\.cpp|dispatch\.hpp)$' |
        grep -vE '^lib/(algorithms|detection|pipelines|search|utils|io|simulation|common)/[a-z_]+\.cpp$' |
        xargs grep -nE 'LOKI_ENABLE_(GPU|CUDA|HIP)')

# 4. Public headers open only the namespaces rule 7 allows.
section "public namespaces follow the directory"
for h in $(find include/loki -mindepth 2 -name '*.hpp'); do
    d=$(basename "$(dirname "$h")")
    case $d in
        common) ok='loki|loki::plans|loki::coord' ;;
        utils) ok='loki::math|loki::memory|loki::psr_utils|loki::detail' ;;
        *) ok="loki::$d" ;;
    esac
    while IFS= read -r ns; do fail "$h: namespace $ns"; done < <(
        grep -oE '^namespace [A-Za-z_:]+' "$h" | awk '{print $2}' | grep -vxE "$ok")
done

# 5. Every public header compiles on its own (rule 1).
if [[ "${SKIP_COMPILE:-0}" != 1 ]]; then
    section "public headers compile standalone"
    cxx=${CXX:-c++}
    for h in $(cd include && find loki -name '*.hpp' | sort); do
        if ! out=$(echo "#include \"$h\"" |
            "$cxx" -std=c++20 -fsyntax-only -Iinclude -x c++ - 2>&1); then
            fail "$h: $(echo "$out" | grep -m1 error)"
        fi
    done
fi

if [[ $status -eq 0 ]]; then
    echo "architecture checks passed"
else
    echo "architecture checks FAILED (see docs/architecture.md)"
fi
exit $status
