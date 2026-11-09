#!/usr/bin/env bash
# Golden-output harness: seeded runs to compare two loki builds (e.g. main vs a branch).
# Usage: run_golden.sh <loki-binary> <outdir> [backends...]   (default: cpu cuda)
# Seeded simulate -> FFA search and accelerated EP search, per backend.
# Compare two outdirs with compare_h5.py. Small: runs in well under a minute.
# The EP case never overflows the world tree, so it does not exercise the
# overflow (trimming) path; a change there needs its own check.
set -euo pipefail
LOKI=$(readlink -f "$1")
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
shift 2
BACKENDS=("${@:-cpu cuda}")
[ $# -eq 0 ] && BACKENDS=(cpu cuda)
# 32768 samples in 8 EP segments: ep.nsegments since ab3637c, bseg_ffa before.
EP_HELP=$("$LOKI" search ep --help 2>&1 || true)
if grep -q -- '--nsegments' <<<"$EP_HELP"; then
  EP_SEG_KEY="nsegments = 8"; EP_SEG_SECTION=ep
else
  EP_SEG_KEY="bseg_ffa = 4096"; EP_SEG_SECTION=performance
fi
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$OUT"

"$LOKI" -q simulate --period 0.01 --dt 0.0001 --nsamps 65536 --snr 30 --seed 1 -o ffa.tim
"$LOKI" -q simulate --period 0.0070921985815602835 --dt 0.000064 --nsamps 32768 \
  --snr 30 --seed 1 --acc 5 -o ep.tim

for be in "${BACKENDS[@]}"; do
  cat > "ffa_$be.toml" <<EOF
[input]
timeseries = "ffa.tim"
[preprocessing]
preprocess = true
[search]
f_min = 99.0
f_max = 101.0
nbins = 64
[performance]
nthreads = 4
max_process_memory_gb = 4.0
backend = "$be"
[output]
outdir = "$be"
prefix = "ffa"
EOF
  cat > "ep_$be.toml" <<EOF
[input]
timeseries = "ep.tim"
[preprocessing]
preprocess = false
[search]
f_min = 140.0
f_max = 142.0
acc_min = -10.0
acc_max = 10.0
nbins = 32
[ep]
n_runs = 1
min_pd = 0.1
prune_poly_order = 2
$([ "$EP_SEG_SECTION" = ep ] && echo "$EP_SEG_KEY")
[performance]
nthreads = 4
max_process_memory_gb = 4.0
$([ "$EP_SEG_SECTION" = performance ] && echo "$EP_SEG_KEY")
backend = "$be"
[output]
outdir = "$be"
prefix = "ep"
EOF
  "$LOKI" -q search ffa --config "ffa_$be.toml" > "ffa_$be.log" 2>&1
  "$LOKI" -q search ep --config "ep_$be.toml" --no-plan-cache > "ep_$be.log" 2>&1
done
find "$OUT" -name '*.h5' | sort
