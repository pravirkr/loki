#!/usr/bin/env python3
"""Fit the brute-vs-merge cost-model constants from ``bseg_brute_sweep.py`` JSON.

The ``bseg_brute`` selector in ``lib/configs.cpp`` minimises

    cost(B) = W_brute * brute_ops(B) + W_table * F0 * B + W_merge * merge_ops(B)

The time-domain selector no longer minimises this: it takes the largest power
of two segment with ``B * tsamp * f_max <= 2`` periods, because the run-length
brute fold costs about one merge level at that depth. Pass ``--tsamp`` to
score chunks with the run-length op count

    brute_ops = 2 * (nsamps / B) * F0 * nbins * ceil(B * tsamp * f_max)
    table_entries = F0 * nbins * ceil(B * tsamp * f_max)

and refit ``W_*`` from a 1-thread ``bseg_brute_sweep.py`` JSON. Without
``--tsamp`` the legacy gather-add counts are used.

Usage:
    python bench/fit_cost_model.py --nsamps 8388608 --nbins-min-lossy-bf 32 \
        sweep23.json [--fourier]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

MIN_TIME_S = 0.01  # ignore chunks too short to time reliably


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--nsamps", type=int, required=True)
    ap.add_argument("--fourier", action="store_true")
    ap.add_argument("--nbins-min-lossy-bf", type=int, default=64)
    ap.add_argument(
        "--tsamp",
        type=float,
        default=None,
        help="sample interval; enables the run-length brute op count",
    )
    args = ap.parse_args()

    runs = []
    for path in args.json:
        runs.extend(json.load(open(path)))

    brute_rate = defaultdict(list)  # seconds per brute op (by mode)
    table_rate = []  # seconds per table entry
    merge_rate = defaultdict(list)  # seconds per merge element-op (by mode)
    per_chunk = defaultdict(list)  # (nbins) -> [(B, total)]

    for run in runs:
        for c in run["chunks"]:
            nbins = c["nbins"]
            nbins_f = nbins // 2 + 1
            direct = args.fourier and nbins <= args.nbins_min_lossy_bf
            mode = (
                "fourier_direct"
                if direct
                else ("fourier_lossy" if args.fourier else "time")
            )
            exec_s = c["brute"] - c["table"]
            f0, B = c["nfreqs0"], c["bseg"]
            f_range = c.get("f_range") or (0.0, 0.0)
            if args.tsamp and not args.fourier and f_range[1] > 0.0:
                periods = max(1.0, math.ceil(B * args.tsamp * f_range[1] - 1e-12))
                ops_b = 2.0 * (args.nsamps / B) * f0 * nbins * periods
                table_entries = f0 * nbins * periods
            else:
                ops_b = 2.0 * args.nsamps * f0 * (nbins_f if direct else 1)
                table_entries = f0 * B
            if exec_s > MIN_TIME_S and ops_b > 0.0:
                brute_rate[mode].append(exec_s / ops_b)
            if c["table"] > 0.002 and table_entries > 0.0:
                table_rate.append(c["table"] / table_entries)
            width = nbins_f if args.fourier else nbins
            ops_m = max(c["levels"] - 1, 1) * 2.0 * c["ncoords_top"] * width
            if c["ffa"] > MIN_TIME_S:
                merge_rate[mode].append(c["ffa"] / ops_m)
            per_chunk[nbins].append((B, c["brute"] + c["ffa"]))

    ref = statistics.median(brute_rate["fourier_lossy" if args.fourier else "time"])
    print(
        f"reference brute gather-add: {ref * 1e9:.4f} ns/op ({1e-9 / ref:.1f} Gops/s)"
    )
    if table_rate:
        t = statistics.median(table_rate)
        print(f"table build: {t * 1e9:.3f} ns/entry -> W_table = {t / ref:.1f}")
    for mode, vals in merge_rate.items():
        m = statistics.median(vals)
        print(f"merge[{mode}]: {m * 1e9:.3f} ns/elem-op -> W_merge = {m / ref:.2f}")
    for mode, vals in brute_rate.items():
        b = statistics.median(vals)
        print(f"brute[{mode}]: {b * 1e9:.4f} ns/op -> relative {b / ref:.2f}")

    print("\nmeasured optimum per nbins (min of brute+merge across the sweep):")
    print("  nbins   best_B   B/nbins   total_s   default_total_s")
    for nbins, pts in sorted(per_chunk.items()):
        pts.sort()
        best = min(pts, key=lambda p: p[1])
        print(f"  {nbins:5d} {best[0]:8d} {best[0] / nbins:9.1f} {best[1]:9.3f}")


if __name__ == "__main__":
    main()
