"""Compare two EPFreqSweep result files: same candidates (as a set), same stats.

Usage: python compare_sweep.py BASELINE.h5 NEW.h5 [--rtol 1e-5]
Exit code 0 when equivalent (up to float rounding), 1 otherwise.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np

COUNT_FIELDS = (
    "level",
    "seg_idx",
    "n_branches",
    "n_leaves",
    "n_leaves_phy",
    "n_leaves_surv",
)
FLOAT_FIELDS = ("threshold", "score_min", "score_max")


def sorted_run(run: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = run["param_sets"][()].reshape(run["param_sets"].shape[0], -1)
    scores = run["scores"][()]
    scores_ep = run["scores_ep"][()]
    order = np.lexsort(params.T[::-1])
    return params[order], scores[order], scores_ep[order]


def compare(base_file: str, new_file: str, rtol: float) -> bool:
    ok = True
    with h5py.File(base_file, "r") as fb, h5py.File(new_file, "r") as fn:
        for chunk in fb["chunks"]:
            for run_name in fb["chunks"][chunk]["runs"]:
                rb = fb["chunks"][chunk]["runs"][run_name]
                rn = fn["chunks"][chunk]["runs"][run_name]
                sb, sn = rb["level_stats"][()], rn["level_stats"][()]
                if sb.shape != sn.shape:
                    print(f"{chunk}/{run_name}: level count {sb.shape} vs {sn.shape}")
                    ok = False
                    continue
                for field in COUNT_FIELDS:
                    if not np.array_equal(sb[field], sn[field]):
                        bad = np.flatnonzero(sb[field] != sn[field])
                        print(
                            f"{chunk}/{run_name}: {field} differs at levels {bad[:10]}"
                        )
                        ok = False
                for field in FLOAT_FIELDS:
                    if not np.allclose(sb[field], sn[field], rtol=rtol, atol=1e-6):
                        d = np.max(np.abs(sb[field] - sn[field]))
                        print(f"{chunk}/{run_name}: {field} max abs diff {d}")
                        ok = False
                pb, scb, sepb = sorted_run(rb)
                pn, scn, sepn = sorted_run(rn)
                if pb.shape != pn.shape:
                    print(f"{chunk}/{run_name}: leaves {pb.shape} vs {pn.shape}")
                    ok = False
                    continue
                exact = np.array_equal(pb, pn)
                if not np.allclose(pb, pn, rtol=rtol, atol=0.0):
                    print(f"{chunk}/{run_name}: param sets differ")
                    ok = False
                if not np.allclose(scb, scn, rtol=rtol, atol=1e-6):
                    print(
                        f"{chunk}/{run_name}: scores max diff "
                        f"{np.max(np.abs(scb - scn))}"
                    )
                    ok = False
                if not np.allclose(sepb, sepn, rtol=rtol, atol=1e-6):
                    print(
                        f"{chunk}/{run_name}: scores_ep max diff "
                        f"{np.max(np.abs(sepb - sepn))}"
                    )
                    ok = False
                same_order = np.array_equal(rb["scores"][()], rn["scores"][()])
                print(
                    f"{chunk}/{run_name}: {pb.shape[0]} leaves, "
                    f"params bit-exact={exact}, same order={same_order}, "
                    f"top score {scb.max():.4f} vs {scn.max():.4f}"
                )
    print("EQUIVALENT" if ok else "DIFFERENT")
    return ok


if __name__ == "__main__":
    rtol = 1e-5
    args = sys.argv[1:]
    if "--rtol" in args:
        i = args.index("--rtol")
        rtol = float(args[i + 1])
        del args[i : i + 2]
    sys.exit(0 if compare(args[0], args[1], rtol) else 1)
