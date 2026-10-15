"""Compare two EPMultiPass result files (``runs/<name>/...``) run by run.

Usage: ``python compare_multipass.py BASE.h5 NEW.h5``

Per run: bit-exactness, per-level survivor counts, the Jaccard overlap of the
final candidates (parameters rounded to ~1e-6 relative), the matched |dS/N|,
whether the top-10 agree, and the best score.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


def keys(params: np.ndarray) -> list[tuple[str, ...]]:
    """Parameter values (not errors), rounded to ~1e-6 relative for matching."""
    values = params.reshape(params.shape[0], -1)[:, ::2]
    return [
        tuple(
            np.format_float_positional(x, precision=7, unique=False, fractional=False)
            for x in row
        )
        for row in values
    ]


def compare_run(ra: h5py.Group, rb: h5py.Group) -> tuple[str, bool]:
    la, lb = ra["level_stats"][()], rb["level_stats"][()]
    counts_same = la.shape == lb.shape and all(
        np.array_equal(la[f], lb[f]) for f in la.dtype.names
    )
    first_diff = next(
        (
            i
            for i in range(min(len(la), len(lb)))
            if la["n_leaves_surv"][i] != lb["n_leaves_surv"][i]
        ),
        None,
    )
    pa, pb = ra["param_sets"][()], rb["param_sets"][()]
    sa, sb = ra["scores"][()], rb["scores"][()]
    exact = (
        pa.shape == pb.shape
        and np.array_equal(pa, pb)
        and np.array_equal(sa, sb)
        and np.array_equal(ra["scores_ep"][()], rb["scores_ep"][()])
    )
    ka, kb = keys(pa), keys(pb)
    da = dict(zip(ka, range(len(ka)), strict=False))
    db = dict(zip(kb, range(len(kb)), strict=False))
    inter = set(da) & set(db)
    jaccard = len(inter) / max(1, len(set(da) | set(db)))
    dsn = max((abs(float(sa[da[k]]) - float(sb[db[k]])) for k in inter), default=0.0)
    top_a = [ka[i] for i in np.argsort(-sa, kind="stable")[:10]]
    top_b = [kb[i] for i in np.argsort(-sb, kind="stable")[:10]]
    line = (
        f"n {len(pa)} vs {len(pb)} | bit-exact {exact} | level counts same "
        f"{counts_same} (first survivor diff at level {first_diff}) | Jaccard "
        f"{jaccard:.4f} | max|dS/N| matched {dsn:.2e} | top10 same {top_a == top_b}"
        f" | best {sa.max():.4f} vs {sb.max():.4f}"
    )
    return line, exact and counts_same


def main(a: str, b: str) -> None:
    all_exact = True
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for run in fa["runs"]:
            if run not in fb["runs"]:
                print(f"{run}: missing in {b}")
                all_exact = False
                continue
            line, exact = compare_run(fa["runs"][run], fb["runs"][run])
            all_exact &= exact
            print(f"{run}: {line}")
    print("BIT-EXACT" if all_exact else "NOT BIT-EXACT")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
