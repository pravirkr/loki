"""Compare two loki FFA result files under the numerics contract.

Usage: ``python compare_ffa.py A.h5 B.h5 [--top 20]``

Rows are matched on (parameters, nbins, width); parameters must be identical,
so compare runs of the same chunking (a different host compiler, or a build
without ``-march=native``, can move grid parameters in the last bits). Prints
the Jaccard overlap, the matched |dS/N| (max, p99, relative max, fraction
exact), whether the top-N sets agree, and the best candidate of each file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import h5py
import numpy as np


def load(path: str) -> tuple[list[tuple], np.ndarray, list[str]]:
    with h5py.File(path, "r") as f:
        params = f["param_sets"][...]
        snr = f["snr"][...]
        width = f["width"][...]
        nbins = f["nbins"][...]
        names = [str(n) for n in f.attrs["param_names"]]
    keys = [
        (*params[i].tolist(), int(nbins[i]), int(width[i])) for i in range(len(snr))
    ]
    return keys, snr, names


def main(a: str, b: str, top: int) -> None:
    ka, sa, names = load(a)
    kb, sb, _ = load(b)
    da: dict[tuple, list[float]] = defaultdict(list)
    db: dict[tuple, list[float]] = defaultdict(list)
    for k, s in zip(ka, sa, strict=True):
        da[k].append(float(s))
    for k, s in zip(kb, sb, strict=True):
        db[k].append(float(s))
    matched = 0
    absd: list[float] = []
    rel: list[float] = []
    only_a: list[float] = []
    only_b: list[float] = []
    for k in set(da) | set(db):
        xa, xb = sorted(da.get(k, [])), sorted(db.get(k, []))
        m = min(len(xa), len(xb))
        matched += m
        for x, y in zip(xa[:m], xb[:m], strict=True):
            absd.append(abs(x - y))
            rel.append(abs(x - y) / max(abs(x), 1e-12))
        only_a += xa[m:]
        only_b += xb[m:]
    d = np.array(absd or [0.0])
    r = np.array(rel or [0.0])
    jaccard = matched / max(len(sa) + len(sb) - matched, 1)
    print(
        f"rows A={len(sa)} B={len(sb)} matched={matched} onlyA={len(only_a)} "
        f"onlyB={len(only_b)} jaccard={jaccard:.6f}"
    )
    print(
        f"matched |dS/N|: max={d.max():.3e} p99={np.percentile(d, 99):.3e} "
        f"rel max={r.max():.3e} exact={np.mean(d == 0) * 100:.2f}%"
    )
    if only_a or only_b:
        print(
            f"unmatched S/N max: A={max(only_a, default=0):.3f} "
            f"B={max(only_b, default=0):.3f}"
        )
    ia = np.argsort(-sa, kind="stable")[:top]
    ib = np.argsort(-sb, kind="stable")[:top]
    same = {ka[i] for i in ia} == {kb[i] for i in ib}
    print(f"top{top} key sets identical: {same}")
    for label, keys, snr, idx in (("A", ka, sa, ia), ("B", kb, sb, ib)):
        if len(idx) == 0:
            continue
        best = keys[idx[0]]
        params = dict(zip(names, best[:-2], strict=False))
        print(
            f"best {label}: S/N={snr[idx[0]]:.5f} {params} "
            f"nbins={best[-2]} w={best[-1]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    main(args.a, args.b, args.top)
