"""Summaries from an nsys sqlite export (nsys export --type sqlite).

Usage: python analyze_nsys.py REPORT.sqlite [--cpu]

Prints the GPU busy fraction over the prune window (first to last kernel of
the pruning loop), and with --cpu the top self/inclusive host symbols from
CPU sampling (needs `nsys profile -s cpu --backtrace=fp`).
"""

from __future__ import annotations

import contextlib
import sqlite3
import sys
from collections import Counter


def gpu_busy(c: sqlite3.Connection) -> None:
    rows = c.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on k.shortName = s.id order by k.start"
    ).fetchall()
    prune = [r for r in rows if "shift_add" in r[2] or "analyze_and_branch" in r[2]]
    t0, t1 = prune[0][0], prune[-1][1]
    busy = 0
    cur_s, cur_e = None, None
    intervals = [(s, e) for s, e, _ in rows if e >= t0 and s <= t1]
    with contextlib.suppress(sqlite3.OperationalError):
        intervals += [
            (s, e)
            for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")
            if e >= t0 and s <= t1
        ]
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = t1 - t0
    print(
        f"prune window {span / 1e9:.3f} s, "
        f"GPU busy {busy / 1e9:.3f} s ({100 * busy / span:.1f}%)"
    )


def cpu_hot(c: sqlite3.Connection, top: int = 40) -> None:
    chains: dict[int, list[tuple[int, str]]] = {}
    for cid, sym, depth in c.execute(
        "select cc.id, s.value, cc.stackDepth from SAMPLING_CALLCHAINS cc "
        "join StringIds s on cc.symbol = s.id"
    ):
        chains.setdefault(cid, []).append((depth, sym))
    self_c: Counter[str] = Counter()
    incl_c: Counter[str] = Counter()
    for frames in chains.values():
        frames.sort()
        self_c[frames[0][1]] += 1
        for sym in {f[1] for f in frames}:
            incl_c[sym] += 1
    n = len(chains)
    print(f"{n} samples")
    print("--- self ---")
    for sym, k in self_c.most_common(top):
        print(f"{100 * k / n:6.2f}%  {sym[:150]}")
    print("--- inclusive ---")
    for sym, k in incl_c.most_common(top):
        print(f"{100 * k / n:6.2f}%  {sym[:150]}")


if __name__ == "__main__":
    conn = sqlite3.connect(sys.argv[1])
    gpu_busy(conn)
    if "--cpu" in sys.argv:
        cpu_hot(conn)
