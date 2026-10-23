#!/usr/bin/env python3
"""Sensitivity-versus-time sweep of ``performance.bseg_brute`` (one FFA chunk).

For each injected pulsar the search window is a single chunk of ratio
``--ratio`` (default 1.5, the planner's ``octave_scale``) around the true
frequency, so one explicit ``bseg_brute`` applies to exactly one chunk. Every
requested B is searched on the same time series; the best S/N near the truth
is compared with the best S/N over all B values of that case, and the brute /
merge times are read from the ``FFA Chunk detail`` log line.

``--bsegs`` takes powers of two plus the tokens

* ``legacy``  the old "two cycles at f_min" rule (four for accel),
* ``cost``    the period-cap rule of ``select_bseg_brute_by_cost`` (largest
              power of two <= 2 cycles at f_max, time domain),
* ``default`` leave ``bseg_brute`` unset (library default of the build).

Usage::

    python bseg_brute_sensitivity.py --app build-dev/applications/loki \\
        --workdir /tmp/loki_bseg_sens --nsamps 2097152 --cases 6 \\
        --bsegs legacy cost 128 256 512 1024 2048 4096 8192 --json out.json

Add ``--acc 50`` for an accelerated search (acc +-acc m/s^2). Output: one
table per case and an aggregate Pareto table (mean/min S/N ratio and total
seconds per B), with B expressed as cycles at f_max.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from workloads import set_key

CHUNK_RE = re.compile(
    r"FFA Chunk detail: nbins=(?P<nbins>\d+) bseg_brute=(?P<bseg>\d+) "
    r"levels=(?P<levels>\d+) (?:fuse_levels=\d+ )?nfreqs0=(?P<nfreqs0>\d+) "
    r"ncoords_top=\d+ brutefold_s=(?P<brute>[\d.]+) "
    r"brute_table_s=(?P<table>[\d.]+) ffa_s=(?P<ffa>[\d.]+)"
)


def legacy_bseg(nsamps: int, f_min: float, tsamp: float, nparams: int) -> int:
    """Mirror of ``get_bseg_brute_legacy`` in lib/search/configs.cpp."""
    cycles = nsamps * tsamp * f_min
    if cycles <= 1.0:
        return min(nsamps // 2, 64)
    levels = math.floor(math.log2(cycles))
    shift = max(1, levels - (1 if nparams == 1 else 2))
    return min(max(nsamps >> shift, 2), nsamps // 2)


def cost_bseg(nsamps: int, f_max: float, tsamp: float) -> int:
    """Largest power of two <= 2 cycles at f_max (time-domain period cap)."""
    cap = 2.0 / (tsamp * f_max)
    return min(max(1 << int(math.floor(math.log2(cap))), 2), nsamps // 2)


def simulate(app: str, work: Path, period: float, acc: float, seed: int, args) -> Path:
    tim = work / f"p{seed}_{period:.6f}_{acc:.1f}_{args.nsamps}.tim"
    if not tim.exists():
        cmd = [app, "simulate", "--period", str(period), "--dt", str(args.tsamp)]
        cmd += ["--nsamps", str(args.nsamps), "--snr", str(args.snr)]
        cmd += ["--ducy", str(args.ducy), "--acc", str(acc), "--seed", str(seed)]
        cmd += ["-o", str(tim)]
        subprocess.run(cmd, check=True, capture_output=True)
    return tim


def run_search(
    app: str,
    template: str,
    tim: Path,
    out: Path,
    f0: float,
    f_true: float,
    acc: float,
    bseg: int | None,
    args,
):
    """Search the chunk centred on ``f0``; score the best S/N near ``f_true``."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = template
    keys = [
        ("timeseries", f'"{tim.resolve()}"'),
        ("f_min", f0 / math.sqrt(args.ratio) * args.f_shift),
        ("f_max", f0 * math.sqrt(args.ratio) * args.f_shift),
        ("acc_min", -args.acc),
        ("acc_max", args.acc),
        ("nbins", args.nbins),
        ("nbins_max", max(args.nbins, args.nbins_max)),
        ("eta", args.eta),
        ("use_fourier", "true" if args.fourier else "false"),
        ("nthreads", args.nthreads),
        ("max_process_memory_gb", args.memory_gb),
        ("outdir", '"./"'),
        ("prefix", '"s"'),
        ("preprocess", "true"),
        ("snr_min", 3.0),
    ]
    for k, v in keys:
        cfg = set_key(cfg, k, v)
    if bseg is not None:
        cfg = re.sub(
            r"^#?\s*bseg_brute = .*$", f"bseg_brute = {bseg}", cfg, flags=re.MULTILINE
        )
        if f"bseg_brute = {bseg}" not in cfg:
            cfg = cfg.replace("[performance]", f"[performance]\nbseg_brute = {bseg}")
    (out / "c.toml").write_text(cfg)
    t0 = time.perf_counter()
    r = subprocess.run(
        [app, "search", "ffa", "-c", "c.toml"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    (out / "log.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return None
    chunks = [m.groupdict() for m in CHUNK_RE.finditer(r.stdout + r.stderr)]
    with h5py.File(out / "s_ffa_results.h5", "r") as f:
        params = f["param_sets"][...]
        snr = f["snr"][...]
    near = np.abs(params[:, -1] - f_true) < 2e-3 * f_true
    if args.acc > 0:
        near &= np.abs(params[:, 0] - acc) < max(0.15 * args.acc, 5.0)
    best = float(snr[near].max()) if near.any() else 0.0
    return {
        "snr": best,
        "wall": wall,
        "brute": sum(float(c["brute"]) + float(c["table"]) for c in chunks),
        "ffa": sum(float(c["ffa"]) for c in chunks),
        "bseg_used": int(chunks[0]["bseg"]) if chunks else bseg,
        "levels": int(chunks[0]["levels"]) if chunks else -1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", required=True)
    ap.add_argument("--workdir", default="/tmp/loki_bseg_sens")
    ap.add_argument("--nsamps", type=int, default=2**21)
    ap.add_argument("--tsamp", type=float, default=6.4e-5)
    ap.add_argument(
        "--periods", type=float, nargs="+", default=[0.002, 0.004, 0.008, 0.016]
    )
    ap.add_argument("--cases", type=int, default=2, help="random f offsets per period")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr", type=float, default=14.0)
    ap.add_argument("--ducy", type=float, default=0.1)
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument(
        "--acc", type=float, default=0.0, help="acc +-ACC m/s^2 (0: freq only)"
    )
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--nbins-max", type=int, default=1024)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--fourier", action="store_true")
    ap.add_argument("--nthreads", type=int, default=4)
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--bsegs", nargs="+", default=["legacy", "cost"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    args.f_shift = 1.0

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    app = str(Path(args.app).resolve())
    tpl = work / "template.toml"
    subprocess.run(
        [app, "search", "ffa", "-g", str(tpl)], check=True, capture_output=True
    )
    template = tpl.read_text()
    nparams = 1 if args.acc == 0 else 2
    rng = random.Random(args.seed)  # noqa: S311

    rows = []
    for period in args.periods:
        for ic in range(args.cases):
            # True frequency uniformly inside the chunk (the chunk centre is f0 below).
            f_chunk = 1.0 / period
            f_true = f_chunk * rng.uniform(0.85, 1.15)
            acc = rng.uniform(-0.6, 0.6) * args.acc
            seed = 100 + int(period * 1e4) + ic
            tim = simulate(app, work, 1.0 / f_true, acc, seed, args)
            f_min = f_chunk / math.sqrt(args.ratio)
            f_max = f_chunk * math.sqrt(args.ratio)
            cands: dict[str, int | None] = {}
            for tok in args.bsegs:
                if tok == "legacy":
                    cands[tok] = legacy_bseg(args.nsamps, f_min, args.tsamp, nparams)
                elif tok == "cost":
                    cands[tok] = cost_bseg(args.nsamps, f_max, args.tsamp)
                elif tok == "default":
                    cands[tok] = None
                else:
                    cands[tok] = int(tok)
            res = {}
            for tok, b in cands.items():
                r = run_search(
                    app,
                    template,
                    tim,
                    work / f"run_{seed}_{tok}",
                    f_chunk,
                    f_true,
                    acc,
                    b,
                    args,
                )
                if r is not None:
                    res[tok] = r
            if not res:
                continue
            ref = max(v["snr"] for v in res.values())
            print(
                f"\nP={period * 1e3:.2f} ms f_true={f_true:.2f} Hz chunk=[{f_min:.1f},{f_max:.1f}] acc={acc:.1f} best={ref:.2f}"
            )
            print("  tok       B  cyc@fmax lvls   snr  ratio  brute_s  merge_s")
            for tok, v in res.items():
                b = v["bseg_used"]
                cyc = b * args.tsamp * f_max
                ratio = v["snr"] / ref if ref > 0 else float("nan")
                print(
                    f"  {tok:8s} {b:6d} {cyc:7.2f} {v['levels']:4d} {v['snr']:6.2f} {ratio:6.3f} {v['brute']:8.3f} {v['ffa']:8.3f}"
                )
                rows.append(
                    {
                        "period": period,
                        "case": ic,
                        "tok": tok,
                        "bseg": b,
                        "cyc_fmax": cyc,
                        "ratio": ratio,
                        **v,
                    }
                )

    # Aggregate Pareto: per token.
    print("\n== Aggregate over cases (ratio = S/N / best S/N of the case) ==")
    print(
        "  tok       mean_cyc  mean_ratio  min_ratio  total_brute_s  total_merge_s  total_s"
    )
    toks = list(dict.fromkeys(r["tok"] for r in rows))
    for tok in toks:
        sel = [r for r in rows if r["tok"] == tok]
        ratios = np.array([r["ratio"] for r in sel])
        brute = sum(r["brute"] for r in sel)
        ffa = sum(r["ffa"] for r in sel)
        cyc = np.mean([r["cyc_fmax"] for r in sel])
        print(
            f"  {tok:8s} {cyc:8.2f} {ratios.mean():10.4f} {ratios.min():10.4f} {brute:13.2f} {ffa:13.2f} {brute + ffa:8.2f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
