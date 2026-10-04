#!/usr/bin/env python3
"""Sweep ``performance.bseg_brute`` for the FFA frequency sweep.

Runs ``loki search ffa`` once per bseg_brute value (plus the library default)
and parses the per-chunk ``FFA Chunk detail`` log lines to report, per chunk
(i.e. per nbins/frequency region): brute-fold time, brute-table build time,
merge ("ffa") time and the number of FFA levels.

Usage:
    python bench/bseg_brute_sweep.py --app build2/applications/loki \
        --timeseries /tmp/loki_p0/pulse21.tim --bsegs default 256 1024 4096 \
        --workdir /tmp/loki_p0 --json out.json

The default search window mirrors ``bench/ffa_freq_b.cpp`` (1-488 Hz,
nbins=32 .. 1024, octave_scale=1.5), so the same regions are exercised.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CHUNK_RE = re.compile(
    r"FFA Chunk detail: nbins=(?P<nbins>\d+) bseg_brute=(?P<bseg>\d+) "
    r"levels=(?P<levels>\d+) (?:fuse_levels=(?P<fuse>\d+) )?nfreqs0=(?P<nfreqs0>\d+) "
    r"ncoords_top=(?P<ncoords_top>\d+) brutefold_s=(?P<brute>[\d.]+) "
    r"brute_table_s=(?P<table>[\d.]+) ffa_s=(?P<ffa>[\d.]+)"
)
RANGE_RE = re.compile(r"Processing chunk f0 \(Hz\): \[\s*([\d.]+),\s*([\d.]+)\]")


def write_config(path: Path, args: argparse.Namespace, bseg: str, outdir: Path) -> None:
    perf = [
        f"nthreads = {args.nthreads}",
        f"max_process_memory_gb = {args.memory_gb}",
        f"octave_scale = {args.octave_scale}",
        f"nbins_max = {args.nbins_max}",
        f"nbins_min_lossy_bf = {args.nbins_min_lossy_bf}",
    ]
    if bseg != "default":
        perf.append(f"bseg_brute = {int(bseg)}")
    text = (
        "[input]\n"
        f'timeseries = "{args.timeseries}"\n'
        "preprocess = false\n\n"
        "[search]\n"
        f"f_min = {args.f_min}\n"
        f"f_max = {args.f_max}\n"
        "acc_min = 0.0\nacc_max = 0.0\n"
        f"nbins = {args.nbins}\n"
        f"eta = {args.eta}\n"
        "ducy_max = 0.5\nwtsp = 1.2\nsnr_min = 8.0\n"
        f"use_fourier = {'true' if args.fourier else 'false'}\n\n"
        "[performance]\n" + "\n".join(perf) + "\n\n"
        "[output]\n"
        f'outdir = "{outdir}"\n'
        'prefix = "sweep"\n'
    )
    path.write_text(text)


def run_one(args: argparse.Namespace, bseg: str) -> dict:
    case = Path(args.workdir) / f"bseg_{bseg}"
    out = case / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg = case / "config.toml"
    write_config(cfg, args, bseg, out)
    log = case / "search.log"
    t0 = time.perf_counter()
    with log.open("w") as fh:
        rc = subprocess.run(
            [args.app, "search", "ffa", "--config", str(cfg)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    wall = time.perf_counter() - t0
    chunks = []
    cur_range = None
    for line in log.read_text().splitlines():
        m = RANGE_RE.search(line)
        if m:
            cur_range = (float(m.group(1)), float(m.group(2)))
        m = CHUNK_RE.search(line)
        if m:
            d = {
                k: float(v) if "." in v else int(v)
                for k, v in m.groupdict().items()
                if v is not None
            }
            d["f_range"] = cur_range
            chunks.append(d)
    return {"bseg": bseg, "rc": rc, "wall_s": wall, "chunks": chunks}


def summarize(res: dict) -> None:
    tot_b = sum(c["brute"] for c in res["chunks"])
    tot_t = sum(c["table"] for c in res["chunks"])
    tot_f = sum(c["ffa"] for c in res["chunks"])
    print(
        f"\n== bseg_brute={res['bseg']}  rc={res['rc']}  wall={res['wall_s']:.2f}s  "
        f"brute={tot_b:.2f}s (table {tot_t:.2f}s)  merge={tot_f:.2f}s  "
        f"brute share={100 * tot_b / max(tot_b + tot_f, 1e-9):.1f}%"
    )
    print("  f_range(Hz)            nbins    B   levels  nfreqs0    brute    table    merge  share")
    for c in res["chunks"]:
        fr = c["f_range"] or (0, 0)
        share = 100 * c["brute"] / max(c["brute"] + c["ffa"], 1e-9)
        print(
            f"  [{fr[0]:8.3f},{fr[1]:8.3f}] {c['nbins']:6d} {c['bseg']:6d} {c['levels']:4d}"
            f" {c['nfreqs0']:8d} {c['brute']:8.3f} {c['table']:8.3f} {c['ffa']:8.3f} {share:5.1f}%"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", required=True)
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--workdir", default="/tmp/loki_bseg_sweep")
    ap.add_argument("--bsegs", nargs="+", default=["default"])
    ap.add_argument("--f-min", type=float, default=1.0)
    ap.add_argument("--f-max", type=float, default=488.0)
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--memory-gb", type=float, default=16.0)
    ap.add_argument("--octave-scale", type=float, default=1.5)
    ap.add_argument("--nbins-max", type=int, default=1024)
    ap.add_argument("--nbins-min-lossy-bf", type=int, default=32)
    ap.add_argument("--fourier", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    for bseg in args.bsegs:
        res = run_one(args, bseg)
        summarize(res)
        results.append(res)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
