#!/usr/bin/env python3
"""Compare loki's frequency-only FFA with riptide on a matched 0.1-1 s search.

Stages reported for each side: read, preprocess (timed separately from the
search), FFA (riptide: downsample + transform; loki: brute fold + merge),
scoring, and post/IO. Trial counts are printed so a 2x gap from loki's
ts_e + ts_v pair can be judged per trial.

Riptide is invoked with ``--python`` (a venv whose ``riptide`` import works).
The C++ stage bench (``--stage-bench``) splits riptide's periodogram the same
way, without the Python wrapper.

Example:
    python bench/riptide_compare.py \\
        --loki build2/applications/loki \\
        --python /tmp/riptide-venv/bin/python \\
        --stage-bench /tmp/riptide_stage_build/riptide_stage_bench \\
        --workdir /tmp/loki_riptide_cmp
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import textwrap
import time
from pathlib import Path

CHUNK_RE = re.compile(
    r"FFA Chunk detail: nbins=(?P<nbins>\d+) bseg_brute=(?P<bseg>\d+) "
    r"levels=(?P<levels>\d+) (?:fuse_levels=(?P<fuse>\d+) )?"
    r"nfreqs0=(?P<nfreqs0>\d+) ncoords_top=(?P<ncoords_top>\d+) "
    r"brutefold_s=(?P<brute>[\d.]+) brute_table_s=(?P<table>[\d.]+) "
    r"ffa_s=(?P<ffa>[\d.]+)"
)
SWEEP_RE = re.compile(r"FFA Freq Sweep: timer: Total:\s*([\d.]+)s")
CANDS_RE = re.compile(r"Total candidates detected :\s*(\d+)")
TOP_SNR_RE = re.compile(r"Top candidate SNR\s*:\s*([\d.]+)")
TOP_FREQ_RE = re.compile(r"(?:f0|freq)=([\d.]+)")

# Injected pulsars for the sensitivity pass (period seconds, duty cycle).
INJECTIONS = (
    (0.15, 0.02),
    (0.15, 0.05),
    (0.40, 0.02),
    (0.40, 0.05),
    (0.90, 0.02),
    (0.90, 0.05),
)

RIPTIDE_WORKER = r"""
import json, sys, time
import numpy as np
from riptide import TimeSeries, Periodogram, find_peaks, libcpp
from riptide.ffautils import generate_width_trials

cfg = json.loads(sys.argv[1])

def load():
    t0 = time.perf_counter()
    try:
        ts = TimeSeries.from_sigproc(cfg["path"])
        return ts, time.perf_counter() - t0, "sigproc"
    except Exception as exc:
        sigproc_error = str(exc)
    if not cfg.get("dat"):
        raise SystemExit("sigproc load failed and no .dat fallback: " + sigproc_error)
    t0 = time.perf_counter()
    ts = TimeSeries.from_presto_inf(cfg["dat"])
    return ts, time.perf_counter() - t0, "presto"

ts, read_s, loader = load()
# Keep the search on the same samples loki will fold (power-of-two prefix).
nsamps = int(cfg["nsamps"])
if ts.data.size < nsamps:
    raise SystemExit(f"file has {ts.data.size} samples, need {nsamps}")
data = np.ascontiguousarray(ts.data[:nsamps], dtype=np.float32)
ts = TimeSeries(data, ts.tsamp, metadata=ts.metadata)

out = {"loader": loader, "read_s": read_s, "nsamps": int(data.size), "tsamp": float(ts.tsamp)}
if cfg.get("load_only"):
    print(json.dumps(out))
    raise SystemExit(0)

if cfg.get("time_preprocess"):
    t0 = time.perf_counter()
    reddened = ts.deredden(cfg["rmed_width"], minpts=int(cfg["rmed_minpts"]))
    out["deredden_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    reddened.normalise()
    out["normalise_s"] = time.perf_counter() - t0

widths = generate_width_trials(int(cfg["bins_min"]), ducy_max=float(cfg["ducy_max"]), wtsp=float(cfg["wtsp"]))
t0 = time.perf_counter()
periods, foldbins, snrs = libcpp.periodogram(
    data, float(ts.tsamp), widths,
    float(cfg["period_min"]), float(cfg["period_max"]),
    int(cfg["bins_min"]), int(cfg["bins_max"]),
)
out["pgram_s"] = time.perf_counter() - t0
out["ntrials"] = int(periods.size)
out["nwidths"] = int(widths.size)
out["mean_bins"] = float(np.mean(foldbins))

if cfg.get("time_peaks"):
    pgram = Periodogram(widths, periods, foldbins, snrs, metadata=ts.metadata)
    t0 = time.perf_counter()
    peaks, _ = find_peaks(pgram, smin=float(cfg["smin"]))
    out["peaks_s"] = time.perf_counter() - t0
    out["npeaks"] = len(peaks)
    if peaks:
        best = max(peaks, key=lambda p: p.snr)
        out["top_snr"] = float(best.snr)
        out["top_period"] = float(best.period)
        out["top_freq"] = float(best.freq)

if cfg.get("period"):
    freq = 1.0 / float(cfg["period"])
    tobs = float(data.size) * float(ts.tsamp)
    half = 2.0 / tobs
    freqs = 1.0 / periods
    mask = np.abs(freqs - freq) <= half
    if not np.any(mask):
        # Fall back to the nearest trial if the window is empty.
        idx = int(np.argmin(np.abs(freqs - freq)))
        out["best_snr"] = float(snrs[idx].max())
        out["best_period"] = float(periods[idx])
        out["best_freq"] = float(freqs[idx])
    else:
        sub = snrs[mask]
        flat = int(np.argmax(sub))
        row, _ = np.unravel_index(flat, sub.shape)
        chosen = np.flatnonzero(mask)[row]
        out["best_snr"] = float(snrs[chosen].max())
        out["best_period"] = float(periods[chosen])
        out["best_freq"] = float(freqs[chosen])

print(json.dumps(out))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loki", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="venv python with riptide")
    parser.add_argument("--stage-bench", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--nsamps", type=int, nargs="+", default=[1 << 21, 1 << 23])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tsamp", type=float, default=6.4e-5)
    parser.add_argument("--f-min", type=float, default=1.0)
    parser.add_argument("--f-max", type=float, default=10.0)
    parser.add_argument("--nbins", type=int, default=256)
    parser.add_argument("--bins-min", type=int, default=256)
    parser.add_argument("--bins-max", type=int, default=280)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ducy-max", type=float, default=0.3)
    parser.add_argument("--wtsp", type=float, default=1.5)
    parser.add_argument("--snr-min", type=float, default=6.0)
    parser.add_argument("--snr", type=float, default=10.0, help="injected folded S/N")
    parser.add_argument("--rmed-width", type=float, default=4.0)
    parser.add_argument("--rmed-minpts", type=int, default=101)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-parallel", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=8)
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def run(cmd: list[str], log: Path | None = None, env: dict | None = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    if log is None:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
        text = proc.stdout + proc.stderr
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            proc = subprocess.run(cmd, check=False, text=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
        text = log.read_text(errors="replace")
    return proc.returncode, text, time.perf_counter() - t0


def write_loki_config(
    path: Path,
    timeseries: Path,
    outdir: Path,
    prefix: str,
    args: argparse.Namespace,
    threads: int,
    preprocess: bool,
) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            [input]
            timeseries = "{timeseries}"
            preprocess = {"true" if preprocess else "false"}
            filter_window = {args.rmed_width}

            [search]
            f_min = {args.f_min}
            f_max = {args.f_max}
            acc_min = 0.0
            acc_max = 0.0
            nbins = {args.nbins}
            eta = {args.eta}
            ducy_max = {args.ducy_max}
            wtsp = {args.wtsp}
            snr_min = {args.snr_min}
            use_fourier = false

            [performance]
            nthreads = {threads}
            max_process_memory_gb = 16.0
            octave_scale = 1.5
            nbins_max = {args.nbins}
            nbins_min_lossy_bf = 32

            [output]
            outdir = "{outdir}"
            prefix = "{prefix}"
            """
        )
    )


def parse_loki(text: str, wall_s: float) -> dict:
    chunks = []
    for match in CHUNK_RE.finditer(text):
        chunks.append({key: float(value) for key, value in match.groupdict().items() if value is not None})
    brute = sum(c["brute"] for c in chunks)
    table = sum(c["table"] for c in chunks)
    merge = sum(c["ffa"] for c in chunks)
    trials = sum(int(c["ncoords_top"]) for c in chunks)
    sweep = SWEEP_RE.findall(text)
    sweep_s = float(sweep[-1]) if sweep else brute + merge
    # The printed sweep total is rounded to 0.1 s, so score = sweep - brute - merge
    # collapses on short runs. Scale the percentage split by the measured fold time.
    score_io = max(0.0, sweep_s - brute - merge)
    pct_match = re.search(r"FFA Freq Sweep: timer: Total:.*\(([^)]*)\)", text)
    if pct_match and brute + merge > 0.0:
        pct = {
            name: int(value)
            for name, value in re.findall(r"(\w+):\s*(\d+)%", pct_match.group(1))
        }
        fold_pct = pct.get("brutefold", 0) + pct.get("ffa", 0)
        if fold_pct > 0:
            score_io = (brute + merge) * (pct.get("score", 0) + pct.get("io", 0)) / fold_pct
    cands = CANDS_RE.findall(text)
    snr = TOP_SNR_RE.findall(text)
    freq = TOP_FREQ_RE.findall(text)
    return {
        "wall_s": wall_s,
        "sweep_s": sweep_s,
        "read_startup_s": max(0.0, wall_s - sweep_s),
        "brutefold_s": brute,
        "brute_table_s": table,
        "merge_s": merge,
        "score_io_s": score_io,
        "ntrials": trials,
        "nchunks": len(chunks),
        "bseg": chunks[0]["bseg"] if chunks else None,
        "fuse_levels": chunks[0].get("fuse") if chunks else None,
        "nbins": chunks[0]["nbins"] if chunks else None,
        "ncands": int(cands[-1]) if cands else 0,
        "top_snr": float(snr[-1]) if snr else None,
        "top_freq": float(freq[-1]) if freq else None,
        "chunks": chunks,
    }


def normalise_tim(python: Path, src: Path, dest: Path) -> None:
    """Rewrite a SIGPROC .tim so the payload has zero mean and unit variance.

    loki simulate scales the noise so the folded profile hits the requested
    S/N while storing the per-sample variance only in memory. The .tim carries
    ts_e alone, and loki reloads it with ts_v = 1. Unit-variance samples make
    that assumption match riptide's already-normalised search.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    script = r"""
import sys
import numpy as np
from riptide.reading.sigproc import SigprocHeader
src, dest = sys.argv[1], sys.argv[2]
header = SigprocHeader(src)
with open(src, "rb") as handle:
    prefix = handle.read(header.bytesize)
    data = np.fromfile(handle, dtype=np.float32)
scale = float(data.std())
if scale == 0.0:
    raise SystemExit("timeseries has zero variance")
data = ((data - float(data.mean())) / scale).astype(np.float32)
with open(dest, "wb") as handle:
    handle.write(prefix)
    data.tofile(handle)
"""
    code, text, _ = run([str(python), "-c", script, str(src), str(dest)])
    if code != 0:
        raise RuntimeError(f"normalise failed ({code}): {text[-500:]}")


def simulate(loki: Path, dest: Path, args: argparse.Namespace, nsamps: int, period: float, ducy: float, seed: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cmd = [
        str(loki),
        "simulate",
        "--period",
        str(period),
        "--dt",
        str(args.tsamp),
        "--nsamps",
        str(nsamps),
        "--snr",
        str(args.snr),
        "--ducy",
        str(ducy),
        "--shape",
        "gaussian",
        "--phi0",
        "0.5",
        "--seed",
        str(seed),
        "-o",
        str(dest),
    ]
    code, text, _ = run(cmd, dest.with_suffix(".simulate.log"))
    if code != 0 or not dest.exists():
        raise RuntimeError(f"loki simulate failed ({code}): {text[-500:]}")
    return dest


def dat_fallback(loki: Path, tim_path: Path) -> Path | None:
    """Write a PRESTO .dat/.inf next to the .tim if riptide cannot read SIGPROC."""
    dat = tim_path.with_suffix(".dat")
    if dat.exists() and dat.with_suffix(".inf").exists():
        return dat.with_suffix(".inf")
    # Re-simulate is unnecessary: convert by asking loki only if we add a tool.
    # The worker tries SIGPROC first. The caller passes this path when present.
    del loki
    return dat.with_suffix(".inf") if dat.exists() else None


def riptide_call(python: Path, cfg: dict, log: Path) -> dict:
    code, text, wall = run([str(python), "-c", RIPTIDE_WORKER, json.dumps(cfg)], log)
    if code != 0:
        raise RuntimeError(f"riptide worker failed ({code}): {text[-800:]}")
    # The worker prints one JSON line; logging libraries may add others.
    line = [row for row in text.splitlines() if row.startswith("{")][-1]
    payload = json.loads(line)
    payload["wall_s"] = wall
    return payload


def loki_search(
    loki: Path,
    cfg_path: Path,
    log: Path,
    args: argparse.Namespace,
    timeseries: Path,
    outdir: Path,
    prefix: str,
    threads: int,
    preprocess: bool,
) -> dict:
    write_loki_config(cfg_path, timeseries, outdir, prefix, args, threads, preprocess)
    code, text, wall = run([str(loki), "search", "ffa", "--config", str(cfg_path)], log)
    parsed = parse_loki(text, wall)
    parsed["rc"] = code
    if code != 0:
        parsed["error"] = text[-500:]
    return parsed


def stage_bench(binary: Path, args: argparse.Namespace, nsamps: int, log: Path) -> dict:
    cmd = [
        str(binary),
        "--nsamps",
        str(nsamps),
        "--tsamp",
        str(args.tsamp),
        "--period-min",
        str(1.0 / args.f_max),
        "--period-max",
        str(1.0 / args.f_min),
        "--bins-min",
        str(args.bins_min),
        "--bins-max",
        str(args.bins_max),
        "--ducy-max",
        str(args.ducy_max),
        "--wtsp",
        str(args.wtsp),
        "--reps",
        str(args.reps),
        "--warmup",
        "1",
        "--seed",
        "1",
    ]
    code, text, wall = run(cmd, log)
    if code != 0:
        raise RuntimeError(f"riptide_stage_bench failed ({code}): {text[-800:]}")
    line = [row for row in text.splitlines() if row.startswith("{")][-1]
    payload = json.loads(line)
    payload["wall_s"] = wall
    return payload


def parallel_pgram(python: Path, cfg: dict, workers: int, log: Path) -> dict:
    """Riptide's own parallelism is one process per time series (per DM)."""
    cfg = dict(cfg)
    cfg["time_preprocess"] = False
    cfg["time_peaks"] = False
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [str(python), "-c", RIPTIDE_WORKER, json.dumps(cfg)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(workers)
    ]
    outputs = []
    rc = 0
    for proc in procs:
        text, _ = proc.communicate()
        outputs.append(text)
        rc = rc or proc.returncode
    wall = time.perf_counter() - t0
    log.write_text("\n".join(outputs))
    if rc != 0:
        raise RuntimeError(f"parallel riptide failed: {outputs[-1][-500:]}")
    return {"workers": workers, "batch_wall_s": wall, "per_series_s": wall / workers}


def summarise_reps(reps: list[dict], keys: tuple[str, ...]) -> dict:
    out = {"reps": len(reps)}
    for key in keys:
        vals = [float(rep[key]) for rep in reps if key in rep and rep[key] is not None]
        if vals:
            out[key] = median(vals)
    return out


def fmt(value, digits=2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, payload: dict) -> None:
    lines = ["# Loki vs riptide (1-10 Hz)", ""]
    lines.append(
        "The primary comparison is single-threaded loki against single-threaded "
        "riptide. Loki folds `ts_e` and `ts_v`. Riptide folds one array. "
        "A factor of about 2 is the budget; lower is better for loki."
    )
    lines.append(
        "`riptide-python` fold time is `libcpp.periodogram` (downsample + transform + S/N). "
        "`riptide-cpp` splits those stages. `riptide-python xN` is N processes at once; "
        "wall s there is batch time divided by N (throughput, not latency)."
    )
    lines.append("")
    for case in payload["cases"]:
        nsamps = case["nsamps"]
        lines.append(f"## nsamps = {nsamps} (tobs = {case['tobs_s']:.2f} s)")
        lines.append("")
        lines.append(
            "| side | threads | wall s | fold s | score s | trials | ns / (trial·bin) | top S/N |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in case["rows"]:
            fold = row.get("ffa_s")
            if row.get("brutefold_s") is not None:
                fold_txt = f"{fmt(fold)} (brute {fmt(row.get('brutefold_s'))} + merge {fmt(row.get('merge_s'))})"
            elif row.get("transform_s") is not None:
                fold_txt = (
                    f"{fmt(fold)} (downsample {fmt(row.get('downsample_s'), 3)} "
                    f"+ transform {fmt(row.get('transform_s'))})"
                )
            else:
                fold_txt = fmt(fold)
            lines.append(
                "| {side} | {threads} | {wall} | {fold} | {score} | {trials} | {ns} | {snr} |".format(
                    side=row["side"],
                    threads=row.get("threads", "-"),
                    wall=fmt(row.get("wall_s")),
                    fold=fold_txt,
                    score=fmt(row.get("score_s")),
                    trials=row.get("ntrials", "-"),
                    ns=fmt(row.get("ns_per_trial_bin"), 3),
                    snr=fmt(row.get("top_snr"), 2),
                )
            )
        ratios = case.get("stage_ratios")
        if ratios:
            lines.append(
                "Single-thread stage ratios (loki / riptide-cpp): "
                f"brute/downsample {fmt(ratios.get('brute_vs_downsample'))}, "
                f"merge/transform {fmt(ratios.get('merge_vs_transform'))}, "
                f"score/snr {fmt(ratios.get('score_vs_snr'))}, "
                f"fold {fmt(ratios.get('fold_vs_fold'))}, "
                f"fold+score {fmt(ratios.get('total'))}."
            )
            lines.append("")
        if case.get("preprocess"):
            pre = case["preprocess"]
            lines.append(
                f"Preprocess (not in the FFA rows): loki read+detrend+normalise "
                f"{fmt(pre.get('loki_wall_s'))} s wall at {pre.get('threads')} threads; "
                f"riptide deredden {fmt(pre.get('deredden_s'))} s, normalise {fmt(pre.get('normalise_s'))} s."
            )
            lines.append("")
        if case.get("sensitivity"):
            lines.append("| P (s) | ducy | loki S/N | loki f (Hz) | riptide S/N | riptide f (Hz) |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for item in case["sensitivity"]:
                lines.append(
                    f"| {item['period']:.2f} | {item['ducy']:.2f} | {fmt(item.get('loki_snr'))} | "
                    f"{fmt(item.get('loki_freq'), 4)} | {fmt(item.get('riptide_snr'))} | "
                    f"{fmt(item.get('riptide_freq'), 4)} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def single_thread_stage_ratios(rows: list[dict]) -> dict | None:
    """Loki at 1 thread over riptide's C++ stage split. This is the primary metric."""
    loki = next(
        (row for row in rows if row.get("side") == "loki" and row.get("threads") == 1),
        None,
    )
    riptide = next((row for row in rows if row.get("side") == "riptide-cpp"), None)
    if loki is None or riptide is None:
        return None
    loki_total = (loki.get("ffa_s") or 0.0) + (loki.get("score_s") or 0.0)
    riptide_total = (riptide.get("ffa_s") or 0.0) + (riptide.get("score_s") or 0.0)
    return {
        "brute_vs_downsample": _ratio(loki.get("brutefold_s"), riptide.get("downsample_s")),
        "merge_vs_transform": _ratio(loki.get("merge_s"), riptide.get("transform_s")),
        "score_vs_snr": _ratio(loki.get("score_s"), riptide.get("score_s")),
        "fold_vs_fold": _ratio(loki.get("ffa_s"), riptide.get("ffa_s")),
        "total": _ratio(loki_total, riptide_total),
    }


def ns_per_trial_bin(ffa_s: float, ntrials: int, bins: float) -> float | None:
    if ntrials <= 0 or bins <= 0:
        return None
    return ffa_s / (ntrials * bins) * 1e9


def main() -> None:
    args = parse_args()
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    period_min = 1.0 / args.f_max
    period_max = 1.0 / args.f_min
    cases = []

    for nsamps in args.nsamps:
        case_dir = work / f"n{nsamps}"
        case_dir.mkdir(parents=True, exist_ok=True)
        # Timing series: one slow pulsar inside the band. Runtime does not
        # depend on it; it gives the top-candidate columns a real peak.
        raw_tim = simulate(
            args.loki,
            case_dir / "timing_raw.tim",
            args,
            nsamps,
            period=0.4,
            ducy=0.05,
            seed=1,
        )
        timing_tim = case_dir / "timing.tim"
        normalise_tim(args.python, raw_tim, timing_tim)
        base_cfg = {
            "path": str(timing_tim),
            "nsamps": nsamps,
            "period_min": period_min,
            "period_max": period_max,
            "bins_min": args.bins_min,
            "bins_max": args.bins_max,
            "ducy_max": args.ducy_max,
            "wtsp": args.wtsp,
            "smin": args.snr_min,
            "rmed_width": args.rmed_width,
            "rmed_minpts": args.rmed_minpts,
            "period": 0.4,
            "time_peaks": True,
            "time_preprocess": False,
        }
        # Probe the loader once. A SIGPROC rejection is reported; the .dat
        # fallback is used when the simulate step also wrote one.
        probe = riptide_call(
            args.python, {**base_cfg, "load_only": True}, case_dir / "probe.log"
        )
        if probe["loader"] != "sigproc":
            inf = dat_fallback(args.loki, timing_tim)
            if inf is not None:
                base_cfg["dat"] = str(inf)

        rows = []
        loki_by_threads = {}
        for threads in args.threads:
            reps = []
            for rep in range(args.reps):
                reps.append(
                    loki_search(
                        args.loki,
                        case_dir / f"loki_t{threads}_r{rep}.toml",
                        case_dir / f"loki_t{threads}_r{rep}.log",
                        args,
                        timing_tim,
                        case_dir / f"out_t{threads}_r{rep}",
                        f"t{threads}r{rep}",
                        threads,
                        preprocess=False,
                    )
                )
            summary = summarise_reps(
                reps,
                ("wall_s", "sweep_s", "read_startup_s", "brutefold_s", "merge_s", "score_io_s", "top_snr"),
            )
            summary.update(
                {
                    "side": "loki",
                    "threads": threads,
                    "ntrials": reps[-1]["ntrials"],
                    "bseg": reps[-1]["bseg"],
                    "fuse_levels": reps[-1]["fuse_levels"],
                    "top_freq": reps[-1]["top_freq"],
                    "ffa_s": summary.get("brutefold_s", 0.0) + summary.get("merge_s", 0.0),
                    "score_s": summary.get("score_io_s"),
                }
            )
            summary["ns_per_trial_bin"] = ns_per_trial_bin(
                summary["ffa_s"], summary["ntrials"], float(args.nbins)
            )
            summary["wall_s"] = summary.get("wall_s")
            rows.append(summary)
            loki_by_threads[threads] = summary

        rip_reps = []
        for rep in range(args.reps):
            rip_reps.append(
                riptide_call(args.python, base_cfg, case_dir / f"riptide_r{rep}.log")
            )
        rip = summarise_reps(
            rip_reps, ("wall_s", "read_s", "pgram_s", "peaks_s", "top_snr", "best_snr")
        )
        rip.update(
            {
                "side": "riptide-python",
                "threads": 1,
                "ntrials": rip_reps[-1]["ntrials"],
                "top_freq": rip_reps[-1].get("best_freq", rip_reps[-1].get("top_freq")),
                "top_snr": rip.get("best_snr", rip.get("top_snr")),
                "ffa_s": rip.get("pgram_s"),
                "score_s": None,
                "loader": rip_reps[-1]["loader"],
                "mean_bins": rip_reps[-1]["mean_bins"],
            }
        )
        rip["ns_per_trial_bin"] = ns_per_trial_bin(
            rip["ffa_s"], rip["ntrials"], rip["mean_bins"]
        )
        rows.append(rip)

        stages = stage_bench(args.stage_bench, args, nsamps, case_dir / "stage_bench.log")
        med = stages["median"]
        stage_row = {
            "side": "riptide-cpp",
            "threads": 1,
            "wall_s": med["total_s"],
            "ffa_s": med["downsample_s"] + med["transform_s"],
            "score_s": med["snr_s"],
            "downsample_s": med["downsample_s"],
            "transform_s": med["transform_s"],
            "ntrials": stages["ntrials"],
            "top_snr": None,
            "ns_per_trial_bin": ns_per_trial_bin(
                med["downsample_s"] + med["transform_s"],
                stages["ntrials"],
                (args.bins_min + args.bins_max) / 2.0,
            ),
        }
        rows.append(stage_row)

        if not args.skip_parallel:
            parallel = parallel_pgram(
                args.python, base_cfg, args.parallel_workers, case_dir / "riptide_parallel.log"
            )
            rows.append(
                {
                    "side": f"riptide-python x{parallel['workers']}",
                    "threads": parallel["workers"],
                    "wall_s": parallel["per_series_s"],
                    "batch_wall_s": parallel["batch_wall_s"],
                    "ffa_s": parallel["per_series_s"],
                    "score_s": None,
                    "ntrials": rip["ntrials"],
                    "top_snr": None,
                    "ns_per_trial_bin": ns_per_trial_bin(
                        parallel["per_series_s"], rip["ntrials"], rip["mean_bins"]
                    ),
                }
            )

        preprocess = None
        if not args.skip_preprocess:
            threads = max(args.threads)
            loki_pre = loki_search(
                args.loki,
                case_dir / "loki_preprocess.toml",
                case_dir / "loki_preprocess.log",
                args,
                timing_tim,
                case_dir / "out_preprocess",
                "preprocess",
                threads,
                preprocess=True,
            )
            rip_pre = riptide_call(
                args.python,
                {**base_cfg, "time_preprocess": True, "time_peaks": False},
                case_dir / "riptide_preprocess.log",
            )
            preprocess = {
                "threads": threads,
                "loki_wall_s": loki_pre["wall_s"],
                "loki_sweep_s": loki_pre["sweep_s"],
                "loki_read_startup_s": loki_pre["read_startup_s"],
                "deredden_s": rip_pre.get("deredden_s"),
                "normalise_s": rip_pre.get("normalise_s"),
            }

        sensitivity = []
        if not args.skip_sensitivity:
            for index, (period, ducy) in enumerate(INJECTIONS):
                raw = simulate(
                    args.loki,
                    case_dir / f"inj_p{period:.2f}_d{ducy:.2f}_raw.tim",
                    args,
                    nsamps,
                    period=period,
                    ducy=ducy,
                    seed=10 + index,
                )
                tim = case_dir / f"inj_p{period:.2f}_d{ducy:.2f}.tim"
                normalise_tim(args.python, raw, tim)
                loki_one = loki_search(
                    args.loki,
                    case_dir / f"inj_{index}.toml",
                    case_dir / f"inj_{index}.log",
                    args,
                    tim,
                    case_dir / f"out_inj_{index}",
                    f"inj{index}",
                    threads=args.threads[0],
                    preprocess=False,
                )
                rip_one = riptide_call(
                    args.python,
                    {**base_cfg, "path": str(tim), "period": period, "time_peaks": False},
                    case_dir / f"inj_{index}_riptide.log",
                )
                sensitivity.append(
                    {
                        "period": period,
                        "ducy": ducy,
                        "loki_snr": loki_one.get("top_snr"),
                        "loki_freq": loki_one.get("top_freq"),
                        "riptide_snr": rip_one.get("best_snr"),
                        "riptide_freq": rip_one.get("best_freq"),
                    }
                )

        stage_ratios = single_thread_stage_ratios(rows)
        cases.append(
            {
                "nsamps": nsamps,
                "tobs_s": nsamps * args.tsamp,
                "stage_ratios": stage_ratios,
                "rows": rows,
                "loki": loki_by_threads,
                "preprocess": preprocess,
                "sensitivity": sensitivity,
                "stage_bench": stages,
            }
        )
        print(f"finished nsamps={nsamps}", flush=True)

    payload = {
        "f_min": args.f_min,
        "f_max": args.f_max,
        "nbins": args.nbins,
        "bins_min": args.bins_min,
        "bins_max": args.bins_max,
        "eta": args.eta,
        "cases": cases,
    }
    (work / "report.json").write_text(json.dumps(payload, indent=2))
    write_report(work / "report.md", payload)
    print(f"wrote {work / 'report.md'}")


if __name__ == "__main__":
    main()
