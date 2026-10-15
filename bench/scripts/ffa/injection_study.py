"""Random injection study: recovered S/N of two loki builds on the same pulsars.

Usage::

    python injection_study.py BASE_LOKI NEW_LOKI [--nsamps 33554432]
        [--cases 24] [--seed 0] [--nbins50] [--backend cuda] [--memgb 10]

Simulates ``--cases`` accelerated pulsars (period 6-80 ms, accel +-80 m/s^2,
S/N 12) with BASE_LOKI, searches each with both builds (time and Fourier
folds, +-25% around the true frequency, accel +-100 m/s^2) and prints the best
S/N near the truth per case, then the mean / min / max difference NEW - BASE.
Any change that moves chunk widths or accel trials needs this study at
2^25 samples (the default), not 2^23. Files go under
``$LOKI_BENCH_WORK/inj/`` (default: the current directory).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
from pathlib import Path

import h5py
import numpy as np
from workloads import set_key

WORK = Path(os.environ.get("LOKI_BENCH_WORK", ".")) / "inj"


def simulate(loki: str, period: float, acc: float, seed: int, nsamps: int) -> Path:
    tim = WORK / f"p{seed}_{period:.6f}_{acc:.3f}_{nsamps}.tim"
    if not tim.exists():
        cmd = [loki, "simulate", "--period", str(period), "--dt", "0.000064"]
        cmd += ["--nsamps", str(nsamps), "--snr", "12", "--ducy", "0.05"]
        cmd += ["--acc", str(acc), "--seed", str(seed), "-o", str(tim)]
        subprocess.run(cmd, check=True, capture_output=True)
    return tim


def best_snr(loki: str, cfg: str, out: Path, f0: float, acc: float) -> float:
    """Best S/N within 0.2% of f0 and 15 m/s^2 of the true accel; nan on error."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "c.toml").write_text(cfg)
    r = subprocess.run(
        [loki, "search", "ffa", "-c", "c.toml"],
        cwd=out,
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        return math.nan
    with h5py.File(out / "inj_ffa_results.h5", "r") as f:
        params = f["param_sets"][...]
        snr = f["snr"][...]
    near = (np.abs(params[:, 1] - f0) < 2e-3 * f0) & (np.abs(params[:, 0] - acc) < 15)
    return float(snr[near].max()) if near.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--nsamps", type=int, default=2**25)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nbins50", action="store_true", help="nbins 50..800")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--memgb", type=float, default=10.0)
    args = parser.parse_args()
    builds = {
        "base": str(Path(args.base).resolve()),
        "new": str(Path(args.new).resolve()),
    }

    WORK.mkdir(parents=True, exist_ok=True)
    template_path = WORK / "template.toml"
    subprocess.run(
        [builds["base"], "search", "ffa", "-g", str(template_path)],
        check=True,
        capture_output=True,
    )
    template = template_path.read_text()
    rng = random.Random(args.seed)  # noqa: S311 (not cryptographic)
    cases = [
        (rng.uniform(0.006, 0.08), rng.uniform(-80, 80), 100 + i)
        for i in range(args.cases)
    ]
    diffs = []
    for period, acc, seed in cases:
        tim = simulate(builds["base"], period, acc, seed, args.nsamps)
        f0 = 1.0 / period
        for fourier in ("false", "true"):
            cfg = template
            for key, value in (
                ("timeseries", f'"{tim.resolve()}"'),
                ("f_min", f0 * 0.8),
                ("f_max", f0 * 1.25),
                ("acc_min", -100.0),
                ("acc_max", 100.0),
                ("use_fourier", fourier),
                ("nthreads", 1),
                ("max_process_memory_gb", args.memgb),
                ("backend", f'"{args.backend}"'),
                ("outdir", '"./"'),
                ("prefix", '"inj"'),
            ):
                cfg = set_key(cfg, key, value)
            if args.nbins50:
                cfg = set_key(set_key(cfg, "nbins", 50), "nbins_max", 800)
            res = {
                name: best_snr(
                    loki,
                    cfg,
                    WORK / f"run_{name}_{seed}_{fourier}_{args.nsamps}",
                    f0,
                    acc,
                )
                for name, loki in builds.items()
            }
            diffs.append(res["new"] - res["base"])
            print(
                f"P={period:.4f} acc={acc:6.1f} fourier={fourier:5s} "
                f"base={res['base']:.3f} new={res['new']:.3f} "
                f"diff={res['new'] - res['base']:+.3f}",
                flush=True,
            )
    d = np.array(diffs)
    d = d[np.isfinite(d)]
    print(
        f"mean diff {d.mean():+.3f}  min {d.min():+.3f}  max {d.max():+.3f}  n={len(d)}"
    )


if __name__ == "__main__":
    main()
