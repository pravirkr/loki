"""EP jerk test: fixed-threshold EPMultiPass on 2^25 samples (3 runs).

Usage::

    python jerk_test.py simulate            # writes the input (needs pyloki)
    python jerk_test.py run NAME [--backend cuda] [--domain fourier]
        [--nbins 64] [--batch 65536] [--max-sugg 4194304] [--kadane]

The input is a 7 ms pulsar with accel 500 m/s^2 and jerk 2 m/s^3 at S/N 12
(``$LOKI_BENCH_WORK/data/test_jerk_snr_12.0_ts.npz``). ``run`` prunes
140-145 Hz with the fixed jerk threshold scheme below (no planner), writes
``$LOKI_BENCH_WORK/ep_runs/NAME_pruning_nstages_128_results.h5`` and prints
the execute wall time. Compare two results with ``compare_multipass.py``.

The plan is fixed, so ``max_sugg`` is not forecast: from nbins 100 the tree
overflows (trims) on some levels, and the survivors then depend on the batch
size and differ between backends; only matched scores compare there.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from loki import libloki

WORK = Path(os.environ.get("LOKI_BENCH_WORK", "."))
DATA = WORK / "data" / "test_jerk_snr_12.0_ts.npz"
TSAMP = 64e-6
NSAMPS = 2**25

# Fixed jerk thresholds, one per pruning level (128 levels).
THRESHOLDS = [
    2.2, 2.2, 2.5, 2.5, 2.6, 3.1, 3.0, 3.1, 3.4, 3.2, 3.4, 3.6, 3.9, 4.0, 3.9,
    3.8, 3.8, 3.7, 3.7, 4.2, 4.2, 3.9, 3.9, 3.9, 4.4, 4.5, 4.4, 4.3, 4.2, 4.2,
    4.1, 4.2, 4.1, 4.0, 3.9, 4.2, 4.2, 3.8, 4.1, 4.2, 4.7, 5.1, 5.2, 5.1, 4.9,
    5.0, 4.8, 4.9, 4.7, 4.8, 4.7, 4.8, 4.8, 4.8, 4.7, 4.7, 4.9, 4.6, 4.9, 4.9,
    4.7, 5.0, 4.9, 4.8, 5.1, 5.1, 5.0, 5.2, 5.1, 5.0, 5.1, 5.2, 5.1, 5.3, 5.2,
    5.3, 5.3, 5.0, 5.5, 5.3, 5.5, 5.4, 5.4, 5.5, 5.7, 5.8, 5.6, 5.7, 5.8, 5.7,
    5.8, 5.9, 5.8, 6.0, 5.9, 6.1, 6.1, 5.9, 6.2, 5.9, 6.2, 5.8, 6.2, 6.2, 6.3,
    6.4, 6.4, 6.3, 6.4, 6.4, 6.4, 6.0, 6.5, 6.4, 6.6, 6.3, 6.7, 6.6, 6.7, 6.7,
    6.8, 6.4, 6.8, 6.7, 6.7, 6.7, 6.3,
]  # fmt: skip


def simulate() -> None:
    from pyloki.simulation.pulse import PulseSignalConfig  # noqa: PLC0415

    cfg = PulseSignalConfig(
        period=0.007,
        dt=TSAMP,
        nsamps=NSAMPS,
        snr=12.0,
        ducy=0.1,
        mod_kwargs={"acc": 500, "jerk": 2.0},
    )
    ts = cfg.generate(shape="gaussian")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, ts_e=ts.ts_e, ts_v=ts.ts_v, dt=TSAMP)
    print(f"wrote {DATA}")


def run(args: argparse.Namespace) -> None:
    data = np.load(DATA)
    fourier = args.domain == "fourier"
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=TSAMP,
        nbins=args.nbins,
        eta=1.0,
        param_limits=np.array([[-4.0, 4.0], [-1000.0, 1000.0], [140.0, 145.0]]),
        ducy_max=0.3,
        wtsp=1.2,
        use_fourier=fourier,
        nthreads=1,
        bseg_brute=NSAMPS // 8192,
        bseg_ffa=NSAMPS // 128,
        prune_poly_order=3,
        use_boxcar_kadane=args.kadane,
    )
    cls = libloki.prune.EPMultiPassFourier if fourier else libloki.prune.EPMultiPassTime
    prune = cls(
        cfg,
        np.array(THRESHOLDS),
        n_runs=3,
        max_sugg=args.max_sugg,
        batch_size=args.batch,
        poly_basis="taylor",
        backend=args.backend,
        device=0,
    )
    outdir = WORK / "ep_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prune.execute(data["ts_e"], data["ts_v"], str(outdir), args.name)
    print(f"execute wall {time.perf_counter() - t0:.3f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("simulate")
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--domain", default="fourier", choices=["fourier", "time"])
    p.add_argument("--nbins", type=int, default=64)
    p.add_argument("--batch", type=int, default=2**16)
    p.add_argument("--max-sugg", type=int, default=2**22)
    p.add_argument("--kadane", action="store_true")
    args = parser.parse_args()
    if args.cmd == "simulate":
        simulate()
    else:
        run(args)


if __name__ == "__main__":
    main()
