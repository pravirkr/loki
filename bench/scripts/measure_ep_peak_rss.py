"""Measure the peak RSS of an EPFreqSweep against the planner's memory model.

Each configuration runs in its own subprocess, so ``ru_maxrss`` is the peak of
that sweep alone. The report gives, per configuration:

- ``base``: RSS after importing loki and building the inputs (interpreter,
  numpy, libloki), which the planner does not model;
- ``peak``: peak RSS of the whole process;
- ``model``: ``EPRegionStats.max_memory_gb``, the planner's sweep peak;
- ``resid``: ``peak - base - model``, the memory the model does not explain.

A residual that stays well below ``kUnmodelledReserveGB`` (0.5 GB, see
``docs/memory.md``) and does not grow with the thread count supports keeping a
fixed reserve. Not part of the test suite.

Usage::

    python bench/scripts/measure_ep_peak_rss.py grid --nthreads 1 2 4 8
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import click
import numpy as np

from loki import libloki

GB = 1024**3


def peak_rss_bytes() -> int:
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def make_config(
    nthreads: int,
    nbins: int,
    use_fourier: bool,
    f_min: float,
    f_max: float,
    nsamps: int,
    max_memory_gb: float,
) -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=64e-6,
        nbins=nbins,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [f_min, f_max]], dtype=np.float64),
        ducy_max=0.3,
        use_fourier=use_fourier,
        nthreads=nthreads,
        max_process_memory_gb=max_memory_gb,
        bseg_brute=1024,
        bseg_ffa=nsamps // 8,
        prune_poly_order=2,
    )


def run_one(params: dict) -> dict:
    cfg = make_config(
        params["nthreads"],
        params["nbins"],
        params["use_fourier"],
        params["f_min"],
        params["f_max"],
        params["nsamps"],
        params["max_memory_gb"],
    )
    rng = np.random.default_rng(42)
    ts_e = rng.standard_normal(params["nsamps"]).astype(np.float32)
    ts_v = np.ones(params["nsamps"], dtype=np.float32)
    base = peak_rss_bytes()

    planner_cls = (
        libloki.prune.EPRegionPlannerFourier
        if params["use_fourier"]
        else libloki.prune.EPRegionPlannerTime
    )
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "plan.h5"
        # Plan once to read the model, then let the sweep reuse the cached plan.
        planner = planner_cls(
            cfg,
            plan_cache_file=str(cache),
            n_workers=min(params["nthreads"], params["n_runs"]),
        )
        model_gb = planner.stats.max_memory_gb
        nchunks = planner.nchunks
        del planner
        sweep = libloki.prune.EPFreqSweep(
            cfg,
            show_progress=False,
            plan_cache_file=str(cache),
            n_runs=params["n_runs"],
        )
        sweep.execute(ts_e, ts_v, outdir=tmp, file_prefix="rss")
    peak = peak_rss_bytes()
    return {
        **params,
        "nchunks": nchunks,
        "base_gb": base / GB,
        "peak_gb": peak / GB,
        "model_gb": model_gb,
        "resid_gb": (peak - base) / GB - model_gb,
    }


@click.group()
def cli() -> None:
    pass


@cli.command("one", hidden=True)
@click.argument("params_json")
def one(params_json: str) -> None:
    print(json.dumps(run_one(json.loads(params_json))))


@cli.command()
@click.option("--nthreads", type=int, multiple=True, default=(1, 2, 4))
@click.option("--nbins", type=int, multiple=True, default=(32, 64))
@click.option("--fourier/--time", "fourier", default=None, help="Default: both")
@click.option("--n-runs", type=int, default=None, help="Default: nthreads")
@click.option("--f-min", type=float, default=140.0)
@click.option("--f-max", type=float, default=142.0)
@click.option("--nsamps", type=int, default=1 << 16)
@click.option("--max-memory-gb", type=float, default=4.0)
def grid(
    nthreads: tuple[int, ...],
    nbins: tuple[int, ...],
    fourier: bool | None,
    n_runs: int | None,
    f_min: float,
    f_max: float,
    nsamps: int,
    max_memory_gb: float,
) -> None:
    """Run each configuration in a fresh subprocess and print a table."""
    modes = (False, True) if fourier is None else (fourier,)
    header = (
        f"{'thr':>4} {'nbins':>5} {'mode':>7} {'chunks':>6} {'base':>7} "
        f"{'peak':>7} {'model':>7} {'resid':>7}  (GB)"
    )
    click.echo(header)
    for nthr, nb, use_fourier in product(nthreads, nbins, modes):
        params = {
            "nthreads": nthr,
            "nbins": nb,
            "use_fourier": use_fourier,
            "n_runs": n_runs if n_runs is not None else nthr,
            "f_min": f_min,
            "f_max": f_max,
            "nsamps": nsamps,
            "max_memory_gb": max_memory_gb,
        }
        proc = subprocess.run(
            [sys.executable, __file__, "one", json.dumps(params)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            click.echo(f"{nthr:>4} {nb:>5} failed:\n{proc.stderr[-2000:]}")
            continue
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        mode = "fourier" if use_fourier else "time"
        click.echo(
            f"{nthr:>4} {nb:>5} {mode:>7} {r['nchunks']:>6} {r['base_gb']:>7.3f} "
            f"{r['peak_gb']:>7.3f} {r['model_gb']:>7.3f} {r['resid_gb']:>7.3f}",
        )


if __name__ == "__main__":
    cli()
