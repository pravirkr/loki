"""Measure the peak RSS of an EPFreqSweep against the planner's memory model.

Each configuration runs in its own subprocess, so ``ru_maxrss`` is the peak of
that sweep alone. The report gives, per configuration:

- ``base``: RSS after importing loki and building the inputs (interpreter,
  numpy, libloki), which the planner does not model;
- ``peak``: peak RSS of the whole process;
- ``model``: ``EPRegionStats.max_memory_gb``, the planner's sweep peak;
- ``resid``: ``peak - base - model``, the memory the model does not explain.

With ``--backend cuda`` a side process samples ``cudaMemGetInfo`` for the whole
sweep (the sweep itself holds the interpreter lock, so a thread in this
process would not run). ``base`` is the device memory after the CUDA context
exists and ``model`` is the device model. On CUDA, host memory is not modelled
or checked. ``tests/cpp/internal/ep_memory_cuda_t.cpp`` checks the model
against the real allocations exactly.

A residual that stays well below ``kUnmodelledReserveGB`` (0.5 GB, see
``docs/memory.md``) and does not grow with the thread count supports keeping a
fixed reserve. Not part of the test suite.

Usage::

    python bench/scripts/measure_ep_peak_rss.py grid --nthreads 1 2 4 8
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import resource
import subprocess
import sys
import tempfile
import time
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


def device_used_bytes() -> int:
    """Device memory in use (bytes), from cudaMemGetInfo. Device-wide."""
    libname = ctypes.util.find_library("cudart")
    if libname is None:
        msg = "libcudart was not found"
        raise RuntimeError(msg)
    lib = ctypes.CDLL(libname)
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    err = lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if err != 0:
        msg = f"cudaMemGetInfo failed ({err})"
        raise RuntimeError(msg)
    return int(total.value - free.value)


def sample_device(stop_file: str, out_file: str, ready_file: str) -> None:
    """Poll device memory until ``stop_file`` appears. Writes baseline and peak."""
    baseline = device_used_bytes()
    Path(ready_file).write_text("ready")
    peak = baseline
    stop = Path(stop_file)
    while not stop.exists():
        peak = max(peak, device_used_bytes())
        time.sleep(0.02)
    peak = max(peak, device_used_bytes())
    Path(out_file).write_text(json.dumps({"baseline": baseline, "peak": peak}))


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
    cuda = params["backend"] == "cuda"
    if cuda:
        # Create the context first, so it counts in the base, not the model.
        libloki.prune.EPRegionPlannerTime(cfg, backend="cuda")
    base = 0 if cuda else peak_rss_bytes()

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
            n_workers=None if cuda else min(params["nthreads"], params["n_runs"]),
            backend=params["backend"],
        )
        model_gb = planner.stats.max_memory_gb
        nchunks = planner.nchunks
        del planner
        sampler = None
        stop = Path(tmp) / "stop"
        ready = Path(tmp) / "ready"
        sampled = Path(tmp) / "device.json"
        if cuda:
            sampler = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    __file__,
                    "device-sample",
                    str(stop),
                    str(sampled),
                    str(ready),
                ],
            )
            while not ready.exists():
                if sampler.poll() is not None:
                    msg = "device sampler exited before it was ready"
                    raise RuntimeError(msg)
                time.sleep(0.01)
        try:
            sweep = libloki.prune.EPFreqSweep(
                cfg,
                show_progress=False,
                plan_cache_file=str(cache),
                n_runs=params["n_runs"],
                backend=params["backend"],
            )
            sweep.execute(ts_e, ts_v, outdir=tmp, file_prefix="rss")
        finally:
            if sampler is not None and not stop.exists():
                stop.write_text("stop")
                sampler.wait(timeout=30)
        if sampler is not None:
            info = json.loads(sampled.read_text())
            base = info["baseline"]
            peak = info["peak"]
        else:
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


@cli.command("device-sample", hidden=True)
@click.argument("stop_file")
@click.argument("out_file")
@click.argument("ready_file")
def device_sample(stop_file: str, out_file: str, ready_file: str) -> None:
    sample_device(stop_file, out_file, ready_file)


@cli.command()
@click.option("--nthreads", type=int, multiple=True, default=(1, 2, 4))
@click.option("--nbins", type=int, multiple=True, default=(32, 64))
@click.option("--fourier/--time", "fourier", default=None, help="Default: both")
@click.option("--n-runs", type=int, default=None, help="Default: nthreads")
@click.option("--f-min", type=float, default=140.0)
@click.option("--f-max", type=float, default=142.0)
@click.option("--nsamps", type=int, default=1 << 16)
@click.option("--max-memory-gb", type=float, default=4.0)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"]),
    default="cpu",
    help="cuda measures device memory instead of RSS",
)
def grid(
    nthreads: tuple[int, ...],
    nbins: tuple[int, ...],
    fourier: bool | None,
    n_runs: int | None,
    f_min: float,
    f_max: float,
    nsamps: int,
    max_memory_gb: float,
    backend: str,
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
            "backend": backend,
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
