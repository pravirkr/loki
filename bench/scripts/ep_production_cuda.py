"""Production-scale EP search on CUDA, from a simulated accelerated-jerk pulse.

This is the EPFreqSweep port of tmp_files/test_jerk_cuda.py. That script used a
tiny chunk and EPMultiPassFourierCUDA with hand-made thresholds. Here the
planner designs the thresholds and chunks the band to the device, and the
sweep runs every chunk of the band.

The pulse and the search are the same as the CLI pair in
ep_production_cuda.toml, so either driver gives the same plan (the cache file
is shared) and the same candidates:

    loki simulate --period 0.007 --dt 0.000064 --nsamps 33554432 --snr 12 \\
        --ducy 0.1 --acc 500 --jerk 2 --seed 42 -o ep_production_pulse.tim
    loki search ep -c bench/scripts/ep_production_cuda.toml

The script reports the plan, the device high-water mark against the model, the
run time, and the best candidate against the injected parameters. The device
is sampled by the sampler of measure_ep_peak_rss.py, in a child process
because the sweep holds the interpreter.

Usage::

    python bench/scripts/ep_production_cuda.py run --outdir pruning_results/ep_production_cuda
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import click
import h5py
import numpy as np

from loki import libloki

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_ep_peak_rss as rss  # noqa: E402

GB = 1024**3

# The injected pulse (tmp_files/test_jerk_cuda.py).
PERIOD = 0.007
DT = 64e-6
SNR = 12.0
DUCY = 0.1
ACC = 500.0
JERK = 2.0
SEED = 42

# Search: jerk and acceleration are wide enough to hold the injected values.
JERK_LIMITS = (-4.0, 4.0)
ACC_LIMITS = (-1000.0, 1000.0)


def simulate(nsamps: int) -> tuple[np.ndarray, np.ndarray]:
    mod = libloki.simulation.ModulatorParams()
    mod.acc = ACC
    mod.jerk = JERK
    pulse = libloki.simulation.PulseSignalConfig(
        period=PERIOD,
        dt=DT,
        nsamps=nsamps,
        snr=SNR,
        ducy=DUCY,
        mod_type="derivative",
        mod=mod,
        seed=SEED,
    )
    series = pulse.generate(shape="gaussian")
    ts_e = np.ascontiguousarray(series.ts_e, dtype=np.float32)
    ts_v = np.ascontiguousarray(series.ts_v, dtype=np.float32)
    return ts_e, ts_v


def make_config(
    nsamps: int, f_min: float, f_max: float, memory_gb: float
) -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=DT,
        nbins=64,
        eta=1.0,
        # Order of the parameters: jerk, acceleration, frequency.
        param_limits=np.array(
            [JERK_LIMITS, ACC_LIMITS, (f_min, f_max)], dtype=np.float64
        ),
        ducy_max=0.3,
        wtsp=1.2,
        use_fourier=True,
        nthreads=1,
        max_process_memory_gb=memory_gb,
        bseg_brute=nsamps // 8192,
        bseg_ffa=nsamps // 128,
        prune_poly_order=3,
    )


def top_candidate(result_file: Path) -> dict:
    """Best leaf over every run, with its parameters (rows 0..n_params-1)."""
    with h5py.File(result_file, "r") as h5:
        names = [str(n) for n in h5.attrs["param_names"]]
        best = {"score": -np.inf, "chunk": "", "run": "", "leaf": -1}
        n_leaves = 0
        for chunk_name in h5["chunks"]:
            runs = h5["chunks"][chunk_name]["runs"]
            for run_name in runs:
                scores = runs[run_name]["scores"][()]
                n_leaves += scores.size
                if scores.size == 0:
                    continue
                leaf = int(np.argmax(scores))
                if scores[leaf] > best["score"]:
                    best = {
                        "score": float(scores[leaf]),
                        "chunk": chunk_name,
                        "run": run_name,
                        "leaf": leaf,
                    }
        if best["leaf"] < 0:
            return {"leaves": n_leaves, "params": {}, **best}
        param_sets = h5["chunks"][best["chunk"]]["runs"][best["run"]]["param_sets"]
        row = param_sets[best["leaf"], : len(names), 0]
    return {
        "leaves": n_leaves,
        "params": {name: float(v) for name, v in zip(names, row, strict=True)},
        **best,
    }


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--outdir", default="pruning_results/ep_production_cuda", show_default=True
)
@click.option(
    "--nsamps", default=1 << 25, show_default=True, help="Sample count (power of 2)."
)
@click.option("--f-min", default=140.0, show_default=True, help="Band start (Hz).")
@click.option("--f-max", default=145.0, show_default=True, help="Band end (Hz).")
@click.option(
    "--memory-gb", default=40.0, show_default=True, help="Device memory cap (GB)."
)
@click.option(
    "--n-runs",
    default=16,
    show_default=True,
    help="Runs (reference segments) to prune.",
)
@click.option("--device", default=0, show_default=True, help="CUDA device ordinal.")
def run(
    outdir: str,
    nsamps: int,
    f_min: float,
    f_max: float,
    memory_gb: float,
    n_runs: int,
    device: int,
) -> None:
    """Plan the band, sweep it on the GPU and check the injected pulse."""
    if nsamps & (nsamps - 1):
        msg = "nsamps must be a power of 2"
        raise click.BadParameter(msg)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "ep_production_ep_plan.h5"
    cfg = make_config(nsamps, f_min, f_max, memory_gb)

    print(f"simulating the pulse: nsamps={nsamps}, injected f={1.0 / PERIOD:.4f} Hz")  # noqa: T201
    ts_e, ts_v = simulate(nsamps)

    # Plan first: the planner reports the model, and the sweep loads the cache.
    planner = libloki.prune.EPRegionPlannerFourier(
        cfg, plan_cache_file=str(cache), backend="cuda", device=device
    )
    stats = planner.stats
    print(  # noqa: T201
        f"plan: {planner.nchunks} chunks, model peak {stats.max_memory_gb:.3f} GB "
        f"of a {stats.memory_limit_gb:.3f} GB device limit"
    )
    model_gb = float(stats.max_memory_gb)
    nchunks = int(planner.nchunks)
    del planner

    # The context exists now, so it counts in the baseline, not in the peak.
    scratch = out / ".sampler"
    scratch.mkdir(exist_ok=True)
    stop = scratch / "stop"
    ready = scratch / "ready"
    sampled = scratch / "device.json"
    for path in (stop, ready, sampled):
        path.unlink(missing_ok=True)
    sampler = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            rss.__file__,
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

    sweep = libloki.prune.EPFreqSweep(
        cfg,
        show_progress=True,
        plan_cache_file=str(cache),
        n_runs=n_runs,
        backend="cuda",
        device=device,
    )
    start = time.perf_counter()
    try:
        sweep.execute(ts_e, ts_v, outdir=str(out), file_prefix="ep_production")
    finally:
        stop.write_text("stop")
        sampler.wait(timeout=60)
    elapsed = time.perf_counter() - start

    info = json.loads(sampled.read_text())
    peak_gb = (info["peak"] - info["baseline"]) / GB
    result_file = out / "ep_production_ep_results.h5"
    best = top_candidate(result_file)

    summary = {
        "nsamps": nsamps,
        "nchunks": nchunks,
        "model_peak_gb": model_gb,
        "device_peak_gb": peak_gb,
        "residual_gb": peak_gb - model_gb,
        "runtime_s": elapsed,
        "n_runs": n_runs,
        "leaves": best["leaves"],
        "top_score": best["score"],
        "top_chunk": best["chunk"],
        "top_run": best["run"],
        "top_params": best["params"],
        "injected": {"jerk": JERK, "acc": ACC, "freq": 1.0 / PERIOD},
    }
    (out / "ep_production_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"runtime {elapsed:.1f} s, {nchunks} chunks, {best['leaves']} leaves")  # noqa: T201
    print(  # noqa: T201
        f"device: model {model_gb:.3f} GB, high-water {peak_gb:.3f} GB "
        f"(residual {peak_gb - model_gb:+.3f} GB)"
    )
    print(  # noqa: T201
        f"best: score {best['score']:.2f} in {best['chunk']}/{best['run']} "
        f"params {best['params']}"
    )
    print(  # noqa: T201
        f"injected: jerk {JERK}, acc {ACC}, f {1.0 / PERIOD:.4f} Hz"
    )


if __name__ == "__main__":
    cli()
