"""Bit-exact CUDA golden harness for DynamicThresholdSchemeCUDA.

Commands:
    make   write ``golden/<config>.npz`` (states + best path)
    check  compare to on-disk goldens and print timings
    repro  run-to-run determinism on the same machine

States are compared field-wise on raw bytes (compound dtype), so any single-bit
change in any cell is reported.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import time

import click
import h5py
import numpy as np
from _branching import BRANCHING_PROD, PROD_KW, SMALL_KW

from loki import libculoki

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
SEED = 20261004

CONFIGS = {
    "prod_improved": (BRANCHING_PROD, {**PROD_KW, "mode": "improved"}, 11),
    "prod_legacy": (BRANCHING_PROD, {**PROD_KW, "mode": "legacy"}, 11),
    "small_improved": (BRANCHING_PROD[:24], {**SMALL_KW, "mode": "improved"}, 6),
    "small_legacy": (BRANCHING_PROD[:24], {**SMALL_KW, "mode": "legacy"}, 6),
    "small32_improved": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "nbins": 32, "mode": "improved"},
        6,
    ),
    "nb48_improved": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "nbins": 48, "mode": "improved"},
        6,
    ),
    "nb50_improved": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "nbins": 50, "mode": "improved"},
        6,
    ),
    "nb128_improved": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "nbins": 128, "mode": "improved"},
        6,
    ),
    "wide_improved": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "ducy_max": 0.6, "mode": "improved"},
        6,
    ),
    "batch64_improved": (
        BRANCHING_PROD[:40],
        {**SMALL_KW, "mode": "improved", "batch_size": 64},
        6,
    ),
    "batch64_legacy": (
        BRANCHING_PROD[:40],
        {**SMALL_KW, "mode": "legacy", "batch_size": 64},
        6,
    ),
    "nb50_legacy": (
        BRANCHING_PROD[:24],
        {**SMALL_KW, "nbins": 50, "mode": "legacy"},
        6,
    ),
}


def run_config(
    name: str,
    seed: int = SEED,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    bp, kw, neigh = CONFIGS[name]
    kw = dict(kw)
    if batch_size is not None:
        kw["batch_size"] = batch_size
    dyn = libculoki.thresholds.DynamicThresholdSchemeCUDA(bp, seed=seed, **kw)
    t0 = time.perf_counter()
    dyn.run(thres_neigh=neigh)
    dt = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as td:
        path = dyn.save(outdir=td)
        with h5py.File(path, "r") as f:
            states = f["states"][...]
    raw = field_bytes(states).ravel()
    best = np.asarray(dyn.get_best_path_thresholds(), dtype=np.float32)
    return states, raw, best, dt


def field_bytes(states: np.ndarray) -> np.ndarray:
    """Concatenate per-field raw bytes (ignores compound padding bytes)."""
    return np.concatenate(
        [
            np.ascontiguousarray(states[f])
            .view(np.uint8)
            .reshape(
                (*states.shape, -1),
            )
            for f in states.dtype.names
        ],
        axis=-1,
    )


def describe_diff(
    states_ref: np.ndarray,
    states_new: np.ndarray,
) -> tuple[int, np.ndarray]:
    cell_diff = (field_bytes(states_ref) != field_bytes(states_new)).any(-1)
    stages = np.unique(np.nonzero(cell_diff)[0])
    return int(cell_diff.sum()), stages


@click.command()
@click.argument("cmd", type=click.Choice(["make", "check", "repro"]))
@click.option("--configs", default=",".join(CONFIGS))
@click.option("--repeat", type=int, default=3)
@click.option("--batch-size", type=int, default=None)
def main(cmd: str, configs: str, repeat: int, batch_size: int | None) -> None:
    names = configs.split(",")
    GOLDEN_DIR.mkdir(exist_ok=True)
    ok = True
    for name in names:
        ok_cfg = True
        if cmd == "make":
            states, raw, best, dt = run_config(name, batch_size=batch_size)
            np.savez_compressed(GOLDEN_DIR / f"{name}.npz", states=states, best=best)
            print(
                f"{name:18s} saved  sha={hashlib.sha256(raw).hexdigest()[:16]}"
                f"  t={dt:.3f}s  nonempty={int((~states['is_empty']).sum())}",
            )
            continue
        if cmd == "check":
            ref = np.load(GOLDEN_DIR / f"{name}.npz")
            ref_states, ref_best = ref["states"], ref["best"]
        times = []
        first = None
        for r in range(repeat + (1 if cmd == "check" else 0)):
            states, raw, best, dt = run_config(name, batch_size=batch_size)
            if cmd == "check" and r == 0:
                pass
            else:
                times.append(dt)
            if cmd == "repro":
                ref_states = first if first is not None else states
                ref_best = best if first is None else ref_best
                if first is None:
                    first = states
                    continue
            same = (
                np.array_equal(field_bytes(ref_states), field_bytes(states))
                and ref_best.tobytes() == best.tobytes()
            )
            if not same:
                ok = ok_cfg = False
                ncells, stages = describe_diff(ref_states, states)
                print(
                    f"{name:18s} run {r}: MISMATCH {ncells} cells, "
                    f"first stages {stages[:10].tolist()}",
                )
        tag = "OK" if ok_cfg else "FAIL"
        print(
            f"{name:18s} {cmd}: {tag}  "
            f"t_min={min(times):.3f}s t_med={np.median(times):.3f}s "
            f"(n={len(times)})",
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
