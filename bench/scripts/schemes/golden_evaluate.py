"""Bit-exact CUDA golden for DynamicThresholdScheme.evaluate().

Commands:
    make   write ``golden/evaluate_cuda.npz`` (states of fixed schedules)
    check  compare to the on-disk golden

Cases cover nbins 32/48/64/128, ntrials that are not a multiple of the block
size, and four schedules: mid (prunes some noise), low (prunes nothing),
floor (no H0 survivor after mid-pattern: the PR #14 floor), extinct (no H1
survivor: later stages stay empty). States are compared field-wise on raw
bytes (State has padding). Run with PYTHONPATH=src, never ``python -I``.
The golden was made with the one-block evaluator that the device-wide
compaction replaced.
"""

from __future__ import annotations

import pathlib
import sys

import click
import numpy as np
from _branching import BRANCHING_PROD, SMALL_KW

from loki import libloki

lt = libloki.thresholds

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "golden" / "evaluate_cuda.npz"


def make_scheme(bp, nbins, ntrials):
    kw = {**SMALL_KW, "nbins": nbins, "ntrials": ntrials}
    return lt.DynamicThresholdScheme(
        np.asarray(bp, dtype=np.float32),
        mode="improved",
        seed=7,
        backend="cuda",
        **kw,
    )


def schedule(nstages: int, kind: str) -> np.ndarray:
    t = np.linspace(2.0, 5.0, nstages, dtype=np.float32)
    if kind == "low":
        return np.full(nstages, -50.0, dtype=np.float32)
    if kind == "floor":
        t[nstages // 2 :] = 9.0
    elif kind == "extinct":
        t[nstages // 3 :] = 1000.0
    return t


def collect() -> dict[str, np.ndarray]:
    out = {}
    bp24 = BRANCHING_PROD[:24]
    for nbins in (32, 48, 64, 128):
        scheme = make_scheme(bp24, nbins, 256)
        for ntrials in (1024, 5000, 32768):
            for kind in ("mid", "low", "floor", "extinct"):
                out[f"bp24_nb{nbins}_n{ntrials}_{kind}"] = scheme.evaluate(
                    schedule(len(bp24), kind),
                    ntrials,
                    12345,
                )
    scheme = make_scheme(BRANCHING_PROD, 64, 256)
    for ntrials in (1024, 32768, 262144):
        for kind in ("mid", "floor"):
            out[f"prod_n{ntrials}_{kind}"] = scheme.evaluate(
                schedule(len(BRANCHING_PROD), kind),
                ntrials,
                99,
            )
    return out


def field_bytes(states: np.ndarray) -> list[np.ndarray]:
    return [
        np.ascontiguousarray(states[f]).view(np.uint8).reshape(len(states), -1)
        for f in states.dtype.names
    ]


@click.command()
@click.argument("cmd", type=click.Choice(["make", "check"]))
def main(cmd: str) -> None:
    res = collect()
    if cmd == "make":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **res)
        click.echo(f"saved {len(res)} cases to {GOLDEN}")
        return
    ref = np.load(GOLDEN)
    bad = 0
    for name, states in res.items():
        old = ref[name]
        same = states.shape == old.shape and all(
            np.array_equal(a, b)
            for a, b in zip(field_bytes(states), field_bytes(old), strict=True)
        )
        if not same:
            bad += 1
            click.echo(f"DIFF {name}")
    click.echo("bit-identical" if bad == 0 else f"{bad} of {len(res)} cases differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
