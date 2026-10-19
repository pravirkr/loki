"""Bit-exact CUDA golden for DynamicThresholdScheme.solve() (mode "shared").

The manifest ``golden/solve_manifest.json`` holds the final thresholds, Pd and
cost of the research workspace's solver at the three refinement levels (tags
keep/E18, keep/E20, keep/E22 of dts_work, built with the same nvcc and Release
flags): refine "none", "shortlist" and "full" must reproduce them bit for
bit. The manifest records the provenance. A different nvcc, flags or GPU
architecture can change the bits; then compare statistically instead
(solve_suite.py).

    python golden_solve.py check [--refine none|shortlist|full]...
"""

from __future__ import annotations

import json
import pathlib
import sys

import click
import numpy as np

from loki import libloki

lt = libloki.thresholds

MANIFEST = (
    pathlib.Path(__file__).resolve().parent.parent / "golden" / "solve_manifest.json"
)


def run_case(man: dict, case: dict, refine: str) -> dict:
    fam = man["families"][case["family"]]
    scheme = lt.DynamicThresholdScheme(
        np.asarray(fam["branching"], dtype=np.float32),
        fam["ref_ducy"],
        nbins=man["provenance"]["nbins"],
        ntrials=man["provenance"]["ntrials_h0"],
        nprobs=man["provenance"]["nprobs_above"],
        snr_final=case["snr"],
        nthresholds=100,
        ducy_max=fam["ducy_max"],
        wtsp=fam["wtsp"],
        beam_width=0.7,
        trials_start=1,
        mode="shared",
        seed=case["seed"],
        backend="cuda",
        batch_size=256,
        guess="window",
        min_pd=case["pd_min"],
        w_terminal=man["provenance"]["w"],
        refine=refine,
    )
    scheme.run(10)
    return scheme.solve()


@click.command()
@click.argument("cmd", type=click.Choice(["check"]))
@click.option(
    "--refine",
    "refines",
    multiple=True,
    default=("none", "shortlist", "full"),
)
def main(cmd: str, refines: tuple[str, ...]) -> None:
    man = json.loads(MANIFEST.read_text())
    bad = 0
    n = 0
    for case in man["cases"]:
        if case["refine"] not in refines:
            continue
        n += 1
        sol = run_case(man, case, case["refine"])
        ref = np.asarray(case["thresholds"], dtype=np.float32)
        got = np.asarray(sol["thresholds"], dtype=np.float32)
        same = ref.shape == got.shape and np.array_equal(
            ref.view(np.uint32),
            got.view(np.uint32),
        )
        tag = f"{case['refine']:9s} {case['family']:6s} snr={case['snr']:<5} pd={case['pd_min']:<5} seed={case['seed']}"
        if same:
            click.echo(f"ok    {tag}  pd={sol['pd']:.4f} (ref {case['pd_claim']:.4f})")
        else:
            bad += 1
            d = np.abs(ref - got).max() if ref.shape == got.shape else float("nan")
            click.echo(
                f"DIFF  {tag}  max|dt|={d:.3g}  pd={sol['pd']:.4f} (ref {case['pd_claim']:.4f})",
            )
    click.echo("bit-identical" if bad == 0 else f"{bad} of {n} cases differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
