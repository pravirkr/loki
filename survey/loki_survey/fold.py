"""``fold``: fold the strongest candidates with loki's own model of their leaf.

For each of the strongest selected groups (``[fold] candidates``), the group's
representative is folded on its own series with loki's phase model of its
leaf (``libloki.psr_utils.leaf_phase``, the model the search used) and scored
with loki's boxcar scorer over the region's width trials. Three checks come
with it:

- a control fold at ``control_factor`` times the frequency, where a real
  pulsar smears and its S/N drops to the noise;
- the two halves of the span folded apart, where a steady pulsar gives about
  S/N / sqrt(2) in each, and interference often only in one;
- with ``dm_curve``, the same model folded on every series (DM) of the run.

An EP candidate's leaf comes from the unit output (``run-unit`` keeps it); an
FFA candidate's is built from its Taylor parameters, as FFA has no other.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from loki_survey.collect import CANDIDATES_H5, load_groups
from loki_survey.collect import SUMMARY as COLLECT_SUMMARY
from loki_survey.prep import load_arrays, load_manifest
from loki_survey.rundir import RunDir, atomic_write_json, read_json

if TYPE_CHECKING:
    from loki_survey.config import Resolved
    from loki_survey.plan import Unit

FOLDS_H5 = "folds.h5"
FOLD_SUMMARY = "summary.json"


def taylor_leaf(native: np.ndarray) -> np.ndarray:
    """Build a leaf from Taylor parameters [..., accel, freq]: no errors, no delay."""
    nparams = native.size
    leaf = np.zeros((nparams + 2, 2), dtype=np.float64)
    leaf[:nparams, 0] = native
    leaf[nparams + 1, 0] = native[-1]
    return leaf


def candidate_leaf(output: h5py.File, candidate_id: int) -> np.ndarray:
    """Return the leaf of the unit candidate with this id."""
    ids = output["candidates/id"][:]
    rows = np.nonzero(ids == candidate_id)[0]
    if rows.size != 1:
        msg = f"candidate {candidate_id} not found once in {output.filename}"
        raise KeyError(msg)
    row = int(rows[0])
    if "survey/leaves" in output:
        return np.asarray(output["survey/leaves"][row])
    return taylor_leaf(np.asarray(output["candidates/param_sets"][row]))


def fold_snr(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    leaf: np.ndarray,
    tsamp: float,
    t_ref: float,
    nbins: int,
    widths: np.ndarray,
) -> tuple[float, int, np.ndarray, str]:
    """Fold a series with a leaf's phase; return S/N, best width, profile, model."""
    from loki import libloki  # noqa: PLC0415

    times = np.arange(ts_e.size, dtype=np.float64) * tsamp
    phase, model = libloki.psr_utils.leaf_phase(leaf, times, t_ref)
    profile = libloki.fold.fold_phase(ts_e, ts_v, phase, nbins)
    snr = libloki.scores.snr_boxcar_3d(profile[None, :, :], widths)[0]
    best = int(np.argmax(snr))
    return float(snr[best]), int(widths[best]), profile, str(model)


def _series_arrays(
    run: RunDir,
    series: str,
    downsample: int,
    nsamps: int,
) -> tuple[np.ndarray, np.ndarray]:
    ts_e, ts_v, _ = load_arrays(run, series, downsample)
    n = min(nsamps, ts_e.size)
    return (
        np.ascontiguousarray(ts_e[:n], dtype=np.float32),
        np.ascontiguousarray(ts_v[:n], dtype=np.float32),
    )


def fold_group(
    run: RunDir,
    unit: Unit,
    leaf: np.ndarray,
    t_ref: float,
    control_factor: float,
    dm_series: list[tuple[str, float]],
) -> dict[str, Any]:
    """Fold one group's representative and its checks."""
    from loki import libloki  # noqa: PLC0415

    widths = np.asarray(
        libloki.scores.generate_box_width_trials(unit.nbins, ducy_max=unit.ducy_max),
        dtype=np.uint64,
    )
    ts_e, ts_v = _series_arrays(run, unit.series, unit.downsample, unit.nsamps)
    snr, width, profile, model = fold_snr(
        ts_e,
        ts_v,
        leaf,
        unit.tsamp,
        t_ref,
        unit.nbins,
        widths,
    )
    control = leaf.copy()
    control[-3, 0] *= control_factor  # the frequency row
    snr_control, _, profile_control, _ = fold_snr(
        ts_e,
        ts_v,
        control,
        unit.tsamp,
        t_ref,
        unit.nbins,
        widths,
    )
    half = ts_e.size // 2
    halves = []
    for lo, hi in ((0, half), (half, ts_e.size)):
        # Each half keeps its own sample times, so it is folded as part of
        # the same span: zero weight outside it.
        e = np.zeros_like(ts_e)
        v = np.zeros_like(ts_v)
        e[lo:hi], v[lo:hi] = ts_e[lo:hi], ts_v[lo:hi]
        halves.append(fold_snr(e, v, leaf, unit.tsamp, t_ref, unit.nbins, widths)[0])
    curve = []
    for series, dm in dm_series:
        e, v = _series_arrays(run, series, unit.downsample, unit.nsamps)
        curve.append(
            (dm, fold_snr(e, v, leaf, unit.tsamp, t_ref, unit.nbins, widths)[0]),
        )
    return {
        "unit": unit.id,
        "model": model,
        "snr": snr,
        "width": width,
        "snr_control": snr_control,
        "snr_halves": halves,
        "dm_curve": curve,
        "profile": profile,
        "profile_control": profile_control,
    }


def fold(run: RunDir, resolved: Resolved, units: list[Unit]) -> dict[str, Any]:
    """Fold the strongest selected groups; write folds/folds.h5 and a summary."""
    cfg = resolved.analysis.fold
    collected = read_json(run.collect_dir / COLLECT_SUMMARY)
    groups = load_groups(run)
    with h5py.File(run.collect_dir / CANDIDATES_H5, "r") as handle:
        unit_ids = handle["units/id"].asstr()[:]
        members_unit = handle["members/unit"][:]
        members_cid = handle["members/unit_candidate_id"][:]
    by_id = {u.id: u for u in units}
    selected = np.nonzero(groups["selected"])[0][: cfg.candidates]
    folded = []
    out_dir = run.root / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{FOLDS_H5}.writing"
    with h5py.File(tmp, "w") as out:
        for g in selected:
            rep = int(groups["rep_member"][g])
            unit = by_id[str(unit_ids[members_unit[rep]])]
            with h5py.File(run.unit_output(unit.id), "r") as output:
                leaf = candidate_leaf(output, int(members_cid[rep]))
                t_ref = float(output["observation"].attrs["reference_time_seconds"])
            dm_series = _dm_series(run, units, unit) if cfg.dm_curve else []
            result = fold_group(
                run,
                unit,
                leaf,
                t_ref,
                cfg.control_factor,
                dm_series,
            )
            group = out.create_group(f"group_{int(g):06d}")
            group.create_dataset("profile", data=result.pop("profile"))
            group.create_dataset("profile_control", data=result.pop("profile_control"))
            group.create_dataset("leaf", data=leaf)
            group.create_dataset(
                "dm_curve",
                data=np.asarray(result["dm_curve"], dtype=np.float64).reshape(-1, 2),
            )
            row = {
                "group": int(g),
                "f0": float(groups["f0"][g]),
                "search_snr": float(groups["snr"][g]),
                **result,
            }
            for key in ("unit", "model", "snr", "width", "snr_control"):
                group.attrs[key] = row[key]
            folded.append(row)
    tmp.replace(out_dir / FOLDS_H5)
    summary = {
        "analysis_sha256": resolved.analysis_sha256,
        "collect_candidates_sha256": collected["candidates_sha256"],
        "fold": asdict(cfg),
        "groups": folded,
        "time": time.time(),
    }
    atomic_write_json(out_dir / FOLD_SUMMARY, summary)
    return summary


def _dm_series(run: RunDir, units: list[Unit], unit: Unit) -> list[tuple[str, float]]:
    """Return every series of the run searched in this unit's region, by DM."""
    seen: dict[str, float] = {}
    for other in units:
        if other.region == unit.region and other.series not in seen:
            obs = load_manifest(run, other.series)["observation"]
            dm = obs.get("dm")
            seen[other.series] = float(dm) if dm is not None else float("nan")
    return sorted(seen.items(), key=lambda item: item[1])
