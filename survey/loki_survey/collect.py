"""``collect``: combine every unit's candidates into one list, with a ledger.

Each unit's output holds loki's candidates for one series (one DM), one band
and one search box, already grouped within the unit by loki's phase tracks.
``collect`` groups them across units: candidates whose frequencies lie within
``radius_bins`` resolutions 1/T of each other are one group, whatever their
box, band or DM. The grouping links neighbours in frequency (friends of
friends), so a group can grow by chaining; its frequency extent is recorded.

Grouping removes nothing: every unit candidate stays in the output as a member
of its group, and the group's representative is its strongest member. Each
group is labelled:

- ``dm_call``, from the DM of its strongest member against the DMs searched:
  ``undispersed`` (strongest at the lowest DM searched, and that DM is at most
  ``dm_zero_max``), ``lowest_dm`` and ``highest_dm`` (strongest at an edge of
  the DMs searched, so the source's DM may lie outside them), ``dispersed``
  (strongest inside them), ``single_dm`` (one DM searched) or ``unknown_dm``;
- ``harmonic_of``, a stronger group whose frequency times p/q (p, q up to
  ``harmonic_max``) is this group's frequency. A group already labelled a
  harmonic is never another's parent.

Candidates leave the list only by a removal rule, each off by default, applied
in this order: ``snr_min``, ``drop_undispersed``, ``remove_harmonics`` and
``max_candidates``. Every group a rule removes is written to the ledger with
the rule and its reason.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from loki_survey.rundir import (
    RunDir,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)

if TYPE_CHECKING:
    from loki_survey.config import Collect, Resolved
    from loki_survey.plan import Unit

SCHEMA = "loki-survey collect 1"
CANDIDATES_H5 = "candidates.h5"
LEDGER = "ledger.jsonl"
SUMMARY = "summary.json"
# Columns read from each unit's /candidates table.
COLUMNS = ("id", "f0", "f1", "f2", "snr", "width", "nbins", "duty_cycle")
NO_PARENT = -1
RULES = ("snr_min", "drop_undispersed", "remove_harmonics", "max_candidates")


class IncompleteRunError(RuntimeError):
    """Units have not finished; collect would leave their candidates out."""


def _unit_tables(
    run: RunDir,
    resolved: Resolved,
    units: list[Unit],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[str]]:
    """Read every finished unit; return its record, its candidates, the missing."""
    from loki.candidates import CandidateFile  # noqa: PLC0415

    records: list[dict[str, Any]] = []
    columns: dict[str, list[np.ndarray]] = {c: [] for c in (*COLUMNS, "unit")}
    missing = []
    for unit in units:
        summary_path = run.unit_summary(unit.id)
        if not summary_path.is_file():
            missing.append(unit.id)
            continue
        summary = read_json(summary_path)
        path = run.unit_output(unit.id)
        digest = sha256_file(path)
        if digest != summary["output_sha256"]:
            msg = f"{path}: its sha256 is not the one its summary recorded"
            raise ValueError(msg)
        with h5py.File(path, "r") as handle:
            science = handle["survey"].attrs["science_sha256"]
            obs = dict(handle["observation"].attrs)
        if science != resolved.science_sha256:
            msg = f"{path} was searched with another survey.toml"
            raise ValueError(msg)
        dm = float(obs["dm"]) if "dm" in obs else float("nan")
        record = {
            "id": unit.id,
            "series": unit.series,
            "region": unit.region,
            "band": unit.band,
            "box": unit.box,
            "kind": unit.kind,
            "f_min": unit.f_min,
            "f_max": unit.f_max,
            "dm": dm,
            "tobs": float(obs["tobs"]),
            "output_sha256": digest,
            "n_candidates": 0,
        }
        index = len(records)
        with CandidateFile(path) as cands:
            for batch in cands.iter_batches():
                rows = len(batch["snr"])
                record["n_candidates"] += rows
                for name in COLUMNS:
                    columns[name].append(np.asarray(batch[name]))
                columns["unit"].append(np.full(rows, index, dtype=np.int32))
        records.append(record)
    merged = {
        name: np.concatenate(parts) if parts else np.zeros(0)
        for name, parts in columns.items()
    }
    return records, merged, missing


def group_by_frequency(
    f0: np.ndarray,
    tobs: np.ndarray,
    radius_bins: float,
) -> np.ndarray:
    """Return a group number per row: neighbours in frequency closer than the radius.

    Two rows next to each other in frequency are linked when they differ by at
    most ``radius_bins`` / T, with T the shorter of their two spans.
    """
    if f0.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(f0, kind="stable")
    f_sorted, t_sorted = f0[order], tobs[order]
    tol = radius_bins / np.minimum(t_sorted[:-1], t_sorted[1:])
    breaks = np.diff(f_sorted) > tol
    labels_sorted = np.concatenate(([0], np.cumsum(breaks)))
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def _representatives(
    labels: np.ndarray,
    snr: np.ndarray,
    f0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's group, numbered by falling S/N, and each group's best row."""
    ngroups = int(labels.max()) + 1 if labels.size else 0
    # Strongest first; ties go to the lower frequency.
    order = np.lexsort((f0, -snr))
    rep = np.full(ngroups, -1, dtype=np.int64)
    for row in order:
        if rep[labels[row]] < 0:
            rep[labels[row]] = row
    # Number the groups by their representative's rank.
    rank = np.argsort(np.lexsort((f0[rep], -snr[rep])), kind="stable")
    renumber = np.empty(ngroups, dtype=np.int64)
    renumber[np.arange(ngroups)] = rank
    groups = renumber[labels]
    reps = np.empty(ngroups, dtype=np.int64)
    reps[renumber] = rep
    return groups, reps


def dm_call(peak_dm: float, dms: np.ndarray, dm_zero_max: float) -> str:
    """Return the DM call of a group whose strongest member is at ``peak_dm``."""
    if dms.size <= 1:
        return "single_dm"
    if not np.isfinite(peak_dm):
        return "unknown_dm"
    if peak_dm == dms[0]:
        return "undispersed" if peak_dm <= dm_zero_max else "lowest_dm"
    if peak_dm == dms[-1]:
        return "highest_dm"
    return "dispersed"


def harmonic_links(
    f0: np.ndarray,
    snr: np.ndarray,
    tol: np.ndarray,
    harmonic_max: int,
) -> list[tuple[int, int, int, int]]:
    """Return (child, parent, p, q) with f_child = (p/q) f_parent, parent stronger."""
    links: list[tuple[int, int, int, int]] = []
    if harmonic_max == 0 or f0.size < 2:
        return links
    order = np.argsort(f0)
    f_sorted = f0[order]
    for p in range(1, harmonic_max + 1):
        for q in range(1, harmonic_max + 1):
            if p == q or np.gcd(p, q) != 1:
                continue
            # The parent of a child at f sits at f q / p.
            target = f0 * q / p
            window = tol * q / p
            lo = np.searchsorted(f_sorted, target - window, side="left")
            hi = np.searchsorted(f_sorted, target + window, side="right")
            for child in np.nonzero(hi > lo)[0]:
                links.extend(
                    (int(child), int(parent), p, q)
                    for parent in order[lo[child] : hi[child]]
                    if snr[parent] > snr[child]
                )
    return links


def assign_harmonics(
    snr: np.ndarray,
    links: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give each group at most one parent: the strongest one not itself a harmonic."""
    n = snr.size
    parent = np.full(n, NO_PARENT, dtype=np.int64)
    num = np.zeros(n, dtype=np.int64)
    den = np.zeros(n, dtype=np.int64)
    by_child: dict[int, list[tuple[int, int, int]]] = {}
    for child, par, p, q in links:
        by_child.setdefault(child, []).append((par, p, q))
    # Strongest first, so a parent's own label is known before its children's.
    for child in np.argsort(-snr, kind="stable"):
        options = [
            (par, p, q)
            for par, p, q in by_child.get(int(child), [])
            if parent[par] == NO_PARENT
        ]
        if options:
            par, p, q = max(options, key=lambda o: (snr[o[0]], -o[2], -o[1]))
            parent[child], num[child], den[child] = par, p, q
    return parent, num, den


def collect(
    run: RunDir,
    resolved: Resolved,
    units: list[Unit],
    *,
    partial: bool = False,
) -> dict[str, Any]:
    """Group, label and select every unit's candidates; write them and the ledger."""
    cfg: Collect = resolved.analysis.collect
    records, rows, missing = _unit_tables(run, resolved, units)
    if missing and not partial:
        msg = (
            f"{len(missing)} of {len(units)} units have not finished (first: "
            f"{missing[0]}); finish them, or collect with --partial"
        )
        raise IncompleteRunError(msg)
    unit_dm = np.array([r["dm"] for r in records], dtype=np.float64)
    unit_tobs = np.array([r["tobs"] for r in records], dtype=np.float64)
    member_unit = rows["unit"].astype(np.int64)
    f0 = rows["f0"].astype(np.float64)
    snr = rows["snr"].astype(np.float64)
    tobs = unit_tobs[member_unit] if member_unit.size else np.zeros(0)
    dm = unit_dm[member_unit] if member_unit.size else np.zeros(0)
    labels = group_by_frequency(f0, tobs, cfg.radius_bins)
    member_group, reps = _representatives(labels, snr, f0)
    ngroups = reps.size
    dms = np.unique(unit_dm[np.isfinite(unit_dm)])

    n_members = np.bincount(member_group, minlength=ngroups)
    t_min = np.full(ngroups, np.inf)
    np.minimum.at(t_min, member_group, tobs)
    f_lo = np.full(ngroups, np.inf)
    f_hi = np.full(ngroups, -np.inf)
    np.minimum.at(f_lo, member_group, f0)
    np.maximum.at(f_hi, member_group, f0)
    curves = _dm_curves(member_group, dm, snr)
    boxes, n_units, n_dms = _membership(member_group, member_unit, records, dm, ngroups)

    g_f0, g_snr, g_dm = f0[reps], snr[reps], dm[reps]
    calls = np.array([dm_call(d, dms, cfg.dm_zero_max) for d in g_dm], dtype=object)
    low_ratio = np.zeros(ngroups)
    if dms.size > 1:
        for (group, curve_dm), best in curves.items():
            if curve_dm == dms[0]:
                low_ratio[group] = best / g_snr[group]
    tol = cfg.radius_bins / t_min
    parent, num, den = assign_harmonics(
        g_snr,
        harmonic_links(g_f0, g_snr, tol, cfg.harmonic_max),
    )

    removed_by = np.full(ngroups, "", dtype=object)
    reasons: dict[int, str] = {}
    if cfg.snr_min > 0:
        for g in np.nonzero(g_snr < cfg.snr_min)[0]:
            removed_by[g] = "snr_min"
            reasons[g] = f"S/N {g_snr[g]:.2f} below {cfg.snr_min:g}"
    if cfg.drop_undispersed:
        for g in np.nonzero((calls == "undispersed") & (removed_by == ""))[0]:
            removed_by[g] = "drop_undispersed"
            reasons[g] = f"strongest at DM {g_dm[g]:g}, the lowest searched"
    if cfg.remove_harmonics:
        for g in np.nonzero((parent != NO_PARENT) & (removed_by == ""))[0]:
            removed_by[g] = "remove_harmonics"
            reasons[g] = f"{num[g]}/{den[g]} times group {parent[g]}"
    if cfg.max_candidates > 0:
        kept = np.nonzero(removed_by == "")[0]  # already in falling S/N
        for g in kept[cfg.max_candidates :]:
            removed_by[g] = "max_candidates"
            reasons[g] = f"beyond the {cfg.max_candidates} strongest"
    selected = removed_by == ""

    groups = {
        "id": np.arange(ngroups, dtype=np.int64),
        "rep_member": reps,
        "f0": g_f0,
        "f1": rows["f1"].astype(np.float64)[reps] if ngroups else np.zeros(0),
        "f2": rows["f2"].astype(np.float64)[reps] if ngroups else np.zeros(0),
        "snr": g_snr,
        "width": rows["width"][reps].astype(np.uint32) if ngroups else np.zeros(0),
        "nbins": rows["nbins"][reps].astype(np.uint32) if ngroups else np.zeros(0),
        "duty_cycle": rows["duty_cycle"].astype(np.float64)[reps]
        if ngroups
        else np.zeros(0),
        "dm": g_dm,
        "unit": member_unit[reps] if ngroups else np.zeros(0, dtype=np.int64),
        "n_members": n_members,
        "n_units": n_units,
        "n_dms": n_dms,
        "boxes": boxes,
        "f_extent_bins": (f_hi - f_lo) * t_min if ngroups else np.zeros(0),
        "dm_call": calls,
        "lowest_dm_snr_ratio": low_ratio,
        "harmonic_of": parent,
        "harmonic_p": num,
        "harmonic_q": den,
        "selected": selected,
        "removed_by": removed_by,
    }
    members = {
        "group": member_group,
        "unit": member_unit,
        "unit_candidate_id": rows["id"].astype(np.uint64),
        "f0": f0,
        "f1": rows["f1"].astype(np.float64),
        "f2": rows["f2"].astype(np.float64),
        "snr": snr,
        "width": rows["width"].astype(np.uint32),
        "nbins": rows["nbins"].astype(np.uint32),
        "duty_cycle": rows["duty_cycle"].astype(np.float64),
        "dm": dm,
    }
    ledger = [
        {
            "rule": removed_by[g],
            "group": int(g),
            "f0": float(g_f0[g]),
            "snr": float(g_snr[g]),
            "dm": float(g_dm[g]),
            "dm_call": calls[g],
            "n_members": int(n_members[g]),
            "reason": reasons[g],
        }
        for g in np.nonzero(~selected)[0]
    ]
    inputs_sha256 = sha256_text(
        canonical_json(sorted(r["output_sha256"] for r in records)),
    )
    summary = {
        "schema": SCHEMA,
        "science_sha256": resolved.science_sha256,
        "analysis_sha256": resolved.analysis_sha256,
        "collect": asdict(cfg),
        "inputs_sha256": inputs_sha256,
        "units": len(units),
        "units_collected": len(records),
        "units_missing": missing,
        "dms": dms.tolist(),
        "members": int(f0.size),
        "groups": int(ngroups),
        "selected": int(selected.sum()),
        "removed": {rule: int((removed_by == rule).sum()) for rule in RULES},
        "dm_calls": {
            call: int((calls == call).sum()) for call in sorted(set(calls.tolist()))
        },
        "harmonics": int((parent != NO_PARENT).sum()),
        "time": time.time(),
    }
    _write(run, records, groups, members, curves, ledger, summary)
    return summary


def _dm_curves(
    member_group: np.ndarray,
    dm: np.ndarray,
    snr: np.ndarray,
) -> dict[tuple[int, float], float]:
    """Return each group's best S/N at each DM where it has a member."""
    curves: dict[tuple[int, float], float] = {}
    for group, d, s in zip(
        member_group.tolist(),
        dm.tolist(),
        snr.tolist(),
        strict=True,
    ):
        key = (group, d)
        if s > curves.get(key, -np.inf):
            curves[key] = s
    return curves


def _membership(
    member_group: np.ndarray,
    member_unit: np.ndarray,
    records: list[dict[str, Any]],
    dm: np.ndarray,
    ngroups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each group's boxes, number of units and number of DMs."""
    box_sets: list[set[str]] = [set() for _ in range(ngroups)]
    unit_sets: list[set[int]] = [set() for _ in range(ngroups)]
    dm_sets: list[set[float]] = [set() for _ in range(ngroups)]
    for group, unit, d in zip(
        member_group.tolist(),
        member_unit.tolist(),
        dm.tolist(),
        strict=True,
    ):
        box_sets[group].add(records[unit]["box"])
        unit_sets[group].add(unit)
        dm_sets[group].add(d)
    boxes = np.array([",".join(sorted(b)) for b in box_sets], dtype=object)
    n_units = np.array([len(u) for u in unit_sets], dtype=np.int64)
    n_dms = np.array([len(d) for d in dm_sets], dtype=np.int64)
    return boxes, n_units, n_dms


def _table(parent: h5py.Group, name: str, columns: dict[str, np.ndarray]) -> None:
    group = parent.create_group(name)
    for key, values in columns.items():
        array = np.asarray(values)
        if array.dtype == object or array.dtype.kind == "U":
            group.create_dataset(
                key,
                data=np.asarray([str(v) for v in array], dtype=object),
                dtype=h5py.string_dtype(),
            )
        else:
            group.create_dataset(key, data=array)


def _write(
    run: RunDir,
    records: list[dict[str, Any]],
    groups: dict[str, np.ndarray],
    members: dict[str, np.ndarray],
    curves: dict[tuple[int, float], float],
    ledger: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out = run.collect_dir
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{CANDIDATES_H5}.writing"
    with h5py.File(tmp, "w") as handle:
        for key in ("schema", "science_sha256", "analysis_sha256", "inputs_sha256"):
            handle.attrs[key] = summary[key]
        handle.attrs["dms"] = np.asarray(summary["dms"], dtype=np.float64)
        units = {
            key: np.asarray([r[key] for r in records], dtype=object)
            if isinstance(records[0][key] if records else "", str)
            else np.asarray([r[key] for r in records])
            for key in (records[0] if records else {})
        }
        _table(handle, "units", units)
        _table(handle, "groups", groups)
        _table(handle, "members", members)
        keys = sorted(curves)
        _table(
            handle,
            "dm_curves",
            {
                "group": np.asarray([k[0] for k in keys], dtype=np.int64),
                "dm": np.asarray([k[1] for k in keys], dtype=np.float64),
                "snr": np.asarray([curves[k] for k in keys], dtype=np.float64),
            },
        )
    tmp.replace(out / CANDIDATES_H5)
    atomic_write_text(
        out / LEDGER,
        "".join(canonical_json(entry) + "\n" for entry in ledger),
    )
    summary["candidates_sha256"] = sha256_file(out / CANDIDATES_H5)
    summary["ledger_sha256"] = sha256_file(out / LEDGER)
    atomic_write_json(out / SUMMARY, summary)


def load_groups(run: RunDir) -> dict[str, np.ndarray]:
    """Read the groups table ``collect`` wrote."""
    with h5py.File(run.collect_dir / CANDIDATES_H5, "r") as handle:
        return {
            key: handle["groups"][key].asstr()[:]
            if h5py.check_string_dtype(handle["groups"][key].dtype)
            else handle["groups"][key][:]
            for key in handle["groups"]
        }
