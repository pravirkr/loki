"""``report``: one static HTML page of a run, made from one JSON document.

``report/report.json`` holds everything the page shows: the run and its
hashes, the units and their failures, the cost, the candidates, the catalogue
recall and the ledger. ``report/index.html`` is rendered from that document
alone, with no other input and no external files, so a page can always be
made again from its JSON and every page records the hashes it came from.
"""

from __future__ import annotations

import html
import json
import math
import time
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from loki_survey.collect import CANDIDATES_H5, LEDGER
from loki_survey.collect import SUMMARY as COLLECT_SUMMARY
from loki_survey.evaluate import EVAL_SUMMARY, MATCHES
from loki_survey.rundir import RunDir, atomic_write_json, atomic_write_text, read_json

if TYPE_CHECKING:
    from loki_survey.config import Resolved
    from loki_survey.plan import Unit

SCHEMA = "loki-survey report 1"
REPORT_JSON = "report.json"
REPORT_HTML = "index.html"
# The page lists this many candidates and ledger rows; the files hold them all.
TOP_CANDIDATES = 100
TOP_LEDGER = 100
SPEED_OF_LIGHT = 299792458.0
SECONDS_PER_HOUR = 3600.0


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _units_section(run: RunDir, units: list[Unit]) -> dict[str, Any]:
    from loki_survey.status import StatusDB  # noqa: PLC0415

    db = StatusDB(run)
    try:
        counts = dict(db.counts())
        failures = [
            {
                "unit": uid,
                "class": failure,
                "attempts": attempts,
                "detail": (detail or "").splitlines()[0] if detail else "",
            }
            for uid, failure, attempts, detail in db.failures()
        ]
        attempts = db.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        excluded = db.excluded_hosts()
    finally:
        db.close()
    return {
        "planned_or_split": len(units),
        "states": counts,
        "attempts": int(attempts),
        "failures": failures,
        "excluded_hosts": excluded,
    }


def _cost_section(run: RunDir, resolved: Resolved, units: list[Unit]) -> dict[str, Any]:
    threads = resolved.site.unit_threads
    by_kind: dict[str, dict[str, list[float]]] = {}
    total_s = 0.0
    for unit in units:
        path = run.unit_summary(unit.id)
        if not path.is_file():
            continue
        summary = read_json(path)
        total_s += summary["total_s"]
        kind = by_kind.setdefault(
            f"{unit.box} / {unit.region}",
            {"total_s": [], "peak_rss_gb": []},
        )
        kind["total_s"].append(summary["total_s"])
        kind["peak_rss_gb"].append(summary["peak_rss_gb"])
    return {
        "threads_per_unit": threads,
        "core_hours": total_s * threads / SECONDS_PER_HOUR,
        "by_kind": {
            name: {
                "units": len(values["total_s"]),
                "wall_s": _percentiles(values["total_s"]),
                "peak_rss_gb": _percentiles(values["peak_rss_gb"]),
            }
            for name, values in sorted(by_kind.items())
        },
    }


def _read_table(handle: h5py.File, name: str) -> dict[str, np.ndarray]:
    return {
        key: handle[name][key].asstr()[:]
        if h5py.check_string_dtype(handle[name][key].dtype)
        else handle[name][key][:]
        for key in handle[name]
    }


def _candidates_section(run: RunDir) -> dict[str, Any] | None:
    path = run.collect_dir / CANDIDATES_H5
    if not path.is_file():
        return None
    collected = read_json(run.collect_dir / COLLECT_SUMMARY)
    with h5py.File(path, "r") as handle:
        groups = _read_table(handle, "groups")
        curves = _read_table(handle, "dm_curves")
    labels: dict[int, dict[str, Any]] = {}
    matches = run.evaluate_dir / MATCHES
    if matches.is_file():
        for line in matches.read_text().splitlines():
            entry = json.loads(line)
            labels[entry["group"]] = entry
    selected = np.nonzero(groups["selected"])[0][:TOP_CANDIDATES]
    rows = []
    for g in selected:
        f0, f1 = float(groups["f0"][g]), float(groups["f1"][g])
        mask = curves["group"] == g
        curve = sorted(
            zip(curves["dm"][mask].tolist(), curves["snr"][mask].tolist(), strict=True),
        )
        label = labels.get(int(g), {})
        catalogue = label.get("catalogue") or []
        rows.append(
            {
                "group": int(g),
                "f0": f0,
                "period_ms": 1e3 / f0,
                "f1": f1,
                "accel": -f1 * SPEED_OF_LIGHT / f0,
                "snr": float(groups["snr"][g]),
                "dm": float(groups["dm"][g]),
                "dm_call": str(groups["dm_call"][g]),
                "n_members": int(groups["n_members"][g]),
                "n_units": int(groups["n_units"][g]),
                "boxes": str(groups["boxes"][g]),
                "harmonic_of": int(groups["harmonic_of"][g]),
                "harmonic": f"{groups['harmonic_p'][g]}/{groups['harmonic_q'][g]}",
                "catalogue": catalogue[0] if catalogue else None,
                "line": label.get("line"),
                "dm_curve": curve,
            },
        )
    return {
        "summary": {
            key: collected[key]
            for key in (
                "units",
                "units_collected",
                "members",
                "groups",
                "selected",
                "removed",
                "dm_calls",
                "harmonics",
                "dms",
            )
        },
        "missing_units": collected["units_missing"],
        "rows": rows,
    }


def _ledger_section(run: RunDir) -> dict[str, Any] | None:
    path = run.collect_dir / LEDGER
    if not path.is_file():
        return None
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["rule"]] = counts.get(entry["rule"], 0) + 1
    return {"total": len(entries), "by_rule": counts, "rows": entries[:TOP_LEDGER]}


def _folds_section(run: RunDir) -> dict[str, Any] | None:
    summary_path = run.root / "folds" / "summary.json"
    if not summary_path.is_file():
        return None
    summary = read_json(summary_path)
    rows = []
    with h5py.File(run.root / "folds" / "folds.h5", "r") as handle:
        for entry in summary["groups"]:
            group = handle[f"group_{entry['group']:06d}"]
            profile = group["profile"][:]
            norm = np.divide(
                profile[0],
                np.sqrt(profile[1]),
                out=np.zeros_like(profile[0]),
                where=profile[1] > 0,
            )
            rows.append({**entry, "profile": (norm - norm.mean()).tolist()})
    return {"control_factor": summary["fold"]["control_factor"], "rows": rows}


def build_report(run: RunDir, resolved: Resolved, units: list[Unit]) -> dict[str, Any]:
    """Gather everything the page shows into one document."""
    resolved_doc = read_json(run.resolved_json)
    evaluated = run.evaluate_dir / EVAL_SUMMARY
    collected = run.collect_dir / COLLECT_SUMMARY
    return {
        "schema": SCHEMA,
        "time": time.time(),
        "run": {
            "name": resolved.survey.name,
            "root": str(run.root),
            "science_sha256": resolved.science_sha256,
            "analysis_sha256": resolved.analysis_sha256,
            "versions": resolved_doc.get("versions", {}),
            "executor": resolved.site.executor,
            "backend": resolved.site.backend,
            "collect_candidates_sha256": read_json(collected)["candidates_sha256"]
            if collected.is_file()
            else None,
        },
        "units": _units_section(run, units),
        "cost": _cost_section(run, resolved, units),
        "candidates": _candidates_section(run),
        "catalogue": read_json(evaluated).get("catalogue")
        if evaluated.is_file()
        else None,
        "lines": read_json(evaluated).get("lines") if evaluated.is_file() else None,
        "folds": _folds_section(run),
        "ledger": _ledger_section(run),
    }


# -- rendering ---------------------------------------------------------------

CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7;
  --accent: #0b6bcb; --warn: #b3261e; --ok: #1b7f3b; --row: #f5f5f7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161617; --fg: #f5f5f7; --muted: #a1a1a6; --line: #3a3a3c;
    --accent: #5aa9ff; --warn: #ff6b60; --ok: #4cd07d; --row: #1f1f21; } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 1200px; padding: 24px 16px 64px;
  background: var(--bg); color: var(--fg);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; border-bottom: 1px solid var(--line);
  padding-bottom: 4px; }
.meta, .note { color: var(--muted); }
.meta code { font-size: 12px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px;
  min-width: 140px; }
.card b { display: block; font-size: 20px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
tbody tr:nth-child(even) { background: var(--row); }
td.num { text-align: right; }
.warn { color: var(--warn); }
.ok { color: var(--ok); }
svg.curve { vertical-align: middle; }
svg.curve polyline { fill: none; stroke: var(--accent); stroke-width: 1.5; }
svg.curve circle { fill: var(--accent); }
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "&ndash;"
    return f"{value:.{digits}g}" if abs(value) >= 1e4 else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], numeric: set[int]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _profile_svg(profile: list[float]) -> str:
    """Return a small pulse-profile line over one turn."""
    if len(profile) < 2 or max(profile) == min(profile):
        return ""
    width, height, pad = 120.0, 28.0, 3.0
    lo, hi = min(profile), max(profile)
    n = len(profile)
    points = " ".join(
        f"{pad + i / (n - 1) * (width - 2 * pad):.1f},"
        f"{height - pad - (v - lo) / (hi - lo) * (height - 2 * pad):.1f}"
        for i, v in enumerate(profile)
    )
    return (
        f'<svg class="curve" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="folded profile"><title>folded profile</title>'
        f'<polyline points="{points}"/></svg>'
    )


def _curve_svg(curve: list[list[float]] | list[tuple[float, float]]) -> str:
    """Return a small S/N against DM line, or nothing for a single DM."""
    if len(curve) < 2:
        return ""
    width, height, pad = 120.0, 28.0, 3.0
    dms = [c[0] for c in curve]
    snrs = [c[1] for c in curve]
    d0, d1 = min(dms), max(dms)
    s1 = max(snrs)
    points = [
        (
            pad + (d - d0) / (d1 - d0) * (width - 2 * pad),
            height - pad - (s / s1) * (height - 2 * pad),
        )
        for d, s in curve
    ]
    peak = points[int(np.argmax(snrs))]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    title = f"S/N against DM, {d0:g} to {d1:g}"
    return (
        f'<svg class="curve" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{title}">'
        f'<title>{title}</title><polyline points="{line}"/>'
        f'<circle cx="{peak[0]:.1f}" cy="{peak[1]:.1f}" r="2.5"/></svg>'
    )


def render_html(doc: dict[str, Any]) -> str:
    """Render the report document as one self-contained HTML page."""
    run, units, cost = doc["run"], doc["units"], doc["cost"]
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(run['name'])} report</title><style>{CSS}</style></head><body>",
        f"<h1>{_e(run['name'])}</h1>",
        '<p class="meta">'
        f"Made {_e(time.strftime('%Y-%m-%d %H:%M', time.localtime(doc['time'])))} "
        f"from <code>{_e(run['root'])}</code>. Science "
        f"<code>{_e(run['science_sha256'][:12])}</code>, analysis "
        f"<code>{_e(run['analysis_sha256'][:12])}</code>, "
        + ", ".join(f"{_e(k)} {_e(v)[:12]}" for k, v in run["versions"].items())
        + ".</p>",
    ]
    states = units["states"]
    done = states.get("done", 0)
    total = sum(v for k, v in states.items() if k != "split")
    failed = states.get("failed", 0)
    cands = doc["candidates"]
    parts.append('<div class="cards">')
    parts.append(f'<div class="card">Units done<b>{done} of {total}</b></div>')
    cls = "warn" if failed else "ok"
    parts.append(f'<div class="card">Failed<b class="{cls}">{failed}</b></div>')
    parts.append(
        f'<div class="card">Core-hours<b>{_num(cost["core_hours"], 2)}</b></div>',
    )
    if cands:
        s = cands["summary"]
        parts.append(f'<div class="card">Candidates<b>{s["selected"]}</b></div>')
        removed = sum(s["removed"].values())
        parts.append(f'<div class="card">Removed (ledger)<b>{removed}</b></div>')
    parts.append("</div>")

    parts.append("<h2>Units</h2>")
    parts.append(
        '<p class="note">'
        + ", ".join(f"{_e(k)} {v}" for k, v in sorted(states.items()))
        + f"; {units['attempts']} attempts in all.</p>",
    )
    if units["excluded_hosts"]:
        hosts = _e(", ".join(units["excluded_hosts"]))
        parts.append(f'<p class="warn">Excluded hosts: {hosts}</p>')
    if units["failures"]:
        parts.append(
            _table(
                ["Unit", "Class", "Attempts", "Detail"],
                [
                    [_e(f["unit"]), _e(f["class"]), str(f["attempts"]), _e(f["detail"])]
                    for f in units["failures"]
                ],
                {2},
            ),
        )

    parts.append("<h2>Cost</h2>")
    parts.append(
        f'<p class="note">{_num(cost["core_hours"], 3)} core-hours at '
        f"{cost['threads_per_unit']} threads per unit.</p>",
    )
    if cost["by_kind"]:
        rows = []
        for name, kind in cost["by_kind"].items():
            wall, mem = kind["wall_s"], kind["peak_rss_gb"]
            rows.append(
                [
                    _e(name),
                    str(kind["units"]),
                    _num(wall["median"], 1),
                    _num(wall["p90"], 1),
                    _num(wall["max"], 1),
                    _num(mem["median"], 2),
                    _num(mem["max"], 2),
                ],
            )
        parts.append(
            _table(
                [
                    "Box / region",
                    "Units",
                    "Wall median (s)",
                    "Wall p90 (s)",
                    "Wall max (s)",
                    "Peak memory median (GB)",
                    "Peak memory max (GB)",
                ],
                rows,
                {1, 2, 3, 4, 5, 6},
            ),
        )

    parts.append("<h2>Candidates</h2>")
    if cands is None:
        parts.append('<p class="note">Not collected yet: run loki-survey collect.</p>')
    else:
        s = cands["summary"]
        calls = ", ".join(f"{_e(k)} {v}" for k, v in s["dm_calls"].items())
        parts.append(
            f'<p class="note">{s["members"]} unit candidates from '
            f"{s['units_collected']} of {s['units']} units, in {s['groups']} groups "
            f"({calls}; {s['harmonics']} labelled harmonics). {s['selected']} "
            f"selected; showing the {min(len(cands['rows']), TOP_CANDIDATES)} "
            "strongest.</p>",
        )
        if cands["missing_units"]:
            parts.append(
                f'<p class="warn">{len(cands["missing_units"])} units had not '
                "finished when collected; their candidates are missing.</p>",
            )
        rows = []
        for r in cands["rows"]:
            cat = r["catalogue"]
            match = (
                f"{_e(cat['name'])} ({cat['p']}/{cat['q']})" if cat is not None else ""
            )
            harmonic = (
                f"{_e(r['harmonic'])} of #{r['harmonic_of']}"
                if r["harmonic_of"] >= 0
                else ""
            )
            line = f"{_num(r['line']['f'], 4)} Hz" if r["line"] else ""
            rows.append(
                [
                    str(r["group"]),
                    _num(r["f0"], 6),
                    _num(r["period_ms"], 6),
                    _num(r["accel"], 2),
                    _num(r["snr"], 2),
                    _num(r["dm"], 2),
                    _e(r["dm_call"]),
                    _curve_svg(r["dm_curve"]),
                    str(r["n_members"]),
                    _e(r["boxes"]),
                    harmonic,
                    match,
                    line,
                ],
            )
        parts.append(
            _table(
                [
                    "#",
                    "f (Hz)",
                    "P (ms)",
                    "Accel (m/s²)",
                    "S/N",
                    "DM",
                    "DM call",
                    "S/N vs DM",
                    "Members",
                    "Boxes",
                    "Harmonic",
                    "Catalogue",
                    "Line",
                ],
                rows,
                {0, 1, 2, 3, 4, 5, 8},
            ),
        )

    catalogue = doc["catalogue"]
    if catalogue:
        parts.append("<h2>Catalogue</h2>")
        found = sum(p["found"] for p in catalogue["pulsars"])
        in_band = sum(p["in_band"] for p in catalogue["pulsars"])
        parts.append(
            f'<p class="note">{found} of {in_band} catalogued pulsars in the searched '
            f"bands found at their own frequency; catalogue "
            f"<code>{_e(catalogue['path'])}</code> "
            f"(<code>{_e(catalogue['sha256'][:12])}</code>).</p>",
        )
        rows = []
        for p in catalogue["pulsars"]:
            best = p["best"]
            if best is None:
                state = '<span class="warn">not found</span>' if p["in_band"] else ""
            elif not best["selected"]:
                state = f'<span class="warn">removed by {_e(best["removed_by"])}</span>'
            else:
                state = '<span class="ok">selected</span>'
            rows.append(
                [
                    _e(p["name"]),
                    _num(p["f0"], 6),
                    _num(p["dm"], 2) if p["dm"] is not None else "&ndash;",
                    "yes" if p["in_band"] else "no",
                    "yes" if p["found"] else "no",
                    str(p["harmonics_found"]),
                    f"#{best['group']} ({best['p']}/{best['q']})" if best else "",
                    _num(best["snr"], 2) if best else "",
                    state,
                ],
            )
        parts.append(
            _table(
                [
                    "Pulsar",
                    "f (Hz)",
                    "DM",
                    "In band",
                    "Found",
                    "Harmonic matches",
                    "Best group",
                    "S/N",
                    "State",
                ],
                rows,
                {1, 2, 5, 7},
            ),
        )
    lines = doc["lines"]
    if lines:
        parts.append(
            f'<p class="note">{lines["n_lines"]} spectral lines in '
            f"<code>{_e(lines['series'])}</code> (DM {_num(lines['dm'], 2)}, "
            f"false-alarm rate {lines['fap']:g}); {lines['groups_on_lines']} groups "
            "lie on one.</p>",
        )

    folds = doc.get("folds")
    if folds:
        parts.append("<h2>Folds</h2>")
        parts.append(
            '<p class="note">The strongest candidates folded on their own series with '
            "loki's model of their leaf. The control is a fold at "
            f"{folds['control_factor']:g} f, where a pulsar smears; a steady pulsar "
            "gives about S/N / &radic;2 in each half.</p>",
        )
        parts.append(
            _table(
                [
                    "#",
                    "f (Hz)",
                    "Search S/N",
                    "Fold S/N",
                    "Control S/N",
                    "Halves",
                    "Model",
                    "Profile",
                    "Fold S/N vs DM",
                ],
                [
                    [
                        str(r["group"]),
                        _num(r["f0"], 6),
                        _num(r["search_snr"], 2),
                        _num(r["snr"], 2),
                        _num(r["snr_control"], 2),
                        " / ".join(_num(h, 2) for h in r["snr_halves"]),
                        _e(r["model"]),
                        _profile_svg(r["profile"]),
                        _curve_svg(r["dm_curve"]),
                    ]
                    for r in folds["rows"]
                ],
                {0, 1, 2, 3, 4},
            ),
        )

    parts.append("<h2>Ledger</h2>")
    ledger = doc["ledger"]
    if ledger is None:
        parts.append('<p class="note">Not collected yet.</p>')
    elif ledger["total"] == 0:
        parts.append('<p class="note">No candidate was removed.</p>')
    else:
        parts.append(
            '<p class="note">'
            + ", ".join(f"{_e(k)} {v}" for k, v in ledger["by_rule"].items())
            + f"; the first {len(ledger['rows'])} of {ledger['total']} below, all in "
            "collect/ledger.jsonl.</p>",
        )
        parts.append(
            _table(
                ["Group", "Rule", "f (Hz)", "S/N", "DM", "Reason"],
                [
                    [
                        str(r["group"]),
                        _e(r["rule"]),
                        _num(r["f0"], 6),
                        _num(r["snr"], 2),
                        _num(r["dm"], 2),
                        _e(r["reason"]),
                    ]
                    for r in ledger["rows"]
                ],
                {0, 2, 3, 4},
            ),
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def write_report(run: RunDir, resolved: Resolved, units: list[Unit]) -> dict[str, Any]:
    """Write ``report/report.json`` and the page made from it."""
    doc = build_report(run, resolved, units)
    run.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run.report_dir / REPORT_JSON, doc)
    atomic_write_text(run.report_dir / REPORT_HTML, render_html(doc))
    return doc
