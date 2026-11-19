"""``evaluate``: match the collected candidates to a pulsar catalogue and to lines.

A catalogue is a CSV file the user supplies (see ``read_catalogue``); its
sha256 is recorded. A group matches a catalogued pulsar when its frequency is
within the window of (p/q) times the pulsar's, for p, q up to
``harmonic_max``. The window is the larger of the Doppler term (``doppler``,
plus the orbit's v/c when the catalogue gives the orbit) times the frequency,
and ``radius_bins`` resolutions 1/T. With ``dm_tol`` above zero the DMs must
also agree that closely.

Lines are the strong narrow features of the power spectrum of the lowest-DM
series, where an undispersed periodic signal (interference) is strongest. A
group whose frequency lies on a line, within ``line_tol_bins`` resolutions, is
labelled with it. Labels remove nothing; removing is ``collect``'s, with a
ledger.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from loki_survey.collect import CANDIDATES_H5, SUMMARY, load_groups
from loki_survey.prep import load_arrays
from loki_survey.rundir import (
    RunDir,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    read_json,
    sha256_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from loki_survey.config import Evaluate, Resolved

MATCHES = "matches.jsonl"
EVAL_SUMMARY = "summary.json"
SECONDS_PER_DAY = 86400.0
# The noise power of a spectrum is estimated as the median of blocks of this
# many bins: the median of n exponential values has a relative error of about
# 1/sqrt(n), so 1024 bins estimate the local noise power to about 3 percent.
LINE_MEDIAN_BINS = 1024
# |X|^2 / (median / ln 2) is exponential with unit mean for Gaussian noise.
LN2 = math.log(2.0)


def read_catalogue(path: Path) -> list[dict[str, Any]]:
    """Read a pulsar catalogue CSV.

    Columns: ``name`` and one of ``f0`` (Hz) or ``p0`` (s); optional ``dm``
    (pc/cm^3), ``pb`` (orbital period, days) and ``a1`` (projected semi-major
    axis, light seconds), as the ATNF catalogue names them. Lines starting with
    ``#`` are comments; an empty value or ``*`` is unknown.
    """
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    pulsars = []
    for i, raw in enumerate(csv.DictReader(rows), start=2):
        row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
        if not row.get("name"):
            msg = f"{path}:{i}: a pulsar needs a name"
            raise ValueError(msg)

        def number(key: str, row: dict[str, str] = row) -> float | None:
            text = row.get(key, "")
            return None if text in {"", "*"} else float(text)

        f0, p0 = number("f0"), number("p0")
        if f0 is None and p0 is not None:
            f0 = 1.0 / p0
        if f0 is None or f0 <= 0:
            msg = f"{path}:{i}: {row['name']} needs a positive f0 or p0"
            raise ValueError(msg)
        pb, a1 = number("pb"), number("a1")
        orbit_vc = (
            2.0 * math.pi * a1 / (pb * SECONDS_PER_DAY)
            if pb and a1 is not None
            else 0.0
        )
        pulsars.append(
            {"name": row["name"], "f0": f0, "dm": number("dm"), "orbit_vc": orbit_vc},
        )
    return pulsars


def match_catalogue(
    f0: np.ndarray,
    dm: np.ndarray,
    tol_hz: np.ndarray,
    pulsars: list[dict[str, Any]],
    cfg: Evaluate,
) -> list[list[dict[str, Any]]]:
    """Return, per group, every (pulsar, p, q) its frequency matches."""
    matches: list[list[dict[str, Any]]] = [[] for _ in range(f0.size)]
    for pulsar in pulsars:
        for p in range(1, cfg.harmonic_max + 1):
            for q in range(1, cfg.harmonic_max + 1):
                if math.gcd(p, q) != 1:
                    continue
                expected = pulsar["f0"] * p / q
                window = np.maximum(
                    expected * (cfg.doppler + pulsar["orbit_vc"]),
                    tol_hz,
                )
                hit = np.abs(f0 - expected) <= window
                if cfg.dm_tol > 0 and pulsar["dm"] is not None:
                    hit &= ~(np.abs(dm - pulsar["dm"]) > cfg.dm_tol)
                for g in np.nonzero(hit)[0]:
                    matches[g].append(
                        {
                            "name": pulsar["name"],
                            "p": p,
                            "q": q,
                            "df_hz": float(f0[g] - expected),
                            "dm_offset": None
                            if pulsar["dm"] is None or not np.isfinite(dm[g])
                            else float(dm[g] - pulsar["dm"]),
                        },
                    )
    # The simplest ratio first, then the closest.
    for found in matches:
        found.sort(key=lambda m: (m["p"] + m["q"], abs(m["df_hz"])))
    return matches


def find_lines(ts_e: np.ndarray, ts_v: np.ndarray, tsamp: float, fap: float) -> dict:
    """Find the spectral lines of one series: runs of bins with excess power.

    The series is whitened by its weights (zero where masked), its power
    spectrum divided by the median of each block of ``LINE_MEDIAN_BINS`` bins,
    and every bin above ln(n_bins / fap) kept. Adjacent bins make one line.
    """
    weight = np.asarray(ts_v, dtype=np.float32)
    x = np.zeros(weight.size, dtype=np.float32)
    good = weight > 0
    x[good] = np.asarray(ts_e, dtype=np.float32)[good] / np.sqrt(weight[good])
    power = np.abs(np.fft.rfft(x)) ** 2
    power[0] = 0.0  # the mean
    nblocks = max(power.size // LINE_MEDIAN_BINS, 1)
    usable = (
        nblocks * LINE_MEDIAN_BINS if power.size >= LINE_MEDIAN_BINS else power.size
    )
    blocks = power[:usable].reshape(nblocks, -1)
    noise = np.median(blocks, axis=1) / LN2
    noise_per_bin = np.repeat(noise, blocks.shape[1])
    noise_per_bin = np.concatenate(
        (noise_per_bin, np.full(power.size - usable, noise[-1])),
    )
    norm = np.divide(
        power,
        noise_per_bin,
        out=np.zeros_like(power),
        where=noise_per_bin > 0,
    )
    threshold = math.log(power.size / fap)
    above = np.nonzero(norm > threshold)[0]
    df = 1.0 / (x.size * tsamp)
    lines = []
    if above.size:
        starts = np.concatenate(([0], np.nonzero(np.diff(above) > 1)[0] + 1))
        ends = np.concatenate((starts[1:], [above.size]))
        for s, e in zip(starts, ends, strict=True):
            bins = above[s:e]
            peak = bins[np.argmax(norm[bins])]
            lines.append(
                {
                    "f": float(peak * df),
                    "f_lo": float(bins[0] * df),
                    "f_hi": float(bins[-1] * df),
                    "power": float(norm[peak]),
                },
            )
    return {"df": df, "threshold": threshold, "n_bins": int(power.size), "lines": lines}


def series_lines(run: RunDir, series: str, fap: float) -> dict:
    """Return a series' lines, measured once per prepared array and false-alarm rate."""
    ts_e, ts_v, manifest = load_arrays(run, series, 1)
    key = {"arrays_sha256": manifest["arrays"]["1"], "fap": fap}
    cache = run.evaluate_dir / f"lines_{series}.json"
    if cache.is_file():
        cached = read_json(cache)
        if cached.get("key") == key:
            return cached
    tsamp = float(manifest["arrays"]["1"]["tsamp"])
    found = find_lines(ts_e, ts_v, tsamp, fap)
    result = {"key": key, "series": series, **found}
    run.evaluate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cache, result)
    return result


def match_lines(
    f0: np.ndarray,
    tol_hz: np.ndarray,
    lines: list[dict],
) -> list[dict | None]:
    """Return, per group, the line its frequency lies on, if any."""
    out: list[dict | None] = [None] * f0.size
    if not lines:
        return out
    lo = np.array([line["f_lo"] for line in lines])
    hi = np.array([line["f_hi"] for line in lines])
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    for g in range(f0.size):
        k = int(np.searchsorted(lo, f0[g] + tol_hz[g], side="right")) - 1
        while k >= 0 and lo[k] <= f0[g] + tol_hz[g]:
            if hi[k] >= f0[g] - tol_hz[g]:
                out[g] = lines[int(order[k])]
                break
            k -= 1
    return out


def evaluate(run: RunDir, resolved: Resolved) -> dict[str, Any]:
    """Label every collected group with its catalogue matches and its line."""
    cfg = resolved.analysis.evaluate
    collected = read_json(run.collect_dir / SUMMARY)
    groups = load_groups(run)
    with h5py.File(run.collect_dir / CANDIDATES_H5, "r") as handle:
        units = {
            key: handle["units"][key].asstr()[:]
            if h5py.check_string_dtype(handle["units"][key].dtype)
            else handle["units"][key][:]
            for key in handle["units"]
        }
    f0 = np.asarray(groups["f0"], dtype=np.float64)
    dm = np.asarray(groups["dm"], dtype=np.float64)
    tobs = (
        np.asarray(units["tobs"], dtype=np.float64)[groups["unit"]] if f0.size else f0
    )
    radius = resolved.analysis.collect.radius_bins
    tol_hz = radius / tobs if f0.size else f0
    summary: dict[str, Any] = {
        "analysis_sha256": resolved.analysis_sha256,
        "collect_candidates_sha256": collected["candidates_sha256"],
        "groups": int(f0.size),
    }

    matches: list[list[dict[str, Any]]] = [[] for _ in range(f0.size)]
    if cfg.catalogue:
        path = run.root / cfg.catalogue
        pulsars = read_catalogue(path)
        matches = match_catalogue(f0, dm, tol_hz, pulsars, cfg)
        summary["catalogue"] = {
            "path": cfg.catalogue,
            "sha256": sha256_file(path),
            "pulsars": _recall(pulsars, matches, groups, units),
        }

    series_dm = {}
    for series, unit_dm in zip(units["series"], units["dm"], strict=True):
        series_dm.setdefault(str(series), float(unit_dm))
    line_hits: list[dict | None] = [None] * f0.size
    if series_dm:
        known = {s: d for s, d in series_dm.items() if np.isfinite(d)}
        lowest = min(known, key=known.get) if known else min(series_dm)
        if not (run.prep_dir / lowest).is_dir():
            summary["lines"] = None
            print(f"evaluate: no prepared series {lowest}; lines not measured")
        else:
            line_hits = _lines(run, lowest, series_dm[lowest], f0, tobs, cfg, summary)

    entries = [
        {
            "group": g,
            "f0": float(f0[g]),
            "snr": float(groups["snr"][g]),
            "dm": float(dm[g]),
            "catalogue": matches[g],
            "line": line_hits[g],
        }
        for g in range(f0.size)
        if matches[g] or line_hits[g] is not None
    ]
    run.evaluate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run.evaluate_dir / MATCHES,
        "".join(canonical_json(e) + "\n" for e in entries),
    )
    summary["matched_groups"] = len(entries)
    atomic_write_json(run.evaluate_dir / EVAL_SUMMARY, summary)
    return summary


def _lines(
    run: RunDir,
    series: str,
    series_dm: float,
    f0: np.ndarray,
    tobs: np.ndarray,
    cfg: Evaluate,
    summary: dict[str, Any],
) -> list[dict | None]:
    """Measure a series' lines, label the groups on them, and summarise."""
    found = series_lines(run, series, cfg.line_fap)
    line_tol = cfg.line_tol_bins / tobs if f0.size else f0
    hits = match_lines(f0, line_tol, found["lines"])
    summary["lines"] = {
        "series": series,
        "dm": series_dm,
        "fap": cfg.line_fap,
        "threshold": found["threshold"],
        "n_lines": len(found["lines"]),
        "groups_on_lines": sum(hit is not None for hit in hits),
    }
    return hits


def _recall(
    pulsars: list[dict[str, Any]],
    matches: list[list[dict[str, Any]]],
    groups: dict[str, np.ndarray],
    units: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Return, per catalogued pulsar, whether it was searched and what found it."""
    f_lo = np.asarray(units["f_min"], dtype=np.float64)
    f_hi = np.asarray(units["f_max"], dtype=np.float64)
    out = []
    for pulsar in pulsars:
        hits = [
            (g, m)
            for g, found in enumerate(matches)
            for m in found
            if m["name"] == pulsar["name"]
        ]
        fundamental = [(g, m) for g, m in hits if m["p"] == 1 and m["q"] == 1]
        best = max(hits, key=lambda h: groups["snr"][h[0]], default=None)
        row = {
            "name": pulsar["name"],
            "f0": pulsar["f0"],
            "dm": pulsar["dm"],
            "in_band": bool(np.any((f_lo <= pulsar["f0"]) & (pulsar["f0"] <= f_hi))),
            "found": bool(fundamental),
            "harmonics_found": len(hits) - len(fundamental),
            "best": None,
        }
        if best is not None:
            g, m = best
            row["best"] = {
                "group": int(g),
                "snr": float(groups["snr"][g]),
                "p": m["p"],
                "q": m["q"],
                "selected": bool(groups["selected"][g]),
                "removed_by": str(groups["removed_by"][g]),
            }
        out.append(row)
    return out
