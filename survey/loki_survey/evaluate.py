"""``evaluate``: match the collected candidates to a pulsar catalogue and to lines.

A catalogue is a CSV file the user supplies (see ``read_catalogue``); its
sha256 is recorded. A group matches a catalogued pulsar when its frequency is
within the window of (p/q) times the pulsar's, for p, q up to
``harmonic_max``. The window is the larger of the Doppler term (``doppler``,
plus the orbit's v/c when the catalogue gives the orbit) times the frequency,
and ``radius_bins`` resolutions 1/T. With ``dm_tol`` above zero the DMs must
also agree that closely.

Lines are the strong narrow features of the power spectrum of each
observation's lowest-DM series, where an undispersed periodic signal
(interference) is strongest. The DM trials of one observation share a name
(``observation_of``). A group whose frequency lies, within ``line_tol_bins``
resolutions, on a line of an observation it was found in is labelled with it.
Labels remove nothing; removing is ``collect``'s, with a ledger.
"""

from __future__ import annotations

import csv
import math
import re
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from loki_survey.collect import CANDIDATES_H5, NO_PARENT, SUMMARY, load_groups
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
        member_group = handle["members"]["group"][:]
        member_unit = handle["members"]["unit"][:]
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

    unit_obs = [observation_of(str(s), cfg.observation) for s in units["series"]]
    group_obs: list[set[str]] = [set() for _ in range(f0.size)]
    for g, u in zip(member_group, member_unit, strict=True):
        group_obs[int(g)].add(unit_obs[int(u)])
    line_hits = _lines(run, units, unit_obs, group_obs, f0, tobs, cfg, summary)

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


# PRESTO's prepsubband names the DM trials of one observation <base>_DM<dm>.
DM_TOKEN = re.compile(r"_DM-?[0-9]+(?:\.[0-9]+)?")


def observation_of(series: str, pattern: str = "") -> str:
    """Name the observation a series belongs to.

    By default, the series id without its DM token, so that the DM trials of
    one observation share a name while other observations and beams, with
    other base names, keep theirs. A pattern (``[evaluate] observation``)
    overrides it: its group "obs", else its one group, else the whole match;
    a series it does not match is its own observation.
    """
    if not pattern:
        return DM_TOKEN.sub("", series, count=1)
    found = re.search(pattern, series)
    if found is None:
        return series
    if "obs" in found.groupdict():
        return found.group("obs")
    return found.group(1) if found.re.groups else found.group(0)


def _lines(
    run: RunDir,
    units: dict[str, np.ndarray],
    unit_obs: list[str],
    group_obs: list[set[str]],
    f0: np.ndarray,
    tobs: np.ndarray,
    cfg: Evaluate,
    summary: dict[str, Any],
) -> list[dict | None]:
    """Label each group with a line of an observation its members came from.

    Lines are searched in each observation's lowest-DM series: undispersed
    interference is strongest at the lowest DM, and it belongs to one
    observation, so a line found in one observation says nothing of another.
    """
    hits: list[dict | None] = [None] * f0.size
    summary["lines"] = None
    series_dm: dict[str, dict[str, float]] = {}
    for series, unit_dm, obs in zip(
        units["series"],
        units["dm"],
        unit_obs,
        strict=True,
    ):
        series_dm.setdefault(obs, {}).setdefault(str(series), float(unit_dm))
    line_tol = cfg.line_tol_bins / tobs if f0.size else f0
    per_obs = []
    obs_hits: dict[str, list[dict | None]] = {}
    for obs in sorted(series_dm):
        dms = series_dm[obs]
        known = {s: d for s, d in dms.items() if np.isfinite(d)}
        lowest = min(known, key=known.get) if known else min(dms)
        if not (run.prep_dir / lowest).is_dir():
            print(f"evaluate: no prepared series {lowest}; lines of {obs} not measured")
            continue
        found = series_lines(run, lowest, cfg.line_fap)
        obs_hits[obs] = [
            None if hit is None else {**hit, "observation": obs}
            for hit in match_lines(f0, line_tol, found["lines"])
        ]
        per_obs.append(
            {
                "observation": obs,
                "series": lowest,
                "dm": dms[lowest],
                "threshold": found["threshold"],
                "n_lines": len(found["lines"]),
            },
        )
    if not per_obs:
        return hits
    for g in range(f0.size):
        for obs in sorted(group_obs[g] & obs_hits.keys()):
            if obs_hits[obs][g] is not None:
                hits[g] = obs_hits[obs][g]
                break
    summary["lines"] = {
        "fap": cfg.line_fap,
        "observations": per_obs,
        "n_lines": sum(o["n_lines"] for o in per_obs),
        "groups_on_lines": sum(hit is not None for hit in hits),
    }
    return hits


def _best_match(
    hits: list[tuple[int, dict[str, Any]]],
    fundamental: list[tuple[int, dict[str, Any]]],
    groups: dict[str, np.ndarray],
) -> tuple[int, dict[str, Any]] | None:
    """Return the match that tells most about a catalogued pulsar.

    In order: its strongest kept fundamental group, its strongest removed
    fundamental group, its kept harmonic group, its removed one. A stronger
    harmonic must not hide a kept fundamental, nor a removed fundamental hide
    that the pulsar was removed. Within each, a group ``collect`` calls a
    harmonic of another comes after the groups it calls a harmonic of none:
    its frequency is that group's times p/q, within ``radius_bins``, a far
    closer tie than the catalogue window, so a pulsar it matches says no more
    than its parent does (a comb of an interference line lies near many
    ratios). Then the lowest-order ratio p/q (smallest p * q) tells most, and
    S/N only between equal orders: with p and q up to ``harmonic_max`` there
    are many high-order ratios, and a bright unrelated group often lies near
    one by chance.
    """

    def most_telling(
        cands: list[tuple[int, dict[str, Any]]],
    ) -> tuple[int, dict[str, Any]] | None:
        return min(
            cands,
            key=lambda h: (
                groups["harmonic_of"][h[0]] != NO_PARENT,
                h[1]["p"] * h[1]["q"],
                -groups["snr"][h[0]],
            ),
            default=None,
        )

    harmonic = [h for h in hits if h not in fundamental]
    for cands in (fundamental, harmonic):
        kept = [h for h in cands if groups["selected"][h[0]]]
        best = most_telling(kept) or most_telling(cands)
        if best is not None:
            return best
    return None


def recall_state(pulsar: dict[str, Any]) -> str:
    """Describe in one phrase what a catalogued pulsar's best match says.

    "selected", "found as p/q" (only a harmonic group is kept), "removed by
    RULE" (with " as p/q" for a harmonic), "not found" (in band, no match),
    or "" (out of band and unmatched).
    """
    best = pulsar["best"]
    if best is None:
        return "not found" if pulsar["in_band"] else ""
    ratio = "" if (best["p"], best["q"]) == (1, 1) else f" as {best['p']}/{best['q']}"
    if not best["selected"]:
        return f"removed by {best['removed_by']}{ratio}"
    return f"found{ratio}" if ratio else "selected"


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
        best = _best_match(hits, fundamental, groups)
        row = {
            "name": pulsar["name"],
            "f0": pulsar["f0"],
            "dm": pulsar["dm"],
            "in_band": bool(np.any((f_lo <= pulsar["f0"]) & (pulsar["f0"] <= f_hi))),
            "found": bool(fundamental),
            "fundamental_selected": any(groups["selected"][g] for g, _ in fundamental),
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
