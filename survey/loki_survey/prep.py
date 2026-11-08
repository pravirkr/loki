"""The ``prep`` stage: read each series, normalise it, mask it, downsample it.

Each series is read with loki's own reader (PRESTO ``.dat``/``.inf`` or SIGPROC
``.tim``) and normalised with loki's robust preprocessing, which gives the
weighted pair (``ts_e``, ``ts_v``) every search consumes. Masked samples get
zero weight. A downsampled copy sums adjacent samples of both arrays, which
keeps them the sufficient statistics of the same matched filter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from loki_survey.rundir import RunDir, atomic_write_json, read_json, sha256_file

if TYPE_CHECKING:
    from loki_survey.config import Resolved, Survey

MANIFEST = "manifest.json"
# PRESTO .inf lines "On/Off bin pair #  1      =  0       , 35326431"
_ONOFF = re.compile(r"On/Off bin pair\s*#\s*\d+\s*=\s*(\d+)\s*,\s*(\d+)")


def discover(run: RunDir, survey: Survey) -> list[Path]:
    """Find the input series from the survey's globs, relative to the run."""
    found: dict[Path, None] = {}
    for pattern in survey.series:
        base = Path(pattern)
        if base.is_absolute():
            matches = sorted(Path(base.anchor).glob(str(base.relative_to(base.anchor))))
        else:
            matches = sorted(run.root.glob(pattern))
        for path in matches:
            if path.suffix in {".dat", ".tim"}:
                found[path.resolve()] = None
    series = list(found)
    if not series:
        msg = f"no .dat or .tim files match {list(survey.series)}"
        raise FileNotFoundError(msg)
    stems = [p.stem for p in series]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        msg = f"series names must be unique; repeated: {duplicates}"
        raise ValueError(msg)
    return series


def series_id(path: Path) -> str:
    return path.stem


def inf_off_ranges(path: Path, nsamps: int) -> list[tuple[int, int]]:
    """Sample ranges a PRESTO ``.inf`` marks as off (data breaks or padding)."""
    inf = path.with_suffix(".inf")
    if path.suffix != ".dat" or not inf.is_file():
        return []
    pairs = [(int(a), int(b)) for a, b in _ONOFF.findall(inf.read_text())]
    if not pairs:
        return []
    off = []
    cursor = 0
    for on, last in sorted(pairs):
        if on > cursor:
            off.append((cursor, on))
        cursor = max(cursor, last + 1)
    if cursor < nsamps:
        off.append((cursor, nsamps))
    return off


def truncated_length(nsamps: int, truncate: str) -> int:
    if truncate == "pow2":
        return 1 << (nsamps.bit_length() - 1)
    return nsamps


def downsample(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``factor`` adjacent samples of both arrays."""
    if factor == 1:
        return ts_e, ts_v
    n = (ts_e.size // factor) * factor
    e = ts_e[:n].reshape(-1, factor).sum(axis=1, dtype=np.float64)
    v = ts_v[:n].reshape(-1, factor).sum(axis=1, dtype=np.float64)
    return e.astype(np.float32), v.astype(np.float32)


def preprocess_options(survey: Survey) -> Any:  # noqa: ANN401
    """Loki's PreprocessOptions, with the survey's [preprocess] fields set."""
    from loki import libloki  # noqa: PLC0415

    options = libloki.io.PreprocessOptions()
    methods = {
        "robust": libloki.io.PreprocessMethod.Robust,
        "zscore": libloki.io.PreprocessMethod.ZScore,
    }
    for key, value in survey.preprocess.items():
        if not hasattr(options, key) or key.startswith("_"):
            msg = f"[preprocess]: loki has no preprocessing option '{key}'"
            raise ValueError(msg)
        if key == "method":
            if value not in methods:
                msg = f"[preprocess].method: expected one of {sorted(methods)}"
                raise ValueError(msg)
            value = methods[value]  # noqa: PLW2901
        setattr(options, key, value)
    options.validate()
    return options


def preprocess_record(options: Any) -> dict[str, Any]:  # noqa: ANN401
    """Every scalar preprocessing option as loki will use it."""
    record = {}
    for key in dir(options):
        if key.startswith("_") or key == "validate":
            continue
        value = getattr(options, key)
        if isinstance(value, bool | int | float | str):
            record[key] = value
        elif hasattr(value, "name"):
            record[key] = value.name
    return record


def _observation(ts: Any) -> dict[str, Any]:  # noqa: ANN401
    obs = ts.observation
    return {
        "dm": obs.dm,
        "start_mjd": obs.start_mjd,
        "barycentric": obs.barycentric,
        "source": obs.source,
    }


def prep_series(
    run: RunDir,
    resolved: Resolved,
    path: Path,
    nthreads: int,
) -> dict[str, Any]:
    """Prepare one series and write its manifest last, so a manifest means done."""
    from loki import libloki  # noqa: PLC0415

    survey = resolved.survey
    sid = series_id(path)
    out = run.prep_dir / sid
    out.mkdir(parents=True, exist_ok=True)
    read_options = libloki.io.ReadOptions()
    read_options.preprocess = False
    read_options.nthreads = nthreads
    ts = libloki.io.TimeSeries.read(str(path), read_options)
    raw = np.asarray(ts.ts_e, dtype=np.float32)
    options = preprocess_options(survey)
    ts_e, ts_v, report = libloki.io.preprocess(raw, ts.dt, options, nthreads)
    ts_e = np.array(ts_e, dtype=np.float32)
    ts_v = np.array(ts_v, dtype=np.float32)
    masked = inf_off_ranges(path, raw.size)
    for t0, t1 in survey.masks.get(path.name, []):
        masked.append((int(np.floor(t0 / ts.dt)), int(np.ceil(t1 / ts.dt))))
    n_masked = 0
    for lo, hi in masked:
        lo_c, hi_c = max(lo, 0), min(hi, raw.size)
        if hi_c > lo_c:
            ts_e[lo_c:hi_c] = 0.0
            ts_v[lo_c:hi_c] = 0.0
            n_masked += hi_c - lo_c
    nsamps = truncated_length(raw.size, survey.truncate)
    factors = sorted({r.downsample for r in survey.regions})
    arrays = {}
    for factor in factors:
        e, v = downsample(ts_e[:nsamps], ts_v[:nsamps], factor)
        e_path, v_path = out / f"ds{factor}_e.npy", out / f"ds{factor}_v.npy"
        np.save(e_path, e)
        np.save(v_path, v)
        arrays[str(factor)] = {
            "nsamps": int(e.size),
            "tsamp": ts.dt * factor,
            "ts_e": e_path.name,
            "ts_v": v_path.name,
            "ts_e_sha256": sha256_file(e_path),
            "ts_v_sha256": sha256_file(v_path),
        }
    inputs = {path.name: sha256_file(path)}
    if path.suffix == ".dat" and path.with_suffix(".inf").is_file():
        inf = path.with_suffix(".inf")
        inputs[inf.name] = sha256_file(inf)
    manifest = {
        "series": sid,
        "path": str(path),
        "inputs_sha256": inputs,
        "science_sha256": resolved.science_sha256,
        "tsamp": ts.dt,
        "nsamps_read": int(raw.size),
        "nsamps": int(nsamps),
        "observation": _observation(ts),
        "masked_ranges": [list(r) for r in masked],
        "n_masked_by_survey": int(n_masked),
        "preprocess": preprocess_record(options),
        "preprocess_report": {
            "n_masked": int(report.n_masked),
            "n_clipped": int(report.n_clipped),
            "n_zapped": int(report.n_zapped),
            "longest_masked_run": int(report.longest_masked_run),
            "zero_weight_fraction": float(report.zero_weight_fraction),
            "baseline_window": float(report.baseline_window),
        },
        "arrays": arrays,
    }
    atomic_write_json(out / MANIFEST, manifest)
    return manifest


def load_manifest(run: RunDir, sid: str) -> dict[str, Any]:
    return read_json(run.prep_dir / sid / MANIFEST)


def is_current(run: RunDir, resolved: Resolved, path: Path) -> bool:
    """Tell whether a series' manifest matches this science and input."""
    manifest_path = run.prep_dir / series_id(path) / MANIFEST
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("science_sha256") != resolved.science_sha256:
        return False
    return manifest["inputs_sha256"].get(path.name) == sha256_file(path)


def run_prep(run: RunDir, resolved: Resolved, *, nthreads: int) -> list[dict[str, Any]]:
    """Prepare every series that is missing or stale; return all manifests."""
    manifests = []
    for path in discover(run, resolved.survey):
        if is_current(run, resolved, path):
            manifests.append(load_manifest(run, series_id(path)))
            print(f"prep: {path.name} is current")
            continue
        print(f"prep: {path.name}")
        manifests.append(prep_series(run, resolved, path, nthreads))
    return manifests


def load_arrays(
    run: RunDir,
    sid: str,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = load_manifest(run, sid)
    entry = manifest["arrays"][str(factor)]
    base = run.prep_dir / sid
    ts_e = np.load(base / entry["ts_e"], mmap_mode="r")
    ts_v = np.load(base / entry["ts_v"], mmap_mode="r")
    return ts_e, ts_v, manifest
