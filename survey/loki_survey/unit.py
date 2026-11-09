"""``run-unit``: search one unit and publish its output atomically.

The unit works in its own temporary directory. loki writes its full result
file there; the unit then copies loki's common tables (candidates, clusters,
search and observation metadata) into a compact output with the run's
provenance, keeps the full file only for a sampled fraction of units, and
renames the output into ``out/``. The summary JSON is written last, so its
presence means the unit is done.
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import loki_survey
from loki_survey.config import Resolved, load_resolved
from loki_survey.plan import (
    Unit,
    candidate_options,
    load_units,
    plan_cache_path,
    search_config,
)
from loki_survey.prep import load_arrays
from loki_survey.rundir import (
    RunDir,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)

# loki's common candidate tables and metadata; anything else stays in the
# full result file.
COMMON_GROUPS = ("observation", "processing", "search", "candidates", "clusters")
RESULT_SUFFIX = {"ffa": "_ffa_results.h5", "ep": "_ep_results.h5"}


def keeps_raw(unit_id: str, raw_fraction: float) -> bool:
    """Decide, reproducibly, whether a unit keeps loki's full result file."""
    if raw_fraction <= 0.0:
        return False
    draw = int(sha256_text(unit_id)[:8], 16) / float(1 << 32)
    return draw < raw_fraction


def versions() -> dict[str, str]:
    """Describe the code behind an output: versions and the built library."""
    import loki  # noqa: PLC0415
    from loki import libloki  # noqa: PLC0415

    return {
        "loki_survey": loki_survey.__version__,
        "loki": getattr(loki, "__version__", "unknown"),
        "libloki_sha256": sha256_file(Path(libloki.__file__)),
        "python": platform.python_version(),
    }


def observation(manifest: dict[str, Any]) -> Any:  # noqa: ANN401
    from loki import libloki  # noqa: PLC0415

    obs = libloki.io.ObservationMetadata()
    fields = manifest["observation"]
    for key in ("dm", "start_mjd", "barycentric"):
        if fields.get(key) is not None:
            setattr(obs, key, fields[key])
    obs.source = fields.get("source") or ""
    obs.input_path = manifest["path"]
    return obs


def search(run: RunDir, resolved: Resolved, unit: Unit, workdir: Path) -> Path:
    """Run loki on the unit's series; return loki's result file."""
    from loki import libloki  # noqa: PLC0415

    site = resolved.site
    ts_e, ts_v, manifest = load_arrays(run, unit.series, unit.downsample)
    ts_e = np.ascontiguousarray(ts_e[: unit.nsamps])
    ts_v = np.ascontiguousarray(ts_v[: unit.nsamps])
    cfg = search_config(resolved, unit)
    candidates = candidate_options()
    obs = observation(manifest)
    if unit.kind == "ffa":
        sweep = libloki.ffa.FFAFreqSweep(
            cfg,
            show_progress=False,
            backend=site.backend,
            device=site.device,
            candidates=candidates,
        )
        sweep.execute(
            ts_e,
            ts_v,
            str(workdir),
            unit.id,
            config_toml=canonical_json(asdict(unit)),
            observation=obs,
        )
    else:
        if unit.plan_key is None:
            msg = f"{unit.id} has no EP plan"
            raise ValueError(msg)
        sweep = libloki.prune.EPFreqSweep(
            cfg,
            show_progress=False,
            min_pd=resolved.survey.min_pd,
            poly_basis="taylor",
            plan_cache_file=str(plan_cache_path(run, unit.plan_key)),
            ref_segs=unit.ref_segs,
            backend=site.backend,
            device=site.device,
            candidates=candidates,
        )
        sweep.execute(ts_e, ts_v, str(workdir), unit.id, observation=obs)
    result = workdir / f"{unit.id}{RESULT_SUFFIX[unit.kind]}"
    if not result.is_file():
        msg = f"loki wrote no result file {result.name}"
        raise RuntimeError(msg)
    return result


def candidate_leaves(src: h5py.File) -> np.ndarray | None:
    """Return each EP candidate's final leaf, (value, error) rows, or None.

    ``fold`` evaluates loki's phase model on the leaf, which needs the errors
    (a five-parameter search decides per leaf, from them, whether it is an
    orbit) that the candidate table does not keep. An FFA result has none.
    """
    if "candidates" not in src or "search/sources/run_path" not in src:
        return None
    paths = src["search/sources/run_path"].asstr()[:]
    regions = src["candidates/source_region"][:]
    rows = src["candidates/source_index"][:]
    nparams = len(src.attrs["param_names"])
    leaves = np.zeros((len(rows), nparams + 2, 2), dtype=np.float64)
    for i, (region, row) in enumerate(zip(regions, rows, strict=True)):
        leaves[i] = src[paths[int(region)]]["param_sets"][int(row)]
    return leaves


def compact(result: Path, out: Path, provenance: dict[str, Any]) -> dict[str, int]:
    """Copy loki's common tables into ``out`` and add the run's provenance."""
    counts = {}
    with h5py.File(result, "r") as src, h5py.File(out, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name in COMMON_GROUPS:
            if name in src:
                src.copy(src[name], dst, name=name)
        for name in ("candidates", "clusters"):
            counts[name] = _rows(src, name)
        meta = dst.create_group("survey")
        for key, value in provenance.items():
            meta.attrs[key] = (
                value if isinstance(value, str | int | float) else canonical_json(value)
            )
        leaves = candidate_leaves(src)
        if leaves is not None:
            meta.create_dataset("leaves", data=leaves)
    return counts


def _rows(handle: h5py.File, name: str) -> int:
    """Rows in a common table (a group of equal-length datasets), or 0."""
    if name not in handle:
        return 0
    node = handle[name]
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.shape else 0
    lengths = [
        int(d.shape[0])
        for d in node.values()
        if isinstance(d, h5py.Dataset) and d.shape
    ]
    return max(lengths, default=0)


def peak_rss_gb() -> float:
    """Return peak resident memory in GB (ru_maxrss: KiB on Linux, B on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if platform.system() == "Darwin" else 1024.0
    return peak * scale / 1e9


def run_unit(run: RunDir, unit_id: str, attempt: int) -> dict[str, Any]:
    """Search one unit; a unit already done is left as it is."""
    resolved = load_resolved(run)
    units = {u.id: u for u in load_units(run)}
    if unit_id not in units:
        msg = f"unknown unit {unit_id}"
        raise KeyError(msg)
    unit = units[unit_id]
    if run.unit_summary(unit_id).is_file():
        print(f"{unit_id} is already done")
        return read_json(run.unit_summary(unit_id))
    workdir = run.unit_tmp(unit_id, attempt)
    if workdir.exists():
        # Left by an earlier run of this same job, which the scheduler
        # stopped and started again (a preemption on a rerunnable queue).
        target = run.quarantine_dir / f"{workdir.name}.{int(time.time())}"
        run.quarantine_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(workdir), str(target))
    workdir.mkdir(parents=True)
    started = time.time()
    result = search(run, resolved, unit, workdir)
    searched = time.time()
    _, _, manifest = load_arrays(run, unit.series, unit.downsample)
    provenance = {
        "unit": asdict(unit),
        "science_sha256": resolved.science_sha256,
        "inputs_sha256": manifest["inputs_sha256"],
        "arrays_sha256": manifest["arrays"][str(unit.downsample)],
        "versions": versions(),
        "host": platform.node(),
        "attempt": attempt,
        "backend": resolved.site.backend,
        "threads": resolved.site.unit_threads,
        "loki_memory_gb": resolved.site.loki_memory_gb,
    }
    staged = workdir / f"{unit_id}.h5"
    counts = compact(result, staged, provenance)
    raw_kept = keeps_raw(unit_id, resolved.survey.raw_fraction)
    if raw_kept:
        run.raw_dir.mkdir(parents=True, exist_ok=True)
        result.replace(run.raw_dir / result.name)
    staged.replace(run.unit_output(unit_id))
    summary = {
        "unit": unit_id,
        "attempt": attempt,
        "host": platform.node(),
        "search_s": searched - started,
        "total_s": time.time() - started,
        "peak_rss_gb": peak_rss_gb(),
        "n_candidates": counts.get("candidates", 0),
        "n_clusters": counts.get("clusters", 0),
        "raw_kept": raw_kept,
        "output_sha256": sha256_file(run.unit_output(unit_id)),
        "science_sha256": resolved.science_sha256,
    }
    atomic_write_json(run.unit_summary(unit_id), summary)
    shutil.rmtree(workdir, ignore_errors=True)
    return summary


def unit_env(threads: int) -> dict[str, str]:
    """Environment for a unit process: thread counts capped to its share."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(threads)
    return env
