"""``run-unit``: search one unit and publish its output atomically.

The unit works in its own temporary directory. loki writes its full result
file there; the unit then copies loki's common tables (candidates, clusters,
search and observation metadata) into a compact output with the run's
provenance, keeps the full file only for a sampled fraction of units, and
renames the output into ``out/``. The summary JSON is written last, so its
presence means the unit is done.

When loki finishes the search but its candidate step fails (most often for
want of memory), loki keeps the search as a spool. The unit then publishes
the spool again with its whole loki budget, since no search runs any more. A
spool an earlier attempt kept, for the same search, is published by the next
attempt without searching again.
"""

from __future__ import annotations

import os
import platform
import re
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import loki_survey
from loki_survey.config import Resolved, load_resolved, science_dict
from loki_survey.plan import (
    Unit,
    build_ep_plan,
    candidate_options,
    load_units,
    plan_cache_path,
    search_config,
)
from loki_survey.prep import load_arrays, load_manifest
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
# A git query for loki's commit that takes longer than this is abandoned.
GIT_TIMEOUT_S = 10
SPOOL_SUFFIX = {"ffa": "_ffa_spool.h5", "ep": "_ep_spool.h5"}
# Written into the work directory before the search: what the search depends
# on, so a later attempt can tell whether a spool kept there is still its own.
SEARCH_JSON = "search.json"


def keeps_raw(unit_id: str, raw_fraction: float) -> bool:
    """Decide, reproducibly, whether a unit keeps loki's full result file."""
    if raw_fraction <= 0.0:
        return False
    draw = int(sha256_text(unit_id)[:8], 16) / float(1 << 32)
    return draw < raw_fraction


def loki_commit(package_dir: Path, libloki: object) -> str:
    """Return the git commit of the loki that runs, as far as it can be told.

    loki writes its commit only into results of its command line (/run_stats),
    not through the Python bindings. So: a commit the bindings report
    (``libloki.git_commit``), else the git checkout loki's package was imported
    from (an editable install), with "-dirty" for uncommitted changes to
    tracked files, else "unknown". An editable install's library is the one
    built at install time; ``libloki_sha256`` names that.
    """
    commit = getattr(libloki, "git_commit", None)
    if isinstance(commit, str) and commit:
        return commit
    root = package_dir.parent.parent  # <checkout>/src/loki
    if not (root / ".git").exists():
        return "unknown"
    git = ["git", "-C", str(root)]
    try:
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_S,
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_S,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def versions() -> dict[str, str]:
    """Describe the code behind an output: versions and the built library."""
    import loki  # noqa: PLC0415
    from loki import libloki  # noqa: PLC0415

    return {
        "loki_survey": loki_survey.__version__,
        "loki": getattr(loki, "__version__", "unknown"),
        "loki_commit": loki_commit(Path(loki.__file__).resolve().parent, libloki),
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


def search_sha256(resolved: Resolved, unit: Unit, manifest: dict[str, Any]) -> str:
    """Hash what a unit's search depends on, apart from its candidate step."""
    survey = replace(resolved.survey, candidates={}, raw_fraction=0.0)
    return sha256_text(
        canonical_json(
            {
                "science": science_dict(survey),
                "unit": asdict(unit),
                "arrays_sha256": manifest["arrays"][str(unit.downsample)],
            },
        ),
    )


def publish_spool(
    resolved: Resolved,
    unit: Unit,
    spool: Path,
    result: Path,
    obs: Any,  # noqa: ANN401
) -> None:
    """Publish a search loki kept, with the unit's whole loki budget."""
    from loki import libloki  # noqa: PLC0415

    options = candidate_options(resolved, unit.kind)
    options.memory_gb = resolved.site.loki_memory_gb
    sweep = (
        libloki.ffa.FFAFreqSweep if unit.kind == "ffa" else libloki.prune.EPFreqSweep
    )
    try:
        sweep.publish_candidates(
            str(spool),
            str(result),
            candidates=options,
            observation=obs,
        )
    except RuntimeError as exc:
        msg = (
            f"{unit.id}: loki's candidate step failed with the unit's whole "
            f"{options.memory_gb:g} GB: {exc}. The search is kept in {spool}. "
            "Keep fewer trials with [candidates] max_per_cell, in cells of "
            "cell_bins or cell_log_width, in survey.toml."
        )
        raise RuntimeError(msg) from exc


def kept_spool(run: RunDir, unit: Unit, key: str) -> Path | None:
    """Return the newest complete spool an earlier attempt kept for this search."""
    if not run.quarantine_dir.is_dir():
        return None
    # Quarantined work directories: <unit>.<attempt>, or with a time after it.
    name = re.compile(rf"{re.escape(unit.id)}\.\d+(\.\d+)?")
    found = []
    for directory in run.quarantine_dir.iterdir():
        spool = directory / f"{unit.id}{SPOOL_SUFFIX[unit.kind]}"
        marker = directory / SEARCH_JSON
        if not (name.fullmatch(directory.name) and spool.is_file()):
            continue
        if not marker.is_file() or read_json(marker).get("search_sha256") != key:
            continue
        try:
            with h5py.File(spool, "r") as handle:
                complete = int(handle.attrs.get("complete", 0)) == 1
        except OSError:
            complete = False
        if complete:
            found.append(spool)
    return max(found, key=lambda p: p.stat().st_mtime, default=None)


def search(
    run: RunDir,
    resolved: Resolved,
    unit: Unit,
    workdir: Path,
) -> tuple[Path, bool]:
    """Run loki on the unit's series; return loki's result file.

    The flag says whether the candidate step ran again from loki's spool.
    """
    from loki import libloki  # noqa: PLC0415

    site = resolved.site
    ts_e, ts_v, manifest = load_arrays(run, unit.series, unit.downsample)
    ts_e = np.ascontiguousarray(ts_e[: unit.nsamps])
    ts_v = np.ascontiguousarray(ts_v[: unit.nsamps])
    cfg = search_config(resolved, unit)
    candidates = candidate_options(resolved, unit.kind)
    obs = observation(manifest)
    extra: dict[str, Any] = {}
    if unit.kind == "ffa":
        sweep = libloki.ffa.FFAFreqSweep(
            cfg,
            show_progress=False,
            backend=site.backend,
            device=site.device,
            candidates=candidates,
        )
        extra["config_toml"] = canonical_json(asdict(unit))
    else:
        if unit.plan_key is None:
            msg = f"{unit.id} has no EP plan"
            raise ValueError(msg)
        if not plan_cache_path(run, unit.plan_key).is_file():
            # A CUDA unit (or a unit split after plan) builds its own plan.
            build_ep_plan(run, resolved, unit)
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
    result = workdir / f"{unit.id}{RESULT_SUFFIX[unit.kind]}"
    republished = False
    try:
        sweep.execute(ts_e, ts_v, str(workdir), unit.id, observation=obs, **extra)
    except RuntimeError as exc:
        spool = workdir / f"{unit.id}{SPOOL_SUFFIX[unit.kind]}"
        if not spool.is_file():
            raise
        # The search is done, so its memory is free for the candidate step.
        del sweep
        print(
            f"{unit.id}: loki's candidate step failed after the search ({exc}); "
            f"publishing the search again with {site.loki_memory_gb:g} GB",
        )
        publish_spool(resolved, unit, spool, result, obs)
        republished = True
    if not result.is_file():
        msg = f"loki wrote no result file {result.name}"
        raise RuntimeError(msg)
    return result, republished


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
    spool_rows = src["candidates/source_index"][:]
    nparams = len(src.attrs["param_names"])
    leaves = np.zeros((len(spool_rows), nparams + 2, 2), dtype=np.float64)
    for i, (region, spool_row) in enumerate(zip(regions, spool_rows, strict=True)):
        run = src[paths[int(region)]]
        leaves[i] = run["param_sets"][run_row(run, int(spool_row))]
    return leaves


def run_row(run: h5py.Group, spool_row: int) -> int:
    """Return the row of an EP run group that holds this spool row's leaf.

    A candidate's ``source_index`` is its leaf's row in loki's search spool. A
    run group loki reduced (``max_per_cell`` > 0) keeps only its retained
    leaves, and ``leaf_index`` gives each kept row's spool row.
    """
    if "leaf_index" not in run:
        return spool_row
    rows = np.nonzero(run["leaf_index"][:] == spool_row)[0]
    if rows.size != 1:
        msg = f"spool row {spool_row} is not kept once in {run.name}"
        raise KeyError(msg)
    return int(rows[0])


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
        counts.update(_copy_level_stats(src, dst))
        meta = dst.create_group("survey")
        for key, value in provenance.items():
            meta.attrs[key] = (
                value if isinstance(value, str | int | float) else canonical_json(value)
            )
        leaves = candidate_leaves(src)
        if leaves is not None:
            meta.create_dataset("leaves", data=leaves)
    return counts


def _copy_level_stats(src: h5py.File, dst: h5py.File) -> dict[str, int]:
    """Keep each EP run's per-level table, and count its overloads.

    loki writes ``level_stats`` (survivors, harvests, overloads per pruning
    level) in every EP run group under ``/chunks``, which the compact output
    otherwise drops. A run whose buffer overflowed had its survivors trimmed.
    """
    n_runs = n_overloaded = n_overloads = 0
    chunks = src.get("chunks", {})
    for chunk_name, chunk in chunks.items():
        runs = chunk.get("runs", {})
        for run_name, run in runs.items():
            if "level_stats" not in run:
                continue
            stats = run["level_stats"]
            src.copy(
                stats,
                dst.require_group(f"level_stats/{chunk_name}"),
                name=run_name,
            )
            n_runs += 1
            if "n_overloads" in (stats.dtype.names or ()):
                k = int(np.asarray(stats["n_overloads"]).sum())
                n_overloads += k
                n_overloaded += k > 0
    return {
        "n_runs": n_runs,
        "n_runs_overloaded": n_overloaded,
        "n_overloads": n_overloads,
    }


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
    manifest = load_manifest(run, unit.series)
    key = search_sha256(resolved, unit, manifest)
    atomic_write_json(workdir / SEARCH_JSON, {"search_sha256": key})
    started = time.time()
    spool = kept_spool(run, unit, key)
    if spool is None:
        result, republished = search(run, resolved, unit, workdir)
    else:
        print(f"{unit_id}: publishing the search kept in {spool}, without searching")
        result = workdir / f"{unit.id}{RESULT_SUFFIX[unit.kind]}"
        publish_spool(resolved, unit, spool, result, observation(manifest))
        republished = True
    searched = time.time()
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
        "candidates_republished": republished,
        "resumed_from": str(spool) if spool is not None else None,
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
        "n_runs": counts["n_runs"],
        "n_runs_overloaded": counts["n_runs_overloaded"],
        "n_overloads": counts["n_overloads"],
        "raw_kept": raw_kept,
        "candidates_republished": republished,
        "resumed_from": str(spool) if spool is not None else None,
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
