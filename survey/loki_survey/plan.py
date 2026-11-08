"""The ``plan`` stage: split the survey into units and build the EP plans.

A unit is one series x region band x search box: the smallest piece that runs
on its own and writes its own output. EP units with the same search geometry
share one plan, built here once, pinned by its hash, and only read by units.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

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
    from pathlib import Path

    from loki_survey.config import Box, Region, Resolved


@dataclass(frozen=True)
class Unit:
    id: str
    series: str
    region: str
    band: int
    box: str
    kind: str
    order: str
    downsample: int
    nsamps: int
    tsamp: float
    f_min: float
    f_max: float
    nbins: int
    ducy_max: float
    param_limits: list[list[float]]
    use_fourier: bool | None
    segments: int | None
    bseg_brute: int | None
    ref_segs: list[int] | None
    orbit: dict[str, float] | None
    plan_key: str | None


def band_edges(region: Region) -> list[tuple[float, float]]:
    """Split a region into bands of equal width in log frequency."""
    ratio = (region.f_max / region.f_min) ** (1.0 / region.bands)
    edges = [region.f_min * ratio**k for k in range(region.bands + 1)]
    edges[-1] = region.f_max
    return list(itertools.pairwise(edges))


def param_limits(box: Box, f_min: float, f_max: float) -> list[list[float]]:
    """Loki's parameter rows for a band: derivatives highest first, then freq."""
    if box.orbit is not None:
        from loki import libloki  # noqa: PLC0415

        limits = libloki.configs.circular_param_limits(
            f_min,
            f_max,
            poly_order=box.nparams,
            p_orb_min=box.orbit.p_orb_min,
            m_c_max=box.orbit.m_c_max,
            m_p_min=box.orbit.m_p_min,
        )
        return np.asarray(limits, dtype=np.float64).tolist()
    return [box.limits[d] for d in box.derivatives] + [[f_min, f_max]]


def plan_key(resolved: Resolved, unit: Unit) -> str:
    """Everything an EP plan depends on; units that agree share one plan."""
    survey, site = resolved.survey, resolved.site
    geometry = {
        "nsamps": unit.nsamps,
        "tsamp": unit.tsamp,
        "nbins": unit.nbins,
        "ducy_max": unit.ducy_max,
        "param_limits": unit.param_limits,
        "order": unit.order,
        "orbit": unit.orbit,
        "use_fourier": unit.use_fourier,
        "segments": unit.segments,
        "bseg_brute": unit.bseg_brute,
        "ref_segs": unit.ref_segs,
        "eta": survey.eta,
        "snr_min": survey.snr_min,
        "min_pd": survey.min_pd,
        "loki_memory_gb": site.loki_memory_gb,
        "backend": site.backend,
    }
    return sha256_text(canonical_json(geometry))[:16]


def make_units(resolved: Resolved, manifests: list[dict[str, Any]]) -> list[Unit]:
    survey = resolved.survey
    units = []
    for manifest in manifests:
        for box in survey.boxes:
            for region_name in box.regions:
                region = survey.region(region_name)
                arrays = manifest["arrays"][str(region.downsample)]
                for band, (f_lo, f_hi) in enumerate(band_edges(region)):
                    uid = f"{manifest['series']}__{region.name}__b{band}__{box.name}"
                    unit = Unit(
                        id=uid,
                        series=manifest["series"],
                        region=region.name,
                        band=band,
                        box=box.name,
                        kind=box.kind,
                        order=box.order,
                        downsample=region.downsample,
                        nsamps=arrays["nsamps"],
                        tsamp=arrays["tsamp"],
                        f_min=f_lo,
                        f_max=f_hi,
                        nbins=region.nbins,
                        ducy_max=region.ducy_max,
                        param_limits=param_limits(box, f_lo, f_hi),
                        use_fourier=box.use_fourier,
                        segments=box.segments,
                        bseg_brute=box.bseg_brute,
                        ref_segs=list(box.ref_segs) if box.ref_segs else None,
                        orbit=asdict(box.orbit) if box.orbit else None,
                        plan_key=None,
                    )
                    check_nyquist(unit)
                    if box.kind == "ep":
                        unit = Unit(
                            **{**asdict(unit), "plan_key": plan_key(resolved, unit)},
                        )
                    units.append(unit)
    return units


def check_nyquist(unit: Unit) -> None:
    """Refuse a unit whose top frequency has fewer than nbins samples per period."""
    f_top = unit.param_limits[-1][1]
    if f_top * unit.nbins * unit.tsamp > 1.0:
        msg = (
            f"{unit.id}: {unit.nbins} bins at {f_top:.6g} Hz need a sampling "
            f"interval of at most {1.0 / (f_top * unit.nbins):.6g} s, but the "
            f"series has {unit.tsamp:.6g} s (downsample {unit.downsample})"
        )
        raise ValueError(msg)


def search_config(resolved: Resolved, unit: Unit) -> Any:  # noqa: ANN401
    """Loki's PulsarSearchConfig for one unit."""
    from loki import libloki  # noqa: PLC0415

    survey, site = resolved.survey, resolved.site
    kwargs: dict[str, Any] = {
        "nsamps": unit.nsamps,
        "tsamp": unit.tsamp,
        "nbins": unit.nbins,
        "eta": survey.eta,
        "param_limits": np.asarray(unit.param_limits, dtype=np.float64),
        "ducy_max": unit.ducy_max,
        "nthreads": site.unit_threads,
        "max_process_memory_gb": site.loki_memory_gb,
        "snr_min": survey.snr_min,
    }
    if unit.use_fourier is not None:
        kwargs["use_fourier"] = unit.use_fourier
    if unit.segments is not None:
        kwargs["bseg_ffa"] = unit.nsamps // unit.segments
    if unit.bseg_brute is not None:
        kwargs["bseg_brute"] = unit.bseg_brute
    if unit.kind == "ep":
        kwargs["prune_poly_order"] = len(unit.param_limits)
        if unit.orbit is not None:
            kwargs["p_orb_min"] = unit.orbit["p_orb_min"]
            kwargs["m_c_max"] = unit.orbit["m_c_max"]
            kwargs["m_p_min"] = unit.orbit["m_p_min"]
        return libloki.configs.PulsarSearchConfig(**kwargs)
    return libloki.configs.FFASearchConfig(**kwargs)


def candidate_options() -> Any:  # noqa: ANN401
    """Loki's candidate options for every unit.

    An EP plan is sized within the search's memory budget less the candidate
    memory, so the plan must be built with the same options the unit runs.
    """
    from loki import libloki  # noqa: PLC0415

    return libloki.configs.CandidateOptions()


def plan_cache_path(run: RunDir, key: str) -> Path:
    return run.plans_dir / f"ep_{key}.h5"


def build_ep_plan(run: RunDir, resolved: Resolved, unit: Unit) -> dict[str, Any]:
    """Build one EP plan cache and record its hash."""
    from loki import libloki  # noqa: PLC0415

    if unit.plan_key is None:
        msg = f"{unit.id} is not an EP unit"
        raise ValueError(msg)
    cache = plan_cache_path(run, unit.plan_key)
    record_path = cache.with_suffix(".json")
    if cache.is_file() and record_path.is_file():
        record = read_json(record_path)
        if record["sha256"] == sha256_file(cache):
            return record
    tmp = cache.with_name(f".{cache.name}.building")
    tmp.unlink(missing_ok=True)
    libloki.prune.EPFreqSweep(
        search_config(resolved, unit),
        show_progress=False,
        min_pd=resolved.survey.min_pd,
        poly_basis="taylor",
        plan_cache_file=str(tmp),
        ref_segs=unit.ref_segs,
        backend=resolved.site.backend,
        device=resolved.site.device,
        candidates=candidate_options(),
    )
    tmp.replace(cache)
    record = {
        "key": unit.plan_key,
        "example_unit": unit.id,
        "sha256": sha256_file(cache),
    }
    atomic_write_json(record_path, record)
    return record


def write_units(run: RunDir, units: list[Unit]) -> None:
    """Write the unit list, refusing to replace a different list once units ran."""
    text = "".join(canonical_json(asdict(u)) + "\n" for u in units)
    if run.units_jsonl.is_file():
        old = run.units_jsonl.read_text()
        if old != text and any(run.out_dir.glob("*.json")):
            msg = (
                f"{run.units_jsonl} would change, but units already finished in "
                f"{run.out_dir}; start a new run directory instead"
            )
            raise RuntimeError(msg)
    atomic_write_text(run.units_jsonl, text)


def load_units(run: RunDir) -> list[Unit]:
    if not run.units_jsonl.is_file():
        msg = f"{run.units_jsonl} not found; run 'loki-survey plan' first"
        raise FileNotFoundError(msg)
    return [
        Unit(**json.loads(line)) for line in run.units_jsonl.read_text().splitlines()
    ]


def run_plan(
    run: RunDir,
    resolved: Resolved,
    manifests: list[dict[str, Any]],
) -> list[Unit]:
    units = make_units(resolved, manifests)
    write_units(run, units)
    seen: dict[str, Unit] = {}
    for unit in units:
        if unit.plan_key is not None and unit.plan_key not in seen:
            seen[unit.plan_key] = unit
    for key, unit in seen.items():
        print(f"plan: EP plan {key} (for {unit.id})")
        build_ep_plan(run, resolved, unit)
    kinds: dict[tuple[str, str], int] = {}
    for unit in units:
        kinds[(unit.box, unit.region)] = kinds.get((unit.box, unit.region), 0) + 1
    print(f"plan: {len(units)} units, {len(seen)} EP plans")
    for (box, region), count in sorted(kinds.items()):
        print(f"  {box:>16} x {region:<12} {count:6d} units")
    return units
