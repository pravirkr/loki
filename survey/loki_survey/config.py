"""Survey and site configuration: read, validate, resolve and hash.

A run directory holds two files the user edits: ``survey.toml`` (the science)
and ``site.toml`` (the machine). ``check`` resolves both into ``resolved.json``
with every default written out, and hashes the science part alone, so moving a
run to another machine never invalidates it.

Science choices (frequency ranges, bin counts, thresholds, search boxes) have
no defaults. Where a value is left out, the default is loki's own, and the
resolved file records the value used.

The ``[collect]`` and ``[evaluate]`` tables of ``survey.toml`` choose how the
searches' candidates are combined and labelled. They change no search, so they
have their own hash: they can be changed after units have run, and every
analysis output records the hash it was made with.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from loki_survey.rundir import RunDir, canonical_json, read_json, sha256_text

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1
# Taylor derivatives in loki's parameter-row order, highest first. A search of
# order "jerk" has rows (jerk, accel, freq).
DERIVATIVES = ("crackle", "snap", "jerk", "accel")
ORDERS = ("freq", "accel", "jerk", "snap", "crackle")
SEARCH_KINDS = ("ffa", "ep")
TRUNCATIONS = ("pow2", "none")
EXECUTORS = ("local", "lsf", "slurm")
BATCH_EXECUTORS = ("lsf", "slurm")
BACKENDS = ("cpu", "cuda")
# Memory units an LSF cluster may use for limits (LSF_UNIT_FOR_LIMITS).
LSF_MEM_UNITS = ("KB", "MB", "GB")
# loki's defaults (PulsarSearchConfig, EPFreqSweep), recorded when not given.
LOKI_DEFAULT_ETA = 1.0
LOKI_DEFAULT_DUCY_MAX = 0.2
LOKI_DEFAULT_MIN_PD = 0.1
LOKI_DEFAULT_M_C_MAX = 10.0
LOKI_DEFAULT_M_P_MIN = 1.2
# loki's EP candidate path takes Taylor searches of 2 to 5 parameters; the
# five-parameter (crackle) search is the circular-orbit one.
EP_ORDERS = ("accel", "jerk", "snap", "crackle")


class ConfigError(ValueError):
    """A survey or site file that cannot be run as written."""


@dataclass(frozen=True)
class Region:
    """A frequency range searched with one bin count and sampling."""

    name: str
    f_min: float
    f_max: float
    nbins: int
    downsample: int
    ducy_max: float
    bands: int


@dataclass(frozen=True)
class Orbit:
    """Circular-orbit priors that set a box's derivative limits per band."""

    p_orb_min: float
    m_c_max: float
    m_p_min: float


@dataclass(frozen=True)
class Box:
    """One search: its algorithm, its order and its parameter limits."""

    name: str
    kind: str
    order: str
    limits: dict[str, list[float]]
    orbit: Orbit | None
    regions: tuple[str, ...]
    use_fourier: bool | None
    segments: int | None
    bseg_brute: int | None
    ref_segs: tuple[int, ...] | None

    @property
    def nparams(self) -> int:
        return ORDERS.index(self.order) + 1

    @property
    def derivatives(self) -> tuple[str, ...]:
        """The searched derivatives in loki's row order, highest first."""
        return DERIVATIVES[len(DERIVATIVES) - (self.nparams - 1) :]


@dataclass(frozen=True)
class Survey:
    name: str
    series: tuple[str, ...]
    truncate: str
    masks: dict[str, list[list[float]]]
    preprocess: dict[str, Any]
    snr_min: float
    eta: float
    min_pd: float
    raw_fraction: float
    regions: tuple[Region, ...]
    boxes: tuple[Box, ...]

    def region(self, name: str) -> Region:
        return next(r for r in self.regions if r.name == name)

    def box(self, name: str) -> Box:
        return next(b for b in self.boxes if b.name == name)


# Analysis defaults, written into resolved.json when survey.toml leaves them out.
# Two candidates from different units are one source when their frequencies
# differ by at most this many resolutions 1 / T_obs.
DEFAULT_RADIUS_BINS = 2.0
# Harmonic ratios p/q are labelled up to this numerator and denominator.
DEFAULT_COLLECT_HARMONIC_MAX = 8
DEFAULT_EVALUATE_HARMONIC_MAX = 16
# Fractional frequency window for a catalogue match: a topocentric frequency
# differs from a barycentric one by up to the Earth's orbital speed over c.
DEFAULT_DOPPLER = 1.0e-4
# A spectral line is a run of FFT bins whose power has this false-alarm
# probability over the whole spectrum.
DEFAULT_LINE_FAP = 1.0e-3
# fold folds this many of the strongest selected groups.
DEFAULT_FOLD_CANDIDATES = 20
# The control fold is at the candidate's frequency times this: a pulsar's
# profile smears there (1.002 f slips 0.002 f T cycles over the span T).
DEFAULT_FOLD_CONTROL = 1.002


@dataclass(frozen=True)
class Collect:
    """How ``collect`` groups candidates across units and what it removes.

    Grouping and labels never remove anything. Each removal rule is off by
    default, and every candidate a rule removes is written to the ledger.
    """

    radius_bins: float = DEFAULT_RADIUS_BINS
    harmonic_max: int = DEFAULT_COLLECT_HARMONIC_MAX
    dm_zero_max: float = 0.0
    snr_min: float = 0.0
    drop_undispersed: bool = False
    remove_harmonics: bool = False
    max_candidates: int = 0


@dataclass(frozen=True)
class Evaluate:
    """How ``evaluate`` matches candidates to a catalogue and to lines."""

    catalogue: str | None = None
    harmonic_max: int = DEFAULT_EVALUATE_HARMONIC_MAX
    doppler: float = DEFAULT_DOPPLER
    dm_tol: float = 0.0
    line_fap: float = DEFAULT_LINE_FAP
    line_tol_bins: float = 1.0


@dataclass(frozen=True)
class Fold:
    """How ``fold`` checks the strongest candidates with loki's own model."""

    candidates: int = DEFAULT_FOLD_CANDIDATES
    control_factor: float = DEFAULT_FOLD_CONTROL
    dm_curve: bool = True


@dataclass(frozen=True)
class Analysis:
    collect: Collect = Collect()
    evaluate: Evaluate = Evaluate()
    fold: Fold = Fold()


# Site defaults, written into resolved.json when site.toml leaves them out.
DEFAULT_POLL_S = {"local": 1.0, "lsf": 60.0, "slurm": 60.0}
DEFAULT_OOM_RETRIES = 1
DEFAULT_OOM_FACTOR = 1.5
DEFAULT_REQUEUE_RETRIES = 3
DEFAULT_HOST_FAILURES = 2
DEFAULT_STALL_MIN = 30.0
DEFAULT_SUBMIT_TIMEOUT_S = 300.0
DEFAULT_MAX_ARRAY = 1000
# A unit that hits its time limit is split in two this many times over.
DEFAULT_TIMEOUT_SPLITS = 1


@dataclass(frozen=True)
class Batch:
    """How to ask an LSF or Slurm cluster for a unit's resources."""

    queue: str | None
    account: str | None
    time_min: int | None
    select: str | None
    constraint: str | None
    hosts: tuple[str, ...]
    gpu: str | None
    mem_unit: str
    reserve_per_task: bool
    mem_limit: bool
    max_array: int
    extra: tuple[str, ...]


@dataclass(frozen=True)
class Site:
    executor: str
    max_memory_gb: float
    max_threads: int
    unit_memory_gb: float
    unit_overhead_gb: float
    unit_threads: int
    backend: str
    device: int
    python: str
    poll_s: float
    oom_retries: int
    oom_factor: float
    prologue: str = ""
    max_running: int = 0
    submit_timeout_s: float = DEFAULT_SUBMIT_TIMEOUT_S
    requeue_retries: int = DEFAULT_REQUEUE_RETRIES
    host_failures: int = DEFAULT_HOST_FAILURES
    stall_min: float = DEFAULT_STALL_MIN
    timeout_splits: int = DEFAULT_TIMEOUT_SPLITS
    batch: Batch | None = None

    @property
    def loki_memory_gb(self) -> float:
        """The memory budget loki plans a unit's search within."""
        return self.unit_memory_gb - self.unit_overhead_gb


def _table(doc: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = doc.get(key, {})
    if not isinstance(value, dict):
        msg = f"{where}: [{key}] must be a table"
        raise ConfigError(msg)
    return value


def _require(table: dict[str, Any], key: str, where: str) -> Any:  # noqa: ANN401
    if key not in table:
        msg = f"{where}: '{key}' is required"
        raise ConfigError(msg)
    return table[key]


def _range(value: Any, where: str) -> list[float]:  # noqa: ANN401
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int | float) for v in value)
        or not value[0] < value[1]
    ):
        msg = f"{where}: expected [min, max] with min < max, got {value!r}"
        raise ConfigError(msg)
    return [float(value[0]), float(value[1])]


def _positive(value: Any, where: str, *, integer: bool = False) -> float:  # noqa: ANN401
    kind = int if integer else int | float
    if not isinstance(value, kind) or isinstance(value, bool) or value <= 0:
        what = "a positive integer" if integer else "a positive number"
        msg = f"{where}: expected {what}, got {value!r}"
        raise ConfigError(msg)
    return value


def _optional_int(table: dict[str, Any], key: str, where: str) -> int | None:
    if key not in table:
        return None
    return int(_positive(table[key], f"{where}.{key}", integer=True))


def _check_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        msg = f"{where}: unknown keys {unknown}; allowed {sorted(allowed)}"
        raise ConfigError(msg)


def _parse_region(raw: dict[str, Any], where: str) -> Region:
    _check_keys(
        raw,
        {"name", "f_hz", "nbins", "downsample", "ducy_max", "bands"},
        where,
    )
    f_min, f_max = _range(_require(raw, "f_hz", where), f"{where}.f_hz")
    if f_min <= 0:
        msg = f"{where}.f_hz: frequencies must be positive"
        raise ConfigError(msg)
    ducy_max = float(raw.get("ducy_max", LOKI_DEFAULT_DUCY_MAX))
    if not 0 < ducy_max < 1:
        msg = f"{where}.ducy_max: expected a duty cycle in (0, 1)"
        raise ConfigError(msg)
    return Region(
        name=str(_require(raw, "name", where)),
        f_min=f_min,
        f_max=f_max,
        nbins=int(
            _positive(_require(raw, "nbins", where), f"{where}.nbins", integer=True),
        ),
        downsample=int(
            _positive(raw.get("downsample", 1), f"{where}.downsample", integer=True),
        ),
        ducy_max=ducy_max,
        bands=int(_positive(raw.get("bands", 1), f"{where}.bands", integer=True)),
    )


def _parse_box(raw: dict[str, Any], region_names: list[str], where: str) -> Box:
    _check_keys(
        raw,
        {
            "name",
            "kind",
            "order",
            "limits",
            "orbit",
            "regions",
            "use_fourier",
            "segments",
            "bseg_brute",
            "ref_segs",
        },
        where,
    )
    name = str(_require(raw, "name", where))
    kind = _require(raw, "kind", where)
    if kind not in SEARCH_KINDS:
        msg = f"{where}.kind: expected one of {SEARCH_KINDS}, got {kind!r}"
        raise ConfigError(msg)
    order = _require(raw, "order", where)
    if order not in ORDERS:
        msg = f"{where}.order: expected one of {ORDERS}, got {order!r}"
        raise ConfigError(msg)
    if kind == "ep" and order not in EP_ORDERS:
        msg = f"{where}.order: EP boxes support orders {EP_ORDERS} for now"
        raise ConfigError(msg)
    limits_raw = raw.get("limits", {})
    orbit_raw = raw.get("orbit")
    box = Box(
        name=name,
        kind=kind,
        order=order,
        limits={},
        orbit=None,
        regions=tuple(raw.get("regions", region_names)),
        use_fourier=raw.get("use_fourier"),
        segments=_optional_int(raw, "segments", where),
        bseg_brute=_optional_int(raw, "bseg_brute", where),
        ref_segs=tuple(raw["ref_segs"]) if "ref_segs" in raw else None,
    )
    if kind != "ep" and (box.segments is not None or box.ref_segs is not None):
        msg = f"{where}: 'segments' and 'ref_segs' are EP settings"
        raise ConfigError(msg)
    if box.segments is not None and box.segments & (box.segments - 1):
        msg = f"{where}.segments: loki needs a power of two, got {box.segments}"
        raise ConfigError(msg)
    if orbit_raw is not None and limits_raw:
        msg = f"{where}: give either 'limits' or 'orbit', not both"
        raise ConfigError(msg)
    if orbit_raw is not None:
        if order == "freq":
            msg = f"{where}: an orbit prior needs an order above 'freq'"
            raise ConfigError(msg)
        _check_keys(orbit_raw, {"p_orb_min_s", "m_c_max", "m_p_min"}, f"{where}.orbit")
        orbit = Orbit(
            p_orb_min=float(
                _positive(
                    _require(orbit_raw, "p_orb_min_s", f"{where}.orbit"),
                    f"{where}.orbit.p_orb_min_s",
                ),
            ),
            m_c_max=float(orbit_raw.get("m_c_max", LOKI_DEFAULT_M_C_MAX)),
            m_p_min=float(orbit_raw.get("m_p_min", LOKI_DEFAULT_M_P_MIN)),
        )
        box = replace(box, orbit=orbit)
    else:
        if not isinstance(limits_raw, dict):
            msg = f"{where}.limits must be a table"
            raise ConfigError(msg)
        _check_keys(limits_raw, set(box.derivatives), f"{where}.limits")
        limits = {}
        for deriv in box.derivatives:
            limits[deriv] = _range(
                _require(limits_raw, deriv, f"{where}.limits"),
                f"{where}.limits.{deriv}",
            )
        box = replace(box, limits=limits)
    unknown = sorted(set(box.regions) - set(region_names))
    if unknown:
        msg = f"{where}.regions: unknown regions {unknown}"
        raise ConfigError(msg)
    return box


def parse_survey(doc: dict[str, Any], where: str = "survey.toml") -> Survey:
    """Validate a parsed ``survey.toml`` and fill in loki's defaults."""
    _check_keys(
        doc,
        {
            "survey",
            "data",
            "preprocess",
            "search",
            "output",
            "collect",
            "evaluate",
            "fold",
        },
        where,
    )
    survey = _table(doc, "survey", where)
    data = _table(doc, "data", where)
    search = _table(doc, "search", where)
    output = _table(doc, "output", where)
    _check_keys(survey, {"name"}, f"{where} [survey]")
    _check_keys(data, {"series", "truncate", "masks"}, f"{where} [data]")
    _check_keys(
        search,
        {"snr_min", "eta", "min_pd", "region", "box"},
        f"{where} [search]",
    )
    _check_keys(output, {"raw_fraction"}, f"{where} [output]")

    series = _require(data, "series", f"{where} [data]")
    if isinstance(series, str):
        series = [series]
    if not series or not all(isinstance(s, str) for s in series):
        msg = f"{where} [data].series: expected a glob or a list of globs"
        raise ConfigError(msg)
    truncate = data.get("truncate", "pow2")
    if truncate not in TRUNCATIONS:
        msg = f"{where} [data].truncate: expected one of {TRUNCATIONS}"
        raise ConfigError(msg)
    masks = {}
    for name, ranges in data.get("masks", {}).items():
        masks[name] = [_range(r, f"{where} [data].masks.{name}") for r in ranges]

    regions_raw = search.get("region", [])
    if not regions_raw:
        msg = f"{where}: at least one [[search.region]] is required"
        raise ConfigError(msg)
    regions = tuple(
        _parse_region(r, f"{where} [[search.region]] {i}")
        for i, r in enumerate(regions_raw)
    )
    region_names = [r.name for r in regions]
    if len(set(region_names)) != len(region_names):
        msg = f"{where}: region names must be unique"
        raise ConfigError(msg)
    boxes_raw = search.get("box", [])
    if not boxes_raw:
        msg = f"{where}: at least one [[search.box]] is required"
        raise ConfigError(msg)
    boxes = tuple(
        _parse_box(b, region_names, f"{where} [[search.box]] {i}")
        for i, b in enumerate(boxes_raw)
    )
    box_names = [b.name for b in boxes]
    if len(set(box_names)) != len(box_names):
        msg = f"{where}: box names must be unique"
        raise ConfigError(msg)
    raw_fraction = float(output.get("raw_fraction", 0.0))
    if not 0.0 <= raw_fraction <= 1.0:
        msg = f"{where} [output].raw_fraction: expected a fraction in [0, 1]"
        raise ConfigError(msg)
    return Survey(
        name=str(_require(survey, "name", f"{where} [survey]")),
        series=tuple(series),
        truncate=truncate,
        masks=masks,
        preprocess=dict(_table(doc, "preprocess", where)),
        snr_min=float(
            _positive(
                _require(search, "snr_min", f"{where} [search]"),
                f"{where} [search].snr_min",
            ),
        ),
        eta=float(
            _positive(search.get("eta", LOKI_DEFAULT_ETA), f"{where} [search].eta"),
        ),
        min_pd=float(search.get("min_pd", LOKI_DEFAULT_MIN_PD)),
        raw_fraction=raw_fraction,
        regions=regions,
        boxes=boxes,
    )


def _bool(table: dict[str, Any], key: str, default: bool, where: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        msg = f"{where}.{key}: expected true or false, got {value!r}"
        raise ConfigError(msg)
    return value


def _nonnegative(table: dict[str, Any], key: str, default: float, where: str) -> float:
    value = table.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        msg = f"{where}.{key}: expected a number >= 0, got {value!r}"
        raise ConfigError(msg)
    return float(value)


def parse_analysis(doc: dict[str, Any], where: str = "survey.toml") -> Analysis:
    """Validate the ``[collect]`` and ``[evaluate]`` tables of ``survey.toml``."""
    raw = _table(doc, "collect", where)
    cw = f"{where} [collect]"
    _check_keys(raw, set(Collect.__dataclass_fields__), cw)
    defaults = Collect()
    collect = Collect(
        radius_bins=float(
            _positive(
                raw.get("radius_bins", defaults.radius_bins),
                f"{cw}.radius_bins",
            ),
        ),
        harmonic_max=_nonnegative_int(raw, "harmonic_max", defaults.harmonic_max, cw),
        dm_zero_max=_nonnegative(raw, "dm_zero_max", defaults.dm_zero_max, cw),
        snr_min=_nonnegative(raw, "snr_min", defaults.snr_min, cw),
        drop_undispersed=_bool(raw, "drop_undispersed", defaults.drop_undispersed, cw),
        remove_harmonics=_bool(raw, "remove_harmonics", defaults.remove_harmonics, cw),
        max_candidates=_nonnegative_int(
            raw,
            "max_candidates",
            defaults.max_candidates,
            cw,
        ),
    )
    if collect.remove_harmonics and collect.harmonic_max == 0:
        msg = f"{cw}: remove_harmonics needs harmonic_max above 0"
        raise ConfigError(msg)
    raw = _table(doc, "evaluate", where)
    ew = f"{where} [evaluate]"
    _check_keys(raw, set(Evaluate.__dataclass_fields__), ew)
    edefaults = Evaluate()
    catalogue = raw.get("catalogue")
    if catalogue is not None and not isinstance(catalogue, str):
        msg = f"{ew}.catalogue: expected a path"
        raise ConfigError(msg)
    line_fap = _positive(raw.get("line_fap", edefaults.line_fap), f"{ew}.line_fap")
    if line_fap >= 1:
        msg = f"{ew}.line_fap: expected a probability in (0, 1)"
        raise ConfigError(msg)
    evaluate = Evaluate(
        catalogue=catalogue,
        harmonic_max=int(
            _positive(
                raw.get("harmonic_max", edefaults.harmonic_max),
                f"{ew}.harmonic_max",
                integer=True,
            ),
        ),
        doppler=_nonnegative(raw, "doppler", edefaults.doppler, ew),
        dm_tol=_nonnegative(raw, "dm_tol", edefaults.dm_tol, ew),
        line_fap=float(line_fap),
        line_tol_bins=float(
            _positive(
                raw.get("line_tol_bins", edefaults.line_tol_bins),
                f"{ew}.line_tol_bins",
            ),
        ),
    )
    raw = _table(doc, "fold", where)
    fw = f"{where} [fold]"
    _check_keys(raw, set(Fold.__dataclass_fields__), fw)
    fdefaults = Fold()
    control = _positive(
        raw.get("control_factor", fdefaults.control_factor),
        f"{fw}.control_factor",
    )
    if control == 1.0:
        msg = f"{fw}.control_factor: must differ from 1"
        raise ConfigError(msg)
    fold = Fold(
        candidates=_nonnegative_int(raw, "candidates", fdefaults.candidates, fw),
        control_factor=float(control),
        dm_curve=_bool(raw, "dm_curve", fdefaults.dm_curve, fw),
    )
    return Analysis(collect=collect, evaluate=evaluate, fold=fold)


def _nonnegative_int(table: dict[str, Any], key: str, default: int, where: str) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{where}.{key}: expected a non-negative integer, got {value!r}"
        raise ConfigError(msg)
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:  # noqa: ANN401
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f"{where}: expected a list of strings"
        raise ConfigError(msg)
    return tuple(value)


def _parse_batch(kind: str, table: dict[str, Any], backend: str, where: str) -> Batch:
    """Validate the [lsf] or [slurm] table: each takes its own scheduler's words."""
    common = {"time_min", "gpu", "max_array", "extra"}
    if kind == "lsf":
        allowed = common | {
            "queue",
            "select",
            "hosts",
            "mem_unit",
            "reserve_per_task",
            "mem_limit",
        }
    else:
        allowed = common | {"partition", "account", "constraint"}
    _check_keys(table, allowed, where)
    mem_unit = "MB"
    reserve_per_task = False
    if kind == "lsf":
        mem_unit = _require(table, "mem_unit", where)
        if mem_unit not in LSF_MEM_UNITS:
            msg = f"{where}.mem_unit: expected one of {LSF_MEM_UNITS}, as in "
            msg += "LSF_UNIT_FOR_LIMITS"
            raise ConfigError(msg)
        reserve_per_task = _require(table, "reserve_per_task", where)
        if not isinstance(reserve_per_task, bool):
            msg = f"{where}.reserve_per_task: expected true or false, as in "
            msg += "RESOURCE_RESERVE_PER_TASK"
            raise ConfigError(msg)
    mem_limit = table.get("mem_limit", True)
    if not isinstance(mem_limit, bool):
        msg = f"{where}.mem_limit: expected true or false"
        raise ConfigError(msg)
    gpu = table.get("gpu")
    if backend == "cuda" and not gpu:
        what = "an LSF -gpu string" if kind == "lsf" else "a Slurm --gres value"
        msg = f"{where}.gpu: CUDA units need {what}"
        raise ConfigError(msg)
    return Batch(
        queue=table.get("queue", table.get("partition")),
        account=table.get("account"),
        time_min=_optional_int(table, "time_min", where),
        select=table.get("select"),
        constraint=table.get("constraint"),
        hosts=_strings(table.get("hosts", []), f"{where}.hosts"),
        gpu=gpu if backend == "cuda" else None,
        mem_unit=mem_unit,
        reserve_per_task=reserve_per_task,
        mem_limit=mem_limit,
        max_array=int(
            _positive(
                table.get("max_array", DEFAULT_MAX_ARRAY),
                f"{where}.max_array",
                integer=True,
            ),
        ),
        extra=_strings(table.get("extra", []), f"{where}.extra"),
    )


def parse_site(doc: dict[str, Any], where: str = "site.toml") -> Site:
    """Validate a parsed ``site.toml``."""
    _check_keys(doc, {"executor", "unit", "retry", "lsf", "slurm"}, where)
    executor = _table(doc, "executor", where)
    unit = _table(doc, "unit", where)
    retry = _table(doc, "retry", where)
    ex_where = f"{where} [executor]"
    kind = _require(executor, "kind", ex_where)
    if kind not in EXECUTORS:
        msg = f"{ex_where}.kind: expected one of {EXECUTORS}, got {kind!r}"
        raise ConfigError(msg)
    batch_keys = {"prologue", "max_running", "submit_timeout_s"}
    local_keys = {"max_threads"}
    _check_keys(
        executor,
        {"kind", "max_memory_gb", "python", "poll_s"}
        | (batch_keys if kind in BATCH_EXECUTORS else local_keys),
        ex_where,
    )
    _check_keys(
        unit,
        {"memory_gb", "overhead_gb", "threads", "backend", "device"},
        f"{where} [unit]",
    )
    _check_keys(
        retry,
        {
            "oom_retries",
            "oom_factor",
            "requeue_retries",
            "host_failures",
            "stall_min",
            "timeout_splits",
        },
        f"{where} [retry]",
    )
    for other in BATCH_EXECUTORS:
        if other in doc and other != kind:
            msg = f"{where}: [{other}] is set but [executor].kind is {kind!r}"
            raise ConfigError(msg)
    backend = unit.get("backend", "cpu")
    if backend not in BACKENDS:
        msg = f"{where} [unit].backend: expected one of {BACKENDS}"
        raise ConfigError(msg)
    unit_threads = int(
        _positive(
            _require(unit, "threads", f"{where} [unit]"),
            f"{where} [unit].threads",
            integer=True,
        ),
    )
    batch = None
    if kind in BATCH_EXECUTORS:
        batch = _parse_batch(
            kind,
            _table(doc, kind, where),
            backend,
            f"{where} [{kind}]",
        )
        max_threads = unit_threads
    else:
        max_threads = int(
            _positive(
                _require(executor, "max_threads", ex_where),
                f"{ex_where}.max_threads",
                integer=True,
            ),
        )
    prologue = executor.get("prologue", "")
    if not isinstance(prologue, str):
        msg = f"{ex_where}.prologue: expected shell lines as a string"
        raise ConfigError(msg)
    stall_min = retry.get("stall_min", DEFAULT_STALL_MIN)
    if not isinstance(stall_min, int | float) or stall_min < 0:
        msg = f"{where} [retry].stall_min: expected minutes >= 0 (0 turns it off)"
        raise ConfigError(msg)
    site = Site(
        executor=kind,
        max_memory_gb=float(
            _positive(
                _require(executor, "max_memory_gb", ex_where),
                f"{ex_where}.max_memory_gb",
            ),
        ),
        max_threads=max_threads,
        unit_memory_gb=float(
            _positive(
                _require(unit, "memory_gb", f"{where} [unit]"),
                f"{where} [unit].memory_gb",
            ),
        ),
        unit_overhead_gb=float(
            _positive(
                _require(unit, "overhead_gb", f"{where} [unit]"),
                f"{where} [unit].overhead_gb",
            ),
        ),
        unit_threads=unit_threads,
        backend=backend,
        device=int(unit.get("device", 0)),
        python=str(executor.get("python", sys.executable)),
        poll_s=float(
            _positive(
                executor.get("poll_s", DEFAULT_POLL_S[kind]),
                f"{ex_where}.poll_s",
            ),
        ),
        oom_retries=_nonnegative_int(
            retry,
            "oom_retries",
            DEFAULT_OOM_RETRIES,
            f"{where} [retry]",
        ),
        oom_factor=float(
            _positive(
                retry.get("oom_factor", DEFAULT_OOM_FACTOR),
                f"{where} [retry].oom_factor",
            ),
        ),
        prologue=prologue,
        max_running=_nonnegative_int(executor, "max_running", 0, ex_where),
        submit_timeout_s=float(
            _positive(
                executor.get("submit_timeout_s", DEFAULT_SUBMIT_TIMEOUT_S),
                f"{ex_where}.submit_timeout_s",
            ),
        ),
        requeue_retries=_nonnegative_int(
            retry,
            "requeue_retries",
            DEFAULT_REQUEUE_RETRIES,
            f"{where} [retry]",
        ),
        host_failures=int(
            _positive(
                retry.get("host_failures", DEFAULT_HOST_FAILURES),
                f"{where} [retry].host_failures",
                integer=True,
            ),
        ),
        stall_min=float(stall_min),
        timeout_splits=_nonnegative_int(
            retry,
            "timeout_splits",
            DEFAULT_TIMEOUT_SPLITS,
            f"{where} [retry]",
        ),
        batch=batch,
    )
    if site.oom_factor <= 1.0:
        msg = f"{where} [retry].oom_factor: must be above 1"
        raise ConfigError(msg)
    if site.unit_overhead_gb >= site.unit_memory_gb:
        msg = f"{where} [unit]: overhead_gb must be below memory_gb"
        raise ConfigError(msg)
    if site.unit_memory_gb > site.max_memory_gb:
        msg = f"{where}: [unit].memory_gb exceeds [executor].max_memory_gb"
        raise ConfigError(msg)
    if site.unit_threads > site.max_threads:
        msg = f"{where}: [unit].threads exceeds [executor].max_threads"
        raise ConfigError(msg)
    return site


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        msg = f"{path} not found"
        raise ConfigError(msg) from None
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from None


def science_dict(survey: Survey) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **asdict(survey)}


def science_sha256(survey: Survey) -> str:
    """Hash everything that changes what a search finds."""
    return sha256_text(canonical_json(science_dict(survey)))


def analysis_sha256(analysis: Analysis) -> str:
    """Hash the choices that combine and label candidates."""
    return sha256_text(canonical_json(asdict(analysis)))


@dataclass(frozen=True)
class Resolved:
    survey: Survey
    site: Site
    science_sha256: str
    analysis: Analysis = Analysis()

    @property
    def analysis_sha256(self) -> str:
        return analysis_sha256(self.analysis)


def resolve(run: RunDir) -> Resolved:
    doc = load_toml(run.survey_toml)
    survey = parse_survey(doc, str(run.survey_toml))
    analysis = parse_analysis(doc, str(run.survey_toml))
    site = parse_site(load_toml(run.site_toml), str(run.site_toml))
    return Resolved(
        survey=survey,
        site=site,
        science_sha256=science_sha256(survey),
        analysis=analysis,
    )


def _from_dict_survey(d: dict[str, Any]) -> Survey:
    regions = tuple(Region(**r) for r in d["regions"])
    boxes = tuple(
        Box(
            **{
                **b,
                "orbit": Orbit(**b["orbit"]) if b["orbit"] else None,
                "regions": tuple(b["regions"]),
                "ref_segs": tuple(b["ref_segs"]) if b["ref_segs"] else None,
            },
        )
        for b in d["boxes"]
    )
    return Survey(
        **{
            **d,
            "series": tuple(d["series"]),
            "regions": regions,
            "boxes": boxes,
        },
    )


def _from_dict_site(d: dict[str, Any]) -> Site:
    batch = d.get("batch")
    if batch is not None:
        batch = Batch(
            **{**batch, "hosts": tuple(batch["hosts"]), "extra": tuple(batch["extra"])},
        )
    return Site(**{**d, "batch": batch})


def load_resolved(run: RunDir) -> Resolved:
    """Read ``resolved.json``, the configuration every later stage uses."""
    if not run.resolved_json.is_file():
        msg = f"{run.resolved_json} not found; run 'loki-survey check' first"
        raise ConfigError(msg)
    doc = read_json(run.resolved_json)
    science = dict(doc["survey"])
    science.pop("schema_version", None)
    survey = _from_dict_survey(science)
    site = _from_dict_site(doc["site"])
    sha = science_sha256(survey)
    if sha != doc["science_sha256"]:
        msg = f"{run.resolved_json}: science hash mismatch; was it edited by hand?"
        raise ConfigError(msg)
    analysis = Analysis()
    if "analysis" in doc:
        raw = doc["analysis"]
        analysis = Analysis(
            collect=Collect(**raw["collect"]),
            evaluate=Evaluate(**raw["evaluate"]),
            fold=Fold(**raw.get("fold", {})),
        )
    return Resolved(survey=survey, site=site, science_sha256=sha, analysis=analysis)


def resolved_dict(resolved: Resolved, versions: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "survey": science_dict(resolved.survey),
        "site": asdict(resolved.site),
        "science_sha256": resolved.science_sha256,
        "analysis": asdict(resolved.analysis),
        "analysis_sha256": resolved.analysis_sha256,
        "versions": versions,
    }
