"""Split a unit that hit its time limit into two half bands.

The unit list written by ``plan`` never changes once units have run. A split is
recorded instead, one line per split in ``splits.jsonl``, and the units a run
works on are the planned ones with each split unit replaced by its two parts.
Each part searches half of its parent's band in log frequency, so it needs
about half the time. EP parts get plans of their own, built when they are made.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

from loki_survey.plan import Unit, check_nyquist, param_limits, plan_key
from loki_survey.rundir import RunDir, canonical_json

if TYPE_CHECKING:
    from loki_survey.config import Resolved

# The suffix of a part's id: the parent's id, then "__s0" or "__s1".
PART_SUFFIX = "__s"
PARTS = 2


def split_depth(unit_id: str) -> int:
    """Return how many times the unit's band has been halved."""
    return unit_id.count(PART_SUFFIX)


def split_unit(resolved: Resolved, unit: Unit) -> list[Unit]:
    """Return the unit's two parts, each half its band in log frequency."""
    box = resolved.survey.box(unit.box)
    middle = math.sqrt(unit.f_min * unit.f_max)
    parts = []
    for index, (f_lo, f_hi) in enumerate(((unit.f_min, middle), (middle, unit.f_max))):
        part = Unit(
            **{
                **asdict(unit),
                "id": f"{unit.id}{PART_SUFFIX}{index}",
                "f_min": f_lo,
                "f_max": f_hi,
                "param_limits": param_limits(box, f_lo, f_hi),
                "plan_key": None,
            },
        )
        check_nyquist(part)
        if part.kind == "ep":
            part = Unit(**{**asdict(part), "plan_key": plan_key(resolved, part)})
        parts.append(part)
    return parts


def load_splits(run: RunDir) -> dict[str, list[Unit]]:
    """Return each split unit's id and its parts."""
    if not run.splits_jsonl.is_file():
        return {}
    splits = {}
    for line in run.splits_jsonl.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            splits[record["parent"]] = [Unit(**u) for u in record["parts"]]
    return splits


def record_split(run: RunDir, parent: Unit, parts: list[Unit], reason: str) -> None:
    """Append a split to ``splits.jsonl``; a line is written whole or not at all."""
    line = canonical_json(
        {
            "parent": parent.id,
            "parts": [asdict(p) for p in parts],
            "reason": reason,
            "time": time.time(),
        },
    )
    with run.splits_jsonl.open("a") as handle:
        handle.write(line + "\n")


def expand(units: list[Unit], splits: dict[str, list[Unit]]) -> list[Unit]:
    """Replace each split unit by its parts, and theirs, in order."""
    out: list[Unit] = []
    for unit in units:
        if unit.id in splits:
            out.extend(expand(splits[unit.id], splits))
        else:
            out.append(unit)
    return out
