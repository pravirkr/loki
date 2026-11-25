"""Match the groups of ``loki sift`` to known pulsars.

``libloki.io.sift`` (``loki sift`` on the command line) groups the candidates
of many result files. This module adds the catalogue step: each group's best
frequency, corrected for the Earth's Doppler shift at its file's reference
MJD, is compared with low-order ratios ``n/m`` of the catalogue frequencies.

The catalogue is a CSV file with a ``name`` column, the frequency as ``f0``
(Hz) or the period as ``p0_ms``, and optionally ``dm`` and ``doppler`` (the
largest fractional orbital Doppler shift of a binary, added to the
tolerance).

The Earth's Doppler shift needs the source position and the observatory;
:func:`astropy_doppler` builds it with astropy (an optional dependency).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from . import libloki

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class KnownPulsar:
    name: str
    f0: float
    dm: float = math.nan
    doppler: float = 0.0


@dataclass(frozen=True)
class Match:
    name: str
    ratio: Fraction
    dm: float


def load_catalog(path: str | Path) -> list[KnownPulsar]:
    """Read a catalogue CSV (``name``, ``f0`` or ``p0_ms``, ``dm``, ``doppler``)."""
    out = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("f0"):
                f0 = float(row["f0"])
            elif row.get("p0_ms"):
                f0 = 1000.0 / float(row["p0_ms"])
            else:
                msg = f"catalogue row {row.get('name', '?')} has no f0 or p0_ms"
                raise ValueError(msg)
            out.append(
                KnownPulsar(
                    name=row["name"],
                    f0=f0,
                    dm=float(row["dm"]) if row.get("dm") else math.nan,
                    doppler=float(row["doppler"]) if row.get("doppler") else 0.0,
                ),
            )
    return out


def astropy_doppler(
    ra: str,
    dec: str,
    lat_deg: float,
    lon_deg: float,
    height_m: float,
) -> Callable[[float], float]:
    """Return ``mjd -> f_topo / f_bary - 1`` for a source and an observatory."""
    try:
        import astropy.units as u  # noqa: PLC0415
        from astropy.coordinates import EarthLocation, SkyCoord  # noqa: PLC0415
        from astropy.time import Time  # noqa: PLC0415
    except ImportError as exc:
        msg = "the Earth Doppler correction needs astropy"
        raise ImportError(msg) from exc
    site = EarthLocation(
        lat=lat_deg * u.deg,
        lon=lon_deg * u.deg,
        height=height_m * u.m,
    )
    source = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    cache: dict[float, float] = {}

    def doppler(mjd: float) -> float:
        key = round(mjd, 3)
        if key not in cache:
            v = source.radial_velocity_correction(
                kind="barycentric",
                obstime=Time(key, format="mjd"),
                location=site,
            )
            cache[key] = v.to(u.m / u.s).value / 299792458.0
        return cache[key]

    return doppler


def _ratios(nmax: int, mmax: int) -> list[Fraction]:
    return sorted(
        {Fraction(n, m) for n in range(1, nmax + 1) for m in range(1, mmax + 1)},
        key=lambda q: (q.numerator * q.denominator, q),
    )


def match_groups(
    result: libloki.io.SiftResult,
    catalog: Sequence[KnownPulsar],
    *,
    doppler: Callable[[float], float] | None = None,
    tolerance: float = 1e-5,
    nmax: int = 8,
    mmax: int = 4,
) -> list[Match | None]:
    """Match each group's best frequency to ``n/m`` times a catalogue frequency.

    With ``doppler`` the frequency is divided by ``1 + doppler(reference_mjd)``
    of the best member's file first, unless that file's sample times were
    already barycentric. The simplest ratio wins; a binary's ``doppler``
    widens its tolerance.
    """
    ratios = _ratios(nmax, mmax)
    out: list[Match | None] = []
    for group in result.groups:
        best = group.members[0]
        freq = best.freq
        file = result.files[best.file]
        mjd = file.reference_mjd
        if doppler is not None and mjd is not None and not file.barycentric:
            freq /= 1.0 + doppler(mjd)
        found = None
        for q in ratios:
            for psr in catalog:
                target = float(q) * psr.f0
                if abs(freq - target) <= (psr.doppler + tolerance) * target:
                    found = Match(psr.name, q, psr.dm)
                    break
            if found is not None:
                break
        out.append(found)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="loki-sift-match",
        description="Group result files with loki sift and match the groups "
        "to a catalogue of known pulsars.",
    )
    ap.add_argument("files", nargs="+")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--snr-min", type=float, default=7.0)
    ap.add_argument("--tol", type=float, default=2e-4, help="grouping tolerance")
    ap.add_argument("--match-tol", type=float, default=1e-5)
    ap.add_argument("--min-segs", type=int, default=1)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--ra", help="source RA (hh:mm:ss) for the Earth Doppler shift")
    ap.add_argument("--dec", help="source Dec (dd:mm:ss)")
    ap.add_argument(
        "--site",
        help="observatory latitude,longitude (deg),height (m)",
    )
    a = ap.parse_args(argv)
    options = libloki.io.SiftOptions()
    options.snr_min = a.snr_min
    options.tolerance = a.tol
    options.min_segments = a.min_segs
    result = libloki.io.sift(a.files, options)
    doppler = None
    if a.ra or a.dec or a.site:
        if not (a.ra and a.dec and a.site):
            ap.error("--ra, --dec and --site go together")
        lat, lon, height = (float(x) for x in a.site.split(","))
        doppler = astropy_doppler(a.ra, a.dec, lat, lon, height)
    matches = match_groups(
        result,
        load_catalog(a.catalog),
        doppler=doppler,
        tolerance=a.match_tol,
    )
    print(  # noqa: T201
        f"{result.n_candidates} candidates from {len(result.files)} files in "
        f"{result.n_segments} segments: {len(result.groups)} groups",
    )
    print(  # noqa: T201
        f"{'#':>4} {'S/N':>7} {'f (Hz)':>14} {'DM':>8} {'segs':>5} {'edge':>5}  match",
    )
    for i, (group, match) in enumerate(zip(result.groups, matches, strict=True)):
        if i >= a.top:
            break
        best = group.members[0]
        dm = result.files[best.file].dm
        edge = group.edge_ratio
        print(  # noqa: T201
            f"{i:>4} {best.snr:7.2f} {best.freq:14.8f} "
            f"{'-' if dm is None else f'{dm:8.2f}':>8} {len(group.segments):>5} "
            f"{'-' if edge is None else f'{edge:5.2f}':>5}  "
            f"{'' if match is None else f'{match.name}:{match.ratio}'}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
