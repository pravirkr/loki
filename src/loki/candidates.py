"""Read Loki candidate files and export folding lists for PulsarX.

The converter handles the common schema and legacy flat FFA results. It uses
SQLite temporary storage to sort large lists without loading them into RAM.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

C = 299792458.0
TABLES = ("candidates", "clusters", "detections")


def _text(value: Any) -> str:  # noqa: ANN401
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class CandidateFile:
    """Context-managed, bounded-batch reader for common and legacy FFA files."""

    def __init__(self, path: str | Path, *, table: str = "candidates") -> None:
        if table not in TABLES:
            msg = f"Unknown candidate table: {table}"
            raise ValueError(msg)
        self.path = Path(path)
        self.table = table
        self._file: h5py.File | None = None
        self.legacy = False
        self.metadata: dict[str, Any] = {}
        self.param_names: list[str] = []

    def __enter__(self) -> Self:
        self._file = h5py.File(self.path, "r")
        try:
            self._validate_file()
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def _validate_file(self) -> None:
        handle = self._file
        if handle is None:
            msg = "Candidate file is not open"
            raise RuntimeError(msg)
        self.legacy = "schema_version" not in handle.attrs
        if int(handle.attrs.get("complete", 0)) != 1:
            msg = "Candidate file is incomplete"
            raise ValueError(msg)
        if not self.legacy:
            version = _text(handle.attrs["schema_version"])
            if version.split(".")[0] != "1":
                msg = f"Unsupported candidate schema: {version}"
                raise ValueError(msg)
            if self.table not in handle:
                msg = f"Table /{self.table} was not retained"
                raise ValueError(msg)
            if "observation" not in handle:
                msg = "Candidate file has no observation metadata"
                raise ValueError(msg)
            self.metadata = dict(handle["observation"].attrs)
        else:
            if "ffa_version" not in handle.attrs:
                msg = "Unsupported legacy file; expected flat FFA results"
                raise ValueError(msg)
            self.metadata = dict(handle.attrs)
        self.param_names = [_text(x) for x in handle.attrs.get("param_names", [])]
        if not self.param_names or self.param_names[-1] != "freq":
            msg = "Missing or unsupported native parameter convention"
            raise ValueError(msg)
        expected = ["crackle", "snap", "jerk", "accel", "freq"]
        if self.param_names != expected[-len(self.param_names) :]:
            msg = "Only kinematic Taylor candidate parameters are supported"
            raise ValueError(msg)
        group = self._group
        nrows = group["snr"].shape[0]
        for name in ("snr", "width", "nbins", "param_sets"):
            if group[name].shape[0] != nrows:
                msg = f"Candidate column length mismatch: {name}"
                raise ValueError(msg)
        if group["param_sets"].shape != (nrows, len(self.param_names)):
            msg = "Invalid native parameter array shape"
            raise ValueError(msg)
        for name in ("width", "nbins"):
            if group[name].dtype.kind not in "iu":
                msg = f"Candidate column must contain integers: {name}"
                raise ValueError(msg)
        if not self.legacy:
            if group["id"].dtype.kind not in "iu":
                msg = "Candidate IDs must contain integers"
                raise ValueError(msg)
            for name in ("id", "cluster_id", "f0", "f1", "f2", "duty_cycle"):
                if group[name].shape != (nrows,):
                    msg = f"Invalid candidate column shape: {name}"
                    raise ValueError(msg)

    @property
    def _group(self) -> h5py.Group:
        if self._file is None:
            msg = "Open CandidateFile using a with statement"
            raise RuntimeError(msg)
        return self._file if self.legacy else self._file[self.table]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self) -> int:
        return self._group["snr"].shape[0]

    def iter_batches(self, batch_size: int = 65536) -> Iterator[dict[str, np.ndarray]]:
        """Yield normalized column dictionaries, preserving row identity."""
        if batch_size < 1:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        group = self._group
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            rows = {
                name: group[name][start:stop]
                for name in ("param_sets", "snr", "width", "nbins")
            }
            if self.legacy:
                params = rows["param_sets"]
                f0 = params[:, -1]
                rows.update(
                    id=np.arange(start, stop, dtype=np.uint64),
                    f0=f0,
                    f1=-f0 * params[:, -2] / C
                    if params.shape[1] >= 2
                    else np.zeros(stop - start),
                    f2=-f0 * params[:, -3] / C
                    if params.shape[1] >= 3
                    else np.zeros(stop - start),
                )
            else:
                for name in ("id", "cluster_id", "f0", "f1", "f2", "duty_cycle"):
                    rows[name] = group[name][start:stop]
            if not self.legacy:
                for name in (
                    "member_count",
                    "parent_cluster_id",
                    "harmonic_numerator",
                    "harmonic_denominator",
                    "selected",
                    "source_region",
                    "source_index",
                    "param_errors",
                    "has_parameter_errors",
                ):
                    if name in group:
                        rows[name] = group[name][start:stop]
            yield rows


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        msg = f"{label} must be finite"
        raise ValueError(msg)
    return value


def _validate_batch(batch: dict[str, np.ndarray]) -> None:
    params = batch["param_sets"]
    if not np.isfinite(params).all():
        msg = "Non-finite native candidate parameters"
        raise ValueError(msg)
    if params.shape[1] > 3 and np.any(params[:, :-3] != 0):
        msg = "PulsarX F0/F1/F2 export cannot represent nonzero derivatives above jerk"
        raise ValueError(msg)
    expected = [params[:, -1]]
    expected.extend(
        -params[:, -1] * params[:, -1 - order] / C
        if params.shape[1] > order
        else np.zeros(len(params))
        for order in (1, 2)
    )
    for name, values in zip(("f0", "f1", "f2"), expected, strict=True):
        if not np.allclose(batch[name], values, rtol=1e-12, atol=0):
            msg = f"Canonical {name} disagrees with native parameters"
            raise ValueError(msg)
    if not np.isfinite(batch["snr"]).all():
        msg = "Non-finite candidate S/N"
        raise ValueError(msg)
    if (
        np.any(batch["width"] <= 0)
        or np.any(batch["nbins"] <= 0)
        or np.any(batch["width"] > batch["nbins"])
    ):
        msg = "Invalid candidate width or fold-bin count"
        raise ValueError(msg)


def export_pulsarx(
    source: str | Path,
    output: str | Path,
    *,
    table: str = "candidates",
    dm: float | None = None,
    pepoch: float | None = None,
    timing_frame: str | None = None,
    snr_min: float | None = None,
    max_candidates: int = 0,
    batch_size: int = 65536,
) -> dict[str, Any]:
    """Atomically write ``id DM accel F0 F1 F2 S/N``, sorted by decreasing S/N.

    ``pepoch`` explicitly supplies the MJD of the native parameter reference
    time; it does not shift coefficients. ``timing_frame`` specifies the
    intended folding frame. Known frame mismatches are rejected.
    """
    output = Path(output)
    if Path(source).resolve() == output.resolve():
        msg = "Input and output paths must differ"
        raise ValueError(msg)
    if max_candidates < 0:
        msg = "max_candidates must be nonnegative"
        raise ValueError(msg)
    if timing_frame not in (None, "topocentric", "barycentric"):
        msg = "timing_frame must be topocentric or barycentric"
        raise ValueError(msg)
    if snr_min is not None:
        snr_min = _finite(snr_min, "snr_min")
    with CandidateFile(source, table=table) as candidates:
        meta = candidates.metadata
        dm_value = dm if dm is not None else meta.get("dm")
        if dm_value is None:
            msg = "DM is unknown; provide dm/--dm explicitly"
            raise ValueError(msg)
        dm_value = _finite(dm_value, "DM")
        if dm_value < 0:
            msg = "DM must be nonnegative"
            raise ValueError(msg)
        epoch = pepoch if pepoch is not None else meta.get("reference_mjd")
        if epoch is not None:
            epoch = _finite(epoch, "Reference MJD")
            if epoch <= 0:
                msg = "Reference MJD must be positive"
                raise ValueError(msg)
        known_frame = (
            ("barycentric" if bool(meta["barycentric"]) else "topocentric")
            if "barycentric" in meta
            else None
        )
        if known_frame is None and timing_frame is None:
            msg = "Timing frame is unknown; provide timing_frame/--timing-frame"
            raise ValueError(msg)
        frame = timing_frame or "topocentric"
        if known_frame is not None and frame != known_frame:
            msg = (
                "Candidate and folding timing frames differ: the file is "
                f"{known_frame}, the export {frame} (frame conversion is not "
                f"supported); pass timing_frame/--timing-frame {known_frame}"
            )
            raise ValueError(msg)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".loki-export-",
            dir=output.parent,
        ) as tmp:
            tmp_path = Path(tmp)
            with sqlite3.connect(tmp_path / "rows.sqlite") as db:
                db.execute("PRAGMA cache_size = -4096")
                db.execute("PRAGMA temp_store = FILE")
                db.execute("CREATE TABLE rows (snr REAL, id TEXT, line TEXT)")
                for batch in candidates.iter_batches(batch_size):
                    _validate_batch(batch)
                    prepared = []
                    for i, snr in enumerate(batch["snr"]):
                        if snr_min is not None and snr < snr_min:
                            continue
                        f0, f1, f2 = (
                            _finite(batch[name][i], name) for name in ("f0", "f1", "f2")
                        )
                        if f0 <= 0:
                            msg = "Candidate frequency must be positive"
                            raise ValueError(msg)
                        if epoch is None and (f1 != 0 or f2 != 0):
                            msg = "Reference epoch is unknown; provide pepoch/--pepoch"
                            raise ValueError(msg)
                        ident = int(batch["id"][i])
                        if ident < 0:
                            msg = "Candidate ID must be nonnegative"
                            raise ValueError(msg)
                        # Text IDs preserve unsigned 64-bit values in SQLite.
                        identity = f"{ident:020d}"
                        accel = -f1 / f0 * C
                        values = (dm_value, accel, f0, f1, f2, float(snr))
                        line = f"{ident} " + " ".join(format(x, ".17g") for x in values)
                        prepared.append((float(snr), identity, line))
                    db.executemany("INSERT INTO rows VALUES (?, ?, ?)", prepared)
                db.commit()
                count = 0
                partial = tmp_path / "pulsarx.cands"
                with partial.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write("#id DM accel F0 F1 F2 S/N\n")
                    handle.write(f"#timing_frame {frame}\n")
                    if epoch is not None:
                        handle.write(
                            f"#Pepoch {epoch:.17g} (pass explicitly with --pepoch)\n",
                        )
                    limit = max_candidates or -1
                    for (line,) in db.execute(
                        "SELECT line FROM rows ORDER BY snr DESC, id ASC LIMIT ?",
                        (limit,),
                    ):
                        handle.write(line + "\n")
                        count += 1
                partial.replace(output)
    return {"count": count, "dm": dm_value, "pepoch": epoch, "timing_frame": frame}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--table", choices=TABLES, default="candidates")
    parser.add_argument("--dm", type=float)
    parser.add_argument("--pepoch", type=float, help="Native parameter reference MJD")
    parser.add_argument("--timing-frame", choices=("topocentric", "barycentric"))
    parser.add_argument("--snr-min", type=float)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = export_pulsarx(**vars(args))
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        parser.exit(2, f"loki-candidates: {exc}\n")
    print(f"Wrote {summary['count']} candidates to {args.output}", file=sys.stderr)  # noqa: T201
    if summary["pepoch"] is not None:
        print(  # noqa: T201
            f"Pass --pepoch {summary['pepoch']:.17g} to psrfold_fil or psrfold_fil2",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
