"""The run directory: its layout, atomic writes and content hashes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

HASH_CHUNK_BYTES = 1 << 20


class DriverBusyError(RuntimeError):
    """Another loki-survey process is submitting or watching this run."""


@dataclass(frozen=True)
class RunDir:
    """Every path a stage reads or writes, relative to one root."""

    root: Path

    @classmethod
    def at(cls, root: str | os.PathLike[str]) -> RunDir:
        return cls(Path(root).resolve())

    @property
    def survey_toml(self) -> Path:
        return self.root / "survey.toml"

    @property
    def site_toml(self) -> Path:
        return self.root / "site.toml"

    @property
    def resolved_json(self) -> Path:
        return self.root / "resolved.json"

    @property
    def prep_dir(self) -> Path:
        return self.root / "prep"

    @property
    def plans_dir(self) -> Path:
        return self.root / "plans"

    @property
    def units_jsonl(self) -> Path:
        return self.root / "units.jsonl"

    @property
    def splits_jsonl(self) -> Path:
        """Units split after hitting their time limit, and their parts."""
        return self.root / "splits.jsonl"

    @property
    def collect_dir(self) -> Path:
        return self.root / "collect"

    @property
    def evaluate_dir(self) -> Path:
        return self.root / "evaluate"

    @property
    def status_db(self) -> Path:
        return self.root / "status.sqlite"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def raw_dir(self) -> Path:
        return self.root / "out" / "raw"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def sched_logs_dir(self) -> Path:
        """Where LSF and Slurm write each job's own output and footer."""
        return self.root / "logs" / "sched"

    @property
    def jobs_dir(self) -> Path:
        """Job scripts and unit lists of the arrays submitted to a cluster."""
        return self.root / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def driver_lock(self) -> Path:
        return self.root / ".driver.lock"

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "quarantine"

    def working_dirs(self) -> tuple[Path, ...]:
        """Return the directories units write to while they run."""
        return (
            self.out_dir,
            self.raw_dir,
            self.logs_dir,
            self.sched_logs_dir,
            self.tmp_dir,
        )

    def ensure(self) -> None:
        """Create the layout and check that every working directory is writable.

        A directory frozen read-only by hand (``chmod 444 *``) makes every unit
        fail at its first write; refuse before anything is submitted.
        """
        for path in (
            self.prep_dir,
            self.plans_dir,
            self.quarantine_dir,
            self.jobs_dir,
            *self.working_dirs(),
        ):
            path.mkdir(parents=True, exist_ok=True)
        blocked = [
            str(p) for p in self.working_dirs() if not os.access(p, os.W_OK | os.X_OK)
        ]
        if blocked:
            msg = f"run directories are not writable: {', '.join(blocked)}"
            raise PermissionError(msg)

    @contextmanager
    def driving(self) -> Iterator[None]:
        """Hold the run's driver lock: one process submits and resubmits."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.driver_lock.open("a+") as handle:
            locked = True
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read().strip() or "unknown"
                msg = (
                    f"another loki-survey process ({holder}) is driving "
                    f"{self.root}; stop it first, or use 'status'"
                )
                raise DriverBusyError(msg) from None
            except OSError:
                # Some network file systems refuse locks altogether.
                locked = False
                print(
                    "warning: cannot lock the run directory; "
                    "run one 'submit' or 'watch' at a time",
                )
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid {os.getpid()} on {os.uname().nodename}\n")
            handle.flush()
            try:
                yield
            finally:
                if locked:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def unit_output(self, unit_id: str) -> Path:
        return self.out_dir / f"{unit_id}.h5"

    def unit_summary(self, unit_id: str) -> Path:
        return self.out_dir / f"{unit_id}.json"

    def unit_log(self, unit_id: str, attempt: int) -> Path:
        return self.logs_dir / f"{unit_id}.{attempt}.log"

    def unit_started(self, unit_id: str, attempt: int) -> Path:
        """Written by ``run-unit`` before anything else: where and when it began."""
        return self.logs_dir / f"{unit_id}.{attempt}.started"

    def unit_alive(self, unit_id: str, attempt: int) -> Path:
        """Touched by a cluster job while it runs; its age shows a stall."""
        return self.logs_dir / f"{unit_id}.{attempt}.alive"

    def unit_tmp(self, unit_id: str, attempt: int) -> Path:
        return self.tmp_dir / f"{unit_id}.{attempt}"


def canonical_json(obj: Any) -> str:  # noqa: ANN401
    """JSON with sorted keys and no whitespace, so equal content hashes equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``path`` so that readers see the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: Path, obj: Any) -> None:  # noqa: ANN401
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text())
