"""The status database: every unit's state and every attempt to run it.

The files in ``out/`` are the truth: a unit with a summary JSON is done. The
database records what the files cannot, such as attempts, the jobs that ran
them, failure classes, memory reservations and excluded hosts, and is brought
up to date from the files on every read.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loki_survey.plan import Unit
    from loki_survey.rundir import RunDir

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    memory_gb REAL,
    failure TEXT,
    updated REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    unit TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    started REAL,
    ended REAL,
    exit_code INTEGER,
    failure TEXT,
    detail TEXT,
    peak_rss_gb REAL,
    memory_gb REAL,
    log TEXT,
    PRIMARY KEY (unit, attempt)
);
CREATE TABLE IF NOT EXISTS arrays (
    name TEXT PRIMARY KEY,
    job TEXT,
    n INTEGER NOT NULL,
    submitted REAL
);
CREATE TABLE IF NOT EXISTS hosts (
    host TEXT PRIMARY KEY,
    failures INTEGER NOT NULL DEFAULT 0,
    blocked INTEGER NOT NULL DEFAULT 0,
    detail TEXT,
    updated REAL
);
"""
# Columns added for cluster executors; a database made before them gains them.
ADDED_COLUMNS = (
    ("attempts", "phase", "TEXT NOT NULL DEFAULT 'ended'"),
    ("attempts", "array_name", "TEXT"),
    ("attempts", "job", "TEXT"),
    ("attempts", "job_index", "INTEGER"),
    ("attempts", "host", "TEXT"),
    ("attempts", "running_since", "REAL"),
    ("attempts", "began", "INTEGER NOT NULL DEFAULT 0"),
    ("attempts", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
)
STATES = ("pending", "queued", "running", "done", "failed")
LIVE_STATES = ("queued", "running")
# Failure classes, each handled its own way by the executors.
OOM = "oom"
CRASHED = "crashed"
INTERRUPTED = "interrupted"
TOO_LARGE = "too_large"
PREEMPTED = "preempted"
KILLED = "killed"
LOST = "lost"
HOST = "host"
STALLED = "stalled"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
# The job id of an array whose submission was given up.
NOT_SUBMITTED = "-"
# Classes resubmitted as they were: the unit itself did nothing wrong.
REQUEUED = (PREEMPTED, KILLED, LOST, HOST, STALLED)


@dataclass(frozen=True)
class UnitState:
    id: str
    state: str
    attempts: int
    memory_gb: float | None
    failure: str | None


@dataclass(frozen=True)
class Attempt:
    """An attempt that is queued on or running in a cluster."""

    unit: str
    attempt: int
    phase: str
    array: str | None
    job: str | None
    index: int | None
    memory_gb: float
    started: float
    running_since: float | None
    host: str | None
    began: bool
    cancel_requested: bool


class StatusDB:
    def __init__(self, run: RunDir) -> None:
        self.run = run
        self.conn = sqlite3.connect(run.status_db)
        self.conn.executescript(SCHEMA)
        self._add_columns()

    def _add_columns(self) -> None:
        with self.conn:
            for table, column, kind in ADDED_COLUMNS:
                have = {
                    row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in have:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {kind}",
                    )

    def close(self) -> None:
        self.conn.close()

    def sync(self, units: list[Unit], default_memory_gb: float) -> None:
        """Add new units as pending and mark every unit with a summary done."""
        now = time.time()
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO units (id, state, memory_gb, updated) "
                "VALUES (?, 'pending', ?, ?)",
                [(u.id, default_memory_gb, now) for u in units],
            )
            for unit in units:
                if self.run.unit_summary(unit.id).is_file():
                    self.conn.execute(
                        "UPDATE units SET state = 'done', failure = NULL, updated = ? "
                        "WHERE id = ? AND state != 'done'",
                        (now, unit.id),
                    )

    def get(self, unit_id: str) -> UnitState:
        row = self.conn.execute(
            "SELECT id, state, attempts, memory_gb, failure FROM units WHERE id = ?",
            (unit_id,),
        ).fetchone()
        return UnitState(*row)

    def states(self, unit_ids: list[str]) -> list[UnitState]:
        return [self.get(u) for u in unit_ids]

    def _new_attempt(
        self,
        unit_id: str,
        memory_gb: float,
        log: str,
        *,
        phase: str,
        array: str | None = None,
        index: int | None = None,
    ) -> int:
        attempt = self.get(unit_id).attempts + 1
        now = time.time()
        self.conn.execute(
            "UPDATE units SET state = ?, attempts = ?, memory_gb = ?, "
            "updated = ? WHERE id = ?",
            (phase, attempt, memory_gb, now, unit_id),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO attempts "
            "(unit, attempt, started, memory_gb, log, phase, array_name, job_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (unit_id, attempt, now, memory_gb, log, phase, array, index),
        )
        return attempt

    def start(self, unit_id: str, memory_gb: float, log: str) -> int:
        """Record a unit started on this machine; return its attempt number."""
        with self.conn:
            return self._new_attempt(unit_id, memory_gb, log, phase="running")

    def queue(
        self,
        unit_id: str,
        memory_gb: float,
        log: str,
        *,
        array: str,
        index: int,
    ) -> int:
        """Record a unit placed in a cluster array; return its attempt number."""
        with self.conn:
            return self._new_attempt(
                unit_id,
                memory_gb,
                log,
                phase="queued",
                array=array,
                index=index,
            )

    def add_array(self, name: str, n: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO arrays (name, n) VALUES (?, ?)",
                (name, n),
            )

    def set_array_job(self, name: str, job: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE arrays SET job = ?, submitted = ? WHERE name = ?",
                (job, time.time(), name),
            )
            self.conn.execute(
                "UPDATE attempts SET job = ? WHERE array_name = ?",
                (job, name),
            )

    def unsubmitted_arrays(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM arrays WHERE job IS NULL")
        return [r[0] for r in rows.fetchall()]

    def drop_array(self, name: str, detail: str) -> None:
        """Return the units of an array that never reached the scheduler."""
        now = time.time()
        with self.conn:
            units = [
                r[0]
                for r in self.conn.execute(
                    "SELECT unit FROM attempts "
                    "WHERE array_name = ? AND phase != 'ended'",
                    (name,),
                )
            ]
            self.conn.execute(
                "UPDATE attempts SET phase = 'ended', ended = ?, detail = ? "
                "WHERE array_name = ? AND phase != 'ended'",
                (now, detail, name),
            )
            self.conn.executemany(
                "UPDATE units SET state = 'pending', updated = ? WHERE id = ?",
                [(now, u) for u in units],
            )
            self.conn.execute(
                "UPDATE arrays SET job = ? WHERE name = ?",
                (NOT_SUBMITTED, name),
            )

    def next_array_seq(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM arrays").fetchone()
        return row[0] + 1

    def live(self) -> list[Attempt]:
        """Every attempt queued on or running in a cluster."""
        rows = self.conn.execute(
            "SELECT unit, attempt, phase, array_name, job, job_index, memory_gb, "
            "started, running_since, host, began, cancel_requested FROM attempts "
            "WHERE phase IN ('queued', 'running') ORDER BY unit, attempt",
        )
        return [
            Attempt(
                *row[:10],
                began=bool(row[10]),
                cancel_requested=bool(row[11]),
            )
            for row in rows.fetchall()
        ]

    def mark_running(self, unit_id: str, attempt: int, host: str | None) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET phase = 'running', "
                "running_since = COALESCE(running_since, ?), "
                "host = COALESCE(?, host) WHERE unit = ? AND attempt = ?",
                (now, host, unit_id, attempt),
            )
            self.conn.execute(
                "UPDATE units SET state = 'running', updated = ? "
                "WHERE id = ? AND state = 'queued'",
                (now, unit_id),
            )

    def restart_stall_clock(self, unit_id: str, attempt: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET running_since = ? WHERE unit = ? AND attempt = ?",
                (time.time(), unit_id, attempt),
            )

    def mark_began(self, unit_id: str, attempt: int, host: str | None) -> None:
        """Record that ``run-unit`` itself started (so the host could run it)."""
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET began = 1, host = COALESCE(?, host) "
                "WHERE unit = ? AND attempt = ?",
                (host, unit_id, attempt),
            )

    def request_cancel(self, unit_id: str, attempt: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET cancel_requested = 1 "
                "WHERE unit = ? AND attempt = ?",
                (unit_id, attempt),
            )

    def finish(
        self,
        unit_id: str,
        attempt: int,
        *,
        state: str,
        exit_code: int | None,
        failure: str | None = None,
        detail: str | None = None,
        peak_rss_gb: float | None = None,
        memory_gb: float | None = None,
        host: str | None = None,
    ) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET phase = 'ended', ended = ?, exit_code = ?, "
                "failure = ?, detail = ?, peak_rss_gb = ?, host = COALESCE(?, host) "
                "WHERE unit = ? AND attempt = ?",
                (now, exit_code, failure, detail, peak_rss_gb, host, unit_id, attempt),
            )
            self.conn.execute(
                "UPDATE units SET state = ?, failure = ?, "
                "memory_gb = COALESCE(?, memory_gb), updated = ? WHERE id = ?",
                (state, failure, memory_gb, now, unit_id),
            )

    def retry_failed(self, unit_ids: list[str]) -> int:
        """Return failed units to pending, keeping their memory reservation."""
        marks = ", ".join("?" for _ in unit_ids)
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE units SET state = 'pending', updated = ? "  # noqa: S608
                f"WHERE state = 'failed' AND id IN ({marks})",
                (time.time(), *unit_ids),
            )
        return cur.rowcount

    def reset_running(self) -> int:
        """Mark units left 'running' by a local driver that died as pending again."""
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET phase = 'ended', ended = ?, failure = ? "
                "WHERE phase = 'running' AND job IS NULL AND array_name IS NULL",
                (time.time(), INTERRUPTED),
            )
            cur = self.conn.execute(
                "UPDATE units SET state = 'pending', updated = ? "
                "WHERE state = 'running' AND id NOT IN "
                "(SELECT unit FROM attempts WHERE phase != 'ended')",
                (time.time(),),
            )
        return cur.rowcount

    def count_failures(self, unit_id: str, classes: tuple[str, ...]) -> int:
        marks = ", ".join("?" for _ in classes)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM attempts WHERE unit = ? AND failure IN ({marks})",  # noqa: S608
            (unit_id, *classes),
        ).fetchone()
        return row[0]

    def host_failed(self, host: str, detail: str) -> int:
        """Count one more failure that points at ``host``; return its total."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO hosts (host, failures, detail, updated) "
                "VALUES (?, 1, ?, ?) "
                "ON CONFLICT(host) DO UPDATE SET failures = failures + 1, "
                "detail = excluded.detail, updated = excluded.updated",
                (host, detail, time.time()),
            )
        row = self.conn.execute(
            "SELECT failures FROM hosts WHERE host = ?",
            (host,),
        ).fetchone()
        return row[0]

    def exclude_host(self, host: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE hosts SET blocked = 1, updated = ? WHERE host = ?",
                (time.time(), host),
            )

    def excluded_hosts(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT host FROM hosts WHERE blocked = 1 ORDER BY host",
        )
        return [r[0] for r in rows.fetchall()]

    def hosts_that_ran_units(self) -> set[str]:
        """Hosts where ``run-unit`` started at least once."""
        rows = self.conn.execute(
            "SELECT DISTINCT host FROM attempts WHERE host IS NOT NULL AND began = 1",
        )
        return {r[0] for r in rows.fetchall()}

    def counts(self) -> Counter[str]:
        rows = self.conn.execute("SELECT state, COUNT(*) FROM units GROUP BY state")
        return Counter(dict(rows.fetchall()))

    def failures(self) -> list[tuple[str, str, int, str | None]]:
        rows = self.conn.execute(
            "SELECT u.id, u.failure, u.attempts, a.detail FROM units u "
            "LEFT JOIN attempts a ON a.unit = u.id AND a.attempt = u.attempts "
            "WHERE u.state = 'failed' ORDER BY u.id",
        )
        return rows.fetchall()
