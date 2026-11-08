"""The status database: every unit's state and every attempt to run it.

The files in ``out/`` are the truth: a unit with a summary JSON is done. The
database records what the files cannot, such as attempts, failure classes
and memory reservations, and is brought up to date from the files on every
read.
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
"""
STATES = ("pending", "running", "done", "failed")
# Failure classes, each handled its own way by the executor.
OOM = "oom"
CRASHED = "crashed"
INTERRUPTED = "interrupted"
TOO_LARGE = "too_large"


@dataclass(frozen=True)
class UnitState:
    id: str
    state: str
    attempts: int
    memory_gb: float | None
    failure: str | None


class StatusDB:
    def __init__(self, run: RunDir) -> None:
        self.run = run
        self.conn = sqlite3.connect(run.status_db)
        self.conn.executescript(SCHEMA)

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

    def start(self, unit_id: str, memory_gb: float, log: str) -> int:
        with self.conn:
            attempt = self.get(unit_id).attempts + 1
            self.conn.execute(
                "UPDATE units SET state = 'running', attempts = ?, memory_gb = ?, "
                "updated = ? WHERE id = ?",
                (attempt, memory_gb, time.time(), unit_id),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO attempts "
                "(unit, attempt, started, memory_gb, log) VALUES (?, ?, ?, ?, ?)",
                (unit_id, attempt, time.time(), memory_gb, log),
            )
        return attempt

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
    ) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET ended = ?, exit_code = ?, failure = ?, "
                "detail = ?, peak_rss_gb = ? WHERE unit = ? AND attempt = ?",
                (now, exit_code, failure, detail, peak_rss_gb, unit_id, attempt),
            )
            self.conn.execute(
                "UPDATE units SET state = ?, failure = ?, "
                "memory_gb = COALESCE(?, memory_gb), updated = ? WHERE id = ?",
                (state, failure, memory_gb, now, unit_id),
            )

    def reset_running(self) -> int:
        """Mark units left 'running' by a driver that died as pending again."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE units SET state = 'pending', updated = ? "
                "WHERE state = 'running'",
                (time.time(),),
            )
        return cur.rowcount

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
