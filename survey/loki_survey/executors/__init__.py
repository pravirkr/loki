"""Executors run units: on this machine (``local``), on LSF or on Slurm."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loki_survey.config import Resolved
    from loki_survey.executors.batch import BatchExecutor
    from loki_survey.rundir import RunDir
    from loki_survey.status import StatusDB


def batch_executor(run: RunDir, resolved: Resolved, db: StatusDB) -> BatchExecutor:
    """Return the cluster executor for the site's scheduler."""
    from loki_survey.executors.batch import BatchExecutor  # noqa: PLC0415
    from loki_survey.executors.lsf import LsfScheduler  # noqa: PLC0415
    from loki_survey.executors.slurm import SlurmScheduler  # noqa: PLC0415

    kinds = {"lsf": LsfScheduler, "slurm": SlurmScheduler}
    scheduler = kinds[resolved.site.executor](run, resolved.site)
    return BatchExecutor(run, resolved, db, scheduler)
