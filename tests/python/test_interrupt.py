"""Ctrl-C and the GIL during long calls (DynamicThresholdScheme.run)."""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from loki import libloki

KW = {
    "ref_ducy": 0.1,
    "nbins": 64,
    "ntrials": 1 << 13,
    "nprobs": 20,
    "snr_final": 8.0,
    "nthresholds": 60,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.0,
    "guess": "window",
}


@pytest.fixture
def safe_sigint() -> Iterator[list[int]]:
    """A SIGINT that lands outside a loki call is recorded, not raised, so a
    slow machine cannot abort the test session."""
    stray: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: stray.append(signum))
    try:
        yield stray
    finally:
        signal.signal(signal.SIGINT, previous)


def _long_scheme() -> libloki.thresholds.DynamicThresholdScheme:
    # About 15 s on 2 threads: far longer than the signal delay below.
    return libloki.thresholds.DynamicThresholdScheme(
        np.full(30, 2.0, dtype=np.float32),
        mode="shared",
        seed=3,
        nthreads=2,
        min_pd=0.3,
        **KW,
    )


def test_ctrl_c_stops_a_run_and_releases_the_gil(safe_sigint: list[int]) -> None:
    scheme = _long_scheme()
    ticks = 0
    done = threading.Event()

    def spin() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.01)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    timer = threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            scheme.run(10, show_progress=False)
        elapsed = time.monotonic() - start
    finally:
        done.set()
        timer.cancel()
        spinner.join()
    assert safe_sigint == []  # the signal reached the run
    # Stopped promptly, and Python threads ran while it was computing.
    assert elapsed < 5.0
    assert ticks > 10
    # The handler in place before the call is back.
    os.kill(os.getpid(), signal.SIGINT)
    assert safe_sigint == [signal.SIGINT]


def test_runs_from_a_worker_thread_leave_sigint_alone() -> None:
    errors: list[BaseException] = []

    def work() -> None:
        try:
            libloki.thresholds.DynamicThresholdScheme(
                np.full(4, 2.0, dtype=np.float32),
                mode="shared",
                seed=3,
                **KW,
            ).run(10, show_progress=False)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    before = signal.getsignal(signal.SIGINT)
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert errors == []
    assert signal.getsignal(signal.SIGINT) is before
