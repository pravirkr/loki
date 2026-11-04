"""Log level control from Python."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from loki import libloki


def test_set_log_level_round_trips() -> None:
    for level in ("warning", "error", "off", "info"):
        libloki.set_log_level(level)
        assert libloki.get_log_level() == level


def test_set_log_level_rejects_unknown_names() -> None:
    previous = libloki.get_log_level()
    with pytest.raises(ValueError, match="level must be one of"):
        libloki.set_log_level("loud")
    assert libloki.get_log_level() == previous


def _ffa_inputs() -> tuple[np.ndarray, np.ndarray, libloki.configs.FFASearchConfig]:
    n = 1 << 14
    ts_e = np.random.default_rng(1).normal(size=n).astype(np.float32)
    ts_v = np.ones(n, dtype=np.float32)
    cfg = libloki.configs.FFASearchConfig(
        n, 0.001, 32, 1, np.array([[-5.0, 5.0], [9.5, 10.5]]), nthreads=1
    )
    return ts_e, ts_v, cfg


@pytest.mark.parametrize("quiet", [False, True])
def test_compute_ffa_quiet_hides_engine_lines(
    capfd: pytest.CaptureFixture[str], quiet: bool
) -> None:
    libloki.set_log_level("debug")
    libloki.ffa.compute_ffa_scores(*_ffa_inputs(), quiet=quiet)
    assert libloki.get_log_level() == "debug"
    out = capfd.readouterr().out
    assert ("P-FFA" in out) is not quiet


def test_compute_ffa_quiet_leaves_the_log_level_alone() -> None:
    """quiet is per call: the level other threads see never changes."""
    libloki.set_log_level("warning")
    seen: set[str] = set()
    done = threading.Event()

    def watch() -> None:
        while not done.is_set():
            seen.add(libloki.get_log_level())

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        for _ in range(3):
            libloki.ffa.compute_ffa_scores(*_ffa_inputs(), quiet=True)
    finally:
        done.set()
        watcher.join()
    assert seen == {"warning"}
