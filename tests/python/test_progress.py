"""Progress bar control from Python: modes and DynamicThresholdScheme.run."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

BRANCHING = np.full(6, 2.0, dtype=np.float32)
KW = {
    "ref_ducy": 0.1,
    "nbins": 32,
    "ntrials": 256,
    "nprobs": 10,
    "snr_final": 8.0,
    "nthresholds": 20,
    "ducy_max": 0.3,
    "beam_width": 1.5,
    "wtsp": 1.0,
    "guess": "window",
}


@pytest.fixture
def quiet_logs() -> None:
    """Hide loki's info lines so that stdout holds only progress output."""
    libloki.set_log_level("warning")


def _run_scheme(*, show_progress: bool) -> None:
    scheme = libloki.thresholds.DynamicThresholdScheme(
        BRANCHING, mode="shared", seed=3, nthreads=2, min_pd=0.3, **KW
    )
    scheme.run(10, show_progress=show_progress)


def test_set_progress_mode_round_trips() -> None:
    for mode in ("rich", "plain", "off", "auto"):
        libloki.set_progress_mode(mode)
        assert libloki.get_progress_mode() == mode


def test_set_progress_mode_rejects_unknown_names() -> None:
    previous = libloki.get_progress_mode()
    with pytest.raises(ValueError, match="Unknown progress mode"):
        libloki.set_progress_mode("fancy")
    assert libloki.get_progress_mode() == previous


@pytest.mark.usefixtures("quiet_logs")
@pytest.mark.parametrize("mode", ["auto", "rich", "plain"])
def test_dts_without_progress_writes_nothing(
    mode: str, capfd: pytest.CaptureFixture[str]
) -> None:
    libloki.set_progress_mode(mode)
    _run_scheme(show_progress=False)
    out, _ = capfd.readouterr()
    assert out == ""


@pytest.mark.usefixtures("quiet_logs")
def test_dts_progress_off_mode_writes_nothing(
    capfd: pytest.CaptureFixture[str],
) -> None:
    libloki.set_progress_mode("off")
    _run_scheme(show_progress=True)
    out, _ = capfd.readouterr()
    assert out == ""


@pytest.mark.usefixtures("quiet_logs")
def test_dts_progress_off_a_terminal_is_plain(
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Captured stdout is not a terminal (as in Jupyter or a log file): the
    # auto mode prints a few plain lines, without control sequences.
    libloki.set_progress_mode("auto")
    _run_scheme(show_progress=True)
    out, _ = capfd.readouterr()
    assert "\x1b" not in out
    assert "\r" not in out
    lines = out.splitlines()
    assert 1 <= len(lines) <= 4
    assert all(line.startswith("Computing scheme (shared)") for line in lines)
    assert "100%" in lines[-1]
