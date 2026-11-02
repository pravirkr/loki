"""Log level control from Python."""

from __future__ import annotations

import pytest

from loki import libloki


def test_set_log_level_round_trips() -> None:
    previous = libloki.get_log_level()
    try:
        for level in ("warning", "error", "off", "info"):
            libloki.set_log_level(level)
            assert libloki.get_log_level() == level
    finally:
        libloki.set_log_level(previous)


def test_set_log_level_rejects_unknown_names() -> None:
    previous = libloki.get_log_level()
    with pytest.raises(ValueError, match="level must be one of"):
        libloki.set_log_level("loud")
    assert libloki.get_log_level() == previous
