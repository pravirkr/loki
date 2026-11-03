from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from loki import libloki


@pytest.fixture
def mock_data() -> tuple[np.ndarray, np.ndarray]:
    nsamps = 2**23
    rng = np.random.default_rng()
    ts_e = rng.normal(size=nsamps).astype(np.float32)
    ts_v = rng.normal(size=nsamps).astype(np.float32)
    return ts_e, ts_v


@pytest.fixture
def default_params() -> dict[str, Any]:
    return {
        "nbins": 64,
        "tsamp": 0.000064,
        "nsamps": 2**23,
        "bseg_brute": 1024,
    }


@pytest.fixture(autouse=True)
def _restore_output_settings() -> Iterator[None]:
    """Each test may change loki's log level and progress mode freely."""
    level = libloki.get_log_level()
    mode = libloki.get_progress_mode()
    yield
    libloki.set_log_level(level)
    libloki.set_progress_mode(mode)
