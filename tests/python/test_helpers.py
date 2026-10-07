"""Smoke tests for the pure-Python helpers in src/loki (they need pyloki)."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("pyloki")

from loki import search, sim_ffa  # noqa: E402


def test_ffa_search_forwards_backend() -> None:
    params = inspect.signature(search.ffa_search).parameters
    assert params["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["device"].kind is inspect.Parameter.KEYWORD_ONLY


def test_sim_ffa_uses_current_config_kwargs() -> None:
    source = inspect.getsource(sim_ffa)
    for removed in ("tol_bins=", "use_fft_shifts="):
        assert removed not in source
