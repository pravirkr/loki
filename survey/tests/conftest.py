from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Units run as subprocesses; this is the Python they use. It defaults to the
# Python running the tests, which must import the same loki.
UNIT_PYTHON = os.environ.get("LOKI_SURVEY_TEST_PYTHON", sys.executable)


@pytest.fixture
def unit_python() -> str:
    return UNIT_PYTHON


def write_site(
    run: Path,
    python: str,
    *,
    max_memory_gb: float = 8.0,
    memory_gb: float = 4.0,
    overhead_gb: float = 1.0,
    oom_retries: int = 1,
) -> None:
    (run / "site.toml").write_text(
        f"""
[executor]
kind = "local"
max_memory_gb = {max_memory_gb}
max_threads = 4
python = "{python}"
poll_s = 0.2

[unit]
memory_gb = {memory_gb}
overhead_gb = {overhead_gb}
threads = 2

[retry]
oom_retries = {oom_retries}
oom_factor = 1.5
""",
    )
