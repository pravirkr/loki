"""loki's version: the project version, known with no install."""

import re
import tomllib
from pathlib import Path

import loki
from loki import libloki

ROOT = Path(__file__).resolve().parents[2]


def test_version_is_the_project_version() -> None:
    # The project version as pip's metadata reads it (pyproject.toml).
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    regex = pyproject["tool"]["scikit-build"]["metadata"]["version"]["regex"]
    match = re.search(regex, (ROOT / "CMakeLists.txt").read_text())
    assert match is not None
    assert libloki.__version__ == match["value"]
    assert loki.__version__ == libloki.__version__
