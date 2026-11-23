"""
LOKI: Leverage Optimal significance to unveil Keplerian orbIt pulsars.

A high-performance C++ library for pulsar searching with Python bindings.
"""

from . import libloki

# The version this module was built as, the project version in CMakeLists.txt
# that pip's metadata also reads: known with no install (PYTHONPATH=src).
__version__: str = libloki.__version__

__all__ = ["libloki"]
