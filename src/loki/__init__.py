"""
LOKI: Leverage Optimal significance to unveil Keplerian orbIt pulsars.

A high-performance C++ library for pulsar searching with Python bindings.
"""

from importlib import metadata

__version__ = metadata.version(__name__)

from . import libloki

__all__ = ["libloki"]
