"""
LOKI: Leverage Optimal significance to unveil Keplerian orbIt pulsars.

A high-performance C++ library for pulsar searching with Python bindings.
"""

from importlib import metadata

__version__ = metadata.version(__name__)

# CPU backend (always available)
from . import libloki

# GPU backend (conditionally available)
try:
    from . import libculoki

    __all__ = ["libculoki", "libloki"]
except ImportError:
    libculoki = None
    __all__ = ["libloki"]
