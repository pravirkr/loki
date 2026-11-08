"""loki-survey: run a loki pulsar survey end to end.

A survey goes from dedispersed time series to a candidate list in stages
(``check``, ``prep``, ``plan``, ``submit``, ``status``). Every stage reads and
writes one run directory, and rerunning a stage does only what is missing.
"""

__version__ = "0.1.0"
