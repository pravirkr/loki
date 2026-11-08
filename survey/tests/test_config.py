from __future__ import annotations

import tomllib
from importlib import resources

import pytest

from loki_survey.config import (
    ConfigError,
    parse_site,
    parse_survey,
    science_sha256,
)

SURVEY = """
[survey]
name = "t"
[data]
series = "data/*.dat"
[search]
snr_min = 7.0
[[search.region]]
name = "high"
f_hz = [100.0, 400.0]
nbins = 32
bands = 2
[[search.box]]
name = "jerk"
kind = "ep"
order = "jerk"
limits = { jerk = [-1.0, 1.0], accel = [-50.0, 50.0] }
"""


def survey(text: str = SURVEY) -> object:
    return parse_survey(tomllib.loads(text))


def test_templates_parse() -> None:
    files = resources.files("loki_survey.templates")
    parse_survey(tomllib.loads(files.joinpath("survey.toml").read_text()))
    parse_site(tomllib.loads(files.joinpath("site.toml").read_text()))


def test_defaults_are_loki_defaults() -> None:
    s = survey()
    assert s.eta == 1.0
    assert s.min_pd == 0.1
    assert s.region("high").ducy_max == 0.2
    assert s.region("high").downsample == 1
    assert s.box("jerk").derivatives == ("jerk", "accel")
    assert s.box("jerk").nparams == 3


def test_hash_ignores_formatting_but_not_science() -> None:
    a = science_sha256(survey())
    b = science_sha256(survey(SURVEY.replace("snr_min = 7.0", "snr_min   =   7.0")))
    c = science_sha256(survey(SURVEY.replace("snr_min = 7.0", "snr_min = 7.5")))
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("snr_min = 7.0", "", "'snr_min' is required"),
        ("nbins = 32", "nbins = 0", "positive integer"),
        ("f_hz = [100.0, 400.0]", "f_hz = [400.0, 100.0]", "min < max"),
        ('kind = "ep"', 'kind = "fft"', "kind"),
        ('order = "jerk"', 'order = "crackle"', "EP boxes support"),
        ("jerk = [-1.0, 1.0], ", "", "'jerk' is required"),
        ("snr_min = 7.0", "snr_min = 7.0\nsnr_mn = 6.0", "unknown keys"),
    ],
)
def test_invalid_survey(old: str, new: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        survey(SURVEY.replace(old, new))


def test_limits_or_orbit_not_both() -> None:
    text = SURVEY.replace(
        "limits = {",
        "orbit = { p_orb_min_s = 3600.0 }\nlimits = {",
    )
    with pytest.raises(ConfigError, match="not both"):
        survey(text)


def test_site_unit_must_fit() -> None:
    doc = tomllib.loads(
        """
[executor]
kind = "local"
max_memory_gb = 4.0
max_threads = 4
[unit]
memory_gb = 8.0
overhead_gb = 1.0
threads = 2
""",
    )
    with pytest.raises(ConfigError, match="exceeds"):
        parse_site(doc)
