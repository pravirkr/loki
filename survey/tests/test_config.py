from __future__ import annotations

import tomllib
from importlib import resources
from typing import TYPE_CHECKING

import pytest

from loki_survey.config import (
    ConfigError,
    Resolved,
    analysis_sha256,
    load_resolved,
    parse_analysis,
    parse_site,
    parse_survey,
    resolve,
    resolved_dict,
    science_sha256,
)
from loki_survey.plan import candidate_options
from loki_survey.rundir import RunDir, atomic_write_json

if TYPE_CHECKING:
    from pathlib import Path

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
        ('order = "jerk"', 'order = "freq"', "EP boxes support"),
        ('kind = "ep"', 'kind = "ep"\nsegments = 6', "power of two"),
        ('kind = "ep"', 'kind = "ffa"\nsegments = 8', "EP settings"),
        ("jerk = [-1.0, 1.0], ", "", "'jerk' is required"),
        ("snr_min = 7.0", "snr_min = 7.0\nsnr_mn = 6.0", "unknown keys"),
        ('name = "t"', 'name = "t"\n[candidates]\nmemory_gb = 2.0', "set by loki"),
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


SITE = """
[executor]
kind = "local"
max_memory_gb = 8.0
max_threads = 4
[unit]
memory_gb = 4.0
overhead_gb = 1.0
threads = 2
"""


def test_candidate_memory_is_reserved_within_the_loki_budget() -> None:
    site = parse_site(tomllib.loads(SITE + "candidate_memory_gb = 2.0\n"))
    assert site.candidate_memory_gb == 2.0
    assert parse_site(tomllib.loads(SITE)).candidate_memory_gb is None
    with pytest.raises(ConfigError, match="candidate_memory_gb must be below"):
        parse_site(tomllib.loads(SITE + "candidate_memory_gb = 3.0\n"))


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("max_per_cel = 1", "no candidate option 'max_per_cel'"),
        ("max_per_cell = 1", "cell_log_width"),
        ('max_per_cell = "one"', r"\[candidates\]\.max_per_cell"),
    ],
)
def test_candidate_options_are_loki_options(table: str, message: str) -> None:
    s = survey(SURVEY + f"[candidates]\n{table}\n")
    resolved = Resolved(
        survey=s,
        site=parse_site(tomllib.loads(SITE)),
        science_sha256=science_sha256(s),
    )
    with pytest.raises(ConfigError, match=message):
        candidate_options(resolved)


def test_candidate_options_take_both_files() -> None:
    s = survey(SURVEY + "[candidates]\nmax_per_cell = 1\ncell_log_width = 2e-4\n")
    site = parse_site(tomllib.loads(SITE + "candidate_memory_gb = 2.0\n"))
    options = candidate_options(Resolved(s, site, science_sha256(s)))
    assert (options.max_per_cell, options.cell_log_width) == (1, 2e-4)
    assert options.memory_gb == 2.0
    # The candidate step is science: it changes what a unit writes.
    assert science_sha256(s) != science_sha256(survey())


def test_analysis_has_its_own_hash() -> None:
    plain = tomllib.loads(SURVEY)
    tuned = tomllib.loads(SURVEY + "[collect]\nmax_candidates = 50\n")
    assert science_sha256(parse_survey(plain)) == science_sha256(parse_survey(tuned))
    assert analysis_sha256(parse_analysis(plain)) != analysis_sha256(
        parse_analysis(tuned),
    )
    assert parse_analysis(tuned).collect.max_candidates == 50


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[collect]\nradius = 2.0\n", "unknown keys"),
        ("[collect]\nremove_harmonics = true\nharmonic_max = 0\n", "harmonic_max"),
        ("[collect]\ndrop_undispersed = 1\n", "true or false"),
        ("[evaluate]\nline_fap = 1.5\n", "probability"),
        ("[evaluate]\ndoppler = -1e-4\n", ">= 0"),
    ],
)
def test_invalid_analysis(extra: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_analysis(tomllib.loads(SURVEY + extra))


def test_resolved_keeps_the_analysis(tmp_path: Path) -> None:
    run = RunDir.at(tmp_path)
    (tmp_path / "survey.toml").write_text(
        SURVEY + '[evaluate]\ncatalogue = "cat.csv"\n[collect]\nsnr_min = 8.0\n',
    )
    site = resources.files("loki_survey.templates").joinpath("site.toml").read_text()
    (tmp_path / "site.toml").write_text(site)
    resolved = resolve(run)
    atomic_write_json(run.resolved_json, resolved_dict(resolved, {}))
    again = load_resolved(run)
    assert again.analysis == resolved.analysis
    assert again.analysis.evaluate.catalogue == "cat.csv"
    assert again.analysis_sha256 == resolved.analysis_sha256
