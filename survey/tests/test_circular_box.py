"""A circular-orbit (five-parameter) EP box finds, keeps and folds an orbit."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from loki import libloki
from loki_survey.cli import main

from .conftest import write_site

if TYPE_CHECKING:
    from pathlib import Path

NSAMPS = 1 << 16
DT = 64e-6
T_OBS = NSAMPS * DT
SPIN_FREQ = 142.3
# loki takes a leaf for an orbit (circular_snap) only if its snap is measured,
# more than twice (propagator_significance) its step: 2^3 eta / nbins turns
# of fourth-order delay over half the span. An orbit's fourth-order delay
# there is x_orb (pi T_OBS / P_ORB)^4 sin(psi) / 24, so the span must cover
# much of the orbit: this one turns 5 radians in T_OBS, a binary scaled down
# to the few seconds a test can search.
ORBIT_RADIANS = 5.0
P_ORB = 2.0 * math.pi * T_OBS / ORBIT_RADIANS
# Kepler's law sets x_orb from the masses (solar masses): 5.7 light-ms, a
# fourth-order delay of 1.1 turns at SPIN_FREQ, 4.5 planned snap steps.
M_C = 0.6
M_P = 1.4
# Orbital phase at mid-span, away from 0 and pi / 2 so that acceleration and
# snap (sin psi) and jerk and crackle (cos psi) are all non-zero.
PSI = 1.0
# The box's prior: a little shorter and heavier than the injected orbit, so
# each of its derivatives lies inside the box.
P_ORB_MIN_S = 4.8
M_C_MAX = 0.8
INJECTED_SNR = 15.0
# The injected pulsar must come back at this S/N or more.
RECOVER_SNR_MIN = 10.0
# Two frequency resolutions 1 / T_obs.
FREQ_TOL = 2.0 / T_OBS
# The fold scores the leaf's phase exactly; the search summed segment folds
# whose phases were good to eta / nbins turns. Over seeds 1 to 10 the two
# agreed within 6%.
FOLD_SNR_RTOL = 0.1
SEED = 1
UNIT = "sim_DM10.00__mid__b0__ep_circular"
MODELS = ("circular_snap", "circular_crackle")


def survey_toml() -> str:
    """One five-parameter EP box, with loki's own reduction of EP leaves."""
    # Without the reduction, the candidate step clusters every leaf above
    # snr_min (about 10^5 here) and takes minutes.
    ep = libloki.configs.CandidateOptions.ep_defaults()
    return f"""
[survey]
name = "circular"
[data]
series = ["data/*.dat"]
[search]
snr_min = 7.0
min_pd = 0.5
[[search.region]]
name = "mid"
f_hz = [140.0, 145.0]
nbins = 32
[[search.box]]
name = "ep_circular"
kind = "ep"
order = "crackle"
orbit = {{ p_orb_min_s = {P_ORB_MIN_S}, m_c_max = {M_C_MAX} }}
segments = 8
bseg_brute = 1024
ref_segs = [4]
[candidates]
max_per_cell = {ep.max_per_cell}
cell_bins = {ep.cell_bins}
dominance_kappa = {ep.dominance_kappa}
[output]
raw_fraction = 1.0
"""


def simulate(path: Path) -> None:
    mod = libloki.simulation.ModulatorParams()
    mod.p_orb = P_ORB
    mod.psi = PSI
    mod.m_c = M_C
    mod.m_p = M_P
    cfg = libloki.simulation.PulseSignalConfig(
        1.0 / SPIN_FREQ,
        DT,
        NSAMPS,
        INJECTED_SNR,
        0.1,
        "circular",
        mod,
        None,
        SEED,
    )
    series = cfg.generate("gaussian", 0.5)
    libloki.io.TimeSeries(
        np.asarray(series.ts_e, dtype=np.float32),
        np.asarray(series.ts_v, dtype=np.float32),
        DT,
    ).write(str(path))


def apparent_freq() -> float:
    """Return the Doppler-shifted frequency at mid-span, which EP reports."""
    orbit = libloki.simulation.CircularModulator(P_ORB, PSI, None, M_C, M_P)
    return float(orbit.to_derivatives_gauge(SPIN_FREQ).freq)


def test_a_circular_box_folds_its_candidate_with_its_orbit(
    tmp_path: Path,
    unit_python: str,
) -> None:
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    simulate(run / "data" / "sim_DM10.00.dat")
    (run / "survey.toml").write_text(survey_toml())
    write_site(run, unit_python)

    assert main(["run", str(run)]) == 0

    truth = apparent_freq()
    with h5py.File(run / "out" / f"{UNIT}.h5", "r") as handle:
        params = np.asarray(handle["candidates/param_sets"][:])
        snr = np.asarray(handle["candidates/snr"][:])
        leaves = np.asarray(handle["survey/leaves"][:])
    # Rows (crackle, snap, jerk, accel, freq).
    assert params.shape[1] == 5
    near = np.nonzero(np.abs(params[:, -1] - truth) <= FREQ_TOL)[0]
    assert near.size, f"no candidate within {FREQ_TOL:.3f} Hz of {truth:.3f} Hz"
    best = int(near[np.argmax(snr[near])])
    assert snr[best] >= RECOVER_SNR_MIN, f"best S/N {snr[best]:.2f}"
    # Each candidate keeps its five-parameter leaf, (value, error) rows.
    assert leaves.shape == (snr.size, 7, 2)
    np.testing.assert_array_equal(leaves[best, :5, 0], params[best])

    # Folded with the orbit its leaf describes: the pulsar keeps its search
    # S/N and the 1.002 f control loses it.
    groups = json.loads((run / "folds" / "summary.json").read_text())["groups"]
    folded = max(
        (g for g in groups if g["unit"] == UNIT and abs(g["f0"] - truth) <= FREQ_TOL),
        key=lambda g: g["snr"],
    )
    assert folded["model"] in MODELS, folded["model"]
    assert folded["snr"] >= RECOVER_SNR_MIN
    assert folded["snr"] == pytest.approx(folded["search_snr"], rel=FOLD_SNR_RTOL)
    assert folded["snr_control"] < folded["snr"] / 2
