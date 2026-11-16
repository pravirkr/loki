"""Refolding a series at a leaf, against a numpy port of the reference."""

from __future__ import annotations

import numpy as np
import pytest

from loki import libloki

C = 299792458.0


def taylor_leaf(accel: float, freq: float) -> np.ndarray:
    """An accel-search leaf: (accel, freq, d0, f0) rows of (value, error)."""
    return np.array([[accel, 0.0], [freq, 0.0], [0.0, 0.0], [freq, 0.0]])


def boxcar_snr(e: np.ndarray, v: np.ndarray, wmax: int) -> tuple[float, int]:
    n = len(e)
    ce = np.concatenate([[0.0], np.cumsum(np.concatenate([e, e]))])
    cv = np.concatenate([[0.0], np.cumsum(np.concatenate([v, v]))])
    best = (-np.inf, 0)
    for w in range(1, wmax + 1):
        se = ce[w : w + n] - ce[:n]
        sv = cv[w : w + n] - cv[:n]
        s = np.where(sv > 0, se / np.sqrt(np.where(sv > 0, sv, 1.0)), -np.inf)
        if s.max() > best[0]:
            best = (float(s.max()), w)
    return best


def reference(e, v, tsamp, t_ref, leaf, nbins, nsub, df_steps, step_bins):
    """fold_check.py's algorithm, with sample-start times."""
    n = e.size
    tobs = n * tsamp
    edges = (np.arange(nsub + 1) * n) // nsub
    times = np.arange(n) * tsamp
    phase, _ = libloki.psr_utils.leaf_phase(leaf, times, t_ref)
    fe = np.empty((nsub, nbins))
    fv = np.empty((nsub, nbins))
    tsub = np.empty(nsub)
    for s in range(nsub):
        a, b = edges[s], edges[s + 1]
        fold = libloki.fold.fold_phase(e[a:b], v[a:b], phase[a:b], nbins)
        fe[s], fv[s] = fold[0], fold[1]
        tsub[s] = (times[a] + times[b - 1]) / 2 - t_ref
    spec_e, spec_v = np.fft.rfft(fe, axis=1), np.fft.rfft(fv, axis=1)
    k = np.arange(spec_e.shape[1])
    fstep = step_bins / (nbins * tobs)
    scores = []
    for j in range(-df_steps, df_steps + 1):
        rot = np.exp(-2j * np.pi * np.outer(j * fstep * tsub, k))
        pe = np.fft.irfft((spec_e * rot).sum(0), n=nbins)
        pv = np.maximum(np.fft.irfft((spec_v * rot).sum(0), n=nbins), 1e-12)
        scores.append((boxcar_snr(pe, pv, nbins // 2)[0], j * fstep))
    s_in = scores[df_steps][0]
    best = scores[df_steps]
    for s in scores:
        if s[0] > best[0]:
            best = s
    return s_in, best[0], best[1]


@pytest.fixture(scope="module")
def series() -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    rng = np.random.default_rng(3)
    nsamps, tsamp, nbins = 1 << 14, 1e-3, 16
    times = np.arange(nsamps) * tsamp
    t_ref = nsamps * tsamp / 2
    leaf = taylor_leaf(4.0e4, 23.0)
    phase, _ = libloki.psr_utils.leaf_phase(leaf, times, t_ref)
    on = np.abs((phase % 1.0) - 0.5) < 0.5 / nbins
    e = (rng.standard_normal(nsamps) + 0.6 * on).astype(np.float32)
    v = np.ones(nsamps, dtype=np.float32)
    return e, v, tsamp, t_ref, leaf


def test_refold_matches_the_reference_algorithm(series) -> None:
    e, v, tsamp, t_ref, leaf = series
    # Off by a few steps in frequency, so the scan has to find the pulse.
    off = leaf.copy()
    off[1, 0] += 3 * 0.25 / (16 * e.size * tsamp)
    out = libloki.fold.refold(e, v, tsamp, t_ref, off, nbins=16, nsub=32, df_steps=8)
    s_in, s_best, df = reference(e, v, tsamp, t_ref, off, 16, 32, 8, 0.25)
    assert out["model"] == "taylor"
    assert out["snr_in"] == pytest.approx(s_in, rel=1e-4)
    assert out["snr"] == pytest.approx(s_best, rel=1e-4)
    assert out["df"] == pytest.approx(df, abs=1e-12)
    assert out["snr"] > 5.0


def test_refold_leaf_model_and_split(series) -> None:
    e, v, tsamp, t_ref, leaf = series
    taylor = libloki.fold.refold(e, v, tsamp, t_ref, leaf, nbins=16, nsub=32)
    own = libloki.fold.refold(
        e, v, tsamp, t_ref, leaf, model="leaf", nbins=16, nsub=32, split=2
    )
    assert own["model"] == "taylor"
    assert own["snr"] == taylor["snr"]
    assert own["split_snr"].shape == (2,)
    assert own["profile"].shape == (16,)
    with pytest.raises(ValueError, match="model"):
        libloki.fold.refold(e, v, tsamp, t_ref, leaf, model="orbit")
