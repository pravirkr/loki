import numpy as np
from pyloki.core import common

import loki


def test_brute_fold() -> None:
    nsamps = 2**24
    segment_len = 2**24 // 2**14
    nbins = 64
    tsamp = 0.000064
    t_ref = 0.0
    nthreads = 8
    freqs_arr = np.linspace(140, 145, 50)

    rng = np.random.default_rng()
    ts_e = rng.normal(size=nsamps).astype(np.float32)
    ts_v = rng.normal(size=nsamps).astype(np.float32)

    out_pyloki = common.brutefold_start(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=nbins,
        tsamp=tsamp,
        t_ref=t_ref,
    )
    out_loki = loki.fold.compute_brute_fold(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=nbins,
        tsamp=tsamp,
        t_ref=t_ref,
        nthreads=nthreads,
    )
    np.testing.assert_allclose(
        out_loki.reshape(out_pyloki.shape),
        out_pyloki,
        atol=1e-5,
    )


def test_brute_fold_complex() -> None:
    nsamps = 2**24
    segment_len = 2**24 // 2**14
    nbins = 64
    tsamp = 0.000064
    t_ref = 0.0
    nthreads = 8
    freqs_arr = np.linspace(140, 145, 50)

    rng = np.random.default_rng()
    ts_e = rng.normal(size=nsamps).astype(np.float32)
    ts_v = rng.normal(size=nsamps).astype(np.float32)

    out_pyloki = common.brutefold_start_complex(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=nbins,
        tsamp=tsamp,
        t_ref=t_ref,
    )
    out_loki = loki.fold.compute_brute_fold_complex(
        ts_e,
        ts_v,
        freqs_arr,
        segment_len=segment_len,
        nbins=nbins,
        tsamp=tsamp,
        t_ref=t_ref,
        nthreads=nthreads,
    )
    np.testing.assert_allclose(
        out_loki.reshape(out_pyloki.shape),
        out_pyloki,
        atol=1e-1,
    )


if __name__ == "__main__":
    test_brute_fold()
    test_brute_fold_complex()
