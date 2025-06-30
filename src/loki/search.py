from __future__ import annotations

from typing import TYPE_CHECKING

from pyloki.periodogram import Periodogram

from loki.config import PulsarSearchConfig as LokiPulsarSearchConfig

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries

    from loki.ffa import FFAPlan


def ffa_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    show_progress: bool = True,
) -> tuple[FFAPlan, Periodogram]:
    """Perform a Fast Folding Algorithm search on a time series.

    Parameters
    ----------
    tseries : TimeSeries
        The time series to search.
    search_cfg : PulsarSearchConfig
        The configuration object for the search.
    show_progress : bool, default=True
        Whether to show progress of FFA computation.

    Returns
    -------
    tuple[FFAPlan, Periodogram]
        The FFAPlan object and the Periodogram object.
    """
    loki_search_cfg = LokiPulsarSearchConfig(
        nsamps=search_cfg.nsamps,
        tsamp=search_cfg.tsamp,
        nbins=search_cfg.nbins,
        tol_bins=search_cfg.tol_bins,
        param_limits=search_cfg.param_limits,
        bseg_brute=search_cfg.bseg_brute,
        bseg_ffa=search_cfg.bseg_ffa,
        use_fft_shifts=search_cfg.use_fft_shifts,
    )
    snrs_flat, ffa_plan = loki.ffa.compute_ffa_scores(
        tseries.ts_e,
        tseries.ts_v,
        loki_search_cfg,
        show_progress=show_progress,
    )
    snrs = snrs_flat.reshape(
        *ffa_plan.fold_shapes[-1][1:-2],
        len(loki_search_cfg.score_widths),
    )
    pgram = Periodogram(
        params={"width": loki_search_cfg.score_widths, **ffa_plan.params_dict},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return ffa_plan, pgram
