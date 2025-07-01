from __future__ import annotations

from typing import TYPE_CHECKING

from pyloki.periodogram import Periodogram

from loki.libloki.ffa import compute_ffa_scores

if TYPE_CHECKING:
    from pyloki.io.timeseries import TimeSeries

    from loki.libloki.configs import PulsarSearchConfig
    from loki.libloki.ffa import FFAPlan


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
    snrs_flat, ffa_plan = compute_ffa_scores(
        tseries.ts_e,
        tseries.ts_v,
        search_cfg,
        show_progress=show_progress,
    )
    snrs = snrs_flat.reshape(
        *ffa_plan.fold_shapes[-1][1:-2],
        len(search_cfg.score_widths),
    )
    pgram = Periodogram(
        params={"width": search_cfg.score_widths, **ffa_plan.params_dict},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return ffa_plan, pgram
