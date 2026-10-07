from __future__ import annotations

from typing import TYPE_CHECKING

from loki.libloki.ffa import compute_ffa_scores
from pyloki.periodogram import Periodogram

if TYPE_CHECKING:
    from loki.libloki.configs import PulsarSearchConfig
    from loki.libloki.plans import FFAPlanTime
    from pyloki.io.timeseries import TimeSeries


def ffa_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool = False,
    show_progress: bool = False,
    backend: str = "cpu",
    device: int = 0,
) -> tuple[FFAPlanTime, Periodogram]:
    """Perform a Fast Folding Algorithm search on a time series.

    Parameters
    ----------
    tseries : TimeSeries
        The time series to search.
    search_cfg : PulsarSearchConfig
        The configuration object for the search.
    quiet : bool, default=False
        Whether to suppress logging.
    show_progress : bool, default=False
        Whether to show progress of FFA computation.
    backend : str, default="cpu"
        Backend to fold and score on ("cpu" or "cuda").
    device : int, default=0
        Device ordinal for the "cuda" backend.

    Returns
    -------
    tuple[FFAPlanTime, Periodogram]
        The FFA plan (time domain) object and the Periodogram object.
    """
    snrs_flat, ffa_plan = compute_ffa_scores(
        tseries.ts_e,
        tseries.ts_v,
        search_cfg,
        quiet=quiet,
        show_progress=show_progress,
        backend=backend,
        device=device,
    )
    snrs = snrs_flat.reshape(
        *ffa_plan.param_counts[-1],
        search_cfg.n_scoring_widths,
    )
    pgram = Periodogram(
        params={"width": search_cfg.score_widths, **ffa_plan.params_dict},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return ffa_plan, pgram
