from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from pyloki.utils import np_utils
from pyloki.utils.misc import CONSOLE, get_logger
from rich.progress import track

from loki.libloki.configs import PulsarSearchConfig
from loki.libloki.scores import generate_box_width_trials, snr_boxcar_1d
from loki.search import ffa_search

if TYPE_CHECKING:
    from numba import types
    from pyloki.io.timeseries import TimeSeries
    from pyloki.simulation.pulse import PulseSignalConfig

logger = get_logger(__name__)

nparam_to_str = {1: "freq", 2: "acc", 3: "jerk", 4: "snap"}


def test_sensitivity_ffa(
    tim_data: TimeSeries,
    signal_cfg: PulseSignalConfig,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool,
) -> tuple[float, float, float]:
    ffa_plan, pgram = ffa_search(
        tim_data,
        search_cfg,
        quiet=quiet,
        show_progress=False,
    )
    snr_shifted = get_shifted_snr(
        ffa_plan.dparams[-1],
        tim_data,
        signal_cfg,
        search_cfg,
    )

    nparams = len(search_cfg.param_limits)
    param_arr = ffa_plan.params[-1]
    true_params_idx = [
        np_utils.find_nearest_sorted_idx(param_arr[-1], signal_cfg.freq),
    ]
    for deriv in range(2, nparams + 1):
        idx = np_utils.find_nearest_sorted_idx(
            param_arr[-deriv],
            signal_cfg.mod_kwargs[nparam_to_str[deriv]],
        )
        true_params_idx.insert(0, idx)
    snr_dynamic = float(pgram.data[tuple(true_params_idx)].max())
    snr_empirical = pgram.find_best_params()["snr"]
    if not quiet:
        logger.info(
            f"snr_dynamic: {snr_dynamic:.2f}, snr_empirical: {snr_empirical:.2f}",
        )
    return snr_shifted, snr_dynamic, snr_empirical


def get_shifted_snr(
    dparams: np.ndarray,
    tim_data: TimeSeries,
    signal_cfg: PulseSignalConfig,
    search_cfg: PulsarSearchConfig,
) -> float:
    # Check the params grid around +- dparam
    nparams = len(dparams)
    shift_snr = []
    grid = [
        [num * sign for num, sign in zip(dparams, signs, strict=False)]
        for signs in product([-1, 1], repeat=nparams)
    ]
    for diff_params in grid:
        freq_shifted = signal_cfg.freq + diff_params[-1]
        fold = tim_data.fold_ephem(
            freq_shifted,
            signal_cfg.fold_bins,
            mod_kwargs={
                nparam_to_str[deriv]: signal_cfg.mod_kwargs[nparam_to_str[deriv]]
                + diff_params[-deriv]
                for deriv in range(2, nparams + 1)
            },
        )
        shift_snr.append(get_best_snr(fold, search_cfg.ducy_max, search_cfg.wtsp))
    return np.max(shift_snr)


def get_best_snr(fold: np.ndarray, ducy_max: float, wtsp: float) -> float:
    widths = generate_box_width_trials(len(fold), ducy_max=ducy_max, wtsp=wtsp)
    return snr_boxcar_1d(fold, widths, 1.0).max()


class TestFFASensitivity:
    def __init__(
        self,
        cfg: PulseSignalConfig,
        param_limits: types.ListType[types.Tuple[float, float]],
        ducy_arr: np.ndarray | None = None,
        tol_bins_arr: np.ndarray | None = None,
        os_arr: np.ndarray | None = None,
        ducy_max: float = 0.5,
        wtsp: float = 1,
        *,
        quiet: bool = False,
    ) -> None:
        self.cfg = cfg
        self.param_limits = param_limits
        if ducy_arr is None:
            ducy_arr = np.linspace(0.01, 0.3, 15)
        if tol_bins_arr is None:
            tol_bins_arr = np.array([1, 2, 4, 8])
        if os_arr is None:
            os_arr = np.array([1, 1.25, 1.5, 1.75, 2])
        self.ducy_arr = ducy_arr
        self.tol_bins_arr = tol_bins_arr
        self.os_arr = os_arr
        self.ducy_max = ducy_max
        self.wtsp = wtsp
        self.quiet = quiet
        self.rng = np.random.default_rng()
        self.losses_real = np.zeros((3, self.ntols, self.nducy), dtype=float)
        self.losses_complex = np.zeros((3, self.ntols, self.nducy), dtype=float)
        self.losses_os = np.zeros((self.nos, self.nducy), dtype=float)

    @property
    def nprams(self) -> int:
        return len(self.param_limits)

    @property
    def ntols(self) -> int:
        return len(self.tol_bins_arr)

    @property
    def nducy(self) -> int:
        return len(self.ducy_arr)

    @property
    def nos(self) -> int:
        return len(self.os_arr)

    @property
    def file_id(self) -> str:
        return (
            f"{nparam_to_str[self.nprams]}_nsamps_{int(np.log2(self.cfg.nsamps)):02d}_"
            f"period_{self.cfg.period:.3f}_os_{self.cfg.os:.1f}"
        )

    def run(self, outdir: str | Path) -> str:
        outpath = Path(outdir)
        if not outpath.is_dir():
            msg = f"Output directory {outdir} does not exist"
            raise FileNotFoundError(msg)
        for idu in track(
            range(self.nducy),
            description="Processing ducy...",
            console=CONSOLE,
            transient=True,
        ):
            cfg_update = self.cfg.get_updated({"ducy": self.ducy_arr[idu]})
            losses_total = self._execute(cfg_update)
            self.losses_os[:, idu] = losses_total[0]
            self.losses_real[:, :, idu] = losses_total[1]
            self.losses_complex[:, :, idu] = losses_total[2]
        return self._save(outpath)

    def _execute(
        self,
        cfg: PulseSignalConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi0 = 0.5 + self.rng.uniform(-cfg.dt, cfg.dt)
        tim_data = cfg.generate(phi0=phi0)
        fold_perfect = tim_data.fold_ephem(
            cfg.freq,
            cfg.fold_bins_ideal,
            mod_kwargs=cfg.mod_kwargs,
        )
        losses_real = np.zeros((3, self.ntols), dtype=float)
        losses_complex = np.zeros((3, self.ntols), dtype=float)
        losses_os = np.zeros((self.nos), dtype=float)
        # Compute signal strength over different oversampling factors
        for ios in range(self.nos):
            fold_bins = int(cfg.fold_bins_ideal / self.os_arr[ios])
            fold_os = tim_data.fold_ephem(
                cfg.freq,
                fold_bins,
                mod_kwargs=cfg.mod_kwargs,
            )
            losses_os[ios] = get_best_snr(fold_os, self.ducy_max, self.wtsp)
        # Compute signal strength over different tol_bins
        for itol in range(self.ntols):
            search_cfg_real = PulsarSearchConfig(
                nsamps=cfg.nsamps,
                tsamp=cfg.dt,
                nbins=cfg.fold_bins,
                tol_bins=self.tol_bins_arr[itol],
                param_limits=self.param_limits,
                ducy_max=self.ducy_max,
                wtsp=self.wtsp,
                bseg_brute=cfg.nsamps // 16384,
                use_fft_shifts=False,
                nthreads=8,
            )
            losses_real[:, itol] = test_sensitivity_ffa(
                tim_data,
                cfg,
                search_cfg_real,
                quiet=self.quiet,
            )
            search_cfg_complex = PulsarSearchConfig(
                nsamps=cfg.nsamps,
                tsamp=cfg.dt,
                nbins=cfg.fold_bins,
                tol_bins=self.tol_bins_arr[itol],
                param_limits=self.param_limits,
                ducy_max=self.ducy_max,
                wtsp=self.wtsp,
                bseg_brute=cfg.nsamps // 16384,
                use_fft_shifts=True,
                nthreads=8,
            )
            losses_complex[:, itol] = test_sensitivity_ffa(
                tim_data,
                cfg,
                search_cfg_complex,
                quiet=self.quiet,
            )
        snr_desired = get_best_snr(fold_perfect, self.ducy_max, self.wtsp)
        losses_real_signi = (losses_real**2) / (snr_desired**2)
        losses_complex_signi = (losses_complex**2) / (snr_desired**2)
        losses_os_signi = (losses_os**2) / (snr_desired**2)
        return losses_os_signi, losses_real_signi, losses_complex_signi

    def _save(self, outpath: Path) -> str:
        outfile = outpath / f"loki_ffa_sensitivity_{self.file_id}.h5"
        with h5py.File(outfile, "w") as f:
            for attr in ["period", "os", "nsamps", "fold_bins_ideal", "dt", "snr"]:
                f.attrs[attr] = getattr(self.cfg, attr)
            for arr in [
                "ducy_arr",
                "tol_bins_arr",
                "os_arr",
                "losses_real",
                "losses_complex",
                "losses_os",
            ]:
                f.create_dataset(
                    arr,
                    data=getattr(self, arr),
                    compression="gzip",
                    compression_opts=9,
                )
            f.create_dataset(
                "param_limits",
                data=np.array(self.param_limits),
                compression="gzip",
                compression_opts=9,
            )

        return outfile.as_posix()
