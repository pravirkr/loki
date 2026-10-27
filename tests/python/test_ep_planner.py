from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loki import libloki


def _search_config(
    nthreads: int, target_snr: float | None = None
) -> libloki.configs.PulsarSearchConfig:
    return libloki.configs.PulsarSearchConfig(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [140.0, 145.0]], dtype=np.float64),
        ducy_max=0.3,
        use_fourier=False,
        nthreads=nthreads,
        max_process_memory_gb=4.0,
        bseg_brute=1024,
        bseg_ffa=(1 << 16) // 8,
        prune_poly_order=2,
        target_snr=target_snr,
    )


def test_ep_region_planner_accepts_workers_and_rfi_config() -> None:
    cfg = _search_config(nthreads=2)
    rfi = libloki.prune.PruneRFIConfig(harvest_scheme=[5.0], max_harvests=64)
    planner = libloki.prune.EPRegionPlannerTime(cfg, rfi_config=rfi, n_workers=1)
    assert planner.nchunks >= 1
    assert 0.0 < planner.stats.max_memory_gb <= 4.0
    for chunk in planner.chunk_cfgs:
        assert chunk.chunk_memory_gb <= planner.stats.max_memory_gb
        assert chunk.ffa_transient_bytes == 0


@pytest.mark.skipif(
    "cuda" not in libloki.available_backends(),
    reason="CUDA backend not built",
)
def test_ep_region_planner_cuda_rejects_an_active_rfi_config() -> None:
    cfg = _search_config(nthreads=1)
    rfi = libloki.prune.PruneRFIConfig(harvest_scheme=[5.0], max_harvests=64)
    with pytest.raises(ValueError):  # noqa: PT011
        libloki.prune.EPRegionPlannerTime(cfg, rfi_config=rfi, backend="cuda")


def _cuda_built() -> bool:
    return "cuda" in libloki.available_backends()


@pytest.mark.skipif(not _cuda_built(), reason="CUDA backend not built")
def test_ep_region_planner_cuda_fits_the_device_budget(tmp_path: Path) -> None:
    cfg = _search_config(nthreads=4)
    planner = libloki.prune.EPRegionPlannerTime(cfg, backend="cuda", device=0)
    assert planner.nchunks >= 1
    stats = planner.stats
    assert 0.0 < stats.max_memory_gb <= stats.memory_limit_gb <= 4.0
    for chunk in planner.chunk_cfgs:
        assert chunk.chunk_memory_gb <= stats.memory_limit_gb
        assert chunk.ffa_transient_bytes > 0
        assert len(chunk.threshold_scheme) > 0

    # A plan written on CUDA loads on the CUDA backend and on the CPU: the peak
    # is rechecked with the policy of the backend that loads it.
    cache = tmp_path / "plan_cuda.h5"
    planner.save_cache(str(cache))
    reloaded = libloki.prune.EPRegionPlannerTime(
        cfg,
        plan_cache_file=str(cache),
        backend="cuda",
    )
    assert reloaded.nchunks == planner.nchunks
    for chunk, cached in zip(planner.chunk_cfgs, reloaded.chunk_cfgs, strict=True):
        assert cached.pd == chunk.pd
        assert cached.pd_se == chunk.pd_se
    on_cpu = libloki.prune.EPRegionPlannerTime(cfg, plan_cache_file=str(cache))
    assert on_cpu.nchunks == planner.nchunks
    for cpu_chunk, cuda_chunk in zip(
        on_cpu.chunk_cfgs, planner.chunk_cfgs, strict=True
    ):
        assert cpu_chunk.predicted_live == cuda_chunk.predicted_live
        assert cpu_chunk.max_sugg >= cpu_chunk.predicted_live
        assert cpu_chunk.max_sugg < cuda_chunk.max_sugg


def test_ep_region_planner_rejects_unknown_backend() -> None:
    cfg = _search_config(nthreads=1)
    with pytest.raises(ValueError, match="backend"):
        libloki.prune.EPRegionPlannerTime(cfg, backend="tpu")


@pytest.mark.skipif(not _cuda_built(), reason="CUDA backend not built")
def test_ep_freq_sweep_cuda_runs_and_rejects_rfi(tmp_path: Path) -> None:
    cfg = _search_config(nthreads=1)
    rng = np.random.default_rng(0)
    ts_e = rng.standard_normal(1 << 16).astype(np.float32)
    ts_v = np.ones(1 << 16, dtype=np.float32)

    sweep = libloki.prune.EPFreqSweep(
        cfg,
        show_progress=False,
        ref_segs=[4],
        backend="cuda",
    )
    sweep.execute(ts_e, ts_v, str(tmp_path), "py_cuda")
    assert (tmp_path / "py_cuda_ep_results.h5").exists()

    rfi = libloki.prune.PruneRFIConfig(harvest_scheme=[5.0], max_harvests=64)
    with pytest.raises(ValueError, match="rfi_config"):
        libloki.prune.EPFreqSweep(cfg, rfi_config=rfi, backend="cuda")


def test_ep_planner_seeded_forecast_and_run_aware_cache(tmp_path: Path) -> None:
    cfg = _search_config(nthreads=1)
    planner = libloki.prune.EPRegionPlannerTime(cfg, ref_segs=[0])
    second = libloki.prune.EPRegionPlannerTime(cfg, ref_segs=[0])
    assert planner.nchunks == second.nchunks
    for first, other in zip(planner.chunk_cfgs, second.chunk_cfgs, strict=True):
        assert first.max_sugg == other.max_sugg
        assert first.predicted_peak == other.predicted_peak
        assert first.predicted_live == other.predicted_live
        assert first.batch_size == other.batch_size
        assert first.pd == other.pd
        assert first.threshold_scheme == other.threshold_scheme
        # The ring is sized in multiples of 1024 leaves, not powers of two.
        assert first.max_sugg % 1024 == 0
        assert first.max_sugg > first.predicted_live
    cache = tmp_path / "run_plan.h5"
    planner.save_cache(str(cache))
    with pytest.raises(ValueError, match="ref_segs"):
        libloki.prune.EPRegionPlannerTime(cfg, ref_segs=[4], plan_cache_file=cache)


@pytest.mark.parametrize("kwargs", [{"n_runs": 9}, {"ref_segs": [8]}])
def test_ep_sweep_rejects_invalid_runs_at_construction(kwargs: dict) -> None:
    cfg = _search_config(nthreads=1)
    with pytest.raises((ValueError, RuntimeError), match="n_runs|ref_segs"):
        libloki.prune.EPFreqSweep(cfg, **kwargs)


def test_ep_target_snr_is_independent_and_invalidates_cache(tmp_path: Path) -> None:
    cfg = _search_config(nthreads=1)
    assert cfg.snr_min == 7.0
    assert cfg.target_snr == 8.5
    cache = tmp_path / "snr_plan.h5"
    libloki.prune.EPRegionPlannerTime(cfg, ref_segs=[0]).save_cache(str(cache))
    changed = _search_config(nthreads=1, target_snr=12.0)
    with pytest.raises(ValueError, match="target_snr"):
        libloki.prune.EPRegionPlannerTime(changed, ref_segs=[0], plan_cache_file=cache)


def test_ep_plan_cache_is_keyed_on_ffa_segmentation(tmp_path: Path) -> None:
    cfg = _search_config(nthreads=1)
    cache = tmp_path / "segmentation_plan.h5"
    libloki.prune.EPRegionPlannerTime(cfg, ref_segs=[0]).save_cache(str(cache))
    args = dict(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [140.0, 145.0]], dtype=np.float64),
        ducy_max=0.3,
        use_fourier=False,
        nthreads=1,
        max_process_memory_gb=4.0,
        prune_poly_order=2,
    )
    for changed in (
        dict(bseg_brute=512, bseg_ffa=(1 << 16) // 8),
        dict(bseg_brute=1024, bseg_ffa=(1 << 16) // 16),
    ):
        other = libloki.configs.PulsarSearchConfig(**args, **changed)
        with pytest.raises(ValueError, match="bseg"):
            libloki.prune.EPRegionPlannerTime(
                other, ref_segs=[0], plan_cache_file=cache
            )


@pytest.mark.parametrize(
    "nparams,expected", [(1, 8.5), (2, 8.5), (3, 9.2), (4, 9.8), (5, 10.0)]
)
def test_ep_target_snr_defaults_and_override(nparams: int, expected: float) -> None:
    args = dict(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[0.0, 0.0]] * (nparams - 1) + [[140.0, 145.0]]),
        bseg_brute=1024,
        bseg_ffa=8192,
        prune_poly_order=3,
    )
    assert libloki.configs.EPSearchConfig(**args).target_snr == expected
    assert (
        libloki.configs.EPSearchConfig(**args, target_snr=None).target_snr == expected
    )
    overridden = libloki.configs.EPSearchConfig(**args, target_snr=11.0, snr_min=6.0)
    assert overridden.target_snr == 11.0
    assert overridden.snr_min == 6.0


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ep_planner_accepts_best_available_path(tmp_path: Path, backend: str) -> None:
    if backend == "cuda" and not _cuda_built():
        pytest.skip("CUDA backend not built")
    cfg = _search_config(nthreads=1)
    cache = tmp_path / "approximate_pd.h5"
    planner = libloki.prune.EPRegionPlannerTime(
        cfg, min_pd=1.0, ref_segs=[0], plan_cache_file=cache, backend=backend
    )
    assert planner.nchunks > 0
    assert cache.exists()
    import h5py

    with h5py.File(cache) as saved:
        assert saved.attrs["min_pd"] == 1.0
        # The threshold pipeline is part of the plan.
        assert saved.attrs["dts_algorithm"] == "shared-per-region"
        assert saved.attrs["dts_mode"] == "shared"
        assert saved.attrs["dts_w_terminal"] == 1.0
    # Reloading preserves the user's requested constraint and chosen scheme.
    reloaded = libloki.prune.EPRegionPlannerTime(
        cfg, min_pd=1.0, ref_segs=[0], plan_cache_file=cache, backend=backend
    )
    assert reloaded.nchunks == planner.nchunks


@pytest.mark.parametrize("min_pd", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_ep_planner_rejects_invalid_min_pd(min_pd: float) -> None:
    with pytest.raises(ValueError, match="min_pd must be in"):
        libloki.prune.EPRegionPlannerTime(_search_config(1), min_pd=min_pd)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_frequency_only_ep_is_rejected_before_planning(backend: str) -> None:
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[140.0, 145.0]]),
        bseg_brute=1024,
        bseg_ffa=8192,
    )
    with pytest.raises(ValueError, match="frequency-only EP"):
        libloki.prune.EPRegionPlannerTime(cfg, backend=backend)
    with pytest.raises(ValueError, match="frequency-only EP"):
        libloki.prune.EPMultiPassTime(cfg, [1.0] * 7, backend=backend)


@pytest.mark.parametrize("nbins,nbins_max", [(1024, 1025), (2048, 2048)])
def test_ep_rejects_bins_above_1024(nbins: int, nbins_max: int) -> None:
    # The config also serves FFA searches, which keep larger folds.
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=nbins,
        nbins_max=nbins_max,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [140.0, 145.0]]),
        bseg_brute=1024,
        bseg_ffa=8192,
    )
    with pytest.raises(ValueError, match="nbins_max must be <= 1024"):
        libloki.prune.EPRegionPlannerTime(cfg)
    with pytest.raises(ValueError, match="nbins_max must be <= 1024"):
        libloki.prune.EPMultiPassTime(cfg, [1.0] * 7)


@pytest.mark.skipif(not _cuda_built(), reason="CUDA backend not built")
@pytest.mark.parametrize("limit_gb", [1.0, 0.6])
def test_cuda_threshold_design_fits_the_budget(limit_gb: float) -> None:
    # nbins 1024 would need far more than the limit for its search pools: the
    # search steps its budget down and runs at 64 bins, and the schedule is
    # re-fitted at 1024 bins.
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=1024,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [0.75, 1.5]]),
        ducy_max=0.3,
        wtsp=1.2,
        nthreads=1,
        max_process_memory_gb=limit_gb,
        bseg_brute=1024,
        bseg_ffa=8192,
    )
    planner = libloki.prune.EPRegionPlannerFourier(cfg, backend="cuda")
    assert planner.nchunks > 0


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ep_planner_runs_region_designs(backend: str) -> None:
    """Every chunk runs its region's design, and the plan fits."""
    if backend == "cuda" and not _cuda_built():
        pytest.skip("CUDA backend not built")
    cfg = libloki.configs.PulsarSearchConfig(
        nsamps=1 << 16,
        tsamp=64e-6,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [100.0, 145.0]], dtype=np.float64),
        ducy_max=0.3,
        use_fourier=False,
        nthreads=1,
        max_process_memory_gb=1.0,
        bseg_brute=1024,
        bseg_ffa=(1 << 16) // 8,
        prune_poly_order=2,
    )
    planner = libloki.prune.EPRegionPlannerTime(
        cfg, min_pd=0.5, ref_segs=[0], backend=backend
    )
    stats = planner.stats
    assert 0.0 < stats.max_memory_gb <= stats.memory_limit_gb
    for chunk in planner.chunk_cfgs:
        assert len(chunk.threshold_scheme) == len(chunk.branching_pattern)
        assert 0.0 < chunk.pd <= 1.0
        assert chunk.pd_se >= 0.0
        assert chunk.max_sugg >= chunk.predicted_live
        assert chunk.pd >= 0.9 * 0.5 - 4.0 * chunk.pd_se
    # One design per region (the regions here differ only in nbins).
    designs = {
        (chunk.cfg.nbins, tuple(chunk.threshold_scheme)) for chunk in planner.chunk_cfgs
    }
    assert len(designs) == len({chunk.cfg.nbins for chunk in planner.chunk_cfgs})


def test_ep_plan_cache_made_on_cuda_serves_cpu(tmp_path: Path) -> None:
    """A CUDA plan is reused on the CPU under the CPU's own memory limit."""
    if not _cuda_built():
        pytest.skip("CUDA backend not built")

    def config(nthreads: int, memory_gb: float) -> libloki.configs.PulsarSearchConfig:
        return libloki.configs.PulsarSearchConfig(
            nsamps=1 << 16,
            tsamp=64e-6,
            nbins=32,
            eta=1.0,
            param_limits=np.array([[-10.0, 10.0], [100.0, 145.0]], dtype=np.float64),
            ducy_max=0.3,
            use_fourier=False,
            nthreads=nthreads,
            max_process_memory_gb=memory_gb,
            bseg_brute=1024,
            bseg_ffa=(1 << 16) // 8,
            prune_poly_order=2,
        )

    cache = str(tmp_path / "plan.h5")
    gpu = libloki.prune.EPRegionPlannerTime(
        config(1, 1.0), min_pd=0.5, ref_segs=[0], plan_cache_file=cache, backend="cuda"
    )
    cpu = libloki.prune.EPRegionPlannerTime(
        config(2, 4.0), min_pd=0.5, ref_segs=[0], plan_cache_file=cache, backend="cpu"
    )
    assert cpu.nchunks == gpu.nchunks
    for a, b in zip(gpu.chunk_cfgs, cpu.chunk_cfgs, strict=True):
        assert a.nominal_f_start == b.nominal_f_start
        assert a.nominal_f_end == b.nominal_f_end
        assert list(a.threshold_scheme) == list(b.threshold_scheme)
        assert a.pd == b.pd
        assert b.max_sugg >= b.predicted_live
    assert 0.0 < cpu.stats.max_memory_gb <= cpu.stats.memory_limit_gb
    # On the same backend the limit is part of the plan.
    with pytest.raises(ValueError, match="max_process_memory_gb"):
        libloki.prune.EPRegionPlannerTime(
            config(1, 2.0),
            min_pd=0.5,
            ref_segs=[0],
            plan_cache_file=cache,
            backend="cuda",
        )
