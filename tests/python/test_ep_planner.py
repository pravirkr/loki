from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loki import libloki


def _search_config(nthreads: int) -> libloki.configs.PulsarSearchConfig:
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
    on_cpu = libloki.prune.EPRegionPlannerTime(cfg, plan_cache_file=str(cache))
    assert on_cpu.nchunks == planner.nchunks


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
