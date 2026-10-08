from __future__ import annotations

import numpy as np

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
