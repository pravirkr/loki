"""Candidate processing starts from completed, independent Taylor EP runs."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from loki import libloki
from loki.candidates import CandidateFile, export_pulsarx

N = 1 << 16
DT = 64e-6
C = 299792458.0


@pytest.fixture(scope="module")
def series() -> tuple[np.ndarray, np.ndarray]:
    modulation = libloki.simulation.ModulatorParams()
    modulation.acc = 5.0
    signal = libloki.simulation.PulseSignalConfig(
        1 / 142.3, DT, N, 15.0, 0.1, "derivative", modulation, None, 1
    ).generate("gaussian", 0.5)
    samples = np.asarray(signal.ts_e, dtype=np.float32) / np.sqrt(signal.ts_v[0])
    return samples, np.ones_like(samples)


def config(fourier: bool, *, threads: int = 2):
    return libloki.configs.PulsarSearchConfig(
        N,
        DT,
        32,
        1.0,
        np.array([[-10.0, 10.0], [140.0, 145.0]]),
        ducy_max=0.3,
        use_fourier=fourier,
        nthreads=threads,
        max_process_memory_gb=4.0,
        bseg_brute=1024,
        bseg_ffa=N // 8,
        prune_poly_order=2,
        snr_min=7,
    )


def options():
    value = libloki.configs.CandidateOptions()
    value.memory_gb = 0.02
    return value


def observation():
    value = libloki.io.ObservationMetadata()
    value.dm = 42.5
    value.start_mjd = 60000
    value.barycentric = False
    value.original_nsamps = N + 100
    return value


@pytest.fixture(
    scope="module",
    params=[("cpu", False), ("cpu", True), ("cuda", False), ("cuda", True)],
)
def product(request, tmp_path_factory, series):
    tmp_path = tmp_path_factory.mktemp("ep-products")
    backend, fourier = request.param
    if backend not in libloki.available_backends():
        pytest.skip(f"{backend} backend not built")
    cfg = config(fourier)
    sweep = libloki.prune.EPFreqSweep(
        cfg,
        show_progress=False,
        min_pd=0.5,
        ref_segs=[2, 4],
        candidates=options(),
        backend=backend,
    )
    sweep.execute(*series, str(tmp_path), "product", observation=observation())
    return tmp_path / "product_ep_results.h5", sweep, backend, fourier


def check_graph(handle, grouping="transitive"):
    raw = handle["detections"]
    f0, f1, duty = (raw[name][:] for name in ("f0", "f1", "duty_cycle"))
    n = len(f0)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            i = int(parent[i])
        return i

    def distance(i, j):
        a, b = f0[i] - f0[j], f1[i] - f1[j]
        times = [-N * DT / 2, N * DT / 2]
        if b != 0 and abs(-a / b) < N * DT / 2:
            times.append(-a / b)
        phases = [a * t + b * t * t / 2 for t in times]
        return max(phases) - min(phases)

    representatives = []
    for i in range(n):
        neighbors = [
            j for j in range(i) if distance(i, j) <= max(0.05, min(duty[i], duty[j]))
        ]
        if grouping == "transitive":
            for j in neighbors:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
        elif grouping == "anchored":
            matches = [j for j in representatives if j in neighbors]
            if matches:
                parent[i] = matches[0]
            else:
                representatives.append(i)
    roots = np.array([root(i) for i in range(n)])
    reps = np.unique(roots)
    membership = np.searchsorted(reps, roots)
    np.testing.assert_array_equal(raw["cluster_id"][:], membership)
    np.testing.assert_array_equal(handle["clusters/id"][:], raw["id"][:][reps])
    np.testing.assert_array_equal(
        handle["clusters/member_count"][:], np.bincount(membership)
    )
    extents = [
        max(distance(i, int(j)) for j in np.flatnonzero(roots == i)) for i in reps
    ]
    np.testing.assert_allclose(handle["clusters/phase_extent"][:], extents, atol=1e-12)


def test_ep_candidates_preserve_runs_and_cluster_the_full_list(product, tmp_path):
    path, _, _, _ = product
    with h5py.File(path) as handle:
        assert handle.attrs["algorithm"] == "ep"
        assert handle.attrs["complete"] == 1
        raw = handle["detections"]
        paths = handle["search/sources/run_path"].asstr()[:]
        assert len(paths) == 2
        assert len(raw["id"]) > len(handle["candidates/id"]) > 0
        assert len(np.unique(raw["id"][:])) == len(raw["id"])
        assert len(np.unique(raw["source_region"][:])) == 2
        assert len(raw["id"]) == sum(
            np.count_nonzero(handle[p]["scores"][:] >= 7) for p in paths
        )
        for i, identity in enumerate(raw["id"][:]):
            source, leaf = int(identity) >> 32, int(identity) & 0xFFFFFFFF
            run = handle[paths[source]]
            assert raw["source_index"][i] == leaf
            assert raw["snr"][i] == run["scores"][leaf]
            assert raw["width"][i] == run["width"][leaf]
            np.testing.assert_array_equal(
                raw["param_sets"][i], run["param_sets"][leaf, :2, 0]
            )
            chunk = run.parent.parent
            assert raw["width"][i] in chunk["scoring_widths"][:]
            assert raw["nbins"][i] == chunk.attrs["nbins"]
            assert "scores_ep" in run
        assert not np.any(raw["has_parameter_errors"][:])
        np.testing.assert_allclose(
            raw["f1"][:], -raw["f0"][:] * raw["param_sets"][:, 0] / C
        )
        check_graph(handle)
    with CandidateFile(path) as candidates:
        assert candidates.metadata["reference_mjd"] == pytest.approx(
            60000 + N * DT / 172800
        )
    summary = export_pulsarx(path, tmp_path / "ep.cands")
    assert summary["count"] > 0


@pytest.mark.parametrize(
    "policy", ["anchored", "unclustered", "no_raw", "capped", "empty"]
)
def test_ep_selection_is_postprocessing(product, tmp_path, policy):
    path, _, _, _ = product
    output = tmp_path / f"{policy}.h5"
    opts = options()
    if policy == "anchored":
        opts.grouping = "anchored"
    elif policy == "unclustered":
        opts.cluster = False
    elif policy == "no_raw":
        opts.retain_detections = False
    elif policy == "capped":
        opts.max_candidates = 1
    libloki.prune.EPFreqSweep.publish_candidates(
        path,
        output,
        candidates=opts,
        observation=observation(),
        snr_min=1e6 if policy == "empty" else None,
    )
    with h5py.File(path) as original, h5py.File(output) as result:
        for run_path in original["search/sources/run_path"].asstr()[:]:
            for field in ("scores", "scores_ep", "param_sets", "width", "level_stats"):
                np.testing.assert_array_equal(
                    original[f"{run_path}/{field}"][:], result[f"{run_path}/{field}"][:]
                )
        if policy == "no_raw":
            assert "detections" not in result
            np.testing.assert_array_equal(
                original["candidates/id"][:], result["candidates/id"][:]
            )
        elif policy == "empty":
            assert result.attrs["n_detections"] == 0
            assert len(result["clusters/id"]) == len(result["candidates/id"]) == 0
        else:
            check_graph(
                result, "singleton" if policy == "unclustered" else opts.grouping
            )
            if policy == "capped":
                assert len(result["candidates/id"]) == 1


def test_ep_sources_remain_unique_across_chunks(product, tmp_path):
    path, _, _, _ = product
    spool, output = tmp_path / "chunks.h5", tmp_path / "grouped.h5"
    spool.write_bytes(path.read_bytes())
    with h5py.File(spool, "r+") as handle:
        handle.copy("chunks/chunk_0000", "chunks/chunk_0001")
        handle["chunks/chunk_0001"].attrs["chunk_id"] = 1
        handle.attrs["nchunks"] = 2
    libloki.prune.EPFreqSweep.publish_candidates(spool, output, candidates=options())
    with h5py.File(path) as original, h5py.File(output) as result:
        assert len(result["detections/id"]) == 2 * len(original["detections/id"])
        ids = result["detections/id"][:]
        assert len(np.unique(ids)) == len(ids)
        assert len(result["search/sources/run_path"]) == 4
        assert set(result["search/sources/chunk_id"][:]) == {0, 1}
        np.testing.assert_array_equal(
            result["candidates/id"][:], original["candidates/id"][:]
        )
        np.testing.assert_array_equal(
            result["clusters/member_count"][:], 2 * original["clusters/member_count"][:]
        )
        check_graph(result)


def test_ep_repeated_execution_does_not_accumulate_candidates(
    product, tmp_path, series
):
    path, sweep, backend, fourier = product
    sweep.execute(*series, str(tmp_path), "again", observation=observation())
    with (
        h5py.File(path) as original,
        h5py.File(tmp_path / "again_ep_results.h5") as repeated,
    ):
        assert repeated.attrs["n_detections"] == original.attrs["n_detections"]
        check_graph(repeated)
        if fourier or backend == "cpu":
            for table in ("detections", "clusters", "candidates"):
                for column in ("id", "snr", "width", "nbins", "cluster_id"):
                    np.testing.assert_array_equal(
                        repeated[f"{table}/{column}"][:],
                        original[f"{table}/{column}"][:],
                    )
        else:
            np.testing.assert_allclose(
                repeated["candidates/snr"][:], original["candidates/snr"][:], rtol=1e-5
            )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ep_failed_postprocessing_keeps_completed_runs(tmp_path, series, backend):
    if backend not in libloki.available_backends():
        pytest.skip(f"{backend} backend not built")
    opts = options()
    bad_scratch = tmp_path / "not-a-directory"
    bad_scratch.write_text("existing file")
    opts.temporary_directory = str(bad_scratch)
    sweep = libloki.prune.EPFreqSweep(
        config(True),
        show_progress=False,
        min_pd=0.5,
        ref_segs=[2, 4],
        candidates=opts,
        backend=backend,
    )
    previous = tmp_path / "retry_ep_results.h5"
    previous.write_bytes(b"previous product")
    with pytest.raises(Exception, match=r"create.directories"):
        sweep.execute(*series, str(tmp_path), "retry")
    assert previous.read_bytes() == b"previous product"
    spool = tmp_path / "retry_ep_spool.h5"
    assert spool.exists()
    assert not list(tmp_path.glob(".loki-ep-*"))
    libloki.prune.EPFreqSweep.publish_candidates(spool, previous, candidates=options())
    with h5py.File(previous) as result:
        assert len(result["candidates/id"]) > 0
    with h5py.File(spool, "r+") as unfinished:
        unfinished.attrs["complete"] = 0
    with pytest.raises(ValueError, match="incomplete"):
        libloki.prune.EPFreqSweep.publish_candidates(
            spool, previous, candidates=options()
        )
    with h5py.File(previous) as result:
        assert result.attrs["complete"] == 1


def test_ep_graph_memory_exhaustion_preserves_previous_product(product, tmp_path):
    path, _, _, _ = product
    spool, output = tmp_path / "many-tracks.h5", tmp_path / "previous.h5"
    spool.write_bytes(path.read_bytes())
    with h5py.File(spool, "r+") as handle:
        source = handle["search/sources/run_path"].asstr()[0]
        run = handle[source]
        leaf = run["param_sets"][0]
        leaves = np.repeat(leaf[None, :, :], 1000, axis=0)
        leaves[:, 0, 0] = np.linspace(-10, 10, 1000)
        leaves[:, 1, 0] = 142.3
        width = run["width"][0]
        for name in ("param_sets", "scores", "scores_ep", "width"):
            del run[name]
        run["param_sets"] = leaves
        run["scores"] = np.full(1000, 12, dtype=np.float32)
        run["scores_ep"] = np.full(1000, 12, dtype=np.float32)
        run["width"] = np.full(1000, width, dtype=np.uint32)
    output.write_bytes(b"previous product")
    opts = options()
    opts.memory_gb = 1e-4
    with pytest.raises(RuntimeError, match="transitive track index"):
        libloki.prune.EPFreqSweep.publish_candidates(spool, output, candidates=opts)
    assert output.read_bytes() == b"previous product"
    opts.cluster = False
    libloki.prune.EPFreqSweep.publish_candidates(spool, output, candidates=opts)
    with h5py.File(output) as result:
        assert len(result["candidates/id"]) >= 1000


@pytest.mark.parametrize("nparams", [3, 4])
def test_ep_reported_taylor_derivatives_are_normalized_without_truncation(
    product, tmp_path, nparams
):
    path, _, _, _ = product
    spool, output = tmp_path / "higher.h5", tmp_path / "normalized.h5"
    spool.write_bytes(path.read_bytes())
    with h5py.File(spool, "r+") as handle:
        del handle.attrs["param_names"]
        names = ["snap", "jerk", "accel", "freq"][-nparams:]
        handle.attrs["param_names"] = np.asarray(names, dtype=h5py.string_dtype())
        for source in handle["search/sources/run_path"].asstr()[:]:
            run = handle[source]
            old = run["param_sets"][:]
            leaves = np.zeros((len(old), nparams + 2, 2))
            leaves[:, -4:, :] = old
            leaves[:, nparams - 3, 0] = 3.0  # reported jerk, m/s^3
            if nparams == 4:
                leaves[:, 0, 0] = 2.0  # reported snap, m/s^4
            del run["param_sets"]
            run["param_sets"] = leaves
    opts = options()
    opts.cluster = False
    libloki.prune.EPFreqSweep.publish_candidates(
        spool, output, candidates=opts, observation=observation()
    )
    with h5py.File(output) as result:
        raw = result["detections"]
        native = raw["param_sets"][:]
        np.testing.assert_array_equal(raw["f0"][:], native[:, -1])
        np.testing.assert_allclose(raw["f1"][:], -native[:, -1] * native[:, -2] / C)
        np.testing.assert_allclose(raw["f2"][:], -native[:, -1] * 3.0 / C)
        assert not np.any(raw["has_parameter_errors"][:])
    if nparams == 4:
        with pytest.raises(ValueError, match="above jerk"):
            export_pulsarx(output, tmp_path / "higher.cands")
    else:
        export_pulsarx(output, tmp_path / "higher.cands")
        rows = np.loadtxt(tmp_path / "higher.cands", comments="#", ndmin=2)
        np.testing.assert_allclose(rows[:, 5], -rows[:, 3] * 3.0 / C, rtol=1e-9)


def test_ep_candidate_scope_is_explicit():
    with pytest.raises(ValueError, match="Taylor"):
        libloki.prune.EPFreqSweep(
            config(False), candidates=options(), poly_basis="chebyshev"
        )
    rfi = libloki.prune.PruneRFIConfig(harvest_scheme=[5.0], max_harvests=64)
    with pytest.raises(ValueError, match="RFI"):
        libloki.prune.EPFreqSweep(config(False), candidates=options(), rfi_config=rfi)


def test_disabled_candidates_write_the_earlier_run_diagnostics(tmp_path, series):
    disabled = options()
    disabled.enabled = False
    # Scope checks apply only to enabled processing.
    libloki.prune.EPFreqSweep(
        config(False), candidates=disabled, poly_basis="chebyshev"
    )
    products = {}
    for name, candidates in (("off", disabled), ("diagnostic", None)):
        sweep = libloki.prune.EPFreqSweep(
            config(False),
            show_progress=False,
            min_pd=0.5,
            ref_segs=[2, 4],
            candidates=candidates,
        )
        sweep.execute(*series, str(tmp_path), name, observation=observation())
        products[name] = tmp_path / f"{name}_ep_results.h5"
    assert not list(tmp_path.glob(".loki-ep-*"))
    with (
        h5py.File(products["off"]) as off,
        h5py.File(products["diagnostic"]) as diagnostic,
    ):
        for name in ("candidates", "detections", "clusters", "search"):
            assert name not in off
        assert set(off.keys()) == set(diagnostic.keys())
        assert set(off.attrs.keys()) == set(diagnostic.attrs.keys())
        runs = []
        off.visit(lambda name: runs.append(name) if name.endswith("/scores") else None)
        assert runs
        for name in runs:
            np.testing.assert_array_equal(off[name][:], diagnostic[name][:])
    with pytest.raises(ValueError, match="enabled"):
        libloki.prune.EPFreqSweep.publish_candidates(
            products["off"], tmp_path / "x.h5", candidates=disabled
        )
