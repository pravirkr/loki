"""Production candidate reading and folding-export contracts."""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from loki import libloki
from loki.candidates import C, CandidateFile, export_pulsarx, main


def _legacy(path: Path, *, jerk: float = 3.0, acceleration: float = -20.0) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs.update(complete=1, ffa_version="0.1.0-cpp")
        handle.attrs["param_names"] = ["jerk", "accel", "freq"]
        handle["param_sets"] = np.array([[jerk, acceleration, 100.0], [0, 0, 50.0]])
        handle["snr"] = np.array([10.0, 20.0], dtype=np.float32)
        handle["width"] = [2, 1]
        handle["nbins"] = [64, 64]


def _rows(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


@pytest.mark.parametrize("accel,jerk", [(-20.0, 3.0), (20.0, -3.0), (0.0, 0.0)])
def test_legacy_conversion(tmp_path: Path, accel: float, jerk: float) -> None:
    source, output = tmp_path / "old.h5", tmp_path / "out.cands"
    _legacy(source, jerk=jerk, acceleration=accel)
    summary = export_pulsarx(
        source,
        output,
        dm=42,
        pepoch=60000.5,
        timing_frame="topocentric",
        batch_size=1,
    )
    rows = _rows(output)
    assert summary["count"] == 2
    assert rows.shape == (2, 7)
    assert rows[:, 0].tolist() == [1, 0]
    assert rows[1, 2] == pytest.approx(accel)
    assert rows[1, 4] == pytest.approx(-accel * 100 / C)
    assert rows[1, 5] == pytest.approx(-jerk * 100 / C)
    assert "#Pepoch 60000.5" in output.read_text()
    with CandidateFile(source) as candidates:
        assert candidates.legacy
        assert len(list(candidates.iter_batches(1))) == 2


def test_export_requires_metadata_and_preserves_previous_output(tmp_path: Path) -> None:
    source, output = tmp_path / "old.h5", tmp_path / "out.cands"
    _legacy(source)
    output.write_text("previous output\n")
    with pytest.raises(ValueError, match="DM is unknown"):
        export_pulsarx(source, output)
    with pytest.raises(ValueError, match="Timing frame is unknown"):
        export_pulsarx(source, output, dm=42)
    with pytest.raises(ValueError, match="Reference epoch is unknown"):
        export_pulsarx(source, output, dm=42, timing_frame="topocentric")
    assert output.read_text() == "previous output\n"
    assert not list(tmp_path.glob(".loki-export-*"))


def test_export_limits_and_validation(tmp_path: Path) -> None:
    source, output = tmp_path / "old.h5", tmp_path / "out.cands"
    _legacy(source)
    export_pulsarx(
        source, output, dm=0, pepoch=60000, timing_frame="topocentric", max_candidates=1
    )
    assert _rows(output)[0, 6] == 20
    with h5py.File(source, "r+") as handle:
        handle["snr"][0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        export_pulsarx(source, output, dm=0, pepoch=60000, timing_frame="topocentric")
    assert _rows(output)[0, 6] == 20
    with h5py.File(source, "r+") as handle:
        handle.attrs["complete"] = 0
    with pytest.raises(ValueError, match="incomplete"), CandidateFile(source):
        pass


def test_higher_derivatives_cannot_be_silently_truncated(tmp_path: Path) -> None:
    source, output = tmp_path / "old.h5", tmp_path / "out.cands"
    _legacy(source)
    with h5py.File(source, "r+") as handle:
        del handle["param_sets"]
        handle["param_sets"] = [[1, 0, 0, 100], [0, 0, 0, 50]]
        handle.attrs["param_names"] = ["snap", "jerk", "accel", "freq"]
    with pytest.raises(ValueError, match="above jerk"):
        export_pulsarx(source, output, dm=42, pepoch=60000, timing_frame="topocentric")
    assert not output.exists()


@pytest.mark.parametrize("grouping", ["transitive", "anchored"])
def test_ffa_common_schema_and_repeated_execution(
    tmp_path: Path, grouping: str
) -> None:
    rng = np.random.default_rng(45)
    n = 8192
    samples = rng.normal(size=n).astype(np.float32)
    samples[np.arange(n) % 100 < 5] += 2
    cfg = libloki.configs.FFASearchConfig(
        n,
        0.001,
        32,
        1,
        np.array([[9.5, 10.5]]),
        use_fourier=True,
        nthreads=1,
        snr_min=4,
    )
    options = libloki.configs.CandidateOptions()
    options.memory_gb = 0.01
    options.grouping = grouping
    observation = libloki.io.ObservationMetadata()
    observation.dm = 42.5
    observation.start_mjd = 60000
    observation.barycentric = False
    observation.original_nsamps = n + 100
    sweep = libloki.ffa.FFAFreqSweep(cfg, show_progress=False, candidates=options)
    for prefix in ("first", "second"):
        sweep.execute(
            samples, np.ones(n, np.float32), tmp_path, prefix, observation=observation
        )
    with CandidateFile(tmp_path / "first_ffa_results.h5") as candidates:
        assert not candidates.legacy
        assert len(candidates) > 0
        assert candidates.metadata["dm"] == 42.5
        assert candidates.metadata["reference_mjd"] == pytest.approx(
            60000 + n * 0.001 / 172800
        )
    with (
        h5py.File(tmp_path / "first_ffa_results.h5") as first,
        h5py.File(tmp_path / "second_ffa_results.h5") as second,
    ):
        assert first["detections/snr"].size > first["candidates/snr"].size
        np.testing.assert_array_equal(
            first["candidates/id"][:], second["candidates/id"][:]
        )
        np.testing.assert_array_equal(
            first["candidates/snr"][:], second["candidates/snr"][:]
        )
        assert first["clusters/member_count"][:].sum() == first["detections/snr"].size
        assert first["processing"].attrs["grouping"] == grouping
        raw = first["detections"]
        f = raw["f0"][:]
        duty = raw["duty_cycle"][:]
        parent = np.arange(len(f))

        def root(i: int) -> int:
            while parent[i] != i:
                i = int(parent[i])
            return i

        representatives: list[int] = []
        for i in range(len(f)):
            neighbors = np.flatnonzero(
                np.abs(f[:i] - f[i]) * (n * 0.001)
                <= np.maximum(0.05, np.minimum(duty[:i], duty[i]))
            )
            if grouping == "transitive":
                for j in neighbors:
                    a, b = root(i), root(int(j))
                    parent[max(a, b)] = min(a, b)
            else:
                matches = [j for j in representatives if j in neighbors]
                if matches:
                    parent[i] = matches[0]
                else:
                    representatives.append(i)
        roots = np.array([root(i) for i in range(len(f))])
        expected_reps = np.unique(roots)
        expected_ids = np.searchsorted(expected_reps, roots)
        np.testing.assert_array_equal(raw["cluster_id"][:], expected_ids)
        np.testing.assert_array_equal(
            first["clusters/id"][:], raw["id"][:][expected_reps]
        )
        extents = [
            np.max(np.abs(f[roots == i] - f[i])) * n * 0.001 for i in expected_reps
        ]
        np.testing.assert_allclose(
            first["clusters/phase_extent"][:], extents, atol=1e-12
        )

    output = tmp_path / "pulsarx.cands"
    export_pulsarx(tmp_path / "first_ffa_results.h5", output)
    assert _rows(output).shape[1] == 7
    assert (
        main(
            [
                str(tmp_path / "first_ffa_results.h5"),
                str(output),
                "--max-candidates",
                "1",
            ]
        )
        == 0
    )
    assert _rows(output).shape[0] == 1
    with pytest.raises(ValueError, match="frames differ"):
        export_pulsarx(
            tmp_path / "first_ffa_results.h5", output, timing_frame="barycentric"
        )


def _tim_header(path: Path, *, metadata: bool) -> None:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<i", len(data)) + data

    data = string("HEADER_START")
    for key, value in (("nbits", 32), ("nchans", 1), ("nifs", 1), ("data_type", 2)):
        data += string(key) + struct.pack("<i", value)
    data += string("tsamp") + struct.pack("<d", 0.001)
    if metadata:
        data += string("tstart") + struct.pack("<d", 60000)
        data += string("refdm") + struct.pack("<d", 0)
        data += string("barycentric") + struct.pack("<i", 0)
    data += string("HEADER_END") + np.ones(64, np.float32).tobytes()
    path.write_bytes(data)


def test_timeseries_metadata_preserves_unknown_and_zero_dm(tmp_path: Path) -> None:
    options = libloki.io.ReadOptions()
    options.preprocess = False
    for known in (True, False):
        path = tmp_path / f"input_{known}.tim"
        _tim_header(path, metadata=known)
        ts = libloki.io.TimeSeries.read(path, options)
        assert ts.observation.dm == (0 if known else None)
        assert ts.observation.start_mjd == (60000 if known else None)
        assert ts.observation.barycentric == (False if known else None)


def test_presto_observation_metadata(tmp_path: Path) -> None:
    source = tmp_path / "trial.dat"
    np.ones(64, np.float32).tofile(source)
    source.with_suffix(".inf").write_text(
        "Data file name without suffix = trial\n"
        "Number of bins in the time series = 64\n"
        "Width of each time series bin (sec) = 0.001\n"
        "Epoch of observation (MJD) = 60001.25\n"
        "Dispersion measure (cm-3 pc) = 12.5\n"
        "Barycentered? (1=yes, 0=no) = 0\n",
    )
    options = libloki.io.ReadOptions()
    options.preprocess = False
    series = libloki.io.TimeSeries.read(source, options)
    assert series.observation.dm == 12.5
    assert series.observation.start_mjd == 60001.25
    assert series.observation.barycentric is False
    assert series.observation.original_nsamps == 64


def test_export_retains_unsigned_ids_and_rejects_malformed_columns(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "common.h5", tmp_path / "out.cands"
    with h5py.File(source, "w") as handle:
        handle.attrs.update(schema_version="1.0", complete=1, param_names=["freq"])
        handle.create_group("observation").attrs.update(dm=0, barycentric=False)
        group = handle.create_group("candidates")
        for name, data in {
            "snr": [10.0],
            "width": [1],
            "nbins": [64],
            "param_sets": [[100.0]],
            "id": np.array([2**64 - 1], np.uint64),
            "cluster_id": [0],
            "f0": [100.0],
            "f1": [0.0],
            "f2": [0.0],
            "duty_cycle": [1 / 64],
        }.items():
            group[name] = data
    export_pulsarx(source, output)
    row = next(
        line for line in output.read_text().splitlines() if not line.startswith("#")
    )
    assert row.split()[0] == str(2**64 - 1)
    with h5py.File(source, "r+") as handle:
        handle["candidates/f1"][0] = 1
    with pytest.raises(ValueError, match="disagrees with native"):
        export_pulsarx(source, output, pepoch=60000)
