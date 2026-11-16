import h5py
import numpy as np
import pytest

from loki import libloki

io = libloki.io


def _write(path, dm, start_mjd, rows):
    with h5py.File(path, "w") as h:
        h.attrs["schema_version"] = "1.0"
        obs = h.create_group("observation")
        obs.attrs["dm"] = dm
        obs.attrs["start_mjd"] = start_mjd
        obs.attrs["reference_mjd"] = start_mjd + 500.0 / 86400.0
        obs.attrs["tobs"] = 1000.0
        c = h.create_group("candidates")
        c["f0"] = np.array([r[0] for r in rows], dtype=np.float64)
        c["f1"] = np.zeros(len(rows))
        c["snr"] = np.array([r[1] for r in rows], dtype=np.float32)
        c["duty_cycle"] = np.full(len(rows), 0.1)
        c["id"] = np.arange(len(rows), dtype=np.uint64)


def test_sift_groups_over_dm(tmp_path) -> None:
    files = []
    for dm, snr in ((10.0, 9.0), (20.0, 12.0), (30.0, 9.0)):
        path = tmp_path / f"dm{dm:g}.h5"
        _write(path, dm, 60000.0, [(100.0, snr), (200.0, snr - 2)])
        files.append(str(path))
    result = io.sift(files)
    assert result.n_candidates == 6
    assert len(result.groups) == 2
    psr, harmonic = result.groups
    assert psr.members[0].snr == pytest.approx(12.0)
    assert [dm for dm, _ in psr.dm_curve] == [10.0, 20.0, 30.0]
    assert psr.edge_ratio == pytest.approx(0.75)
    assert (harmonic.harmonic_of, harmonic.harmonic_numerator) == (0, 2)
    options = io.SiftOptions()
    options.snr_min = 11.0
    assert len(io.sift(files, options).groups) == 1


def test_sift_matches_a_catalogue(tmp_path) -> None:
    from loki import sift as loki_sift  # noqa: PLC0415

    files = []
    for dm in (10.0, 20.0):
        path = tmp_path / f"dm{dm:g}.h5"
        _write(path, dm, 60000.0, [(100.001, 12.0), (50.0005, 9.0), (77.0, 8.0)])
        files.append(str(path))
    catalog = tmp_path / "cat.csv"
    catalog.write_text("name,f0,p0_ms,dm,doppler\nA,100.0,,15,1e-5\nB,,25.0,30,\n")
    known = loki_sift.load_catalog(catalog)
    assert known[1].f0 == pytest.approx(40.0)
    result = io.sift(files)
    matches = loki_sift.match_groups(result, known)
    by_freq = {
        round(g.members[0].freq): m for g, m in zip(result.groups, matches, strict=True)
    }
    assert by_freq[100].name == "A"
    assert by_freq[100].ratio == 1
    assert by_freq[50].ratio == loki_sift.Fraction(1, 2)
    assert by_freq[77] is None
    # Without its orbital tolerance, A matches only after the Earth Doppler
    # shift of 1e-5 is divided out.
    isolated = [loki_sift.KnownPulsar("A", 100.0)]
    assert loki_sift.match_groups(result, isolated, tolerance=1e-7)[0] is None
    shifted = loki_sift.match_groups(
        result, isolated, doppler=lambda _mjd: 1e-5, tolerance=1e-7
    )
    assert shifted[0] is not None
    assert shifted[0].name == "A"
    assert loki_sift.main([*files, "--catalog", str(catalog), "--top", "2"]) == 0
