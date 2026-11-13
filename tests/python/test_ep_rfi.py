"""[ep.rfi] in the EP config and the harvest defaults of PruneRFIConfig."""

import pytest

from loki import libloki

DOC = """
[input]
nsamps = 65536
tsamp = 0.000064
[search]
f_min = 140.0
f_max = 142.0
acc_min = -10.0
acc_max = 10.0
nbins = 32
[ep]
n_runs = 2
nsegments = 8
{rfi}
[performance]
nthreads = 2
"""


def test_ep_rfi_table_is_optional():
    cfg = libloki.configs.EPTomlConfig.from_string(DOC.format(rfi=""))
    assert cfg.rfi is None
    assert not cfg.prune_rfi_config().is_active
    assert "rfi" not in cfg.to_toml_string()


def test_ep_rfi_table_reads_and_maps_to_the_library():
    rfi = """[ep.rfi]
pulsar_mask = [[49.9, 50.1]]
harvest = true
harvest_min_level = 4
harvest_min_snr = 12.0
impulsive_veto = true
"""
    cfg = libloki.configs.EPTomlConfig.from_string(DOC.format(rfi=rfi))
    assert cfg.rfi.is_active
    assert cfg.rfi.pulsar_mask == [[49.9, 50.1]]
    lib = cfg.prune_rfi_config()
    assert lib.has_harvest
    assert lib.harvest_scheme == []
    assert lib.harvest_defaults.min_level == 4
    assert lib.harvest_defaults.min_snr == 12.0
    assert lib.pulsar_mask[0].f_lo == 49.9
    again = libloki.configs.EPTomlConfig.from_string(cfg.to_toml_string())
    assert again.to_toml_string() == cfg.to_toml_string()

    cfg.backend = "cuda"
    with pytest.raises(ValueError, match="CPU only"):
        cfg.check_rfi_backend()


def test_ep_rfi_table_is_strict():
    with pytest.raises(ValueError, match="ep.rfi.harvst"):
        libloki.configs.EPTomlConfig.from_string(
            DOC.format(rfi="[ep.rfi]\nharvst = true\n")
        )


def test_harvest_defaults_expand_from_a_threshold_scheme():
    defaults = libloki.prune.HarvestDefaults(min_level=2, offset=1.0, min_snr=3.0)
    rfi = libloki.prune.PruneRFIConfig(harvest_defaults=defaults)
    assert rfi.has_harvest and rfi.is_active
    expanded = rfi.expand_harvest([1.0, 2.5, 3.0])
    assert expanded.harvest_defaults is None
    assert expanded.harvest_scheme == [libloki.prune.HARVEST_DISABLED, 3.5, 4.0]
    expanded.validate(4)
