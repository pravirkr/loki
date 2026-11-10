#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/search/configs.hpp"

#include "temp_dir.hpp"

namespace {

/// A complete EP document. Each part is spliced into its table, so a test can
/// add or replace one key without writing a second [ep] or [performance].
std::string make_doc(const std::string& ep           = "",
                     const std::string& perf         = "nthreads = 2\n",
                     const std::string& search_extra = "",
                     const std::string& tables       = "") {
    return "[input]\n"
           "nsamps = 65536\n"
           "tsamp = 0.000064\n"
           "[search]\n"
           "f_min = 140.0\n"
           "f_max = 142.0\n"
           "acc_min = -10.0\n"
           "acc_max = 10.0\n"
           "nbins = 32\n" +
           search_extra +
           "[ep]\n"
           "n_runs = 2\n"
           "nsegments = 8\n" +
           ep + "[performance]\n" + perf + tables;
}

} // namespace

TEST_CASE("EPTomlConfig reads a complete document", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(make_doc());
    REQUIRE(cfg.poly_basis == "taylor");
    REQUIRE(cfg.n_runs.value() == 2);
    REQUIRE(cfg.nthreads == 2);
    REQUIRE(cfg.nbins == 32);
    REQUIRE(cfg.prune_poly_order == 3);
    REQUIRE(cfg.use_conservative_tile == false);
    REQUIRE_FALSE(cfg.plan_cache.has_value());

    const auto search = cfg.to_ep_search_config();
    REQUIRE(search.get_nbins() == 32);
    REQUIRE(search.get_nsamps() == 65536);
    REQUIRE(search.get_prune_poly_order() == 3);
}

TEST_CASE("EPTomlConfig rejects unknown and removed keys", "[config][ep]") {
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("bogus = 1\n")),
        std::invalid_argument);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "", "", "[cuda]\nenable = true\n")),
                      std::invalid_argument);
}

TEST_CASE("EP candidate controls do not replace the pruning target",
          "[config][ep][candidates]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(make_doc(
        "target_snr = 8.5\n", "nthreads = 2\n", "snr_min = 12.0\n",
        "[candidates]\ncluster = false\ngrouping = \"anchored\"\n"
        "memory_gb = 0.02\nretain_detections = false\nmax_candidates = 3\n"));
    CHECK_FALSE(cfg.candidates.cluster);
    CHECK(cfg.candidates.grouping == "anchored");
    CHECK(cfg.candidates.memory_gb == 0.02);
    CHECK_FALSE(cfg.candidates.retain_detections);
    CHECK(cfg.candidates.max_candidates == 3);
    const auto search = cfg.to_ep_search_config();
    CHECK(search.get_target_snr() == 8.5);
    CHECK(search.get_snr_min() == 12.0);
    CHECK_THROWS_AS(loki::search::EPTomlConfig::from_string(
                        make_doc("", "", "", "[candidates]\nunknown = true\n")),
                    std::invalid_argument);
}

TEST_CASE("EPTomlConfig reads the [preprocessing] table", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nthreads = 2\n", "",
                 "[preprocessing]\nmethod = \"zscore\"\nclip_sigma = 4.0\n"));
    REQUIRE(cfg.preprocessing.method == loki::io::PreprocessMethod::kZScore);
    REQUIRE(cfg.preprocessing.clip_sigma == 4.0);
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc(
            "", "nthreads = 2\n", "", "[preprocessing]\nblock_sigma = -1.0\n")),
        std::invalid_argument);
}

TEST_CASE("EPTomlConfig rejects RFI options, which the EP search lacks",
          "[config][ep]") {
    const auto message_for = [](const std::string& toml) -> std::string {
        try {
            (void)loki::search::EPTomlConfig::from_string(toml);
        } catch (const std::invalid_argument& err) {
            return err.what();
        }
        return {};
    };
    REQUIRE(message_for(make_doc("harvest_scheme = [5.0]\n")).find("RFI") !=
            std::string::npos);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "", "", "[rfi]\npulsar_mask = []\n")),
                      std::invalid_argument);
}

TEST_CASE("The FFA reader rejects an [ep] table", "[config][ep]") {
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(
                          "[search]\nf_min = 1.0\nf_max = 10.0\n[ep]\n"
                          "n_runs = 2\n"),
                      std::invalid_argument);
}

TEST_CASE("EPTomlConfig checks the pruning parameters", "[config][ep]") {
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("poly_basis = \"bogus\"\n")),
                      std::invalid_argument);
    REQUIRE_NOTHROW(loki::search::EPTomlConfig::from_string(
        make_doc("poly_basis = \"chebyshev\"\n")));

    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("min_pd = 0.0\n")),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("ref_ducy = 1.5\n")),
        std::invalid_argument);
    REQUIRE_NOTHROW(
        loki::search::EPTomlConfig::from_string(make_doc("min_pd = 1.0\n")));
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("prune_poly_order = 0\n")),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("m_c_max = nan\n")),
        std::invalid_argument);
}

TEST_CASE("EPTomlConfig rejects nan and inf by their literal", "[config][ep]") {
    // toml++'s NaN and Inf are garbage under -ffast-math: the reader must
    // reject the literal itself, with either sign, before any other check.
    using Catch::Matchers::ContainsSubstring;
    const auto ep_doc = [](const std::string& ep) {
        return loki::search::EPTomlConfig::from_string(make_doc(ep));
    };
    for (const auto* value : {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}) {
        CAPTURE(value);
        REQUIRE_THROWS_WITH(ep_doc(std::string("min_pd = ") + value + "\n"),
                            ContainsSubstring("ep.min_pd must be finite"));
    }
    // Positions count code points: non-ASCII text earlier on the line does
    // not hide the literal, nor flag a finite value (the unknown key is then
    // the error).
    REQUIRE_THROWS_WITH(ep_doc("\"\xC3\xA9\xC3\xA9\" = [1.5, inf]\n"),
                        ContainsSubstring("[1] must be finite"));
    REQUIRE_THROWS_WITH(ep_doc("\"\xC3\xA9\xC3\xA9\" = [1.5, 2.5]\n"),
                        !ContainsSubstring("finite"));
    REQUIRE_NOTHROW(ep_doc("target_snr = 9.5 # \xC3\xA9 inf nan\n"));
}

TEST_CASE("EPTomlConfig reads whole numbers as reals", "[config][ep]") {
    const auto cfg =
        loki::search::EPTomlConfig::from_string(make_doc("m_c_max = 12\n"));
    REQUIRE(cfg.m_c_max == 12.0);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("p_orb_min = \"x\"\n")),
                      std::invalid_argument);
}

TEST_CASE("EPTomlConfig takes the runs from the file or the command line",
          "[config][ep]") {
    // Neither set: the search config does not depend on the runs, so it
    // builds (the CLI then defaults to 16 runs, the library to every
    // reference segment).
    const std::string no_runs = "[input]\n"
                                "nsamps = 65536\n"
                                "tsamp = 0.000064\n"
                                "[search]\n"
                                "f_min = 140.0\n"
                                "f_max = 142.0\n"
                                "nbins = 32\n"
                                "acc_min = -10.0\n"
                                "acc_max = 10.0\n"
                                "[ep]\n"
                                "nsegments = 8\n";
    const auto cfg = loki::search::EPTomlConfig::from_string(no_runs);
    REQUIRE_NOTHROW(cfg.to_ep_search_config());

    // A command line value is applied after the parse, so it can complete it.
    auto with_runs   = cfg;
    with_runs.n_runs = 4;
    REQUIRE_NOTHROW(with_runs.to_ep_search_config());

    // Both set is ambiguous.
    auto both     = with_runs;
    both.ref_segs = std::vector<loki::SizeType>{0, 1};
    REQUIRE_THROWS_AS(both.to_ep_search_config(), std::invalid_argument);
}

TEST_CASE("EPTomlConfig validates explicit reference segments",
          "[config][ep]") {
    // n_runs is set by make_doc, so these cases only test ref_segs itself.
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("ref_segs = []\n")),
        std::invalid_argument);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("ref_segs = [1, 1]\n")),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("ref_segs = [0, \"a\"]\n")),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("ref_segs = 1\n")),
        std::invalid_argument);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("ref_segs = [0, true]\n")),
                      std::invalid_argument);

    const auto cfg =
        loki::search::EPTomlConfig::from_string("[input]\n"
                                                "nsamps = 65536\n"
                                                "tsamp = 0.000064\n"
                                                "[search]\n"
                                                "f_min = 140.0\n"
                                                "f_max = 142.0\n"
                                                "nbins = 32\n"
                                                "[ep]\n"
                                                "ref_segs = [0, 2]\n"
                                                "nsegments = 8\n");
    REQUIRE(cfg.ref_segs.has_value());
    REQUIRE(cfg.ref_segs->size() == 2);
    REQUIRE_FALSE(cfg.n_runs.has_value());
}

TEST_CASE("EP search needs at least two power-of-2 segments", "[config][ep]") {
    const auto doc = [](const std::string& nsegments) {
        return "[input]\nnsamps = 65536\ntsamp = 0.000064\n"
               "[search]\nf_min = 140.0\nf_max = 142.0\nacc_min = -10.0\n"
               "acc_max = 10.0\nnbins = 32\n[ep]\nn_runs = 2\nnsegments = " +
               nsegments + "\n";
    };
    // One segment leaves no pruning levels.
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(doc("1")),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(doc("12")),
                      std::invalid_argument);
    REQUIRE_NOTHROW(loki::search::EPTomlConfig::from_string(doc("16"))
                        .to_ep_search_config());
    // The removed key says what replaces it.
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "nthreads = 2\nbseg_ffa = 8192\n")),
                      std::invalid_argument);
}

TEST_CASE("EP search rejects Kadane peak detection", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nthreads = 2\n", "use_boxcar_kadane = true\n"));
    REQUIRE_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
}

TEST_CASE("EPTomlConfig reads the backend, device and plan cache",
          "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nthreads = 2\nbackend = \"cuda\"\ndevice = 1\n", "",
                 "[output]\nplan_cache = \"plans/ep.h5\"\n"));
    REQUIRE(cfg.backend == loki::Backend::kCUDA);
    REQUIRE(cfg.device == 1);
    REQUIRE(cfg.plan_cache.has_value());
    REQUIRE(cfg.plan_cache->string() == "plans/ep.h5");

    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "nthreads = 2\nbackend = \"gpu\"\n")),
                      std::invalid_argument);
}

TEST_CASE("The default EP template requires a physical acceleration range",
          "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        std::string(loki::search::EPTomlConfig::default_toml_string()));
    REQUIRE_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
    REQUIRE(cfg.n_runs.value() == 16);
}

TEST_CASE("EPTomlConfig::write_default writes a document that loads",
          "[config][ep]") {
    const loki::test::TempDir dir("loki-ep-config");
    const auto path = dir / "ep_config.toml";
    loki::search::EPTomlConfig::write_default(path);
    const auto cfg = loki::search::EPTomlConfig::load(path);
    REQUIRE_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
}

TEST_CASE("EP threshold simulation has an independent target SNR",
          "[config][ep]") {
    const auto defaults = loki::search::EPTomlConfig::from_string(make_doc())
                              .to_ep_search_config();
    REQUIRE(defaults.get_target_snr() == 8.5);
    REQUIRE(defaults.get_snr_min() == 7.0);
    const auto custom =
        loki::search::EPTomlConfig::from_string(
            make_doc("target_snr = 12\n", "nthreads = 2\n", "snr_min = 6.0\n"))
            .to_ep_search_config();
    REQUIRE(custom.get_target_snr() == 12.0);
    REQUIRE(custom.get_snr_min() == 6.0);
    const auto updated = custom.get_updated_ep_config(64, 1.0, 140.0, 141.0);
    REQUIRE(updated.get_target_snr() == 12.0);
    REQUIRE(updated.get_snr_min() == 6.0);
    const auto copied = custom;
    REQUIRE(copied.get_target_snr() == 12.0);
    for (const auto value : {"0.0", "-1.0", "nan", "inf"}) {
        REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(make_doc(
                              std::string("target_snr = ") + value + "\n")),
                          std::invalid_argument);
    }
}

TEST_CASE("EP overflow keep fraction defaults to the median and is kept",
          "[config][ep]") {
    const auto defaults = loki::search::EPTomlConfig::from_string(make_doc())
                              .to_ep_search_config();
    REQUIRE(defaults.get_overflow_keep_fraction() == 0.5);
    const auto custom = loki::search::EPTomlConfig::from_string(
                            make_doc("overflow_keep_fraction = 0.9\n"))
                            .to_ep_search_config();
    REQUIRE(custom.get_overflow_keep_fraction() == 0.9);
    const auto updated = custom.get_updated_ep_config(64, 1.0, 140.0, 141.0);
    REQUIRE(updated.get_overflow_keep_fraction() == 0.9);
    const auto copied = custom;
    REQUIRE(copied.get_overflow_keep_fraction() == 0.9);
    REQUIRE_NOTHROW(loki::search::EPTomlConfig::from_string(
        make_doc("overflow_keep_fraction = 1\n")));
    for (const auto value : {"0.0", "-0.5", "1.5", "nan", "inf"}) {
        CAPTURE(value);
        REQUIRE_THROWS_AS(
            loki::search::EPTomlConfig::from_string(make_doc(
                std::string("overflow_keep_fraction = ") + value + "\n")),
            std::invalid_argument);
    }
}

TEST_CASE("EP target SNR defaults follow the search parameter count",
          "[config][ep]") {
    const std::vector<double> expected{8.5, 8.5, 9.2, 9.8, 10.0};
    for (loki::SizeType nparams = 1; nparams <= expected.size(); ++nparams) {
        std::vector<loki::ParamLimit> limits(nparams, {.min = 0.0, .max = 0.0});
        limits.back() = {.min = 140.0, .max = 142.0};
        const loki::search::FFASearchConfig ffa(65536, 64e-6, 32, 1.0, limits);
        // Keep the default pruning-order metadata: dispatch uses nparams.
        const loki::search::EPSearchConfig cfg(ffa);
        REQUIRE(cfg.get_target_snr() == expected[nparams - 1]);
        REQUIRE(
            cfg.get_updated_ep_config(64, 1.0, 140.0, 141.0).get_target_snr() ==
            expected[nparams - 1]);
        const loki::search::EPSearchConfig override(ffa, 3, 1e-5, 10.0, 1.4,
                                                    2.0, 5.0, false, 11.0);
        REQUIRE(override.get_target_snr() == 11.0);
        const std::vector<loki::ParamLimit> freq_limits{
            {.min = 140.0, .max = 142.0}};
        REQUIRE(
            cfg.get_updated_ep_config(32, 1.0, freq_limits).get_target_snr() ==
            8.5);
        REQUIRE(override.get_updated_ep_config(32, 1.0, freq_limits)
                    .get_target_snr() == 11.0);
    }
    REQUIRE_THROWS_AS(
        loki::search::EPTomlConfig::from_string(make_doc("snr_final = 10\n")),
        std::invalid_argument);
    const auto toml = loki::search::EPTomlConfig::from_string(make_doc());
    REQUIRE_FALSE(toml.target_snr.has_value());
}

TEST_CASE("EP config rejects bin limits above 1024", "[config][ep]") {
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "nbins_max = 1025\n")),
                      std::invalid_argument);
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nbins_max = 1024\n"));
    REQUIRE_NOTHROW(cfg.to_ep_search_config());
}

TEST_CASE("EP config defaults to 128 FFA segments when nsegments is omitted",
          "[config][ep]") {
    // Power-of-two nsamps: 128 segments exactly.
    const std::string doc_pow2 = "[input]\n"
                                 "nsamps = 1048576\n"
                                 "tsamp = 0.000064\n"
                                 "[search]\n"
                                 "f_min = 140.0\n"
                                 "f_max = 142.0\n"
                                 "acc_min = -10.0\n"
                                 "acc_max = 10.0\n"
                                 "nbins = 32\n"
                                 "[ep]\n"
                                 "n_runs = 2\n"
                                 "[performance]\n"
                                 "nthreads = 1\n";
    const auto toml_pow2 = loki::search::EPTomlConfig::from_string(doc_pow2);
    REQUIRE_FALSE(toml_pow2.nsegments.has_value());
    const auto ep_pow2 = toml_pow2.to_ep_search_config();
    REQUIRE(ep_pow2.get_nsegments() ==
            loki::search::EPSearchConfig::kDefaultEPSegments);
    REQUIRE(ep_pow2.get_bseg_ffa() ==
            1048576U / loki::search::EPSearchConfig::kDefaultEPSegments);
    const loki::plans::FFAPlan<float> plan_pow2(ep_pow2);
    REQUIRE(plan_pow2.get_nsegments_active() ==
            loki::search::EPSearchConfig::kDefaultEPSegments);

    // Non-power-of-two nsamps: 128 equal, full segments of a whole number
    // of brute segments; the tail that fills no brute segment of the finest
    // tree (fewer than 2^L samples) is dropped.
    const std::vector<loki::ParamLimit> limits{
        {.min = -10.0, .max = 10.0},
        {.min = 140.0, .max = 142.0},
    };
    const loki::search::PulsarSearchConfig raw_cfg(70000, 64e-6, 32, 1.0,
                                                   limits);
    REQUIRE_FALSE(raw_cfg.has_explicit_nsegments());
    REQUIRE(raw_cfg.get_nsegments() == 1U);
    const auto norm_cfg = raw_cfg.with_ep_default_nsegments();
    REQUIRE(norm_cfg.has_explicit_nsegments());
    REQUIRE(norm_cfg.get_nsegments() == 128U);
    REQUIRE(norm_cfg.get_nsamps_input() == 70000U);
    REQUIRE(norm_cfg.get_nsamps() == norm_cfg.get_bseg_ffa() * 128U);
    REQUIRE(norm_cfg.get_nsamps() <= 70000U);
    const loki::SizeType nseg_brute =
        norm_cfg.get_nsamps() / norm_cfg.get_bseg_brute();
    REQUIRE(70000U - norm_cfg.get_nsamps() < nseg_brute);
    // FFA searches keep the whole series.
    REQUIRE(raw_cfg.get_nsamps() == 70000U);
    const loki::plans::FFAPlan<float> plan_np2(norm_cfg);
    REQUIRE(plan_np2.get_nsegments().back() == 128U);
    REQUIRE(plan_np2.get_nsegments_active() == 128U);
}

TEST_CASE("EP chunk configs keep the EP segments of the base config",
          "[config][ep]") {
    // Non-power-of-two nsamps: each chunk picks its own brute length, but
    // only among bseg_ffa >> k, so segment k covers the same samples in
    // every chunk.
    const std::vector<loki::ParamLimit> limits{
        {.min = -10.0, .max = 10.0},
        {.min = 140.0, .max = 142.0},
    };
    const auto base =
        loki::search::PulsarSearchConfig(70000, 64e-6, 32, 1.0, limits)
            .with_ep_default_nsegments();
    // Unconstrained, the 20-30 Hz chunk would take other EP segments.
    const auto slow = base.get_updated_ep_config(32, 1.0, 20.0, 30.0);
    const auto fast = base.get_updated_ep_config(32, 1.0, 500.0, 700.0);
    CAPTURE(base.get_bseg_brute(), base.get_bseg_ffa(), slow.get_bseg_brute(),
            fast.get_bseg_brute());
    REQUIRE(slow.get_nsegments() == base.get_nsegments());
    REQUIRE(fast.get_nsegments() == base.get_nsegments());
    REQUIRE(slow.get_bseg_ffa() == base.get_bseg_ffa());
    REQUIRE(fast.get_bseg_ffa() == base.get_bseg_ffa());
    // A fresh config of the slow chunk is not constrained.
    const std::vector<loki::ParamLimit> slow_limits{
        {.min = -10.0, .max = 10.0},
        {.min = 20.0, .max = 30.0},
    };
    const auto fresh =
        loki::search::PulsarSearchConfig(70000, 64e-6, 32, 1.0, slow_limits)
            .with_nsegments(128);
    CAPTURE(fresh.get_bseg_brute(), fresh.get_bseg_ffa());
    REQUIRE(fresh.get_bseg_ffa() != base.get_bseg_ffa());
}

TEST_CASE("EPSearchConfig TOML helpers round-trip", "[config][ep]") {
    const auto default_str =
        loki::search::EPSearchConfig::default_toml_string();
    REQUIRE(default_str.find("# nsegments = 128") != std::string_view::npos);
    REQUIRE(default_str.find("bseg_ffa") == std::string_view::npos);
    const auto ep_cfg =
        loki::search::EPSearchConfig::from_toml_string(make_doc());
    REQUIRE(ep_cfg.get_nbins() == 32U);
    REQUIRE(ep_cfg.get_bseg_ffa() == 8192U);

    const loki::test::TempDir dir("loki-ep-search-cfg");
    const auto path = dir / "ep_search_cfg.toml";
    loki::search::EPSearchConfig::write_default_toml(path);
    REQUIRE(std::filesystem::exists(path));
}
