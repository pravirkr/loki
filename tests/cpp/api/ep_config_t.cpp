#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/search/configs.hpp"

namespace {

/// A complete EP document. Each part is spliced into its table, so a test can
/// add or replace one key without writing a second [ep] or [performance].
std::string make_doc(const std::string& ep           = "",
                     const std::string& perf         = "nthreads = 2\n"
                                                       "bseg_ffa = 8192\n",
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
           "n_runs = 2\n" +
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

TEST_CASE("EPTomlConfig reads the [preprocessing] table", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nthreads = 2\nbseg_ffa = 8192\n", "",
                 "[preprocessing]\nmethod = \"zscore\"\nclip_sigma = 4.0\n"));
    REQUIRE(cfg.preprocessing.method == loki::io::PreprocessMethod::kZScore);
    REQUIRE(cfg.preprocessing.clip_sigma == 4.0);
    REQUIRE_THROWS_AS(loki::search::EPTomlConfig::from_string(
                          make_doc("", "nthreads = 2\nbseg_ffa = 8192\n", "",
                                   "[preprocessing]\nblock_sigma = -1.0\n")),
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
    // Neither set: the document parses, the search config does not.
    const std::string no_runs = "[input]\n"
                                "nsamps = 65536\n"
                                "tsamp = 0.000064\n"
                                "[search]\n"
                                "f_min = 140.0\n"
                                "f_max = 142.0\n"
                                "nbins = 32\n"
                                "[performance]\n"
                                "bseg_ffa = 8192\n";
    const auto cfg = loki::search::EPTomlConfig::from_string(no_runs);
    REQUIRE_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);

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
                                                "[performance]\n"
                                                "bseg_ffa = 8192\n");
    REQUIRE(cfg.ref_segs.has_value());
    REQUIRE(cfg.ref_segs->size() == 2);
    REQUIRE_FALSE(cfg.n_runs.has_value());
}

TEST_CASE("EP search needs an FFA level below the series", "[config][ep]") {
    // bseg_ffa == nsamps leaves one segment and no pruning levels.
    const auto cfg = loki::search::EPTomlConfig::from_string(
        make_doc("", "nthreads = 2\nbseg_ffa = 65536\n"));
    REQUIRE_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
}

TEST_CASE("EP search rejects Kadane peak detection", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(make_doc(
        "", "nthreads = 2\nbseg_ffa = 8192\n", "use_boxcar_kadane = true\n"));
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

TEST_CASE("The default EP document is valid", "[config][ep]") {
    const auto cfg = loki::search::EPTomlConfig::from_string(
        std::string(loki::search::EPTomlConfig::default_toml_string()));
    REQUIRE_NOTHROW(cfg.to_ep_search_config());
    REQUIRE(cfg.n_runs.value() == 16);
}

TEST_CASE("EPTomlConfig::write_default writes a document that loads",
          "[config][ep]") {
    const auto path =
        std::filesystem::temp_directory_path() / "loki_ep_config_t.toml";
    loki::search::EPTomlConfig::write_default(path);
    const auto cfg = loki::search::EPTomlConfig::load(path);
    REQUIRE_NOTHROW(cfg.to_ep_search_config());
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
