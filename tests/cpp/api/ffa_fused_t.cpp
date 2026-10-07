#include <cstring>
#include <optional>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::FFATime;
using loki::search::PulsarSearchConfig;

namespace {

[[nodiscard]] std::vector<float> run_ffa(const PulsarSearchConfig& cfg,
                                         const std::vector<float>& ts_e,
                                         const std::vector<float>& ts_v,
                                         std::optional<SizeType> fuse_levels) {
    FFATime ffa(cfg, /*show_progress=*/false);
    ffa.set_fuse_levels(fuse_levels);
    std::vector<float> fold(ffa.get_plan().get_buffer_size(), 0.0F);
    ffa.execute(ts_e, ts_v, fold);
    // Only the final-level fold is meaningful.
    fold.resize(ffa.get_plan().get_fold_size());
    return fold;
}

} // namespace

TEST_CASE("Fused brute fold + FFA levels is bit-exact with the plain path",
          "[ffa][fused]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.1F * (dist(rng) * dist(rng)));
    }

    for (const SizeType bseg_brute : {SizeType{64}, SizeType{256}}) {
        for (const int nthreads : {1, 4}) {
            const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 14.0}};
            const PulsarSearchConfig cfg(
                kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5, limits,
                /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/false, nthreads,
                /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
                /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, bseg_brute);
            const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
            for (const SizeType k : {
                     SizeType{1},
                     SizeType{2},
                     SizeType{3},
                     SizeType{4},
                     SizeType{5},
                 }) {
                const auto fused = run_ffa(cfg, ts_e, ts_v, k);
                REQUIRE(fused.size() == reference.size());
                const bool equal =
                    std::memcmp(fused.data(), reference.data(),
                                reference.size() * sizeof(float)) == 0;
                INFO("bseg_brute=" << bseg_brute << " nthreads=" << nthreads
                                   << " fuse_levels=" << k);
                CHECK(equal);
            }
            // The automatic choice must agree as well.
            const auto automatic = run_ffa(cfg, ts_e, ts_v, std::nullopt);
            CHECK(std::memcmp(automatic.data(), reference.data(),
                              reference.size() * sizeof(float)) == 0);
        }
    }
}
