#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/planner_cost.hpp"
#include "lib/algorithms/planner_memory.hpp"
#include "lib/detail/utils.hpp"

using Catch::Approx;
using loki::ComplexType;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::FFARegionPlanner;
using loki::algorithms::generate_ffa_regions;
using loki::plans::FFAPlan;
using loki::search::FFASearchConfig;
namespace detail = loki::algorithms::detail;

namespace {

// 2^22 samples at 64 us (tobs ~ 268 s), accel +-200 m/s^2, 20-80 Hz, nbins
// 32: two regions; memory binds below ~1 GB.
constexpr SizeType kNsamps = SizeType{1} << 22U;
constexpr double kTsamp    = 6.4e-5;
constexpr double kAccel    = 200.0;
constexpr double kFMin     = 20.0;
constexpr double kFMax     = 80.0;
constexpr SizeType kNbins  = 32;

FFASearchConfig make_cfg(bool use_fourier, double max_memory_gb, bool accel) {
    std::vector<ParamLimit> limits;
    if (accel) {
        limits.push_back({.min = -kAccel, .max = kAccel});
    }
    limits.push_back({.min = kFMin, .max = kFMax});
    return {kNsamps, kTsamp, kNbins,      /*eta=*/1.0, limits,
            0.2,     1.5,    use_fourier, 1,           max_memory_gb};
}

double max_drift(bool accel) {
    constexpr double kC = 299792458.0;
    return accel ? kAccel * (static_cast<double>(kNsamps) * kTsamp / 2.0) / kC
                 : 0.0;
}

template <typename FoldType>
double modelled_cost(const FFARegionPlanner<FoldType>& planner,
                     bool use_fourier) {
    double cost = 0.0;
    for (const auto& cfg : planner.get_cfgs()) {
        cost += detail::estimate_chunk_cost(
            FFAPlan<FoldType>(cfg), detail::cpu_chunk_cost_model(use_fourier));
    }
    return cost;
}

double sweep_budget_gb(double max_memory_gb) {
    const double input_gb = static_cast<double>(kNsamps) * 2.0 * sizeof(float) /
                            loki::utils::kBytesPerGB;
    return detail::effective_memory_limit_gb(max_memory_gb) - input_gb;
}

// Chunks run from the top of each region down; their nominal ranges tile the
// region list exactly.
template <typename FoldType>
void check_covers_regions(const FFARegionPlanner<FoldType>& planner,
                          bool accel) {
    const auto regions = generate_ffa_regions(1.0 / kFMax, 1.0 / kFMin, kTsamp,
                                              kNbins, /*eta_min=*/1.0);
    const double drift = max_drift(accel);
    const auto& cfgs   = planner.get_cfgs();
    REQUIRE(!cfgs.empty());
    constexpr double kRelTol = 1.0e-9;
    SizeType ichunk          = 0;
    for (const auto& region : regions) {
        double top = region.f_end;
        while (ichunk < cfgs.size() &&
               cfgs[ichunk].get_nbins() == region.nbins &&
               top > region.f_start * (1.0 + kRelTol)) {
            const auto freq         = cfgs[ichunk].get_param_limits().back();
            const double nominal_hi = freq.max / (1.0 + drift);
            const double nominal_lo = freq.min / (1.0 - drift);
            CHECK(std::abs(nominal_hi - top) <= kRelTol * top);
            CHECK(nominal_lo < nominal_hi);
            top = nominal_lo;
            ++ichunk;
        }
        CHECK(std::abs(top - region.f_start) <= kRelTol * region.f_start);
    }
    CHECK(ichunk == cfgs.size());
}

} // namespace

TEST_CASE("FFARegionPlanner CPU: cost-model chunks tile the regions and fit "
          "memory",
          "[regions][planner_cost]") {
    for (const bool use_fourier : {false, true}) {
        for (const double mem_gb : {0.65, 0.7, 0.8, 1.0, 8.0}) {
            CAPTURE(use_fourier, mem_gb);
            const auto cfg = make_cfg(use_fourier, mem_gb, /*accel=*/true);
            if (use_fourier) {
                const FFARegionPlanner<ComplexType> planner(cfg);
                check_covers_regions(planner, true);
                CHECK(planner.get_stats().get_freq_sweep_memory_usage() <=
                      sweep_budget_gb(mem_gb) + 1.0e-6);
            } else {
                const FFARegionPlanner<float> planner(cfg);
                check_covers_regions(planner, true);
                CHECK(planner.get_stats().get_freq_sweep_memory_usage() <=
                      sweep_budget_gb(mem_gb) + 1.0e-6);
            }
        }
    }
}

TEST_CASE("FFARegionPlanner CPU: more memory never costs more and the plan "
          "saturates",
          "[regions][planner_cost]") {
    const std::vector<double> mems = {0.65, 0.7, 0.75, 0.8, 0.9,
                                      1.0,  1.5, 2.0,  8.0};
    for (const bool use_fourier : {false, true}) {
        CAPTURE(use_fourier);
        std::vector<double> costs;
        std::vector<SizeType> nchunks;
        std::vector<std::vector<ParamLimit>> limits_2gb;
        std::vector<std::vector<ParamLimit>> limits_8gb;
        for (const double mem_gb : mems) {
            const auto cfg    = make_cfg(use_fourier, mem_gb, /*accel=*/true);
            const auto record = [&](const auto& planner) {
                costs.push_back(modelled_cost(planner, use_fourier));
                nchunks.push_back(planner.get_nregions());
                for (const auto& chunk : planner.get_cfgs()) {
                    const auto lim = chunk.get_param_limits();
                    if (mem_gb == 2.0) {
                        limits_2gb.emplace_back(lim.begin(), lim.end());
                    } else if (mem_gb == 8.0) {
                        limits_8gb.emplace_back(lim.begin(), lim.end());
                    }
                }
            };
            if (use_fourier) {
                record(FFARegionPlanner<ComplexType>(cfg));
            } else {
                record(FFARegionPlanner<float>(cfg));
            }
        }
        for (SizeType i = 1; i < mems.size(); ++i) {
            CAPTURE(mems[i]);
            CHECK(costs[i] <= costs[i - 1] * (1.0 + 1.0e-12));
            CHECK(nchunks[i] <= nchunks[i - 1]);
        }
        // Memory stops binding: the plan no longer depends on the limit.
        REQUIRE(limits_2gb.size() == limits_8gb.size());
        for (SizeType i = 0; i < limits_2gb.size(); ++i) {
            for (SizeType p = 0; p < limits_2gb[i].size(); ++p) {
                CHECK(limits_2gb[i][p].min == limits_8gb[i][p].min);
                CHECK(limits_2gb[i][p].max == limits_8gb[i][p].max);
            }
        }
    }
}

TEST_CASE("FFARegionPlanner CPU: an accel search is cheaper than one chunk "
          "per region",
          "[regions][planner_cost]") {
    // The accel grid of a chunk is designed at its highest frequency, so the
    // widest chunk oversamples the low end; the cost model splits it even
    // though memory allows one chunk per region.
    const auto cfg = make_cfg(/*use_fourier=*/true, 8.0, /*accel=*/true);
    const FFARegionPlanner<ComplexType> planner(cfg);
    const auto regions = generate_ffa_regions(1.0 / kFMax, 1.0 / kFMin, kTsamp,
                                              kNbins, /*eta_min=*/1.0);
    REQUIRE(planner.get_nregions() > regions.size());

    const double drift = max_drift(true);
    double widest_cost = 0.0;
    for (const auto& region : regions) {
        const auto chunk = cfg.get_updated_config(
            region.nbins, region.eta, region.f_start * (1.0 - drift),
            region.f_end * (1.0 + drift));
        widest_cost += detail::estimate_chunk_cost(
            FFAPlan<ComplexType>(chunk), detail::cpu_chunk_cost_model(true));
    }
    CHECK(modelled_cost(planner, true) < widest_cost);
}

TEST_CASE("FFARegionPlanner CPU: a frequency-only search keeps one chunk per "
          "region when it fits",
          "[regions][planner_cost]") {
    // Without an accel grid nothing oversamples: splitting would only move
    // the frequency grid's origin, so the widest chunks are kept.
    const auto regions = generate_ffa_regions(1.0 / kFMax, 1.0 / kFMin, kTsamp,
                                              kNbins, /*eta_min=*/1.0);
    for (const bool use_fourier : {false, true}) {
        CAPTURE(use_fourier);
        const auto cfg = make_cfg(use_fourier, 8.0, /*accel=*/false);
        if (use_fourier) {
            const FFARegionPlanner<ComplexType> planner(cfg);
            CHECK(planner.get_nregions() == regions.size());
            check_covers_regions(planner, false);
        } else {
            const FFARegionPlanner<float> planner(cfg);
            CHECK(planner.get_nregions() == regions.size());
            check_covers_regions(planner, false);
        }
    }
}

namespace {

// Closed-form FLOP count of a frequency-only plan: data arithmetic only, a
// real FFT of length N costs 2.5 N log2 N.
double expected_gflops(const FFAPlan<ComplexType>& plan,
                       const FFASearchConfig& cfg,
                       bool return_in_time) {
    const auto nbins   = static_cast<double>(cfg.get_nbins());
    const auto nbins_f = static_cast<double>(cfg.get_nbins_f());
    const auto nsamps  = static_cast<double>(cfg.get_nsamps());
    const auto nfreqs  = static_cast<double>(plan.get_ncoords()[0]);
    const auto rfft    = 2.5 * nbins * std::log2(nbins);
    double flops{};
    if (cfg.get_nbins() > cfg.get_nbins_min_lossy_bf()) {
        flops = (2.0 * nsamps * nfreqs) +
                (2.0 * static_cast<double>(plan.get_nsegments()[0]) * nfreqs *
                 rfft);
    } else {
        flops = 8.0 * nsamps * nfreqs * nbins_f;
    }
    for (SizeType i = 1; i < plan.get_n_levels(); ++i) {
        flops += static_cast<double>(plan.get_ncoords()[i]) *
                 static_cast<double>(plan.get_nsegments()[i]) * 2.0 * nbins_f *
                 8.0;
    }
    if (return_in_time) {
        flops +=
            static_cast<double>(plan.get_fold_size_time() / cfg.get_nbins()) *
            rfft;
    }
    return flops * 1e-9;
}

} // namespace

TEST_CASE("FFAPlan: FLOP count matches the closed form (time fold)",
          "[ffa][plan][flops]") {
    const std::vector<ParamLimit> limits{{.min = 20.0, .max = 40.0}};
    const FFASearchConfig cfg(SizeType{1} << 14U, kTsamp, 32, 1.0, limits, 0.2,
                              1.5, /*use_fourier=*/false);
    const FFAPlan<float> plan(cfg);
    double flops = 2.0 * static_cast<double>(cfg.get_nsamps()) *
                   static_cast<double>(plan.get_ncoords()[0]);
    for (SizeType i = 1; i < plan.get_n_levels(); ++i) {
        flops += static_cast<double>(plan.get_ncoords()[i]) *
                 static_cast<double>(plan.get_nsegments()[i]) * 2.0 * 32.0;
    }
    REQUIRE(plan.get_gflops(false) == Approx(flops * 1e-9).epsilon(1e-4));
}

TEST_CASE("FFAPlan: FLOP count matches the closed form (Fourier fold)",
          "[ffa][plan][flops]") {
    const std::vector<ParamLimit> limits{{.min = 20.0, .max = 40.0}};
    // nbins 32 uses the direct DFT brute fold, nbins 128 the lossy one.
    for (const SizeType nbins : {SizeType{32}, SizeType{128}}) {
        const FFASearchConfig cfg(SizeType{1} << 14U, kTsamp, nbins, 1.0,
                                  limits, 0.2, 1.5, /*use_fourier=*/true);
        const FFAPlan<ComplexType> plan(cfg);
        INFO("nbins " << nbins);
        REQUIRE(plan.get_gflops(false) ==
                Approx(expected_gflops(plan, cfg, false)).epsilon(1e-4));
        REQUIRE(plan.get_gflops(true) ==
                Approx(expected_gflops(plan, cfg, true)).epsilon(1e-4));
    }
}
