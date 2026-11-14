#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"

#include "lib/core/circular.hpp"
#include "lib/utils/workspace_impl.hpp"

#include "circ_branch_replay.hpp"

// The circular branch workspace is sized from BranchingForecast::max_children.
// Driving the CPU branch and transform kernels level by level, with every
// frequency of the band and parents whose children all fall in the crackle
// hole, no parent may have more children than the forecast, and the worst
// parent has exactly that many.

namespace {

using loki::SizeType;
using loki::test::kCircBatch;
using loki::test::kCircParams;
using loki::test::kCircSlots;

struct CpuReplay {
    SizeType max_children{0};   // most children of one parent
    SizeType max_product{0};    // most d4..d1 children (no hole expansion)
    SizeType crackle_levels{0}; // levels at which crackle split
    bool all_in_hole{true};     // every crackle split reached every child
};

CpuReplay replay_cpu(const loki::search::PulsarSearchConfig& cfg,
                     const loki::plans::FFAPlan<loki::ComplexType>& plan) {
    loki::memory::BranchingWorkspace ws(kCircBatch, kCircSlots, kCircParams);
    CpuReplay out;
    SizeType last_crackle_level = 0;
    out.max_children            = loki::test::replay_circ_widths(
        cfg, plan,
        [&](std::span<const double> tree, SizeType nb, SizeType level,
            std::pair<double, double> coord_cur, std::vector<double>& children,
            std::vector<SizeType>& origins) {
            // The no-branching path leaves the counts unwritten.
            std::ranges::fill(ws.scratch_counts, 0);
            const SizeType n_out = loki::core::circ_taylor_branch_batch(
                tree, children, origins, coord_cur, cfg.get_nbins(),
                cfg.get_eta(), kCircSlots, nb,
                cfg.get_propagator_significance(), ws);
            std::vector<SizeType> n_children(nb, 0);
            for (SizeType c = 0; c < n_out; ++c) {
                ++n_children[origins[c]];
            }
            for (SizeType i = 0; i < nb; ++i) {
                const auto* counts =
                    ws.scratch_counts.data() + (i * kCircParams);
                SizeType product = 1;
                for (SizeType j = 1; j < kCircParams; ++j) {
                    product *= std::max(counts[j], SizeType{1});
                }
                const SizeType n_d5 = std::max(counts[0], SizeType{1});
                if (n_d5 > 1 && level != last_crackle_level) {
                    ++out.crackle_levels;
                    last_crackle_level = level;
                }
                out.all_in_hole =
                    out.all_in_hole && n_children[i] == product * n_d5;
                out.max_product = std::max(out.max_product, product);
            }
            return n_out;
        });
    return out;
}

void check_bound(bool conservative) {
    const auto cfg = loki::test::make_circ_cfg(conservative);
    const loki::plans::FFAPlan<loki::ComplexType> plan(cfg);
    const auto forecast = plan.forecast_branching("taylor", 0);
    const auto run      = replay_cpu(cfg, plan);
    INFO("conservative " << conservative << ", forecast "
                         << forecast.max_children << ", kernels "
                         << run.max_children << " (d4..d1 alone "
                         << run.max_product << "), crackle levels "
                         << run.crackle_levels);
    // Widths must be carried across several crackle splits, and the split
    // must set the worst parent.
    REQUIRE(run.crackle_levels >= 2);
    REQUIRE(run.all_in_hole);
    REQUIRE(run.max_children < kCircSlots);
    CHECK(run.max_children > run.max_product);
    CHECK(run.max_children == forecast.max_children);
}

} // namespace

TEST_CASE("Circular branch_max bounds the kernels' crackle hole expansion",
          "[ep_branching][circular]") {
    check_bound(false);
}

TEST_CASE("Circular branch_max bounds the kernels with the conservative tile",
          "[ep_branching][circular]") {
    // The plan of an EP config must see its conservative tile, which widens
    // the cells the kernels branch.
    check_bound(true);
}
