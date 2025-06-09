#include "loki/core/taylor.hpp"

#include <cstddef>
#include <span>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/psr_utils.hpp"
#include "loki/utils.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, double>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins) {
    const auto nparams = pset_cur.size();
    std::vector<double> pset_prev(nparams, 0.0);
    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
    double delta_t{}, delay{};

    if (nparams == 1) {
        delta_t      = static_cast<double>(latter) * tsegment;
        pset_prev[0] = pset_cur[0];
        delay        = 0.0;
    } else {
        delta_t = (static_cast<double>(latter) - 0.5) * tsegment;
        std::tie(pset_prev, delay) = psr_utils::shift_params(pset_cur, delta_t);
    }
    const auto relative_phase =
        psr_utils::get_phase_idx(delta_t, pset_cur[nparams - 1], nbins, delay);

    std::vector<SizeType> pindex_prev(nparams);
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = utils::find_nearest_sorted_idx(
            std::span(param_arr[ip]), pset_prev[ip]);
    }
    return {pindex_prev, relative_phase};
}

} // namespace loki::core
