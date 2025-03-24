#include "loki/loki_types.hpp"
#include <loki/basic.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <loki/math.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

std::tuple<std::vector<SizeType>, SizeType>
loki::ffa_resolve(std::span<const FloatType> pset_cur,
                  std::span<const std::vector<FloatType>> parr_prev,
                  SizeType ffa_level,
                  SizeType latter,
                  FloatType tseg_brute,
                  SizeType nbins) {
    const auto nparams = pset_cur.size();
    const auto t_ref_prev =
        (static_cast<FloatType>(latter) - 0.5F) *
        std::pow(2.0F, static_cast<FloatType>(ffa_level - 1)) * tseg_brute;
    std::vector<FloatType> pset_prev(nparams, 0.0F);
    FloatType delay_rel = 0.0;

    if (nparams == 1) {
        pset_prev[0] = pset_cur[0];
    } else {
        std::vector<FloatType> dvec_cur(nparams + 1, 0.0F);
        std::copy(pset_cur.begin(), pset_cur.end() - 1, dvec_cur.begin());
        std::vector<FloatType> dvec_prev = loki::utils::shift_params(
            dvec_cur, static_cast<FloatType>(t_ref_prev));
        std::copy(dvec_prev.begin(), dvec_prev.end() - 1, pset_prev.begin());
        pset_prev[nparams - 1] = pset_cur[nparams - 1] *
                                 (1.0F + dvec_prev[nparams - 2] / loki::kCval);
        delay_rel = dvec_prev[nparams - 1] / loki::kCval;
    }

    const SizeType relative_phase = loki::utils::get_phase_idx(
        t_ref_prev, pset_prev[nparams - 1], nbins, delay_rel);

    std::vector<SizeType> pindex_prev(nparams);
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = loki::find_nearest_sorted_idx(
            std::span(parr_prev[ip]), pset_prev[ip]);
    }

    return {pindex_prev, relative_phase};
}
