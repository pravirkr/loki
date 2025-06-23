#pragma once

#include <span>
#include <tuple>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::core {

template <typename FoldType> class PruneTaylorDPFuncts {
public:
    PruneTaylorDPFuncts(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        double tseg_ffa,
                        search::PulsarSearchConfig cfg);

    // Core interface methods
    auto load(const xt::xtensor<FoldType, 2>& fold, SizeType seg_idx) const
        -> xt::xtensor<FoldType, 1>;

    auto resolve(const xt::xtensor<double, 3>& leaf_batch,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_init) const
        -> std::tuple<std::vector<SizeType>, std::vector<double>>;

    auto branch(const xt::xtensor<double, 3>& param_set_batch,
                std::pair<double, double> coord_cur) const
        -> std::tuple<xt::xtensor<double, 3>, std::vector<SizeType>>;

    void suggest(std::span<const FoldType> fold_segment,
                 std::pair<double, double> coord_init,
                 utils::SuggestionStruct<FoldType>& sugg_struct) const;

    void score(xt::xtensor<FoldType, 3>& combined_res_batch,
               std::span<float> out) const;

    auto pack(const xt::xtensor<FoldType, 2>& data) const
        -> xt::xtensor<FoldType, 1>;

    void shift_add(const xt::xtensor<FoldType, 3>& segment_batch,
                   std::span<const double> shift_batch,
                   const xt::xtensor<FoldType, 3>& folds,
                   std::span<const SizeType> isuggest_batch,
                   xt::xtensor<FoldType, 3>& out) const;

    auto transform(const xt::xtensor<double, 2>& leaf,
                   std::pair<double, double> coord_cur,
                   const xt::xtensor<double, 2>& trans_matrix) const
        -> xt::xtensor<double, 2>;

    auto get_transform_matrix(std::pair<double, double> coord_cur,
                              std::pair<double, double> coord_prev) const
        -> xt::xtensor<double, 2>;

    auto validate(const xt::xtensor<double, 3>& leaves,
                  std::pair<double, double> coord_valid,
                  const std::tuple<xt::xtensor<double, 1>,
                                   xt::xtensor<double, 1>,
                                   double>& validation_params) const
        -> xt::xtensor<double, 3>;

    auto get_validation_params(std::pair<double, double> coord_add) const
        -> std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, double>;

private:
    std::vector<std::vector<double>> m_param_arr;
    std::vector<double> m_dparams;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
};

// Type aliases for convenience
using PruneTaylorDPFunctsFloat   = PruneTaylorDPFuncts<float>;
using PruneTaylorDPFunctsComplex = PruneTaylorDPFuncts<ComplexType>;

} // namespace loki::core