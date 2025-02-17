#include <loki/ffa.hpp>

#include <numeric>
#include <ranges>
#include <utility>

#include <spdlog/spdlog.h>

#include <loki/basic.hpp>
#include <loki/cartesian.hpp>
#include <loki/defaults.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

FFAPlan::FFAPlan(PulsarSearchConfig cfg) : m_cfg(std::move(cfg)) {
    configure_plan();
}

SizeType FFAPlan::get_buffer_size() const noexcept {
    return std::ranges::max(
        fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<size_t>());
        }));
}

void FFAPlan::configure_plan() {
    const auto levels = m_cfg.niters_ffa + 1;
    segment_lens.resize(levels);
    tsegments.resize(levels);
    params.resize(levels);
    dparams.resize(levels);
    fold_shapes.resize(levels);
    coordinates.resize(levels);

    for (SizeType i_level = 0; i_level < levels; ++i_level) {
        const auto segment_len = m_cfg.bseg_brute * (1 << i_level);
        const auto tsegment    = static_cast<float>(segment_len) * m_cfg.tsamp;
        const auto nsegments   = m_cfg.nsamps / segment_len;
        const auto dparam_arr  = m_cfg.get_dparams(tsegment);

        segment_lens[i_level] = segment_len;
        tsegments[i_level]    = tsegment;
        dparams[i_level]      = dparam_arr;
        params[i_level].resize(m_cfg.nparams);
        fold_shapes[i_level].resize(m_cfg.nparams + 2);

        fold_shapes[i_level][0] = nsegments;
        for (SizeType iparam = 0; iparam < m_cfg.nparams; ++iparam) {
            const auto param_arr = loki::utils::range_param(
                m_cfg.param_limits[iparam][0], m_cfg.param_limits[iparam][1],
                dparam_arr[iparam]);
            params[i_level][iparam]          = param_arr;
            fold_shapes[i_level][iparam + 1] = param_arr.size();
        }
        fold_shapes[i_level][m_cfg.nparams + 1] = m_cfg.nbins;
    }
    // Check param arr lengths for the initialization
    if (m_cfg.nparams > 1) {
        for (SizeType iparam = 0; iparam < m_cfg.nparams - 1; ++iparam) {
            if (params[0][iparam].size() != 1) {
                throw std::runtime_error(
                    "Only one parameter for higher order derivatives is "
                    "supported for the initialization");
            }
        }
    }

    // Now resolve the params for the FFA plan
    for (SizeType i_level = 1; i_level < levels; ++i_level) {
        std::vector<float> p_set_cur(m_cfg.nparams);
        const auto strides = calculate_strides(params[i_level - 1]);
        for (const auto& p_set_view : cartesian_product_view(params[i_level])) {
            for (SizeType iparam = 0; iparam < m_cfg.nparams; ++iparam) {
                p_set_cur[iparam] = p_set_view[iparam];
            }
            const auto [p_idx_tail, phase_shift_tail] =
                loki::ffa_resolve(p_set_cur, params[i_level - 1], i_level, 0,
                                  m_cfg.tseg_brute, m_cfg.nbins);
            const auto [p_idx_head, phase_shift_head] =
                loki::ffa_resolve(p_set_cur, params[i_level - 1], i_level, 1,
                                  m_cfg.tseg_brute, m_cfg.nbins);
            const auto& i_coord_tail =
                std::inner_product(p_idx_tail.begin(), p_idx_tail.end(),
                                   strides.begin(), SizeType{0});
            const auto& i_coord_head =
                std::inner_product(p_idx_head.begin(), p_idx_head.end(),
                                   strides.begin(), SizeType{0});

            const auto coord_cur = FFACoord{.i_coord_tail = i_coord_tail,
                                            .shift_tail   = phase_shift_tail,
                                            .i_coord_head = i_coord_head,
                                            .shift_head   = phase_shift_head};
            coordinates[i_level].emplace_back(coord_cur);
        }
    }
}

std::vector<SizeType>
FFAPlan::calculate_strides(std::span<const std::vector<float>> p_arr) {
    const auto nparams = p_arr.size();
    std::vector<SizeType> strides(nparams);
    strides[nparams - 1] = 1;
    for (SizeType i = nparams - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * p_arr[i + 1].size();
    }
    return strides;
}

FFA::FFA(PulsarSearchConfig cfg) : m_cfg(std::move(cfg)), m_ffa_plan(m_cfg) {
    // Allocate memory for the FFA buffers
    m_fold_in.resize(m_ffa_plan.get_buffer_size(), 0.0F);
    m_fold_out.resize(m_ffa_plan.get_buffer_size(), 0.0F);
}

const FFAPlan& FFA::get_plan() const { return m_ffa_plan; }

void FFA::initialize(std::span<const float> ts, std::span<float> fold) {
    spdlog::debug("Initializing FFA");
    const auto t_ref = m_ffa_plan.tsegments[0] / 2.0F;
    brute_fold(ts, fold, m_ffa_plan.params[0].back(),
               m_ffa_plan.segment_lens[0], m_cfg.nbins, m_cfg.nsamps,
               m_cfg.tsamp, t_ref);
}

void FFA::execute(std::span<const float> ts, std::span<float> fold) {
    if (ts.size() != m_cfg.nsamps) {
        throw std::runtime_error("ts must have size nsamps");
    }
    if (fold.size() != m_cfg.nbins) {
        throw std::runtime_error("fold must have size nbins");
    }
    float* fold_in_ptr  = m_fold_in.data();
    float* fold_out_ptr = m_fold_out.data();

    initialize(ts, std::span<float>(fold_in_ptr, m_ffa_plan.get_buffer_size()));
    for (SizeType i_iter = 1; i_iter < m_cfg.niters_ffa + 1; ++i_iter) {
        execute_iter(fold_in_ptr, fold_out_ptr, i_iter);
        std::swap(fold_in_ptr, fold_out_ptr);
    }
    // Last iteration directly writes to the output buffer
    execute_iter(fold_in_ptr, fold_out_ptr, m_cfg.niters_ffa);
}