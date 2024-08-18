#include <numeric>
#include <ranges>
#include <utility>

#include <spdlog/spdlog.h>

#include <loki/defaults.hpp>
#include <loki/ffa.hpp>
#include <loki/utils.hpp>

SearchConfig::SearchConfig(SizeType nsamps,
                           float tsamp,
                           SizeType nbins,
                           float tolerance,
                           const std::vector<ParamLimitType>& param_limits)
    : nsamps(nsamps),
      tsamp(tsamp),
      nbins(nbins),
      tolerance(tolerance),
      param_limits(param_limits) {
    if (param_limits.empty()) {
        throw std::runtime_error("coord_limits must be non-empty");
    }
    nparams = param_limits.size();
    validate_config();
    tobs  = static_cast<float>(nsamps) * tsamp;
    f_min = param_limits[nparams - 1][0];
    f_max = param_limits[nparams - 1][1];
}

std::vector<float> SearchConfig::get_ffa_step(float tsegment_cur) const {
    const auto step_freq =
        loki::freq_step(tsegment_cur, nbins, f_max, tolerance);
    std::vector<float> step_arr{step_freq};
    const auto t_ref = tsegment_cur / 2.0F;
    for (SizeType deriv = 2; deriv <= nparams; ++deriv) {
        const auto step_param =
            loki::deriv_step(tsegment_cur, tsamp, deriv, tolerance, t_ref);
        step_arr.insert(step_arr.begin(), step_param);
    }
    return step_arr;
};

SizeType SearchConfig::get_segment_len_init_default() const {
    const auto levels = static_cast<SizeType>(
        std::log2(static_cast<float>(nsamps) * tsamp * f_max));
    return static_cast<SizeType>(nsamps / (1 << levels));
}

SizeType SearchConfig::get_segment_len_final_default() const { return nsamps; }

void SearchConfig::validate_config() const {
    if (nsamps <= 0) {
        throw std::runtime_error("nsamps must be positive");
    }
    if ((nsamps & (nsamps - 1)) != 0) {
        throw std::runtime_error("nsamps must be power of 2");
    }
    if (tsamp <= 0) {
        throw std::runtime_error("tsamp must be positive");
    }
    if (nbins <= 0) {
        throw std::runtime_error("nbins must be positive");
    }
    if (tolerance <= 0) {
        throw std::runtime_error("tolerance must be positive");
    }
    if (nparams > 4) {
        throw std::runtime_error("nparams > 4 is not supported");
    }
    for (const auto& param_limit : param_limits) {
        if (param_limit[0] >= param_limit[1]) {
            throw std::runtime_error("coord_limits must be increasing");
        }
    }
}

FFADPFunctions::FFADPFunctions(SearchConfig cfg, FFAPlan& ffa_plan)
    : m_cfg(std::move(cfg)),
      m_ffa_plan(ffa_plan) {
    // Initialize the BruteFold object for initialization
    const auto& freq_arr = m_ffa_plan.params[0].back();
    const auto t_ref     = m_ffa_plan.tsegments[0] / 2.0F;
    m_brute_fold         = std::make_unique<BruteFold>(
        freq_arr, m_cfg.nbins, m_ffa_plan.segment_lens[0], m_cfg.tsamp, t_ref);
}

void FFADPFunctions::init(std::span<const float> ts, std::span<float> fold) {
    m_brute_fold->execute(ts, fold);
}

void FFADPFunctions::resolve(std::span<const float> pset_cur,
                             const std::vector<std::vector<float>>& parr_prev,
                             int ffa_level,
                             int latter) {
    loki::ffa_resolve(pset_cur, parr_prev, ffa_level, latter,
                      m_ffa_plan.tsegments[ffa_level], m_cfg.nbins);
}

FFA::FFA(SearchConfig cfg,
         std::optional<SizeType> segment_len_init,
         std::optional<SizeType> segment_len_final)
    : m_cfg(std::move(cfg)),
      m_segment_len_init(
          segment_len_init.value_or(m_cfg.get_segment_len_init_default())),
      m_segment_len_final(
          segment_len_final.value_or(m_cfg.get_segment_len_final_default())),
      m_niters(calculate_niters()) {
    validate_params();
    configure_plan();
    // Allocate memory for the FFA buffers
    m_fold_in.resize(m_ffa_plan.buffer_size, 0.0F);
    m_fold_out.resize(m_ffa_plan.buffer_size, 0.0F);
}

const FFAPlan& FFA::get_plan() const { return m_ffa_plan; }

void FFA::execute(std::span<const float> ts, std::span<float> fold) {
    if (ts.size() != m_cfg.nsamps) {
        throw std::runtime_error("ts must have size nsamps");
    }
    if (fold.size() != m_cfg.nbins) {
        throw std::runtime_error("fold must have size nbins");
    }
    initialize(ts, fold);
    for (SizeType i_iter = 0; i_iter < m_niters + 1; ++i_iter) {
        execute_iter(ts, fold);
    }
}

void FFA::initialize(std::span<const float> ts, std::span<float> fold) {
    spdlog::debug("Initializing FFA");
    // m_ffa_functions.ffa_init(ts, fold, m_ffa_plan.params[0][0],
    //                          m_ffa_plan.tsegments[0]);
}

void FFA::configure_plan() {
    m_ffa_plan.segment_lens.resize(m_niters + 1);
    m_ffa_plan.tsegments.resize(m_niters + 1);
    m_ffa_plan.params.resize(m_niters + 1);
    m_ffa_plan.dparams.resize(m_niters + 1);
    m_ffa_plan.fold_shapes.resize(m_niters + 1);

    for (SizeType i_iter = 0; i_iter < m_niters + 1; ++i_iter) {
        const auto segment_len = m_segment_len_init * (1 << i_iter);
        const auto nsegments   = m_cfg.nsamps / segment_len;
        const auto tsegment    = static_cast<float>(segment_len) * m_cfg.tsamp;
        const auto dparam_arr  = m_cfg.get_ffa_step(tsegment);

        m_ffa_plan.segment_lens[i_iter] = segment_len;
        m_ffa_plan.tsegments[i_iter]    = tsegment;
        m_ffa_plan.dparams[i_iter]      = dparam_arr;
        m_ffa_plan.params[i_iter].resize(m_cfg.nparams);
        m_ffa_plan.fold_shapes[i_iter].resize(m_cfg.nparams + 2);

        m_ffa_plan.fold_shapes[i_iter][0] = nsegments;
        for (SizeType iparam = 0; iparam < m_cfg.nparams; ++iparam) {
            const auto param_arr = loki::range_param(
                m_cfg.param_limits[iparam][0], m_cfg.param_limits[iparam][1],
                dparam_arr[iparam]);
            m_ffa_plan.params[i_iter][iparam]          = param_arr;
            m_ffa_plan.fold_shapes[i_iter][iparam + 1] = param_arr.size();
        }
        m_ffa_plan.fold_shapes[i_iter][m_cfg.nparams + 1] = m_cfg.nbins;
    }
    m_ffa_plan.buffer_size = std::ranges::max(
        m_ffa_plan.fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<size_t>());
        }));

    // Check param arr lengths for the initialization
    if (m_cfg.nparams > 1) {
        for (SizeType iparam = 0; iparam < m_cfg.nparams - 1; ++iparam) {
            if (m_ffa_plan.params[0][iparam].size() != 1) {
                throw std::runtime_error(
                    "Only one parameter for higher order derivatives is "
                    "supported for the initialization");
            }
        }
    }
}

void FFA::validate_params() const {
    if (!((m_segment_len_init & (m_segment_len_init - 1)) == 0)) {
        throw std::runtime_error("Segment length must be power of 2");
    }
    if (m_segment_len_init > m_cfg.nsamps) {
        throw std::runtime_error("Segment length must be less than nsamps");
    }
    if (!((m_segment_len_final & (m_segment_len_final - 1)) == 0)) {
        throw std::runtime_error("Segment length must be power of 2");
    }
    if (m_segment_len_final > m_cfg.nsamps) {
        throw std::runtime_error("Segment length must be less than nsamps");
    }
    if (m_segment_len_final <= m_segment_len_init) {
        throw std::runtime_error(
            "Final segment length must be greater than initial");
    }
}

SizeType FFA::calculate_niters() const {
    const auto nsegments_init  = m_cfg.nsamps / m_segment_len_init;
    const auto nsegments_final = m_cfg.nsamps / m_segment_len_final;
    return static_cast<SizeType>(std::log2(nsegments_init / nsegments_final));
}