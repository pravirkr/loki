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

std::vector<float> SearchConfig::ffa_step(float tsegment_cur) const {
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

FFA::FFA(SearchConfig cfg, int segment_len_init, int segment_len_final)
    : m_cfg(std::move(cfg)),
      m_segment_len_init(check_segment_len_init(segment_len_init)),
      m_segment_len_final(check_segment_len_final(segment_len_final)),
      m_niters(calculate_niters()) {
    configure_plan();
    // Allocate memory for the FFA buffers
    m_fold_in.resize(m_ffa_plan.buffer_size, 0.0F);
    m_fold_out.resize(m_ffa_plan.buffer_size, 0.0F);
}

const FFAPlan& FFA::get_plan() const { return m_ffa_plan; }

void FFA::execute(std::span<const float> ts, std::span<float> fold) {
    initialize(ts, fold);
    for (SizeType i_iter = 0; i_iter < m_niters + 1; ++i_iter) {
        execute_iter(ts, fold);
    }
}

void FFA::initialize(std::span<const float> ts, std::span<float> fold) {
    spdlog::debug("Initializing FFA");
    const auto nsegments = m_cfg.nsamps / m_segment_len_init;
    m_ffa_plan.params[0].resize(m_cfg.nparams);
    m_ffa_plan.fold_shapes[0].resize(m_cfg.nparams + 2);
    m_ffa_plan.fold_shapes[0][0] = nsegments;
    for (SizeType iparam = 0; iparam < m_cfg.nparams; ++iparam) {
        const auto param_arr = loki::range_param(m_cfg.param_limits[iparam][0],
                                                 m_cfg.param_limits[iparam][1],
                                                 m_ffa_plan.dparams[0][iparam]);
        m_ffa_plan.params[0][iparam]          = param_arr;
        m_ffa_plan.fold_shapes[0][iparam + 1] = param_arr.size();
    }
    m_ffa_plan.fold_shapes[0][m_cfg.nparams + 1] = m_cfg.nbins;
}

void FFA::configure_plan() {
    m_ffa_plan.tsegments.resize(m_niters + 1);
    m_ffa_plan.params.resize(m_niters + 1);
    m_ffa_plan.dparams.resize(m_niters + 1);
    m_ffa_plan.fold_shapes.resize(m_niters + 1);

    for (SizeType i_iter = 0; i_iter < m_niters + 1; ++i_iter) {
        const auto segment_size = m_segment_len_init * (1 << i_iter);
        const auto tsegment   = static_cast<float>(segment_size) * m_cfg.tsamp;
        const auto nsegments  = m_cfg.nsamps / segment_size;
        const auto dparam_arr = m_cfg.ffa_step(tsegment);

        m_ffa_plan.tsegments[i_iter] = tsegment;
        m_ffa_plan.dparams[i_iter]   = dparam_arr;
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
}

SizeType FFA::check_segment_len_init(int segment_len_init) const {
    if (segment_len_init > 0) {
        if (!((segment_len_init & (segment_len_init - 1)) == 0)) {
            throw std::runtime_error("Segment length must be power of 2");
        }
        if (static_cast<SizeType>(segment_len_init) > m_cfg.nsamps) {
            throw std::runtime_error("Segment length must be less than nsamps");
        }
        return static_cast<SizeType>(segment_len_init);
    }
    if (segment_len_init == -1) {
        const auto levels = static_cast<SizeType>(std::log2(
            static_cast<float>(m_cfg.nsamps) * m_cfg.tsamp * m_cfg.f_max));
        return static_cast<SizeType>(m_cfg.nsamps / (1 << levels));
    }
    throw std::runtime_error("Segment length must be positive or -1");
}

SizeType FFA::check_segment_len_final(int segment_len_final) const {
    if (segment_len_final > 0) {
        if (!((segment_len_final & (segment_len_final - 1)) == 0)) {
            throw std::runtime_error("Segment length must be power of 2");
        }
        if (static_cast<SizeType>(segment_len_final) > m_cfg.nsamps) {
            throw std::runtime_error("Segment length must be less than nsamps");
        }
        if (static_cast<SizeType>(segment_len_final) <= m_segment_len_init) {
            throw std::runtime_error(
                "Final segment length must be greater than initial");
        }
        return static_cast<SizeType>(segment_len_final);
    }
    if (segment_len_final == -1) {
        return m_cfg.nsamps;
    }
    throw std::runtime_error("Segment length must be positive or -1");
}

SizeType FFA::calculate_niters() const {
    const auto nsegments_init  = m_cfg.nsamps / m_segment_len_init;
    const auto nsegments_final = m_cfg.nsamps / m_segment_len_final;
    return static_cast<SizeType>(std::log2(nsegments_init / nsegments_final));
}