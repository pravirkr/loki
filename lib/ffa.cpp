#include "loki/algorithms/ffa.hpp"

#include <chrono>
#include <format>
#include <utility>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>
#include <xsimd/xsimd.hpp>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

namespace {
inline void shift_add(const float* __restrict__ data_tail,
                      SizeType phase_shift_tail,
                      const float* __restrict__ data_head,
                      SizeType phase_shift_head,
                      float* __restrict__ out,
                      SizeType nbins) {

    const SizeType shift_tail = phase_shift_tail % nbins;
    const SizeType shift_head = phase_shift_head % nbins;

    const float* __restrict__ data_tail_row1 = data_tail + nbins;
    const float* __restrict__ data_head_row1 = data_head + nbins;
    float* __restrict__ out_row1             = out + nbins;
    for (SizeType j = 0; j < nbins; ++j) {
        const SizeType idx_tail =
            (j < shift_tail) ? (j + nbins - shift_tail) : (j - shift_tail);
        const SizeType idx_head =
            (j < shift_head) ? (j + nbins - shift_head) : (j - shift_head);
        out[j]      = data_tail[idx_tail] + data_head[idx_head];
        out_row1[j] = data_tail_row1[idx_tail] + data_head_row1[idx_head];
    }
}
} // namespace

class FFA::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()) {
        // Allocate memory for the FFA buffers
        m_fold_in.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        m_fold_out.resize(m_ffa_plan.get_buffer_size(), 0.0F);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const FFAPlan& get_plan() const { return m_ffa_plan; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold) {
        if (ts_e.size() != m_cfg.get_nsamps()) {
            throw std::runtime_error(
                std::format("ts must have size nsamps (got "
                            "{} != {})",
                            ts_e.size(), m_cfg.get_nsamps()));
        }
        if (ts_v.size() != ts_e.size()) {
            throw std::runtime_error(
                std::format("ts variance must have size nsamps "
                            "(got {} != {})",
                            ts_v.size(), ts_e.size()));
        }
        if (fold.size() != m_ffa_plan.get_fold_size()) {
            throw std::runtime_error(
                std::format("Output array has wrong size (got "
                            "{} != {})",
                            fold.size(), m_ffa_plan.get_fold_size()));
        }
        auto start = std::chrono::steady_clock::now();
        indicators::show_console_cursor(false);
        indicators::ProgressBar bar{
            indicators::option::PrefixText{"Computing FFA"},
            indicators::option::ShowPercentage(true),
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };
        initialize(ts_e, ts_v);
        // Use raw pointers for swapping buffers
        float* fold_in_ptr  = m_fold_in.data();
        float* fold_out_ptr = m_fold_out.data();
        const auto levels   = m_cfg.get_niters_ffa() + 1;

        for (SizeType i_level = 1; i_level < levels - 1; ++i_level) {
            execute_iter(fold_in_ptr, fold_out_ptr, i_level);
            std::swap(fold_in_ptr, fold_out_ptr);
            const auto progress = static_cast<float>(i_level) /
                                  static_cast<float>(levels - 1) * 100.0F;
            bar.set_progress(static_cast<SizeType>(progress));
        }
        // Last iteration directly writes to the output buffer
        execute_iter(fold_in_ptr, fold.data(), m_cfg.get_niters_ffa());
        bar.set_progress(100);
        indicators::show_console_cursor(true);
        spdlog::info("FFA finished");
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        spdlog::info("FFA::execute took {} ms", elapsed_ms);
    }

private:
    search::PulsarSearchConfig m_cfg;
    FFAPlan m_ffa_plan;
    int m_nthreads;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void initialize(std::span<const float> ts_e, std::span<const float> ts_v) {
        auto start = std::chrono::steady_clock::now();
        spdlog::info("FFA initialize");
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();
        algorithms::BruteFold bf(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_cfg.get_nthreads());
        bf.execute(ts_e, ts_v, std::span(m_fold_in.data(), bf.get_fold_size()));
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        spdlog::info("FFA::initialize took {} ms", elapsed_ms);
    }

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level) {
        const auto& coords_cur  = m_ffa_plan.coordinates[i_level];
        const auto& coords_prev = m_ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();
#pragma omp parallel for num_threads(m_nthreads)
        for (SizeType icoord = 0; icoord < ncoords_cur; ++icoord) {
            const auto& coord_cur = coords_cur[icoord];
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                const auto tail_offset =
                    ((iseg * 2) * ncoords_prev * 2 * nbins) +
                    (coord_cur.i_tail * 2 * nbins);
                const auto head_offset =
                    ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                    (coord_cur.i_head * 2 * nbins);
                const auto out_offset =
                    (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);
                shift_add(fold_in + tail_offset, coord_cur.shift_tail,
                          fold_in + head_offset, coord_cur.shift_head,
                          fold_out + out_offset, nbins);
            }
        }
    }
}; // End FFA::Impl definition

FFA::FFA(const search::PulsarSearchConfig& cfg)
    : m_impl(std::make_unique<Impl>(cfg)) {}
FFA::~FFA()                               = default;
FFA::FFA(FFA&& other) noexcept            = default;
FFA& FFA::operator=(FFA&& other) noexcept = default;
const FFAPlan& FFA::get_plan() const noexcept { return m_impl->get_plan(); }
void FFA::execute(std::span<const float> ts_e,
                  std::span<const float> ts_v,
                  std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

std::vector<float> compute_ffa(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               const search::PulsarSearchConfig& cfg) {
    FFA ffa(cfg);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms