#include "loki/algorithms/ffa.hpp"

#include <utility>

#include <indicators/cursor_control.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"

namespace loki::algorithms {

namespace {
void shift_add(const float* __restrict__ data_tail,
               double phase_shift_tail,
               const float* __restrict__ data_head,
               double phase_shift_head,
               float* __restrict__ out,
               SizeType nbins) {

    const auto shift_tail =
        static_cast<SizeType>(std::round(phase_shift_tail)) % nbins;
    const auto shift_head =
        static_cast<SizeType>(std::round(phase_shift_head)) % nbins;

    const float* __restrict__ data_tail_e = data_tail;
    const float* __restrict__ data_tail_v = data_tail + nbins;
    const float* __restrict__ data_head_e = data_head;
    const float* __restrict__ data_head_v = data_head + nbins;
    float* __restrict__ out_e             = out;
    float* __restrict__ out_v             = out + nbins;

    for (SizeType j = 0; j < nbins; ++j) {
        const auto idx_tail =
            (j < shift_tail) ? (j + nbins - shift_tail) : (j - shift_tail);
        const auto idx_head =
            (j < shift_head) ? (j + nbins - shift_head) : (j - shift_head);
        out_e[j] = data_tail_e[idx_tail] + data_head_e[idx_head];
        out_v[j] = data_tail_v[idx_tail] + data_head_v[idx_head];
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
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan& get_plan() const { return m_ffa_plan; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold) {
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size(),
            "FFA::Impl::execute: fold must have size fold_size");

        ScopeTimer timer("FFA::execute");
        initialize(ts_e, ts_v);

        // Use raw pointers for swapping buffers
        float* fold_in_ptr     = m_fold_in.data();
        float* fold_out_ptr    = m_fold_out.data();
        float* fold_result_ptr = fold.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;

        indicators::show_console_cursor(false);
        auto bar = utils::make_standard_bar("Computing FFA...");
        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            // Determine output buffer: final iteration writes to output buffer
            const bool is_last     = i_level == levels - 1;
            float* current_out_ptr = is_last ? fold_result_ptr : fold_out_ptr;
            execute_iter(fold_in_ptr, current_out_ptr, i_level);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(fold_in_ptr, fold_out_ptr);
            }
            const auto progress = static_cast<float>(i_level) /
                                  static_cast<float>(levels - 1) * 100.0F;
            bar.set_progress(static_cast<SizeType>(progress));
        }
        indicators::show_console_cursor(true);
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    int m_nthreads;
    std::unique_ptr<algorithms::BruteFold> m_the_bf;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void initialize(std::span<const float> ts_e, std::span<const float> ts_v) {
        ScopeTimer timer("FFA::initialize");
        m_the_bf->execute(
            ts_e, ts_v, std::span(m_fold_in.data(), m_the_bf->get_fold_size()));
    }

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_ffa_plan.coordinates[i_level];
        const auto coords_prev  = m_ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();

        constexpr SizeType kBlockSize = 8;
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, coords_prev, nsegments, nbins,       \
               ncoords_cur, ncoords_prev)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_tail * 2 * nbins);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_head * 2 * nbins);
                    const auto out_offset =
                        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

                    const float* fold_tail = &fold_in[tail_offset];
                    const float* fold_head = &fold_in[head_offset];
                    float* fold_sum        = &fold_out[out_offset];
                    shift_add(fold_tail, coord_cur.shift_tail, fold_head,
                              coord_cur.shift_head, fold_sum, nbins);
                }
            }
        }
    }
}; // End FFA::Impl definition

FFA::FFA(const search::PulsarSearchConfig& cfg)
    : m_impl(std::make_unique<Impl>(cfg)) {}
FFA::~FFA()                               = default;
FFA::FFA(FFA&& other) noexcept            = default;
FFA& FFA::operator=(FFA&& other) noexcept = default;
const plans::FFAPlan& FFA::get_plan() const noexcept {
    return m_impl->get_plan();
}
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