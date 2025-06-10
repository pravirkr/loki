#include "loki/algorithms/ffa_complex.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
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
#include "loki/utils/fft.hpp"

namespace loki::algorithms {

namespace {

void shift_add_complex(const ComplexType* __restrict__ data_tail,
                       double phase_shift_tail,
                       const ComplexType* __restrict__ data_head,
                       double phase_shift_head,
                       ComplexType* __restrict__ out,
                       SizeType nbins_f,
                       SizeType nbins) {

    // Precompute phase factor constants
    const auto phase_factor_tail =
        -2.0 * std::numbers::pi * phase_shift_tail / static_cast<double>(nbins);
    const auto phase_factor_head =
        -2.0 * std::numbers::pi * phase_shift_head / static_cast<double>(nbins);

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    for (SizeType k = 0; k < nbins_f; ++k) {
        const auto k_phase_tail = static_cast<double>(k) * phase_factor_tail;
        const auto k_phase_head = static_cast<double>(k) * phase_factor_head;

        // Fast complex exponential: exp(i * theta) = cos(theta) + i *
        // sin(theta)
        const ComplexType phase_tail{
            static_cast<float>(std::cos(k_phase_tail)),
            static_cast<float>(std::sin(k_phase_tail))};
        const ComplexType phase_head{
            static_cast<float>(std::cos(k_phase_head)),
            static_cast<float>(std::sin(k_phase_head))};
        out_e[k] =
            (data_tail_e[k] * phase_tail) + (data_head_e[k] * phase_head);
        out_v[k] =
            (data_tail_v[k] * phase_tail) + (data_head_v[k] * phase_head);
    }
}

} // namespace

class FFACOMPLEX::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()) {
        // Allocate memory for the FFA buffers
        m_fold_in.resize(m_ffa_plan.get_buffer_size_complex(),
                         ComplexType(0.0F, 0.0F));
        m_fold_out.resize(m_ffa_plan.get_buffer_size_complex(),
                          ComplexType(0.0F, 0.0F));
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);
        m_fold_in_brute.resize(m_the_bf->get_fold_size(), 0.0F);
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
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size(),
            "FFACOMPLEX::Impl::execute: fold must have size fold_size");

        ScopeTimer timer("FFACOMPLEX::execute");
        std::vector<ComplexType> fold_complex(
            m_ffa_plan.get_fold_size_complex(), ComplexType(0.0F, 0.0F));
        execute_core(ts_e, ts_v, fold_complex);
        // IRFFT the output
        const auto nfft = m_ffa_plan.get_fold_size() / m_cfg.get_nbins();
        utils::irfft_batch(fold_complex, fold, static_cast<int>(nfft),
                           static_cast<int>(m_cfg.get_nbins()), m_nthreads);
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold) {
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size_complex(),
            "FFACOMPLEX::Impl::execute: fold must have size fold_size_complex");

        ScopeTimer timer("FFACOMPLEX::execute");
        execute_core(ts_e, ts_v, fold);
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    int m_nthreads;
    std::unique_ptr<algorithms::BruteFold> m_the_bf;

    // Buffers for the FFA plan
    std::vector<ComplexType> m_fold_in;
    std::vector<ComplexType> m_fold_out;
    std::vector<float> m_fold_in_brute;

    void initialize(std::span<const float> ts_e, std::span<const float> ts_v) {
        ScopeTimer timer("FFACOMPLEX::initialize");
        m_the_bf->execute(ts_e, ts_v, m_fold_in_brute);

        // RFFT the input
        const auto nfft         = m_the_bf->get_fold_size() / m_cfg.get_nbins();
        const auto complex_size = nfft * ((m_cfg.get_nbins() / 2) + 1);
        utils::rfft_batch(
            m_fold_in_brute,
            std::span<ComplexType>(m_fold_in.data(), complex_size),
            static_cast<int>(nfft), static_cast<int>(m_cfg.get_nbins()),
            m_nthreads);
    }

    void execute_core(std::span<const float> ts_e,
                      std::span<const float> ts_v,
                      std::span<ComplexType> fold_complex) {
        initialize(ts_e, ts_v);

        // Use raw pointers for swapping buffers
        ComplexType* fold_in_ptr     = m_fold_in.data();
        ComplexType* fold_out_ptr    = m_fold_out.data();
        ComplexType* fold_result_ptr = fold_complex.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;

        indicators::show_console_cursor(false);
        auto bar = utils::make_standard_bar("Computing FFA...");
        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            // Determine output buffer: final iteration writes to output buffer
            const bool is_last = i_level == levels - 1;
            ComplexType* current_out_ptr =
                is_last ? fold_result_ptr : fold_out_ptr;
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

    void execute_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_ffa_plan.coordinates[i_level];
        const auto coords_prev  = m_ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto nbins_f      = (nbins / 2) + 1;
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();

        constexpr SizeType kBlockSize = 8;
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, coords_prev, nsegments, nbins,       \
               nbins_f, ncoords_cur, ncoords_prev)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_tail * 2 * nbins_f);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_head * 2 * nbins_f);
                    const auto out_offset = (iseg * ncoords_cur * 2 * nbins_f) +
                                            (icoord * 2 * nbins_f);

                    const ComplexType* fold_tail = &fold_in[tail_offset];
                    const ComplexType* fold_head = &fold_in[head_offset];
                    ComplexType* fold_sum        = &fold_out[out_offset];
                    shift_add_complex(fold_tail, coord_cur.shift_tail,
                                      fold_head, coord_cur.shift_head, fold_sum,
                                      nbins_f, nbins);
                }
            }
        }
    }
}; // End FFA::Impl definition

FFACOMPLEX::FFACOMPLEX(const search::PulsarSearchConfig& cfg)
    : m_impl(std::make_unique<Impl>(cfg)) {}
FFACOMPLEX::~FFACOMPLEX()                                      = default;
FFACOMPLEX::FFACOMPLEX(FFACOMPLEX&& other) noexcept            = default;
FFACOMPLEX& FFACOMPLEX::operator=(FFACOMPLEX&& other) noexcept = default;
const plans::FFAPlan& FFACOMPLEX::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FFACOMPLEX::execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}
void FFACOMPLEX::execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<ComplexType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

std::vector<float> compute_ffa_complex(std::span<const float> ts_e,
                                       std::span<const float> ts_v,
                                       const search::PulsarSearchConfig& cfg) {
    FFACOMPLEX ffa(cfg);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms