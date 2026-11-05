#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#include "lib/algorithms/fold_engine.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/core/kernels.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/psr_utils.hpp"

namespace loki::algorithms {

namespace {

template <SupportedFoldType FoldType>
class BruteFoldCpuEngine : public detail::BruteFoldEngine<FoldType> {
public:
    BruteFoldCpuEngine(std::span<const double> freq_arr,
                       SizeType segment_len,
                       SizeType nbins,
                       SizeType nsamps,
                       double tsamp,
                       double t_ref,
                       int nthreads)
        : m_freq_arr(freq_arr.begin(), freq_arr.end()),
          m_segment_len(segment_len),
          m_nbins(nbins),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_t_ref(t_ref),
          m_nthreads(nthreads) {
        error_check::check(!m_freq_arr.empty(),
                           "BruteFold::Impl: Frequency array is empty");
        error_check::check_greater(m_nsamps, 0,
                                   "BruteFold::Impl: nsamps must be > 0");
        error_check::check_greater(m_segment_len, 0,
                                   "BruteFold::Impl: segment_len must be > 0");
        m_nthreads  = std::max(m_nthreads, 1);
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = (m_nsamps + m_segment_len - 1) / m_segment_len;
        m_nbins_f   = (nbins / 2) + 1;
        if constexpr (std::is_same_v<FoldType, float>) {
            compute_phase_time_domain();
        }
    }

    ~BruteFoldCpuEngine() override                           = default;
    BruteFoldCpuEngine(const BruteFoldCpuEngine&)            = delete;
    BruteFoldCpuEngine& operator=(const BruteFoldCpuEngine&) = delete;
    BruteFoldCpuEngine(BruteFoldCpuEngine&&)                 = delete;
    BruteFoldCpuEngine& operator=(BruteFoldCpuEngine&&)      = delete;

    SizeType get_fold_size() const override {
        if constexpr (std::is_same_v<FoldType, float>) {
            return m_nsegments * m_nfreqs * 2 * m_nbins;
        } else {
            return m_nsegments * m_nfreqs * 2 * m_nbins_f;
        }
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) override {
        error_check::check_equal(
            ts_e.size(), m_nsamps,
            "BruteFold::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "BruteFold::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), get_fold_size(),
            "BruteFold::Impl::execute: fold must have size fold_size");

        // Ensure output fold is zeroed
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            std::ranges::fill(fold, ComplexType(0.0F, 0.0F));
        } else {
            std::ranges::fill(fold, 0.0F);
        }
        if constexpr (std::is_same_v<FoldType, float>) {
            core::brute_fold_ts(ts_e.data(), ts_v.data(), fold.data(),
                                m_runs.data(), m_run_offsets.data(),
                                m_nsegments, m_nfreqs, m_segment_len, m_nbins,
                                m_nthreads, m_nsamps);

        } else {
            core::brute_fold_ts_complex(ts_e.data(), ts_v.data(), fold.data(),
                                        m_freq_arr.data(), m_nfreqs,
                                        m_nsegments, m_segment_len, m_nbins,
                                        m_tsamp, m_t_ref, m_nthreads, m_nsamps);
        }
    }

    void execute(DeviceSpan<const float> /*ts_e*/,
                 DeviceSpan<const float> /*ts_v*/,
                 DeviceSpan<FoldType> /*fold*/,
                 Stream /*stream*/) override {
        loki::detail::throw_no_device_memory("BruteFold", Backend::kCPU);
    }

    void execute_fused_freq(
        std::span<const float> ts_e,
        std::span<const float> ts_v,
        std::span<float> fold_out,
        std::span<const coord::FFACoordFreq* const> coords_levels,
        std::span<const SizeType> ncoords,
        SizeType nlevels) override {
        if constexpr (std::is_same_v<FoldType, float>) {
            error_check::check_equal(ts_e.size(), m_nsamps,
                                     "BruteFoldCpuEngine::execute_fused_freq: "
                                     "ts_e must have size nsamps");
            error_check::check_equal(
                ts_v.size(), ts_e.size(),
                "BruteFoldCpuEngine::execute_fused_freq: ts_v "
                "must have size nsamps");
            const SizeType nsegments_padded = std::bit_ceil(m_nsegments);
            error_check::check_greater_equal(
                nlevels, 1,
                "BruteFoldCpuEngine::execute_fused_freq: nlevels must be >= 1");
            error_check::check_equal(
                nsegments_padded % (SizeType{1} << nlevels), 0,
                "BruteFoldCpuEngine::execute_fused_freq: nsegments must be a "
                "multiple of 2^nlevels");
            error_check::check_greater_equal(
                coords_levels.size(), nlevels + 1,
                "BruteFoldCpuEngine::execute_fused_freq: coords_levels too "
                "short");
            error_check::check_greater_equal(
                ncoords.size(), nlevels + 1,
                "BruteFoldCpuEngine::execute_fused_freq: ncoords too short");
            error_check::check_equal(ncoords[0], m_nfreqs,
                                     "BruteFoldCpuEngine::execute_fused_freq: "
                                     "ncoords[0] must equal nfreqs");
            error_check::check_equal(fold_out.size(),
                                     (nsegments_padded >> nlevels) *
                                         ncoords[nlevels] * 2 * m_nbins,
                                     "BruteFoldCpuEngine::execute_fused_freq: "
                                     "fold_out has wrong size");
            core::brute_fold_ffa_fused_freq(
                ts_e.data(), ts_v.data(), fold_out.data(), m_runs.data(),
                m_run_offsets.data(), coords_levels.data(), ncoords.data(),
                nsegments_padded, m_nfreqs, m_segment_len, m_nbins, nlevels,
                m_nthreads, m_nsamps);
        } else {
            throw std::logic_error(
                "execute_fused_freq only supported on float engine");
        }
    }

    [[nodiscard]] std::span<const coord::PhaseRun> runs() const override {
        if constexpr (std::is_same_v<FoldType, float>) {
            return m_runs;
        } else {
            return {};
        }
    }

    [[nodiscard]] std::span<const SizeType> run_offsets() const override {
        if constexpr (std::is_same_v<FoldType, float>) {
            return m_run_offsets;
        } else {
            return {};
        }
    }

private:
    std::vector<double> m_freq_arr;
    SizeType m_segment_len;
    SizeType m_nbins;
    SizeType m_nsamps;
    double m_tsamp;
    double m_t_ref;
    int m_nthreads;

    SizeType m_nfreqs;
    SizeType m_nsegments;
    SizeType m_nbins_f;

    // Time domain only. One (end, bin) per contiguous phase run.
    std::vector<coord::PhaseRun> m_runs;
    std::vector<SizeType> m_run_offsets;

    /// Build the run table: per frequency, the (exclusive end, bin) runs of
    /// consecutive samples in one phase bin. The runs of a frequency cover
    /// its segment by construction.
    void compute_phase_time_domain() {
        error_check::check_less_equal(
            m_segment_len,
            static_cast<SizeType>(std::numeric_limits<uint32_t>::max()),
            "BruteFold segment length does not fit in a phase-run index");
        const auto nt = static_cast<SizeType>(m_nthreads);
        std::vector<std::vector<coord::PhaseRun>> thread_runs(nt);
        std::vector<SizeType> freq_counts(m_nfreqs, 0);
        // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel num_threads(m_nthreads)
        {
            const auto tid    = static_cast<SizeType>(omp_get_thread_num());
            const auto nthr   = static_cast<SizeType>(omp_get_num_threads());
            const SizeType f0 = (tid * m_nfreqs) / nthr;
            const SizeType f1 = ((tid + 1) * m_nfreqs) / nthr;
            auto& local       = thread_runs[tid];
            for (SizeType ifreq = f0; ifreq < f1; ++ifreq) {
                const SizeType before = local.size();
                append_phase_runs(m_freq_arr[ifreq], local);
                freq_counts[ifreq] = local.size() - before;
            }
        }
        m_run_offsets.resize(m_nfreqs + 1);
        m_run_offsets[0] = 0;
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            m_run_offsets[ifreq + 1] =
                m_run_offsets[ifreq] + freq_counts[ifreq];
        }
        std::vector<SizeType> thread_offsets(nt + 1, 0);
        for (SizeType t = 0; t < nt; ++t) {
            thread_offsets[t + 1] = thread_offsets[t] + thread_runs[t].size();
        }
        m_runs.resize(m_run_offsets.back());
        // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (SizeType t = 0; t < nt; ++t) {
            if (!thread_runs[t].empty()) {
                std::ranges::copy(thread_runs[t],
                                  m_runs.begin() + static_cast<std::ptrdiff_t>(
                                                       thread_offsets[t]));
            }
        }
    }

    /// Append phase runs of one frequency over a segment to `runs`.
    void append_phase_runs(double freq,
                           std::vector<coord::PhaseRun>& runs) const {
        error_check::check_greater_equal(freq, 0.0,
                                         "Frequency must be positive");
        error_check::check_greater_equal(m_nbins, 1,
                                         "Number of bins must be positive");
        if (m_segment_len == 0) {
            return;
        }
        const auto nbins_d    = static_cast<double>(m_nbins);
        const auto nbins_u32  = static_cast<uint32_t>(m_nbins);
        const double step_bin = m_tsamp * freq * nbins_d;
        const double inv_step =
            (step_bin > 0.0 && step_bin < 0.5) ? (1.0 / step_bin) : 0.0;
        // Margin of 2e-3 bins (> 32 float ULPs at nbins <= 1024) before the
        // half-integer boundary so the exact formula always decides boundary
        // samples.
        constexpr double kBoundaryMargin = 2.0e-3;

        uint32_t prev_bin = 0;
        for (SizeType isamp = 0; isamp < m_segment_len;) {
            const double proper_time =
                (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
            const double total_phase = proper_time * freq;
            const double norm_phase  = total_phase - std::floor(total_phase);
            double bin_phase         = norm_phase * nbins_d;
            if (bin_phase >= nbins_d) {
                bin_phase = 0.0;
            }
            const auto iphase = static_cast<float>(bin_phase);
            auto iphase_int   = static_cast<uint32_t>(iphase + 0.5F);
            if (iphase_int == nbins_u32) {
                iphase_int = 0;
            }
            if (isamp > 0 && iphase_int != prev_bin) {
                runs.push_back(
                    {.end = static_cast<uint32_t>(isamp), .bin = prev_bin});
            }
            prev_bin = iphase_int;

            if (inv_step > 0.0) {
                const double target =
                    (iphase_int == 0 && bin_phase >= nbins_d - 0.5)
                        ? (nbins_d + 0.5)
                        : (static_cast<double>(iphase_int) + 0.5);
                const double rem = target - bin_phase - kBoundaryMargin;
                if (rem > step_bin) {
                    const auto skip = static_cast<SizeType>(rem * inv_step);
                    isamp = std::min(m_segment_len,
                                     isamp + std::max<SizeType>(1, skip));
                    continue;
                }
            }
            ++isamp;
        }
        runs.push_back(
            {.end = static_cast<uint32_t>(m_segment_len), .bin = prev_bin});
    }

}; // End BruteFoldCpuEngine definition

} // namespace

namespace detail {

template <SupportedFoldType FoldType>
std::unique_ptr<BruteFoldEngine<FoldType>>
make_brute_fold_cpu(std::span<const double> freq_arr,
                    SizeType segment_len,
                    SizeType nbins,
                    SizeType nsamps,
                    double tsamp,
                    double t_ref,
                    int nthreads) {
    return std::make_unique<BruteFoldCpuEngine<FoldType>>(
        freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads);
}

} // namespace detail

template std::unique_ptr<detail::BruteFoldEngine<float>>
detail::make_brute_fold_cpu<float>(
    std::span<const double>, SizeType, SizeType, SizeType, double, double, int);
template std::unique_ptr<detail::BruteFoldEngine<ComplexType>>
detail::make_brute_fold_cpu<ComplexType>(
    std::span<const double>, SizeType, SizeType, SizeType, double, double, int);

} // namespace loki::algorithms
