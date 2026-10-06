#include "loki/algorithms/fold.hpp"

#include <algorithm>
#include <limits>

#include <omp.h>

#include "algorithms/fold_engine.hpp"
#include "common/dispatch.hpp"
#include "core/kernels.hpp"
#include "detail/exceptions.hpp"
#include "detail/psr_utils.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms {

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
        error_check::check_equal(m_nsamps % m_segment_len, 0,
                                 "BruteFold::Impl: Number of samples is not a "
                                 "multiple of segment length");
        m_nthreads  = std::clamp(m_nthreads, 1, omp_get_max_threads());
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = m_nsamps / m_segment_len;
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
            kernels::brute_fold_ts(ts_e.data(), ts_v.data(), fold.data(),
                                   m_runs.data(), m_run_offsets.data(),
                                   m_nsegments, m_nfreqs, m_segment_len,
                                   m_nbins, m_nthreads);

        } else {
            kernels::brute_fold_ts_complex(
                ts_e.data(), ts_v.data(), fold.data(), m_freq_arr.data(),
                m_nfreqs, m_nsegments, m_segment_len, m_nbins, m_tsamp, m_t_ref,
                m_nthreads);
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
            error_check::check_greater_equal(
                nlevels, 1,
                "BruteFoldCpuEngine::execute_fused_freq: nlevels must be >= 1");
            error_check::check_equal(
                m_nsegments % (SizeType{1} << nlevels), 0,
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
                                     (m_nsegments >> nlevels) *
                                         ncoords[nlevels] * 2 * m_nbins,
                                     "BruteFoldCpuEngine::execute_fused_freq: "
                                     "fold_out has wrong size");
            kernels::brute_fold_ffa_fused_freq(
                ts_e.data(), ts_v.data(), fold_out.data(), m_runs.data(),
                m_run_offsets.data(), coords_levels.data(), ncoords.data(),
                m_nsegments, m_nfreqs, m_segment_len, m_nbins, nlevels,
                m_nthreads);
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

    /// Build the run table and check it reproduces the per-sample phase map.
    void compute_phase_time_domain() {
        error_check::check_less_equal(
            m_segment_len,
            static_cast<SizeType>(std::numeric_limits<uint32_t>::max()),
            "BruteFold segment length does not fit in a phase-run index");
        m_run_offsets.assign(m_nfreqs + 1, 0);
        std::vector<SizeType> counts(m_nfreqs, 0);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            uint32_t prev_bin = std::numeric_limits<uint32_t>::max();
            SizeType nruns    = 0;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto proper_time =
                    (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
                const uint32_t iphase = psr_utils::get_phase_idx_uint(
                    proper_time, m_freq_arr[ifreq], m_nbins, 0.0);
                if (iphase != prev_bin) {
                    ++nruns;
                    prev_bin = iphase;
                }
            }
            counts[ifreq] = nruns;
        }
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            m_run_offsets[ifreq + 1] = m_run_offsets[ifreq] + counts[ifreq];
        }
        m_runs.resize(m_run_offsets.back());
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            coord::PhaseRun* out = m_runs.data() + m_run_offsets[ifreq];
            uint32_t prev_bin    = std::numeric_limits<uint32_t>::max();
            uint32_t run_end     = 0;
            SizeType written     = 0;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto proper_time =
                    (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
                const uint32_t iphase = psr_utils::get_phase_idx_uint(
                    proper_time, m_freq_arr[ifreq], m_nbins, 0.0);
                if (iphase != prev_bin) {
                    if (prev_bin != std::numeric_limits<uint32_t>::max()) {
                        out[written++] = {run_end, prev_bin};
                    }
                    prev_bin = iphase;
                }
                run_end = static_cast<uint32_t>(isamp + 1);
            }
            if (m_segment_len > 0) {
                out[written++] = {run_end, prev_bin};
            }
            // Contiguous cover of the segment: the runs abut and end at B.
            error_check::check_equal(
                written, counts[ifreq],
                "BruteFold phase runs do not match the counted run total");
            error_check::check_equal(
                static_cast<SizeType>(run_end), m_segment_len,
                "BruteFold phase runs do not cover the segment");
        }
    }

}; // End BruteFoldCpuEngine definition

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
