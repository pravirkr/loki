#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "loki/cands.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

/**
 * @brief Everything needed to turn a flat score index back into a parameter
 * set, for one FFA chunk.
 *
 * @details Built once per chunk from its own PulsarSearchConfig. Holding the
 * chunk's own scoring widths is essential: the boxcar width set is derived
 * from nbins, which varies between FFA regions, so decoding with the base
 * configuration's widths silently mislabels every chunk with a larger nbins.
 */
struct RegionDecode {
    std::vector<ParamLimit> param_limits; ///< Drift-expanded chunk limits.
    std::vector<SizeType> param_counts;   ///< Grid counts at the last level.
    std::vector<SizeType> param_strides;  ///< Cartesian strides, last level.
    std::vector<SizeType> widths;         ///< This chunk's scoring widths.
    SizeType n_widths{};
    SizeType ncoords{};
    SizeType nbins{};
    SizeType nsegments{};
    double gflops{}; ///< FFA plus scoring cost for this chunk.

    /// @brief Number of raw scores this chunk produces (ncoords * n_widths).
    [[nodiscard]] SizeType get_n_scores() const noexcept {
        return ncoords * n_widths;
    }
};

/**
 * @brief Build the per-chunk decode tables for a planned frequency sweep.
 *
 * @tparam FoldType float or ComplexType (host fold type).
 * @param cfgs The per-chunk search configurations from FFARegionPlanner.
 * @return One RegionDecode per chunk, in planner order.
 */
template <SupportedFoldType FoldType>
std::vector<RegionDecode>
build_region_decode_table(std::span<const search::PulsarSearchConfig> cfgs) {
    std::vector<RegionDecode> table;
    table.reserve(cfgs.size());
    for (const auto& cfg : cfgs) {
        const plans::FFAPlan<FoldType> plan(cfg);
        const auto limits = cfg.get_param_limits();

        RegionDecode dec;
        dec.param_limits.assign(limits.begin(), limits.end());
        dec.param_counts  = plan.get_param_counts().back();
        dec.param_strides = plan.get_param_cart_strides().back();
        dec.widths        = cfg.get_scoring_widths();
        dec.n_widths      = dec.widths.size();
        dec.ncoords       = plan.get_ncoords().back();
        dec.nbins         = cfg.get_nbins();
        dec.nsegments     = plan.get_nsegments().back();

        // Scoring cost: 2 passes over (e, v) profiles per width trial.
        const auto score_flops =
            static_cast<double>(dec.ncoords * 2) *
            static_cast<double>(dec.n_widths * 2 * dec.nbins);
        dec.gflops = static_cast<double>(plan.get_gflops(
                         /*return_in_time=*/true)) +
                     (score_flops * 1e-9);

        error_check::check_less_equal(
            dec.get_n_scores(),
            static_cast<SizeType>(std::numeric_limits<uint32_t>::max()),
            std::format("build_region_decode_table: chunk produces {} scores, "
                        "which exceeds the uint32 index range used by the "
                        "candidate buffer",
                        dec.get_n_scores()));
        table.push_back(std::move(dec));
    }
    return table;
}

/**
 * @brief Fixed-capacity, append-only store of surviving FFA candidates.
 *
 * @details The buffer is sized once and never grows, so a pathological chunk
 * (RFI, red noise) cannot drive the process out of memory. Each candidate
 * carries the id of the chunk it came from, which decouples the buffer layout
 * from the chunk iteration order and lets it be drained at any point rather
 * than only on a chunk boundary.
 */
class CandidateBuffer {
public:
    explicit CandidateBuffer(SizeType capacity) : m_capacity(capacity) {
        error_check::check_greater(capacity, SizeType{0},
                                   "CandidateBuffer: capacity must be > 0");
        m_scores.resize(capacity);
        m_local_idx.resize(capacity);
        m_region_id.resize(capacity);
    }

    [[nodiscard]] SizeType get_capacity() const noexcept { return m_capacity; }
    [[nodiscard]] SizeType get_size() const noexcept { return m_size; }
    [[nodiscard]] SizeType get_space() const noexcept {
        return m_capacity - m_size;
    }
    [[nodiscard]] bool is_full() const noexcept { return m_size == m_capacity; }
    void clear() noexcept { m_size = 0; }

    /**
     * @brief Append a single candidate.
     * @note Precondition: !is_full(). Callers must flush first.
     */
    void push(float score, uint32_t local_idx, uint32_t region_id) noexcept {
        m_scores[m_size]    = score;
        m_local_idx[m_size] = local_idx;
        m_region_id[m_size] = region_id;
        ++m_size;
    }

    /**
     * @brief Writable view of the next `n` score slots, for bulk fills.
     * @note Must be paired with commit(n, region_id). Throws if n > space.
     */
    [[nodiscard]] std::span<float> get_scores_tail(SizeType n) {
        check_space(n);
        return std::span(m_scores).subspan(m_size, n);
    }

    /**
     * @brief Writable view of the next `n` local-index slots, for bulk fills.
     * @note Must be paired with commit(n, region_id). Throws if n > space.
     */
    [[nodiscard]] std::span<uint32_t> get_indices_tail(SizeType n) {
        check_space(n);
        return std::span(m_local_idx).subspan(m_size, n);
    }

    /// @brief Publish `n` bulk-filled slots, all belonging to `region_id`.
    void commit(SizeType n, uint32_t region_id) {
        check_space(n);
        std::fill_n(m_region_id.begin() + static_cast<IndexType>(m_size), n,
                    region_id);
        m_size += n;
    }

    [[nodiscard]] std::span<const float> get_scores() const noexcept {
        return std::span(m_scores).first(m_size);
    }
    [[nodiscard]] std::span<const uint32_t> get_local_indices() const noexcept {
        return std::span(m_local_idx).first(m_size);
    }
    [[nodiscard]] std::span<const uint32_t> get_region_ids() const noexcept {
        return std::span(m_region_id).first(m_size);
    }

private:
    SizeType m_capacity;
    SizeType m_size{};
    std::vector<float> m_scores;
    std::vector<uint32_t> m_local_idx; ///< Score index within its own chunk.
    std::vector<uint32_t> m_region_id;

    void check_space(SizeType n) const {
        error_check::check_less_equal(
            n, get_space(),
            std::format("CandidateBuffer: requested {} slots but only {} of {} "
                        "are free",
                        n, get_space(), m_capacity));
    }
};

/**
 * @brief Decode every buffered candidate and append it to the result file.
 *
 * @details Each candidate is decoded against its own chunk's width count and
 * grid strides. Writes are batched through `param_sets_scratch`, whose size
 * fixes the batch length. FFAResultWriter::write_results appends, so calling
 * this repeatedly is equivalent to one call with the whole sweep. The buffer
 * is cleared on return.
 *
 * @param buf The candidate buffer to drain.
 * @param decode_table Per-chunk decode tables, indexed by region id.
 * @param writer The destination result file.
 * @param param_sets_scratch Staging buffer of size batch * (n_params + 1).
 * @param n_params Number of search parameters (excluding the width column).
 */
inline void flush_candidates(CandidateBuffer& buf,
                             std::span<const RegionDecode> decode_table,
                             cands::FFAResultWriter& writer,
                             std::span<double> param_sets_scratch,
                             SizeType n_params) {
    if (buf.get_size() == 0) {
        return;
    }
    const SizeType total_params = n_params + 1; // includes width
    const SizeType batch_max    = param_sets_scratch.size() / total_params;
    error_check::check_greater(batch_max, SizeType{0},
                               "flush_candidates: param_sets_scratch is too "
                               "small to hold a single candidate");

    const auto scores     = buf.get_scores();
    const auto local_idx  = buf.get_local_indices();
    const auto region_ids = buf.get_region_ids();

    SizeType batch_start = 0;
    while (batch_start < buf.get_size()) {
        const SizeType batch_count =
            std::min(batch_max, buf.get_size() - batch_start);

        for (SizeType k = 0; k < batch_count; ++k) {
            const SizeType global_idx = batch_start + k;
            const SizeType region_id  = region_ids[global_idx];
            const auto& dec           = decode_table[region_id];
            const SizeType score_idx  = local_idx[global_idx];
            const SizeType coord_idx  = score_idx / dec.n_widths;
            const SizeType width_idx  = score_idx % dec.n_widths;

            // Reconstruct parameters from coord_idx using index arithmetic
            SizeType remaining = coord_idx;
            for (SizeType j = 0; j < n_params; ++j) {
                const SizeType param_idx = remaining / dec.param_strides[j];
                remaining -= param_idx * dec.param_strides[j];
                param_sets_scratch[(k * total_params) + j] =
                    psr_utils::get_param_val_at_idx(
                        dec.param_limits[j], dec.param_counts[j], param_idx);
            }
            param_sets_scratch[(k * total_params) + n_params] =
                static_cast<double>(dec.widths[width_idx]);
        }

        writer.write_results(
            param_sets_scratch.first(batch_count * total_params),
            scores.subspan(batch_start, batch_count), batch_count,
            total_params);
        batch_start += batch_count;
    }
    buf.clear();
}

} // namespace loki::algorithms
