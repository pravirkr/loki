#pragma once

/**
 * @file slice_scratch.hpp
 * @brief The final leaves of a run pruned in seed slices, kept on disk
 * between its slices. Internal.
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>

#include "loki/common/types.hpp"

namespace loki::search::detail {

/**
 * @brief Row store of one run's final leaves, one raw file per column:
 * leaf rows (`leaves_stride` doubles), scores, scores_ep and widths.
 *
 * A run pruned in K seed slices appends each finished slice's rows, so the
 * host holds at most what the caller passes to one append(); the run is
 * written from the files afterwards (PruneResultWriter::write_run_results
 * with a scratch). Rows keep the order they were appended in. The files
 * live beside the run's result file and are removed by the destructor, on
 * every exit.
 */
class SliceScratch {
public:
    /// Creates `<stem>.leaves`, `.scores`, `.scores_ep` and `.widths`.
    SliceScratch(std::filesystem::path stem, SizeType leaves_stride);
    ~SliceScratch();
    SliceScratch(const SliceScratch&)            = delete;
    SliceScratch& operator=(const SliceScratch&) = delete;
    SliceScratch(SliceScratch&&)                 = delete;
    SliceScratch& operator=(SliceScratch&&)      = delete;

    /// Appends rows to one column. Every column must reach rows() before a
    /// read (widths: rows() or none at all).
    void append_leaves(std::span<const double> leaves);
    void append_scores(std::span<const float> scores);
    void append_scores_ep(std::span<const float> scores_ep);
    void append_widths(std::span<const std::uint32_t> widths);

    /// Rows of the run: those of the scores column.
    [[nodiscard]] SizeType rows() const noexcept { return m_scores_rows; }
    [[nodiscard]] SizeType leaves_stride() const noexcept { return m_stride; }
    [[nodiscard]] bool has_widths() const noexcept { return m_widths_rows > 0; }
    /// Bytes written to disk.
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    /// Wall seconds the appends took (the caller waits for them).
    [[nodiscard]] double write_seconds() const noexcept {
        return m_write_seconds;
    }

    /// Ends the appends; the columns must have equal row counts.
    void seal();

    /// Reads rows [begin, begin + n) of every column (sealed scratch only;
    /// sequential calls are the cheap ones). @p widths may be empty when
    /// the scratch has none.
    void read(SizeType begin,
              SizeType n,
              std::span<double> leaves,
              std::span<float> scores,
              std::span<float> scores_ep,
              std::span<std::uint32_t> widths);

private:
    std::filesystem::path m_stem;
    SizeType m_stride;
    std::fstream m_leaves;
    std::fstream m_scores;
    std::fstream m_scores_ep;
    std::fstream m_widths;
    SizeType m_leaves_rows{};
    SizeType m_scores_rows{};
    SizeType m_scores_ep_rows{};
    SizeType m_widths_rows{};
    bool m_sealed{false};
    double m_write_seconds{};
};

/// A new scratch stem for run @p run_name, beside its @p result_file:
/// unique within the process.
[[nodiscard]] std::filesystem::path
slice_scratch_stem(const std::filesystem::path& result_file,
                   std::string_view run_name);

/// Logs a run's scratch volume when it exceeds 1 GB: a source or RFI that
/// filled its slices' trees.
void log_scratch_volume(const SliceScratch& scratch, std::string_view run_name);

} // namespace loki::search::detail
