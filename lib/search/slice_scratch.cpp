#include "lib/search/slice_scratch.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"

namespace loki::search::detail {
namespace {

constexpr std::array<const char*, 4> kColumns{".leaves", ".scores",
                                              ".scores_ep", ".widths"};

std::filesystem::path column_path(const std::filesystem::path& stem,
                                  std::size_t i) {
    return std::filesystem::path(stem.string() + kColumns[i]);
}

void open_column(std::fstream& file, const std::filesystem::path& path) {
    file.open(path, std::ios::binary | std::ios::in | std::ios::out |
                        std::ios::trunc);
    if (!file) {
        throw std::runtime_error(
            fmt::format("seed-slice scratch: cannot create {} ({})",
                        path.string(), std::strerror(errno)));
    }
}

template <typename T>
void append_column(std::fstream& file,
                   std::span<const T> values,
                   const std::filesystem::path& stem,
                   double& seconds) {
    if (values.empty()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
    if (!file) {
        throw std::runtime_error(fmt::format(
            "seed-slice scratch: writing {} bytes beside {} failed (disk "
            "full?)",
            values.size_bytes(), stem.string()));
    }
    seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
}

template <typename T>
void read_column(std::fstream& file,
                 SizeType begin_value,
                 std::span<T> out,
                 const std::filesystem::path& stem) {
    if (out.empty()) {
        return;
    }
    const auto offset = static_cast<std::streamoff>(begin_value * sizeof(T));
    if (file.tellg() != offset) {
        file.seekg(offset);
    }
    file.read(reinterpret_cast<char*>(out.data()),
              static_cast<std::streamsize>(out.size_bytes()));
    if (!file) {
        throw std::runtime_error(fmt::format(
            "seed-slice scratch: reading beside {} failed", stem.string()));
    }
}

} // namespace

SliceScratch::SliceScratch(std::filesystem::path stem, SizeType leaves_stride)
    : m_stem(std::move(stem)),
      m_stride(leaves_stride) {
    if (m_stride == 0) {
        throw std::invalid_argument("seed-slice scratch: zero leaf stride");
    }
    open_column(m_leaves, column_path(m_stem, 0));
    open_column(m_scores, column_path(m_stem, 1));
    open_column(m_scores_ep, column_path(m_stem, 2));
    open_column(m_widths, column_path(m_stem, 3));
}

SliceScratch::~SliceScratch() {
    m_leaves.close();
    m_scores.close();
    m_scores_ep.close();
    m_widths.close();
    for (std::size_t i = 0; i < kColumns.size(); ++i) {
        std::error_code error;
        std::filesystem::remove(column_path(m_stem, i), error);
    }
}

void SliceScratch::append_leaves(std::span<const double> leaves) {
    if (m_sealed || leaves.size() % m_stride != 0) {
        throw std::logic_error("seed-slice scratch: bad leaf append");
    }
    append_column(m_leaves, leaves, m_stem, m_write_seconds);
    m_leaves_rows += leaves.size() / m_stride;
}

void SliceScratch::append_scores(std::span<const float> scores) {
    if (m_sealed) {
        throw std::logic_error("seed-slice scratch: append after seal");
    }
    append_column(m_scores, scores, m_stem, m_write_seconds);
    m_scores_rows += scores.size();
}

void SliceScratch::append_scores_ep(std::span<const float> scores_ep) {
    if (m_sealed) {
        throw std::logic_error("seed-slice scratch: append after seal");
    }
    append_column(m_scores_ep, scores_ep, m_stem, m_write_seconds);
    m_scores_ep_rows += scores_ep.size();
}

void SliceScratch::append_widths(std::span<const std::uint32_t> widths) {
    if (m_sealed) {
        throw std::logic_error("seed-slice scratch: append after seal");
    }
    append_column(m_widths, widths, m_stem, m_write_seconds);
    m_widths_rows += widths.size();
}

std::uint64_t SliceScratch::bytes() const noexcept {
    return (m_leaves_rows * m_stride * sizeof(double)) +
           ((m_scores_rows + m_scores_ep_rows) * sizeof(float)) +
           (m_widths_rows * sizeof(std::uint32_t));
}

void SliceScratch::seal() {
    if (m_leaves_rows != m_scores_rows || m_scores_ep_rows != m_scores_rows ||
        (m_widths_rows != 0 && m_widths_rows != m_scores_rows)) {
        throw std::logic_error(fmt::format(
            "seed-slice scratch: column rows differ (leaves {}, scores {}, "
            "scores_ep {}, widths {})",
            m_leaves_rows, m_scores_rows, m_scores_ep_rows, m_widths_rows));
    }
    for (auto* file : {&m_leaves, &m_scores, &m_scores_ep, &m_widths}) {
        file->flush();
        if (!*file) {
            throw std::runtime_error(fmt::format(
                "seed-slice scratch: flushing beside {} failed (disk full?)",
                m_stem.string()));
        }
        file->seekg(0);
    }
    m_sealed = true;
}

void SliceScratch::read(SizeType begin,
                        SizeType n,
                        std::span<double> leaves,
                        std::span<float> scores,
                        std::span<float> scores_ep,
                        std::span<std::uint32_t> widths) {
    if (!m_sealed || begin + n > m_scores_rows ||
        leaves.size() != n * m_stride || scores.size() != n ||
        scores_ep.size() != n ||
        (!widths.empty() && (widths.size() != n || !has_widths()))) {
        throw std::logic_error("seed-slice scratch: bad read");
    }
    read_column(m_leaves, begin * m_stride, leaves, m_stem);
    read_column(m_scores, begin, scores, m_stem);
    read_column(m_scores_ep, begin, scores_ep, m_stem);
    read_column(m_widths, begin, widths, m_stem);
}

namespace {
std::atomic<SizeType> g_slice_spill_rows{SizeType{1} << 20U};
} // namespace

SizeType slice_spill_rows() noexcept { return g_slice_spill_rows.load(); }

void set_slice_spill_rows(SizeType rows) noexcept {
    g_slice_spill_rows.store(rows);
}

std::filesystem::path
slice_scratch_stem(const std::filesystem::path& result_file,
                   std::string_view run_name) {
    // Runs of different chunks can share a result file name, and a CUDA
    // sweep writes one run while the next one fills its scratch: a process
    // counter keeps every scratch's files apart.
    static std::atomic<std::uint64_t> counter{0};
    return result_file.parent_path() /
           fmt::format("{}.{}.{}.slices", result_file.stem().string(), run_name,
                       counter.fetch_add(1));
}

void log_scratch_volume(const SliceScratch& scratch,
                        std::string_view run_name) {
    constexpr double kBytesPerGB = 1024.0 * 1024.0 * 1024.0;
    const double gb = static_cast<double>(scratch.bytes()) / kBytesPerGB;
    if (gb > 1.0) {
        spdlog::warn("Pruning run {}: {} final leaves of its seed slices took "
                     "{:.1f} GB of scratch beside the results ({:.1f} s to "
                     "write; a source or RFI filled the slices' trees)",
                     run_name, scratch.rows(), gb, scratch.write_seconds());
    }
}

} // namespace loki::search::detail
