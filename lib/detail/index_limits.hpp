#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "loki/common/types.hpp"

namespace loki::index_limits {

/// Largest fold-buffer length addressable with uint32_t element indices in FFA
/// CUDA kernels.
inline constexpr std::uint64_t kMaxUint32FoldElements =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

/// Conservative limit for scoring kernels that index folds with signed int
/// strides (profile_idx * 2 * nbins).
inline constexpr std::uint64_t kMaxIntFoldStrideElements =
    static_cast<std::uint64_t>(std::numeric_limits<int>::max());

struct ChunkIndexUsage {
    SizeType buffer_size{};
    SizeType ncoords{};
    SizeType nbins{};
    SizeType n_scoring_widths{};
    SizeType nfreqs{};
    SizeType segment_len{};
};

[[nodiscard]] inline bool chunk_exceeds_cuda_index_limits(
    const ChunkIndexUsage& u) noexcept {
    if (u.buffer_size > kMaxUint32FoldElements) {
        return true;
    }
    const auto fold_stride =
        static_cast<std::uint64_t>(u.ncoords) * static_cast<std::uint64_t>(u.nbins) *
        2ULL;
    if (fold_stride > kMaxIntFoldStrideElements) {
        return true;
    }
    const auto n_scores =
        static_cast<std::uint64_t>(u.ncoords) * u.n_scoring_widths;
    if (n_scores > kMaxUint32FoldElements) {
        return true;
    }
    const auto phase_elems =
        static_cast<std::uint64_t>(u.nfreqs) * u.segment_len;
    return phase_elems > kMaxUint32FoldElements;
}

[[nodiscard]] inline std::string chunk_index_limit_message(
    const ChunkIndexUsage& u) {
    return std::string(
        "CUDA index limits exceeded for chunk "
        "(buffer_size=") +
           std::to_string(u.buffer_size) + ", ncoords=" +
           std::to_string(u.ncoords) + ", nbins=" + std::to_string(u.nbins) +
           ", n_widths=" + std::to_string(u.n_scoring_widths) +
           ", nfreqs=" + std::to_string(u.nfreqs) +
           ", segment_len=" + std::to_string(u.segment_len) +
           "). Reduce max_memory_gb or narrow the parameter search.";
}

inline void validate_chunk_cuda_index_limits(const ChunkIndexUsage& u) {
    if (chunk_exceeds_cuda_index_limits(u)) {
        throw std::runtime_error(chunk_index_limit_message(u));
    }
}

} // namespace loki::index_limits
