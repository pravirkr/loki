#include "loki/utils/world_tree.hpp"

#include <algorithm>

#include <cuda/std/limits>
#include <cuda/std/span>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::utils {
namespace {

template <typename T>
__global__ void compact_in_place_kernel(const uint8_t* __restrict__ keep,
                                        const uint32_t* __restrict__ prefix,
                                        T* __restrict__ data,
                                        uint32_t start,
                                        uint32_t capacity,
                                        uint32_t size,
                                        uint32_t stride) {
    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= size || keep[i] == 0) {
        return;
    }

    const uint32_t dst_logical = prefix[i];
    const uint32_t src_lin     = start + i;
    const uint32_t dst_lin     = start + dst_logical;
    const uint32_t src_phys = src_lin < capacity ? src_lin : src_lin - capacity;
    const uint32_t dst_phys = dst_lin < capacity ? dst_lin : dst_lin - capacity;
    // Copy full row
    for (uint32_t j = 0; j < stride; ++j) {
        data[(dst_phys * stride) + j] = data[(src_phys * stride) + j];
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
__global__ void
compact_and_copy_kernel(const double* __restrict__ batch_leaves_ptr,
                        const FoldTypeCUDA* __restrict__ batch_folds_ptr,
                        const float* __restrict__ batch_scores_ptr,
                        const uint8_t* __restrict__ mask_ptr,
                        const uint32_t* __restrict__ prefix_ptr,
                        double* __restrict__ tmp_leaves_ptr,
                        FoldTypeCUDA* __restrict__ tmp_folds_ptr,
                        float* __restrict__ tmp_scores_ptr,
                        uint32_t slots_to_write,
                        uint32_t m_leaves_stride,
                        uint32_t m_folds_stride) {
    const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= slots_to_write) {
        return;
    }

    if (mask_ptr[idx] == 1) {
        const uint32_t out_idx = prefix_ptr[idx];
        for (uint32_t j = 0; j < m_leaves_stride; ++j) {
            tmp_leaves_ptr[(out_idx * m_leaves_stride) + j] =
                batch_leaves_ptr[(idx * m_leaves_stride) + j];
        }

        for (uint32_t j = 0; j < m_folds_stride; ++j) {
            tmp_folds_ptr[(out_idx * m_folds_stride) + j] =
                batch_folds_ptr[(idx * m_folds_stride) + j];
        }

        tmp_scores_ptr[out_idx] = batch_scores_ptr[idx];
    }
}

struct CircularIndexFunctor {
    uint32_t start;
    uint32_t capacity;

    __host__ __device__ uint32_t operator()(uint32_t logical_idx) const {
        const uint32_t x = start + logical_idx;
        return (x < capacity) ? x : x - capacity;
    }
};

struct LinearMaskFunctor {
    const float* __restrict__ scores_ptr;
    float threshold;

    __host__ __device__ uint8_t operator()(uint32_t idx) const {
        return (scores_ptr[idx] > threshold) ? 1 : 0;
    }
};

struct CircularMaskFunctor {
    const float* __restrict__ scores_ptr;
    uint32_t start;
    uint32_t capacity;
    float threshold;

    __host__ __device__ uint8_t operator()(uint32_t logical_idx) const {
        const uint32_t x    = start + logical_idx;
        const uint32_t phys = (x < capacity) ? x : x - capacity;
        return (scores_ptr[phys] > threshold) ? 1 : 0;
    }
};
} // namespace

/**
 * @brief GPU-resident circular buffer for world tree data
 *
 * This class maintains all data in GPU global memory and provides
 * the same circular buffer semantics as the CPU version.
 *
 * Layout (all indices modulo m_capacity):
 *
 * ┌────────┬───────────────┬───────────────┐
 * │ unused │  READ REGION  │  WRITE REGION │
 * └────────┴───────────────┴───────────────┘
 *            ^ m_head         ^ m_write_start
 *            size = m_size_old      size = m_size
 *
 * @tparam FoldTypeCUDA Either float or ComplexTypeCUDA
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class WorldTreeCUDA<FoldTypeCUDA>::Impl {
public:
    /**
     * @brief Constructor for the WorldTreeCUDA class.
     *
     * Initializes the internal arrays with the given maximum number of
     * candidates, number of parameters, and number of bins.
     *
     * @param capacity Maximum number of candidates to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     * @param max_batch_size Maximum number of candidates that can be added in a
     * single batch. This is used to allocate the scratch buffer.
     */
    Impl(SizeType capacity,
         SizeType nparams,
         SizeType nbins,
         SizeType max_batch_size)
        : m_capacity(capacity),
          m_nparams(nparams),
          m_nbins(nbins),
          m_max_batch_size(max_batch_size),
          m_leaves_stride((nparams + 2) * kParamStride),
          m_folds_stride(2 * nbins),
          m_leaves(m_capacity * m_leaves_stride, 0.0),
          m_folds(m_capacity * m_folds_stride, FoldTypeCUDA{}),
          m_scores(m_capacity, 0.0F),
          m_scratch_leaves(m_capacity * (nparams * kParamStride), 0.0),
          m_scratch_folds(max_batch_size * m_folds_stride, FoldTypeCUDA{}),
          m_scratch_scores((m_capacity + max_batch_size), 0.0F),
          m_scratch_prefix(m_capacity, 0),
          m_scratch_mask(m_capacity, 0) {
        // Validate inputs
        error_check::check_greater(m_capacity, SizeType{0},
                                   "SuggestionTreeCUDA: capacity must be > 0");
        error_check::check_greater(m_nparams, SizeType{0},
                                   "SuggestionTreeCUDA: nparams must be > 0");
        error_check::check_greater(m_nbins, SizeType{0},
                                   "SuggestionTreeCUDA: nbins must be > 0");
    }

    ~Impl()                          = default;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;
    Impl(const Impl&)                = delete;
    Impl& operator=(const Impl&)     = delete;

    // Size and capacity queries
    SizeType get_capacity() const noexcept { return m_capacity; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    SizeType get_max_batch_size() const noexcept { return m_max_batch_size; }
    SizeType get_leaves_stride() const noexcept { return m_leaves_stride; }
    SizeType get_folds_stride() const noexcept { return m_folds_stride; }
    SizeType get_size() const noexcept { return m_size; }
    SizeType get_size_old() const noexcept { return m_size_old; }
    float get_size_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }

    // Get raw span of data
    cuda::std::span<const double> get_leaves_span() const noexcept {
        return {thrust::raw_pointer_cast(m_leaves.data()), m_leaves.size()};
    }
    cuda::std::span<const FoldTypeCUDA> get_folds_span() const noexcept {
        return {thrust::raw_pointer_cast(m_folds.data()), m_folds.size()};
    }
    cuda::std::span<const float> get_scores_span() const noexcept {
        return {thrust::raw_pointer_cast(m_scores.data()), m_scores.size()};
    }

    /**
     * @brief Get maximum score in current region
     */
    float get_score_max() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const SizeType start = get_current_start_idx();
        if (start + m_size <= m_capacity) {
            // Contiguous case - direct reduction
            return thrust::reduce(thrust::device, m_scores.begin() + start,
                                  m_scores.begin() + start + m_size,
                                  -cuda::std::numeric_limits<float>::infinity(),
                                  ThrustMaxOp<float>());
        } // Wrapped case - two reductions
        const SizeType first_count = m_capacity - start;

        float max1 = thrust::reduce(
            thrust::device, m_scores.begin() + start, m_scores.end(),
            -cuda::std::numeric_limits<float>::infinity(),
            ThrustMaxOp<float>());
        float max2 =
            thrust::reduce(thrust::device, m_scores.begin(),
                           m_scores.begin() + (m_size - first_count),
                           -cuda::std::numeric_limits<float>::infinity(),
                           ThrustMaxOp<float>());
        return std::max(max1, max2);
    }

    /**
     * @brief Get minimum score in current region
     */
    float get_score_min() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const SizeType start = get_current_start_idx();

        if (start + m_size <= m_capacity) {
            return thrust::reduce(thrust::device, m_scores.begin() + start,
                                  m_scores.begin() + start + m_size,
                                  cuda::std::numeric_limits<float>::infinity(),
                                  ThrustMinOp<float>());
        }
        const SizeType first_count = m_capacity - start;

        float min1 = thrust::reduce(
            thrust::device, m_scores.begin() + start, m_scores.end(),
            cuda::std::numeric_limits<float>::infinity(), ThrustMinOp<float>());
        float min2 = thrust::reduce(
            thrust::device, m_scores.begin(),
            m_scores.begin() + (m_size - first_count),
            cuda::std::numeric_limits<float>::infinity(), ThrustMinOp<float>());
        return std::min(min1, min2);
    }

    /**
     * @brief Estimate GPU memory usage in GiB
     *
     * Includes both base storage and estimated peak temporary allocations.
     */
    float get_memory_usage() const noexcept {
        const auto base_bytes =
            (m_leaves.size() * sizeof(double)) +
            (m_folds.size() * sizeof(FoldTypeCUDA)) +
            (m_scores.size() * sizeof(float)) +
            (m_scratch_leaves.size() * sizeof(double)) +
            (m_scratch_folds.size() * sizeof(FoldTypeCUDA)) +
            (m_scratch_scores.size() * sizeof(float)) +
            (m_scratch_prefix.size() * sizeof(uint32_t)) +
            (m_scratch_mask.size() * sizeof(uint8_t));

        return static_cast<float>(base_bytes) / static_cast<float>(1ULL << 30U);
    }

    /**
     * @brief Get span over leaves for processing
     *
     * During updates, returns span over readable (old) region.
     * The span may be truncated at buffer wrap point.
     *
     * @param n_leaves Number of leaves to access
     * @return Pair of (span, actual_size), where actual_size <= requested
     * n_leaves, limited to contiguous segment before wrap.
     */
    std::pair<cuda::std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const {
        const SizeType available = m_size_old - m_read_consumed;
        error_check::check_less_equal(
            n_leaves, available,
            "get_leaves_span: n_leaves exceeds available space");

        // Compute physical start: relative to current head of read region
        const auto physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        // Compute how many contiguous elements we can take before wrap
        const auto actual_size =
            std::min(n_leaves, m_capacity - physical_start);
        // Return span with correct byte size (actual_size elements, not bytes)
        return {cuda::std::span<const double>(
                    thrust::raw_pointer_cast(m_leaves.data()) +
                        (physical_start * m_leaves_stride),
                    actual_size * m_leaves_stride),
                actual_size};
    }

    cuda::std::span<double> get_leaves_contiguous_span() noexcept {
        const auto start_idx     = get_current_start_idx();
        const auto report_stride = m_nparams * kParamStride;
        const SizeType src_pitch = m_leaves_stride * sizeof(double);
        const SizeType dst_pitch = report_stride * sizeof(double);
        const SizeType row_bytes = report_stride * sizeof(double);

        const double* __restrict__ src =
            thrust::raw_pointer_cast(m_leaves.data());
        double* __restrict__ dst =
            thrust::raw_pointer_cast(m_scratch_leaves.data());

        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single copy
            cuda_utils::check_cuda_call(
                cudaMemcpy2DAsync(dst, dst_pitch,
                                  src + start_idx * m_leaves_stride, src_pitch,
                                  row_bytes, m_size, cudaMemcpyDeviceToDevice),
                "cudaMemcpy2DAsync leaves failed");
        } else {
            // Wrapped case - two contiguous copies
            const auto first_part      = m_capacity - start_idx;
            const SizeType second_part = m_size - first_part;
            cuda_utils::check_cuda_call(
                cudaMemcpy2DAsync(
                    dst, dst_pitch, src + start_idx * m_leaves_stride,
                    src_pitch, row_bytes, first_part, cudaMemcpyDeviceToDevice),
                "cudaMemcpy2DAsync leaves failed");
            cuda_utils::check_cuda_call(
                cudaMemcpy2DAsync(dst + first_part * report_stride, dst_pitch,
                                  src, src_pitch, row_bytes, second_part,
                                  cudaMemcpyDeviceToDevice),
                "cudaMemcpy2DAsync leaves failed");
        }
        return {thrust::raw_pointer_cast(m_scratch_leaves.data()),
                m_size * report_stride};
    }

    /**
     * @brief Get span over contiguous scores (for saving to file)
     *
     * Returns span over contiguous scores for m_size elements in the current
     * region.
     *
     * @return Span over contiguous scores
     */
    cuda::std::span<float> get_scores_contiguous_span() noexcept {
        const auto start_idx = get_current_start_idx();
        copy_from_circular(thrust::raw_pointer_cast(m_scores.data()), start_idx,
                           m_size, SizeType{1},
                           thrust::raw_pointer_cast(m_scratch_scores.data()));
        return {thrust::raw_pointer_cast(m_scratch_scores.data()), m_size};
    }

    // Mutation operations

    /**
     * @brief Set size externally (for initialization)
     */
    void set_size(SizeType size) noexcept {
        m_size     = size;
        m_head     = 0;
        m_size_old = 0;
        error_check::check_less_equal(
            m_size, m_capacity, "WorldTreeCUDA: Invalid size after set_size");
    }

    /**
     * @brief Reset buffer to empty state
     */
    void reset() noexcept {
        m_size          = 0;
        m_head          = 0;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
        m_write_head    = 0;
        m_write_start   = 0;
    }

    /**
     * @brief Prepare for in-place update
     *
     * Freezes current data as read region, opens write region.
     */
    void prepare_in_place_update() {
        error_check::check_equal(
            m_is_updating, false,
            "Cannot prepare for update while already updating");
        m_size_old      = m_size;
        m_write_start   = get_circular_index(m_size, m_head, m_capacity);
        m_write_head    = m_write_start;
        m_size          = 0; // The new size starts at 0
        m_is_updating   = true;
        m_read_consumed = 0;
        validate_circular_buffer_state();
    }

    /**
     * @brief Finalize in-place update
     *
     * Promotes the write region to be the new read region.
     * All old data must have been consumed.
     */
    void finalize_in_place_update() {
        error_check::check_equal(
            m_read_consumed, m_size_old,
            "finalize_in_place_update: not all old data consumed");
        m_head          = m_write_start;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
        m_write_head    = 0; // Reset for safety
        m_write_start   = 0;
    }

    /**
     * @brief Advance read consumed counter
     */
    void consume_read(SizeType n) {
        error_check::check_less_equal(m_read_consumed + n, m_size_old,
                                      "WorldTreeCUDA: read_consumed overflow");
        m_read_consumed += n;

        // Validate circular buffer invariant
        error_check::check_less_equal(
            m_size_old - m_read_consumed + m_size, m_capacity,
            "WorldTreeCUDA: circular buffer invariant violated");
    }

    /**
     * @brief Convert logical indices to physical indices
     */
    void convert_to_physical_indices(cuda::std::span<uint32_t> logical_indices,
                                     SizeType n_leaves,
                                     cudaStream_t stream) const {
        error_check::check(
            m_is_updating,
            "WorldTreeCUDA: convert_to_physical_indices only valid "
            "during updates");
        error_check::check_greater_equal(
            logical_indices.size(), n_leaves,
            "convert_to_physical_indices: n_leaves size insufficient");

        // Compute physical start: relative to current head of read region
        const SizeType physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);

        CircularIndexFunctor functor{
            .start    = static_cast<uint32_t>(physical_start),
            .capacity = static_cast<uint32_t>(m_capacity)};
        thrust::transform(thrust::cuda::par.on(stream), logical_indices.begin(),
                          logical_indices.begin() +
                              static_cast<IndexType>(n_leaves),
                          logical_indices.begin(), functor);
        cuda_utils::check_last_cuda_error("convert_to_physical_indices failed");
    }

    /**
     * @brief Add initial batch (resets buffer first)
     */
    void add_initial(cuda::std::span<const double> leaves_batch,
                     cuda::std::span<const FoldTypeCUDA> folds_batch,
                     cuda::std::span<const float> scores_batch,
                     SizeType slots_to_write) {
        error_check::check_less_equal(
            slots_to_write, m_capacity,
            "WorldTreeCUDA: Suggestions too large to add.");
        error_check::check_equal(slots_to_write, scores_batch.size(),
                                 "slots_to_write must match batch_scores size");

        reset(); // Start fresh
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()),
                            leaves_batch.data(),
                            slots_to_write * m_leaves_stride * sizeof(double),
                            cudaMemcpyDeviceToDevice),
            "cudaMemcpyAsync leaves failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                thrust::raw_pointer_cast(m_folds.data()), folds_batch.data(),
                slots_to_write * m_folds_stride * sizeof(FoldTypeCUDA),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpyAsync folds failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_scores.data()),
                            scores_batch.data(), slots_to_write * sizeof(float),
                            cudaMemcpyDeviceToDevice),
            "cudaMemcpyAsync scores failed");
        m_size = slots_to_write;
        error_check::check_less_equal(
            m_size, m_capacity,
            "WorldTreeCUDA: Invalid size after add_initial.");
    }

    /**
     * @brief Add batch during update with threshold filtering
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * top-k threshold. Makes sure all candidates fit, reclaiming space
     * from consumed old candidates.
     */
    float add_batch(cuda::std::span<const double> leaves_batch,
                    cuda::std::span<const FoldTypeCUDA> folds_batch,
                    cuda::std::span<const float> scores_batch,
                    float current_threshold,
                    SizeType slots_to_write,
                    cudaStream_t stream) {
        if (slots_to_write == 0) {
            return current_threshold;
        }
        const double* __restrict__ leaves_batch_ptr =
            thrust::raw_pointer_cast(leaves_batch.data());
        const FoldTypeCUDA* __restrict__ folds_batch_ptr =
            thrust::raw_pointer_cast(folds_batch.data());
        const float* __restrict__ scores_batch_ptr =
            thrust::raw_pointer_cast(scores_batch.data());
        double* __restrict__ m_leaves_ptr =
            thrust::raw_pointer_cast(m_leaves.data());
        FoldTypeCUDA* __restrict__ m_folds_ptr =
            thrust::raw_pointer_cast(m_folds.data());
        float* __restrict__ m_scores_ptr =
            thrust::raw_pointer_cast(m_scores.data());

        // Fast path: Check if we have enough space immediately
        SizeType space_left = calculate_space_left();
        if (slots_to_write <= space_left) {
            copy_to_circular(leaves_batch_ptr, m_leaves_ptr, m_write_head,
                             slots_to_write, m_leaves_stride, stream);
            copy_to_circular(folds_batch_ptr, m_folds_ptr, m_write_head,
                             slots_to_write, m_folds_stride, stream);
            copy_to_circular(scores_batch_ptr, m_scores_ptr, m_write_head,
                             slots_to_write, SizeType{1}, stream);
            m_write_head =
                get_circular_index(slots_to_write, m_write_head, m_capacity);
            m_size += slots_to_write;
            return current_threshold;
        }

        // Slow path: Overflow & Pruning (Union Top-K Strategy)
        error_check::check_less_equal(
            slots_to_write, m_max_batch_size,
            "WorldTreeCUDA: Suggestions too large to add.");
        // Determine the Global Pruning Threshold
        const float effective_threshold = get_prune_threshold(
            scores_batch, slots_to_write, current_threshold);
        // Remove old items that are strictly worse than the new global pruning
        // threshold.
        prune_on_overload(effective_threshold, stream);

        double* __restrict__ tmp_leaves_ptr =
            thrust::raw_pointer_cast(m_scratch_leaves.data());
        FoldTypeCUDA* __restrict__ tmp_folds_ptr =
            thrust::raw_pointer_cast(m_scratch_folds.data());
        float* __restrict__ tmp_scores_ptr =
            thrust::raw_pointer_cast(m_scratch_scores.data());

        LinearMaskFunctor functor{.scores_ptr = scores_batch_ptr,
                                  .threshold  = effective_threshold};
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<uint32_t>(0),
                          thrust::counting_iterator<uint32_t>(slots_to_write),
                          m_scratch_mask.begin(), functor);
        cuda_utils::check_last_cuda_error("LinearMaskFunctor failed");
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream), m_scratch_mask.begin(),
            m_scratch_mask.begin() + static_cast<IndexType>(slots_to_write),
            m_scratch_prefix.begin(), uint32_t{0});
        cuda_utils::check_last_cuda_error("exclusive_scan failed");

        const dim3 block_dim(256);
        const dim3 grid_dim((slots_to_write + block_dim.x - 1) / block_dim.x);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        compact_and_copy_kernel<<<grid_dim, block_dim, 0, stream>>>(
            leaves_batch_ptr, folds_batch_ptr, scores_batch_ptr,
            thrust::raw_pointer_cast(m_scratch_mask.data()),
            thrust::raw_pointer_cast(m_scratch_prefix.data()), tmp_leaves_ptr,
            tmp_folds_ptr, tmp_scores_ptr, slots_to_write, m_leaves_stride,
            m_folds_stride);
        cuda_utils::check_last_cuda_error("compact_and_copy_kernel failed");
        const SizeType pending_count = m_scratch_prefix[slots_to_write - 1] +
                                       m_scratch_mask[slots_to_write - 1];

        space_left          = calculate_space_left();
        const auto n_to_add = std::min(pending_count, space_left);
        if (n_to_add == 0) {
            return effective_threshold;
        }
        copy_to_circular(tmp_leaves_ptr, m_leaves_ptr, m_write_head, n_to_add,
                         m_leaves_stride, stream);
        copy_to_circular(tmp_folds_ptr, m_folds_ptr, m_write_head, n_to_add,
                         m_folds_stride, stream);
        copy_to_circular(tmp_scores_ptr, m_scores_ptr, m_write_head, n_to_add,
                         SizeType{1}, stream);
        m_write_head = get_circular_index(n_to_add, m_write_head, m_capacity);
        m_size += n_to_add;
        return effective_threshold;
    }

private:
    static constexpr SizeType kParamStride = 2;

    // Configuration
    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    SizeType m_max_batch_size;
    SizeType m_leaves_stride;
    SizeType m_folds_stride;

    // Device storage
    thrust::device_vector<double> m_leaves;
    thrust::device_vector<FoldTypeCUDA> m_folds;
    thrust::device_vector<float> m_scores;

    // Circular buffer state (host-side tracking)
    SizeType m_head{0};
    SizeType m_size{0};
    SizeType m_size_old{0};
    SizeType m_write_head{0};
    SizeType m_write_start{0};
    bool m_is_updating{false};
    SizeType m_read_consumed{0};

    // Scratch buffer for in-place operations
    thrust::device_vector<double> m_scratch_leaves;
    thrust::device_vector<FoldTypeCUDA> m_scratch_folds;
    thrust::device_vector<float> m_scratch_scores;
    thrust::device_vector<uint32_t> m_scratch_prefix;
    thrust::device_vector<uint8_t> m_scratch_mask;

    /**
     * @brief Copy slots from contiguous source to circular buffer
     *
     * @param src Source pointer (contiguous)
     * @param dst Destination pointer (circular buffer)
     * @param dst_start_slot Starting slot in destination (circular buffer)
     * @param slots Number of slots to copy
     * @param stride Stride of the elements
     */
    template <typename T>
    void copy_to_circular(const T* __restrict__ src,
                          T* __restrict__ dst,
                          SizeType dst_start_slot,
                          SizeType slots,
                          SizeType stride,
                          cudaStream_t stream) const noexcept {
        const auto first_slots = std::min(slots, m_capacity - dst_start_slot);
        const auto first_elems = first_slots * stride;
        const auto dst_offset  = dst_start_slot * stride;

        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(dst + dst_offset, src, first_elems * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync copy_to_circular failed");
        const auto second_slots = slots - first_slots;
        if (second_slots > 0) {
            const auto second_elems = second_slots * stride;
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(dst, src + first_elems,
                                second_elems * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync copy_to_circular failed");
        }
    }

    /**
     * @brief Copy slots from circular buffer to contiguous destination
     *
     * @param src Source pointer (circular buffer)
     * @param src_start_slot Starting slot in source (circular buffer)
     * @param slots Number of slots to copy from circular buffer
     * @param stride Stride of the elements
     * @param dst Destination pointer (contiguous)
     */
    template <typename T>
    void copy_from_circular(const T* __restrict__ src,
                            SizeType src_start_slot,
                            SizeType slots,
                            SizeType stride,
                            T* __restrict__ dst) const noexcept {
        const auto first_slots = std::min(slots, m_capacity - src_start_slot);
        const auto first_elems = first_slots * stride;
        const auto src_offset  = src_start_slot * stride;

        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(dst, src + src_offset, first_elems * sizeof(T),
                            cudaMemcpyDeviceToDevice),
            "cudaMemcpyAsync copy_from_circular failed");
        const auto second_slots = slots - first_slots;
        if (second_slots > 0) {
            const auto second_elems = second_slots * stride;
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(dst + first_elems, src,
                                second_elems * sizeof(T),
                                cudaMemcpyDeviceToDevice),
                "cudaMemcpyAsync copy_from_circular failed");
        }
    }

    /**
     * @brief Get the starting index for current region
     */
    SizeType get_current_start_idx() const noexcept {
        return m_is_updating ? m_write_start : m_head;
    }

    /**
     * @brief Compute physical index from logical index in circular buffer
     * @param logical_idx Logical index (0-based from start of valid region)
     * @param start Starting physical index of the region
     * @param capacity Total buffer capacity
     * @return Physical index in the buffer
     */
    static constexpr SizeType get_circular_index(SizeType logical_idx,
                                                 SizeType start,
                                                 SizeType capacity) noexcept {
        return (start + logical_idx) % capacity;
    }

    /**
     * @brief Calculate available space in buffer
     */
    SizeType calculate_space_left() const noexcept {
        const auto remaining_old = static_cast<IndexType>(m_size_old) -
                                   static_cast<IndexType>(m_read_consumed);
        error_check::check_greater_equal(
            remaining_old, 0, "calculate_space_left: Invalid buffer state");

        const auto space_left =
            static_cast<IndexType>(m_capacity) -
            (remaining_old + static_cast<IndexType>(m_size));
        error_check::check_greater_equal(
            space_left, 0, "calculate_space_left: Invalid buffer state");
        return static_cast<SizeType>(space_left);
    }

    /**
     * @brief Get prune threshold in current region
     *
     * find a threshold in (Buffer + Batch) so that keeping only scores strictly
     * above it yields at most total_capacity items.
     */
    float get_prune_threshold(cuda::std::span<const float> scores_batch,
                              SizeType slots_to_write,
                              float current_threshold) noexcept {
        if (slots_to_write == 0) {
            return current_threshold;
        }
        const auto space_left       = calculate_space_left();
        const auto total_candidates = m_size + slots_to_write;
        const auto total_capacity   = m_size + space_left;
        if (total_candidates <= total_capacity) {
            return current_threshold;
        }
        error_check::check_greater_equal(
            m_scratch_scores.size(), total_candidates,
            "get_prune_threshold: Invalid scratch buffer size");
        // Gather all candidates (Buffer + Batch) into scratch
        // Copy active circular buffer scores to scratch [0 ... m_size]
        const auto start_idx = get_current_start_idx();
        copy_from_circular(thrust::raw_pointer_cast(m_scores.data()), start_idx,
                           m_size, SizeType{1},
                           thrust::raw_pointer_cast(m_scratch_scores.data()));
        // Append batch scores [m_size ... total]
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                thrust::raw_pointer_cast(m_scratch_scores.data()) + m_size,
                thrust::raw_pointer_cast(scores_batch.data()),
                slots_to_write * sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpyAsync get_prune_threshold failed");

        // The item at 'total_capacity-1' (in descending order) is the smallest
        // score we keep (i.e. the threshold).
        thrust::sort(thrust::device, m_scratch_scores.begin(),
                     m_scratch_scores.begin() + total_candidates,
                     cuda::std::greater<float>());
        const float kth = m_scratch_scores[total_capacity - 1];
        // Median score for severity (restricted range)
        const float mid = m_scratch_scores[total_candidates / 2];
        // To break ties, we use the next representable value greater than the
        // topk and median scores.
        const float topk_threshold =
            std::nextafter(kth, std::numeric_limits<float>::max());
        const float median_threshold =
            std::nextafter(mid, std::numeric_limits<float>::max());
        return std::max({current_threshold, topk_threshold, median_threshold});
    }

    /**
     * @brief Prune write region by threshold with in-place update.
     * @param mode The prune mode to use
     * @return The threshold used
     *
     * Single-pass approach: safely compacts in-place because write never
     * overtakes read (write_logical ≤ read_logical always holds).
     */
    void prune_on_overload(float threshold, cudaStream_t stream) {
        error_check::check(
            m_is_updating,
            "WorldTreeCUDA: prune_on_overload only allowed during updates");
        if (m_size == 0) {
            return;
        }

        const SizeType start_idx = get_current_start_idx();
        // Mark survivors (logical indices) in keep mask
        CircularMaskFunctor functor{
            .scores_ptr = thrust::raw_pointer_cast(m_scores.data()),
            .start      = static_cast<uint32_t>(start_idx),
            .capacity   = static_cast<uint32_t>(m_capacity),
            .threshold  = threshold};
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<uint32_t>(0),
                          thrust::counting_iterator<uint32_t>(m_size),
                          m_scratch_mask.begin(), functor);

        // Exclusive prefix sum to get logical indices to keep
        thrust::exclusive_scan(thrust::cuda::par.on(stream),
                               m_scratch_mask.begin(),
                               m_scratch_mask.begin() + m_size,
                               m_scratch_prefix.begin(), uint32_t{0});

        // In-place compaction (three arrays)
        const dim3 block_dim(256);
        const dim3 grid_dim((m_size + block_dim.x - 1) / block_dim.x);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
            thrust::raw_pointer_cast(m_scratch_mask.data()),
            thrust::raw_pointer_cast(m_scratch_prefix.data()),
            thrust::raw_pointer_cast(m_leaves.data()), start_idx, m_capacity,
            m_size, m_leaves_stride);
        cuda_utils::check_last_cuda_error(
            "compact_in_place_kernel leaves failed");
        compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
            thrust::raw_pointer_cast(m_scratch_mask.data()),
            thrust::raw_pointer_cast(m_scratch_prefix.data()),
            thrust::raw_pointer_cast(m_folds.data()), start_idx, m_capacity,
            m_size, m_folds_stride);
        cuda_utils::check_last_cuda_error(
            "compact_in_place_kernel folds failed");
        compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
            thrust::raw_pointer_cast(m_scratch_mask.data()),
            thrust::raw_pointer_cast(m_scratch_prefix.data()),
            thrust::raw_pointer_cast(m_scores.data()), start_idx, m_capacity,
            m_size, 1);
        cuda_utils::check_last_cuda_error(
            "compact_in_place_kernel scores failed");
        auto last_prefix = m_scratch_prefix[m_size - 1];
        auto last_keep   = m_scratch_mask[m_size - 1];
        m_size           = last_prefix + last_keep;
        m_write_head     = get_circular_index(m_size, start_idx, m_capacity);
        error_check::check_less_equal(
            m_size, m_capacity,
            "WorldTreeCUDA: Invalid size after prune_on_overload");
    }

    void validate_circular_buffer_state() {
        // Core invariant: total used space <= capacity
        const auto total_used = (m_size_old - m_read_consumed) + m_size;
        error_check::check_less_equal(total_used, m_capacity,
                                      "Circular buffer invariant violated: "
                                      "total_used > capacity");

        error_check::check_less_equal(m_read_consumed, m_size_old,
                                      "read_consumed cannot exceed size_old");

        error_check::check_less(m_head, m_capacity,
                                "head must be within capacity");

        if (m_is_updating) {
            error_check::check_less(m_write_start, m_capacity,
                                    "write_start must be within capacity");
            error_check::check_less(m_write_head, m_capacity,
                                    "write_head must be within capacity");
        }
    }

}; // End WorldTreeCUDA::Impl definition

// Public interface implementation
template <SupportedFoldTypeCUDA FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::WorldTreeCUDA(SizeType capacity,
                                           SizeType nparams,
                                           SizeType nbins,
                                           SizeType max_batch_size)
    : m_impl(std::make_unique<Impl>(capacity, nparams, nbins, max_batch_size)) {
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::~WorldTreeCUDA() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::WorldTreeCUDA(WorldTreeCUDA&& other) noexcept =
    default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>& WorldTreeCUDA<FoldTypeCUDA>::operator=(
    WorldTreeCUDA<FoldTypeCUDA>&& other) noexcept = default;
// Getters
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const double>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_span() const noexcept {
    return m_impl->get_leaves_span();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::get_folds_span() const noexcept {
    return m_impl->get_folds_span();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const float>
WorldTreeCUDA<FoldTypeCUDA>::get_scores_span() const noexcept {
    return m_impl->get_scores_span();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_capacity() const noexcept {
    return m_impl->get_capacity();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_max_batch_size() const noexcept {
    return m_impl->get_max_batch_size();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_leaves_stride() const noexcept {
    return m_impl->get_leaves_stride();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_folds_stride() const noexcept {
    return m_impl->get_folds_stride();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_size() const noexcept {
    return m_impl->get_size();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_size_old() const noexcept {
    return m_impl->get_size_old();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_size_lb() const noexcept {
    return m_impl->get_size_lb();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_score_max() const noexcept {
    return m_impl->get_score_max();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_score_min() const noexcept {
    return m_impl->get_score_min();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::pair<cuda::std::span<const double>, SizeType>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_span(SizeType n_leaves) const {
    return m_impl->get_leaves_span(n_leaves);
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<double>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_contiguous_span() noexcept {
    return m_impl->get_leaves_contiguous_span();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<float>
WorldTreeCUDA<FoldTypeCUDA>::get_scores_contiguous_span() noexcept {
    return m_impl->get_scores_contiguous_span();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::set_size(SizeType size) noexcept {
    m_impl->set_size(size);
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::reset() noexcept {
    m_impl->reset();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::prepare_in_place_update() {
    m_impl->prepare_in_place_update();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::finalize_in_place_update() {
    m_impl->finalize_in_place_update();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::consume_read(SizeType n) {
    m_impl->consume_read(n);
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::convert_to_physical_indices(
    cuda::std::span<uint32_t> logical_indices,
    SizeType n_leaves,
    cudaStream_t stream) const {
    m_impl->convert_to_physical_indices(logical_indices, n_leaves, stream);
}
// Other methods
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::add_initial(
    cuda::std::span<const double> leaves_batch,
    cuda::std::span<const FoldTypeCUDA> folds_batch,
    cuda::std::span<const float> scores_batch,
    SizeType slots_to_write) {
    m_impl->add_initial(leaves_batch, folds_batch, scores_batch,
                        slots_to_write);
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::add_batch(
    cuda::std::span<const double> leaves_batch,
    cuda::std::span<const FoldTypeCUDA> folds_batch,
    cuda::std::span<const float> scores_batch,
    float current_threshold,
    SizeType slots_to_write,
    cudaStream_t stream) {
    return m_impl->add_batch(leaves_batch, folds_batch, scores_batch,
                             current_threshold, slots_to_write, stream);
}
// Explicit instantiation
template class WorldTreeCUDA<float>;
template class WorldTreeCUDA<ComplexTypeCUDA>;
} // namespace loki::utils