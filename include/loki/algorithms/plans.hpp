#pragma once

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::plans {

/**
 * @brief Base class for an FFA search plan.
 * @details
 * This class holds all type-invariant (non-template) data and logic
 * for an FFA plan. This includes parameter grid sizes and segment lengths.
 */
class FFAPlanBase {
public:
    /**
     * @brief Constructs the FFA plan from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlanBase(const search::PulsarSearchConfig& cfg);

    // --- Rule of five: PIMPL ---
    virtual ~FFAPlanBase();
    FFAPlanBase(FFAPlanBase&&) noexcept;
    FFAPlanBase& operator=(FFAPlanBase&&) noexcept;
    FFAPlanBase(const FFAPlanBase&)            = delete;
    FFAPlanBase& operator=(const FFAPlanBase&) = delete;

    // --- Getters ---
    /// @brief Get the search configuration object used to build the plan.
    [[nodiscard]] const search::PulsarSearchConfig& get_config() const noexcept;
    /// @brief Number of parameters to search over (..., a, f).
    [[nodiscard]] SizeType get_n_params() const noexcept;
    /// @brief Number of FFA merge levels.
    [[nodiscard]] SizeType get_n_levels() const noexcept;
    /// @brief Segment length for each level.
    [[nodiscard]] std::span<const SizeType> get_segment_lens() const noexcept;
    /// @brief Number of segments for each level.
    [[nodiscard]] std::span<const SizeType> get_nsegments() const noexcept;
    /// @brief Segment lengths in seconds.
    [[nodiscard]] std::span<const double> get_tsegments() const noexcept;
    /// @brief Number of coordinates for each level.
    [[nodiscard]] std::span<const SizeType> get_ncoords() const noexcept;
    /// @brief Log2 of number of coordinates for each level.
    [[nodiscard]] std::span<const float> get_ncoords_lb() const noexcept;
    /// @brief Offset number of coordinates for each level (cumulative sum)
    /// including final sentinel.
    [[nodiscard]] std::span<const uint32_t>
    get_ncoords_offsets() const noexcept;
    /// @brief Parameter grid count for each parameter.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_param_counts() const noexcept;
    // Get the flattened parameter counts for CUDA plans
    [[nodiscard]] std::span<const uint32_t>
    get_param_counts_flat() const noexcept;
    /// @brief Cartesian strides for each parameter in the parameter grid.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_param_cart_strides() const noexcept;
    /// @brief Grid step sizes for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams() const noexcept;
    /// @brief Grid step size (limited) for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams_lim() const noexcept;
    /// @brief Get the total number of coordinates across all levels.
    SizeType get_coord_size() const noexcept;
    /// @brief Get the memory usage of the coordinate storage (in GB).
    float get_coord_memory_usage() const noexcept;

    // --- Methods ---
    /**
     * @brief Resolve the coordinates for the plan.
     * @param coords A span of FFACoord objects.
     */
    void resolve_coordinates(std::span<coord::FFACoord> coords);

    /**
     * @brief Resolve the coordinates for the plan.
     * @return A vector of vectors of FFACoord for each level.
     */
    std::vector<std::vector<coord::FFACoord>> resolve_coordinates();

    /**
     * @brief Resolve the coordinates for the plan (frequency-only coordinates).
     * @param coords_freq A span of FFACoordFreq objects.
     */
    void resolve_coordinates_freq(std::span<coord::FFACoordFreq> coords_freq);

    /**
     * @brief Resolve the coordinates for the plan (frequency-only coordinates).
     * @return A vector of vectors of FFACoordFreq for each level.
     */
    std::vector<std::vector<coord::FFACoordFreq>> resolve_coordinates_freq();

    /// @brief Compute the parameter grid for a given FFA level.
    [[nodiscard]] std::vector<std::vector<double>>
    compute_param_grid(SizeType ffa_level) const noexcept;

    /// @brief Compute the parameter grid for the entire plan.
    [[nodiscard]] std::vector<std::vector<std::vector<double>>>
    compute_param_grid_full() const noexcept;

    /// @brief Get a dictionary of parameters for the last level of the plan.
    [[nodiscard]] std::map<std::string, std::vector<double>>
    get_params_dict() const;

    /**
     * @brief Get the approximate branching pattern for the plan.
     * @param poly_basis The polynomial basis for the branching pattern (e.g.
     * "taylor").
     * @param ref_seg The reference segment for the branching pattern.
     * @param isuggest The index of the leaf to use for the branching pattern.
     * @return A vector of branching pattern values.
     */
    std::vector<double>
    get_branching_pattern_approx(std::string_view poly_basis = "taylor",
                                 SizeType ref_seg            = 0,
                                 IndexType isuggest          = 0) const;

    /**
     * @brief Get the exact branching pattern for the plan.
     * @param poly_basis The polynomial basis for the branching pattern (e.g.
     * "taylor").
     * @param ref_seg The reference segment for the branching pattern.
     * @return A vector of branching pattern values.
     */
    std::vector<double>
    get_branching_pattern(std::string_view poly_basis = "taylor",
                          SizeType ref_seg            = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief A type-aware FFA plan (Time or Fourier domain).
 * @details
 * This class inherits all common plan logic from FFAPlanBase and adds
 * type-specific data.
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> class FFAPlan final : public FFAPlanBase {
public:
    /**
     * @brief Constructs the full, type-aware plan.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlan(const search::PulsarSearchConfig& cfg);

    // --- Rule of five: PIMPL ---
    ~FFAPlan() override;
    FFAPlan(FFAPlan&&) noexcept            = default;
    FFAPlan& operator=(FFAPlan&&) noexcept = default;
    FFAPlan(const FFAPlan&)                = delete;
    FFAPlan& operator=(const FFAPlan&)     = delete;

    // --- Getters ---
    /// @brief Get the fold shapes for each level.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_fold_shapes() const noexcept {
        return m_fold_shapes;
    }
    /// @brief Get the fold shapes for each level (time domain).
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_fold_shapes_time() const noexcept {
        return m_fold_shapes_time;
    }
    /// @brief Get the size of the brute fold buffer.
    SizeType get_brute_fold_size() const noexcept;
    /// @brief Get the size of the fold buffer.
    SizeType get_fold_size() const noexcept;
    /// @brief Get the size of the fold buffer (time domain).
    SizeType get_fold_size_time() const noexcept;
    /// @brief Get the size of the FFA workspace buffer.
    SizeType get_buffer_size() const noexcept;
    /// @brief Get the size of the FFA workspace buffer (time domain).
    SizeType get_buffer_size_time() const noexcept;
    /// @brief Get the memory usage of the FFA workspace buffer (in GB).
    float get_buffer_memory_usage() const noexcept;
    /// @brief Get the compute FLOPS for the FFA plan (in GFLOPS).
    float get_gflops(bool return_in_time) const noexcept;

private:
    std::vector<std::vector<SizeType>> m_fold_shapes;
    std::vector<std::vector<SizeType>> m_fold_shapes_time;
    SizeType m_flops_required{0U};
    SizeType m_flops_required_return_in_time{0U};

    void configure_fold_shapes() noexcept;
    void compute_flops() noexcept;
    void validate() const;
};

} // namespace loki::plans