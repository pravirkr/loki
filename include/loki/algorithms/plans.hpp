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

namespace detail {

/**
 * @brief Base class for an FFA search plan.
 * @details
 * This class holds all lightweight type-invariant (non-template) data and logic
 * for an FFA plan. This includes parameter grid sizes and segment lengths.
 */
class FFAPlanBase {
protected:
    search::PulsarSearchConfig m_cfg;
    SizeType m_n_params{};
    SizeType m_n_levels{};
    std::vector<SizeType> m_segment_lens;
    std::vector<SizeType> m_nsegments;
    std::vector<double> m_tsegments;
    std::vector<SizeType> m_ncoords;
    std::vector<float> m_ncoords_lb;
    std::vector<uint32_t> m_ncoords_offsets;
    std::vector<uint32_t> m_params_flat_offsets;
    std::vector<uint32_t> m_params_flat_sizes;
    std::vector<std::vector<SizeType>> m_param_counts;
    std::vector<std::vector<SizeType>> m_param_cart_strides;
    std::vector<std::vector<double>> m_dparams;
    std::vector<std::vector<double>> m_dparams_lim;

public:
    /**
     * @brief Constructs the FFA plan from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlanBase(search::PulsarSearchConfig cfg);

    virtual ~FFAPlanBase()                         = default;
    FFAPlanBase(const FFAPlanBase&)                = delete;
    FFAPlanBase& operator=(const FFAPlanBase&)     = delete;
    FFAPlanBase(FFAPlanBase&&) noexcept            = default;
    FFAPlanBase& operator=(FFAPlanBase&&) noexcept = default;

    // --- Getters ---
    /// @brief Get the search configuration object used to build the plan.
    [[nodiscard]] const search::PulsarSearchConfig&
    get_config() const noexcept {
        return m_cfg;
    }
    /// @brief Number of parameters to search over (..., a, f).
    SizeType get_n_params() const noexcept { return m_n_params; }
    /// @brief Number of FFA merge levels.
    SizeType get_n_levels() const noexcept { return m_n_levels; }
    /// @brief Segment length for each level.
    [[nodiscard]] std::span<const SizeType> get_segment_lens() const noexcept {
        return m_segment_lens;
    }
    /// @brief Number of segments for each level.
    [[nodiscard]] std::span<const SizeType> get_nsegments() const noexcept {
        return m_nsegments;
    }
    /// @brief Segment lengths in seconds.
    [[nodiscard]] std::span<const double> get_tsegments() const noexcept {
        return m_tsegments;
    }
    /// @brief Number of coordinates for each level.
    [[nodiscard]] std::span<const SizeType> get_ncoords() const noexcept {
        return m_ncoords;
    }
    /// @brief Log2 of number of coordinates for each level.
    [[nodiscard]] std::span<const float> get_ncoords_lb() const noexcept {
        return m_ncoords_lb;
    }
    /// @brief Offset number of coordinates for each level (cumulative sum)
    /// including final sentinel.
    [[nodiscard]] std::span<const uint32_t>
    get_ncoords_offsets() const noexcept {
        return m_ncoords_offsets;
    }
    /// @brief Offset of the flattened parameters counts for each level
    /// (cumulative sum) excluding final sentinel.
    [[nodiscard]] std::span<const uint32_t>
    get_params_flat_offsets() const noexcept {
        return m_params_flat_offsets;
    }
    /// @brief Flattened parameters counts for each level.
    [[nodiscard]] std::span<const uint32_t>
    get_params_flat_sizes() const noexcept {
        return m_params_flat_sizes;
    }
    /// @brief Parameter grid count for each parameter.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_param_counts() const noexcept {
        return m_param_counts;
    }
    /// @brief Cartesian strides for each parameter in the parameter grid.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_param_cart_strides() const noexcept {
        return m_param_cart_strides;
    }
    /// @brief Grid step sizes for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams() const noexcept {
        return m_dparams;
    }
    /// @brief Grid step size (limited) for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams_lim() const noexcept {
        return m_dparams_lim;
    }
    /// @brief Get the total number of coordinates across all levels.
    SizeType get_coord_size() const noexcept;
    /// @brief Get the total number of flattened parameters counts.
    SizeType get_total_params_flat_count() const noexcept;
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
    void configure_base_plan();
    void validate_base_plan() const;
    static std::vector<SizeType>
    calculate_strides(std::span<const SizeType> p_arr_counts);
};

} // namespace detail

/**
 * @brief A type-aware full FFA plan (Time or Fourier domain).
 * @details
 * This class inherits all common plan logic from FFAPlanBase and adds
 * type-specific data (fold shapes, etc.). This class is used to store the
 * metadata for an FFA plan, which is used to store the fold shapes, etc.
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType>
class FFAPlan : public detail::FFAPlanBase {
protected:
    std::vector<std::vector<SizeType>> m_fold_shapes;
    std::vector<std::vector<SizeType>> m_fold_shapes_time;
    SizeType m_flops_required{0U};
    SizeType m_flops_required_return_in_time{0U};

public:
    /**
     * @brief Constructs the metadata for a type-aware FFA plan.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlan(search::PulsarSearchConfig cfg);

    ~FFAPlan() override                    = default;
    FFAPlan(const FFAPlan&)                = delete;
    FFAPlan& operator=(const FFAPlan&)     = delete;
    FFAPlan(FFAPlan&&) noexcept            = default;
    FFAPlan& operator=(FFAPlan&&) noexcept = default;

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

    // Get the flattened parameters for CUDA plans
    [[nodiscard]] std::vector<double> get_params_flat() const noexcept;
    // Get the flattened parameter counts for CUDA plans
    [[nodiscard]] std::vector<uint32_t> get_param_counts_flat() const noexcept;

private:
    void configure_fold_shapes();
    void compute_flops();
    void validate_plan() const;
};

} // namespace loki::plans