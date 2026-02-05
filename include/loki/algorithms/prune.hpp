#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/progress.hpp"
#include "loki/search/configs.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

class PruningManager {
public:
    PruningManager(search::PulsarSearchConfig cfg,
                   std::span<const float> threshold_scheme,
                   std::optional<SizeType> n_runs                = std::nullopt,
                   std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                   SizeType max_sugg                             = 1U << 18U,
                   SizeType batch_size                           = 1024U);
    ~PruningManager();
    PruningManager(PruningManager&&) noexcept;
    PruningManager& operator=(PruningManager&&) noexcept;
    PruningManager(const PruningManager&)            = delete;
    PruningManager& operator=(const PruningManager&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test",
                 std::string_view poly_basis         = "taylor",
                 bool show_progress                  = true);

    // Opaque handle to the implementation
    class BaseImpl;

private:
    std::unique_ptr<BaseImpl> m_impl;
};

template <SupportedFoldType FoldType> class Prune {
public:
    Prune(plans::FFAPlan<FoldType> ffa_plan,
          search::PulsarSearchConfig cfg,
          std::span<const float> threshold_scheme,
          SizeType max_sugg           = 1U << 18U,
          SizeType batch_size         = 1024U,
          std::string_view poly_basis = "taylor");

    ~Prune();
    Prune(Prune&&) noexcept;
    Prune& operator=(Prune&&) noexcept;
    Prune(const Prune&)            = delete;
    Prune& operator=(const Prune&) = delete;

    SizeType get_memory_usage() const noexcept;

    void execute(
        std::span<const FoldType> ffa_fold,
        SizeType ref_seg,
        const std::filesystem::path& outdir                     = "./",
        const std::optional<std::filesystem::path>& log_file    = std::nullopt,
        const std::optional<std::filesystem::path>& result_file = std::nullopt,
        progress::MultiprocessProgressTracker* tracker          = nullptr,
        int task_id                                             = 0,
        bool show_progress                                      = true);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using PruneTime    = Prune<float>;
using PruneFourier = Prune<ComplexType>;

#ifdef LOKI_ENABLE_CUDA

class PruningManagerCUDA {
public:
    PruningManagerCUDA(
        search::PulsarSearchConfig cfg,
        std::span<const float> threshold_scheme,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        SizeType max_sugg                             = 1U << 20U,
        SizeType batch_size                           = 4096U,
        int device_id                                 = 0);
    ~PruningManagerCUDA();
    PruningManagerCUDA(PruningManagerCUDA&&) noexcept;
    PruningManagerCUDA& operator=(PruningManagerCUDA&&) noexcept;
    PruningManagerCUDA(const PruningManagerCUDA&)            = delete;
    PruningManagerCUDA& operator=(const PruningManagerCUDA&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test",
                 std::string_view poly_basis         = "taylor");

    // Opaque handle to the implementation
    class BaseImpl;

private:
    std::unique_ptr<BaseImpl> m_impl;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> class PruneCUDA {
public:
    using HostFoldType   = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = typename FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    PruneCUDA(plans::FFAPlan<HostFoldType> ffa_plan,
              search::PulsarSearchConfig cfg,
              std::span<const float> threshold_scheme,
              SizeType max_sugg           = 1U << 18U,
              SizeType batch_size         = 1024U,
              std::string_view poly_basis = "taylor",
              int device_id               = 0);

    ~PruneCUDA();
    PruneCUDA(PruneCUDA&&) noexcept;
    PruneCUDA& operator=(PruneCUDA&&) noexcept;
    PruneCUDA(const PruneCUDA&)            = delete;
    PruneCUDA& operator=(const PruneCUDA&) = delete;

    SizeType get_memory_usage() const noexcept;

    void execute(
        cuda::std::span<const FoldTypeCUDA> ffa_fold,
        SizeType ref_seg,
        const std::filesystem::path& outdir                     = "./",
        const std::optional<std::filesystem::path>& log_file    = std::nullopt,
        const std::optional<std::filesystem::path>& result_file = std::nullopt,
        int task_id                                             = 0);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using PruneTimeCUDA    = PruneCUDA<float>;
using PruneFourierCUDA = PruneCUDA<ComplexTypeCUDA>;

#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms