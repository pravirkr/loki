#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

template <typename FoldType> class PruningManager {
public:
    PruningManager(const search::PulsarSearchConfig& cfg,
                   const std::vector<float>& threshold_scheme,
                   std::optional<SizeType> n_runs                = std::nullopt,
                   std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                   SizeType max_sugg                             = 1U << 18U,
                   SizeType batch_size                           = 1024U,
                   int nthreads                                  = 1);
    ~PruningManager();
    PruningManager(PruningManager&&) noexcept;
    PruningManager& operator=(PruningManager&&) noexcept;
    PruningManager(const PruningManager&)            = delete;
    PruningManager& operator=(const PruningManager&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test",
                 std::string_view kind               = "taylor");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using PruningManagerFloat   = PruningManager<float>;
using PruningManagerComplex = PruningManager<ComplexType>;

template <typename FoldType> class Prune {
public:
    Prune(const plans::FFAPlan& ffa_plan,
          const search::PulsarSearchConfig& cfg,
          std::span<const float> threshold_scheme,
          SizeType max_sugg     = 1U << 18U,
          SizeType batch_size   = 1024U,
          std::string_view kind = "taylor");

    ~Prune();
    Prune(Prune&&) noexcept;
    Prune& operator=(Prune&&) noexcept;
    Prune(const Prune&)            = delete;
    Prune& operator=(const Prune&) = delete;

    void execute(
        std::span<const FoldType> ffa_fold,
        SizeType ref_seg,
        const std::filesystem::path& outdir                     = "./",
        const std::optional<std::filesystem::path>& log_file    = std::nullopt,
        const std::optional<std::filesystem::path>& result_file = std::nullopt);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms