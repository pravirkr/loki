#include <utility>

#include "loki/algorithms/prune.hpp"

namespace loki::algorithms {

class PruningManager::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         const std::vector<float>& threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<const std::vector<SizeType>&> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         int nthreads)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme),
          m_n_runs(n_runs),
          m_ref_segs(ref_segs),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_nthreads(nthreads) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

private:
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    std::optional<int> m_n_runs;
    std::optional<std::vector<int>> m_ref_segs;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    int m_nthreads;
};

class Prune::Impl {
public:
    Impl(plans::FFAPlan& ffa_plan,
         search::PulsarSearchConfig& cfg,
         std::span<const float> threshold_scheme,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view kind)
        : m_ffa_plan(std::move(ffa_plan)),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_kind(kind) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

private:
    plans::FFAPlan m_ffa_plan;
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string_view m_kind;
};

PruningManager::PruningManager(
    const search::PulsarSearchConfig& cfg,
    const std::vector<float>& threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<const std::vector<SizeType>&> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    int nthreads)
    : m_impl(std::make_unique<Impl>(cfg,
                                    threshold_scheme,
                                    n_runs,
                                    ref_segs,
                                    max_sugg,
                                    batch_size,
                                    nthreads)) {}
PruningManager::~PruningManager()                               = default;
PruningManager::PruningManager(PruningManager&& other) noexcept = default;
PruningManager&
PruningManager::operator=(PruningManager&& other) noexcept = default;
void PruningManager::execute(std::span<const float> ts_e,
                             std::span<const float> ts_v,
                             std::filesystem::path outdir,
                             std::string_view file_prefix,
                             std::string_view kind) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix, kind);
}

Prune::Prune(const plans::FFAPlan& ffa_plan,
             const search::PulsarSearchConfig& cfg,
             std::span<const float> threshold_scheme,
             SizeType max_sugg,
             SizeType batch_size,
             std::string_view kind)
    : m_impl(std::make_unique<Impl>(
          ffa_plan, cfg, threshold_scheme, max_sugg, batch_size, kind)) {}
Prune::~Prune()                                 = default;
Prune::Prune(Prune&& other) noexcept            = default;
Prune& Prune::operator=(Prune&& other) noexcept = default;
void Prune::execute(std::span<const float> ffa_fold,
                    SizeType ref_seg,
                    std::filesystem::path outdir,
                    std::optional<std::filesystem::path> log_file,
                    std::optional<std::filesystem::path> result_file) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file);
}

} // namespace loki::algorithms