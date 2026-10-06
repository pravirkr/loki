#include "loki/algorithms/fold.hpp"

#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "algorithms/fold_engine.hpp"
#include "common/dispatch.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms {

namespace {

template <SupportedFoldType FoldType>
std::unique_ptr<detail::BruteFoldEngine<FoldType>>
make_brute_fold_engine(std::span<const double> freq_arr,
                       SizeType segment_len,
                       SizeType nbins,
                       SizeType nsamps,
                       double tsamp,
                       double t_ref,
                       Exec exec) {
    if (exec.backend == Backend::kCPU) {
        return detail::make_brute_fold_cpu<FoldType>(
            freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, exec.nthreads);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_brute_fold_gpu<FoldType>(
            freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, exec.device);
    }
#endif
    loki::detail::throw_unavailable("BruteFold", exec.backend);
}

} // namespace

template <SupportedFoldType FoldType> class BruteFold<FoldType>::Impl {
public:
    Impl(std::span<const double> freq_arr,
         SizeType segment_len,
         SizeType nbins,
         SizeType nsamps,
         double tsamp,
         double t_ref,
         Exec exec)
        : m_exec(exec),
          m_engine(make_brute_fold_engine<FoldType>(
              freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, exec)) {}

    SizeType get_fold_size() const { return m_engine->get_fold_size(); }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) {
        m_engine->execute(ts_e, ts_v, fold);
    }

    void execute(DeviceSpan<const float> ts_e,
                 DeviceSpan<const float> ts_v,
                 DeviceSpan<FoldType> fold,
                 Stream stream) {
        check_device(ts_e.device, "BruteFold::execute");
        check_device(ts_v.device, "BruteFold::execute");
        check_device(fold.device, "BruteFold::execute");
        m_engine->execute(ts_e, ts_v, fold, stream);
    }

    void execute_fused_freq(
        std::span<const float> ts_e,
        std::span<const float> ts_v,
        std::span<float> fold_out,
        std::span<const coord::FFACoordFreq* const> coords_levels,
        std::span<const SizeType> ncoords,
        SizeType nlevels)
        requires(std::is_same_v<FoldType, float>)
    {
        m_engine->execute_fused_freq(ts_e, ts_v, fold_out, coords_levels,
                                     ncoords, nlevels);
    }

    [[nodiscard]] std::span<const coord::PhaseRun> runs() const
        requires(std::is_same_v<FoldType, float>)
    {
        return m_engine->runs();
    }

    [[nodiscard]] std::span<const SizeType> run_offsets() const
        requires(std::is_same_v<FoldType, float>)
    {
        return m_engine->run_offsets();
    }

private:
    Exec m_exec;
    std::unique_ptr<detail::BruteFoldEngine<FoldType>> m_engine;

    void check_device(const Device& view, std::string_view what) const {
        loki::detail::check_device(view, m_exec.backend, m_exec.device, what);
    }
};

template <SupportedFoldType FoldType>
BruteFold<FoldType>::BruteFold(std::span<const double> freq_arr,
                               SizeType segment_len,
                               SizeType nbins,
                               SizeType nsamps,
                               double tsamp,
                               double t_ref,
                               Exec exec)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, exec)) {}

template <SupportedFoldType FoldType>
BruteFold<FoldType>::~BruteFold() = default;

template <SupportedFoldType FoldType>
BruteFold<FoldType>::BruteFold(BruteFold&& other) noexcept = default;

template <SupportedFoldType FoldType>
BruteFold<FoldType>&
BruteFold<FoldType>::operator=(BruteFold&& other) noexcept = default;

template <SupportedFoldType FoldType>
SizeType BruteFold<FoldType>::get_fold_size() const {
    return m_impl->get_fold_size();
}

template <SupportedFoldType FoldType>
void BruteFold<FoldType>::execute(std::span<const float> ts_e,
                                  std::span<const float> ts_v,
                                  std::span<FoldType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

template <SupportedFoldType FoldType>
void BruteFold<FoldType>::execute(DeviceSpan<const float> ts_e,
                                  DeviceSpan<const float> ts_v,
                                  DeviceSpan<FoldType> fold,
                                  Stream stream) {
    m_impl->execute(ts_e, ts_v, fold, stream);
}

template <SupportedFoldType FoldType>
std::span<const coord::PhaseRun> BruteFold<FoldType>::runs() const
    requires(std::is_same_v<FoldType, float>)
{
    return m_impl->runs();
}

template <SupportedFoldType FoldType>
std::span<const SizeType> BruteFold<FoldType>::run_offsets() const
    requires(std::is_same_v<FoldType, float>)
{
    return m_impl->run_offsets();
}

template <SupportedFoldType FoldType>
void BruteFold<FoldType>::execute_fused_freq(
    std::span<const float> ts_e,
    std::span<const float> ts_v,
    std::span<float> fold_out,
    std::span<const coord::FFACoordFreq* const> coords_levels,
    std::span<const SizeType> ncoords,
    SizeType nlevels)
    requires(std::is_same_v<FoldType, float>)
{
    m_impl->execute_fused_freq(ts_e, ts_v, fold_out, coords_levels, ncoords,
                               nlevels);
}

template <SupportedFoldType FoldType>
std::vector<FoldType> compute_brute_fold(std::span<const float> ts_e,
                                         std::span<const float> ts_v,
                                         std::span<const double> freq_arr,
                                         SizeType segment_len,
                                         SizeType nbins,
                                         double tsamp,
                                         double t_ref,
                                         Exec exec) {
    const SizeType nsamps = ts_e.size();
    BruteFold<FoldType> bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref,
                           exec);
    std::vector<FoldType> fold(bf.get_fold_size(), FoldType{});
    bf.execute(ts_e, ts_v, std::span<FoldType>(fold));
    return fold;
}

// Explicit instantiation
template class BruteFold<float>;
template class BruteFold<ComplexType>;

template std::vector<float> compute_brute_fold<float>(std::span<const float>,
                                                      std::span<const float>,
                                                      std::span<const double>,
                                                      SizeType,
                                                      SizeType,
                                                      double,
                                                      double,
                                                      Exec);
template std::vector<ComplexType>
compute_brute_fold<ComplexType>(std::span<const float>,
                                std::span<const float>,
                                std::span<const double>,
                                SizeType,
                                SizeType,
                                double,
                                double,
                                Exec);

} // namespace loki::algorithms