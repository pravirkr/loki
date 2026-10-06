#pragma once

/**
 * @file fold_engine.hpp
 * @brief Backend engine interface for BruteFold. Internal.
 */

#include <memory>
#include <span>
#include <stdexcept>

#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

template <SupportedFoldType FoldType> class BruteFoldEngine {
protected:
    BruteFoldEngine() = default;

public:
    virtual ~BruteFoldEngine()                     = default;
    virtual SizeType get_fold_size() const         = 0;
    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<FoldType> fold) = 0;
    virtual void execute(DeviceSpan<const float> ts_e,
                         DeviceSpan<const float> ts_v,
                         DeviceSpan<FoldType> fold,
                         Stream stream)            = 0;
    virtual void execute_fused_freq(
        std::span<const float> /*ts_e*/,
        std::span<const float> /*ts_v*/,
        std::span<float> /*fold_out*/,
        std::span<const coord::FFACoordFreq* const> /*coords_levels*/,
        std::span<const SizeType> /*ncoords*/,
        SizeType /*nlevels*/) {
        throw std::logic_error(
            "execute_fused_freq only supported on CPU float engine");
    }
    virtual std::span<const coord::PhaseRun> runs() const { return {}; }
    virtual std::span<const SizeType> run_offsets() const { return {}; }

    BruteFoldEngine(const BruteFoldEngine&)            = delete;
    BruteFoldEngine& operator=(const BruteFoldEngine&) = delete;
    BruteFoldEngine(BruteFoldEngine&&)                 = delete;
    BruteFoldEngine& operator=(BruteFoldEngine&&)      = delete;
};

template <SupportedFoldType FoldType>
std::unique_ptr<BruteFoldEngine<FoldType>>
make_brute_fold_cpu(std::span<const double> freq_arr,
                    SizeType segment_len,
                    SizeType nbins,
                    SizeType nsamps,
                    double tsamp,
                    double t_ref,
                    int nthreads);

template <SupportedFoldType FoldType>
std::unique_ptr<BruteFoldEngine<FoldType>>
make_brute_fold_gpu(std::span<const double> freq_arr,
                    SizeType segment_len,
                    SizeType nbins,
                    SizeType nsamps,
                    double tsamp,
                    double t_ref,
                    int device_id);

} // namespace loki::algorithms::detail
