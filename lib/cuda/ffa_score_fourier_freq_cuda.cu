#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/ffa_score_fourier_impl.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

void ffa_freq_final_score_fourier_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    coord::FFACoordFreqDPtrs coords_final,
    coord::FFACoordFreqDPtrs coords_prev,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType ncoords_src,
    int depth,
    SizeType nbins,
    const FFAFinalScoreArgs& args,
    cudaStream_t stream) {
    (void)ncoords_prev;
    detail::launch_fourier(fold_in, detail::FreqParentsC{coords_final},
                           detail::FreqParentsC{coords_prev}, ncoords_cur,
                           ncoords_src, depth, nbins, args, stream);
}

} // namespace loki::core
