#include "loki/utils/workspace.hpp"

#include "loki/exceptions.hpp"

namespace loki::memory {

// --- FFAWorkspace implementation ---
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan) {
    const auto buffer_size  = ffa_plan.get_buffer_size();
    const auto coord_size   = ffa_plan.get_coord_size();
    const bool is_freq_only = ffa_plan.get_n_params() == 1;
    fold_internal.resize(buffer_size, FoldType{});
    if (is_freq_only) {
        coords_freq.resize(coord_size);
    } else {
        coords.resize(coord_size);
    }
}

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(SizeType buffer_size,
                                     SizeType coord_size,
                                     SizeType n_params) {
    const bool is_freq_only = n_params == 1;
    fold_internal.resize(buffer_size, FoldType{});
    if (is_freq_only) {
        coords_freq.resize(coord_size);
    } else {
        coords.resize(coord_size);
    }
}

template <SupportedFoldType FoldType>
void FFAWorkspace<FoldType>::validate(
    const plans::FFAPlan<FoldType>& ffa_plan) const {
    const auto buffer_size  = ffa_plan.get_buffer_size();
    const bool is_freq_only = ffa_plan.get_n_params() == 1;
    error_check::check_greater_equal(
        fold_internal.size(), buffer_size,
        "FFAWorkspace: fold_internal buffer too small");
    if (is_freq_only) {
        error_check::check_greater_equal(
            coords_freq.size(), ffa_plan.get_coord_size(),
            "FFAWorkspace: coordinates not allocated for enough levels");
    } else {
        error_check::check_greater_equal(
            coords.size(), ffa_plan.get_coord_size(),
            "FFAWorkspace: coordinates not allocated for enough levels");
    }
}

// --- BranchingWorkspace implementation ---
BranchingWorkspace::BranchingWorkspace(SizeType batch_size,
                                       SizeType branch_max,
                                       SizeType n_params)
    : scratch_params(batch_size * n_params * branch_max),
      scratch_dparams(batch_size * n_params),
      scratch_counts(batch_size * n_params),
      scratch_shifts(batch_size * n_params) {}

[[nodiscard]] float BranchingWorkspace::get_memory_usage_gib() const noexcept {
    const auto total_memory = (scratch_params.size() * sizeof(double)) +
                              (scratch_dparams.size() * sizeof(double)) +
                              (scratch_counts.size() * sizeof(SizeType)) +
                              (scratch_shifts.size() * sizeof(double));
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

void BranchingWorkspace::validate(SizeType batch_size,
                                  SizeType branch_max,
                                  SizeType nparams) const {
    error_check::check_equal(
        scratch_params.size(), batch_size * nparams * branch_max,
        "BranchingWorkspace: scratch_params size is too small");
    error_check::check_equal(
        scratch_dparams.size(), batch_size * nparams,
        "BranchingWorkspace: scratch_dparams size is too small");
    error_check::check_equal(
        scratch_counts.size(), batch_size * nparams,
        "BranchingWorkspace: scratch_counts size is too small");
    error_check::check_equal(
        scratch_shifts.size(), batch_size * nparams,
        "BranchingWorkspace: scratch_shifts size is too small");
}

// --- PruneWorkspace implementation ---
template <SupportedFoldType FoldType>
PruneWorkspace<FoldType>::PruneWorkspace(SizeType batch_size,
                                         SizeType branch_max,
                                         SizeType nparams,
                                         SizeType nbins,
                                         SizeType nsegments)
    : batch_size(batch_size),
      branch_max(branch_max),
      nparams(nparams),
      nbins(nbins),
      nsegments(nsegments),
      max_branched_leaves(batch_size * branch_max),
      max_branched_param_idx(
          std::max(max_branched_leaves, nsegments * batch_size)),
      leaves_stride((nparams + 2) * kLeavesParamStride),
      folds_stride(2 * nbins),
      branched_leaves(max_branched_leaves * leaves_stride),
      branched_folds(max_branched_leaves * folds_stride),
      branched_scores(max_branched_leaves),
      branched_indices(max_branched_leaves),
      branched_param_idx(max_branched_param_idx),
      branched_phase_shift(max_branched_param_idx) {}

template <SupportedFoldType FoldType>
float PruneWorkspace<FoldType>::get_memory_usage_gib() const noexcept {
    const auto total_memory = (branched_leaves.size() * sizeof(double)) +
                              (branched_folds.size() * sizeof(FoldType)) +
                              (branched_scores.size() * sizeof(float)) +
                              (branched_indices.size() * sizeof(SizeType)) +
                              (branched_param_idx.size() * sizeof(SizeType)) +
                              (branched_phase_shift.size() * sizeof(float));
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

template <SupportedFoldType FoldType>
void PruneWorkspace<FoldType>::validate(SizeType batch_size,
                                        SizeType branch_max,
                                        SizeType nsegments) const {
    const auto max_branched_param_idx =
        std::max(batch_size * branch_max, nsegments * batch_size);
    error_check::check_equal(
        branched_leaves.size(), batch_size * branch_max * leaves_stride,
        "PruneWorkspace: branched_leaves size is too small");
    error_check::check_equal(
        branched_folds.size(), batch_size * branch_max * folds_stride,
        "PruneWorkspace: branched_folds size is too small");
    error_check::check_equal(
        branched_scores.size(), batch_size * branch_max,
        "PruneWorkspace: branched_scores size is too small");
    error_check::check_equal(
        branched_indices.size(), batch_size * branch_max,
        "PruneWorkspace: branched_indices size is too small");
    error_check::check_equal(
        branched_param_idx.size(), max_branched_param_idx,
        "PruneWorkspace: branched_param_idx size is too small");
    error_check::check_equal(
        branched_phase_shift.size(), max_branched_param_idx,
        "PruneWorkspace: branched_phase_shift size is too small");
}

// --- EPWorkspace implementation ---
template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::EPWorkspace(SizeType batch_size,
                                   SizeType branch_max,
                                   SizeType max_sugg,
                                   SizeType ncoords_ffa,
                                   SizeType nparams,
                                   SizeType nbins,
                                   SizeType nsegments)
    : world_tree(max_sugg, nparams, nbins, batch_size * branch_max),
      prune(batch_size, branch_max, nparams, nbins, nsegments),
      branch(batch_size, branch_max, nparams) {
    seed_leaves.resize(ncoords_ffa * world_tree.get_leaves_stride());
    seed_scores.resize(ncoords_ffa);
}

template <SupportedFoldType FoldType>
float EPWorkspace<FoldType>::get_memory_usage_gib() const noexcept {
    const auto base_gb      = world_tree.get_memory_usage_gib() +
                              prune.get_memory_usage_gib() +
                              branch.get_memory_usage_gib();
    const auto extra_memory = (seed_leaves.size() * sizeof(double)) +
                              (seed_scores.size() * sizeof(float));
    const auto extra_gb =
        static_cast<float>(extra_memory) / static_cast<float>(1ULL << 30U);
    return base_gb + extra_gb;
}

template <SupportedFoldType FoldType>
void EPWorkspace<FoldType>::validate(SizeType batch_size,
                                     SizeType branch_max,
                                     SizeType max_sugg,
                                     SizeType ncoords_ffa,
                                     SizeType nparams,
                                     SizeType nbins,
                                     SizeType nsegments) const {
    const auto leaves_stride = (nparams + 2) * 2;
    error_check::check_greater_equal(
        seed_scores.size(), ncoords_ffa,
        "EPWorkspace: seed_scores size is too small");
    error_check::check_equal(seed_leaves.size(), ncoords_ffa * leaves_stride,
                             "EPWorkspace: seed_leaves size is too small");
    world_tree.validate(max_sugg, nparams, nbins, batch_size * branch_max);
    prune.validate(batch_size, branch_max, nsegments);
    branch.validate(batch_size, branch_max, nparams);
}

// --- Explicit template instantiations ---
template struct FFAWorkspace<float>;
template struct FFAWorkspace<ComplexType>;
template struct PruneWorkspace<float>;
template struct PruneWorkspace<ComplexType>;
template struct EPWorkspace<float>;
template struct EPWorkspace<ComplexType>;

} // namespace loki::memory