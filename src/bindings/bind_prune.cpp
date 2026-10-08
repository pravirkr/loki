#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/loki.hpp"

#include "bindings/bind.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using algorithms::EPChunkConfig;
using algorithms::EPChunkStats;
using algorithms::EPRegionStats;
using pipelines::EPFreqSweep;
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_prune(py::module_& m) {
    auto m_prune = m.def_submodule("prune", "Pruning submodule");

    // EP submodule
    bind_prune_rfi(m_prune);
    bind_ep_multi_pass<float>(m_prune, "EPMultiPassTime");
    bind_ep_multi_pass<ComplexType>(m_prune, "EPMultiPassFourier");

    py::class_<EPChunkConfig>(m_prune, "EPChunkConfig")
        .def_readonly("cfg", &EPChunkConfig::cfg)
        .def_readonly("threshold_scheme", &EPChunkConfig::threshold_scheme)
        .def_readonly("branching_pattern", &EPChunkConfig::branching_pattern)
        .def_readonly("max_sugg", &EPChunkConfig::max_sugg)
        .def_readonly("branch_max", &EPChunkConfig::branch_max)
        .def_readonly("nominal_f_start", &EPChunkConfig::nominal_f_start)
        .def_readonly("nominal_f_end", &EPChunkConfig::nominal_f_end)
        .def_readonly("actual_f_start", &EPChunkConfig::actual_f_start)
        .def_readonly("actual_f_end", &EPChunkConfig::actual_f_end)
        .def_readonly("peak_complexity", &EPChunkConfig::peak_complexity)
        .def_readonly("chunk_memory_gb", &EPChunkConfig::chunk_memory_gb,
                      "Memory (GB) of this chunk run alone. The sweep peak, "
                      "with the shared FFA buffers grown across chunks, is "
                      "EPRegionStats.max_memory_gb.")
        .def_readonly("nsegments", &EPChunkConfig::nsegments)
        .def_readonly("ncoords", &EPChunkConfig::ncoords)
        .def_readonly("buffer_size", &EPChunkConfig::buffer_size)
        .def_readonly("coord_size", &EPChunkConfig::coord_size)
        .def_readonly("fold_size", &EPChunkConfig::fold_size);

    py::class_<EPChunkStats>(m_prune, "EPChunkStats")
        .def_readonly("chunk_id", &EPChunkStats::chunk_id)
        .def_readonly("nominal_f_start", &EPChunkStats::nominal_f_start)
        .def_readonly("nominal_f_end", &EPChunkStats::nominal_f_end)
        .def_readonly("actual_f_start", &EPChunkStats::actual_f_start)
        .def_readonly("actual_f_end", &EPChunkStats::actual_f_end)
        .def_readonly("nominal_width", &EPChunkStats::nominal_width)
        .def_readonly("actual_width", &EPChunkStats::actual_width)
        .def_readonly("nbins", &EPChunkStats::nbins)
        .def_readonly("eta", &EPChunkStats::eta)
        .def_readonly("ncoords", &EPChunkStats::ncoords)
        .def_readonly("max_sugg", &EPChunkStats::max_sugg)
        .def_readonly("branch_max", &EPChunkStats::branch_max)
        .def_readonly("peak_complexity", &EPChunkStats::peak_complexity)
        .def_readonly("memory_gb", &EPChunkStats::memory_gb,
                      "Memory (GB) of this chunk run alone: its workers, its "
                      "own shared FFA buffers and the input series.")
        .def_readonly("overlap_fraction", &EPChunkStats::overlap_fraction);

    py::class_<EPRegionStats>(m_prune, "EPRegionStats")
        .def_property_readonly("max_sugg", &EPRegionStats::get_max_sugg)
        .def_property_readonly("max_ncoords", &EPRegionStats::get_max_ncoords)
        .def_property_readonly("max_branch_max",
                               &EPRegionStats::get_max_branch_max)
        .def_property_readonly("max_memory_gb",
                               &EPRegionStats::get_max_memory_gb,
                               "Peak memory (GB) of the whole sweep, with the "
                               "shared FFA buffers grown to their largest "
                               "size. This is the value checked against the "
                               "memory limit.")
        .def_property_readonly("max_buffer_size",
                               &EPRegionStats::get_max_buffer_size)
        .def_property_readonly("max_coord_size",
                               &EPRegionStats::get_max_coord_size)
        .def_property_readonly("max_fold_size",
                               &EPRegionStats::get_max_fold_size)
        .def_property_readonly("nchunks", &EPRegionStats::get_nchunks)
        .def_property_readonly("chunk_stats", &EPRegionStats::get_chunk_stats);

    bind_ep_region_planner<float>(m_prune, "EPRegionPlannerTime");
    bind_ep_region_planner<ComplexType>(m_prune, "EPRegionPlannerFourier");

    py::class_<EPFreqSweep>(m_prune, "EPFreqSweep")
        .def(py::init(
                 [](const PulsarSearchConfig& cfg, bool show_progress,
                    float min_pd, std::string_view poly_basis, float ref_ducy,
                    const algorithms::PruneRFIConfig& rfi_config,
                    const std::optional<std::filesystem::path>& plan_cache_file,
                    std::optional<SizeType> n_runs,
                    const std::optional<std::vector<SizeType>>& ref_segs,
                    std::string_view backend, int device) {
                     return std::make_unique<EPFreqSweep>(
                         cfg, show_progress, min_pd, poly_basis, ref_ducy,
                         rfi_config, plan_cache_file, n_runs, ref_segs,
                         make_exec(backend, device));
                 }),
             py::arg("cfg"), py::arg("show_progress") = true,
             py::arg("min_pd") = 0.1F, py::arg("poly_basis") = "taylor",
             py::arg("ref_ducy")        = 0.1F,
             py::arg("rfi_config")      = algorithms::PruneRFIConfig(),
             py::arg("plan_cache_file") = std::nullopt,
             py::arg("n_runs")          = std::nullopt,
             py::arg("ref_segs")        = std::nullopt, py::kw_only(),
             py::arg("backend") = "cpu", py::arg("device") = 0)
        .def(
            "execute",
            [](EPFreqSweep& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir") = "./",
            py::arg("file_prefix") = "test");
}

} // namespace loki
