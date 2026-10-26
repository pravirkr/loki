#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"

using Catch::Approx;
using loki::Backend;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::FFAFreqSweep;
using loki::search::PulsarSearchConfig;

namespace {

struct SweepResults {
    std::vector<float> snr;
    std::vector<double> f0; // one parameter: the frequency
    std::vector<std::uint16_t> width;
    std::vector<std::uint16_t> nbins;
};

SweepResults read_results(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    // Compare raw scoring trials; representative selection is tested
    // separately.
    const auto detections = file.getGroup("detections");
    SweepResults out;
    detections.getDataSet("snr").read(out.snr);
    detections.getDataSet("width").read(out.width);
    detections.getDataSet("nbins").read(out.nbins);
    std::vector<std::vector<double>> rows;
    detections.getDataSet("param_sets").read(rows);
    for (const auto& row : rows) {
        REQUIRE(row.size() == 1);
        out.f0.push_back(row[0]);
    }
    return out;
}

// Candidates are matched by chunk (nbins) and frequency. The two backends
// decode the same grid cell to frequencies that differ in the last bits, so
// the match is the nearest frequency within a relative 1e-9. The widths are
// compared as sets: the GPU's Fourier brute fold differs from the CPU's at
// the 1e-1 level, which can move a strong candidate to the neighbouring
// boxcar width, so a cell's S/N is its best over widths.
struct CellSnr {
    double f0;
    float snr;
};

// Per nbins, the cells sorted by frequency with their best S/N.
using CellTable = std::map<std::uint16_t, std::vector<CellSnr>>;

CellTable make_cell_table(const SweepResults& r) {
    CellTable table;
    for (SizeType i = 0; i < r.snr.size(); ++i) {
        table[r.nbins[i]].push_back({r.f0[i], r.snr[i]});
    }
    for (auto& [nbins, cells] : table) {
        std::ranges::sort(cells, {}, &CellSnr::f0);
        std::vector<CellSnr> merged;
        for (const auto& c : cells) {
            if (!merged.empty() &&
                std::abs(merged.back().f0 - c.f0) <= 1e-9 * c.f0) {
                merged.back().snr = std::max(merged.back().snr, c.snr);
            } else {
                merged.push_back(c);
            }
        }
        cells = std::move(merged);
    }
    return table;
}

// The cell of @p table at (nbins, f0), or nullptr.
const CellSnr*
find_cell(const CellTable& table, std::uint16_t nbins, double f0) {
    const auto it = table.find(nbins);
    if (it == table.end()) {
        return nullptr;
    }
    const auto& cells   = it->second;
    auto lo             = std::ranges::lower_bound(cells, f0, {}, &CellSnr::f0);
    const CellSnr* best = nullptr;
    for (auto cand : {lo, lo == cells.begin() ? cells.end() : lo - 1}) {
        if (cand != cells.end() && std::abs(cand->f0 - f0) <= 1e-9 * f0 &&
            (best == nullptr ||
             std::abs(cand->f0 - f0) < std::abs(best->f0 - f0))) {
            best = &*cand;
        }
    }
    return best;
}

} // namespace

TEST_CASE("FFAFreqSweep CUDA matches the CPU sweep across regions",
          "[ffa_sweep][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    // A wide band, so the planner cuts several chunks with different bin
    // counts, and with them different boxcar width sets. The CUDA sweep
    // scores and decodes each chunk with its own widths; the base config's
    // widths would be too narrow for the chunks with more bins. 200 Hz keeps
    // nbins_min samples per period at this sampling.
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 6.4e-5;
    const std::vector<ParamLimit> limits{ParamLimit{.min = 50.0, .max = 200.0}};

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (auto& sample : ts_e) {
        sample = noise(rng);
    }
    constexpr double kInjectedF0 = 123.0;
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double phase =
            std::fmod(static_cast<double>(i) * kTsamp * kInjectedF0, 1.0);
        if (phase < 0.05) {
            ts_e[i] += 8.0F;
        }
    }

    const auto outdir = std::filesystem::temp_directory_path();
    const auto run    = [&](Exec exec, const std::string& prefix) {
        const PulsarSearchConfig cfg(
            kNsamps, kTsamp, /*nbins=*/64, /*eta=*/1.0, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true,
            /*nthreads=*/1, /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0, /*nbins_max=*/256,
            /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/std::nullopt,
            /*bseg_ffa=*/std::nullopt, /*snr_min=*/3.0,
            /*max_passing_candidates=*/1U << 24U);
        FFAFreqSweep sweep(cfg, /*show_progress=*/false, exec);
        sweep.execute(ts_e, ts_v, outdir, prefix);
        const auto path = outdir / (prefix + "_ffa_results.h5");
        auto results    = read_results(path);
        std::filesystem::remove(path);
        return results;
    };

    const auto cpu = run(Exec::cpu(1), "loki_sweep_parity_cpu");
    const auto gpu = run(Exec::cuda(0), "loki_sweep_parity_gpu");
    REQUIRE_FALSE(cpu.snr.empty());
    REQUIRE_FALSE(gpu.snr.empty());

    // Several chunks, and the same (nbins, width) cells on both backends.
    std::set<std::uint16_t> cpu_nbins(cpu.nbins.begin(), cpu.nbins.end());
    REQUIRE(cpu_nbins.size() >= 2);
    std::set<std::pair<std::uint16_t, std::uint16_t>> cpu_cells;
    std::set<std::pair<std::uint16_t, std::uint16_t>> gpu_cells;
    for (SizeType i = 0; i < cpu.snr.size(); ++i) {
        cpu_cells.emplace(cpu.nbins[i], cpu.width[i]);
    }
    for (SizeType i = 0; i < gpu.snr.size(); ++i) {
        gpu_cells.emplace(gpu.nbins[i], gpu.width[i]);
    }
    CHECK(cpu_cells == gpu_cells);

    // The strongest CPU candidates are found on the GPU at the same
    // frequency cell, with the cell's best S/N over widths within 1%: the
    // GPU brute fold is not bit-identical to the CPU's (observed differences
    // are below 1e-3 relative on an A40).
    const auto cpu_cells_by_f0 = make_cell_table(cpu);
    const auto gpu_cells_by_f0 = make_cell_table(gpu);
    std::vector<SizeType> order(cpu.snr.size());
    std::iota(order.begin(), order.end(), SizeType{0});
    std::ranges::sort(
        order, [&](SizeType a, SizeType b) { return cpu.snr[a] > cpu.snr[b]; });
    const auto n_top = std::min<SizeType>(50, order.size());
    for (SizeType k = 0; k < n_top; ++k) {
        const auto i = order[k];
        INFO("CPU candidate f0=" << cpu.f0[i] << " width=" << cpu.width[i]
                                 << " nbins=" << cpu.nbins[i]
                                 << " snr=" << cpu.snr[i]);
        const auto* cpu_cell =
            find_cell(cpu_cells_by_f0, cpu.nbins[i], cpu.f0[i]);
        const auto* gpu_cell =
            find_cell(gpu_cells_by_f0, cpu.nbins[i], cpu.f0[i]);
        REQUIRE(cpu_cell != nullptr);
        REQUIRE(gpu_cell != nullptr);
        CHECK(gpu_cell->snr == Approx(cpu_cell->snr).epsilon(1.0e-2));
    }

    // Both find the injected pulsar.
    const auto cpu_best = order.front();
    REQUIRE(std::abs(cpu.f0[cpu_best] - kInjectedF0) < 1.0);
    const auto gpu_best = static_cast<SizeType>(
        std::distance(gpu.snr.begin(), std::ranges::max_element(gpu.snr)));
    REQUIRE(std::abs(gpu.f0[gpu_best] - kInjectedF0) < 1.0);
    CHECK(cpu.snr[cpu_best] > 8.0F);
}
