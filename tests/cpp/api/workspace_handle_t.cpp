#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/prune.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

using Catch::Approx;
using loki::Backend;
using loki::ComplexType;
using loki::Device;
using loki::DeviceSpan;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::EPMultiPass;
using loki::algorithms::FFA;
using loki::algorithms::PruneRFIConfig;
using loki::math::FFTManager;
using loki::memory::EPWorkspace;
using loki::memory::FFAWorkspace;
using loki::search::PulsarSearchConfig;

namespace {

constexpr SizeType kNsamps = 1U << 14U;
constexpr double kTsamp    = 1.0e-3;

PulsarSearchConfig make_cfg(bool use_fourier) {
    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 10.0}};
    return {kNsamps,
            kTsamp,
            /*nbins=*/32,
            /*eta=*/0.5,
            limits,
            /*ducy_max=*/0.2,
            /*wtsp=*/1.5,
            use_fourier,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/32,
            /*bseg_brute=*/128};
}

std::vector<float> make_noise(SizeType nsamps, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(nsamps);
    for (auto& x : ts_e) {
        x = dist(rng);
    }
    return ts_e;
}

template <typename FoldType>
void check_shared_matches_owned(bool use_fourier, Exec exec) {
    const auto ts_e = make_noise(kNsamps, 7);
    const std::vector<float> ts_v(kNsamps, 1.0F);
    const auto cfg = make_cfg(use_fourier);

    FFA<FoldType> owned(cfg, /*show_progress=*/false, exec);
    std::vector<FoldType> fold_owned(owned.get_plan().get_buffer_size());
    owned.execute(ts_e, ts_v, fold_owned);

    FFAWorkspace<FoldType> workspace(owned.get_plan(), exec);
    FFTManager fft_manager(exec);
    // Two instances on the same buffers: the second must not see stale state.
    for (int run = 0; run < 2; ++run) {
        FFA<FoldType> shared(workspace, fft_manager, cfg,
                             /*show_progress=*/false, exec);
        std::vector<FoldType> fold_shared(shared.get_plan().get_buffer_size());
        shared.execute(ts_e, ts_v, fold_shared);
        const auto fold_size = shared.get_plan().get_fold_size();
        for (SizeType i = 0; i < fold_size; ++i) {
            if constexpr (std::is_same_v<FoldType, float>) {
                if (exec.backend != Backend::kCPU) {
                    // The CUDA time-domain brute fold accumulates with float
                    // atomics, so it is not bit-reproducible run to run.
                    REQUIRE(fold_shared[i] ==
                            Approx(fold_owned[i]).epsilon(1e-5).margin(1e-3));
                    continue;
                }
            }
            REQUIRE(fold_shared[i] == fold_owned[i]);
        }
    }
}

// Small two-parameter search: one EP run, a handful of segments.
PulsarSearchConfig make_ep_cfg(bool use_fourier, int nthreads) {
    constexpr SizeType kEPNsamps = 1U << 16U;
    constexpr double kEPTsamp    = 64e-6;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 145.0},
    };
    return {kEPNsamps,
            kEPTsamp,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            use_fourier,
            nthreads,
            /*max_process_memory_gb=*/8.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/kEPNsamps / 16,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

constexpr SizeType kEPMaxSugg   = 1U << 14U;
constexpr SizeType kEPBatchSize = 256;

/// An EPWorkspace sized the way EPMultiPass sizes its own.
template <typename FoldType>
EPWorkspace<FoldType> make_ep_workspace(const PulsarSearchConfig& cfg,
                                        Exec exec) {
    const loki::plans::FFAPlan<FoldType> plan(cfg);
    const auto pattern = plan.get_branching_pattern("taylor");
    const auto branch_max =
        std::max(static_cast<SizeType>(
                     std::ceil(*std::ranges::max_element(pattern) * 2)),
                 SizeType{32});
    const SizeType nbins = std::is_same_v<FoldType, ComplexType>
                               ? cfg.get_nbins_f()
                               : cfg.get_nbins();
    return {kEPBatchSize,
            branch_max,
            kEPMaxSugg,
            plan.get_ncoords().back(),
            cfg.get_nparams(),
            nbins,
            plan.get_nsegments().back(),
            exec};
}

struct EPInputs {
    std::vector<float> thresholds;
    std::vector<SizeType> ref_segs;
    SizeType nsegments;
};

EPInputs make_ep_inputs(const PulsarSearchConfig& cfg) {
    const auto nsegments =
        loki::plans::FFAPlan<float>(cfg).get_nsegments().back();
    return {
        std::vector<float>(nsegments - 1, 1.5F), {nsegments / 2}, nsegments};
}

struct Dataset {
    std::vector<char> bytes;
    bool is_float32;
};

bool is_float32(const HighFive::DataType& dtype) {
    return dtype.getClass() == HighFive::DataTypeClass::Float &&
           dtype.getSize() == sizeof(float);
}

/// Every dataset of an EP result file, except the wall-clock timers.
std::map<std::string, Dataset>
read_ep_datasets(const std::filesystem::path& path) {
    std::map<std::string, Dataset> out;
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    auto walk = [&](auto&& self, const HighFive::Group& group,
                    const std::string& prefix) -> void {
        for (const auto& name : group.listObjectNames()) {
            const auto full = prefix + "/" + name;
            if (group.getObjectType(name) == HighFive::ObjectType::Group) {
                self(self, group.getGroup(name), full);
                continue;
            }
            if (group.getObjectType(name) != HighFive::ObjectType::Dataset ||
                name == "timer_stats") {
                continue;
            }
            const auto ds    = group.getDataSet(name);
            auto dtype       = ds.getDataType();
            const auto nelem = ds.getElementCount();
            const auto esize = dtype.getSize();
            std::vector<char> bytes(nelem * esize);
            ds.read_raw(bytes.data(), dtype);
            if (dtype.getClass() != HighFive::DataTypeClass::Compound) {
                out.emplace(full, Dataset{std::move(bytes), is_float32(dtype)});
                continue;
            }
            // Compare compound members one by one: padding bytes between
            // them are uninitialised in the writer and differ run to run.
            const HighFive::CompoundType ctype(std::move(dtype));
            for (const auto& member : ctype.getMembers()) {
                const auto msize = member.base_type.getSize();
                std::vector<char> field(nelem * msize);
                for (std::size_t i = 0; i < nelem; ++i) {
                    std::copy_n(bytes.data() + (i * esize) + member.offset,
                                msize, field.data() + (i * msize));
                }
                out.emplace(
                    full + "." + member.name,
                    Dataset{std::move(field), is_float32(member.base_type)});
            }
        }
    };
    walk(walk, file.getGroup("/"), "");
    return out;
}

template <typename FoldType>
std::filesystem::path run_ep(EPMultiPass<FoldType>& ep,
                             const PulsarSearchConfig& cfg,
                             const EPInputs& in,
                             const std::string& prefix) {
    const auto ts_e = make_noise(cfg.get_nsamps(), 99);
    const std::vector<float> ts_v(cfg.get_nsamps(), 1.0F);
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_workspace_handle_ep";
    std::filesystem::create_directories(outdir);
    ep.execute(ts_e, ts_v, outdir, prefix);
    return outdir / std::format("{}_pruning_nstages_{}_results.h5", prefix,
                                in.nsegments);
}

/// EPMultiPass on caller-owned workspaces writes the same results as the
/// owning constructor.
template <typename FoldType>
void check_ep_shared_matches_owned(bool use_fourier, Exec exec) {
    const int nthreads = exec.backend == Backend::kCPU ? 2 : 1;
    const auto cfg     = make_ep_cfg(use_fourier, nthreads);
    const auto in      = make_ep_inputs(cfg);

    EPMultiPass<FoldType> owned(cfg, in.thresholds, std::nullopt, in.ref_segs,
                                {}, kEPMaxSugg, kEPBatchSize, "taylor",
                                /*show_progress=*/false, {}, exec);
    // ctest runs test cases in parallel processes: keep file names distinct.
    const auto tag        = std::format("{}_{}", loki::to_string(exec.backend),
                                        use_fourier ? "fourier" : "time");
    const auto owned_path = run_ep(owned, cfg, in, tag + "_owned");

    std::vector<EPWorkspace<FoldType>> workspaces;
    for (int i = 0; i < nthreads; ++i) {
        workspaces.push_back(make_ep_workspace<FoldType>(cfg, exec));
    }
    const std::span<EPWorkspace<FoldType>> ws_view(workspaces);
    EPMultiPass<FoldType> shared(ws_view, cfg, in.thresholds, std::nullopt,
                                 in.ref_segs, {}, kEPMaxSugg, kEPBatchSize,
                                 "taylor", /*show_progress=*/false, {}, exec);
    const auto shared_path = run_ep(shared, cfg, in, tag + "_shared");

    const auto owned_ds  = read_ep_datasets(owned_path);
    const auto shared_ds = read_ep_datasets(shared_path);
    REQUIRE(!owned_ds.empty());
    REQUIRE(owned_ds.size() == shared_ds.size());
    // The CUDA time-domain fold accumulates with float atomics, so its
    // scores are reproducible only to rounding; everything else is exact.
    const bool exact = exec.backend == Backend::kCPU || use_fourier;
    for (const auto& [name, ds] : owned_ds) {
        CAPTURE(name);
        REQUIRE(shared_ds.contains(name));
        const auto& other = shared_ds.at(name);
        if (exact || !ds.is_float32) {
            const bool same_bytes = other.bytes == ds.bytes;
            REQUIRE(same_bytes);
            continue;
        }
        REQUIRE(other.bytes.size() == ds.bytes.size());
        std::vector<float> a(ds.bytes.size() / sizeof(float));
        std::vector<float> b(a.size());
        std::memcpy(a.data(), ds.bytes.data(), ds.bytes.size());
        std::memcpy(b.data(), other.bytes.data(), other.bytes.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            REQUIRE(b[i] == Approx(a[i]).epsilon(1e-4));
        }
    }
    std::filesystem::remove(owned_path);
    std::filesystem::remove(shared_path);
}

} // namespace

TEST_CASE("FFA on a shared workspace matches an owning FFA",
          "[workspace][ffa]") {
    SECTION("time domain") {
        check_shared_matches_owned<float>(false, Exec::cpu());
    }
    SECTION("Fourier domain") {
        check_shared_matches_owned<ComplexType>(true, Exec::cpu());
    }
}

TEST_CASE("Workspace and FFT handles report their backend",
          "[workspace][fft]") {
    const auto cfg = make_cfg(false);
    const FFA<float> ffa(cfg, /*show_progress=*/false);

    const FFAWorkspace<float> ws(ffa.get_plan(), Exec::cpu());
    REQUIRE_FALSE(ws.empty());
    REQUIRE(ws.exec().backend == Backend::kCPU);

    const EPWorkspace<float> ep_ws(/*batch_size=*/16, /*branch_max=*/4,
                                   /*max_sugg=*/64, /*ncoords_ffa=*/8,
                                   /*nparams=*/2, /*nbins=*/32,
                                   /*nsegments=*/4, Exec::cpu());
    REQUIRE(ep_ws.exec().backend == Backend::kCPU);
    REQUIRE(ep_ws.get_memory_usage_gib() > 0.0F);

    FFTManager fft(Exec::cpu());
    const std::vector<SizeType> n_reals = {32};
    REQUIRE_FALSE(fft.has_prepared(32));
    fft.prepare_plans(n_reals);
    REQUIRE(fft.has_prepared(32));
    REQUIRE(fft.n_cached_plans() > 0);
}

TEST_CASE("Empty handles are rejected", "[workspace][ffa]") {
    const auto cfg = make_cfg(false);
    FFAWorkspace<float> ws;
    FFTManager fft;
    REQUIRE(ws.empty());
    REQUIRE(fft.empty());
    REQUIRE_FALSE(fft.has_prepared(32));
    REQUIRE_THROWS_AS(FFA<float>(ws, fft, cfg, /*show_progress=*/false),
                      std::invalid_argument);
}

TEST_CASE("GPU handles are rejected by a CPU-only build", "[workspace][fft]") {
    if (loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend is built");
    }
    REQUIRE_THROWS_AS(FFTManager(Exec::cuda(0)), std::invalid_argument);
    REQUIRE_THROWS_AS(FFAWorkspace<float>(1024, 64, 4, 1, Exec::cuda(0)),
                      std::invalid_argument);
}

TEST_CASE("A workspace on another backend than the FFA is rejected",
          "[workspace][ffa]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto cfg = make_cfg(false);
    const FFA<float> owned(cfg, /*show_progress=*/false);
    FFAWorkspace<float> ws(owned.get_plan(), Exec::cpu());
    FFTManager fft(Exec::cpu());
    REQUIRE_THROWS_AS(FFA<float>(ws, fft, cfg, Exec::cuda(0)),
                      std::invalid_argument);
}

TEST_CASE("FFA on a shared GPU workspace matches an owning GPU FFA",
          "[workspace][ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    SECTION("time domain") {
        check_shared_matches_owned<float>(false, Exec::cuda(0));
    }
    SECTION("Fourier domain") {
        check_shared_matches_owned<ComplexType>(true, Exec::cuda(0));
    }
}

TEST_CASE("FFA device views on another device are rejected",
          "[workspace][ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto cfg = make_cfg(false);
    FFA<float> ffa(cfg, /*show_progress=*/false, Exec::cuda(0));
    // check_device runs before any memory is touched, so null views suffice.
    const Device other{.backend = Backend::kCUDA, .id = 1};
    const Device same{.backend = Backend::kCUDA, .id = 0};
    const DeviceSpan<const float> ts_ok(nullptr, 0, same);
    const DeviceSpan<const float> ts_bad(nullptr, 0, other);
    const DeviceSpan<float> fold_ok(nullptr, 0, same);
    const DeviceSpan<float> fold_bad(nullptr, 0, other);
    REQUIRE_THROWS_AS(ffa.execute(ts_bad, ts_ok, fold_ok),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ffa.execute(ts_ok, ts_bad, fold_ok),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ffa.execute(ts_ok, ts_ok, fold_bad),
                      std::invalid_argument);
    const DeviceSpan<float> fold_cpu(nullptr, 0,
                                     {.backend = Backend::kCPU, .id = 0});
    REQUIRE_THROWS_AS(ffa.execute(ts_ok, ts_ok, fold_cpu),
                      std::invalid_argument);
}

TEST_CASE("EPMultiPass on caller-owned workspaces matches an owning one",
          "[workspace][prune]") {
    SECTION("time domain") {
        check_ep_shared_matches_owned<float>(false, Exec::cpu());
    }
    SECTION("Fourier domain") {
        check_ep_shared_matches_owned<ComplexType>(true, Exec::cpu());
    }
}

TEST_CASE("EPMultiPass on a GPU workspace matches an owning GPU EPMultiPass",
          "[workspace][prune][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    SECTION("time domain") {
        check_ep_shared_matches_owned<float>(false, Exec::cuda(0));
    }
    SECTION("Fourier domain") {
        check_ep_shared_matches_owned<ComplexType>(true, Exec::cuda(0));
    }
}

TEST_CASE("EPMultiPass rejects unsupported GPU workspace setups",
          "[workspace][prune][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto cfg = make_ep_cfg(false, 1);
    const auto in  = make_ep_inputs(cfg);
    const auto gpu = Exec::cuda(0);

    SECTION("more than one GPU workspace") {
        std::vector<EPWorkspace<float>> workspaces;
        workspaces.push_back(make_ep_workspace<float>(cfg, gpu));
        workspaces.push_back(make_ep_workspace<float>(cfg, gpu));
        REQUIRE_THROWS_AS(
            EPMultiPass<float>(std::span(workspaces), cfg, in.thresholds,
                               std::nullopt, in.ref_segs, {}, kEPMaxSugg,
                               kEPBatchSize, "taylor", false, {}, gpu),
            std::invalid_argument);
    }
    SECTION("a CPU workspace") {
        std::vector<EPWorkspace<float>> workspaces;
        workspaces.push_back(make_ep_workspace<float>(cfg, Exec::cpu()));
        REQUIRE_THROWS_AS(
            EPMultiPass<float>(std::span(workspaces), cfg, in.thresholds,
                               std::nullopt, in.ref_segs, {}, kEPMaxSugg,
                               kEPBatchSize, "taylor", false, {}, gpu),
            std::invalid_argument);
    }
    SECTION("an active rfi_config") {
        PruneRFIConfig rfi;
        rfi.harvest_scheme = loki::algorithms::make_default_harvest_scheme(
            in.thresholds, /*min_level=*/10, /*offset=*/10.0F,
            /*min_snr=*/15.0F);
        REQUIRE(rfi.is_active());
        std::vector<EPWorkspace<float>> workspaces;
        workspaces.push_back(make_ep_workspace<float>(cfg, gpu));
        REQUIRE_THROWS_AS(
            EPMultiPass<float>(std::span(workspaces), cfg, in.thresholds,
                               std::nullopt, in.ref_segs, {}, kEPMaxSugg,
                               kEPBatchSize, "taylor", false, rfi, gpu),
            std::invalid_argument);
        REQUIRE_THROWS_AS(EPMultiPass<float>(cfg, in.thresholds, std::nullopt,
                                             in.ref_segs, {}, kEPMaxSugg,
                                             kEPBatchSize, "taylor", false, rfi,
                                             gpu),
                          std::invalid_argument);
    }
}
