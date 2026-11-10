#include "lib/io/candidate_results.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <highfive/highfive.hpp>

#include "loki/io/observation.hpp"
#include "loki/search/candidates.hpp"

#include "lib/search/candidate_processing.hpp"
#include "lib/search/cands.hpp"

namespace {
// A directory of its own per test: ctest runs the cases in parallel.
struct Files {
    std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "loki-candidate-results-test" /
        std::to_string(std::mt19937_64{std::random_device{}()}());
    Files() { std::filesystem::create_directories(directory); }
    ~Files() { std::filesystem::remove_all(directory); }
};
void spool(const std::filesystem::path& path, std::size_t count) {
    loki::search::FFAResultWriter writer(path);
    loki::search::FFAResultMetadata meta;
    meta.param_names = {"accel", "freq"};
    meta.tobs        = 10;
    meta.tsamp       = 0.01;
    meta.nsamps      = 1000;
    writer.write_metadata(meta);
    std::vector<double> params;
    std::vector<float> snr;
    std::vector<std::uint16_t> widths(count, 1);
    std::vector<std::uint16_t> bins(count, 10);
    for (std::size_t i = 0; i < count; ++i) {
        params.push_back(0);
        params.push_back(10 + static_cast<double>(i % 4));
        snr.push_back(static_cast<float>(i + 6));
    }
    writer.write_results(params, snr, widths, bins, count, 2);
    writer.finalize();
}
template <typename T>
std::vector<T> column(const HighFive::File& file, const std::string& path) {
    std::vector<T> values;
    file.getDataSet(path).read(values);
    return values;
}
} // namespace

TEST_CASE("Candidate external sorting is independent of batch size",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 120);
    loki::io::ObservationMetadata obs;
    obs.dm              = 42;
    obs.start_mjd       = 60000;
    obs.barycentric     = false;
    obs.original_nsamps = 1234;
    loki::search::CandidateOptions small;
    small.memory_gb  = 0.0001;
    const auto first = files.directory / "first.h5";
    loki::io::detail::write_candidate_results(raw, first, obs, small);
    auto large        = small;
    large.memory_gb   = 0.01;
    const auto second = files.directory / "second.h5";
    loki::io::detail::write_candidate_results(raw, second, obs, large);
    const HighFive::File a(first.string(), HighFive::File::ReadOnly);
    const HighFive::File b(second.string(), HighFive::File::ReadOnly);
    REQUIRE(column<float>(a, "candidates/snr") ==
            column<float>(b, "candidates/snr"));
    REQUIRE(column<std::uint64_t>(a, "detections/cluster_id") ==
            column<std::uint64_t>(b, "detections/cluster_id"));
    REQUIRE(column<std::uint64_t>(a, "clusters/member_count") ==
            std::vector<std::uint64_t>(4, 30));
    REQUIRE(a.getDataSet("candidates/snr").getDimensions()[0] == 4);
    int complete = 0;
    a.getAttribute("complete").read(complete);
    REQUIRE(complete == 1);
    double epoch = 0;
    a.getGroup("observation").getAttribute("reference_mjd").read(epoch);
    REQUIRE(epoch > 60000);
    REQUIRE(a.getGroup("search").hasAttribute("config_toml"));
    double phase_radius = 0;
    a.getGroup("processing").getAttribute("phase_radius").read(phase_radius);
    REQUIRE(phase_radius == small.phase_radius);
}

TEST_CASE("Candidate retention selection and failed publication",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 120);
    loki::search::CandidateOptions options;
    options.retain_detections = false;
    options.max_candidates    = 2;
    const auto output         = files.directory / "out.h5";
    loki::io::detail::write_candidate_results(raw, output, {}, options);
    {
        const HighFive::File file(output.string(), HighFive::File::ReadOnly);
        REQUIRE_FALSE(file.exist("detections"));
        REQUIRE(file.getDataSet("clusters/snr").getDimensions()[0] == 4);
        REQUIRE(file.getDataSet("candidates/snr").getDimensions()[0] == 2);
        REQUIRE_FALSE(file.getGroup("observation").hasAttribute("dm"));
    }
    options.cluster        = false;
    options.flag_harmonics = true;
    options.memory_gb      = 0.0001;
    REQUIRE_THROWS(
        loki::io::detail::write_candidate_results(raw, output, {}, options));
    const HighFive::File preserved(output.string(), HighFive::File::ReadOnly);
    REQUIRE(preserved.getDataSet("candidates/snr").getDimensions()[0] == 2);
    // The partial file lives in a scratch directory beside the output; a failed
    // publication removes it.
    for (const auto& entry :
         std::filesystem::directory_iterator(files.directory)) {
        CAPTURE(entry.path().string());
        REQUIRE_FALSE(
            entry.path().filename().string().starts_with("loki-candidates-"));
    }
}

TEST_CASE("Abandoned raw writers do not publish", "[candidates][hdf5]") {
    Files files;
    const auto path = files.directory / "aborted.h5";
    {
        loki::search::FFAResultWriter writer(path);
        loki::search::FFAResultMetadata meta;
        meta.param_names = {"freq"};
        writer.write_metadata(meta);
    }
    REQUIRE_FALSE(std::filesystem::exists(path));
    REQUIRE_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST_CASE("Incomplete search spools are not published", "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 4);
    {
        HighFive::File file(raw.string(), HighFive::File::ReadWrite);
        file.getAttribute("complete").write(0);
    }
    const auto out = files.directory / "out.h5";
    REQUIRE_THROWS_WITH(
        loki::io::detail::write_candidate_results(raw, out, {}, {}),
        "search spool is incomplete");
    REQUIRE_FALSE(std::filesystem::exists(out));
}

TEST_CASE("Spools outside the kinematic Taylor convention are rejected",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 4);
    {
        HighFive::File file(raw.string(), HighFive::File::ReadWrite);
        file.getAttribute("param_names")
            .write(std::vector<std::string>{"freq", "accel"});
    }
    const auto out = files.directory / "out.h5";
    REQUIRE_THROWS_WITH(
        loki::io::detail::write_candidate_results(raw, out, {}, {}),
        "unsupported native parameter convention");
}

TEST_CASE("Empty common candidate files have valid zero-row tables",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 0);
    const auto out = files.directory / "out.h5";
    loki::io::detail::write_candidate_results(raw, out, {}, {});
    const HighFive::File file(out.string(), HighFive::File::ReadOnly);
    REQUIRE(file.getDataSet("candidates/snr").getDimensions()[0] == 0);
    REQUIRE(file.getDataSet("detections/snr").getDimensions()[0] == 0);
}

TEST_CASE("Candidate products preserve optional EP-style native uncertainties",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 1);
    {
        HighFive::File file(raw.string(), HighFive::File::ReadWrite);
        file.createDataSet("param_errors",
                           std::vector<std::vector<double>>{{0.5, 0.001}});
        file.createDataSet("has_parameter_errors",
                           std::vector<std::uint8_t>{1});
    }
    const auto out = files.directory / "out.h5";
    loki::io::detail::write_candidate_results(raw, out, {}, {});
    const HighFive::File file(out.string(), HighFive::File::ReadOnly);
    REQUIRE(column<std::uint8_t>(file, "candidates/has_parameter_errors") ==
            std::vector<std::uint8_t>{1});
    std::vector<std::vector<double>> errors;
    file.getDataSet("candidates/param_errors").read(errors);
    REQUIRE(errors == std::vector<std::vector<double>>{{0.5, 0.001}});
}

TEST_CASE("Unclustered non-harmonic results stream beyond the index budget",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 120);
    loki::search::CandidateOptions options;
    options.cluster        = false;
    options.memory_gb      = 0.0001;
    options.max_candidates = 3;
    const auto out         = files.directory / "out.h5";
    loki::io::detail::write_candidate_results(raw, out, {}, options);
    const HighFive::File file(out.string(), HighFive::File::ReadOnly);
    REQUIRE(file.getDataSet("clusters/snr").getDimensions()[0] == 120);
    REQUIRE(file.getDataSet("candidates/snr").getDimensions()[0] == 3);
    REQUIRE(column<std::uint64_t>(file, "clusters/member_count") ==
            std::vector<std::uint64_t>(120, 1));
}

TEST_CASE(
    "Disk candidate publication remaps provisional IDs after a late bridge",
    "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "bridge.h5";
    {
        loki::search::FFAResultWriter writer(raw);
        loki::search::FFAResultMetadata meta;
        meta.param_names = {"freq"};
        meta.tobs        = 1;
        writer.write_metadata(meta);
        const std::vector<double> params{10, 10.18, 10.09};
        const std::vector<float> snr{20, 19, 10};
        const std::vector<std::uint16_t> widths(3, 1), bins(3, 10);
        writer.write_results(params, snr, widths, bins, 3, 1);
        writer.finalize();
    }
    loki::search::CandidateOptions options;
    options.memory_gb = .0001; // one record per sort and serialization batch
    for (const bool retain : {true, false}) {
        options.retain_detections = retain;
        const auto out =
            files.directory / (retain ? "bridge-out.h5" : "bridge-no-raw.h5");
        loki::io::detail::write_candidate_results(raw, out, {}, options);
        const HighFive::File file(out.string(), HighFive::File::ReadOnly);
        REQUIRE(column<float>(file, "candidates/snr") ==
                std::vector<float>{20});
        REQUIRE(column<std::uint64_t>(file, "clusters/member_count") ==
                std::vector<std::uint64_t>{3});
        REQUIRE(column<double>(file, "clusters/phase_extent")[0] > .17);
        std::string grouping;
        file.getGroup("processing").getAttribute("grouping").read(grouping);
        REQUIRE(grouping == "transitive");
        if (retain) {
            REQUIRE(column<std::uint64_t>(file, "detections/cluster_id") ==
                    std::vector<std::uint64_t>{0, 0, 0});
            REQUIRE(column<float>(file, "detections/snr") ==
                    std::vector<float>{20, 19, 10});
        } else {
            REQUIRE_FALSE(file.exist("detections"));
        }
    }
}

TEST_CASE("A cell cap keeps the strongest trials of each frequency cell",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    // Frequencies 10, 11, 12 and 13 Hz in turn, S/N 6 to 125.
    spool(raw, 120);
    loki::search::CandidateOptions options;
    options.max_per_cell   = 2;
    options.cell_log_width = 1e-3; // one frequency per cell
    std::vector<float> first_snr;
    for (const double memory_gb : {0.0001, 0.01}) {
        CAPTURE(memory_gb);
        options.memory_gb = memory_gb;
        const auto out    = files.directory / "out.h5";
        loki::io::detail::write_candidate_results(raw, out, {}, options);
        const HighFive::File file(out.string(), HighFive::File::ReadOnly);
        std::uint64_t kept    = 0;
        std::uint64_t dropped = 0;
        file.getAttribute("n_detections").read(kept);
        file.getAttribute("n_cell_dropped").read(dropped);
        REQUIRE(kept == 8);
        REQUIRE(dropped == 112);
        const auto snr = column<float>(file, "detections/snr");
        REQUIRE(snr ==
                std::vector<float>{125, 124, 123, 122, 121, 120, 119, 118});
        REQUIRE(column<std::uint64_t>(file, "clusters/member_count") ==
                std::vector<std::uint64_t>(4, 2));
        std::uint64_t max_per_cell = 0;
        double width               = 0;
        file.getGroup("processing")
            .getAttribute("max_per_cell")
            .read(max_per_cell);
        file.getGroup("processing").getAttribute("cell_log_width").read(width);
        REQUIRE(max_per_cell == 2);
        REQUIRE(width == options.cell_log_width);
        std::filesystem::remove(out);
    }
    // One cell holds every frequency: only the three strongest trials stay.
    options.max_per_cell   = 3;
    options.cell_log_width = 1;
    options.memory_gb      = 0.01;
    const auto wide        = files.directory / "wide.h5";
    loki::io::detail::write_candidate_results(raw, wide, {}, options);
    const HighFive::File file(wide.string(), HighFive::File::ReadOnly);
    REQUIRE(column<float>(file, "detections/snr") ==
            std::vector<float>{125, 124, 123});
}

TEST_CASE("A cell cap keeps the same trials in any arrival order",
          "[candidates]") {
    loki::search::CandidateOptions options;
    options.max_per_cell   = 3;
    options.cell_log_width = 0.05;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> freq(10, 12);
    std::uniform_int_distribution<int> snr(7, 12); // ties are common
    std::vector<loki::search::Candidate> trials(500);
    for (std::size_t i = 0; i < trials.size(); ++i) {
        auto& c          = trials[i];
        c.derivatives[0] = freq(rng);
        c.parameters[0]  = c.derivatives[0];
        c.snr            = static_cast<float>(snr(rng));
        c.width          = 1;
        c.nbins          = 10;
        c.id             = i;
    }
    const auto kept = [&](const std::vector<loki::search::Candidate>& in) {
        loki::search::detail::CellCap cap(options);
        for (const auto& c : in) {
            cap.add(c);
        }
        REQUIRE(cap.dropped() > 0);
        auto out = cap.take();
        std::vector<std::uint64_t> ids;
        for (const auto& c : out) {
            ids.push_back(c.id);
        }
        std::ranges::sort(ids);
        return ids;
    };
    const auto reference = kept(trials);
    // ln(f) from 10 to 12 Hz spans four cells of width 0.05.
    REQUIRE(reference.size() == 12);
    for (int shuffle = 0; shuffle < 5; ++shuffle) {
        std::ranges::shuffle(trials, rng);
        REQUIRE(kept(trials) == reference);
    }
}

TEST_CASE("A cell cap is bounded by the candidate memory budget",
          "[candidates][hdf5]") {
    Files files;
    const auto raw = files.directory / "raw.h5";
    spool(raw, 120);
    loki::search::CandidateOptions options;
    options.max_per_cell   = 120;
    options.cell_log_width = 1;
    options.memory_gb      = 0.0001;
    REQUIRE_THROWS_WITH(
        loki::io::detail::write_candidate_results(raw, files.directory / "o.h5",
                                                  {}, options),
        Catch::Matchers::ContainsSubstring("cell index exceeds"));
    options.cell_log_width = 0;
    REQUIRE_THROWS_AS(options.validate(), std::invalid_argument);
}
