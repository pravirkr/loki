#include "lib/search/harvested.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"

#include "temp_dir.hpp"

namespace {

using loki::SizeType;

struct Rec {
    double a;
    double f;
    float snr;
    std::uint64_t level;
};

/// A minimal accel EP result: param_names, tobs, one chunk with limits.
HighFive::File make_result(const std::filesystem::path& path) {
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    file.createAttribute("param_names",
                         std::vector<std::string>{"accel", "freq"});
    file.createAttribute("tobs", 1000.0);
    auto chunk = file.createGroup("chunks/chunk_0000");
    chunk.createDataSet("param_limits", std::vector<std::vector<double>>{
                                            {-10.0, 10.0}, {90.0, 160.0}});
    chunk.createGroup("rfi_config").createAttribute("harvest_mask_ntiles", 4.0);
    chunk.createGroup("runs");
    return file;
}

void add_run(HighFive::File& file,
             const std::string& run,
             const std::vector<Rec>& recs) {
    auto g = file.createGroup("chunks/chunk_0000/runs/" + run + "/harvest");
    const SizeType n = recs.size();
    // Rows: accel, freq, then two extra rows, as the pruner writes them.
    std::vector<double> params(n * 4 * 2, 0.0);
    std::vector<float> scores;
    std::vector<std::uint64_t> levels;
    std::vector<std::uint64_t> seg_idx;
    std::vector<double> t_ref;
    for (SizeType i = 0; i < n; ++i) {
        params[(i * 8) + 0] = recs[i].a;
        params[(i * 8) + 1] = 1.0; // da
        params[(i * 8) + 2] = recs[i].f;
        params[(i * 8) + 3] = 1e-3; // df
        scores.push_back(recs[i].snr);
        levels.push_back(recs[i].level);
        seg_idx.push_back(recs[i].level);
        t_ref.push_back(100.0);
    }
    auto ds = g.createDataSet("param_sets", HighFive::DataSpace({n, 4, 2}),
                              HighFive::create_datatype<double>());
    if (n > 0) {
        ds.write_raw(params.data());
    }
    g.createDataSet("scores", scores);
    g.createDataSet("levels", levels);
    g.createDataSet("seg_idx", seg_idx);
    g.createDataSet("t_ref", t_ref);
}

} // namespace

TEST_CASE("write_harvested merges records of one source over runs",
          "[harvested]") {
    const loki::test::TempDir dir("loki-harvested");
    auto file = make_result(dir / "r.h5");
    add_run(file, "000_00",
            {{.a = 0.0, .f = 100.0, .snr = 30.0F, .level = 12},
             {.a = 0.0, .f = 150.0, .snr = 20.0F, .level = 20}});
    add_run(file, "004_01",
            {{.a = 0.0, .f = 100.0005, .snr = 40.0F, .level = 15}});
    add_run(file, "008_02", {}); // harvest on, nothing harvested

    REQUIRE(loki::search::write_harvested(file) == 2);
    const auto g = file.getGroup("harvested");
    CHECK(g.getAttribute("n_sources").read<std::uint64_t>() == 2);
    CHECK(g.getAttribute("n_records").read<std::uint64_t>() == 3);
    std::vector<float> snr;
    std::vector<std::uint64_t> n_records;
    std::vector<std::uint64_t> n_runs;
    std::vector<std::uint64_t> level;
    std::vector<std::string> run;
    g.getDataSet("snr").read(snr);
    g.getDataSet("n_records").read(n_records);
    g.getDataSet("n_runs").read(n_runs);
    g.getDataSet("level").read(level);
    g.getDataSet("run").read(run);
    REQUIRE(snr.size() == 2);
    // Strongest first: the 100 Hz source, seen in two runs.
    CHECK(snr == std::vector<float>{40.0F, 20.0F});
    CHECK(n_records == std::vector<std::uint64_t>{2, 1});
    CHECK(n_runs == std::vector<std::uint64_t>{2, 1});
    CHECK(level == std::vector<std::uint64_t>{15, 20});
    CHECK(run.front() == "chunks/chunk_0000/runs/004_01");
    std::vector<double> params(4);
    g.getDataSet("param_sets").read_raw(params.data());
    CHECK(params[1] == 100.0005);
    CHECK(params[3] == 150.0);
    std::vector<double> f_lo;
    std::vector<double> f_hi;
    g.getDataSet("f_lo").read(f_lo);
    g.getDataSet("f_hi").read(f_hi);
    CHECK(f_lo[0] < 100.0);
    CHECK(f_hi[0] > 100.0005);

    // Rewriting replaces the table.
    REQUIRE(loki::search::write_harvested(file) == 2);
}

TEST_CASE("write_harvested writes nothing without harvest records",
          "[harvested]") {
    const loki::test::TempDir dir("loki-harvested-none");
    auto file = make_result(dir / "r.h5");
    file.createGroup("chunks/chunk_0000/runs/000_00");
    CHECK(loki::search::write_harvested(file) == 0);
    CHECK_FALSE(file.exist("harvested"));
}
