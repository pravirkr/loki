#include "lib/search/harvested.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <highfive/highfive.hpp>

#include "loki/algorithms/prune_rfi.hpp"

#include "lib/detail/utils.hpp"

namespace loki::search {

namespace {

struct Record {
    std::vector<double> values;
    std::vector<double> errors;
    float snr{};
    std::uint64_t level{};
    std::uint64_t seg_idx{};
    double t_ref{};
    double f_lo{};
    double f_hi{};
    std::string run;
};

double observation_length(const HighFive::File& file) {
    double tobs = 0.0;
    if (file.hasAttribute("tobs")) {
        file.getAttribute("tobs").read(tobs);
    } else if (file.hasAttribute("nsamps") && file.hasAttribute("tsamp")) {
        std::uint64_t nsamps = 0;
        double tsamp         = 0.0;
        file.getAttribute("nsamps").read(nsamps);
        file.getAttribute("tsamp").read(tsamp);
        tobs = static_cast<double>(nsamps) * tsamp;
    }
    return utils::is_finite(tobs) && tobs > 0.0 ? tobs : 0.0;
}

/// Largest |acceleration| of a chunk (its param_limits row n_params - 2).
double chunk_accel_limit(const HighFive::Group& chunk, SizeType n_params) {
    if (!chunk.exist("param_limits") || n_params < 2) {
        return std::numeric_limits<double>::max();
    }
    const auto ds   = chunk.getDataSet("param_limits");
    const auto dims = ds.getDimensions();
    if (dims.size() != 2 || dims[0] != n_params || dims[1] != 2) {
        return std::numeric_limits<double>::max();
    }
    std::vector<double> limits(n_params * 2);
    ds.read_raw(limits.data());
    const auto row = (n_params - 2) * 2;
    return std::max(std::abs(limits[row]), std::abs(limits[row + 1]));
}

double chunk_ntiles(const HighFive::Group& chunk) {
    double ntiles = algorithms::PruneRFIConfig{}.harvest_mask_ntiles;
    if (chunk.exist("rfi_config")) {
        const auto rfi = chunk.getGroup("rfi_config");
        if (rfi.hasAttribute("harvest_mask_ntiles")) {
            rfi.getAttribute("harvest_mask_ntiles").read(ntiles);
        }
    }
    return ntiles;
}

void read_run(const HighFive::Group& harvest,
              const std::string& run_path,
              SizeType n_params,
              double tobs,
              double ntiles,
              double a_max,
              std::vector<Record>& out) {
    if (!harvest.exist("scores") || !harvest.exist("param_sets")) {
        return;
    }
    std::vector<float> scores;
    harvest.getDataSet("scores").read(scores);
    const auto n = scores.size();
    if (n == 0) {
        return;
    }
    const auto ps   = harvest.getDataSet("param_sets");
    const auto dims = ps.getDimensions();
    if (dims.size() != 3 || dims[0] != n || dims[1] < n_params ||
        dims[2] != 2) {
        return;
    }
    std::vector<double> params(dims[0] * dims[1] * dims[2]);
    ps.read_raw(params.data());
    std::vector<std::uint64_t> levels(n, 0);
    std::vector<std::uint64_t> seg_idx(n, 0);
    std::vector<double> t_ref(n, 0.0);
    if (harvest.exist("levels")) {
        harvest.getDataSet("levels").read(levels);
    }
    if (harvest.exist("seg_idx")) {
        harvest.getDataSet("seg_idx").read(seg_idx);
    }
    if (harvest.exist("t_ref")) {
        harvest.getDataSet("t_ref").read(t_ref);
    }
    const auto row_stride = dims[1] * dims[2];
    for (SizeType i = 0; i < n; ++i) {
        Record r;
        r.values.resize(n_params);
        r.errors.resize(n_params);
        for (SizeType p = 0; p < n_params; ++p) {
            r.values[p] = params[(i * row_stride) + (p * 2)];
            r.errors[p] = std::abs(params[(i * row_stride) + (p * 2) + 1]);
        }
        const double f  = r.values[n_params - 1];
        const double df = r.errors[n_params - 1];
        const double a  = n_params >= 2 ? r.values[n_params - 2] : 0.0;
        const double da = n_params >= 2 ? r.errors[n_params - 2] : 0.0;
        if (!utils::is_finite(f) || f <= 0.0 || !utils::is_finite(df) ||
            !utils::is_finite(a) || !utils::is_finite(da)) {
            continue;
        }
        const double tr   = utils::is_finite(t_ref[i]) ? t_ref[i] : 0.0;
        const auto window = algorithms::make_harvest_window(
            f, a, df, da, tr, tobs, ntiles, a_max);
        r.snr     = scores[i];
        r.level   = levels[i];
        r.seg_idx = seg_idx[i];
        r.t_ref   = tr;
        r.f_lo    = window.f_lo;
        r.f_hi    = window.f_hi;
        r.run     = run_path;
        out.push_back(std::move(r));
    }
}

} // namespace

SizeType write_harvested(HighFive::File& file) {
    if (!file.exist("chunks") || !file.hasAttribute("param_names")) {
        return 0;
    }
    std::vector<std::string> names;
    file.getAttribute("param_names").read(names);
    const SizeType n_params = names.size();
    if (n_params == 0) {
        return 0;
    }
    const double tobs = observation_length(file);
    std::vector<Record> records;
    bool any_harvest  = false;
    const auto chunks = file.getGroup("chunks");
    for (const auto& chunk_name : chunks.listObjectNames()) {
        const auto chunk = chunks.getGroup(chunk_name);
        if (!chunk.exist("runs")) {
            continue;
        }
        const double ntiles = chunk_ntiles(chunk);
        const double a_max  = chunk_accel_limit(chunk, n_params);
        const auto runs     = chunk.getGroup("runs");
        for (const auto& run_name : runs.listObjectNames()) {
            const auto run = runs.getGroup(run_name);
            if (!run.exist("harvest")) {
                continue;
            }
            any_harvest = true;
            read_run(run.getGroup("harvest"),
                     "chunks/" + chunk_name + "/runs/" + run_name, n_params,
                     tobs, ntiles, a_max, records);
        }
    }
    if (!any_harvest) {
        return 0;
    }

    // Sources: overlapping frequency windows, merged in order of f_lo.
    std::vector<SizeType> order(records.size());
    std::iota(order.begin(), order.end(), SizeType{0});
    std::ranges::sort(order, [&](SizeType x, SizeType y) {
        return records[x].f_lo != records[y].f_lo
                   ? records[x].f_lo < records[y].f_lo
                   : x < y;
    });
    struct Source {
        SizeType best{};
        double f_lo{};
        double f_hi{};
        std::uint64_t n_records{};
        std::set<std::string> runs;
    };
    std::vector<Source> sources;
    for (const auto i : order) {
        const auto& r = records[i];
        if (sources.empty() || r.f_lo > sources.back().f_hi) {
            sources.push_back(Source{.best      = i,
                                     .f_lo      = r.f_lo,
                                     .f_hi      = r.f_hi,
                                     .n_records = 0,
                                     .runs      = {}});
        }
        auto& s = sources.back();
        s.f_hi  = std::max(s.f_hi, r.f_hi);
        ++s.n_records;
        s.runs.insert(r.run);
        const auto& b = records[s.best];
        if (r.snr > b.snr || (r.snr == b.snr && i < s.best)) {
            s.best = i;
        }
    }
    std::ranges::stable_sort(sources, [&](const Source& x, const Source& y) {
        return records[x.best].snr > records[y.best].snr;
    });

    const SizeType n = sources.size();
    std::vector<double> values(n * n_params);
    std::vector<double> errors(n * n_params);
    std::vector<float> snr(n);
    std::vector<std::uint64_t> level(n);
    std::vector<std::uint64_t> seg_idx(n);
    std::vector<double> t_ref(n);
    std::vector<double> f_lo(n);
    std::vector<double> f_hi(n);
    std::vector<std::uint64_t> n_records(n);
    std::vector<std::uint64_t> n_runs(n);
    std::vector<std::string> run(n);
    for (SizeType k = 0; k < n; ++k) {
        const auto& s = sources[k];
        const auto& b = records[s.best];
        std::ranges::copy(b.values, values.begin() + (k * n_params));
        std::ranges::copy(b.errors, errors.begin() + (k * n_params));
        snr[k]       = b.snr;
        level[k]     = b.level;
        seg_idx[k]   = b.seg_idx;
        t_ref[k]     = b.t_ref;
        f_lo[k]      = s.f_lo;
        f_hi[k]      = s.f_hi;
        n_records[k] = s.n_records;
        n_runs[k]    = s.runs.size();
        run[k]       = b.run;
    }

    if (file.exist("harvested")) {
        file.unlink("harvested");
    }
    auto g = file.createGroup("harvested");
    g.createAttribute(
        "description",
        std::string("sources removed from the pruning tree by early harvest, "
                    "merged over runs: the strongest record of each, with "
                    "its partial S/N at the harvest level (not comparable "
                    "with detection S/N) and parameters at its t_ref"));
    g.createAttribute("param_names", names);
    g.createAttribute("n_sources", static_cast<std::uint64_t>(n));
    g.createAttribute("n_records", static_cast<std::uint64_t>(records.size()));
    const auto write_2d = [&](const char* name,
                              const std::vector<double>& data) {
        auto ds = g.createDataSet(name, HighFive::DataSpace({n, n_params}),
                                  HighFive::create_datatype<double>());
        if (n > 0) {
            ds.write_raw(data.data());
        }
    };
    write_2d("param_sets", values);
    write_2d("param_errors", errors);
    g.createDataSet("snr", snr);
    g.createDataSet("level", level);
    g.createDataSet("seg_idx", seg_idx);
    g.createDataSet("t_ref", t_ref);
    g.createDataSet("f_lo", f_lo);
    g.createDataSet("f_hi", f_hi);
    g.createDataSet("n_records", n_records);
    g.createDataSet("n_runs", n_runs);
    g.createDataSet("run", run);
    return n;
}

} // namespace loki::search
