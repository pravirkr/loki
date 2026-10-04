// Stage-level timing of riptide's frequency-only FFA periodogram.
//
// Replays the loop in riptide's periodogram() and times downsampling, the FFA
// transform, and boxcar S/N separately. Built against riptide's headers with
// riptide's own optimisation flags (see CMakeLists.txt).

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "downsample.hpp"
#include "periodogram.hpp"
#include "snr.hpp"
#include "transforms.hpp"

namespace {

struct Options {
    std::size_t nsamps{1U << 21};
    double tsamp{6.4e-5};
    double period_min{0.1};
    double period_max{1.0};
    std::size_t bins_min{256};
    std::size_t bins_max{280};
    double ducy_max{0.3};
    double wtsp{1.5};
    int reps{3};
    int warmup{1};
    unsigned seed{1};
    std::string input;
};

[[noreturn]] void die(const std::string& message) {
    std::cerr << "riptide_stage_bench: " << message << '\n';
    std::exit(2);
}

template <typename T> T parse_as(std::string_view text, const char* name) {
    T value{};
    const char* const begin = text.data();
    const char* const end   = begin + text.size();
    const auto [ptr, ec]    = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        die(std::string("invalid value for ") + name + ": " +
            std::string(text));
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto need                  = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                die(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--nsamps") {
            opt.nsamps = parse_as<std::size_t>(need("--nsamps"), "--nsamps");
        } else if (arg == "--tsamp") {
            opt.tsamp = std::stod(std::string(need("--tsamp")));
        } else if (arg == "--period-min") {
            opt.period_min = std::stod(std::string(need("--period-min")));
        } else if (arg == "--period-max") {
            opt.period_max = std::stod(std::string(need("--period-max")));
        } else if (arg == "--bins-min") {
            opt.bins_min =
                parse_as<std::size_t>(need("--bins-min"), "--bins-min");
        } else if (arg == "--bins-max") {
            opt.bins_max =
                parse_as<std::size_t>(need("--bins-max"), "--bins-max");
        } else if (arg == "--ducy-max") {
            opt.ducy_max = std::stod(std::string(need("--ducy-max")));
        } else if (arg == "--wtsp") {
            opt.wtsp = std::stod(std::string(need("--wtsp")));
        } else if (arg == "--reps") {
            opt.reps = parse_as<int>(need("--reps"), "--reps");
        } else if (arg == "--warmup") {
            opt.warmup = parse_as<int>(need("--warmup"), "--warmup");
        } else if (arg == "--seed") {
            opt.seed = parse_as<unsigned>(need("--seed"), "--seed");
        } else if (arg == "--input") {
            opt.input = std::string(need("--input"));
        } else if (arg == "--help") {
            std::cout
                << "usage: riptide_stage_bench [--nsamps N] [--tsamp S] "
                   "[--period-min S] [--period-max S] [--bins-min N] "
                   "[--bins-max N] [--ducy-max F] [--wtsp F] [--reps N] "
                   "[--warmup N] [--seed N] [--input raw_f32]\n";
            std::exit(0);
        } else {
            die("unknown argument " + std::string(arg));
        }
    }
    if (opt.reps < 1 || opt.warmup < 0) {
        die("--reps must be >= 1 and --warmup >= 0");
    }
    return opt;
}

// Same recurrence as riptide.ffautils.generate_width_trials, anchored on bins_min.
std::vector<std::size_t>
width_trials(std::size_t bins_min, double ducy_max, double wtsp) {
    const auto wmax = static_cast<std::size_t>(
        std::max(1.0, ducy_max * static_cast<double>(bins_min)));
    std::vector<std::size_t> widths;
    std::size_t w = 1;
    while (w <= wmax) {
        widths.push_back(w);
        const auto grown = static_cast<std::size_t>(wtsp * static_cast<double>(w));
        w                = std::max(w + 1, grown);
    }
    return widths;
}

std::vector<float> load_or_make(const Options& opt) {
    std::vector<float> data(opt.nsamps);
    if (!opt.input.empty()) {
        std::ifstream input(opt.input, std::ios::binary);
        if (!input) {
            die("cannot open " + opt.input);
        }
        input.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (static_cast<std::size_t>(input.gcount()) !=
            data.size() * sizeof(float)) {
            die("--input is shorter than --nsamps float32 samples");
        }
        return data;
    }
    std::mt19937 rng(opt.seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& sample : data) {
        sample = normal(rng);
    }
    return data;
}

struct RepTimes {
    double downsample_s{0.0};
    double transform_s{0.0};
    double snr_s{0.0};
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
}

// One full periodogram, matching riptide::periodogram's control flow.
RepTimes run_once(const float* data,
                  const Options& opt,
                  const std::vector<std::size_t>& widths,
                  std::size_t ntrials) {
    const double ds_ini = opt.period_min / (opt.tsamp * static_cast<double>(opt.bins_min));
    const double ds_geo =
        (static_cast<double>(opt.bins_max) + 1.0) / static_cast<double>(opt.bins_min);
    const std::size_t num_downsamplings = static_cast<std::size_t>(
        std::ceil(std::log(opt.period_max / opt.period_min) / std::log(ds_geo)));

    const std::size_t bufsize = riptide::downsampled_size(opt.nsamps, ds_ini);
    std::vector<float> input_mem(bufsize);
    std::vector<float> ffabuf_mem(bufsize);
    std::vector<float> ffaout_mem(bufsize);
    std::vector<float> snr(ntrials * widths.size(), 0.0F);
    float* snr_cursor = snr.data();

    RepTimes times;
    for (std::size_t ids = 0; ids < num_downsamplings; ++ids) {
        const double f   = ds_ini * std::pow(ds_geo, static_cast<double>(ids));
        const double tau = f * opt.tsamp;
        const double period_max_samples = opt.period_max / tau;
        const std::size_t n = riptide::downsampled_size(opt.nsamps, f);

        const float* input = data;
        if (f != 1.0) {
            const auto t0 = std::chrono::steady_clock::now();
            riptide::downsample(data, opt.nsamps, f, input_mem.data());
            times.downsample_s += seconds_since(t0);
            input = input_mem.data();
        }

        const std::size_t bstop = std::min(
            {opt.bins_max, n, static_cast<std::size_t>(period_max_samples)});
        for (std::size_t bins = opt.bins_min; bins <= bstop; ++bins) {
            const std::size_t rows = n / bins;
            const float stdnoise   = static_cast<float>(
                std::sqrt(static_cast<double>(rows) *
                          riptide::downsampled_variance(opt.nsamps, f)));
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const std::size_t rows_eval =
                std::min(rows, riptide::ceilshift(rows, bins, period_ceil));

            const auto t_transform = std::chrono::steady_clock::now();
            riptide::transform(input, rows, bins, ffabuf_mem.data(),
                               ffaout_mem.data());
            times.transform_s += seconds_since(t_transform);

            const auto t_snr = std::chrono::steady_clock::now();
            auto block       = riptide::ConstBlock(ffaout_mem.data(), rows_eval, bins);
            riptide::snr2(block, widths.data(), widths.size(), stdnoise, snr_cursor);
            times.snr_s += seconds_since(t_snr);
            snr_cursor += rows_eval * widths.size();
        }
    }
    // Touch the sink so a future compiler cannot drop snr2.
    if (snr_cursor != snr.data() + snr.size()) {
        die("stage bench wrote a different number of S/N rows than periodogram_length");
    }
    volatile float sink = snr.back();
    (void)sink;
    return times;
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void print_json(const Options& opt,
                std::size_t ntrials,
                std::size_t nwidths,
                const std::vector<RepTimes>& reps) {
    std::vector<double> down;
    std::vector<double> xform;
    std::vector<double> snr;
    std::vector<double> total;
    down.reserve(reps.size());
    for (const auto& rep : reps) {
        down.push_back(rep.downsample_s);
        xform.push_back(rep.transform_s);
        snr.push_back(rep.snr_s);
        total.push_back(rep.downsample_s + rep.transform_s + rep.snr_s);
    }
    auto emit_rep = [](const RepTimes& rep) {
        std::cout << "{\"downsample_s\":" << rep.downsample_s
                  << ",\"transform_s\":" << rep.transform_s
                  << ",\"snr_s\":" << rep.snr_s << ",\"total_s\":"
                  << (rep.downsample_s + rep.transform_s + rep.snr_s) << "}";
    };
    std::cout << std::setprecision(8);
    std::cout << "{\"nsamps\":" << opt.nsamps << ",\"tsamp\":" << opt.tsamp
              << ",\"period_min\":" << opt.period_min
              << ",\"period_max\":" << opt.period_max
              << ",\"bins_min\":" << opt.bins_min << ",\"bins_max\":" << opt.bins_max
              << ",\"ntrials\":" << ntrials << ",\"nwidths\":" << nwidths
              << ",\"reps\":[";
    for (std::size_t i = 0; i < reps.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        emit_rep(reps[i]);
    }
    std::cout << "],\"median\":{\"downsample_s\":" << median_of(down)
              << ",\"transform_s\":" << median_of(xform)
              << ",\"snr_s\":" << median_of(snr)
              << ",\"total_s\":" << median_of(total) << "}}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const auto widths =
            width_trials(opt.bins_min, opt.ducy_max, opt.wtsp);
        const std::size_t ntrials = riptide::periodogram_length(
            opt.nsamps, opt.tsamp, opt.period_min, opt.period_max, opt.bins_min,
            opt.bins_max);
        const auto data = load_or_make(opt);
        for (int i = 0; i < opt.warmup; ++i) {
            (void)run_once(data.data(), opt, widths, ntrials);
        }
        std::vector<RepTimes> reps;
        reps.reserve(static_cast<std::size_t>(opt.reps));
        for (int i = 0; i < opt.reps; ++i) {
            reps.push_back(run_once(data.data(), opt, widths, ntrials));
        }
        print_json(opt, ntrials, widths.size(), reps);
        return 0;
    } catch (const std::exception& ex) {
        die(ex.what());
    }
}
