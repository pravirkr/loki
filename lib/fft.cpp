#include "loki/utils/fft.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"

namespace loki::math {

namespace {

// FFTW planner create/destroy is not thread-safe. Serialize those calls
// with a library-wide mutex. Execute is already thread-safe and is not
// serialized.
std::mutex& fftw_planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

void destroy_fftw_plan(fftwf_plan plan) noexcept {
    if (plan == nullptr) {
        return;
    }
    std::scoped_lock lock(fftw_planner_mutex());
    fftwf_destroy_plan(plan);
}

class FFTWPlan {
public:
    explicit FFTWPlan(fftwf_plan plan) : m_plan(plan) {
        if (m_plan == nullptr) {
            throw std::runtime_error("FFTW plan creation failed");
        }
    }
    ~FFTWPlan() { destroy_fftw_plan(m_plan); }
    FFTWPlan(const FFTWPlan&)            = delete;
    FFTWPlan& operator=(const FFTWPlan&) = delete;
    FFTWPlan(FFTWPlan&& other) noexcept
        : m_plan(std::exchange(other.m_plan, nullptr)) {}
    FFTWPlan& operator=(FFTWPlan&& other) noexcept {
        if (this != &other) {
            destroy_fftw_plan(m_plan);
            m_plan = std::exchange(other.m_plan, nullptr);
        }
        return *this;
    }

    [[nodiscard]] fftwf_plan get() const noexcept { return m_plan; }

private:
    fftwf_plan m_plan{nullptr};
};

struct HowmanyPlan {
    SizeType n_howmany{};
    FFTWPlan fft_plan;
};

struct BatchSlice {
    SizeType offset{};
    SizeType count{};
};

struct PreparedPlans {
    SizeType max_howmany{};
    bool exact_howmany_cache{false};
    std::vector<HowmanyPlan> rfft;
    std::vector<HowmanyPlan> irfft;
    std::unordered_map<SizeType, FFTWPlan> irfft_exact;
};

[[nodiscard]] int howmany_to_fftw(SizeType howmany) {
    error_check::check_less_equal(
        howmany, static_cast<SizeType>(std::numeric_limits<int>::max()),
        "FFTWManager: howmany exceeds FFTW int limit");
    error_check::check_less_equal(
        howmany, static_cast<SizeType>(kFFTBatchSizeMax),
        "FFTWManager: howmany exceeds kFFTBatchSizeMax");
    return static_cast<int>(howmany);
}

[[nodiscard]] fftwf_plan get_cached_plan(const std::vector<HowmanyPlan>& plans,
                                         SizeType howmany) {
    assert(std::has_single_bit(howmany));
    const auto idx = static_cast<SizeType>(std::countr_zero(howmany));
    assert(idx < plans.size());
    return plans[idx].fft_plan.get();
}

[[nodiscard]] fftwf_plan
get_ephemeral_plan(const std::vector<HowmanyPlan>& plans, SizeType howmany) {
    for (const auto& entry : plans) {
        if (entry.n_howmany == howmany) {
            return entry.fft_plan.get();
        }
    }
    throw error_check::DetailedException(
        std::format("FFTWManager: no ephemeral plan for howmany={}", howmany));
}

void consume_howmany(SizeType& remaining, SizeType howmany) {
    error_check::check_greater(howmany, SizeType{0},
                               "FFTWManager: howmany must be positive");
    error_check::check_less_equal(
        howmany, remaining,
        "FFTWManager: howmany exceeds remaining transforms");
    remaining -= howmany;
}

// Distinct howmany values needed to cover `count` transforms with chunks ≤ max.
void add_howmany_sizes(std::vector<SizeType>& sizes,
                       SizeType count,
                       SizeType max_howmany) {
    error_check::check_greater(max_howmany, SizeType{0},
                               "FFTWManager: max_howmany must be positive");
    while (count > 0) {
        const SizeType howmany = std::min(max_howmany, count);
        if (std::ranges::find(sizes, howmany) == sizes.end()) {
            sizes.push_back(howmany);
        }
        consume_howmany(count, howmany);
    }
}

SizeType
next_howmany(SizeType remaining, SizeType max_howmany, bool pow2_decompose) {
    const SizeType capped = std::min(
        {remaining, static_cast<SizeType>(kFFTBatchSizeMax), max_howmany});
    error_check::check_greater(
        capped, SizeType{0},
        "FFTWManager: next_howmany cap is zero (remaining or max_howmany is "
        "zero)");
    if (!pow2_decompose) {
        return capped;
    }
    return std::bit_floor(capped);
}

void check_batch_extent(SizeType batch_size,
                        SizeType length,
                        std::string_view what) {
    const bool fits =
        length == 0 ||
        batch_size <= (std::numeric_limits<SizeType>::max() / length);
    error_check::check(fits, std::format("FFTWManager: {} * batch_size "
                                         "overflows SizeType",
                                         what));
}

std::vector<BatchSlice> build_batch_slices(SizeType batch_size,
                                           SizeType n_workers) {
    const auto base  = batch_size / n_workers;
    const auto extra = batch_size % n_workers;
    std::vector<BatchSlice> slices(n_workers);
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        slices[worker] = BatchSlice{
            .offset = (worker * base) + std::min(worker, extra),
            .count  = base + (worker < extra ? SizeType{1} : SizeType{0}),
        };
    }
    return slices;
}

FFTWPlan make_rfft_plan(SizeType n_real, SizeType n_complex, SizeType howmany) {
    const int n_real_i    = static_cast<int>(n_real);
    const int n_complex_i = static_cast<int>(n_complex);
    const int howmany_i   = howmany_to_fftw(howmany);
    fftwf_plan raw        = nullptr;
    {
        std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_many_dft_r2c(
            1,                       // rank
            &n_real_i,               // transform size
            howmany_i,               // number of transforms
            nullptr,                 // input (dummy for planning)
            nullptr, 1, n_real_i,    // input layout: stride=1, dist=n_real
            nullptr,                 // output (dummy for planning)
            nullptr, 1, n_complex_i, // output layout: stride=1, dist=n_complex
            FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        throw std::runtime_error(
            std::format("Failed to create RFFT plan for n_real={}, howmany={}",
                        n_real, howmany));
    }
    return FFTWPlan{raw};
}

FFTWPlan
make_irfft_plan(SizeType n_real, SizeType n_complex, SizeType howmany) {
    const int n_real_i    = static_cast<int>(n_real);
    const int n_complex_i = static_cast<int>(n_complex);
    const int howmany_i   = howmany_to_fftw(howmany);
    fftwf_plan raw        = nullptr;
    {
        std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_many_dft_c2r(
            1,                       // rank
            &n_real_i,               // transform size
            howmany_i,               // number of transforms
            nullptr,                 // input (dummy for planning)
            nullptr, 1, n_complex_i, // input layout: stride=1, dist=n_complex
            nullptr,                 // output (dummy for planning)
            nullptr, 1, n_real_i,    // output layout: stride=1, dist=n_real
            FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        throw std::runtime_error(
            std::format("Failed to create IRFFT plan for n_real={}, howmany={}",
                        n_real, howmany));
    }
    return FFTWPlan{raw};
}

void build_rfft_ladder(std::vector<HowmanyPlan>& out,
                       SizeType n_real,
                       SizeType n_complex,
                       SizeType from_howmany,
                       SizeType to_howmany) {
    error_check::check_greater(from_howmany, SizeType{0},
                               "FFTWManager: ladder from_howmany must be "
                               "positive");
    for (SizeType howmany = from_howmany; howmany <= to_howmany;) {
        out.push_back(HowmanyPlan{
            .n_howmany = howmany,
            .fft_plan  = make_rfft_plan(n_real, n_complex, howmany)});
        if (howmany > to_howmany / 2) {
            break;
        }
        howmany *= 2;
    }
}

void build_irfft_ladder(std::vector<HowmanyPlan>& out,
                        SizeType n_real,
                        SizeType n_complex,
                        SizeType from_howmany,
                        SizeType to_howmany) {
    error_check::check_greater(from_howmany, SizeType{0},
                               "FFTWManager: ladder from_howmany must be "
                               "positive");
    for (SizeType howmany = from_howmany; howmany <= to_howmany;) {
        out.push_back(HowmanyPlan{
            .n_howmany = howmany,
            .fft_plan  = make_irfft_plan(n_real, n_complex, howmany)});
        if (howmany > to_howmany / 2) {
            break;
        }
        howmany *= 2;
    }
}

fftwf_plan get_or_create_exact_irfft_plan(PreparedPlans& prepared,
                                          std::mutex& exact_mutex,
                                          SizeType n_real,
                                          SizeType n_complex,
                                          SizeType howmany) {
    {
        std::scoped_lock lock(exact_mutex);
        const auto it = prepared.irfft_exact.find(howmany);
        if (it != prepared.irfft_exact.end()) {
            return it->second.get();
        }
    }
    FFTWPlan plan     = make_irfft_plan(n_real, n_complex, howmany);
    fftwf_plan result = nullptr;
    {
        std::scoped_lock lock(exact_mutex);
        const auto [it, inserted] =
            prepared.irfft_exact.try_emplace(howmany, std::move(plan));
        result = it->second.get();
    }
    return result;
}

void execute_rfft_cached(const std::vector<BatchSlice>& slices,
                         SizeType n_workers,
                         float* in_ptr,
                         fftwf_complex* out_ptr,
                         SizeType n_real,
                         SizeType n_complex,
                         const std::vector<HowmanyPlan>& plans,
                         SizeType max_howmany) {
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(n_workers, slices, in_ptr, out_ptr, n_real, n_complex, plans,       \
               max_howmany)
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        const BatchSlice slice = slices[worker];
        SizeType offset        = slice.offset;
        SizeType remaining     = slice.count;
        while (remaining > 0) {
            const SizeType howmany =
                next_howmany(remaining, max_howmany, /*pow2_decompose=*/true);
            fftwf_plan plan = get_cached_plan(plans, howmany);
            fftwf_execute_dft_r2c(plan, in_ptr + (offset * n_real),
                                  out_ptr + (offset * n_complex));
            offset += howmany;
            consume_howmany(remaining, howmany);
        }
    }
}

void execute_rfft_ephemeral(const std::vector<BatchSlice>& slices,
                            SizeType n_workers,
                            float* in_ptr,
                            fftwf_complex* out_ptr,
                            SizeType n_real,
                            SizeType n_complex,
                            const std::vector<HowmanyPlan>& plans) {
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(n_workers, slices, in_ptr, out_ptr, n_real, n_complex, plans)
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        const BatchSlice slice = slices[worker];
        SizeType offset        = slice.offset;
        SizeType remaining     = slice.count;
        while (remaining > 0) {
            const SizeType howmany = next_howmany(remaining, kFFTBatchSizeMax,
                                                  /*pow2_decompose=*/false);
            fftwf_plan plan        = get_ephemeral_plan(plans, howmany);
            fftwf_execute_dft_r2c(plan, in_ptr + (offset * n_real),
                                  out_ptr + (offset * n_complex));
            offset += howmany;
            consume_howmany(remaining, howmany);
        }
    }
}

void execute_irfft_cached(const std::vector<BatchSlice>& slices,
                          SizeType n_workers,
                          fftwf_complex* in_ptr,
                          float* out_ptr,
                          SizeType n_real,
                          SizeType n_complex,
                          const std::vector<HowmanyPlan>& plans,
                          SizeType max_howmany,
                          float norm) {
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(n_workers, slices, in_ptr, out_ptr, n_real, n_complex, plans,       \
               max_howmany, norm)
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        const BatchSlice slice = slices[worker];
        SizeType offset        = slice.offset;
        SizeType remaining     = slice.count;
        while (remaining > 0) {
            const SizeType howmany =
                next_howmany(remaining, max_howmany, /*pow2_decompose=*/true);
            fftwf_plan plan = get_cached_plan(plans, howmany);
            fftwf_execute_dft_c2r(plan, in_ptr + (offset * n_complex),
                                  out_ptr + (offset * n_real));
            float* chunk_out      = out_ptr + (offset * n_real);
            const SizeType n_elem = howmany * n_real;
            for (SizeType i = 0; i < n_elem; ++i) {
                chunk_out[i] *= norm;
            }
            offset += howmany;
            consume_howmany(remaining, howmany);
        }
    }
}

void execute_irfft_ephemeral(const std::vector<BatchSlice>& slices,
                             SizeType n_workers,
                             fftwf_complex* in_ptr,
                             float* out_ptr,
                             SizeType n_real,
                             SizeType n_complex,
                             const std::vector<HowmanyPlan>& plans,
                             float norm) {
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(n_workers, slices, in_ptr, out_ptr, n_real, n_complex, plans, norm)
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        const BatchSlice slice = slices[worker];
        SizeType offset        = slice.offset;
        SizeType remaining     = slice.count;
        while (remaining > 0) {
            const SizeType howmany = next_howmany(remaining, kFFTBatchSizeMax,
                                                  /*pow2_decompose=*/false);
            fftwf_plan plan        = get_ephemeral_plan(plans, howmany);
            fftwf_execute_dft_c2r(plan, in_ptr + (offset * n_complex),
                                  out_ptr + (offset * n_real));
            float* chunk_out      = out_ptr + (offset * n_real);
            const SizeType n_elem = howmany * n_real;
            for (SizeType i = 0; i < n_elem; ++i) {
                chunk_out[i] *= norm;
            }
            offset += howmany;
            consume_howmany(remaining, howmany);
        }
    }
}

void execute_irfft_exact(PreparedPlans& prepared,
                         std::mutex& exact_mutex,
                         const std::vector<BatchSlice>& slices,
                         SizeType n_workers,
                         fftwf_complex* in_ptr,
                         float* out_ptr,
                         SizeType n_real,
                         SizeType n_complex,
                         SizeType max_howmany,
                         float norm) {
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(prepared, exact_mutex, n_workers, slices, in_ptr, out_ptr, n_real,  \
               n_complex, max_howmany, norm)
    for (SizeType worker = 0; worker < n_workers; ++worker) {
        const BatchSlice slice = slices[worker];
        SizeType offset        = slice.offset;
        SizeType remaining     = slice.count;
        while (remaining > 0) {
            const SizeType howmany =
                next_howmany(remaining, max_howmany, /*pow2_decompose=*/false);
            fftwf_plan plan = get_or_create_exact_irfft_plan(
                prepared, exact_mutex, n_real, n_complex, howmany);
            fftwf_execute_dft_c2r(plan, in_ptr + (offset * n_complex),
                                  out_ptr + (offset * n_real));
            float* chunk_out      = out_ptr + (offset * n_real);
            const SizeType n_elem = howmany * n_real;
            for (SizeType i = 0; i < n_elem; ++i) {
                chunk_out[i] *= norm;
            }
            offset += howmany;
            consume_howmany(remaining, howmany);
        }
    }
}
} // namespace

class FFTWManager::Impl {
public:
    std::unordered_map<SizeType, PreparedPlans> plans;
    std::mutex exact_plans_mutex;
};

FFTWManager::FFTWManager() : m_impl(std::make_unique<Impl>()) {}
FFTWManager::~FFTWManager()                                 = default;
FFTWManager::FFTWManager(FFTWManager&&) noexcept            = default;
FFTWManager& FFTWManager::operator=(FFTWManager&&) noexcept = default;

void FFTWManager::prepare_plans(std::span<const SizeType> n_reals,
                                SizeType max_howmany) {
    error_check::check_greater(max_howmany, SizeType{0},
                               "FFTWManager::prepare_plans: max_howmany "
                               "must be positive");
    error_check::check_less_equal(
        max_howmany, static_cast<SizeType>(kFFTBatchSizeMax),
        "FFTWManager::prepare_plans: max_howmany exceeds kFFTBatchSizeMax");
    error_check::check_power_of_2(max_howmany,
                                  "FFTWManager::prepare_plans: max_howmany");

    std::vector<SizeType> unique_n(n_reals.begin(), n_reals.end());
    std::ranges::sort(unique_n);
    unique_n.erase(std::ranges::unique(unique_n).begin(), unique_n.end());

    SizeType n_new = 0;
    for (const auto n_real : unique_n) {
        if (n_real == 0) {
            continue;
        }
        error_check::check_less_equal(
            n_real, static_cast<SizeType>(std::numeric_limits<int>::max()),
            "FFTWManager::prepare_plans: n_real exceeds FFTW int limit");
        const auto n_complex = (n_real / 2) + 1;

        const auto existing = m_impl->plans.find(n_real);
        if (existing == m_impl->plans.end()) {
            PreparedPlans candidate;
            candidate.max_howmany         = max_howmany;
            candidate.exact_howmany_cache = false;
            build_rfft_ladder(candidate.rfft, n_real, n_complex, 1,
                              max_howmany);
            build_irfft_ladder(candidate.irfft, n_real, n_complex, 1,
                               max_howmany);
            n_new += candidate.rfft.size() + candidate.irfft.size();
            m_impl->plans.emplace(n_real, std::move(candidate));
            continue;
        }

        PreparedPlans& prepared = existing->second;
        if (prepared.exact_howmany_cache) {
            throw error_check::DetailedException(std::format(
                "FFTWManager: n_real={} already prepared with {} cache mode",
                n_real,
                prepared.exact_howmany_cache ? "exact-howmany"
                                             : "power-of-two"));
        }
        if (prepared.max_howmany == max_howmany) {
            continue;
        }
        if (max_howmany < prepared.max_howmany) {
            throw error_check::DetailedException(std::format(
                "FFTWManager::prepare_plans: cannot shrink max_howmany for "
                "n_real={} from {} to {}",
                n_real, prepared.max_howmany, max_howmany));
        }

        std::vector<HowmanyPlan> new_rfft;
        std::vector<HowmanyPlan> new_irfft;
        build_rfft_ladder(new_rfft, n_real, n_complex, prepared.max_howmany * 2,
                          max_howmany);
        build_irfft_ladder(new_irfft, n_real, n_complex,
                           prepared.max_howmany * 2, max_howmany);
        n_new += new_rfft.size() + new_irfft.size();
        prepared.rfft.insert(prepared.rfft.end(),
                             std::make_move_iterator(new_rfft.begin()),
                             std::make_move_iterator(new_rfft.end()));
        prepared.irfft.insert(prepared.irfft.end(),
                              std::make_move_iterator(new_irfft.begin()),
                              std::make_move_iterator(new_irfft.end()));
        prepared.max_howmany = max_howmany;
    }
    spdlog::info("FFTWManager: cached {} plans for {} n_real values",
                 n_cached_plans(), m_impl->plans.size());
    spdlog::debug("FFTWManager: created {} new ladder plans this call", n_new);
}

void FFTWManager::prepare_exact_plans(std::span<const SizeType> n_reals,
                                      SizeType max_howmany) {
    error_check::check_greater(max_howmany, SizeType{0},
                               "FFTWManager::prepare_exact_plans: max_howmany "
                               "must be positive");
    error_check::check_less_equal(
        max_howmany, static_cast<SizeType>(kFFTBatchSizeMax),
        "FFTWManager::prepare_exact_plans: max_howmany exceeds "
        "kFFTBatchSizeMax");

    std::vector<SizeType> unique_n(n_reals.begin(), n_reals.end());
    std::ranges::sort(unique_n);
    unique_n.erase(std::ranges::unique(unique_n).begin(), unique_n.end());

    SizeType n_registered = 0;
    for (const auto n_real : unique_n) {
        if (n_real == 0) {
            continue;
        }
        error_check::check_less_equal(
            n_real, static_cast<SizeType>(std::numeric_limits<int>::max()),
            "FFTWManager::prepare_exact_plans: n_real exceeds FFTW int limit");

        const auto existing = m_impl->plans.find(n_real);
        if (existing == m_impl->plans.end()) {
            PreparedPlans candidate;
            candidate.max_howmany         = max_howmany;
            candidate.exact_howmany_cache = true;
            m_impl->plans.emplace(n_real, std::move(candidate));
            ++n_registered;
            continue;
        }

        PreparedPlans& prepared = existing->second;
        if (!prepared.exact_howmany_cache) {
            throw error_check::DetailedException(std::format(
                "FFTWManager: n_real={} already prepared with {} cache mode",
                n_real,
                prepared.exact_howmany_cache ? "exact-howmany"
                                             : "power-of-two"));
        }
        if (max_howmany < prepared.max_howmany) {
            throw error_check::DetailedException(std::format(
                "FFTWManager::prepare_exact_plans: cannot shrink max_howmany "
                "for n_real={} from {} to {}",
                n_real, prepared.max_howmany, max_howmany));
        }
        prepared.max_howmany = max_howmany;
        ++n_registered;
    }
    spdlog::info(
        "FFTWManager: registered {} n_real values for exact-howmany caching",
        n_registered);
}

bool FFTWManager::has_prepared(SizeType n_real) const noexcept {
    return m_impl->plans.contains(n_real);
}

SizeType FFTWManager::n_cached_plans() const noexcept {
    SizeType n_plans = 0;
    for (const auto& [n_real, prepared] : m_impl->plans) {
        n_plans += prepared.rfft.size() + prepared.irfft.size();
        n_plans += prepared.irfft_exact.size();
    }
    return n_plans;
}

void FFTWManager::rfft_batch(std::span<float> real_input,
                             std::span<ComplexType> complex_output,
                             SizeType batch_size,
                             SizeType n_real,
                             int nthreads) {
    error_check::check_greater(n_real, SizeType{0},
                               "FFTWManager::rfft_batch: n_real must be "
                               "positive");
    if (batch_size == 0) {
        return;
    }
    error_check::check_less_equal(
        n_real, static_cast<SizeType>(std::numeric_limits<int>::max()),
        "FFTWManager::rfft_batch: n_real exceeds FFTW int limit");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch_extent(batch_size, n_real, "n_real");
    check_batch_extent(batch_size, n_complex, "n_complex");
    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "FFTWManager::rfft_batch: real_input size does not match batch size");
    error_check::check_equal(complex_output.size(), batch_size * n_complex,
                             "FFTWManager::rfft_batch: complex_output size "
                             "does not match batch size");

    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    const auto n_workers =
        std::min(static_cast<SizeType>(nthreads), batch_size);
    const auto slices = build_batch_slices(batch_size, n_workers);

    // May overwrite real input. Be prepared to make a copy if needed.
    auto* in_ptr  = real_input.data();
    auto* out_ptr = reinterpret_cast<fftwf_complex*>(complex_output.data());

    if (has_prepared(n_real)) {
        PreparedPlans& prepared = m_impl->plans.at(n_real);
        if (prepared.exact_howmany_cache) {
            throw error_check::DetailedException(std::format(
                "FFTWManager::rfft_batch: n_real={} uses exact-howmany cache; "
                "use prepare_plans for RFFT",
                n_real));
        }
        execute_rfft_cached(slices, n_workers, in_ptr, out_ptr, n_real,
                            n_complex, prepared.rfft, prepared.max_howmany);
        return;
    }

    const auto base  = batch_size / n_workers;
    const auto extra = batch_size % n_workers;
    std::vector<SizeType> howmany_sizes;
    add_howmany_sizes(howmany_sizes, base, kFFTBatchSizeMax);
    if (extra > 0) {
        add_howmany_sizes(howmany_sizes, base + 1, kFFTBatchSizeMax);
    }
    std::vector<HowmanyPlan> local_plans;
    local_plans.reserve(howmany_sizes.size());
    for (const SizeType howmany : howmany_sizes) {
        local_plans.push_back(HowmanyPlan{
            .n_howmany = howmany,
            .fft_plan  = make_rfft_plan(n_real, n_complex, howmany)});
    }
    execute_rfft_ephemeral(slices, n_workers, in_ptr, out_ptr, n_real,
                           n_complex, local_plans);
}

void FFTWManager::irfft_batch(std::span<ComplexType> complex_input,
                              std::span<float> real_output,
                              SizeType batch_size,
                              SizeType n_real,
                              int nthreads) {
    error_check::check_greater(n_real, SizeType{0},
                               "FFTWManager::irfft_batch: n_real must be "
                               "positive");
    if (batch_size == 0) {
        return;
    }
    error_check::check_less_equal(
        n_real, static_cast<SizeType>(std::numeric_limits<int>::max()),
        "FFTWManager::irfft_batch: n_real exceeds FFTW int limit");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch_extent(batch_size, n_real, "n_real");
    check_batch_extent(batch_size, n_complex, "n_complex");
    error_check::check_equal(real_output.size(), batch_size * n_real,
                             "FFTWManager::irfft_batch: real_output size "
                             "does not match batch size");
    error_check::check_equal(complex_input.size(), batch_size * n_complex,
                             "FFTWManager::irfft_batch: complex_input size "
                             "does not match batch size");

    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    const auto n_workers =
        std::min(static_cast<SizeType>(nthreads), batch_size);
    const auto slices = build_batch_slices(batch_size, n_workers);
    const float norm  = 1.0F / static_cast<float>(n_real);

    // May overwrite complex input. Be prepared to make a copy if needed.
    auto* in_ptr  = reinterpret_cast<fftwf_complex*>(complex_input.data());
    auto* out_ptr = real_output.data();

    if (has_prepared(n_real)) {
        PreparedPlans& prepared = m_impl->plans.at(n_real);
        if (prepared.exact_howmany_cache) {
            execute_irfft_exact(prepared, m_impl->exact_plans_mutex, slices,
                                n_workers, in_ptr, out_ptr, n_real, n_complex,
                                prepared.max_howmany, norm);
        } else {
            execute_irfft_cached(slices, n_workers, in_ptr, out_ptr, n_real,
                                 n_complex, prepared.irfft,
                                 prepared.max_howmany, norm);
        }
        return;
    }

    const auto base  = batch_size / n_workers;
    const auto extra = batch_size % n_workers;
    std::vector<SizeType> howmany_sizes;
    add_howmany_sizes(howmany_sizes, base, kFFTBatchSizeMax);
    if (extra > 0) {
        add_howmany_sizes(howmany_sizes, base + 1, kFFTBatchSizeMax);
    }
    std::vector<HowmanyPlan> local_plans;
    local_plans.reserve(howmany_sizes.size());
    for (const SizeType howmany : howmany_sizes) {
        local_plans.push_back(HowmanyPlan{
            .n_howmany = howmany,
            .fft_plan  = make_irfft_plan(n_real, n_complex, howmany)});
    }
    execute_irfft_ephemeral(slices, n_workers, in_ptr, out_ptr, n_real,
                            n_complex, local_plans, norm);
}

// --- FFT2D Implementation ---
FFT2D::FFT2D(SizeType n1x, SizeType n2x, SizeType ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_fft_size((m_ny / 2) + 1),
      m_n1_fft(fftwf_alloc_complex(n1x * m_fft_size)),
      m_n2_fft(fftwf_alloc_complex(n2x * m_fft_size)),
      m_n1n2_fft(fftwf_alloc_complex(n1x * n2x * m_fft_size)),
      m_plan_forward(fftwf_plan_dft_r2c_2d(static_cast<int>(n1x),
                                           static_cast<int>(m_ny),
                                           nullptr,
                                           nullptr,
                                           FFTW_ESTIMATE)),
      m_plan_inverse(fftwf_plan_dft_c2r_2d(static_cast<int>(n1x),
                                           static_cast<int>(m_ny),
                                           nullptr,
                                           nullptr,
                                           FFTW_ESTIMATE)) {

      };

FFT2D::~FFT2D() {
    fftwf_free(m_n1_fft);
    fftwf_free(m_n2_fft);
    fftwf_free(m_n1n2_fft);
    fftwf_destroy_plan(m_plan_forward);
    fftwf_destroy_plan(m_plan_inverse);
}

void FFT2D::circular_convolve(std::span<float> n1,
                              std::span<float> n2,
                              std::span<float> out) {
    // Forward FFT
    fftwf_execute_dft_r2c(m_plan_forward, n1.data(), m_n1_fft);
    fftwf_execute_dft_r2c(m_plan_forward, n2.data(), m_n2_fft);
    // Multiply the FFTs
    for (SizeType i = 0; i < m_n1x * m_n2x * m_fft_size; ++i) {
        const SizeType idx_n1 =
            ((i / (m_n2x * m_fft_size)) * m_fft_size) + (i % m_fft_size);
        const SizeType idx_n2 =
            ((i / m_fft_size) % m_n2x * m_fft_size) + (i % m_fft_size);
        m_n1n2_fft[i][0] = (m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][0]) -
                           (m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][1]);
        m_n1n2_fft[i][1] = (m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][1]) +
                           (m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][0]);
    }
    // Inverse FFT
    fftwf_execute_dft_c2r(m_plan_inverse, m_n1n2_fft, out.data());
}

void rfft_batch(std::span<float> real_input,
                std::span<ComplexType> complex_output,
                SizeType batch_size,
                SizeType n_real,
                int nthreads) {
    FFTWManager manager;
    manager.rfft_batch(real_input, complex_output, batch_size, n_real,
                       nthreads);
}

void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 SizeType batch_size,
                 SizeType n_real,
                 int nthreads) {
    FFTWManager manager;
    manager.irfft_batch(complex_input, real_output, batch_size, n_real,
                        nthreads);
}

} // namespace loki::math