#include "loki/io/search_series.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/detail/utils.hpp"
#include "lib/io/search_series_impl.hpp"

namespace loki::io {

SearchSeries::SearchSeries() noexcept                          = default;
SearchSeries::~SearchSeries()                                  = default;
SearchSeries::SearchSeries(SearchSeries&&) noexcept            = default;
SearchSeries& SearchSeries::operator=(SearchSeries&&) noexcept = default;

SearchSeries::SearchSeries(std::unique_ptr<Impl> impl) noexcept
    : m_impl(std::move(impl)) {}

SearchSeries::Impl& SearchSeries::impl() {
    if (!m_impl) {
        throw std::logic_error("SearchSeries: empty handle");
    }
    return *m_impl;
}

const SearchSeries::Impl& SearchSeries::impl() const {
    if (!m_impl) {
        throw std::logic_error("SearchSeries: empty handle");
    }
    return *m_impl;
}

SizeType SearchSeries::size() const noexcept { return m_impl ? m_impl->n : 0; }

double SearchSeries::tsamp() const noexcept {
    return m_impl ? m_impl->tsamp : 0.0;
}

Exec SearchSeries::exec() const { return impl().exec; }

std::pair<SearchSeries, PreprocessReport>
SearchSeries::preprocess(std::span<const float> raw,
                         double tsamp,
                         const PreprocessOptions& options,
                         Exec exec) {
    auto impl   = std::make_unique<Impl>();
    impl->exec  = exec;
    impl->tsamp = tsamp;
    impl->n     = raw.size();
    if (exec.backend == Backend::kCPU) {
        impl->host_e.resize(raw.size());
        impl->host_v.resize(raw.size());
        auto rep = io::preprocess(raw, tsamp, impl->host_e, impl->host_v,
                                  options, exec);
        return {SearchSeries(std::move(impl)), std::move(rep)};
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        if (options.method != PreprocessMethod::kRobust) {
            // The z-score method has no GPU kernels: run it on the host and
            // upload the result once.
            std::vector<float> ts_e(raw.size());
            std::vector<float> ts_v(raw.size());
            auto rep = io::preprocess(raw, tsamp, ts_e, ts_v, options, exec);
            impl->device =
                detail::make_series_storage_gpu(ts_e, ts_v, exec.device);
            return {SearchSeries(std::move(impl)), std::move(rep)};
        }
        // io::preprocess's argument checks, then the resident GPU path.
        if (raw.empty()) {
            throw std::invalid_argument("preprocess: input is empty");
        }
        if (!utils::is_finite(tsamp) || tsamp <= 0.0) {
            throw std::invalid_argument("preprocess: tsamp must be positive");
        }
        options.validate();
        auto [storage, rep] =
            detail::preprocess_gpu_resident(raw, tsamp, options, exec.device);
        impl->device = std::move(storage);
        return {SearchSeries(std::move(impl)), std::move(rep)};
    }
#endif
    loki::detail::throw_unavailable("SearchSeries::preprocess", exec.backend);
}

SearchSeries SearchSeries::from_host(std::span<const float> ts_e,
                                     std::span<const float> ts_v,
                                     double tsamp,
                                     Exec exec) {
    if (ts_e.size() != ts_v.size()) {
        throw std::invalid_argument(
            fmt::format("SearchSeries::from_host: ts_e and ts_v sizes differ "
                        "({} and {})",
                        ts_e.size(), ts_v.size()));
    }
    if (ts_e.empty()) {
        throw std::invalid_argument("SearchSeries::from_host: input is empty");
    }
    if (!utils::is_finite(tsamp) || tsamp <= 0.0) {
        throw std::invalid_argument(
            "SearchSeries::from_host: tsamp must be positive");
    }
    auto impl   = std::make_unique<Impl>();
    impl->exec  = exec;
    impl->tsamp = tsamp;
    impl->n     = ts_e.size();
    if (exec.backend == Backend::kCPU) {
        impl->host_e.assign(ts_e.begin(), ts_e.end());
        impl->host_v.assign(ts_v.begin(), ts_v.end());
        return SearchSeries(std::move(impl));
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        impl->device = detail::make_series_storage_gpu(ts_e, ts_v, exec.device);
        return SearchSeries(std::move(impl));
    }
#endif
    loki::detail::throw_unavailable("SearchSeries::from_host", exec.backend);
}

void SearchSeries::copy_to_host(std::span<float> ts_e,
                                std::span<float> ts_v) const {
    const auto& s = impl();
    if (ts_e.size() != s.n || ts_v.size() != s.n) {
        throw std::invalid_argument(fmt::format(
            "SearchSeries::copy_to_host: outputs need {} samples (got {} and "
            "{})",
            s.n, ts_e.size(), ts_v.size()));
    }
    if (s.device) {
        s.device->copy_to_host(ts_e, ts_v);
        return;
    }
    std::ranges::copy(s.host_e, ts_e.begin());
    std::ranges::copy(s.host_v, ts_v.begin());
}

namespace detail {

std::pair<std::span<const float>, std::span<const float>>
host_series(const SearchSeries& series, std::string_view what) {
    const auto& s = loki::detail::HandleAccess::impl(series);
    if (s.exec.backend != Backend::kCPU || s.device) {
        throw std::invalid_argument(
            fmt::format("{}: expected a CPU SearchSeries", what));
    }
    return {s.host_e, s.host_v};
}

std::pair<DeviceSpan<const float>, DeviceSpan<const float>>
device_series(const SearchSeries& series, std::string_view what) {
    const auto& s = loki::detail::HandleAccess::impl(series);
    if (!s.device) {
        throw std::invalid_argument(
            fmt::format("{}: expected a GPU SearchSeries", what));
    }
    const Device dev{.backend = s.exec.backend, .id = s.exec.device};
    return {DeviceSpan<const float>(s.device->ts_e(), s.n, dev),
            DeviceSpan<const float>(s.device->ts_v(), s.n, dev)};
}

} // namespace detail

} // namespace loki::io
