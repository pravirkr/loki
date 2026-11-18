#pragma once

#include <memory>
#include <span>
#include <utility>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/utils/handle_access.hpp"

namespace loki::io {

/**
 * @brief The folding inputs of a search (`ts_e`, `ts_v`), held on one
 * backend: host memory on the CPU, device memory on CUDA.
 *
 * A CUDA search preprocesses its raw series on the device and passes the
 * result to the sweep (`EPFreqSweep::execute`, `FFAFreqSweep::execute`)
 * without copying it back to the host. On the CPU the handle simply owns
 * the two arrays.
 */
class SearchSeries {
public:
    /// Empty handle.
    SearchSeries() noexcept;
    ~SearchSeries();
    SearchSeries(SearchSeries&&) noexcept;
    SearchSeries& operator=(SearchSeries&&) noexcept;
    SearchSeries(const SearchSeries&)            = delete;
    SearchSeries& operator=(const SearchSeries&) = delete;

    /**
     * @brief Preprocesses @p raw (io::preprocess) on @p exec's backend and
     * keeps the result there.
     *
     * On CUDA the raw series is uploaded once and the robust method runs on
     * the device (the z-score method runs on the host and its result is
     * uploaded). The report is the same as preprocess() gives.
     */
    [[nodiscard]] static std::pair<SearchSeries, PreprocessReport>
    preprocess(std::span<const float> raw,
               double tsamp,
               const PreprocessOptions& options = {},
               Exec exec                        = {});

    /// Folding inputs that are already built, copied to @p exec's backend.
    [[nodiscard]] static SearchSeries from_host(std::span<const float> ts_e,
                                                std::span<const float> ts_v,
                                                double tsamp,
                                                Exec exec = {});

    [[nodiscard]] SizeType size() const noexcept;
    [[nodiscard]] double tsamp() const noexcept;
    /// Backend and device that hold the series.
    [[nodiscard]] Exec exec() const;
    [[nodiscard]] bool empty() const noexcept { return m_impl == nullptr; }

    /// Copies `ts_e` and `ts_v` into @p ts_e and @p ts_v (each size()).
    void copy_to_host(std::span<float> ts_e, std::span<float> ts_v) const;

private:
    friend struct loki::detail::HandleAccess;

    /// Backend storage. Defined inside the library only.
    class Impl;
    [[nodiscard]] Impl& impl();
    [[nodiscard]] const Impl& impl() const;

    explicit SearchSeries(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::io
