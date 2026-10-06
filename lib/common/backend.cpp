#include "loki/common/backend.hpp"

#include <algorithm>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/spdlog.h>

#include "lib/common/dispatch.hpp"

namespace loki {

std::string_view to_string(Backend backend) noexcept {
    switch (backend) {
    case Backend::kCPU:
        return "cpu";
    case Backend::kCUDA:
        return "cuda";
    }
    return "unknown";
}

Backend parse_backend(std::string_view name) {
    for (const auto b : {Backend::kCPU, Backend::kCUDA}) {
        if (name == to_string(b)) {
            return b;
        }
    }
    throw std::invalid_argument(
        std::format("Unknown backend '{}'. Expected 'cpu' or 'cuda'", name));
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out{Backend::kCPU};
#ifdef LOKI_ENABLE_GPU
    out.push_back(detail::kGPUBackend);
#endif
    return out;
}

bool is_available(Backend backend) {
    const auto avail = available_backends();
    return std::ranges::find(avail, backend) != avail.end();
}

} // namespace loki

namespace loki::detail {

namespace {

std::string available_names() {
    std::string names;
    for (const auto b : available_backends()) {
        names += names.empty() ? "" : ", ";
        names += to_string(b);
    }
    return names;
}

} // namespace

void throw_no_device_memory(std::string_view what, Backend backend) {
    throw std::invalid_argument(
        std::format("{}: device memory is not supported by the {} backend; "
                    "pass host memory (std::span) instead",
                    what, to_string(backend)));
}

void throw_unavailable(std::string_view algorithm, Backend backend) {
    throw std::invalid_argument(
        std::format("{}: backend '{}' is not available in this build "
                    "(available: {})",
                    algorithm, to_string(backend), available_names()));
}

void throw_unimplemented(std::string_view algorithm, Backend backend) {
    throw std::invalid_argument(
        std::format("{}: backend '{}' is built but not implemented for this "
                    "algorithm yet",
                    algorithm, to_string(backend)));
}

void check_device(const Device& view,
                  Backend backend,
                  int device,
                  std::string_view what) {
    if (view.id < 0) {
        return;
    }
    if (view.backend != backend || view.id != device) {
        throw std::invalid_argument(std::format(
            "{}: memory is on {}:{} but this instance runs on {}:{}", what,
            to_string(view.backend), view.id, to_string(backend), device));
    }
}

Device common_device(std::initializer_list<Device> views,
                     std::string_view what) {
    Device common{};
    for (const auto& view : views) {
        if (view.id < 0) {
            continue;
        }
        if (common.id < 0) {
            common = view;
        } else if (view.backend != common.backend || view.id != common.id) {
            throw std::invalid_argument(std::format(
                "{}: device views are on different devices ({}:{} and {}:{})",
                what, to_string(common.backend), common.id,
                to_string(view.backend), view.id));
        }
    }
    return common;
}

void check_same_exec(const Exec& resource,
                     const Exec& algorithm,
                     std::string_view what) {
    const bool same_device = resource.backend == Backend::kCPU ||
                             resource.device == algorithm.device;
    if (resource.backend != algorithm.backend || !same_device) {
        throw std::invalid_argument(std::format(
            "{}: shared resource was built for {}:{} but the algorithm runs "
            "on {}:{}",
            what, to_string(resource.backend), resource.device,
            to_string(algorithm.backend), algorithm.device));
    }
}

void warn_ignored_nthreads(const Exec& exec, std::string_view algorithm) {
    if (exec.backend == Backend::kCPU && exec.nthreads != Exec{}.nthreads) {
        spdlog::warn("{}: Exec.nthreads={} is ignored; the thread count comes "
                     "from the search config",
                     algorithm, exec.nthreads);
    }
}

} // namespace loki::detail
