#pragma once

/**
 * @file bind.hpp
 * @brief One bind_<submodule>() per libloki submodule. Each lives in
 * src/bindings/bind_<submodule>.cpp and uses the public headers only.
 */

#include <pybind11/pybind11.h>

namespace loki {

void bind_scores(pybind11::module_& m);
void bind_thresholds(pybind11::module_& m);
void bind_fold(pybind11::module_& m);
void bind_configs(pybind11::module_& m);
void bind_plans(pybind11::module_& m);
void bind_ffa(pybind11::module_& m);
void bind_psr_utils(pybind11::module_& m);
void bind_prune(pybind11::module_& m);
void bind_io(pybind11::module_& m);
void bind_simulation(pybind11::module_& m);

} // namespace loki
