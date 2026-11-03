#pragma once

#include <exception>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/common/backend.hpp"
#include "loki/common/interrupt.hpp"

namespace py = pybind11;

/// Python-facing backend selection: `backend="cpu" | "cuda"` and `device=`
/// are keyword-only on every binding. Unknown names raise ValueError.
inline loki::Exec
make_exec(std::string_view backend, int device = 0, int nthreads = 1) {
    const auto b = loki::parse_backend(backend);
    return {.backend = b, .nthreads = nthreads, .device = device};
}

/// Whether the caller runs on Python's main thread, the one that receives
/// SIGINT. Needs the GIL.
inline bool on_python_main_thread() {
    const auto threading = py::module_::import("threading");
    return threading.attr("current_thread")().is(
        threading.attr("main_thread")());
}

/**
 * Scope of a long call into the library: the GIL is released, so other
 * Python threads (Jupyter's, a progress display) keep running, and on the
 * main thread Ctrl-C stops the call at its next checkpoint with
 * KeyboardInterrupt (loki::Interrupted, translated in loki_python.cpp).
 * Nothing in the scope may touch a Python object: take spans of the
 * arrays before it and build the results after it. Also usable as
 * py::call_guard<LongCall>() for methods with C++ arguments and results.
 */
class LongCall {
public:
    LongCall() {
        if (on_python_main_thread()) {
            m_interrupt.emplace();
        }
        m_release.emplace();
    }

    ~LongCall() {
        m_release.reset(); // the GIL is back
        // A Ctrl-C that no checkpoint consumed (the call ended first, or had
        // none) is handed to Python, which raises KeyboardInterrupt next.
        if (m_interrupt && loki::interrupt_requested() &&
            std::uncaught_exceptions() == 0) {
            PyErr_SetInterrupt(); // NOLINT(misc-include-cleaner): Python.h
        }
    }
    LongCall(const LongCall&)            = delete;
    LongCall& operator=(const LongCall&) = delete;
    LongCall(LongCall&&)                 = delete;
    LongCall& operator=(LongCall&&)      = delete;

private:
    // Destroyed in reverse: the GIL is taken back before the signal
    // handlers are restored.
    std::optional<loki::InterruptGuard> m_interrupt;
    std::optional<py::gil_scoped_release> m_release;
};

/// Runs @p fn in a LongCall scope and returns its (C++) result.
template <typename Fn> decltype(auto) long_call(Fn&& fn) {
    const LongCall scope;
    return std::forward<Fn>(fn)();
}

template <typename T>
using PyArrayT = py::array_t<T, py::array::c_style | py::array::forcecast>;

// helper function to avoid making a copy when returning a py::array_t
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    const auto size = seq.size();
    auto data       = seq.data();
    std::unique_ptr<Sequence> seq_ptr =
        std::make_unique<Sequence>(std::forward<Sequence>(seq));
    const auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); // NOLINT
    });
    seq_ptr.release();
    return py::array({size}, {sizeof(typename Sequence::value_type)}, data,
                     capsule);
}

template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray_ref(const Sequence& seq) {
    const auto size  = seq.size();
    const auto* data = seq.data();
    return py::array_t<typename Sequence::value_type>(size, data);
}

template <typename T>
inline std::span<const T> to_span(const PyArrayT<T>& arr) {
    static_assert(!std::is_pointer_v<T>, "T must not be a pointer type");
    static_assert(!std::is_reference_v<T>, "T must not be a reference type");
    const py::buffer_info buffer = arr.request();
    return std::span<const T>(static_cast<const T*>(buffer.ptr), buffer.size);
}

template <typename T> inline std::span<T> to_span(PyArrayT<T>& arr) {
    static_assert(!std::is_pointer_v<T>, "T must not be a pointer type");
    static_assert(!std::is_reference_v<T>, "T must not be a reference type");
    const py::buffer_info buffer = arr.request();
    return std::span<T>(static_cast<T*>(buffer.ptr), buffer.size);
}

template <typename T>
inline py::list
as_listof_pyarray(const std::vector<std::vector<T>>& vec_of_vecs) {
    py::list result;
    for (const auto& inner : vec_of_vecs) {
        result.append(as_pyarray_ref(inner));
    }
    return result;
}

template <typename T>
inline py::list as_listof_pyarray(
    const std::vector<std::vector<std::vector<T>>>& vec_of_vec_of_vecs) {
    py::list result;
    for (const auto& inner : vec_of_vec_of_vecs) {
        result.append(as_listof_pyarray(inner));
    }
    return result;
}

// Simple RAII wrapper for stream redirection
class StreamRedirection {
public:
    StreamRedirection()
        : m_stdout_redirect(std::cout,
                            py::module_::import("sys").attr("stdout")),
          m_stderr_redirect(std::cerr,
                            py::module_::import("sys").attr("stderr")) {}

private:
    py::scoped_ostream_redirect m_stdout_redirect;
    py::scoped_estream_redirect m_stderr_redirect;
};
