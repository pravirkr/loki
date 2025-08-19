#pragma once

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>

namespace py = pybind11;

template <typename T>
using PyArrayT = py::array_t<T, py::array::c_style | py::array::forcecast>;

// helper function to avoid making a copy when returning a py::array_t
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr =
        std::make_unique<Sequence>(std::forward<Sequence>(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); // NOLINT
    });
    seq_ptr.release();
    return py::array({size}, {sizeof(typename Sequence::value_type)}, data,
                     capsule);
}

template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray_ref(const Sequence& seq) {
    auto size        = seq.size();
    const auto* data = seq.data();
    return py::array_t<typename Sequence::value_type>(size, data);
}

template <typename T>
inline std::span<const T> to_span(const PyArrayT<T>& arr) {
    static_assert(!std::is_pointer_v<T>, "T must not be a pointer type");
    static_assert(!std::is_reference_v<T>, "T must not be a reference type");
    py::buffer_info buffer = arr.request();
    return std::span<const T>(static_cast<const T*>(buffer.ptr), buffer.size);
}

template <typename T> inline std::span<T> to_span(PyArrayT<T>& arr) {
    static_assert(!std::is_pointer_v<T>, "T must not be a pointer type");
    static_assert(!std::is_reference_v<T>, "T must not be a reference type");
    py::buffer_info buffer = arr.request();
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
