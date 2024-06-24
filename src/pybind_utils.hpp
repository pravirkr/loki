#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    return py::array(size, data, capsule);
}

template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray_ref(const Sequence& seq) {
    auto size        = seq.size();
    const auto* data = seq.data();
    return py::array_t<typename Sequence::value_type>(size, data);
}