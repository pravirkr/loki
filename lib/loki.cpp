#include "loki/loki.hpp"

#include <numeric>

namespace loki {

template <typename T>
std::vector<T> make_vector(std::size_t size) {
  std::vector<T> vec(size);
  std::iota(vec.begin(), vec.end(), 0);
  return vec;
}

}  // namespace loki