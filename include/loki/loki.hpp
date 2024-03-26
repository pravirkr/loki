#pragma once

#include <vector>

namespace loki {

/** @brief Helper function that returns a vector of given size and type
 *
 * @tparam T The type of element
 * @param size The size of the vector to return
 * @returns a vector of given size and type
 */
template <typename T>
std::vector<T> make_vector(std::size_t size);

}  // namespace loki
