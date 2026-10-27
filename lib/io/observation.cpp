#include "loki/io/observation.hpp"

#include <stdexcept>

#include "lib/detail/utils.hpp"

namespace loki::io {
void ObservationMetadata::validate() const {
    if ((dm && (!utils::is_finite(*dm) || *dm < 0)) ||
        (start_mjd && (!utils::is_finite(*start_mjd) || *start_mjd <= 0))) {
        throw std::invalid_argument(
            "observation DM must be finite and nonnegative; start MJD must be "
            "finite and positive");
    }
}
} // namespace loki::io
