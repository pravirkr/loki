#include "loki/common/terminal.hpp"

#include <cstdlib>
#include <string_view>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace loki {

bool stdout_is_terminal() noexcept {
#ifdef _WIN32
    return false;
#else
    if (isatty(STDOUT_FILENO) == 0) {
        return false;
    }
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read only
    const char* term = std::getenv("TERM");
    return term == nullptr || std::string_view(term) != "dumb";
#endif
}

bool color_allowed() noexcept {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read only
    const char* no_color = std::getenv("NO_COLOR");
    return no_color == nullptr || no_color[0] == '\0';
}

bool color_forced() noexcept {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read only
    const char* force = std::getenv("CLICOLOR_FORCE");
    return force != nullptr && force[0] != '\0' &&
           std::string_view(force) != "0";
}

} // namespace loki
