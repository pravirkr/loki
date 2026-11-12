#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>

#include <fmt/format.h>

namespace loki::test {

/// A new directory under the system temp directory, removed with its
/// contents when this goes out of scope. Its name ends in a random suffix,
/// so tests running in parallel, and other users of a shared machine, never
/// meet in it. A fixed name under /tmp fails once another user owns it.
class TempDir {
public:
    explicit TempDir(std::string_view prefix) {
        std::random_device seed;
        std::mt19937_64 rng((static_cast<std::uint64_t>(seed()) << 32U) ^
                            seed());
        const auto base = std::filesystem::temp_directory_path();
        do {
            m_path = base / fmt::format("{}-{:016x}", prefix, rng());
        } while (!std::filesystem::create_directory(m_path));
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&)                 = delete;
    TempDir& operator=(TempDir&&)      = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return m_path; }
    [[nodiscard]] std::filesystem::path
    operator/(const std::filesystem::path& name) const {
        return m_path / name;
    }

private:
    std::filesystem::path m_path;
};

} // namespace loki::test
