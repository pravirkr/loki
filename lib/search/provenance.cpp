#include "loki/search/provenance.hpp"

#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include <highfive/H5File.hpp>
#include <highfive/highfive.hpp>

namespace loki::search {

void write_run_provenance(const std::filesystem::path& results_file,
                          const RunProvenance& provenance) {
    if (!std::filesystem::exists(results_file)) {
        throw std::runtime_error(
            std::format("write_run_provenance: results file not found: '{}'",
                        results_file.string()));
    }
    HighFive::File file(results_file.string(), HighFive::File::ReadWrite);
    const auto write = [&](std::string_view name, const std::string& value) {
        if (value.empty()) {
            return;
        }
        const std::string key(name);
        if (file.hasAttribute(key)) {
            file.deleteAttribute(key);
        }
        file.createAttribute(key, value);
    };
    write("config_effective", provenance.config_effective);
    write("command_line", provenance.command_line);
    write("loki_version", provenance.loki_version);
    file.flush();
}

} // namespace loki::search
