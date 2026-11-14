#pragma once

#include <string>
#include <vector>

#include <H5Fpublic.h>
#include <H5Zpublic.h>
#include <H5public.h>
#include <highfive/highfive.hpp>

namespace loki::io::detail {

/// Access properties of every handle that writes search results, spools or
/// their run groups. Objects are written in the HDF5 1.10 format: with it a
/// compressed dataset of one chunk needs no chunk B-tree, and groups of many
/// small objects take less metadata. Readers need HDF5 1.10 or newer (the
/// build's minimum). Bounds act per open handle, so a reopen for writing
/// takes them too.
inline HighFive::FileAccessProps result_file_access() {
    HighFive::FileAccessProps fapl;
#if H5_VERSION_GE(1, 10, 2)
    fapl.add(HighFive::FileVersionBounds(H5F_LIBVER_V110, H5F_LIBVER_V110));
#else
    // 1.10.0 and 1.10.1 have no H5F_LIBVER_V110; their latest is 1.10.
    fapl.add(HighFive::FileVersionBounds(H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
#endif
    return fapl;
}

/// Opens a result, spool or run-group file with result_file_access().
inline HighFive::File open_result_file(const std::string& path,
                                       HighFive::File::AccessMode mode) {
    return HighFive::File(path, mode, result_file_access());
}

/// Creation properties of a small per-run table (for example level_stats):
/// one chunk, byte shuffle and deflate. Mostly small integers and zeros, it
/// compresses about sevenfold. Contiguous when deflate is unavailable or
/// the table is empty.
inline HighFive::DataSetCreateProps
small_table_props(const std::vector<std::size_t>& dims) {
    HighFive::DataSetCreateProps props;
    for (const auto d : dims) {
        if (d == 0) {
            return props;
        }
    }
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) {
        return props;
    }
    props.add(
        HighFive::Chunking(std::vector<hsize_t>(dims.begin(), dims.end())));
    props.add(HighFive::Shuffle());
    props.add(HighFive::Deflate(4));
    return props;
}

} // namespace loki::io::detail
