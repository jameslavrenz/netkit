#pragma once

#include "netkit_config.h"
#include <cstddef>
#include <cstdint>

/* mmap helpers when NETKIT_USE_MMAP=1 (POSIX on macOS/Linux; Win32 on Windows).
 * Default on for cpu + any MPU; forbidden on MCU. Opt out with NETKIT_MMAP=0 on
 * RTOS/bare-metal MPU. */

namespace NkMmap
{
    struct Mapping
    {
        const uint8_t* data = nullptr;
        std::size_t size = 0;
        bool valid() const { return data != nullptr; }
    };

    /* Map an entire .nk file privately (POSIX MAP_PRIVATE / Win32 FILE_MAP_COPY).
     * Pages stay file-backed until a write (e.g. in-place BN fold) copy-on-writes
     * that page into anonymous RAM. The on-disk file is never modified. Returns
     * false when mmap is disabled or unavailable. Caller must eventually Unmap
     * (Arena does this). */
    bool MapFile(const char* path, Mapping& out);
    void Unmap(Mapping& mapping);
    void Unmap(const void* data, std::size_t size);
}
