#pragma once

#include "netkit_config.h"
#include <cstddef>
#include <cstdint>

/* POSIX mmap helpers when NETKIT_USE_MMAP=1 (macOS/Linux). MCU and RTOS/bare-metal
 * MPU builds leave NETKIT_USE_MMAP=0 and use flash buffer or fread fallback. */

namespace NkMmap
{
    struct Mapping
    {
        const uint8_t* data = nullptr;
        std::size_t size = 0;
        bool valid() const { return data != nullptr; }
    };

    /* Map an entire .nk file with MAP_PRIVATE. Pages stay file-backed until a
     * write (e.g. in-place BN fold) copy-on-writes that page into anonymous RAM.
     * The on-disk file is never modified. Returns false when mmap is disabled
     * or unavailable. Caller must eventually Unmap (Arena does this). */
    bool MapFile(const char* path, Mapping& out);
    void Unmap(Mapping& mapping);
    void Unmap(const void* data, std::size_t size);
}
