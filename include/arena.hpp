#pragma once
#include "netkit_config.h"
#include <cstddef>
#include <cstdint>

struct Arena {
#if defined(NETKIT_TARGET_MCU)
    static constexpr std::size_t kDefaultCapacity = 64 * 1024; // 64 KiB
#else
    static constexpr std::size_t kDefaultCapacity = 64 * 1024 * 1024; // 64 MiB (CPU and MPU)
#endif

    std::byte* base{};
    std::size_t capacity = 0;
    std::size_t offset = 0;
    bool heap_owned = false;
    /* Optional mmap of a .nk file when NETKIT_USE_MMAP=1 (CPU default; opt-in
     * Linux MPU). Released on reset(), init(), and destroy_heap(). */
    const void* mapped_file = nullptr;
    std::size_t mapped_size = 0;

    void init(void* memory, std::size_t size);
#if defined(NETKIT_ARENA_HEAP)
    // Heap-backed backing store (malloc once). CPU may free via destroy_heap(); MCU/MPU never free.
    bool init_heap(std::size_t size);
    void destroy_heap();
#endif
    // Returns nullptr when the arena is exhausted, not initialized, or alignment is invalid.
    // alignment must be a power of two. Padding is inserted before the block when needed.
    void* alloc(std::size_t size, std::size_t alignment);
    void reset();
    std::size_t remaining() const;

    /* Take ownership of a POSIX mmap region. Releases any previous mapping.
     * Does not consume arena bump capacity. */
    void attach_mapped_file(const void* data, std::size_t size);
    void release_mapped_file();
};
