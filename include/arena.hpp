#pragma once
#include <cstddef>
#include <cstdint>

struct Arena {
    static constexpr std::size_t kDefaultCapacity = 64 * 1024; // 64 KiB

    std::byte* base{};
    std::size_t capacity = 0;
    std::size_t offset = 0;

    void init(void* memory, std::size_t size);
    // Returns nullptr when the arena is exhausted or not initialized.
    void* alloc(std::size_t size);
    void reset();
    std::size_t remaining() const;
};
