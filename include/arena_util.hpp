#pragma once

#include "arena.hpp"
#include "netkit_config.h"

#include <cstddef>
#include <cstdint>

namespace ArenaUtil
{
    // Named aliases of the target default (MCU 64 KiB; CPU/MPU 64 MiB).
    constexpr std::size_t kHandCapacity = Arena::kDefaultCapacity;
    constexpr std::size_t kMnistMlpCapacity = Arena::kDefaultCapacity;
    constexpr std::size_t kMnistCnnCapacity = Arena::kDefaultCapacity;
    constexpr std::size_t kLargeCnnCapacity = Arena::kDefaultCapacity;
    constexpr std::size_t kXLargeCnnCapacity = Arena::kDefaultCapacity;

    inline std::size_t CapacityForInputElements(uint32_t /*input_elements*/, bool /*is_cnn*/)
    {
        return Arena::kDefaultCapacity;
    }

    inline std::size_t CapacityForModel(uint32_t /*input_elements*/,
                                        bool /*is_cnn*/,
                                        uint32_t /*weights_bytes*/,
                                        uint32_t /*biases_bytes*/)
    {
        return Arena::kDefaultCapacity;
    }

    // Heap backing: one malloc via init_heap(); never realloc. CPU may free via Release().
    inline bool Init(Arena& arena, std::size_t capacity, void* global_buffer = nullptr)
    {
#if defined(NETKIT_ARENA_HEAP)
        (void)global_buffer;
        return arena.init_heap(capacity);
#else
        if (!global_buffer || capacity == 0)
            return false;
        arena.init(global_buffer, capacity);
        return true;
#endif
    }

    // CPU only: release heap backing. No-op on MCU/MPU (heap is never freed there).
    inline void Release(Arena& arena)
    {
#if defined(NETKIT_ARENA_HEAP) && defined(NETKIT_TARGET_CPU)
        arena.destroy_heap();
#endif
    }

    class Scoped
    {
    public:
        Scoped(std::size_t capacity, void* global_buffer = nullptr)
        {
            ok_ = Init(arena_, capacity, global_buffer);
        }

        ~Scoped() { Release(arena_); }

        Scoped(const Scoped&) = delete;
        Scoped& operator=(const Scoped&) = delete;

        Arena& Get() { return arena_; }
        const Arena& Get() const { return arena_; }

        explicit operator bool() const { return ok_; }

    private:
        Arena arena_{};
        bool ok_{false};
    };
}
