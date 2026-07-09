#pragma once

#include "arena.hpp"
#include "netkit_config.h"

#include <cstddef>
#include <cstdint>

namespace ArenaUtil
{
    constexpr std::size_t kHandCapacity = 64 * 1024; // hand models; independent of target default
    constexpr std::size_t kMnistMlpCapacity = 2u * 1024u * 1024u;
    constexpr std::size_t kMnistCnnCapacity = 4u * 1024u * 1024u;
    constexpr std::size_t kLargeCnnCapacity = 32u * 1024u * 1024u;
    constexpr std::size_t kXLargeCnnCapacity = 64u * 1024u * 1024u;

    inline std::size_t CapacityForInputElements(uint32_t input_elements, bool is_cnn)
    {
        if (input_elements >= 784)
            return is_cnn ? kMnistCnnCapacity : kMnistMlpCapacity;
        return kHandCapacity;
    }

    inline std::size_t CapacityForModel(uint32_t input_elements,
                                        bool is_cnn,
                                        uint32_t weights_bytes,
                                        uint32_t biases_bytes)
    {
        const std::size_t payload =
            static_cast<std::size_t>(weights_bytes) + static_cast<std::size_t>(biases_bytes);
        if (is_cnn && payload > kLargeCnnCapacity)
            return kXLargeCnnCapacity;
        if (is_cnn && payload > kMnistCnnCapacity)
            return kLargeCnnCapacity;
        // Quantized UIB backbones (e.g. MobileNetV4 56×56×3) need large scratch even when
        // int8 weights are smaller than the float payload threshold above.
        if (is_cnn && input_elements >= 56u * 56u * 3u)
            return kXLargeCnnCapacity;
        return CapacityForInputElements(input_elements, is_cnn);
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
