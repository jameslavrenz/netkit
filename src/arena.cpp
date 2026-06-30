#include "arena.hpp"

void Arena::init(void* memory, std::size_t size)
{
    base = static_cast<std::byte*>(memory);
    capacity = size;
    offset = 0;
}

void* Arena::alloc(std::size_t size)
{
    if (!base)
        return nullptr;

    if (size == 0)
        return base + offset;

    if (offset > capacity || size > capacity - offset)
        return nullptr;

    void* p = base + offset;
    offset += size;
    return p;
}

void Arena::reset()
{
    offset = 0;
}

std::size_t Arena::remaining() const
{
    if (!base || offset > capacity)
        return 0;
    return capacity - offset;
}
