/*
 * MCU hard ban on dynamic allocation.
 *
 * Providing non-throwing aborting operator new/new[] prevents libstdc++ from
 * pulling the throwing new + __cxa_demangle (~33 KiB) path that previously
 * dominated netkit flash vs TFLM. malloc/calloc/realloc abort the same way;
 * _sbrk in syscalls.c refuses heap growth.
 *
 * Placement new (arena / network objects) uses the inline operator new(size_t,
 * void*) overload and does not call these.
 */
#include <cstddef>
#include <cstdlib>
#include <new>

[[noreturn]] static void NetkitMcuHeapAbort()
{
    for (;;)
    {
    }
}

__attribute__((used, noinline)) void* operator new(std::size_t)
{
    NetkitMcuHeapAbort();
}

__attribute__((used, noinline)) void* operator new[](std::size_t)
{
    NetkitMcuHeapAbort();
}

__attribute__((used, noinline)) void* operator new(std::size_t, const std::nothrow_t&) noexcept
{
    NetkitMcuHeapAbort();
}

__attribute__((used, noinline)) void* operator new[](std::size_t, const std::nothrow_t&) noexcept
{
    NetkitMcuHeapAbort();
}

__attribute__((used)) void operator delete(void*) noexcept {}
__attribute__((used)) void operator delete[](void*) noexcept {}
__attribute__((used)) void operator delete(void*, std::size_t) noexcept {}
__attribute__((used)) void operator delete[](void*, std::size_t) noexcept {}
__attribute__((used)) void operator delete(void*, const std::nothrow_t&) noexcept {}
__attribute__((used)) void operator delete[](void*, const std::nothrow_t&) noexcept {}

extern "C" {

__attribute__((used, noinline)) void* malloc(size_t)
{
    NetkitMcuHeapAbort();
}

__attribute__((used, noinline)) void* calloc(size_t, size_t)
{
    NetkitMcuHeapAbort();
}

__attribute__((used, noinline)) void* realloc(void*, size_t)
{
    NetkitMcuHeapAbort();
}

__attribute__((used)) void free(void*) {}

} // extern "C"
