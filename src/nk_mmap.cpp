#include "nk_mmap.hpp"

#include <cerrno>
#include <cstring>

#if NETKIT_USE_MMAP && (defined(__APPLE__) || defined(__linux__))
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define NETKIT_HAS_POSIX_MMAP 1
#elif NETKIT_USE_MMAP && defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define NETKIT_HAS_WIN32_MMAP 1
#endif

namespace NkMmap
{
#if defined(NETKIT_HAS_WIN32_MMAP)
    namespace
    {
        bool Utf8ToWide(const char* utf8, wchar_t* wide, int wide_chars)
        {
            if (!utf8 || !wide || wide_chars <= 0)
                return false;
            const int n = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, wide, wide_chars);
            return n > 0;
        }
    }
#endif

    bool MapFile(const char* path, Mapping& out)
    {
        out = {};
        if (!path || !path[0])
            return false;

#if defined(NETKIT_HAS_POSIX_MMAP)
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0)
            return false;

        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < 0)
        {
            ::close(fd);
            return false;
        }

        const std::size_t size = static_cast<std::size_t>(st.st_size);
        if (size == 0)
        {
            ::close(fd);
            out.data = nullptr;
            out.size = 0;
            return true;
        }

        /* MAP_PRIVATE + PROT_WRITE: pages stay file-backed until a write
         * (e.g. in-place BN fold) copy-on-writes that page. Disk is never modified. */
        void* mapped = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mapped == MAP_FAILED)
            return false;

#if defined(MADV_WILLNEED)
        (void)::madvise(mapped, size, MADV_WILLNEED);
#endif

        out.data = static_cast<const uint8_t*>(mapped);
        out.size = size;
        return true;

#elif defined(NETKIT_HAS_WIN32_MMAP)
        /* FILE_MAP_COPY + PAGE_WRITECOPY ≈ POSIX MAP_PRIVATE: writes COW into
         * private pages; the on-disk .nk is never modified. */
        wchar_t wpath[MAX_PATH * 4];
        if (!Utf8ToWide(path, wpath, static_cast<int>(sizeof(wpath) / sizeof(wpath[0]))))
            return false;

        HANDLE file = ::CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return false;

        LARGE_INTEGER file_size {};
        if (!::GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0)
        {
            ::CloseHandle(file);
            return false;
        }

        const std::size_t size = static_cast<std::size_t>(file_size.QuadPart);
        if (size == 0)
        {
            ::CloseHandle(file);
            out.data = nullptr;
            out.size = 0;
            return true;
        }

        HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_WRITECOPY, 0, 0, nullptr);
        ::CloseHandle(file);
        if (!mapping)
            return false;

        void* mapped = ::MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, 0);
        ::CloseHandle(mapping);
        if (!mapped)
            return false;

        out.data = static_cast<const uint8_t*>(mapped);
        out.size = size;
        return true;

#else
        (void)path;
        (void)out;
        return false;
#endif
    }

    void Unmap(const void* data, std::size_t size)
    {
#if defined(NETKIT_HAS_POSIX_MMAP)
        if (!data || size == 0)
            return;
        (void)::munmap(const_cast<void*>(data), size);
#elif defined(NETKIT_HAS_WIN32_MMAP)
        (void)size;
        if (!data)
            return;
        (void)::UnmapViewOfFile(data);
#else
        (void)data;
        (void)size;
#endif
    }

    void Unmap(Mapping& mapping)
    {
        Unmap(mapping.data, mapping.size);
        mapping = {};
    }
}
