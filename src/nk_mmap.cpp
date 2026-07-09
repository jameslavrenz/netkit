#include "nk_mmap.hpp"

#include <cerrno>
#include <cstring>

#if NETKIT_USE_MMAP && (defined(__APPLE__) || defined(__linux__))
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define NETKIT_HAS_POSIX_MMAP 1
#endif

namespace NkMmap
{
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
