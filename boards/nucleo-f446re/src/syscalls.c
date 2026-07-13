#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

void* _sbrk(int incr)
{
    /* MCU policy: no heap. Refuse growth so malloc/new cannot succeed quietly. */
    (void)incr;
    errno = ENOMEM;
    return (void*)-1;
}

int _close(int fd)
{
    (void)fd;
    return -1;
}

int _fstat(int fd, struct stat* st)
{
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int fd)
{
    (void)fd;
    return 1;
}

int _lseek(int fd, int ptr, int dir)
{
    (void)fd;
    (void)ptr;
    (void)dir;
    return 0;
}

int _read(int fd, char* ptr, int len)
{
    (void)fd;
    (void)ptr;
    (void)len;
    return 0;
}

int _write(int fd, char* ptr, int len)
{
    (void)fd;
    (void)ptr;
  return len;
}

void _exit(int code)
{
    (void)code;
    for (;;)
    {
    }
}

void _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    for (;;)
    {
    }
}

int _getpid(void)
{
    return 1;
}
