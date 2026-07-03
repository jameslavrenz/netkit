#pragma once

#ifdef NETKIT_DISABLE_IOSTREAM
#define NETKIT_LOG(msg) ((void)0)
#else
#include <iostream>
#define NETKIT_LOG(msg)            \
    do                             \
    {                              \
        std::cout << msg;          \
    } while (0)
#endif
