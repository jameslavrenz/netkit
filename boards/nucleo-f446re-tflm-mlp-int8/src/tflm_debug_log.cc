#include "uart.h"

extern "C" void uart_debug_log(const char* message)
{
    if (message)
    {
        uart_write(message);
    }
}
