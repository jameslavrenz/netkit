/* Bare-metal platform stubs for microTVM CRT (unpacked AOT). */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/error_codes.h"
#include "uart.h"

#ifdef __cplusplus
extern "C" {
#endif

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code)
{
    uart_printf("TVMPlatformAbort: %d\r\n", (int)error_code);
    for (;;)
    {
    }
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)
{
    (void)num_bytes;
    (void)dev;
    (void)out_ptr;
    return kTvmErrorFunctionCallNotImplemented;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)
{
    (void)ptr;
    (void)dev;
    return kTvmErrorFunctionCallNotImplemented;
}

void TVMLogf(const char* msg, ...)
{
    char buf[192];
    va_list args;
    va_start(args, msg);
    vsnprintf(buf, sizeof(buf), msg, args);
    va_end(args);
    uart_write(buf);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override)
{
    (void)name;
    (void)f;
    (void)override;
    return 0;
}

#ifdef __cplusplus
}
#endif
