#pragma once

#include <cstddef>
#include <cstdint>

/*
 * Shared kernel scratch allocated from the inference arena at model load.
 * CMSIS-NN conv / depthwise / GELU reuse this buffer during forward (TFLM-style).
 */
struct KernelWorkspace
{
    uint8_t* data = nullptr;
    std::size_t size_bytes = 0;
};

class KernelWorkspaceScope
{
public:
    explicit KernelWorkspaceScope(KernelWorkspace* workspace);
    ~KernelWorkspaceScope();

    KernelWorkspaceScope(const KernelWorkspaceScope&) = delete;
    KernelWorkspaceScope& operator=(const KernelWorkspaceScope&) = delete;

private:
    KernelWorkspace* previous_ = nullptr;
};

KernelWorkspace* GetActiveKernelWorkspace();

bool BindCmsisWorkspace(void*& buf, int32_t& size, int32_t required_bytes);
