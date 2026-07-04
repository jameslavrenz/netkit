#include "kernel_workspace.hpp"

namespace
{
    KernelWorkspace* g_active_kernel_workspace = nullptr;
}

KernelWorkspaceScope::KernelWorkspaceScope(KernelWorkspace* workspace)
    : previous_(g_active_kernel_workspace)
{
    g_active_kernel_workspace = workspace;
}

KernelWorkspaceScope::~KernelWorkspaceScope()
{
    g_active_kernel_workspace = previous_;
}

KernelWorkspace* GetActiveKernelWorkspace()
{
    return g_active_kernel_workspace;
}

bool BindCmsisWorkspace(void*& buf, int32_t& size, int32_t required_bytes)
{
    buf = nullptr;
    size = 0;
    if (required_bytes <= 0)
        return true;

    KernelWorkspace* workspace = GetActiveKernelWorkspace();
    if (!workspace || !workspace->data ||
        static_cast<std::size_t>(required_bytes) > workspace->size_bytes)
        return false;

    buf = workspace->data;
    size = required_bytes;
    return true;
}
