# Target environment: NETKIT_ARCH drives desktop vs embedded and ARM_MATH_* flags.

include(cmake/netkit_arch.cmake)

set(MCU_CORE "" CACHE STRING "Deprecated — use NETKIT_ARCH instead")

netkit_resolve_architecture()
