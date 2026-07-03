.syntax unified
.cpu cortex-m4
.fpu fpv4-sp-d16
.thumb

.section .isr_vector, "a"
.global g_pfnVectors
g_pfnVectors:
.word Reset_Handler
.word NMI_Handler
.word HardFault_Handler
.word MemManage_Handler
.word BusFault_Handler
.word UsageFault_Handler
.word 0
.word 0
.word 0
.word 0
.word SVC_Handler
.word DebugMon_Handler
.word 0
.word PendSV_Handler
.word SysTick_Handler

.text
.global Reset_Handler
Reset_Handler:
    ldr r0, =_estack
    mov sp, r0
    ldr r0, =_sidata
    ldr r1, =_sdata
    ldr r2, =_edata
    movs r3, #0
    b LoopCopyDataInit

CopyDataInit:
    ldr r4, [r0, r3]
    str r4, [r1, r3]
    adds r3, r3, #4

LoopCopyDataInit:
    adds r4, r1, r3
    cmp r4, r2
    bcc CopyDataInit

    ldr r2, =_sbss
    ldr r4, =_ebss
    movs r3, #0
    b LoopFillZeroBss

FillZeroBss:
    str r3, [r2]
    adds r2, r2, #4

LoopFillZeroBss:
    cmp r2, r4
    bcc FillZeroBss

    bl main
    b .

.thumb_func
.weak NMI_Handler
.thumb_func
.weak HardFault_Handler
.thumb_func
.weak MemManage_Handler
.thumb_func
.weak BusFault_Handler
.thumb_func
.weak UsageFault_Handler
.thumb_func
.weak SVC_Handler
.thumb_func
.weak DebugMon_Handler
.thumb_func
.weak PendSV_Handler
.thumb_func
.weak SysTick_Handler
NMI_Handler:
HardFault_Handler:
MemManage_Handler:
BusFault_Handler:
UsageFault_Handler:
SVC_Handler:
DebugMon_Handler:
PendSV_Handler:
SysTick_Handler:
    b .
