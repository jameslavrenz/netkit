ENTRY(Reset_Handler)

_estack = ORIGIN(RAM) + LENGTH(RAM);

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 16M
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 512K
}

SECTIONS
{
    .isr_vector :
    {
        KEEP(*(.isr_vector))
    } > FLASH

    .text :
    {
        *(.text*)
        *(.rodata*)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss :
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(8);
        _ebss = .;
        PROVIDE(end = .);
        PROVIDE(__bss_start__ = _sbss);
        PROVIDE(__bss_end__ = _ebss);
        PROVIDE(__end__ = end);
    } > RAM

    . = ALIGN(8);
    _heap_start = .;
    . = . + 0x1000;
    _heap_end = .;

    .stack (NOLOAD) :
    {
        . = ALIGN(8);
        . = . + 0x8000;
        . = ALIGN(8);
    } > RAM
}
