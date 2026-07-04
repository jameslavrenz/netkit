/*
 * nk_infer — run one forward pass; print outputs as comma-separated float32 (full precision).
 *
 * Usage: nk_infer <model.nk> <input floats...>
 * Build: make nk_infer (cpu only)
 */
#include "netkit.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <model.nk> <input floats...>\n", argv[0]);
        return 1;
    }

    const char* nk_path = argv[1];
    const int input_arg_count = argc - 2;

    nk_arch_info_t arch = {0};
    if (nk_parse_architecture(nk_path, &arch) != NK_OK)
        return 1;

    if ((uint32_t)input_arg_count != arch.input_elements)
        return 1;

    float input[NK_MAX_CASE_FLOATS];
    for (int i = 0; i < input_arg_count; ++i)
        input[i] = strtof(argv[i + 2], NULL);

#if defined(NETKIT_ARENA_HEAP)
    const size_t arena_capacity = nk_recommended_arena_bytes(nk_path);
    if (arena_capacity == 0)
        return 1;

    nk_arena_t arena;
    if (nk_arena_init_heap(&arena, arena_capacity) != NK_OK)
        return 1;
#else
    alignas(max_align_t) static unsigned char arena_memory[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
    nk_arena_init(&arena, arena_memory, sizeof(arena_memory));
#endif

    nk_model_t model;
    if (nk_model_load(nk_path, &arena, &model) != NK_OK)
    {
#if defined(NETKIT_ARENA_HEAP)
        nk_arena_destroy_heap(&arena);
#endif
        return 1;
    }

    float output[NK_MAX_CASE_FLOATS];
    uint32_t output_count = 0;
    if (nk_model_run(&model, &arena, input, arch.input_elements, output,
                     (uint32_t)(sizeof(output) / sizeof(output[0])), &output_count) != NK_OK)
    {
#if defined(NETKIT_ARENA_HEAP)
        nk_arena_destroy_heap(&arena);
#endif
        return 1;
    }

    for (uint32_t i = 0; i < output_count; ++i)
    {
        if (i > 0)
            putchar(',');
        printf("%.9g", output[i]);
    }
    putchar('\n');

#if defined(NETKIT_ARENA_HEAP)
    nk_arena_destroy_heap(&arena);
#endif
    return 0;
}
