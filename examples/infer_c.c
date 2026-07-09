/*
 * infer_c.c — C23 example: load a .nk model and run inference via netkit.h
 *
 * Build: make example-c
 * Run:   ./examples/infer_c models/test_mlp.nk 1 2
 */
#include "netkit.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <model.nk> <input floats...>\n", argv[0]);
        fprintf(stderr, "Example: %s models/test_mlp.nk 1 2\n", argv[0]);
        return 1;
    }

    const char* nk_path = argv[1];
    const int input_arg_count = argc - 2;

    nk_arch_info_t arch = {0};
    const nk_status_t arch_status = nk_parse_architecture(nk_path, &arch);
    if (arch_status != NK_OK)
    {
        fprintf(stderr, "parse failed: %s (%s)\n", nk_status_string(arch_status), nk_last_error());
        return 1;
    }

    if ((uint32_t)input_arg_count != arch.input_elements)
    {
        fprintf(stderr, "expected %u input values, got %d\n", arch.input_elements, input_arg_count);
        return 1;
    }

    if (arch.input_elements > NK_MAX_CASE_FLOATS || arch.output_elements > NK_MAX_CASE_FLOATS)
    {
        fprintf(stderr, "model I/O exceeds example buffer limit (%u floats)\n", (unsigned)NK_MAX_CASE_FLOATS);
        return 1;
    }

    float input[NK_MAX_CASE_FLOATS];
    for (int i = 0; i < input_arg_count; ++i)
        input[i] = strtof(argv[i + 2], nullptr);

    nk_arena_t arena;
    int exit_code = 0;

#if defined(NETKIT_ARENA_HEAP)
    if (nk_arena_init_heap(&arena, NK_ARENA_DEFAULT_CAPACITY) != NK_OK)
    {
        fprintf(stderr, "arena init failed\n");
        return 1;
    }
#else
    alignas(max_align_t) static unsigned char arena_memory[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_init(&arena, arena_memory, sizeof(arena_memory));
#endif

    nk_model_t model;
    const nk_status_t load_status = nk_model_load(nk_path, &arena, &model);
    if (load_status != NK_OK)
    {
        fprintf(stderr, "load failed: %s (%s)\n", nk_status_string(load_status), nk_last_error());
        exit_code = 1;
        goto done;
    }

    float output[NK_MAX_CASE_FLOATS];
    uint32_t output_count = 0;
    const nk_status_t run_status = nk_model_run(&model,
                                                  &arena,
                                                  input,
                                                  arch.input_elements,
                                                  output,
                                                  (uint32_t)(sizeof(output) / sizeof(output[0])),
                                                  &output_count);
    if (run_status != NK_OK)
    {
        fprintf(stderr, "run failed: %s (%s)\n", nk_status_string(run_status), nk_last_error());
        exit_code = 1;
        goto done;
    }

    printf("Model: %s\n", nk_path);
    printf("Input (%u): ", arch.input_elements);
    for (uint32_t i = 0; i < arch.input_elements; ++i)
        printf("%s%.4f", i ? ", " : "", input[i]);
    printf("\n");

    printf("Output (%u): ", output_count);
    for (uint32_t i = 0; i < output_count; ++i)
        printf("%s%.4f", i ? ", " : "", output[i]);
    printf("\n");

done:
#if defined(NETKIT_ARENA_HEAP)
    nk_arena_destroy_heap(&arena);
#endif
    return exit_code;
}
