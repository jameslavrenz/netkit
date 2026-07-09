/*
 * infer_cpp.cpp — C++26 example: load a .nk model and run inference
 *
 * Build: make example-cpp
 * Run:   ./examples/infer_cpp models/test_mlp.nk 1 2
 */
#include "arena.hpp"
#include "arena_util.hpp"
#include "cnn.hpp"
#include "nk_loader.hpp"
#include "mlp.hpp"
#include "netkit.h"
#include "tensor_factory.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <span>

namespace
{
    Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        const std::array<uint32_t, 3> shape{h, w, c};
        return TensorFactory::ViewND(data, 3, std::span<const uint32_t>(shape));
    }
}

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model.nk> <input floats...>\n";
        std::cerr << "Example: " << argv[0] << " models/test_mlp.nk 1 2\n";
        return 1;
    }

    const char* nk_path = argv[1];

    NkLoader::ParsedModel parsed{};
    const NkLoader::LoadResult arch_result = NkLoader::ParseFile(nk_path, parsed);
    if (arch_result.status != NkLoader::LoadStatus::Ok)
    {
        std::cerr << "parse failed: "
                  << (arch_result.message ? arch_result.message : NkLoader::StatusMessage(arch_result.status))
                  << "\n";
        return 1;
    }

    const uint32_t input_elements = NkLoader::InputElements(parsed);
    const uint32_t output_elements = NkLoader::OutputElements(parsed);
    const int input_arg_count = argc - 2;
    if (static_cast<uint32_t>(input_arg_count) != input_elements)
    {
        std::cerr << "expected " << input_elements << " input values, got " << input_arg_count << "\n";
        return 1;
    }

    if (input_elements > NK_MAX_CASE_FLOATS || output_elements > NK_MAX_CASE_FLOATS)
    {
        std::cerr << "model I/O exceeds example buffer limit (" << NK_MAX_CASE_FLOATS << " floats)\n";
        return 1;
    }

    float input_buffer[NK_MAX_CASE_FLOATS] = {};
    for (int i = 0; i < input_arg_count; ++i)
        input_buffer[i] = std::strtof(argv[i + 2], nullptr);

#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity);
#else
    alignas(std::max_align_t) static unsigned char arena_memory[Arena::kDefaultCapacity];
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity, arena_memory);
#endif
    if (!arena_scope)
    {
        std::cerr << "arena init failed\n";
        return 1;
    }
    Arena& arena = arena_scope.Get();

    std::cout << "Model: " << nk_path << "\n";

    if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
    {
        MLPNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const NkLoader::LoadResult load_result =
            NkLoader::LoadMLP(nk_path, arena, network, input_shape, input_rank);

        if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
        {
            std::cerr << "load failed: "
                      << (load_result.message ? load_result.message : NkLoader::StatusMessage(load_result.status))
                      << "\n";
            return 1;
        }

        Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
        float* input_data = static_cast<float*>(input.data);
        for (uint32_t i = 0; i < input_elements; ++i)
            input_data[i] = input_buffer[i];

        const uint32_t output_cols = NkLoader::OutputElements(parsed) / input_shape[0];
        Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);

        TensorFactory::PrintLabeled("Input", input);
        network->forward(input, output, arena);
        TensorFactory::PrintLabeled("Output", output);
    }
    else if (parsed.header.network_kind == NkFormat::NetworkKind::Cnn)
    {
        CNNNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const NkLoader::LoadResult load_result =
            NkLoader::LoadCNN(nk_path, arena, network, input_shape, input_rank);

        if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
        {
            std::cerr << "load failed: "
                      << (load_result.message ? load_result.message : NkLoader::StatusMessage(load_result.status))
                      << "\n";
            return 1;
        }

        Tensor input = MakeNhwcView(input_buffer, input_shape[0], input_shape[1], input_shape[2]);

        TensorFactory::PrintLabeled("Input", input);
        Tensor& output = network->forward(input, arena);
        if (!output.data)
        {
            std::cerr << "forward pass failed (arena overflow)\n";
            return 1;
        }

        TensorFactory::PrintLabeled("Output", output);
    }
    else
    {
        std::cerr << "unsupported network kind\n";
        return 1;
    }

    return 0;
}
