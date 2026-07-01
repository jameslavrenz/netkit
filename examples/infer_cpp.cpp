/*
 * infer_cpp.cpp — C++26 example: load a model and run inference via the native API
 *
 * Build: make example-cpp
 * Run:   ./examples/infer_cpp models/test_mlp.json 1 2
 */
#include "arena.hpp"
#include "cnn.hpp"
#include "model_loader.hpp"
#include "mlp.hpp"
#include "tensor_factory.hpp"
#include <array>
#include <cstdlib>
#include <iostream>

namespace
{
    uint32_t InputElementCount(const ModelLoader::ArchitectureSpec& spec)
    {
        uint32_t count = 1;
        for (uint32_t i = 0; i < spec.input_rank; ++i)
            count *= spec.input_shape[i];
        return count;
    }

    Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        Tensor input{};
        input.data = data;
        input.type = DataType::Float32;
        input.rank = 3;
        input.shape[0] = h;
        input.shape[1] = w;
        input.shape[2] = c;
        input.stride[0] = w * c;
        input.stride[1] = c;
        input.stride[2] = 1;
        input.num_elements = h * w * c;
        input.bytes = input.num_elements * sizeof(float);
        return input;
    }
}

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model.json> <input floats...>\n";
        std::cerr << "Example: " << argv[0] << " models/test_mlp.json 1 2\n";
        return 1;
    }

    const char* json_path = argv[1];

    ModelLoader::ArchitectureSpec spec{};
    const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(json_path, spec);
    if (arch_result.status != ModelLoader::LoadStatus::Ok)
    {
        std::cerr << "parse failed: "
                  << (arch_result.message ? arch_result.message : "unknown error") << "\n";
        return 1;
    }

    const uint32_t input_elements = InputElementCount(spec);
    const int input_arg_count = argc - 2;
    if (static_cast<uint32_t>(input_arg_count) != input_elements)
    {
        std::cerr << "expected " << input_elements << " input values, got " << input_arg_count << "\n";
        return 1;
    }

    if (input_elements > 4096)
    {
        std::cerr << "input too large for example buffer\n";
        return 1;
    }

    float input_buffer[4096] = {};
    for (int i = 0; i < input_arg_count; ++i)
        input_buffer[i] = std::strtof(argv[i + 2], nullptr);

    alignas(std::max_align_t) static unsigned char arena_memory[Arena::kDefaultCapacity];
    Arena arena;
    arena.init(arena_memory, sizeof(arena_memory));

    std::cout << "Model: " << json_path << "\n";

    if (spec.kind == ModelLoader::NetworkKind::MLP)
    {
        MLPNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadMLP(json_path, arena, network, input_shape, input_rank);

        if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
        {
            std::cerr << "load failed: "
                      << (load_result.message ? load_result.message : "unknown error") << "\n";
            return 1;
        }

        Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
        float* input_data = static_cast<float*>(input.data);
        for (uint32_t i = 0; i < input_elements; ++i)
            input_data[i] = input_buffer[i];

        const uint32_t output_cols = spec.dense_layers[spec.num_layers - 1].units;
        Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);

        TensorFactory::PrintLabeled("Input", input);
        network->forward(input, output, arena);
        TensorFactory::PrintLabeled("Output", output);
    }
    else if (spec.kind == ModelLoader::NetworkKind::CNN)
    {
        CNNNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadCNN(json_path, arena, network, input_shape, input_rank);

        if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
        {
            std::cerr << "load failed: "
                      << (load_result.message ? load_result.message : "unknown error") << "\n";
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
