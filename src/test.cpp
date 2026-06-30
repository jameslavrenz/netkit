#include "vectors_loader.hpp"
#include <iostream>

static VectorsLoader::RunSummary RunSuite(const char* vectors_path)
{
    return VectorsLoader::RunVectorsFile(vectors_path);
}

void test_mlp()
{
    std::cout << "\n============================\n";
    std::cout << " MLP TESTS\n";
    std::cout << "============================\n";
    RunSuite("models/test_mlp.vectors.json");
    RunSuite("models/mlp_hand.vectors.json");
}

void test_cnn()
{
    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";
    RunSuite("models/test_cnn.vectors.json");
    RunSuite("models/cnn_4x4_single.vectors.json");
    RunSuite("models/cnn_hand.vectors.json");
}

VectorsLoader::RunSummary run_all_tests()
{
    VectorsLoader::RunSummary total{};

    auto merge = [&](const VectorsLoader::RunSummary& part) {
        total.passed += part.passed;
        total.failed += part.failed;
    };

    std::cout << "\n============================\n";
    std::cout << " MLP TESTS\n";
    std::cout << "============================\n";
    merge(RunSuite("models/test_mlp.vectors.json"));
    merge(RunSuite("models/mlp_hand.vectors.json"));

    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";
    merge(RunSuite("models/test_cnn.vectors.json"));
    merge(RunSuite("models/cnn_4x4_single.vectors.json"));
    merge(RunSuite("models/cnn_hand.vectors.json"));

    return total;
}
