#include "nk_regression.hpp"
#include <iostream>

NkRegression::RunSummary run_all_tests()
{
    NkRegression::RunSummary total{};

    NkRegression::BeginRegressionArena();

    auto merge = [&](const NkRegression::RunSummary& part) {
        total.passed += part.passed;
        total.failed += part.failed;
    };

    std::cout << "\n============================\n";
    std::cout << " C++ API TESTS\n";
    std::cout << "============================\n";

    std::cout << "\n============================\n";
    std::cout << " MLP TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/test_mlp.nk"));
    merge(NkRegression::RunModelTests("models/mlp_hand.nk"));

    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/test_cnn.nk"));
    merge(NkRegression::RunModelTests("models/cnn_4x4_single.nk"));
    merge(NkRegression::RunModelTests("models/cnn_hand.nk"));

    std::cout << "\n============================\n";
    std::cout << " SPEECH KWS TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/speech_kws.nk"));

    std::cout << "\n============================\n";
    std::cout << " CONVNEXT V2 ATTO BLOCK TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/convnextv2_atto_block.nk"));
    merge(NkRegression::RunModelTests("models/convnextv2_atto.nk"));

    std::cout << "\n============================\n";
    std::cout << " RESNET-18 BASIC BLOCK TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/resnet18_basic_block.nk"));

    std::cout << "\n============================\n";
    std::cout << " RESNET-18 TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/resnet18.nk"));

    std::cout << "\n============================\n";
    std::cout << " MOBILENETV4 SMALL UIB TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mobilenetv4_small_uib.nk"));

    std::cout << "\n============================\n";
    std::cout << " MOBILENETV4 CONV-SMALL TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mobilenetv4_small.nk"));

    std::cout << "\n============================\n";
    std::cout << " MNIST MLP TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mnist_mlp.nk"));

    std::cout << "\n============================\n";
    std::cout << " MNIST CNN TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mnist_cnn.nk"));

    std::cout << "\n============================\n";
    std::cout << " OP MATRIX TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/op_matrix_mlp.nk"));
    merge(NkRegression::RunModelTests("models/op_matrix_cnn.nk"));
    merge(NkRegression::RunModelTests("models/cnn_extended_ops.nk"));
    merge(NkRegression::RunModelTests("models/deep_mlp.nk"));

    std::cout << "\n============================\n";
    std::cout << " FASHION-MNIST MLP TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/fashion_mnist_mlp.nk"));

    std::cout << "\n============================\n";
    std::cout << " FASHION-MNIST CNN TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/fashion_mnist_cnn.nk"));

    NkRegression::EndRegressionArena();

    return total;
}
