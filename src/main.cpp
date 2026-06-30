#include "test.hpp"
#include <iostream>

int main()
{
    std::cout << std::unitbuf;

    const VectorsLoader::RunSummary summary = run_all_tests();

    std::cout << "\n============================\n";
    std::cout << " SUMMARY\n";
    std::cout << "============================\n";
    std::cout << "Passed: " << summary.passed << "\n";
    std::cout << "Failed: " << summary.failed << "\n";

    return summary.failed == 0 ? 0 : 1;
}
