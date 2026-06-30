#pragma once
#include <cstdint>

namespace VectorsLoader
{
    constexpr uint32_t kMaxCases = 8;
    constexpr uint32_t kMaxFloats = 256;

    struct RunSummary
    {
        uint32_t passed = 0;
        uint32_t failed = 0;
    };

    // Load model from vectors file, run all cases, print results.
    RunSummary RunVectorsFile(const char* vectors_path);
}
