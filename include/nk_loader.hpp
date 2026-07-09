#pragma once

#include "arena.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "nk_format.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace NkLoader
{
    constexpr std::size_t kMaxPathLen = 256;

    enum class NetworkKind : uint8_t
    {
        Unknown = 0,
        Mlp = 1,
        Cnn = 2
    };

    enum class LoadStatus
    {
        Ok,
        FileOpenFailed,
        ReadFailed,
        InvalidMagic,
        UnsupportedVersion,
        TruncatedFile,
        UnsupportedLayer,
        SizeMismatch,
        ArenaOverflow
    };

    struct ArchInfo
    {
        uint32_t version = 0;
        NetworkKind kind = NetworkKind::Unknown;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;
        uint32_t num_layers = 0;
        uint32_t input_elements = 0;
        uint32_t output_elements = 0;
        std::size_t weight_floats = 0;
    };

    struct ParsedModel
    {
        NkFormat::FileHeader header{};
        NkFormat::LayerDesc layers[NkFormat::kMaxLayers]{};
        NkFormat::TensorDesc weight_tensors[NkFormat::kMaxTensorCatalog]{};
        NkFormat::TensorDesc bias_tensors[NkFormat::kMaxTensorCatalog]{};
        bool has_quant = false;
        uint32_t num_quant_layers = 0;
        NkFormat::MlpLayerQuantDesc layer_quant[NkFormat::kMaxLayers]{};
        // Heap-owned per-channel weight scales (optional QUAN flag). ParseFile
        // allocates; LoadCNN relocates into the Arena and clears this pointer.
        // Call FreeParsedModelExtras (or delete[] the blob) if you only ParseFile.
        float* weight_channel_scale_blob = nullptr;
        std::size_t weight_channel_scale_floats = 0;
        std::size_t payload_offset = 0;
    };

    struct TestCase
    {
        char name[NkFormat::kMaxCaseNameLen + 1]{};
        int32_t label = NkFormat::kNoLabel;
        uint32_t input_count = 0;
        uint32_t output_count = 0;
        // Float models: float32 inputs. Quantized models (kFlagHasInt8Tests):
        // native int8 in input_i8 (Python-prequantized; no C++ float→int8).
        float input[NkFormat::kMaxCaseFloats]{};
        int8_t input_i8[NkFormat::kMaxCaseFloats]{};
        float expected[NkFormat::kMaxCaseFloats]{};
    };

    struct TestSuite
    {
        float tolerance = 1e-5f;
        uint32_t num_cases = 0;
        bool inputs_are_int8 = false;
        TestCase cases[NkFormat::kMaxTestCases]{};
    };

    struct LoadResult
    {
        LoadStatus status = LoadStatus::Ok;
        NetworkKind kind = NetworkKind::Unknown;
        const char* message = nullptr;
    };

    LoadResult ParseFile(const char* nk_path, ParsedModel& out);
    LoadResult ParseBuffer(const uint8_t* data, std::size_t size, ParsedModel& out);
    // Frees heap extras owned by ParsedModel (per-channel scale blob). Safe no-op
    // after LoadCNN, which relocates scales into the Arena.
    void FreeParsedModelExtras(ParsedModel& parsed);
    LoadResult ReadTestSuite(const char* nk_path, TestSuite& out);
    std::size_t ModelPayloadBytes(const ParsedModel& model);
    void FillArchInfo(const ParsedModel& model, ArchInfo& info);
    uint32_t InputElements(const ParsedModel& model);
    uint32_t OutputElements(const ParsedModel& model);

    void PrintNetworkSummary(const char* nk_path, const ParsedModel& model);

    /* File load: when NETKIT_USE_MMAP=1 (CPU default on macOS/Linux; opt-in on
       embedded Linux MPU), mmap MAP_PRIVATE and arena owns the mapping until
       reset()/destroy. Otherwise fread into the arena. Prefer Load*FromBuffer
       / flash for MCU and RTOS/bare-metal MPU. */
    LoadResult LoadMLP(const char* nk_path,
                       Arena& arena,
                       MLPNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank);

    LoadResult LoadMLPFromBuffer(const uint8_t* data,
                                 std::size_t size,
                                 Arena& arena,
                                 MLPNetwork*& network,
                                 std::array<uint32_t, kMaxTensorRank>& input_shape,
                                 uint32_t& input_rank);
    /* Weights stay in the blob: `data` must outlive the network (flash .rodata
       or caller-owned buffer). Misaligned payloads return SizeMismatch. */

    LoadResult LoadCNN(const char* nk_path,
                       Arena& arena,
                       CNNNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank);

    LoadResult LoadCNNFromBuffer(const uint8_t* data,
                                 std::size_t size,
                                 Arena& arena,
                                 CNNNetwork*& network,
                                 std::array<uint32_t, kMaxTensorRank>& input_shape,
                                 uint32_t& input_rank);
    /* Weights stay in the blob: `data` must outlive the network (flash .rodata
       or caller-owned buffer). Misaligned payloads return SizeMismatch. */

    LoadResult Load(const char* nk_path,
                    Arena& arena,
                    NetworkKind& kind,
                    MLPNetwork*& mlp,
                    CNNNetwork*& cnn,
                    std::array<uint32_t, kMaxTensorRank>& input_shape,
                    uint32_t& input_rank);

    const char* StatusMessage(LoadStatus status);
    const char* NetworkKindName(NetworkKind kind);
}
