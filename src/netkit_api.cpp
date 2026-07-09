#include "netkit.h"
#include "arena_util.hpp"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include "ops.hpp"
#include "conv2d.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include "cmsis_quant_plan.hpp"
#include "quant_output.hpp"
#if defined(NETKIT_DESKTOP)
#include "nk_regression.hpp"
#include "cli.hpp"
#include "test.hpp"
#endif
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace
{
#if defined(NETKIT_TARGET_MCU) || defined(NETKIT_TARGET_MPU)
    char g_last_error[NK_MAX_MESSAGE_LEN] = {};
#else
    thread_local char g_last_error[NK_MAX_MESSAGE_LEN] = {};
#endif

    void SetLastError(const char* message)
    {
        if (!message)
        {
            g_last_error[0] = '\0';
            return;
        }
        std::strncpy(g_last_error, message, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }

    nk_status_t FromLoadStatus(NkLoader::LoadStatus status)
    {
        switch (status)
        {
            case NkLoader::LoadStatus::Ok: return NK_OK;
            case NkLoader::LoadStatus::FileOpenFailed: return NK_ERR_MODEL_OPEN;
            case NkLoader::LoadStatus::ReadFailed: return NK_ERR_MODEL_READ;
            case NkLoader::LoadStatus::InvalidMagic:
            case NkLoader::LoadStatus::UnsupportedVersion:
            case NkLoader::LoadStatus::TruncatedFile: return NK_ERR_MODEL_PARSE;
            case NkLoader::LoadStatus::UnsupportedLayer: return NK_ERR_LAYER_CONFIG;
            case NkLoader::LoadStatus::SizeMismatch: return NK_ERR_WEIGHT_MISMATCH;
            case NkLoader::LoadStatus::ArenaOverflow: return NK_ERR_ARENA_OVERFLOW;
        }
        return NK_ERR_INVALID_ARGUMENT;
    }

    nk_network_kind_t FromNetworkKind(NkLoader::NetworkKind kind)
    {
        switch (kind)
        {
            case NkLoader::NetworkKind::Mlp: return NK_NETWORK_MLP;
            case NkLoader::NetworkKind::Cnn: return NK_NETWORK_CNN;
            default: return NK_NETWORK_UNKNOWN;
        }
    }

    nk_dtype_t FromDataType(DataType type)
    {
        switch (type)
        {
            case DataType::Float32: return NK_DTYPE_FLOAT32;
            case DataType::Int8: return NK_DTYPE_INT8;
            case DataType::UInt8: return NK_DTYPE_UINT8;
            case DataType::Int16: return NK_DTYPE_INT16;
        }
        return NK_DTYPE_FLOAT32;
    }

    void CopyModelOutputToFloat(const Tensor& output, float* dest, uint32_t count)
    {
        // Int8 models must not be dequantized in C++. Callers that need float
        // probabilities should use Python offline (or an int8 output API).
        if (output.type == DataType::Int8)
        {
            for (uint32_t i = 0; i < count; ++i)
                dest[i] = 0.0f;
            return;
        }

        const float* src = static_cast<const float*>(output.data);
        for (uint32_t i = 0; i < count; ++i)
            dest[i] = src[i];
    }

    ActivationType ToMlpActivation(nk_activation_t act)
    {
        switch (act)
        {
            case NK_ACTIVATION_RELU: return ActivationType::ReLU;
            case NK_ACTIVATION_SIGMOID: return ActivationType::Sigmoid;
            case NK_ACTIVATION_TANH: return ActivationType::Tanh;
            case NK_ACTIVATION_LEAKY_RELU: return ActivationType::LeakyReLU;
            case NK_ACTIVATION_RELU6: return ActivationType::ReLU6;
            case NK_ACTIVATION_SOFTMAX: return ActivationType::Softmax;
            default: return ActivationType::None;
        }
    }

    ConvActivationType ToCnnActivation(nk_conv_activation_t act)
    {
        switch (act)
        {
            case NK_CONV_ACTIVATION_RELU: return ConvActivationType::ReLU;
            case NK_CONV_ACTIVATION_SIGMOID: return ConvActivationType::Sigmoid;
            case NK_CONV_ACTIVATION_TANH: return ConvActivationType::Tanh;
            case NK_CONV_ACTIVATION_LEAKY_RELU: return ConvActivationType::LeakyReLU;
            case NK_CONV_ACTIVATION_RELU6: return ConvActivationType::ReLU6;
            case NK_CONV_ACTIVATION_SOFTMAX: return ConvActivationType::Softmax;
            default: return ConvActivationType::None;
        }
    }

    struct MlpHolder { MLPNetwork* net; };
    struct CnnHolder { CNNNetwork* net; };

    struct ModelState
    {
        nk_arch_info_t arch{};
        nk_network_kind_t kind = NK_NETWORK_UNKNOWN;
        MLPNetwork* mlp = nullptr;
        CNNNetwork* cnn = nullptr;
        bool loaded = false;
    };

    static_assert(sizeof(Arena) <= NK_ARENA_STORAGE_BYTES);
    static_assert(sizeof(MlpHolder) <= NK_MLP_STORAGE_BYTES);
    static_assert(sizeof(CnnHolder) <= NK_CNN_STORAGE_BYTES);
    static_assert(sizeof(ModelState) <= NK_MODEL_STORAGE_BYTES);

    Arena* ArenaPtr(nk_arena_t* arena) { return reinterpret_cast<Arena*>(arena->storage); }
    const Arena* ArenaPtr(const nk_arena_t* arena) { return reinterpret_cast<const Arena*>(arena->storage); }
    MlpHolder* MlpPtr(nk_mlp_t* mlp) { return reinterpret_cast<MlpHolder*>(mlp->storage); }
    const MlpHolder* MlpPtr(const nk_mlp_t* mlp) { return reinterpret_cast<const MlpHolder*>(mlp->storage); }
    CnnHolder* CnnPtr(nk_cnn_t* cnn) { return reinterpret_cast<CnnHolder*>(cnn->storage); }
    const CnnHolder* CnnPtr(const nk_cnn_t* cnn) { return reinterpret_cast<const CnnHolder*>(cnn->storage); }
    ModelState* ModelPtr(nk_model_t* model) { return reinterpret_cast<ModelState*>(model->storage); }
    const ModelState* ModelPtr(const nk_model_t* model) { return reinterpret_cast<const ModelState*>(model->storage); }

    Tensor* AsTensor(nk_tensor_t* t) { return reinterpret_cast<Tensor*>(t); }
    const Tensor* AsTensor(const nk_tensor_t* t) { return reinterpret_cast<const Tensor*>(t); }

    void ToNkTensor(const Tensor& src, nk_tensor_t* dst)
    {
        dst->data = src.data;
        dst->dtype = FromDataType(src.type);
        dst->rank = src.rank;
        dst->num_elements = src.num_elements;
        dst->bytes = src.bytes;
        for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
        {
            dst->shape[i] = src.shape[i];
            dst->stride[i] = src.stride[i];
        }
    }

    bool FileReadable(const char* path)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;
        std::fclose(file);
        return true;
    }

    const char* ResolveModelPath(const char* rel_path, char* buffer, std::size_t buffer_size)
    {
        if (FileReadable(rel_path))
            return rel_path;
        std::snprintf(buffer, buffer_size, "../%s", rel_path);
        if (FileReadable(buffer))
            return buffer;
        return rel_path;
    }

    void FillArchInfo(const NkLoader::ArchInfo& arch, nk_arch_info_t* info)
    {
        info->version = arch.version;
        info->kind = FromNetworkKind(arch.kind);
        info->input_rank = arch.input_rank;
        info->num_layers = arch.num_layers;
        info->expected_weight_floats = arch.weight_floats;
        info->input_elements = arch.input_elements;
        info->output_elements = arch.output_elements;
        for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
            info->input_shape[i] = i < arch.input_rank ? arch.input_shape[i] : 0;
    }

    void FillArchInfoFromParsed(const NkLoader::ParsedModel& parsed, nk_arch_info_t* info)
    {
        NkLoader::ArchInfo arch{};
        NkLoader::FillArchInfo(parsed, arch);
        FillArchInfo(arch, info);
        info->weights_bytes = parsed.header.weights_bytes;
        info->biases_bytes = parsed.header.biases_bytes;
    }

    std::size_t WeightPayloadBytes(const NkLoader::ParsedModel& parsed)
    {
        return static_cast<std::size_t>(parsed.header.weights_bytes) +
               static_cast<std::size_t>(parsed.header.biases_bytes);
    }

    bool ReadNkFile(const char* path, std::vector<uint8_t>& out)
    {
        if (!path)
            return false;
        FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;
        if (std::fseek(file, 0, SEEK_END) != 0)
        {
            std::fclose(file);
            return false;
        }
        const long file_size = std::ftell(file);
        if (file_size < 0)
        {
            std::fclose(file);
            return false;
        }
        if (std::fseek(file, 0, SEEK_SET) != 0)
        {
            std::fclose(file);
            return false;
        }
        out.resize(static_cast<std::size_t>(file_size));
        if (file_size > 0 &&
            std::fread(out.data(), 1, static_cast<std::size_t>(file_size), file) !=
                static_cast<std::size_t>(file_size))
        {
            std::fclose(file);
            return false;
        }
        std::fclose(file);
        return true;
    }

    Tensor MakeNhwcInput(float* data, uint32_t h, uint32_t w, uint32_t c)
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

    Tensor MakeNhwcInputInt8(int8_t* data, uint32_t h, uint32_t w, uint32_t c)
    {
        Tensor input{};
        input.data = data;
        input.type = DataType::Int8;
        input.rank = 3;
        input.shape[0] = h;
        input.shape[1] = w;
        input.shape[2] = c;
        input.stride[0] = w * c;
        input.stride[1] = c;
        input.stride[2] = 1;
        input.num_elements = h * w * c;
        input.bytes = input.num_elements * sizeof(int8_t);
        return input;
    }

    nk_status_t InspectModelFull(const NkLoader::ParsedModel& parsed,
                                 const char* resolved_path,
                                 const uint8_t* buffer,
                                 std::size_t buffer_size,
                                 nk_arena_t* arena,
                                 nk_inspect_info_t* info)
    {
        FillArchInfoFromParsed(parsed, &info->arch);
        info->weight_floats = info->arch.expected_weight_floats;
#if NETKIT_WEIGHTS_IN_RAM
        info->flash_payload_bytes = 0;
#else
        info->flash_payload_bytes = WeightPayloadBytes(parsed);
#endif

        if (info->arch.input_elements > NK_MAX_CASE_FLOATS)
            return NK_ERR_INVALID_ARGUMENT;

        const uint8_t* load_data = buffer;
        std::size_t load_size = buffer_size;

        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;
        Arena& arena_ref = *ArenaPtr(arena);

        if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
        {
            MLPNetwork* network = nullptr;
            const NkLoader::LoadResult load_result =
                resolved_path
                    ? NkLoader::LoadMLP(resolved_path, arena_ref, network, input_shape, input_rank)
                    : NkLoader::LoadMLPFromBuffer(
                          load_data, load_size, arena_ref, network, input_shape, input_rank);
            if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
            {
                SetLastError(load_result.message ? load_result.message : "MLP load failed");
                return FromLoadStatus(load_result.status);
            }

            info->arena_bytes_after_load = nk_arena_used(arena);
            const uint32_t output_cols = NkLoader::OutputElements(parsed) / input_shape[0];
            if (network->IsQuantized())
            {
                int8_t* in_i8 = static_cast<int8_t*>(
                    arena_ref.alloc(info->arch.input_elements * sizeof(int8_t), alignof(int8_t)));
                int8_t* out_i8 = static_cast<int8_t*>(
                    arena_ref.alloc(info->arch.output_elements * sizeof(int8_t), alignof(int8_t)));
                if (!in_i8 || !out_i8)
                    return NK_ERR_ARENA_OVERFLOW;
                std::memset(in_i8, 0, info->arch.input_elements * sizeof(int8_t));
                Tensor input = TensorFactory::View2DInt8(in_i8, input_shape[0], input_shape[1]);
                Tensor output = TensorFactory::View2DInt8(out_i8, input_shape[0], output_cols);
                network->forward(input, output, arena_ref);
            }
            else
            {
                Tensor input = TensorFactory::Create2D(arena_ref, input_shape[0], input_shape[1]);
                Tensor output = TensorFactory::Create2D(arena_ref, input_shape[0], output_cols);
                network->forward(input, output, arena_ref);
            }
            info->arena_bytes_after_forward = nk_arena_used(arena);
            info->arena_remaining = nk_arena_remaining(arena);
            SetLastError(nullptr);
            return NK_OK;
        }

        if (parsed.header.network_kind == NkFormat::NetworkKind::Cnn)
        {
            CNNNetwork* network = nullptr;
            const NkLoader::LoadResult load_result =
                resolved_path
                    ? NkLoader::LoadCNN(resolved_path, arena_ref, network, input_shape, input_rank)
                    : NkLoader::LoadCNNFromBuffer(
                          load_data, load_size, arena_ref, network, input_shape, input_rank);
            if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
            {
                SetLastError(load_result.message ? load_result.message : "CNN load failed");
                return FromLoadStatus(load_result.status);
            }

            info->arena_bytes_after_load = nk_arena_used(arena);
            if (network->IsQuantized())
            {
                int8_t zero_input_i8[NK_MAX_CASE_FLOATS] = {};
                Tensor input = MakeNhwcInputInt8(
                    zero_input_i8, input_shape[0], input_shape[1], input_shape[2]);
                Tensor& output = network->forward(input, arena_ref);
                if (!output.data)
                    return NK_ERR_ARENA_OVERFLOW;
            }
            else
            {
                float zero_input[NK_MAX_CASE_FLOATS] = {};
                Tensor input = MakeNhwcInput(
                    zero_input, input_shape[0], input_shape[1], input_shape[2]);
                Tensor& output = network->forward(input, arena_ref);
                if (!output.data)
                    return NK_ERR_ARENA_OVERFLOW;
            }
            info->arena_bytes_after_forward = nk_arena_used(arena);
            info->arena_remaining = nk_arena_remaining(arena);
            SetLastError(nullptr);
            return NK_OK;
        }

        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    nk_status_t ParseNkModel(const char* nk_path, NkLoader::ParsedModel& parsed, const char** resolved_out)
    {
#if defined(NETKIT_TARGET_MCU) || defined(NETKIT_TARGET_MPU)
        static char path_buffer[NkLoader::kMaxPathLen];
#else
        static thread_local char path_buffer[NkLoader::kMaxPathLen];
#endif
        const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));
        const NkLoader::LoadResult result = NkLoader::ParseFile(resolved, parsed);
        if (result.status != NkLoader::LoadStatus::Ok)
        {
            SetLastError(result.message ? result.message : NkLoader::StatusMessage(result.status));
            return FromLoadStatus(result.status);
        }
        if (resolved_out)
            *resolved_out = resolved;
        return NK_OK;
    }

    nk_status_t ParseNkBuffer(const uint8_t* data, size_t size, NkLoader::ParsedModel& parsed)
    {
        const NkLoader::LoadResult result = NkLoader::ParseBuffer(data, size, parsed);
        if (result.status != NkLoader::LoadStatus::Ok)
        {
            SetLastError(result.message ? result.message : NkLoader::StatusMessage(result.status));
            return FromLoadStatus(result.status);
        }
        return NK_OK;
    }
}

extern "C" {

const char* nk_version_string(void) { return "0.1.0"; }

const char* nk_status_string(nk_status_t status)
{
    switch (status)
    {
        case NK_OK: return "ok";
        case NK_ERR_MODEL_OPEN: return "model file open failed";
        case NK_ERR_MODEL_READ: return "model read failed";
        case NK_ERR_MODEL_PARSE: return "model parse failed";
        case NK_ERR_UNSUPPORTED_NETWORK: return "unsupported network";
        case NK_ERR_VERSION_MISMATCH: return "version mismatch";
        case NK_ERR_LAYER_CONFIG: return "layer config error";
        case NK_ERR_WEIGHT_MISMATCH: return "weight payload size mismatch";
        case NK_ERR_ARENA_OVERFLOW: return "arena overflow";
        case NK_ERR_INVALID_ARGUMENT: return "invalid argument";
        case NK_ERR_BUFFER_TOO_SMALL: return "buffer too small";
        case NK_ERR_MODEL_NOT_LOADED: return "model not loaded";
        case NK_ERR_NOT_INITIALIZED: return "not initialized";
    }
    return "unknown";
}

const char* nk_last_error(void) { return g_last_error; }

void nk_arena_init(nk_arena_t* arena, void* memory, size_t size)
{
    if (!arena)
        return;
    std::memset(arena->storage, 0, sizeof(arena->storage));
    ArenaPtr(arena)->init(memory, size);
}

void* nk_arena_alloc(nk_arena_t* arena, size_t size, size_t alignment)
{
    if (!arena)
        return nullptr;
    return ArenaPtr(arena)->alloc(size, alignment);
}

void nk_arena_reset(nk_arena_t* arena)
{
    if (arena)
        ArenaPtr(arena)->reset();
}

size_t nk_arena_capacity(const nk_arena_t* arena)
{
    return arena ? ArenaPtr(arena)->capacity : 0;
}

size_t nk_arena_used(const nk_arena_t* arena)
{
    return arena ? ArenaPtr(arena)->offset : 0;
}

size_t nk_arena_remaining(const nk_arena_t* arena)
{
    return arena ? ArenaPtr(arena)->remaining() : 0;
}

#if defined(NETKIT_ARENA_HEAP)
nk_status_t nk_arena_init_heap(nk_arena_t* arena, size_t capacity)
{
    if (!arena)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(arena->storage, 0, sizeof(arena->storage));
    if (!ArenaPtr(arena)->init_heap(capacity))
        return NK_ERR_ARENA_OVERFLOW;
    return NK_OK;
}

void nk_arena_destroy_heap(nk_arena_t* arena)
{
    if (!arena)
        return;
    ArenaPtr(arena)->destroy_heap();
}
#endif

nk_status_t nk_tensor_create_2d(nk_arena_t* arena, uint32_t rows, uint32_t cols, nk_tensor_t* out)
{
    if (!arena || !out)
        return NK_ERR_INVALID_ARGUMENT;
    Tensor t = TensorFactory::Create2D(*ArenaPtr(arena), rows, cols);
    if (!t.data)
        return NK_ERR_ARENA_OVERFLOW;
    ToNkTensor(t, out);
    return NK_OK;
}

nk_status_t nk_tensor_create_nd(nk_arena_t* arena, uint32_t rank, const uint32_t* shape, nk_tensor_t* out)
{
    if (!arena || !shape || !out || rank == 0 || rank > NK_MAX_TENSOR_RANK)
        return NK_ERR_INVALID_ARGUMENT;
    Tensor t = TensorFactory::CreateND(*ArenaPtr(arena), rank, std::span<const uint32_t>(shape, rank));
    if (!t.data)
        return NK_ERR_ARENA_OVERFLOW;
    ToNkTensor(t, out);
    return NK_OK;
}

void nk_tensor_view_2d(float* data, uint32_t rows, uint32_t cols, nk_tensor_t* out)
{
    if (!out)
        return;
    ToNkTensor(TensorFactory::View2D(data, rows, cols), out);
}

nk_status_t nk_tensor_fill(nk_tensor_t* tensor, const float* values, uint32_t count)
{
    if (!tensor || !values || count == 0)
        return NK_ERR_INVALID_ARGUMENT;
    if (count > AsTensor(tensor)->num_elements)
        return NK_ERR_INVALID_ARGUMENT;
    TensorFactory::Fill(*AsTensor(tensor), std::span<const float>(values, count));
    return NK_OK;
}

void nk_tensor_print(const nk_tensor_t* tensor)
{
    if (tensor)
        TensorFactory::Print(*AsTensor(tensor));
}

void nk_tensor_print_labeled(const char* label, const nk_tensor_t* tensor)
{
    if (tensor)
        TensorFactory::PrintLabeled(label, *AsTensor(tensor));
}

float* nk_tensor_data_f32(nk_tensor_t* tensor)
{
    return tensor ? tensor_data_f32(*AsTensor(tensor)) : nullptr;
}

const float* nk_tensor_data_f32_const(const nk_tensor_t* tensor)
{
    return tensor ? tensor_data_f32(*AsTensor(tensor)) : nullptr;
}

uint32_t nk_tensor_index_nhwc(const nk_tensor_t* tensor, uint32_t h, uint32_t w, uint32_t c)
{
    return tensor ? index_nhwc(*AsTensor(tensor), h, w, c) : 0;
}

bool nk_ops_is_elementwise_valid(const nk_tensor_t* a, const nk_tensor_t* b)
{
    return a && b && Ops::IsElementwiseValid(*AsTensor(a), *AsTensor(b));
}

bool nk_ops_check_same_shape_2d(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c)
{
    return a && b && c && Ops::CheckSameShape2D(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

bool nk_ops_check_same_shape_nd(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c)
{
    return a && b && c && Ops::CheckSameShapeND(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

bool nk_ops_is_matmul_valid(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c)
{
    return a && b && c && Ops::IsMatMulValid(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

bool nk_ops_is_elementwise_valid_nd(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c)
{
    return a && b && c && Ops::IsElementwiseValidND(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

bool nk_ops_is_unary_op_valid(const nk_tensor_t* a, const nk_tensor_t* c)
{
    return a && c && Ops::IsUnaryOpValid(*AsTensor(a), *AsTensor(c));
}

void nk_ops_mul(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c)
{
    if (a && b && c)
        Ops::Mul(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

void nk_ops_mul_scalar(const nk_tensor_t* a, float scalar, nk_tensor_t* c)
{
    if (a && c)
        Ops::MulScalar(*AsTensor(a), scalar, *AsTensor(c));
}

void nk_ops_mat_add(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c)
{
    if (a && b && c)
        Ops::MatAdd(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

void nk_ops_mat_add_nd(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c)
{
    if (a && b && c)
        Ops::MatAddND(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

void nk_ops_mat_mul(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c)
{
    if (a && b && c)
        Ops::MatMul(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

void nk_ops_mul_nd(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c)
{
    if (a && b && c)
        Ops::MulND(*AsTensor(a), *AsTensor(b), *AsTensor(c));
}

void nk_ops_relu(const nk_tensor_t* a, nk_tensor_t* c)
{
    if (a && c)
        Ops::ReLU(*AsTensor(a), *AsTensor(c));
}

void nk_ops_sigmoid(const nk_tensor_t* a, nk_tensor_t* c)
{
    if (a && c)
        Ops::Sigmoid(*AsTensor(a), *AsTensor(c));
}

void nk_ops_tanh(const nk_tensor_t* a, nk_tensor_t* c)
{
    if (a && c)
        Ops::Tanh(*AsTensor(a), *AsTensor(c));
}

void nk_ops_leaky_relu(const nk_tensor_t* a, nk_tensor_t* c, float alpha)
{
    if (a && c)
        Ops::LeakyReLU(*AsTensor(a), *AsTensor(c), alpha);
}

void nk_ops_relu6(const nk_tensor_t* a, nk_tensor_t* c)
{
    if (a && c)
        Ops::ReLU6(*AsTensor(a), *AsTensor(c));
}

void nk_ops_softmax(const nk_tensor_t* a, nk_tensor_t* c)
{
    if (a && c)
        Ops::Softmax(*AsTensor(a), *AsTensor(c));
}

void nk_conv2d_forward(const nk_conv2d_t* conv, const nk_tensor_t* input, nk_tensor_t* output)
{
    if (!conv || !input || !output)
        return;
    Conv2D c{};
    c.kernel_size = conv->kernel_size;
    c.stride = conv->stride;
    c.pad_h = conv->pad_h;
    c.pad_w = conv->pad_w;
    c.pad_h_end = conv->pad_h_end < 0 ? conv->pad_h : conv->pad_h_end;
    c.pad_w_end = conv->pad_w_end < 0 ? conv->pad_w : conv->pad_w_end;
    c.in_channels = conv->in_channels;
    c.out_channels = conv->out_channels;
    c.weights = conv->weights;
    c.bias = conv->bias;
    c.forward(*AsTensor(input), *AsTensor(output));
}

nk_status_t nk_mlp_create(nk_arena_t* arena, uint32_t num_layers, nk_mlp_t* mlp)
{
    if (!arena || !mlp)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(mlp->storage, 0, sizeof(mlp->storage));
    void* mem = ArenaPtr(arena)->alloc(sizeof(MLPNetwork), alignof(MLPNetwork));
    if (!mem)
        return NK_ERR_ARENA_OVERFLOW;
    MLPNetwork* net = new (mem) MLPNetwork(num_layers, *ArenaPtr(arena));
    if (!net->IsValid())
        return NK_ERR_ARENA_OVERFLOW;
    MlpPtr(mlp)->net = net;
    return NK_OK;
}

bool nk_mlp_is_valid(const nk_mlp_t* mlp)
{
    return mlp && MlpPtr(mlp)->net && MlpPtr(mlp)->net->IsValid();
}

nk_status_t nk_mlp_init_layer(nk_mlp_t* mlp,
                              uint32_t layer_idx,
                              const nk_tensor_t* weights,
                              const nk_tensor_t* bias,
                              nk_activation_t activation,
                              float leaky_alpha)
{
    if (!nk_mlp_is_valid(mlp) || !weights || !bias)
        return NK_ERR_INVALID_ARGUMENT;
    MlpPtr(mlp)->net->InitLayer(layer_idx,
                                *AsTensor(weights),
                                *AsTensor(bias),
                                ToMlpActivation(activation),
                                leaky_alpha);
    return NK_OK;
}

nk_status_t nk_mlp_init_activation_buffers(nk_mlp_t* mlp, nk_arena_t* arena, uint32_t batch_rows)
{
    if (!nk_mlp_is_valid(mlp) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    if (!MlpPtr(mlp)->net->InitActivationBuffers(*ArenaPtr(arena), batch_rows))
        return NK_ERR_ARENA_OVERFLOW;
    return NK_OK;
}

bool nk_mlp_has_activation_buffers(const nk_mlp_t* mlp)
{
    return mlp && MlpPtr(mlp)->net && MlpPtr(mlp)->net->HasActivationBuffers();
}

nk_status_t nk_mlp_forward(nk_mlp_t* mlp,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output)
{
    if (!nk_mlp_is_valid(mlp) || !arena || !input || !output)
        return NK_ERR_INVALID_ARGUMENT;
    if (!MlpPtr(mlp)->net->HasActivationBuffers())
        return NK_ERR_NOT_INITIALIZED;
    MlpPtr(mlp)->net->forward(*AsTensor(input), *AsTensor(output), *ArenaPtr(arena));
    if (!AsTensor(output)->data)
        return NK_ERR_ARENA_OVERFLOW;
    return NK_OK;
}

nk_status_t nk_cnn_create(nk_arena_t* arena, uint32_t num_layers, nk_cnn_t* cnn)
{
    if (!arena || !cnn)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(cnn->storage, 0, sizeof(cnn->storage));
    void* mem = ArenaPtr(arena)->alloc(sizeof(CNNNetwork), alignof(CNNNetwork));
    if (!mem)
        return NK_ERR_ARENA_OVERFLOW;
    CNNNetwork* net = new (mem) CNNNetwork(num_layers, *ArenaPtr(arena));
    if (!net->IsValid())
        return NK_ERR_ARENA_OVERFLOW;
    CnnPtr(cnn)->net = net;
    return NK_OK;
}

bool nk_cnn_is_valid(const nk_cnn_t* cnn)
{
    return cnn && CnnPtr(cnn)->net && CnnPtr(cnn)->net->IsValid();
}

nk_status_t nk_cnn_init_conv_layer(nk_cnn_t* cnn,
                                   uint32_t layer_idx,
                                   int kernel_size,
                                   int stride,
                                   int in_channels,
                                   int out_channels,
                                   float* weights,
                                   float* bias,
                                   nk_conv_activation_t activation,
                                   float leaky_alpha,
                                   int pad_h,
                                   int pad_w,
                                   int pad_h_end,
                                   int pad_w_end)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitConvLayer(static_cast<uint32_t>(layer_idx),
                                    kernel_size,
                                    stride,
                                    in_channels,
                                    out_channels,
                                    weights,
                                    bias,
                                    ToCnnActivation(activation),
                                    leaky_alpha,
                                    pad_h,
                                    pad_w,
                                    pad_h_end,
                                    pad_w_end);
    return NK_OK;
}

nk_status_t nk_cnn_init_depthwise_conv_layer(nk_cnn_t* cnn,
                                             uint32_t layer_idx,
                                             int kernel_h,
                                             int kernel_w,
                                             int stride,
                                             int channels,
                                             float* weights,
                                             float* bias,
                                             nk_conv_activation_t activation,
                                             float leaky_alpha,
                                             int pad_h,
                                             int pad_w,
                                             int pad_h_end,
                                             int pad_w_end)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitDepthwiseConvLayer(layer_idx,
                                             kernel_h,
                                             kernel_w,
                                             stride,
                                             channels,
                                             weights,
                                             bias,
                                             ToCnnActivation(activation),
                                             leaky_alpha,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end);
    return NK_OK;
}

nk_status_t nk_cnn_init_pool_layer(nk_cnn_t* cnn,
                                   uint32_t layer_idx,
                                   int pool_h,
                                   int pool_w,
                                   int stride,
                                   int pad_h,
                                   int pad_w,
                                   int pad_h_end,
                                   int pad_w_end)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitPoolLayer(
        layer_idx, pool_h, pool_w, stride, pad_h, pad_w, pad_h_end, pad_w_end);
    return NK_OK;
}

nk_status_t nk_cnn_init_avg_pool_layer(nk_cnn_t* cnn,
                                       uint32_t layer_idx,
                                       int pool_h,
                                       int pool_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int pad_h_end,
                                       int pad_w_end)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitAvgPoolLayer(
        layer_idx, pool_h, pool_w, stride, pad_h, pad_w, pad_h_end, pad_w_end);
    return NK_OK;
}

nk_status_t nk_cnn_init_batch_norm_layer(nk_cnn_t* cnn,
                                         uint32_t layer_idx,
                                         int channels,
                                         float* scale,
                                         float* bias)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitBatchNormLayer(layer_idx, channels, scale, bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_layernorm_layer(nk_cnn_t* cnn,
                                        uint32_t layer_idx,
                                        int channels,
                                        float eps,
                                        float* weight,
                                        float* bias)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitLayerNormLayer(layer_idx, channels, eps, weight, bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_convnextv2_block_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int channels,
                                                 float eps,
                                                 float* dw_weights,
                                                 float* dw_bias,
                                                 float* ln_weight,
                                                 float* ln_bias,
                                                 float* pw1_weight,
                                                 float* pw1_bias,
                                                 float* grn_gamma,
                                                 float* grn_beta,
                                                 float* pw2_weight,
                                                 float* pw2_bias)
{
    if (!nk_cnn_is_valid(cnn) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    CnnPtr(cnn)->net->InitConvNeXtV2BlockLayer(layer_idx,
                                               *ArenaPtr(arena),
                                               spatial_h,
                                               spatial_w,
                                               channels,
                                               eps,
                                               dw_weights,
                                               dw_bias,
                                               ln_weight,
                                               ln_bias,
                                               pw1_weight,
                                               pw1_bias,
                                               grn_gamma,
                                               grn_beta,
                                               pw2_weight,
                                               pw2_bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_mobilenetv4_uib_layer(nk_cnn_t* cnn,
                                              nk_arena_t* arena,
                                              uint32_t layer_idx,
                                              uint32_t spatial_h,
                                              uint32_t spatial_w,
                                              int in_channels,
                                              int out_channels,
                                              int start_dw_kernel,
                                              int middle_dw_kernel,
                                              int stride,
                                              int middle_dw_downsample,
                                              float expand_ratio,
                                              float* start_dw_weights,
                                              float* start_dw_bias,
                                              float* start_bn_scale,
                                              float* start_bn_bias,
                                              float* expand_weights,
                                              float* expand_bias,
                                              float* expand_bn_scale,
                                              float* expand_bn_bias,
                                              float* middle_dw_weights,
                                              float* middle_dw_bias,
                                              float* middle_bn_scale,
                                              float* middle_bn_bias,
                                              float* proj_weights,
                                              float* proj_bias,
                                              float* proj_bn_scale,
                                              float* proj_bn_bias)
{
    if (!nk_cnn_is_valid(cnn) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    CnnPtr(cnn)->net->InitMobilenetV4UibLayer(layer_idx,
                                              *ArenaPtr(arena),
                                              spatial_h,
                                              spatial_w,
                                              in_channels,
                                              out_channels,
                                              start_dw_kernel,
                                              middle_dw_kernel,
                                              stride,
                                              middle_dw_downsample != 0,
                                              expand_ratio,
                                              start_dw_weights,
                                              start_dw_bias,
                                              start_bn_scale,
                                              start_bn_bias,
                                              expand_weights,
                                              expand_bias,
                                              expand_bn_scale,
                                              expand_bn_bias,
                                              middle_dw_weights,
                                              middle_dw_bias,
                                              middle_bn_scale,
                                              middle_bn_bias,
                                              proj_weights,
                                              proj_bias,
                                              proj_bn_scale,
                                              proj_bn_bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_resnet_basic_block_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int in_channels,
                                                 int out_channels,
                                                 int stride,
                                                 float* conv1_weights,
                                                 float* conv1_bias,
                                                 float* bn1_scale,
                                                 float* bn1_bias,
                                                 float* conv2_weights,
                                                 float* conv2_bias,
                                                 float* bn2_scale,
                                                 float* bn2_bias,
                                                 float* shortcut_weights,
                                                 float* shortcut_bias,
                                                 float* shortcut_bn_scale,
                                                 float* shortcut_bn_bias)
{
    if (!nk_cnn_is_valid(cnn) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    CnnPtr(cnn)->net->InitResNetBasicBlockLayer(layer_idx,
                                               *ArenaPtr(arena),
                                               spatial_h,
                                               spatial_w,
                                               in_channels,
                                               out_channels,
                                               stride,
                                               conv1_weights,
                                               conv1_bias,
                                               bn1_scale,
                                               bn1_bias,
                                               conv2_weights,
                                               conv2_bias,
                                               bn2_scale,
                                               bn2_bias,
                                               shortcut_weights,
                                               shortcut_bias,
                                               shortcut_bn_scale,
                                               shortcut_bn_bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_yolox_decoupled_head_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int in_channels,
                                                 int hidden_dim,
                                                 int num_classes,
                                                 int num_convs,
                                                 float* stem_weights,
                                                 float* stem_bias,
                                                 float* const* cls_conv_weights,
                                                 float* const* cls_conv_bias,
                                                 float* const* reg_conv_weights,
                                                 float* const* reg_conv_bias,
                                                 float* cls_pred_weights,
                                                 float* cls_pred_bias,
                                                 float* reg_pred_weights,
                                                 float* reg_pred_bias,
                                                 float* obj_pred_weights,
                                                 float* obj_pred_bias)
{
    if (!nk_cnn_is_valid(cnn) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    CnnPtr(cnn)->net->InitYoloxDecoupledHeadLayer(layer_idx,
                                                 *ArenaPtr(arena),
                                                 spatial_h,
                                                 spatial_w,
                                                 in_channels,
                                                 hidden_dim,
                                                 num_classes,
                                                 num_convs,
                                                 stem_weights,
                                                 stem_bias,
                                                 cls_conv_weights,
                                                 cls_conv_bias,
                                                 reg_conv_weights,
                                                 reg_conv_bias,
                                                 cls_pred_weights,
                                                 cls_pred_bias,
                                                 reg_pred_weights,
                                                 reg_pred_bias,
                                                 obj_pred_weights,
                                                 obj_pred_bias);
    return NK_OK;
}

nk_status_t nk_cnn_init_flatten_layer(nk_cnn_t* cnn, uint32_t layer_idx)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitFlattenLayer(layer_idx);
    return NK_OK;
}

nk_status_t nk_cnn_init_dense_layer(nk_cnn_t* cnn,
                                    uint32_t layer_idx,
                                    const nk_tensor_t* weights,
                                    const nk_tensor_t* bias,
                                    nk_activation_t activation,
                                    float leaky_alpha)
{
    if (!nk_cnn_is_valid(cnn) || !weights || !bias)
        return NK_ERR_INVALID_ARGUMENT;
    CnnPtr(cnn)->net->InitDenseLayer(layer_idx,
                                     *AsTensor(weights),
                                     *AsTensor(bias),
                                     ToMlpActivation(activation),
                                     leaky_alpha);
    return NK_OK;
}

nk_status_t nk_cnn_init_activation_buffers(nk_cnn_t* cnn,
                                           nk_arena_t* arena,
                                           uint32_t in_h,
                                           uint32_t in_w,
                                           uint32_t in_c)
{
    if (!nk_cnn_is_valid(cnn) || !arena)
        return NK_ERR_INVALID_ARGUMENT;
    if (!CnnPtr(cnn)->net->InitActivationBuffers(*ArenaPtr(arena), in_h, in_w, in_c))
        return NK_ERR_ARENA_OVERFLOW;
    return NK_OK;
}

bool nk_cnn_has_activation_buffers(const nk_cnn_t* cnn)
{
    return cnn && CnnPtr(cnn)->net && CnnPtr(cnn)->net->HasActivationBuffers();
}

nk_status_t nk_cnn_forward(nk_cnn_t* cnn,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output)
{
    if (!nk_cnn_is_valid(cnn) || !arena || !input || !output)
        return NK_ERR_INVALID_ARGUMENT;
    if (!CnnPtr(cnn)->net->HasActivationBuffers())
        return NK_ERR_NOT_INITIALIZED;
    Tensor& out = CnnPtr(cnn)->net->forward(*AsTensor(input), *ArenaPtr(arena));
    if (!out.data)
        return NK_ERR_ARENA_OVERFLOW;
    ToNkTensor(out, output);
    return NK_OK;
}

nk_status_t nk_parse_architecture(const char* nk_path, nk_arch_info_t* info)
{
    if (!nk_path || !info)
        return NK_ERR_INVALID_ARGUMENT;
    NkLoader::ParsedModel parsed{};
    const nk_status_t status = ParseNkModel(nk_path, parsed, nullptr);
    if (status != NK_OK)
        return status;
    NkLoader::ArchInfo arch{};
    NkLoader::FillArchInfo(parsed, arch);
    std::memset(info, 0, sizeof(*info));
    FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

size_t nk_recommended_arena_bytes(const char* nk_path)
{
    if (!nk_path)
        return 0;
    NkLoader::ParsedModel parsed{};
    if (ParseNkModel(nk_path, parsed, nullptr) != NK_OK)
        return 0;
    const bool is_cnn = parsed.header.network_kind == NkFormat::NetworkKind::Cnn;
    std::size_t capacity = ArenaUtil::CapacityForModel(NkLoader::InputElements(parsed),
                                                       is_cnn,
                                                       parsed.header.weights_bytes,
                                                       parsed.header.biases_bytes);
#if defined(NETKIT_ARENA_HEAP)
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        nk_arena_t arena{};
        if (nk_arena_init_heap(&arena, capacity) != NK_OK)
        {
            capacity *= 2;
            continue;
        }

        nk_inspect_info_t info{};
        const nk_status_t status = nk_inspect_model(nk_path, &arena, &info);
        nk_arena_destroy_heap(&arena);
        if (status == NK_OK)
        {
            const std::size_t peak = info.arena_bytes_after_forward;
            const std::size_t headroom = peak / 8 + 65536;
            return peak + headroom;
        }

        if (status != NK_ERR_ARENA_OVERFLOW)
            return 0;
        capacity *= 2;
    }
    return 0;
#else
    if (!NETKIT_WEIGHTS_IN_RAM)
        capacity = capacity > WeightPayloadBytes(parsed) ? capacity - WeightPayloadBytes(parsed) : 0;
    return capacity;
#endif
}

nk_status_t nk_parse_architecture_memory(const uint8_t* data, size_t size, nk_arch_info_t* info)
{
    if (!data || size == 0 || !info)
        return NK_ERR_INVALID_ARGUMENT;
    NkLoader::ParsedModel parsed{};
    const nk_status_t status = ParseNkBuffer(data, size, parsed);
    if (status != NK_OK)
        return status;
    NkLoader::ArchInfo arch{};
    NkLoader::FillArchInfo(parsed, arch);
    std::memset(info, 0, sizeof(*info));
    FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_arch_print(const char* nk_path)
{
    if (!nk_path)
        return NK_ERR_INVALID_ARGUMENT;
    NkLoader::ParsedModel parsed{};
    const char* resolved = nullptr;
    const nk_status_t status = ParseNkModel(nk_path, parsed, &resolved);
    if (status != NK_OK)
        return status;
    NkLoader::PrintNetworkSummary(resolved, parsed);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_mlp_load(const char* nk_path, nk_arena_t* arena, nk_mlp_t* mlp, nk_arch_info_t* info)
{
    if (!nk_path || !arena || !mlp)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(mlp->storage, 0, sizeof(mlp->storage));
    NkLoader::ParsedModel parsed{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseNkModel(nk_path, parsed, &resolved);
    if (ps != NK_OK)
        return ps;
    if (parsed.header.network_kind != NkFormat::NetworkKind::Mlp)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    MLPNetwork* network = nullptr;
    const NkLoader::LoadResult lr =
        NkLoader::LoadMLP(resolved, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "MLP load failed");
        return FromLoadStatus(lr.status);
    }
    MlpPtr(mlp)->net = network;
    if (info)
        FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_mlp_load_memory(const uint8_t* data,
                             size_t size,
                             nk_arena_t* arena,
                             nk_mlp_t* mlp,
                             nk_arch_info_t* info)
{
    if (!data || size == 0 || !arena || !mlp)
        return NK_ERR_INVALID_ARGUMENT;

    NkLoader::ParsedModel parsed{};
    const nk_status_t ps = ParseNkBuffer(data, size, parsed);
    if (ps != NK_OK)
        return ps;
    if (parsed.header.network_kind != NkFormat::NetworkKind::Mlp)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::memset(mlp->storage, 0, sizeof(mlp->storage));
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    MLPNetwork* network = nullptr;
    const NkLoader::LoadResult lr =
        NkLoader::LoadMLPFromBuffer(data, size, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "MLP load failed");
        return FromLoadStatus(lr.status);
    }
    MlpPtr(mlp)->net = network;
    if (info)
        FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

bool nk_mlp_is_quantized(const nk_mlp_t* mlp)
{
    return mlp && nk_mlp_is_valid(mlp) && MlpPtr(mlp)->net->IsQuantized();
}

nk_status_t nk_cnn_load(const char* nk_path, nk_arena_t* arena, nk_cnn_t* cnn, nk_arch_info_t* info)
{
    if (!nk_path || !arena || !cnn)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(cnn->storage, 0, sizeof(cnn->storage));
    NkLoader::ParsedModel parsed{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseNkModel(nk_path, parsed, &resolved);
    if (ps != NK_OK)
        return ps;
    if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    CNNNetwork* network = nullptr;
    const NkLoader::LoadResult lr =
        NkLoader::LoadCNN(resolved, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "CNN load failed");
        return FromLoadStatus(lr.status);
    }
    CnnPtr(cnn)->net = network;
    if (info)
        FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_cnn_load_memory(const uint8_t* data,
                             size_t size,
                             nk_arena_t* arena,
                             nk_cnn_t* cnn,
                             nk_arch_info_t* info)
{
    if (!data || size == 0 || !arena || !cnn)
        return NK_ERR_INVALID_ARGUMENT;

    NkLoader::ParsedModel parsed{};
    const nk_status_t ps = ParseNkBuffer(data, size, parsed);
    if (ps != NK_OK)
        return ps;
    if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::memset(cnn->storage, 0, sizeof(cnn->storage));
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    CNNNetwork* network = nullptr;
    const NkLoader::LoadResult lr =
        NkLoader::LoadCNNFromBuffer(data, size, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "CNN load failed");
        return FromLoadStatus(lr.status);
    }
    CnnPtr(cnn)->net = network;
    if (info)
        FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

bool nk_cnn_is_quantized(const nk_cnn_t* cnn)
{
    return cnn && nk_cnn_is_valid(cnn) && CnnPtr(cnn)->net->IsQuantized();
}

nk_status_t nk_model_load_auto(const char* nk_path,
                               nk_arena_t* arena,
                               nk_network_kind_t* kind,
                               nk_mlp_t* mlp,
                               nk_cnn_t* cnn,
                               nk_arch_info_t* info)
{
    if (!nk_path || !arena || !kind)
        return NK_ERR_INVALID_ARGUMENT;
    NkLoader::ParsedModel parsed{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseNkModel(nk_path, parsed, &resolved);
    if (ps != NK_OK)
        return ps;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    MLPNetwork* mlp_net = nullptr;
    CNNNetwork* cnn_net = nullptr;
    NkLoader::NetworkKind detected = NkLoader::NetworkKind::Unknown;

    const NkLoader::LoadResult lr = NkLoader::Load(resolved,
                                                   *ArenaPtr(arena),
                                                   detected,
                                                   mlp_net,
                                                   cnn_net,
                                                   input_shape,
                                                   input_rank);
    if (lr.status != NkLoader::LoadStatus::Ok)
    {
        SetLastError(lr.message ? lr.message : "model load failed");
        return FromLoadStatus(lr.status);
    }

    *kind = FromNetworkKind(detected);
    if (detected == NkLoader::NetworkKind::Mlp && mlp)
    {
        std::memset(mlp->storage, 0, sizeof(mlp->storage));
        MlpPtr(mlp)->net = mlp_net;
    }
    if (detected == NkLoader::NetworkKind::Cnn && cnn)
    {
        std::memset(cnn->storage, 0, sizeof(cnn->storage));
        CnnPtr(cnn)->net = cnn_net;
    }
    if (info)
        FillArchInfoFromParsed(parsed, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_load(const char* nk_path, nk_arena_t* arena, nk_model_t* model)
{
    if (!nk_path || !arena || !model)
        return NK_ERR_INVALID_ARGUMENT;

    char path_buffer[NkLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));
    NkLoader::ParsedModel parsed{};
    const NkLoader::LoadResult arch_result = NkLoader::ParseFile(resolved, parsed);
    if (arch_result.status != NkLoader::LoadStatus::Ok)
    {
        SetLastError(arch_result.message ? arch_result.message : "model parse failed");
        return FromLoadStatus(arch_result.status);
    }

    std::memset(model->storage, 0, sizeof(model->storage));
    ModelState* state = ModelPtr(model);
    NkLoader::ArchInfo arch{};
    NkLoader::FillArchInfo(parsed, arch);
    FillArchInfoFromParsed(parsed, &state->arch);
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    for (uint32_t i = 0; i < state->arch.input_rank; ++i)
        input_shape[i] = state->arch.input_shape[i];

    if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
    {
        const NkLoader::LoadResult load_result =
            NkLoader::LoadMLP(resolved, *ArenaPtr(arena), state->mlp, input_shape, state->arch.input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !state->mlp || !state->mlp->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "MLP load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_MLP;
        state->loaded = true;
    }
    else if (parsed.header.network_kind == NkFormat::NetworkKind::Cnn)
    {
        const NkLoader::LoadResult load_result =
            NkLoader::LoadCNN(resolved, *ArenaPtr(arena), state->cnn, input_shape, state->arch.input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !state->cnn || !state->cnn->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "CNN load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_CNN;
        state->loaded = true;
    }
    else
    {
        SetLastError("unsupported network kind");
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_load_memory(const uint8_t* data, size_t size, nk_arena_t* arena, nk_model_t* model)
{
    if (!data || size == 0 || !arena || !model)
        return NK_ERR_INVALID_ARGUMENT;

    NkLoader::ParsedModel parsed{};
    const nk_status_t parse_status = ParseNkBuffer(data, size, parsed);
    if (parse_status != NK_OK)
        return parse_status;

    std::memset(model->storage, 0, sizeof(model->storage));
    ModelState* state = ModelPtr(model);
    NkLoader::ArchInfo arch{};
    NkLoader::FillArchInfo(parsed, arch);
    FillArchInfoFromParsed(parsed, &state->arch);
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    for (uint32_t i = 0; i < state->arch.input_rank; ++i)
        input_shape[i] = state->arch.input_shape[i];

    if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
    {
        const NkLoader::LoadResult load_result = NkLoader::LoadMLPFromBuffer(
            data, size, *ArenaPtr(arena), state->mlp, input_shape, state->arch.input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !state->mlp || !state->mlp->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "MLP load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_MLP;
        state->loaded = true;
    }
    else if (parsed.header.network_kind == NkFormat::NetworkKind::Cnn)
    {
        const NkLoader::LoadResult load_result = NkLoader::LoadCNNFromBuffer(
            data, size, *ArenaPtr(arena), state->cnn, input_shape, state->arch.input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !state->cnn || !state->cnn->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "CNN load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_CNN;
        state->loaded = true;
    }
    else
    {
        SetLastError("unsupported network kind");
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info)
{
    if (!model || !info)
        return NK_ERR_INVALID_ARGUMENT;
    const ModelState* state = ModelPtr(model);
    if (!state->loaded)
        return NK_ERR_MODEL_NOT_LOADED;
    *info = state->arch;
    return NK_OK;
}

uint32_t nk_model_input_count(const nk_model_t* model)
{
    return model && ModelPtr(model)->loaded ? ModelPtr(model)->arch.input_elements : 0;
}

uint32_t nk_model_output_count(const nk_model_t* model)
{
    return model && ModelPtr(model)->loaded ? ModelPtr(model)->arch.output_elements : 0;
}

nk_network_kind_t nk_model_kind(const nk_model_t* model)
{
    return model ? ModelPtr(model)->kind : NK_NETWORK_UNKNOWN;
}

bool nk_model_is_quantized(const nk_model_t* model)
{
    if (!model || !ModelPtr(model)->loaded)
        return false;
    const ModelState* state = ModelPtr(model);
    if (state->kind == NK_NETWORK_MLP)
        return state->mlp && state->mlp->IsQuantized();
    if (state->kind == NK_NETWORK_CNN)
        return state->cnn && state->cnn->IsQuantized();
    return false;
}

nk_status_t nk_model_run(const nk_model_t* model,
                         nk_arena_t* arena,
                         const float* input,
                         uint32_t input_count,
                         float* output,
                         uint32_t output_capacity,
                         uint32_t* output_count)
{
    if (!model || !arena || !input || !output || !output_count)
        return NK_ERR_INVALID_ARGUMENT;
    const ModelState* state = ModelPtr(model);
    if (!state->loaded)
        return NK_ERR_MODEL_NOT_LOADED;
    if (nk_model_is_quantized(model))
        return NK_ERR_INVALID_ARGUMENT;  // use nk_model_run_int8
    if (input_count != state->arch.input_elements)
        return NK_ERR_INVALID_ARGUMENT;
    if (output_capacity < state->arch.output_elements)
        return NK_ERR_BUFFER_TOO_SMALL;

    if (state->kind == NK_NETWORK_MLP)
    {
        Tensor input_tensor =
            TensorFactory::Create2D(*ArenaPtr(arena), state->arch.input_shape[0], state->arch.input_shape[1]);
        if (!input_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        float* input_data = static_cast<float*>(input_tensor.data);
        for (uint32_t i = 0; i < input_count; ++i)
            input_data[i] = input[i];
        const uint32_t output_cols = state->arch.output_elements / state->arch.input_shape[0];
        Tensor output_tensor =
            TensorFactory::Create2D(*ArenaPtr(arena), state->arch.input_shape[0], output_cols);
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        state->mlp->forward(input_tensor, output_tensor, *ArenaPtr(arena));
        CopyModelOutputToFloat(output_tensor, output, state->arch.output_elements);
    }
    else if (state->kind == NK_NETWORK_CNN)
    {
        float input_buffer[NK_MAX_CASE_FLOATS] = {};
        if (input_count > NK_MAX_CASE_FLOATS)
            return NK_ERR_INVALID_ARGUMENT;
        for (uint32_t i = 0; i < input_count; ++i)
            input_buffer[i] = input[i];
        Tensor input_tensor = MakeNhwcInput(input_buffer,
                                            state->arch.input_shape[0],
                                            state->arch.input_shape[1],
                                            state->arch.input_shape[2]);
        Tensor& output_tensor = state->cnn->forward(input_tensor, *ArenaPtr(arena));
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        const uint32_t actual_output = output_tensor.num_elements;
        if (output_capacity < actual_output)
            return NK_ERR_BUFFER_TOO_SMALL;
        CopyModelOutputToFloat(output_tensor, output, actual_output);
        *output_count = actual_output;
        return NK_OK;
    }
    else
    {
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    *output_count = state->arch.output_elements;
    return NK_OK;
}

nk_status_t nk_model_run_int8(const nk_model_t* model,
                              nk_arena_t* arena,
                              const int8_t* input,
                              uint32_t input_count,
                              int8_t* output,
                              uint32_t output_capacity,
                              uint32_t* output_count)
{
    if (!model || !arena || !input || !output || !output_count)
        return NK_ERR_INVALID_ARGUMENT;
    const ModelState* state = ModelPtr(model);
    if (!state->loaded)
        return NK_ERR_MODEL_NOT_LOADED;
    if (!nk_model_is_quantized(model))
        return NK_ERR_INVALID_ARGUMENT;  // use nk_model_run for float32
    if (input_count != state->arch.input_elements)
        return NK_ERR_INVALID_ARGUMENT;
    if (output_capacity < state->arch.output_elements)
        return NK_ERR_BUFFER_TOO_SMALL;

    if (state->kind == NK_NETWORK_MLP)
    {
        Tensor input_tensor = TensorFactory::View2DInt8(
            const_cast<int8_t*>(input), state->arch.input_shape[0], state->arch.input_shape[1]);
        const uint32_t output_cols = state->arch.output_elements / state->arch.input_shape[0];
        Tensor output_tensor =
            TensorFactory::View2DInt8(output, state->arch.input_shape[0], output_cols);
        if (!input_tensor.data || !output_tensor.data)
            return NK_ERR_INVALID_ARGUMENT;
        state->mlp->forward(input_tensor, output_tensor, *ArenaPtr(arena));
        if (output_tensor.type != DataType::Int8)
            return NK_ERR_INVALID_ARGUMENT;
        *output_count = state->arch.output_elements;
        return NK_OK;
    }

    if (state->kind == NK_NETWORK_CNN)
    {
        CmsisQuantPlan::Runtime* runtime = state->cnn->quant_runtime();
        if (!runtime)
            return NK_ERR_INVALID_ARGUMENT;
        if (!CmsisQuantPlan::ForwardInt8ToBuffer(
                *runtime, *state->cnn, input, output, state->arch.output_elements))
            return NK_ERR_INVALID_ARGUMENT;
        *output_count = state->arch.output_elements;
        return NK_OK;
    }

    return NK_ERR_UNSUPPORTED_NETWORK;
}

nk_status_t nk_inspect_model(const char* nk_path, nk_arena_t* arena, nk_inspect_info_t* info)
{
    if (!nk_path || !arena || !info)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));

    NkLoader::ParsedModel parsed{};
    const char* resolved = nullptr;
    const nk_status_t parse_status = ParseNkModel(nk_path, parsed, &resolved);
    if (parse_status != NK_OK)
        return parse_status;

    return InspectModelFull(parsed, resolved, nullptr, 0, arena, info);
}

nk_status_t nk_inspect_model_memory(const uint8_t* data,
                                    size_t size,
                                    nk_arena_t* arena,
                                    nk_inspect_info_t* info)
{
    if (!data || size == 0 || !arena || !info)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));

    NkLoader::ParsedModel parsed{};
    const nk_status_t parse_status = ParseNkBuffer(data, size, parsed);
    if (parse_status != NK_OK)
        return parse_status;

    return InspectModelFull(parsed, nullptr, data, size, arena, info);
}

#if defined(NETKIT_DESKTOP)
nk_test_summary_t nk_run_model_tests(const char* nk_path)
{
    nk_test_summary_t summary{};
    const NkRegression::RunSummary result = NkRegression::RunModelTests(nk_path);
    summary.passed = result.passed;
    summary.failed = result.failed;
    return summary;
}

nk_test_summary_t nk_run_all_tests(void)
{
    nk_test_summary_t summary{};
    const NkRegression::RunSummary result = run_all_tests();
    summary.passed = result.passed;
    summary.failed = result.failed;
    return summary;
}

int nk_cli_run(int argc, char** argv)
{
    return Cli::Run(argc, argv);
}
#endif /* NETKIT_DESKTOP */

} /* extern "C" */
