#include "netkit.h"
#include "model_loader.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include "ops.hpp"
#include "conv2d.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include "vectors_loader.hpp"
#include "cli.hpp"
#include "test.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>
#include <span>

namespace
{
    thread_local char g_last_error[NK_MAX_MESSAGE_LEN] = {};

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

    nk_status_t FromLoadStatus(ModelLoader::LoadStatus status)
    {
        switch (status)
        {
            case ModelLoader::LoadStatus::Ok: return NK_OK;
            case ModelLoader::LoadStatus::JsonOpenFailed: return NK_ERR_JSON_OPEN;
            case ModelLoader::LoadStatus::BinOpenFailed: return NK_ERR_BIN_OPEN;
            case ModelLoader::LoadStatus::JsonParseFailed: return NK_ERR_JSON_PARSE;
            case ModelLoader::LoadStatus::UnsupportedNetwork: return NK_ERR_UNSUPPORTED_NETWORK;
            case ModelLoader::LoadStatus::VersionMismatch: return NK_ERR_VERSION_MISMATCH;
            case ModelLoader::LoadStatus::LayerConfigError: return NK_ERR_LAYER_CONFIG;
            case ModelLoader::LoadStatus::BinSizeMismatch: return NK_ERR_BIN_SIZE_MISMATCH;
            case ModelLoader::LoadStatus::ArenaOverflow: return NK_ERR_ARENA_OVERFLOW;
        }
        return NK_ERR_INVALID_ARGUMENT;
    }

    nk_network_kind_t FromNetworkKind(ModelLoader::NetworkKind kind)
    {
        switch (kind)
        {
            case ModelLoader::NetworkKind::MLP: return NK_NETWORK_MLP;
            case ModelLoader::NetworkKind::CNN: return NK_NETWORK_CNN;
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
        nk_network_kind_t kind = NK_NETWORK_UNKNOWN;
        MLPNetwork* mlp = nullptr;
        CNNNetwork* cnn = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;
        uint32_t input_elements = 0;
        uint32_t output_elements = 0;
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

    uint32_t InputElementCount(const ModelLoader::ArchitectureSpec& spec)
    {
        uint32_t count = 1;
        for (uint32_t i = 0; i < spec.input_rank; ++i)
            count *= spec.input_shape[i];
        return count;
    }

    uint32_t MlpOutputElements(const ModelLoader::ArchitectureSpec& spec)
    {
        if (spec.num_layers == 0)
            return 0;
        return spec.input_shape[0] * spec.dense_layers[spec.num_layers - 1].units;
    }

    uint32_t CnnOutputElements(const ModelLoader::ArchitectureSpec& spec)
    {
        uint32_t h = spec.input_shape[0];
        uint32_t w = spec.input_shape[1];
        uint32_t c = spec.input_shape[2];
        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            const ModelLoader::ConvLayerConfig& layer = spec.conv_layers[i];
            h = (h - layer.kernel_size) / layer.stride + 1;
            w = (w - layer.kernel_size) / layer.stride + 1;
            c = layer.filters;
        }
        return h * w * c;
    }

    void FillArchInfo(const ModelLoader::ArchitectureSpec& spec, nk_arch_info_t* info)
    {
        info->version = spec.version;
        info->kind = FromNetworkKind(spec.kind);
        info->input_rank = spec.input_rank;
        info->num_layers = spec.num_layers;
        info->expected_weight_floats = spec.expected_weight_floats;
        info->input_elements = InputElementCount(spec);
        for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
            info->input_shape[i] = i < spec.input_rank ? spec.input_shape[i] : 0;
        if (spec.kind == ModelLoader::NetworkKind::MLP)
            info->output_elements = MlpOutputElements(spec);
        else if (spec.kind == ModelLoader::NetworkKind::CNN)
            info->output_elements = CnnOutputElements(spec);
        else
            info->output_elements = 0;
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

    nk_status_t ParseSpec(const char* json_path, ModelLoader::ArchitectureSpec& spec, const char** resolved_out)
    {
        static thread_local char path_buffer[ModelLoader::kMaxPathLen];
        const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));
        const ModelLoader::LoadResult result = ModelLoader::ParseArchitecture(resolved, spec);
        if (result.status != ModelLoader::LoadStatus::Ok)
        {
            SetLastError(result.message ? result.message : "architecture parse failed");
            return FromLoadStatus(result.status);
        }
        if (resolved_out)
            *resolved_out = resolved;
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
        case NK_ERR_JSON_OPEN: return "json open failed";
        case NK_ERR_BIN_OPEN: return "bin open failed";
        case NK_ERR_JSON_PARSE: return "json parse failed";
        case NK_ERR_UNSUPPORTED_NETWORK: return "unsupported network";
        case NK_ERR_VERSION_MISMATCH: return "version mismatch";
        case NK_ERR_LAYER_CONFIG: return "layer config error";
        case NK_ERR_BIN_SIZE_MISMATCH: return "bin size mismatch";
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

void* nk_arena_alloc(nk_arena_t* arena, size_t size)
{
    if (!arena)
        return nullptr;
    return ArenaPtr(arena)->alloc(size);
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
    float* dst = static_cast<float*>(AsTensor(tensor)->data);
    for (uint32_t i = 0; i < count; ++i)
        dst[i] = values[i];
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
    void* mem = ArenaPtr(arena)->alloc(sizeof(MLPNetwork));
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

nk_status_t nk_mlp_forward(nk_mlp_t* mlp,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output)
{
    if (!nk_mlp_is_valid(mlp) || !arena || !input || !output)
        return NK_ERR_INVALID_ARGUMENT;
    MlpPtr(mlp)->net->forward(*AsTensor(input), *AsTensor(output), *ArenaPtr(arena));
    return NK_OK;
}

nk_status_t nk_cnn_create(nk_arena_t* arena, uint32_t num_layers, nk_cnn_t* cnn)
{
    if (!arena || !cnn)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(cnn->storage, 0, sizeof(cnn->storage));
    void* mem = ArenaPtr(arena)->alloc(sizeof(CNNNetwork));
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

nk_status_t nk_cnn_init_layer(nk_cnn_t* cnn,
                              uint32_t layer_idx,
                              int kernel_size,
                              int stride,
                              int in_channels,
                              int out_channels,
                              float* weights,
                              float* bias,
                              nk_conv_activation_t activation,
                              float leaky_alpha)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitLayer(static_cast<uint32_t>(layer_idx),
                                kernel_size,
                                stride,
                                in_channels,
                                out_channels,
                                weights,
                                bias,
                                ToCnnActivation(activation),
                                leaky_alpha);
    return NK_OK;
}

nk_status_t nk_cnn_forward(nk_cnn_t* cnn,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output)
{
    if (!nk_cnn_is_valid(cnn) || !arena || !input || !output)
        return NK_ERR_INVALID_ARGUMENT;
    Tensor& out = CnnPtr(cnn)->net->forward(*AsTensor(input), *ArenaPtr(arena));
    if (!out.data)
        return NK_ERR_ARENA_OVERFLOW;
    ToNkTensor(out, output);
    return NK_OK;
}

nk_status_t nk_parse_architecture(const char* json_path, nk_arch_info_t* info)
{
    if (!json_path || !info)
        return NK_ERR_INVALID_ARGUMENT;
    ModelLoader::ArchitectureSpec spec{};
    const nk_status_t status = ParseSpec(json_path, spec, nullptr);
    if (status != NK_OK)
        return status;
    std::memset(info, 0, sizeof(*info));
    FillArchInfo(spec, info);
    SetLastError(nullptr);
    return NK_OK;
}

void nk_arch_print(const char* json_path)
{
    if (!json_path)
        return;
    ModelLoader::ArchitectureSpec spec{};
    const char* resolved = nullptr;
    if (ParseSpec(json_path, spec, &resolved) != NK_OK)
        return;
    std::cout << "Model: " << resolved << "\n\nArchitecture:\n";
    ModelLoader::PrintArchitecture(spec);
}

bool nk_json_path_to_bin_path(const char* json_path, char* bin_path, size_t bin_path_capacity)
{
    return ModelLoader::JsonPathToBinPath(json_path, bin_path, bin_path_capacity);
}

nk_status_t nk_load_weights_bin(const char* json_path,
                                nk_arena_t* arena,
                                float** weights,
                                size_t* float_count)
{
    if (!json_path || !arena || !weights || !float_count)
        return NK_ERR_INVALID_ARGUMENT;
    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));
    float* loaded = nullptr;
    std::size_t count = 0;
    const char* err = nullptr;
    const ModelLoader::LoadStatus status =
        ModelLoader::LoadWeightsBin(resolved, *ArenaPtr(arena), loaded, count, &err);
    *weights = loaded;
    *float_count = count;
    if (status != ModelLoader::LoadStatus::Ok)
    {
        SetLastError(err ? err : "weight load failed");
        return FromLoadStatus(status);
    }
    return NK_OK;
}

nk_status_t nk_mlp_load(const char* json_path, nk_arena_t* arena, nk_mlp_t* mlp, nk_arch_info_t* info)
{
    if (!json_path || !arena || !mlp)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(mlp->storage, 0, sizeof(mlp->storage));
    ModelLoader::ArchitectureSpec spec{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseSpec(json_path, spec, &resolved);
    if (ps != NK_OK)
        return ps;
    if (spec.kind != ModelLoader::NetworkKind::MLP)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    MLPNetwork* network = nullptr;
    const ModelLoader::LoadResult lr =
        ModelLoader::LoadMLP(resolved, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "MLP load failed");
        return FromLoadStatus(lr.status);
    }
    MlpPtr(mlp)->net = network;
    if (info)
        FillArchInfo(spec, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_cnn_load(const char* json_path, nk_arena_t* arena, nk_cnn_t* cnn, nk_arch_info_t* info)
{
    if (!json_path || !arena || !cnn)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(cnn->storage, 0, sizeof(cnn->storage));
    ModelLoader::ArchitectureSpec spec{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseSpec(json_path, spec, &resolved);
    if (ps != NK_OK)
        return ps;
    if (spec.kind != ModelLoader::NetworkKind::CNN)
        return NK_ERR_UNSUPPORTED_NETWORK;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    CNNNetwork* network = nullptr;
    const ModelLoader::LoadResult lr =
        ModelLoader::LoadCNN(resolved, *ArenaPtr(arena), network, input_shape, input_rank);
    if (lr.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        SetLastError(lr.message ? lr.message : "CNN load failed");
        return FromLoadStatus(lr.status);
    }
    CnnPtr(cnn)->net = network;
    if (info)
        FillArchInfo(spec, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_load_auto(const char* json_path,
                               nk_arena_t* arena,
                               nk_network_kind_t* kind,
                               nk_mlp_t* mlp,
                               nk_cnn_t* cnn,
                               nk_arch_info_t* info)
{
    if (!json_path || !arena || !kind)
        return NK_ERR_INVALID_ARGUMENT;
    ModelLoader::ArchitectureSpec spec{};
    const char* resolved = nullptr;
    const nk_status_t ps = ParseSpec(json_path, spec, &resolved);
    if (ps != NK_OK)
        return ps;

    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    MLPNetwork* mlp_net = nullptr;
    CNNNetwork* cnn_net = nullptr;
    ModelLoader::NetworkKind detected = ModelLoader::NetworkKind::Unknown;

    const ModelLoader::LoadResult lr = ModelLoader::Load(resolved,
                                                         *ArenaPtr(arena),
                                                         detected,
                                                         mlp_net,
                                                         cnn_net,
                                                         input_shape,
                                                         input_rank);
    if (lr.status != ModelLoader::LoadStatus::Ok)
    {
        SetLastError(lr.message ? lr.message : "model load failed");
        return FromLoadStatus(lr.status);
    }

    *kind = FromNetworkKind(detected);
    if (detected == ModelLoader::NetworkKind::MLP && mlp)
    {
        std::memset(mlp->storage, 0, sizeof(mlp->storage));
        MlpPtr(mlp)->net = mlp_net;
    }
    if (detected == ModelLoader::NetworkKind::CNN && cnn)
    {
        std::memset(cnn->storage, 0, sizeof(cnn->storage));
        CnnPtr(cnn)->net = cnn_net;
    }
    if (info)
        FillArchInfo(spec, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_load(const char* json_path, nk_arena_t* arena, nk_model_t* model)
{
    if (!json_path || !arena || !model)
        return NK_ERR_INVALID_ARGUMENT;

    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));
    ModelLoader::ArchitectureSpec spec{};
    const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(resolved, spec);
    if (arch_result.status != ModelLoader::LoadStatus::Ok)
    {
        SetLastError(arch_result.message ? arch_result.message : "architecture parse failed");
        return FromLoadStatus(arch_result.status);
    }

    std::memset(model->storage, 0, sizeof(model->storage));
    ModelState* state = ModelPtr(model);
    state->input_rank = spec.input_rank;
    state->input_elements = InputElementCount(spec);
    for (uint32_t i = 0; i < spec.input_rank; ++i)
        state->input_shape[i] = spec.input_shape[i];

    if (spec.kind == ModelLoader::NetworkKind::MLP)
    {
        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadMLP(resolved, *ArenaPtr(arena), state->mlp, state->input_shape, state->input_rank);
        if (load_result.status != ModelLoader::LoadStatus::Ok || !state->mlp || !state->mlp->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "MLP load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_MLP;
        state->output_elements = MlpOutputElements(spec);
        state->loaded = true;
    }
    else if (spec.kind == ModelLoader::NetworkKind::CNN)
    {
        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadCNN(resolved, *ArenaPtr(arena), state->cnn, state->input_shape, state->input_rank);
        if (load_result.status != ModelLoader::LoadStatus::Ok || !state->cnn || !state->cnn->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "CNN load failed");
            return FromLoadStatus(load_result.status);
        }
        state->kind = NK_NETWORK_CNN;
        state->output_elements = CnnOutputElements(spec);
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
    std::memset(info, 0, sizeof(*info));
    info->version = 1;
    info->kind = state->kind;
    info->input_rank = state->input_rank;
    info->input_elements = state->input_elements;
    info->output_elements = state->output_elements;
    for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
        info->input_shape[i] = i < state->input_rank ? state->input_shape[i] : 0;
    return NK_OK;
}

uint32_t nk_model_input_count(const nk_model_t* model)
{
    return model ? ModelPtr(model)->input_elements : 0;
}

uint32_t nk_model_output_count(const nk_model_t* model)
{
    return model ? ModelPtr(model)->output_elements : 0;
}

nk_network_kind_t nk_model_kind(const nk_model_t* model)
{
    return model ? ModelPtr(model)->kind : NK_NETWORK_UNKNOWN;
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
    if (input_count != state->input_elements)
        return NK_ERR_INVALID_ARGUMENT;
    if (output_capacity < state->output_elements)
        return NK_ERR_BUFFER_TOO_SMALL;

    if (state->kind == NK_NETWORK_MLP)
    {
        Tensor input_tensor =
            TensorFactory::Create2D(*ArenaPtr(arena), state->input_shape[0], state->input_shape[1]);
        if (!input_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        float* input_data = static_cast<float*>(input_tensor.data);
        for (uint32_t i = 0; i < input_count; ++i)
            input_data[i] = input[i];
        const uint32_t output_cols = state->output_elements / state->input_shape[0];
        Tensor output_tensor = TensorFactory::Create2D(*ArenaPtr(arena), state->input_shape[0], output_cols);
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        state->mlp->forward(input_tensor, output_tensor, *ArenaPtr(arena));
        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->output_elements; ++i)
            output[i] = out_data[i];
    }
    else if (state->kind == NK_NETWORK_CNN)
    {
        float input_buffer[4096] = {};
        if (input_count > 4096)
            return NK_ERR_INVALID_ARGUMENT;
        for (uint32_t i = 0; i < input_count; ++i)
            input_buffer[i] = input[i];
        Tensor input_tensor = MakeNhwcInput(input_buffer,
                                            state->input_shape[0],
                                            state->input_shape[1],
                                            state->input_shape[2]);
        Tensor& output_tensor = state->cnn->forward(input_tensor, *ArenaPtr(arena));
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->output_elements; ++i)
            output[i] = out_data[i];
    }
    else
    {
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    *output_count = state->output_elements;
    return NK_OK;
}

nk_status_t nk_inspect_model(const char* json_path, nk_arena_t* arena, nk_inspect_info_t* info)
{
    if (!json_path || !arena || !info)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));
    const nk_status_t arch_status = nk_parse_architecture(json_path, &info->arch);
    if (arch_status != NK_OK)
        return arch_status;

    nk_model_t model{};
    const nk_status_t load_status = nk_model_load(json_path, arena, &model);
    if (load_status != NK_OK)
        return load_status;

    info->arena_bytes_after_load = nk_arena_used(arena);

    float zero_input[4096] = {};
    if (info->arch.input_elements > 4096)
        return NK_ERR_INVALID_ARGUMENT;

    float output_buffer[4096] = {};
    uint32_t output_count = 0;
    const nk_status_t run_status = nk_model_run(&model,
                                                arena,
                                                zero_input,
                                                info->arch.input_elements,
                                                output_buffer,
                                                4096,
                                                &output_count);
    if (run_status != NK_OK)
        return run_status;

    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));
    float* weights = nullptr;
    std::size_t float_count = 0;
    const char* weight_error = nullptr;
    Arena scratch{};
    alignas(std::max_align_t) unsigned char scratch_buffer[512];
    scratch.init(scratch_buffer, sizeof(scratch_buffer));
    if (ModelLoader::LoadWeightsBin(resolved, scratch, weights, float_count, &weight_error) ==
        ModelLoader::LoadStatus::Ok)
        info->weight_floats = float_count;

    info->arena_bytes_after_forward = nk_arena_used(arena);
    info->arena_remaining = nk_arena_remaining(arena);
    return NK_OK;
}

nk_test_summary_t nk_run_vectors_file(const char* vectors_path)
{
    nk_test_summary_t summary{};
    const VectorsLoader::RunSummary result = VectorsLoader::RunVectorsFile(vectors_path);
    summary.passed = result.passed;
    summary.failed = result.failed;
    return summary;
}

nk_test_summary_t nk_run_all_tests(void)
{
    nk_test_summary_t summary{};
    const VectorsLoader::RunSummary result = run_all_tests();
    summary.passed = result.passed;
    summary.failed = result.failed;
    return summary;
}

int nk_cli_run(int argc, char** argv)
{
    return Cli::Run(argc, argv);
}

} /* extern "C" */
