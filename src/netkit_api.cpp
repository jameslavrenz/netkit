#include "netkit.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include "ops.hpp"
#include "conv2d.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#if defined(NETKIT_DESKTOP)
#include "nk_regression.hpp"
#include "cli.hpp"
#include "test.hpp"
#endif
#include <cstdio>
#include <cstring>
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

    nk_status_t ParseNkModel(const char* nk_path, NkLoader::ParsedModel& parsed, const char** resolved_out)
    {
        static thread_local char path_buffer[NkLoader::kMaxPathLen];
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
    c.pad_h = conv->pad_h;
    c.pad_w = conv->pad_w;
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
                                   int pad_w)
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
                                    pad_w);
    return NK_OK;
}

nk_status_t nk_cnn_init_pool_layer(nk_cnn_t* cnn,
                                   uint32_t layer_idx,
                                   int pool_size,
                                   int stride,
                                   int pad_h,
                                   int pad_w)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitPoolLayer(layer_idx, pool_size, stride, pad_h, pad_w);
    return NK_OK;
}

nk_status_t nk_cnn_init_avg_pool_layer(nk_cnn_t* cnn,
                                       uint32_t layer_idx,
                                       int pool_size,
                                       int stride,
                                       int pad_h,
                                       int pad_w)
{
    if (!nk_cnn_is_valid(cnn))
        return NK_ERR_NOT_INITIALIZED;
    CnnPtr(cnn)->net->InitAvgPoolLayer(layer_idx, pool_size, stride, pad_h, pad_w);
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
    FillArchInfo(arch, info);
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
    {
        NkLoader::ArchInfo arch{};
        NkLoader::FillArchInfo(parsed, arch);
        FillArchInfo(arch, info);
    }
    SetLastError(nullptr);
    return NK_OK;
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
    {
        NkLoader::ArchInfo arch{};
        NkLoader::FillArchInfo(parsed, arch);
        FillArchInfo(arch, info);
    }
    SetLastError(nullptr);
    return NK_OK;
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
    {
        NkLoader::ArchInfo arch{};
        NkLoader::FillArchInfo(parsed, arch);
        FillArchInfo(arch, info);
    }
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
    FillArchInfo(arch, &state->arch);
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
        Tensor output_tensor = TensorFactory::Create2D(*ArenaPtr(arena), state->arch.input_shape[0], output_cols);
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        state->mlp->forward(input_tensor, output_tensor, *ArenaPtr(arena));
        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->arch.output_elements; ++i)
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
                                            state->arch.input_shape[0],
                                            state->arch.input_shape[1],
                                            state->arch.input_shape[2]);
        Tensor& output_tensor = state->cnn->forward(input_tensor, *ArenaPtr(arena));
        if (!output_tensor.data)
            return NK_ERR_ARENA_OVERFLOW;
        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->arch.output_elements; ++i)
            output[i] = out_data[i];
    }
    else
    {
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    *output_count = state->arch.output_elements;
    return NK_OK;
}

nk_status_t nk_inspect_model(const char* nk_path, nk_arena_t* arena, nk_inspect_info_t* info)
{
    if (!nk_path || !arena || !info)
        return NK_ERR_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));
    const nk_status_t arch_status = nk_parse_architecture(nk_path, &info->arch);
    if (arch_status != NK_OK)
        return arch_status;

    nk_model_t model{};
    const nk_status_t load_status = nk_model_load(nk_path, arena, &model);
    if (load_status != NK_OK)
        return load_status;

    info->arena_bytes_after_load = nk_arena_used(arena);
    info->weight_floats = info->arch.expected_weight_floats;

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

    info->arena_bytes_after_forward = nk_arena_used(arena);
    info->arena_remaining = nk_arena_remaining(arena);
    return NK_OK;
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
