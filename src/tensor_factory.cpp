#include "tensor_factory.hpp"
#ifndef NETKIT_DISABLE_IOSTREAM
#include <algorithm>
#include <iomanip>
#endif

namespace TensorFactory
{
    void Print(const Tensor& t)
    {
#ifndef NETKIT_DISABLE_IOSTREAM
        PrintLabeled("Tensor", t);
#else
        (void)t;
#endif
    }

    void PrintLabeled(const char* label, const Tensor& t, uint32_t max_values)
    {
#ifndef NETKIT_DISABLE_IOSTREAM
        std::cout << label << ": shape=[";

        for (uint32_t i = 0; i < t.rank; i++)
        {
            std::cout << t.shape[i];
            if (i + 1 < t.rank)
                std::cout << ", ";
        }

        std::cout << "] values=[";

        if (!t.data || t.num_elements == 0)
        {
            std::cout << "empty";
        }
        else
        {
            const uint32_t print_count =
                max_values == 0 ? t.num_elements : std::min(t.num_elements, max_values);
            if (t.type == DataType::Int8)
            {
                const int8_t* p = static_cast<const int8_t*>(t.data);
                for (uint32_t i = 0; i < print_count; i++)
                {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << static_cast<int>(p[i]);
                }
            }
            else
            {
                const float* p = static_cast<const float*>(t.data);
                std::cout << std::fixed << std::setprecision(4);
                for (uint32_t i = 0; i < print_count; i++)
                {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << p[i];
                }
            }
            if (print_count < t.num_elements)
                std::cout << ", ... (" << t.num_elements << " total)";
        }

        std::cout << "]\n" << std::flush;
#else
        (void)label;
        (void)t;
#endif
    }

    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols)
    {
        Tensor t;

        t.type = DataType::Float32;
        t.rank = 2;

        t.shape[0] = rows;
        t.shape[1] = cols;

        t.stride[0] = cols;
        t.stride[1] = 1;

        t.num_elements = rows * cols;
        t.bytes = t.num_elements * sizeof(float);

        t.data = arena.alloc(t.bytes, alignof(float));

        return t;
    }

    Tensor CreateND(Arena& arena, uint32_t rank, std::span<const uint32_t> shape)
    {
        Tensor t;

        t.type = DataType::Float32;
        t.rank = rank;

        uint32_t num_elements = 1;

        for (uint32_t i = 0; i < rank; i++)
        {
            t.shape[i] = shape[i];
            num_elements *= shape[i];
        }

        uint32_t stride_val = 1;
        for (int i = static_cast<int>(rank) - 1; i >= 0; i--)
        {
            t.stride[i] = stride_val;
            stride_val *= shape[i];
        }

        t.num_elements = num_elements;
        t.bytes = num_elements * sizeof(float);

        t.data = arena.alloc(t.bytes, alignof(float));

        return t;
    }

    Tensor View2D(float* data, uint32_t rows, uint32_t cols)
    {
        Tensor t{};

        t.data = data;
        t.type = DataType::Float32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = cols;
        t.stride[0] = cols;
        t.stride[1] = 1;
        t.num_elements = rows * cols;
        t.bytes = t.num_elements * sizeof(float);

        return t;
    }

    Tensor View2DInt8(int8_t* data, uint32_t rows, uint32_t cols)
    {
        Tensor t{};

        t.data = data;
        t.type = DataType::Int8;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = cols;
        t.stride[0] = cols;
        t.stride[1] = 1;
        t.num_elements = rows * cols;
        t.bytes = t.num_elements * sizeof(int8_t);

        return t;
    }

    Tensor View3DInt8(int8_t* data, uint32_t depth, uint32_t rows, uint32_t cols)
    {
        Tensor t{};

        t.data = data;
        t.type = DataType::Int8;
        t.rank = 3;
        t.shape[0] = depth;
        t.shape[1] = rows;
        t.shape[2] = cols;
        t.stride[0] = rows * cols;
        t.stride[1] = cols;
        t.stride[2] = 1;
        t.num_elements = depth * rows * cols;
        t.bytes = t.num_elements * sizeof(int8_t);

        return t;
    }

    Tensor View1DInt32(int32_t* data, uint32_t length)
    {
        Tensor t{};

        t.data = data;
        t.type = DataType::Int32;
        t.rank = 2;
        t.shape[0] = 1;
        t.shape[1] = length;
        t.stride[0] = length;
        t.stride[1] = 1;
        t.num_elements = length;
        t.bytes = t.num_elements * sizeof(int32_t);

        return t;
    }

    Tensor ViewND(float* data, uint32_t rank, std::span<const uint32_t> shape)
    {
        Tensor t{};

        t.data = data;
        t.type = DataType::Float32;
        t.rank = rank;

        uint32_t num_elements = 1;
        for (uint32_t i = 0; i < rank; ++i)
        {
            t.shape[i] = shape[i];
            num_elements *= shape[i];
        }

        uint32_t stride_val = 1;
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i)
        {
            t.stride[i] = stride_val;
            stride_val *= t.shape[i];
        }

        t.num_elements = num_elements;
        t.bytes = num_elements * sizeof(float);

        return t;
    }

    void Fill(Tensor& t, std::span<const float> values)
    {
        float* p = static_cast<float*>(t.data);
        uint32_t i = 0;
        for (float v : values)
        {
            if (i >= t.num_elements)
                break;
            p[i++] = v;
        }
    }
}
