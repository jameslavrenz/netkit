#include "cnn.hpp"

NkOpCode ToOpCode(CnnBlockType block_type)
{
    switch (block_type)
    {
        case CnnBlockType::Conv2D:
            return NkOpCode::Conv2D;
        case CnnBlockType::MaxPool2D:
            return NkOpCode::MaxPool2D;
        case CnnBlockType::AvgPool2D:
            return NkOpCode::AvgPool2D;
        case CnnBlockType::BatchNorm2d:
            return NkOpCode::BatchNorm2d;
        case CnnBlockType::Flatten:
            return NkOpCode::Flatten;
        case CnnBlockType::Dense:
            return NkOpCode::Dense;
    }
    return NkOpCode::Dense;
}
