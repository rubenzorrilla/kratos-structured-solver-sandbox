#include <vector>

#pragma once

template<int TDim>
class PressurePreconditioner
{
public:

    using SizeType = std::size_t;

    using IndexType = std::size_t;

    using VectorType = std::vector<double>;

    PressurePreconditioner() = default;

    virtual void SetUp()
    {
    }

    virtual void Apply(
        const VectorType& rInput,
        VectorType& rOutput)
    {
        for (IndexType i = 0; i < rInput.size(); ++i) {
            rOutput[i] = rInput[i];
        }
    }

    virtual void Clear()
    {
    }

};