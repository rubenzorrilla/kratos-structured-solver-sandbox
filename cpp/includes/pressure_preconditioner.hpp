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

    virtual void Apply(const std::vector<double>& rInput, std::vector<double>& rOutput)
    {
        std::copy(rInput.begin(), rInput.end(), rOutput.begin());  
    }

    virtual void Clear()
    {
    }

};