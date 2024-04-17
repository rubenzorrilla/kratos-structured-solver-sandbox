#include <array>
#include <utility>
#include <vector>
#include "include/experimental/mdspan"

#pragma once

class MdspanUtilities
{
public:

    template<class TInput1, class TInput2, class TOutput>
    static void Mult(
        const TInput1& rA,
        const TInput2& rB,
        TOutput& rOut)
    {
        Mult(1.0, rA, rB, rOut);
    }

    template<class TInput1, class TInput2, class TOutput>
    static void Mult(
        const double Factor,
        const TInput1& rA,
        const TInput2& rB,
        TOutput& rOut)
    {
        if (rA.extent(1) != rB.extent(0)) {
            throw std::logic_error("Mult: A and B input matrices do not conform.");
        }
        if (rA.extent(0) != rOut.extent(0) | rB.extent(1) != rOut.extent(1)) {
            throw std::logic_error("Mult: Output extent is wrong.");
        }

        for (std::size_t i = 0; i < rA.extent(0); ++i) {
            for (std::size_t j = 0; j < rB.extent(1); ++j) {
                rOut(i,j) = 0.0;
                for (std::size_t k = 0; k < rA.extent(1); ++k) {
                    rOut(i,j) += rA(i,k) * rB(k,j);
                }
                rOut(i,j) *= Factor;
            }
        }
    }

    template<class TInput1, class TInput2, class TOutput>
    static void TransposeMult(
        const TInput1& rA,
        const TInput2& rB,
        TOutput& rOut)
    {
        TransposeMult(1.0, rA, rB, rOut);
    }

    template<class TInput1, class TInput2, class TOutput>
    static void TransposeMult(
        const double Factor,
        const TInput1& rA,
        const TInput2& rB,
        TOutput& rOut)
    {
        if (rA.extent(0) != rB.extent(0)) {
            throw std::logic_error("TransposeMult: A and B input matrices do not conform.");
        }
        if (rA.extent(1) != rOut.extent(0) | rB.extent(1) != rOut.extent(1)) {
            throw std::logic_error("TransposeMult: Output extent is wrong.");
        }

        for (std::size_t i = 0; i < rA.extent(1); ++i) {
            for (std::size_t j = 0; j < rB.extent(1); ++j) {
                rOut(i,j) = 0.0;
                for (std::size_t k = 0; k < rA.extent(0); ++k) {
                    rOut(i,j) += rA(i,k) * rB(k,j);
                }
                rOut(i,j) *= Factor;
            }
        }
    }

};