#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "mesh_utilities.hpp"
#include "operators.hpp"

#pragma once

template<int TDim>
class MatrixReplacement;

// using Eigen::SparseMatrix;

namespace Eigen::internal
{
    // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
    template<>
    struct traits<MatrixReplacement<2>> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};

    // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
    template<>
    struct traits<MatrixReplacement<3>> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
}

template<int TDim>
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement<TDim>>
{
public:

    // Required typedefs, constants, and method:
    typedef double Scalar;

    typedef double RealScalar;

    typedef int StorageIndex;

    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Eigen::Index rows() const
    {
        return mProblemSize;
    }

    Eigen::Index cols() const
    {
        return mProblemSize;
    }

    template<typename Rhs>
    Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const
    {
        return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    MatrixReplacement(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVectorInv)
        : mProblemSize(std::get<1>(MeshUtilities<TDim>::CalculateMeshData(rBoxDivisions)))
        , pBoxDivisions(&rBoxDivisions)
        , pCellSize(&rCellSize)
        , pActiveCells(&rActiveCells)
        , pLumpedMassVectorInv(&rLumpedMassVectorInv)
        {}

    const std::array<int, TDim>& GetBoxDivisions() const
    {
        return *pBoxDivisions;
    }

    const std::array<double, TDim>& GetCellSize() const
    {
        return *pCellSize;
    }

    const Eigen::Array<bool, Eigen::Dynamic, 1>& GetActiveCells() const
    {
        return *pActiveCells;
    }

    const Eigen::Array<double, Eigen::Dynamic, TDim>& GetLumpedMassVectorInv() const
    {
        return *pLumpedMassVectorInv;
    }

private:

    const unsigned int mProblemSize;
    const std::array<int, TDim>* pBoxDivisions = nullptr;
    const std::array<double, TDim>* pCellSize = nullptr;
    const Eigen::Array<bool, Eigen::Dynamic, 1>* pActiveCells = nullptr;
    const Eigen::Array<double, Eigen::Dynamic, TDim>* pLumpedMassVectorInv = nullptr;

};

// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen::internal {

template<typename Rhs>
struct generic_product_impl<MatrixReplacement<2>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
: generic_product_impl_base<MatrixReplacement<2>,Rhs,generic_product_impl<MatrixReplacement<2>,Rhs> >
{
    typedef typename Product<MatrixReplacement<2>,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const MatrixReplacement<2>& lhs, const Rhs& rhs, const Scalar& alpha)
    {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        assert(alpha==Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        // but let's do something fancier (and less efficient):
        Eigen::VectorXd aux;
        Operators<2>::ApplyPressureOperator(
            lhs.GetBoxDivisions(),
            lhs.GetCellSize(),
            lhs.GetActiveCells(),
            lhs.GetLumpedMassVectorInv(),
            rhs,
            aux);
        dst += aux;
    }
};

template<typename Rhs>
struct generic_product_impl<MatrixReplacement<3>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
: generic_product_impl_base<MatrixReplacement<3>,Rhs,generic_product_impl<MatrixReplacement<3>,Rhs> >
{
    typedef typename Product<MatrixReplacement<3>,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const MatrixReplacement<3>& lhs, const Rhs& rhs, const Scalar& alpha)
    {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        assert(alpha==Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        // but let's do something fancier (and less efficient):
        Eigen::VectorXd aux;
        Operators<3>::ApplyPressureOperator(
            lhs.GetBoxDivisions(),
            lhs.GetCellSize(),
            lhs.GetActiveCells(),
            lhs.GetLumpedMassVectorInv(),
            rhs,
            aux);
        dst += aux;
    }
    };
}


