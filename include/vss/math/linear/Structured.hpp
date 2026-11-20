#ifndef VSS_MATH_LINEAR_STRUCTURED_HPP
#define VSS_MATH_LINEAR_STRUCTURED_HPP

#include <functional>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

namespace vss::math::linear {

/// @brief Synchronous callback description of a fixed real linear map.
/// @details The caller owns all state captured by the callbacks. Callback input
///          views expire on return; returned vectors are owned by the result.
///          The callbacks are never retained by a numerical operation.
struct LinearMap final {
  /// @brief Number of output coordinates.
  Extent rows;
  /// @brief Number of input coordinates.
  Extent columns;
  /// @brief Applies the map to a vector of length `columns`.
  std::function<Result<Vector>(ConstVectorView)> apply;
  /// @brief Optional Euclidean adjoint action, from rows to columns.
  std::function<Result<Vector>(ConstVectorView)> transpose;
};

/// @brief Projects a vector onto the retained left singular span of a basis.
[[nodiscard]] Result<Vector> orthogonalProjection(ConstMatrixView basis,
                                                  ConstVectorView vector,
                                                  SolvePolicy policy,
                                                  Limits limits);

/// @brief Solves a square compact-band system for one or more right-hand sides.
[[nodiscard]] Result<LinearEstimate> solveBanded(const BandMatrix& band,
                                                 ConstMatrixView rhs,
                                                 SolvePolicy policy,
                                                 Limits limits);

/// @brief Solves a block-diagonal least-squares system with one global cutoff.
[[nodiscard]] Result<LinearEstimate>
solveBlockDiagonal(const BlockDiagonal& matrix, ConstMatrixView rhs,
                   SolvePolicy policy, Limits limits);

/// @brief Solves a bounded square-root augmented regularized least-squares
/// problem.
[[nodiscard]] Result<LinearEstimate>
regularizedLeastSquares(ConstMatrixView design, ConstMatrixView observations,
                        const LinearMap& penalty, Scalar lambda,
                        SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_STRUCTURED_HPP
