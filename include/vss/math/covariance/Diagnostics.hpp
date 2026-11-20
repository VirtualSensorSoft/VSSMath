#ifndef VSS_MATH_COVARIANCE_DIAGNOSTICS_HPP
#define VSS_MATH_COVARIANCE_DIAGNOSTICS_HPP

#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {

/// @brief Returns a selected, ordered principal covariance block.
/// @param covariance Finite square joint PSD covariance, borrowed unchanged.
/// @param keep Nonempty, unique, in-range coordinate indices; output follows
///        this order.
/// @param policy Iteration and spectrum-certification policy. Rank cutoffs and
///        reciprocal-condition floor are ignored because no inversion occurs.
/// @param limits Peak input-validation workspace and output budget.
/// @return The owned principal covariance block, or a validation/resource
///         failure without partial output.
[[nodiscard]] Result<Matrix> marginalCovariance(ConstMatrixView covariance,
                                                std::span<const Index> keep,
                                                SolvePolicy policy,
                                                Limits limits);

/// @brief Computes the symmetric inverse square root on retained covariance
///        support.
/// @details The numeric matrix is stored in `CovarianceResult::covariance`.
///          Classification, rank, cutoff and reciprocal condition refer to the
///          input covariance. Supported rank policies return a zero transform
///          on excluded modes; strict rank policy rejects any excluded mode.
/// @param covariance Finite nonempty square covariance, borrowed unchanged.
/// @param policy Covariance cutoff, retained condition and eigensolve policy.
/// @param limits Peak element and byte budget for the transform and workspace.
/// @return The owned inverse square root with input-spectrum diagnostics.
[[nodiscard]] Result<CovarianceResult>
inverseSquareRoot(ConstMatrixView covariance, SolvePolicy policy,
                  Limits limits);

/// @brief Computes the signed determinant of a finite square real matrix.
/// @details The empty determinant is one. Exact singularity returns positive
///          zero. A nonzero determinant rounded to zero retains its sign.
///          Overflow returns NonFiniteResult. Permissive rank policies use
///          the full determinant. Strict rank and condition checks apply to
///          nonsingular inputs.
/// @param matrix Borrowed square input, unchanged during the operation.
/// @param policy Numerical rank, reciprocal condition, and iteration limits.
/// @param limits Peak element and byte budgets, including exact fallback.
/// @return Signed determinant or an error without a partial result.
[[nodiscard]] Result<Scalar> determinant(ConstMatrixView matrix,
                                         SolvePolicy policy, Limits limits);

/// @brief Computes the natural logarithm of the absolute determinant.
/// @details Returns zero for the empty matrix and negative infinity for
///          exact singularity. Computes the log directly from a scaled
///          magnitude, including when the determinant is outside Scalar range.
/// @param matrix Borrowed square input, unchanged during the operation.
/// @param policy Same rank, reciprocal condition, and iteration rules as
///        determinant.
/// @param limits Peak element and byte budgets, including exact fallback.
/// @return Log magnitude or an error without a partial result.
[[nodiscard]] Result<Scalar> logAbsoluteDeterminant(ConstMatrixView matrix,
                                                    SolvePolicy policy,
                                                    Limits limits);

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_DIAGNOSTICS_HPP
