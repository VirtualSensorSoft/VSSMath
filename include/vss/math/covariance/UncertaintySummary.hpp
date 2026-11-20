#ifndef VSS_MATH_COVARIANCE_UNCERTAINTY_SUMMARY_HPP
#define VSS_MATH_COVARIANCE_UNCERTAINTY_SUMMARY_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>

namespace vss::math::covariance {

/// @brief Transforms an ordered cross-covariance block through two linear maps.
/// @details For left map L (m×a), cross-covariance C (a×b), and right map R
///          (n×b), returns L C Rᵀ (m×n). C is a rectangular signed cross block,
///          not a standalone covariance, so it is not subjected to a PSD test.
///          Inputs are borrowed, may alias, and are never modified. Empty
///          contraction dimensions produce zero sums while preserving the
///          requested output shape.
/// @param left Finite m×a map for the left random vector.
/// @param crossCovariance Finite a×b cross-covariance in the declared source
///        coordinate orders.
/// @param right Finite n×b map for the right random vector.
/// @param limits Output and scratch element/byte limits.
/// @return The owned m×n transformed cross-covariance or a shape, finite,
///         resource or representability failure without partial output.
[[nodiscard]] Result<Matrix>
crossCovarianceFromMaps(ConstMatrixView left, ConstMatrixView crossCovariance,
                        ConstMatrixView right, Limits limits);

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_UNCERTAINTY_SUMMARY_HPP
