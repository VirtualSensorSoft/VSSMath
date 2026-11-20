#ifndef VSS_MATH_COVARIANCE_UNCERTAINTY_SUMMARY_HPP
#define VSS_MATH_COVARIANCE_UNCERTAINTY_SUMMARY_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>

namespace vss::math {

/// @brief Componentwise standard and expanded uncertainty for a propagated
///        estimate, optionally retaining caller-supplied interval bounds.
/// @details Bounds are copied unchanged and do not imply a probability or
///          coverage level. `coverage_factor` applies only to `expanded`.
struct UncertaintySummary final {
  /// @brief Nominal output values, copied from the propagation result.
  Vector nominal;
  /// @brief Component standard uncertainties from covariance diagonals.
  Vector standard;
  /// @brief Standard uncertainty multiplied by coverage_factor.
  Vector expanded;
  /// @brief Optional caller-supplied lower bounds; no coverage is inferred.
  std::optional<Vector> lower;
  /// @brief Optional caller-supplied upper bounds; no coverage is inferred.
  std::optional<Vector> upper;
  /// @brief Explicit multiplier used only for expanded uncertainty.
  Scalar coverage_factor;
};

} // namespace vss::math

namespace vss::math::covariance {

/// @brief Summarizes a propagated estimate's componentwise uncertainty.
/// @details Revalidates the mean and actual covariance instead of trusting
///          producer diagnostics. The covariance must be finite, symmetric
///          within the scaled arithmetic allowance, and have no materially
///          negative eigenvalue. It is never projected or otherwise repaired.
///          Standard uncertainty is the square root of each covariance
///          diagonal; expanded uncertainty is that value times the supplied
///          finite, nonnegative factor. Optional bounds must be supplied as a
///          pair, be finite and ordered componentwise; they are not assigned
///          an inferred coverage probability.
/// @param value Propagated estimate; borrowed and unchanged.
/// @param coverageFactor Finite nonnegative multiplier for expanded values.
/// @param lowerQuantile Optional lower bounds, paired with upperQuantile.
/// @param upperQuantile Optional upper bounds, paired with lowerQuantile.
/// @param limits Peak scalar-element and byte limits for spectral workspace
///        and owned result vectors.
/// @return An owned summary or a validation, resource or representability
///         failure without a partial result.
[[nodiscard]] Result<UncertaintySummary>
summarizeUncertainty(const PropagationResult& value, Scalar coverageFactor,
                     std::optional<ConstVectorView> lowerQuantile,
                     std::optional<ConstVectorView> upperQuantile,
                     Limits limits);

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
