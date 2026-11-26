#ifndef VSS_MATH_STATISTICS_CIRCULAR_HPP
#define VSS_MATH_STATISTICS_CIRCULAR_HPP

#include <optional>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {

/// @brief Circular mean and dispersion for one common angular period.
/// @details `resultant_length` is the normalized weighted phasor magnitude and
///          `circular_variance` is one minus that magnitude. Direction is
///          absent when phase is unresolved at the documented epsilon cutoff.
struct CircularEstimate final {
  /// @brief Principal mean direction in `[0,period)`, when resolved.
  std::optional<Scalar> direction;
  /// @brief Length of the normalized resultant phasor in `[0,1]`.
  Scalar resultant_length;
  /// @brief Circular variance `1-resultant_length` in `[0,1]`.
  Scalar circular_variance;
  /// @brief Weight and support assumptions, including unresolved warning.
  EstimatorDiagnostics diagnostics;
};

/// @brief Estimates a circular mean using weighted unit phasors.
/// @param angles Borrowed finite observations, each interpreted modulo
/// `period`.
/// @param weights Unit weights or nonnegative empirical row weights.
/// @param period Strictly positive finite period.
/// @param limits Bounds returned diagnostics and fixed workspace.
/// @return Circular mean, resultant length, variance and support diagnostics.
/// @retval ErrorCode::InvalidDomain Invalid weights, zero total weight or
/// period.
/// @retval ErrorCode::DimensionMismatch Row weights do not match angles.
/// @retval ErrorCode::NonFiniteInput Non-finite angle or weight.
/// @retval ErrorCode::ResourceLimit Diagnostic/workspace storage exceeds
/// limits.
/// @details The phasor phase is `2*pi*remainder(angle,period)/period`. When the
///          resultant is at or below `32*epsilon`, direction is omitted and an
///          `IllConditioned` warning is retained while dispersion remains
///          valid.
[[nodiscard]] Result<CircularEstimate>
circularStatistics(ConstVectorView angles, const WeightSpec& weights,
                   Scalar period, Limits limits);

} // namespace vss::math::statistics

namespace vss::math {
using statistics::CircularEstimate;
using statistics::circularStatistics;
using statistics::WeightSpec;
} // namespace vss::math

#endif
