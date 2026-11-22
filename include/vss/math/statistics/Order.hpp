#ifndef VSS_MATH_STATISTICS_ORDER_HPP
#define VSS_MATH_STATISTICS_ORDER_HPP

#include <cstdint>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math {

/// @brief Selects the empirical convention used to evaluate a quantile.
enum class QuantileRule : std::uint8_t {
  /// @brief Linear interpolation at rank (N - 1) p.
  linear,
  /// @brief First sorted observation whose empirical CDF is at least p.
  inverse_empirical
};

} // namespace vss::math

namespace vss::math::statistics {

/// @brief Selects the variance convention and requested summary quantiles.
/// @details Quantile interpolation is selected from the weight rule: unweighted
///          and frequency weights use the linear type-7 convention; general
///          empirical, fixed-IID, and importance masses use inverse empirical
///          quantiles. The probability range is borrowed for the call.
struct SummarySpec final {
  /// @brief Probabilities in [0, 1], retained in this order in the output.
  std::span<const Scalar> quantile_probabilities;
  /// @brief Empirical population spread or compatible IID sample correction.
  VarianceMode variance_mode{VarianceMode::empirical};
};

/// @brief Owns complete common-support descriptive summaries by coordinate.
/// @details Quantiles have shape d-by-Q. Variance and standard-deviation
///          elements whose status is not None are unavailable and stored as
///          zero placeholders; callers must inspect @ref variance_status.
struct Summary final {
  /// @brief Weighted sum of each coordinate, using the supplied row masses.
  Vector sum;
  /// @brief Minimum positive-weight observation for each coordinate.
  Vector minimum;
  /// @brief Maximum positive-weight observation for each coordinate.
  Vector maximum;
  /// @brief Square root of the weighted empirical mean of squared values.
  Vector rms;
  /// @brief Weighted empirical mean for each coordinate.
  Vector mean;
  /// @brief Requested empirical or corrected variance; inspect its status.
  Vector variance;
  /// @brief Requested quantiles, with rows for coordinates and columns for
  ///        probabilities in @ref SummarySpec::quantile_probabilities order.
  Matrix quantiles;
  /// @brief Weight interpretation, support, estimand, and estimator premises.
  EstimatorDiagnostics diagnostics;
  /// @brief Maximum minus minimum; non-finite range fails the whole call.
  Vector range{};
  /// @brief Square root of each available variance.
  Vector standard_deviation{};
  /// @brief ErrorCode::None or the per-coordinate variance-unavailable cause.
  Array<ErrorCode> variance_status{};
};

/// @brief Evaluates an empirical quantile of finite observations.
/// @param values Borrowed observations; they are not modified.
/// @param probability Probability in the closed interval [0, 1].
/// @param rule Empirical quantile convention.
/// @param limits Per-call scratch and result storage limits.
/// @return The selected quantile, or a status for invalid input, insufficient
///         observations, an unsupported rule, or an exceeded resource limit.
/// @note Complexity: O(N log N) time and O(N) auxiliary storage. Input is
///       validated before scratch allocation.
[[nodiscard]] Result<Scalar> quantile(ConstVectorView values,
                                      Scalar probability, QuantileRule rule,
                                      Limits limits);

/// @brief Evaluates the median using the linear empirical quantile rule.
/// @param values Borrowed finite observations; they are not modified.
/// @param limits Per-call scratch and result storage limits.
/// @return The median or a documented input/resource failure.
/// @note Complexity: O(N log N) time and O(N) auxiliary storage.
[[nodiscard]] Result<Scalar> median(ConstVectorView values, Limits limits);

/// @brief Computes a symmetrically trimmed arithmetic mean.
/// @param values Borrowed finite observations; they are not modified.
/// @param fraction Fraction in [0, 1/2); floor(N*fraction) observations are
///        removed from each tail.
/// @param limits Per-call scratch and result storage limits.
/// @return The mean of the retained observations or a documented failure.
/// @note Complexity: O(N log N) time and O(N) auxiliary storage.
[[nodiscard]] Result<Scalar> trimmedMean(ConstVectorView values,
                                         Scalar fraction, Limits limits);

/// @brief Computes a symmetrically winsorized arithmetic mean.
/// @param values Borrowed finite observations; they are not modified.
/// @param fraction Fraction in [0, 1/2); floor(N*fraction) tail values are
///        replaced by the nearest retained order statistic.
/// @param limits Per-call scratch and result storage limits.
/// @return The winsorized mean or a documented failure.
/// @note Complexity: O(N log N) time and O(N) auxiliary storage.
[[nodiscard]] Result<Scalar> winsorizedMean(ConstVectorView values,
                                            Scalar fraction, Limits limits);

/// @brief Computes the geometric mean of strictly positive finite observations.
/// @param values Borrowed observations; they are not modified.
/// @return The geometric mean or an insufficient-sample, domain, non-finite
///         input, or numerical-result failure.
/// @note Complexity: O(N) time and O(1) auxiliary storage. The logarithmic
///       reduction is referenced to a power-of-two scale.
[[nodiscard]] Result<Scalar> geometricMean(ConstVectorView values);

/// @brief Computes the harmonic mean of strictly positive finite observations.
/// @param values Borrowed observations; they are not modified.
/// @return The harmonic mean or an insufficient-sample, domain, non-finite
///         input, or numerical-result failure.
/// @note Complexity: O(N) time and O(1) auxiliary storage. Reciprocal terms
///       are scaled by the smallest observation.
[[nodiscard]] Result<Scalar> harmonicMean(ConstVectorView values);

/// @brief Computes the scaled median absolute deviation of finite observations.
/// @param values Borrowed observations; they are not modified.
/// @param scale Finite nonnegative scale applied to the unscaled MAD.
/// @param limits Per-call scratch and result storage limits.
/// @return The scaled MAD, or a documented input, resource, or range failure.
/// @note The center and deviation median both use the linear empirical median.
///       Complexity is O(N log N) time and O(N) auxiliary storage.
[[nodiscard]] Result<Scalar>
medianAbsoluteDeviation(ConstVectorView values, Scalar scale, Limits limits);

/// @brief Computes the interquartile range under the selected quantile rule.
/// @param values Borrowed observations; they are not modified.
/// @param rule Empirical quantile convention used at probabilities 0.25/0.75.
/// @param limits Per-call scratch and result storage limits.
/// @return q(0.75)-q(0.25), or a documented input, resource, or range failure.
/// @note Complexity is O(N log N) time and O(N) auxiliary storage.
[[nodiscard]] Result<Scalar>
interquartileRange(ConstVectorView values, QuantileRule rule, Limits limits);

/// @brief Computes a weighted coordinate-wise descriptive summary.
/// @param sample Complete finite N-by-d observations and their shape metadata.
/// @param weights Unweighted, exact frequency, empirical/fixed-IID row, or
///        importance weights as specified by C114.
/// @param spec Requested quantiles and variance convention.
/// @param limits Maximum result and simultaneously required workspace storage.
/// @return Summary values and common-support interpretation diagnostics.
/// @details Complexity is O(Nd + dQ + dN log N) for Q quantiles; frequency
///          quantiles use cumulative integer counts without expanding rows.
[[nodiscard]] Result<Summary> descriptiveSummary(const SampleView& sample,
                                                 const WeightSpec& weights,
                                                 const SummarySpec& spec,
                                                 Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_ORDER_HPP
