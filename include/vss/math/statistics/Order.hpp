#ifndef VSS_MATH_STATISTICS_ORDER_HPP
#define VSS_MATH_STATISTICS_ORDER_HPP

#include <cstdint>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>

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

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_ORDER_HPP
