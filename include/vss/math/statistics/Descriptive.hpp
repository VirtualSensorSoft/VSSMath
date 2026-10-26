#ifndef VSS_MATH_STATISTICS_DESCRIPTIVE_HPP
#define VSS_MATH_STATISTICS_DESCRIPTIVE_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::statistics {

/// @brief Computes the arithmetic mean of a finite sample.
/// @param values Immutable view of the sample values.
/// @return The mean, or `InvalidExtent`, `NonFiniteInput`, or
///         `NonFiniteResult` when the contract cannot be satisfied.
/// @note Complexity: O(N) time and O(1) auxiliary storage. Values are read in
///       increasing index order and the input is not modified.
[[nodiscard]] Result<Scalar> mean(ConstVectorView values);

/// @brief Computes the population variance of a finite sample.
/// @param values Immutable view of the sample values.
/// @return The variance divided by N, or a documented numerical failure.
/// @note Complexity: O(N) time and O(1) auxiliary storage. The stable
///       scale-normalized two-pass algorithm avoids avoidable overflow.
[[nodiscard]] Result<Scalar> populationVariance(ConstVectorView values);

/// @brief Computes the unbiased sample variance of a finite sample.
/// @param values Immutable view containing at least two values.
/// @return The variance divided by N-1, or `InsufficientSamples` when fewer
///         than two values are supplied.
/// @note Complexity: O(N) time and O(1) auxiliary storage.
[[nodiscard]] Result<Scalar> sampleVariance(ConstVectorView values);

/// @brief Computes the population standard deviation.
/// @param values Immutable view of the sample values.
/// @return The non-negative square root of the population variance.
/// @note Complexity: O(N) time and O(1) auxiliary storage.
[[nodiscard]] Result<Scalar>
populationStandardDeviation(ConstVectorView values);

/// @brief Computes the unbiased sample standard deviation.
/// @param values Immutable view containing at least two values.
/// @return The non-negative square root of the sample variance, or
///         `InsufficientSamples` when fewer than two values are supplied.
/// @note Complexity: O(N) time and O(1) auxiliary storage.
[[nodiscard]] Result<Scalar> sampleStandardDeviation(ConstVectorView values);

/// @brief Computes the root mean square of a finite sample.
/// @param values Immutable view of the sample values.
/// @return The root mean square, or a documented numerical failure.
/// @note Complexity: O(N) time and O(1) auxiliary storage. A scaled
///       sum-of-squares algorithm avoids avoidable overflow and underflow.
[[nodiscard]] Result<Scalar> rootMeanSquare(ConstVectorView values);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_DESCRIPTIVE_HPP
