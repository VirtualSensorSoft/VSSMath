#ifndef VSS_MATH_PROBABILITY_CHI_SQUARE_HPP
#define VSS_MATH_PROBABILITY_CHI_SQUARE_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>

namespace vss::math::probability {

/// @brief Computes the chi-square upper-tail survival probability.
/// @param chiSquare Non-negative chi-square statistic.
/// @param degreesOfFreedom Degrees of freedom in the supported range
///        `[1,10000]`.
/// @return The probability in `[0,1]`, or a documented domain/numerical
///         failure.
/// @note Complexity: O(min(10000, convergence iterations)) time and O(1)
///       auxiliary storage. No external numerical library is required.
[[nodiscard]] Result<Scalar> chiSquareSurvival(Scalar chiSquare,
                                               Extent degreesOfFreedom);

/// @brief Computes the natural logarithm of the chi-square upper tail.
/// @param chiSquare Non-negative chi-square statistic.
/// @param degreesOfFreedom Degrees of freedom in the supported range
///        `[1,10000]`.
/// @return A finite logarithmic probability, or a documented failure when the
///         finite-result contract cannot be met.
/// @note Complexity: O(min(10000, convergence iterations)) time and O(1)
///       auxiliary storage.
[[nodiscard]] Result<Scalar> chiSquareLogSurvival(Scalar chiSquare,
                                                  Extent degreesOfFreedom);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_CHI_SQUARE_HPP
