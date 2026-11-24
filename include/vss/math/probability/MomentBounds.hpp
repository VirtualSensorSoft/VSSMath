#ifndef VSS_MATH_PROBABILITY_MOMENT_BOUNDS_HPP
#define VSS_MATH_PROBABILITY_MOMENT_BOUNDS_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>

namespace vss::math::probability {

/// @brief Bound the two-sided deviation probability using only variance.
[[nodiscard]] Result<Scalar> chebyshevBound(Scalar variance, Scalar radius);

/// @brief Bound a one-sided deviation probability using only variance.
[[nodiscard]] Result<Scalar> cantelliBound(Scalar variance, Scalar radius);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_MOMENT_BOUNDS_HPP
