#ifndef VSS_MATH_FUNCTIONS_HPP
#define VSS_MATH_FUNCTIONS_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Applies the scalar exponential smoothing recurrence to a sequence.
/// @param values Finite observations in chronological input order.
/// @param alpha Smoothing coefficient in the closed interval [0,1].
/// @param initial State immediately before the first observation.
/// @param limits Output allocation limits.
/// @return One owned updated state per observation, or a validation/resource
///         error. No timestamps or missing-value policy are inferred.
[[nodiscard]] Result<Vector> exponentialSmooth(ConstVectorView values,
                                               Scalar alpha, Scalar initial,
                                               Limits limits);

} // namespace vss::math

#endif // VSS_MATH_FUNCTIONS_HPP
