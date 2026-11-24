#ifndef VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP
#define VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP

#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability {

[[nodiscard]] Result<Scalar> mean(const Distribution& law,
                                  ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<Scalar> variance(const Distribution& law,
                                      ProbabilityPolicy policy, Limits limits);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP
