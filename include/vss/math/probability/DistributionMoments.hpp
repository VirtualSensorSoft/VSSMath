#ifndef VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP
#define VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP

#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability {

class JointDistribution;

[[nodiscard]] Result<Scalar> mean(const Distribution& law,
                                  ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<Scalar> variance(const Distribution& law,
                                      ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<Vector> mean(const JointDistribution& law,
                                  ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<Matrix> covariance(const JointDistribution& law,
                                        ProbabilityPolicy policy,
                                        Limits limits);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_DISTRIBUTION_MOMENTS_HPP
