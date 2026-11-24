#ifndef VSS_MATH_PROBABILITY_LIKELIHOOD_HPP
#define VSS_MATH_PROBABILITY_LIKELIHOOD_HPP

#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/Gaussian.hpp>

namespace vss::math::probability {

/// @brief Evaluates the log likelihood of independent scalar observations.
/// @param law Scalar distribution shared by every observation.
/// @param independentDraws Borrowed finite observation sequence.
/// @param policy Bounds probability evaluation work.
/// @return Compensated sum of log densities or masses; zero probability is
///         represented by negative infinity.
[[nodiscard]] Result<Scalar> logLikelihood(const Distribution& law,
                                           ConstVectorView independentDraws,
                                           ProbabilityPolicy policy);

[[nodiscard]] Result<Scalar> logLikelihood(const JointDistribution& law,
                                           ConstVectorView jointDraw,
                                           ProbabilityPolicy policy);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_LIKELIHOOD_HPP
