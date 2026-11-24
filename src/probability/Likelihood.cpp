#include <cmath>
#include <limits>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/Likelihood.hpp>

namespace vss::math::probability {

Result<Scalar> logLikelihood(const Distribution& law,
                             ConstVectorView independentDraws,
                             ProbabilityPolicy policy) {
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "likelihood tolerances must be positive");
  }
  if (policy.iteration.max_evaluations == 0U ||
      independentDraws.size() > policy.iteration.max_evaluations) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "likelihood evaluation cap exceeded");
  }
  long double sum = 0.0L;
  long double correction = 0.0L;
  for (Index index = 0U; index < independentDraws.size(); ++index) {
    const Scalar observation = independentDraws.at(index);
    if (!std::isfinite(observation)) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                     "likelihood observation is not finite");
    }
    auto value = evaluate(law, observation, policy);
    if (!value) {
      return Result<Scalar>::failure(value.status().code(),
                                     std::string(value.status().message()));
    }
    const Scalar logValue = value.value().log_density_or_mass;
    if (logValue == -std::numeric_limits<Scalar>::infinity()) {
      return Result<Scalar>::success(logValue);
    }
    if (logValue == std::numeric_limits<Scalar>::infinity()) {
      return Result<Scalar>::success(logValue);
    }
    const long double adjusted =
        static_cast<long double>(logValue) - correction;
    const long double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  if (!std::isfinite(sum) ||
      std::abs(sum) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "log likelihood is outside scalar range");
  }
  return Result<Scalar>::success(static_cast<Scalar>(sum));
}

} // namespace vss::math::probability
