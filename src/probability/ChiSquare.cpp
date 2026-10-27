#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/NumericalClassification.hpp>
#include <limits>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/ChiSquare.hpp>

namespace vss::math::probability {
namespace {

constexpr Extent maximumDegreesOfFreedom = 10000U;
constexpr Extent maximumIterations = 10000U;

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

Status validateInputs(Scalar chiSquare, Extent degreesOfFreedom) {
  if (!std::isfinite(chiSquare)) {
    return Status::failure(ErrorCode::NonFiniteInput,
                           "chi-square statistic must be finite");
  }
  if (chiSquare < 0.0) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "chi-square statistic must be non-negative");
  }
  if (degreesOfFreedom == 0U || degreesOfFreedom > maximumDegreesOfFreedom) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "chi-square degrees of freedom must be in [1,10000]");
  }
  return Status::success();
}

Scalar smallestPositiveScale() noexcept {
  return std::numeric_limits<Scalar>::min() /
         std::numeric_limits<Scalar>::epsilon();
}

Result<Scalar> noConvergence(const char* branch, Scalar chiSquare,
                             Extent degreesOfFreedom) {
  return Result<Scalar>::failure(
      ErrorCode::NoConvergence,
      std::string("chi-square ") + branch +
          " did not converge for x=" + std::to_string(chiSquare) +
          ", dof=" + std::to_string(degreesOfFreedom));
}

Result<Scalar> normalizeLogProbability(Scalar logProbability) {
  if (!std::isfinite(logProbability)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "chi-square log probability is non-finite");
  }

  const Scalar logDenormalizedMinimum =
      std::log(std::numeric_limits<Scalar>::denorm_min());
  if (!std::isfinite(logDenormalizedMinimum)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square denormalized minimum logarithm is non-finite");
  }
  if (logProbability < logDenormalizedMinimum) {
    return Result<Scalar>::success(logProbability);
  }

  const Scalar candidate = std::exp(logProbability);
  if (!std::isfinite(candidate)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square reconstructed probability is non-finite");
  }
  auto classified = detail::classifyUnitProbability(candidate);
  if (!classified.hasValue()) {
    return copyFailure<Scalar>(classified.status());
  }
  if (classified.value() == 1.0) {
    return Result<Scalar>::success(0.0);
  }
  if (classified.value() == 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square logarithm cannot represent a zero probability");
  }
  return Result<Scalar>::success(logProbability);
}

Result<Scalar> lowerSeriesLogSurvival(Scalar a, Scalar x, Scalar epsilon,
                                      Scalar chiSquare,
                                      Extent degreesOfFreedom) {
  detail::CompensatedSum sum;
  Scalar term = 1.0 / a;
  if (!std::isfinite(term)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "chi-square series term is non-finite");
  }
  sum.add(term);
  if (!std::isfinite(sum.total())) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "chi-square series sum is non-finite");
  }

  bool converged = false;
  for (Extent iteration = 1U; iteration <= maximumIterations; ++iteration) {
    const Scalar denominator = a + static_cast<Scalar>(iteration);
    const Scalar ratio = x / denominator;
    term *= ratio;
    if (!std::isfinite(denominator) || !std::isfinite(ratio) ||
        !std::isfinite(term)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "chi-square series recurrence became non-finite");
    }
    sum.add(term);
    const Scalar total = sum.total();
    if (!std::isfinite(total)) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "chi-square series sum is non-finite");
    }
    if (std::abs(term) <= (8.0 * epsilon * std::abs(total))) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    return noConvergence("lower series", chiSquare, degreesOfFreedom);
  }

  const Scalar logPrefactor = -x + (a * std::log(x)) - std::lgamma(a);
  const Scalar logProbability = logPrefactor + std::log(sum.total());
  if (!std::isfinite(logPrefactor) || !std::isfinite(logProbability)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square lower-series log state is non-finite");
  }
  const Scalar candidate = -std::expm1(logProbability);
  auto classified = detail::classifyUnitProbability(candidate);
  if (!classified.hasValue()) {
    return copyFailure<Scalar>(classified.status());
  }
  if (classified.value() == 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square lower-series tail is not finitely representable");
  }
  return normalizeLogProbability(std::log(classified.value()));
}

void floorContinuedFractionState(Scalar& value, Scalar tiny) noexcept {
  if (value == 0.0) {
    value = tiny;
  } else if (std::abs(value) < tiny) {
    value = std::copysign(tiny, value);
  }
}

Result<Scalar> continuedFractionLogSurvival(Scalar a, Scalar x, Scalar epsilon,
                                            Scalar tiny, Scalar chiSquare,
                                            Extent degreesOfFreedom) {
  Scalar b = x + 1.0 - a;
  Scalar c = 1.0 / tiny;
  Scalar d = 1.0 / b;
  Scalar h = d;
  if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d) ||
      !std::isfinite(h)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square continued-fraction state is non-finite");
  }

  bool converged = false;
  Scalar delta = 0.0;
  for (Extent iteration = 1U; iteration <= maximumIterations; ++iteration) {
    const auto index = static_cast<Scalar>(iteration);
    const Scalar an = -index * (index - a);
    b += 2.0;
    d = (an * d) + b;
    floorContinuedFractionState(d, tiny);
    c = b + (an / c);
    floorContinuedFractionState(c, tiny);
    d = 1.0 / d;
    delta = c * d;
    h *= delta;
    if (!std::isfinite(an) || !std::isfinite(b) || !std::isfinite(c) ||
        !std::isfinite(d) || !std::isfinite(delta) || !std::isfinite(h)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "chi-square continued-fraction recurrence became non-finite");
    }
    if (std::abs(delta - 1.0) <= (8.0 * epsilon)) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    return noConvergence("continued fraction", chiSquare, degreesOfFreedom);
  }
  if (!(h > 0.0)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square continued-fraction result is not positive");
  }

  const Scalar logPrefactor = -x + (a * std::log(x)) - std::lgamma(a);
  const Scalar logProbability = logPrefactor + std::log(h);
  if (!std::isfinite(logPrefactor) || !std::isfinite(logProbability)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square continued-fraction log state is non-finite");
  }
  return normalizeLogProbability(logProbability);
}

Result<Scalar> logSurvivalImpl(Scalar chiSquare, Extent degreesOfFreedom) {
  const Status validation = validateInputs(chiSquare, degreesOfFreedom);
  if (!validation.ok()) {
    return copyFailure<Scalar>(validation);
  }
  if (chiSquare == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  const Scalar a = static_cast<Scalar>(degreesOfFreedom) * 0.5;
  const Scalar x = chiSquare * 0.5;
  if (!std::isfinite(a) || !std::isfinite(x)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "chi-square parameters are non-finite");
  }
  if (x == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  const Scalar tiny = smallestPositiveScale();
  if (!std::isfinite(epsilon) || !std::isfinite(tiny) || tiny <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "chi-square numerical constants invalid");
  }
  if (x < (a + 1.0)) {
    return lowerSeriesLogSurvival(a, x, epsilon, chiSquare, degreesOfFreedom);
  }
  return continuedFractionLogSurvival(a, x, epsilon, tiny, chiSquare,
                                      degreesOfFreedom);
}

} // namespace

Result<Scalar> chiSquareSurvival(Scalar chiSquare, Extent degreesOfFreedom) {
  const Result<Scalar> logResult = logSurvivalImpl(chiSquare, degreesOfFreedom);
  if (!logResult.hasValue()) {
    return copyFailure<Scalar>(logResult.status());
  }
  const Scalar logProbability = logResult.value();
  const Scalar logDenormalizedMinimum =
      std::log(std::numeric_limits<Scalar>::denorm_min());
  if (!std::isfinite(logDenormalizedMinimum)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square denormalized minimum logarithm is non-finite");
  }
  if (logProbability < logDenormalizedMinimum) {
    return Result<Scalar>::success(0.0);
  }
  const Scalar candidate = std::exp(logProbability);
  if (!std::isfinite(candidate)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "chi-square reconstructed probability is non-finite");
  }
  auto classified = detail::classifyUnitProbability(candidate);
  if (!classified.hasValue()) {
    return copyFailure<Scalar>(classified.status());
  }
  return classified;
}

Result<Scalar> chiSquareLogSurvival(Scalar chiSquare, Extent degreesOfFreedom) {
  return logSurvivalImpl(chiSquare, degreesOfFreedom);
}

} // namespace vss::math::probability
