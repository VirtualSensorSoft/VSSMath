#include <cmath>
#include <limits>
#include <string>
#include <vss/math/Error.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Inference.hpp>

namespace vss::math::statistics {
namespace {

Result<Scalar> varianceEndpoint(Scalar sampleVariance, Scalar degreesOfFreedom,
                                Scalar chiSquareQuantile) {
  int varianceExponent = 0;
  int degreesExponent = 0;
  int quantileExponent = 0;
  const Scalar varianceMantissa = std::frexp(sampleVariance, &varianceExponent);
  const Scalar degreesMantissa = std::frexp(degreesOfFreedom, &degreesExponent);
  const Scalar quantileMantissa =
      std::frexp(chiSquareQuantile, &quantileExponent);
  const Scalar mantissa =
      (varianceMantissa * degreesMantissa) / quantileMantissa;
  int mantissaExponent = 0;
  const Scalar normalizedMantissa = std::frexp(mantissa, &mantissaExponent);
  const int exponent =
      varianceExponent + degreesExponent - quantileExponent + mantissaExponent;
  const Scalar endpoint = std::scalbn(normalizedMantissa, exponent);
  if (!std::isfinite(endpoint)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "variance confidence interval endpoint is not representable");
  }
  return Result<Scalar>::success(endpoint);
}

template <class T> Result<T> propagateFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

} // namespace

Result<Interval> varianceConfidenceInterval(Scalar sampleVariance, Index n,
                                            Scalar confidence,
                                            NormalIID premise) {
  if (!std::isfinite(sampleVariance) || !std::isfinite(confidence)) {
    return Result<Interval>::failure(
        ErrorCode::NonFiniteInput,
        "variance confidence interval input is non-finite");
  }
  if (!premise.acknowledged) {
    return Result<Interval>::failure(
        ErrorCode::InvalidDomain,
        "variance confidence interval requires acknowledged NormalIID premise");
  }
  if (n < 2U) {
    return Result<Interval>::failure(
        ErrorCode::InsufficientSamples,
        "variance confidence interval requires at least two observations");
  }
  if (sampleVariance < 0.0 || confidence <= 0.0 || confidence >= 1.0) {
    return Result<Interval>::failure(
        ErrorCode::InvalidDomain,
        "variance must be non-negative and confidence must be in (0,1)");
  }
  if (sampleVariance == 0.0) {
    return Result<Interval>::success(Interval{0.0, 0.0});
  }

  const Scalar alpha = 1.0 - confidence;
  const Scalar lowerProbability = alpha * 0.5;
  const Scalar upperProbability = 1.0 - lowerProbability;
  const Scalar degreesOfFreedom = static_cast<Scalar>(n - 1U);
  const auto law = probability::chiSquare(degreesOfFreedom);
  if (!law.hasValue()) {
    return propagateFailure<Interval>(law.status());
  }
  const ProbabilityPolicy accuracy{.iteration = {.abs = 1.0e-13,
                                                 .rel = 1.0e-13,
                                                 .max_iterations = 256U,
                                                 .max_evaluations = 1024U,
                                                 .bracket = std::nullopt},
                                   .max_draws = 0U,
                                   .abs_log_error = 1.0e-13};
  const auto lowerQuantile =
      probability::quantile(law.value(), lowerProbability, accuracy);
  if (!lowerQuantile.hasValue()) {
    return propagateFailure<Interval>(lowerQuantile.status());
  }
  const auto upperQuantile =
      probability::quantile(law.value(), upperProbability, accuracy);
  if (!upperQuantile.hasValue()) {
    return propagateFailure<Interval>(upperQuantile.status());
  }
  if (!(lowerQuantile.value() > 0.0) ||
      !(upperQuantile.value() > lowerQuantile.value())) {
    return Result<Interval>::failure(
        ErrorCode::NoConvergence,
        "chi-square quantiles do not certify ordered positive endpoints");
  }
  const auto lower =
      varianceEndpoint(sampleVariance, degreesOfFreedom, upperQuantile.value());
  if (!lower.hasValue()) {
    return propagateFailure<Interval>(lower.status());
  }
  const auto upper =
      varianceEndpoint(sampleVariance, degreesOfFreedom, lowerQuantile.value());
  if (!upper.hasValue()) {
    return propagateFailure<Interval>(upper.status());
  }
  if (lower.value() > upper.value()) {
    return Result<Interval>::failure(
        ErrorCode::NoConvergence,
        "variance confidence interval endpoints are not ordered");
  }
  return Result<Interval>::success(Interval{lower.value(), upper.value()});
}

} // namespace vss::math::statistics
