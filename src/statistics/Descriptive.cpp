#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/NumericalClassification.hpp>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Descriptive.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

Status validateValues(ConstVectorView values, Scalar& scale) {
  if (values.empty()) {
    return Status::failure(ErrorCode::InvalidExtent,
                           "statistics requires at least one value");
  }

  scale = 0.0;
  for (Index index = 0U; index < values.size(); ++index) {
    const Scalar value = values.at(index);
    if (!std::isfinite(value)) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "statistics input is non-finite at index " +
                                 std::to_string(index));
    }
    scale = std::max(scale, std::abs(value));
  }
  if (!std::isfinite(scale)) {
    return Status::failure(ErrorCode::NonFiniteResult,
                           "statistics input scale is non-finite");
  }
  return Status::success();
}

bool addFinite(detail::CompensatedSum& sum, Scalar value) noexcept {
  if (!std::isfinite(value)) {
    return false;
  }
  sum.add(value);
  return std::isfinite(sum.total());
}

Result<Scalar> scaledMean(ConstVectorView values, Scalar scale) {
  if (scale == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  detail::CompensatedSum normalizedSum;
  for (Index index = 0U; index < values.size(); ++index) {
    const Scalar normalized = values.at(index) / scale;
    if (!addFinite(normalizedSum, normalized)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "normalized mean accumulation became non-finite");
    }
  }
  const auto count = static_cast<Scalar>(values.size());
  const Scalar normalizedMean = normalizedSum.total() / count;
  if (!std::isfinite(normalizedMean)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "normalized mean became non-finite");
  }
  const Scalar result = scale * normalizedMean;
  if (!std::isfinite(result)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "mean became non-finite");
  }
  return Result<Scalar>::success(result);
}

Result<Scalar> normalizedSquareSum(ConstVectorView values, Scalar scale) {
  detail::CompensatedSum sum;
  for (Index index = 0U; index < values.size(); ++index) {
    const Scalar normalized = values.at(index) / scale;
    const Scalar square = normalized * normalized;
    if (!addFinite(sum, square)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "normalized square accumulation became non-finite");
    }
  }
  return Result<Scalar>::success(sum.total());
}

Result<Scalar> varianceImpl(ConstVectorView values, bool sample) {
  Scalar scale = 0.0;
  const Status validation = validateValues(values, scale);
  if (!validation.ok()) {
    return copyFailure<Scalar>(validation);
  }
  if (sample && values.size() < 2U) {
    return Result<Scalar>::failure(
        ErrorCode::InsufficientSamples,
        "sample variance requires at least two values");
  }
  if (scale == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  const Result<Scalar> meanResult = scaledMean(values, scale);
  if (!meanResult.hasValue()) {
    return copyFailure<Scalar>(meanResult.status());
  }
  const Scalar meanValue = meanResult.value();

  Scalar deviationScale = 0.0;
  for (Index index = 0U; index < values.size(); ++index) {
    const Scalar deviation = values.at(index) - meanValue;
    if (!std::isfinite(deviation)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "variance deviation became non-finite at index " +
              std::to_string(index));
    }
    deviationScale = std::max(deviationScale, std::abs(deviation));
  }
  if (!std::isfinite(deviationScale)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "variance deviation scale is non-finite");
  }
  if (deviationScale == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  detail::CompensatedSum sumR;
  detail::CompensatedSum sumR2;
  for (Index index = 0U; index < values.size(); ++index) {
    const Scalar deviation = values.at(index) - meanValue;
    const Scalar normalized = deviation / deviationScale;
    const Scalar square = normalized * normalized;
    if (!addFinite(sumR, normalized) || !addFinite(sumR2, square)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "normalized variance accumulation became non-finite");
    }
  }

  const auto count = static_cast<Scalar>(values.size());
  const Scalar sum = sumR.total();
  const Scalar sumSquares = sumR2.total();
  const Scalar correction = (sum * sum) / count;
  const Scalar corrected = sumSquares - correction;
  if (!std::isfinite(sum) || !std::isfinite(sumSquares) ||
      !std::isfinite(correction) || !std::isfinite(corrected)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "normalized variance correction became non-finite");
  }

  const auto divisor =
      static_cast<Scalar>(sample ? values.size() - 1U : values.size());
  const Scalar scaledDivisor = deviationScale / divisor;
  const Scalar candidate = deviationScale * (scaledDivisor * corrected);
  if (!std::isfinite(scaledDivisor) || !std::isfinite(candidate)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "variance result became non-finite");
  }
  if (candidate >= 0.0) {
    return Result<Scalar>::success(candidate);
  }

  const Scalar qScale = std::max(scale, std::abs(meanValue));
  if (!std::isfinite(qScale) || qScale == 0.0) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "variance reference scale is invalid");
  }
  const Result<Scalar> squareSumResult = normalizedSquareSum(values, scale);
  if (!squareSumResult.hasValue()) {
    return copyFailure<Scalar>(squareSumResult.status());
  }
  const Scalar scaleRatio = scale / qScale;
  const Scalar meanRatio = meanValue / qScale;
  const Scalar reference =
      std::max((scaleRatio * scaleRatio) * (squareSumResult.value() / count),
               meanRatio * meanRatio);
  const Scalar candidateNorm = (std::abs(candidate) / qScale) / qScale;
  const ErrorCode classification =
      detail::classifyVarianceRoundoff(candidateNorm, reference);
  if (classification == ErrorCode::None) {
    return Result<Scalar>::success(0.0);
  }
  return Result<Scalar>::failure(
      ErrorCode::NonFiniteResult,
      "variance is negative beyond the permitted roundoff boundary");
}

Result<Scalar> standardDeviation(ConstVectorView values, bool sample) {
  const Result<Scalar> variance = varianceImpl(values, sample);
  if (!variance.hasValue()) {
    return copyFailure<Scalar>(variance.status());
  }
  const Scalar result = std::sqrt(variance.value());
  if (!std::isfinite(result)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "standard deviation became non-finite");
  }
  return Result<Scalar>::success(result);
}

} // namespace

Result<Scalar> mean(ConstVectorView values) {
  Scalar scale = 0.0;
  const Status validation = validateValues(values, scale);
  if (!validation.ok()) {
    return copyFailure<Scalar>(validation);
  }
  return scaledMean(values, scale);
}

Result<Scalar> populationVariance(ConstVectorView values) {
  return varianceImpl(values, false);
}

Result<Scalar> sampleVariance(ConstVectorView values) {
  return varianceImpl(values, true);
}

Result<Scalar> populationStandardDeviation(ConstVectorView values) {
  return standardDeviation(values, false);
}

Result<Scalar> sampleStandardDeviation(ConstVectorView values) {
  return standardDeviation(values, true);
}

Result<Scalar> rootMeanSquare(ConstVectorView values) {
  Scalar scale = 0.0;
  const Status validation = validateValues(values, scale);
  if (!validation.ok()) {
    return copyFailure<Scalar>(validation);
  }
  if (scale == 0.0) {
    return Result<Scalar>::success(0.0);
  }

  const Result<Scalar> sumResult = normalizedSquareSum(values, scale);
  if (!sumResult.hasValue()) {
    return copyFailure<Scalar>(sumResult.status());
  }
  const auto count = static_cast<Scalar>(values.size());
  const Scalar normalizedMeanSquare = sumResult.value() / count;
  const Scalar root = std::sqrt(normalizedMeanSquare);
  const Scalar result = scale * root;
  if (!std::isfinite(normalizedMeanSquare) || !std::isfinite(root) ||
      !std::isfinite(result)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "root mean square became non-finite");
  }
  return Result<Scalar>::success(result);
}

} // namespace vss::math::statistics
