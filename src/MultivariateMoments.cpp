#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/ThirdMoment.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {
namespace {
Status failure(ErrorCode code, const char* message) {
  return Status::failure(code, message);
}

Result<Scalar> resultFailure(ErrorCode code, std::string message) {
  return Result<Scalar>::failure(code, std::move(message));
}

Scalar tensorValue(const Tensor& tensor, Index component, Extent rank) {
  std::array<Index, 4U> indices{};
  indices.fill(component);
  return tensor.at(std::span<const Index>{indices.data(), rank});
}
} // namespace

MultivariateMoments::MultivariateMoments(Extent dimension) noexcept
    : dimension_(dimension) {}

Result<MultivariateMoments> MultivariateMoments::create(Extent dimension) {
  if (dimension == 0U) {
    return Result<MultivariateMoments>::failure(
        ErrorCode::InvalidExtent,
        "moment component dimension must be positive");
  }
  return Result<MultivariateMoments>::success(MultivariateMoments{dimension});
}

Extent MultivariateMoments::dimension() const noexcept { return dimension_; }

Status MultivariateMoments::validateDimension(const Tensor& tensor) const {
  if (tensor.shape().extent(0U) != dimension_) {
    return failure(ErrorCode::DimensionMismatch,
                   "moment component dimension does not match aggregate");
  }
  return Status::success();
}

Status MultivariateMoments::setFirstMoment(const FirstMoment& moment) {
  Status status = validateDimension(moment.tensor());
  if (!status.ok()) {
    return status;
  }
  std::optional<FirstMoment> replacement{moment};
  first_.swap(replacement);
  return Status::success();
}

Status MultivariateMoments::setSecondMoment(const SecondMoment& moment) {
  Status status = validateDimension(moment.tensor());
  if (!status.ok()) {
    return status;
  }
  std::optional<SecondMoment> replacement{moment};
  second_.swap(replacement);
  return Status::success();
}

Status MultivariateMoments::setThirdMoment(const ThirdMoment& moment) {
  Status status = validateDimension(moment.tensor());
  if (!status.ok()) {
    return status;
  }
  std::optional<ThirdMoment> replacement{moment};
  third_.swap(replacement);
  return Status::success();
}

Status MultivariateMoments::setFourthMoment(const FourthMoment& moment) {
  Status status = validateDimension(moment.tensor());
  if (!status.ok()) {
    return status;
  }
  std::optional<FourthMoment> replacement{moment};
  fourth_.swap(replacement);
  return Status::success();
}

const FirstMoment* MultivariateMoments::getFirstMoment() const noexcept {
  return first_ ? &*first_ : nullptr;
}
const SecondMoment* MultivariateMoments::getSecondMoment() const noexcept {
  return second_ ? &*second_ : nullptr;
}
const ThirdMoment* MultivariateMoments::getThirdMoment() const noexcept {
  return third_ ? &*third_ : nullptr;
}
const FourthMoment* MultivariateMoments::getFourthMoment() const noexcept {
  return fourth_ ? &*fourth_ : nullptr;
}

void MultivariateMoments::clearFirstMoment() noexcept { first_.reset(); }
void MultivariateMoments::clearSecondMoment() noexcept { second_.reset(); }
void MultivariateMoments::clearThirdMoment() noexcept { third_.reset(); }
void MultivariateMoments::clearFourthMoment() noexcept { fourth_.reset(); }

Result<Scalar> MultivariateMoments::componentVariance(Index component) const {
  if (component >= dimension_) {
    return resultFailure(ErrorCode::IndexOutOfRange,
                         "moment component index is out of range");
  }
  if (!second_) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "second central moment is unavailable");
  }
  const Scalar variance = tensorValue(second_->tensor(), component, 2U);
  if (variance < 0.0) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "component variance must be non-negative");
  }
  return Result<Scalar>::success(variance);
}

Result<Scalar>
MultivariateMoments::componentStandardDeviation(Index component) const {
  const auto variance = componentVariance(component);
  if (!variance) {
    return resultFailure(variance.status().code(),
                         std::string{variance.status().message()});
  }
  const Scalar deviation = std::sqrt(variance.value());
  if (!isFinite(deviation)) {
    return resultFailure(ErrorCode::NonFiniteResult,
                         "component standard deviation is not finite");
  }
  return Result<Scalar>::success(deviation);
}

Status MultivariateMoments::validateStandardizedInputs(Index component) const {
  if (component >= dimension_) {
    return failure(ErrorCode::IndexOutOfRange,
                   "moment component index is out of range");
  }
  if (!first_) {
    return failure(ErrorCode::InvalidDomain,
                   "first moment is required for standardized quantities");
  }
  const auto variance = componentVariance(component);
  if (!variance) {
    return Status::failure(variance.status().code(),
                           std::string{variance.status().message()});
  }
  if (variance.value() == 0.0) {
    return failure(ErrorCode::InvalidDomain,
                   "standardized quantity is undefined for zero variance");
  }
  return Status::success();
}

Result<Scalar> MultivariateMoments::componentSkewness(Index component) const {
  const Status input_status = validateStandardizedInputs(component);
  if (!input_status.ok()) {
    return Result<Scalar>::failure(input_status.code(),
                                   std::string{input_status.message()});
  }
  if (!second_) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "second central moment is unavailable");
  }
  if (!third_) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "third central moment is unavailable");
  }
  const Scalar variance = tensorValue(second_->tensor(), component, 2U);
  const Scalar third = tensorValue(third_->tensor(), component, 3U);
  const Scalar skewness = (third / variance) / std::sqrt(variance);
  if (!isFinite(skewness)) {
    return resultFailure(ErrorCode::NonFiniteResult,
                         "component skewness is not finite");
  }
  return Result<Scalar>::success(skewness);
}

Result<Scalar> MultivariateMoments::componentKurtosis(Index component) const {
  const Status input_status = validateStandardizedInputs(component);
  if (!input_status.ok()) {
    return Result<Scalar>::failure(input_status.code(),
                                   std::string{input_status.message()});
  }
  if (!second_) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "second central moment is unavailable");
  }
  if (!fourth_) {
    return resultFailure(ErrorCode::InvalidDomain,
                         "fourth central moment is unavailable");
  }
  const Scalar variance = tensorValue(second_->tensor(), component, 2U);
  const Scalar fourth = tensorValue(fourth_->tensor(), component, 4U);
  const Scalar kurtosis = (fourth / variance) / variance;
  if (!isFinite(kurtosis)) {
    return resultFailure(ErrorCode::NonFiniteResult,
                         "component kurtosis is not finite");
  }
  return Result<Scalar>::success(kurtosis);
}

Result<Scalar>
MultivariateMoments::componentExcessKurtosis(Index component) const {
  auto kurtosis = componentKurtosis(component);
  if (!kurtosis) {
    return resultFailure(kurtosis.status().code(),
                         std::string{kurtosis.status().message()});
  }
  const Scalar excess = kurtosis.value() - 3.0;
  if (!isFinite(excess)) {
    return resultFailure(ErrorCode::NonFiniteResult,
                         "component excess kurtosis is not finite");
  }
  return Result<Scalar>::success(excess);
}

} // namespace vss::math
