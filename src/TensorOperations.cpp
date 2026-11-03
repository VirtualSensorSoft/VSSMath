#include <algorithm>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/NumericalError.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {
namespace {

template <class T> Result<T> fromStatus(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

Status validateOptions(OperationOptions options) {
  return validate(options.tolerance);
}

bool sameShape(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape().rank() != rhs.shape().rank()) {
    return false;
  }
  for (Index axis = 0U; axis < lhs.shape().rank(); ++axis) {
    if (lhs.shape().extent(axis) != rhs.shape().extent(axis)) {
      return false;
    }
  }
  return true;
}

std::span<const Scalar> elements(const Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

std::span<Scalar> elements(Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

bool finite(const Tensor& tensor) {
  return std::ranges::all_of(elements(tensor),
                             [](Scalar value) { return isFinite(value); });
}

template <class Operation>
Result<Tensor> binaryOperation(const Tensor& lhs, const Tensor& rhs,
                               TensorLimits limits, OperationOptions options,
                               Operation operation, const char* label) {
  const Status optionStatus = validateOptions(options);
  if (!optionStatus.ok()) {
    return fromStatus<Tensor>(optionStatus);
  }
  if (!sameShape(lhs, rhs)) {
    return failure<Tensor>(ErrorCode::DimensionMismatch,
                           "tensor dimensions do not match");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor arithmetic input is non-finite");
  }
  auto result = Tensor::create(lhs.shape(), 0.0, limits);
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  const auto lhsValues = elements(lhs);
  const auto rhsValues = elements(rhs);
  auto resultValues = elements(result.value());
  for (Extent index = 0U; index < lhsValues.size(); ++index) {
    const Scalar value = operation(lhsValues[index], rhsValues[index]);
    if (options.requireFinite && !isFinite(value)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult, label);
    }
    resultValues[index] = value;
  }
  return result;
}

template <class T> T unwrap(Result<T> result) {
  if (result.hasValue()) {
    return std::move(result).value();
  }
  throw NumericalError(result.status().code(), result.status().message());
}

} // namespace

Result<Tensor> add(const Tensor& lhs, const Tensor& rhs, TensorLimits limits,
                   OperationOptions options) {
  return binaryOperation(
      lhs, rhs, limits, options,
      [](Scalar left, Scalar right) { return left + right; },
      "tensor addition produced a non-finite result");
}

Result<Tensor> subtract(const Tensor& lhs, const Tensor& rhs,
                        TensorLimits limits, OperationOptions options) {
  return binaryOperation(
      lhs, rhs, limits, options,
      [](Scalar left, Scalar right) { return left - right; },
      "tensor subtraction produced a non-finite result");
}

Result<Tensor> scale(const Tensor& value, Scalar factor, TensorLimits limits,
                     OperationOptions options) {
  const Status optionStatus = validateOptions(options);
  if (!optionStatus.ok()) {
    return fromStatus<Tensor>(optionStatus);
  }
  if (options.requireFinite && (!finite(value) || !isFinite(factor))) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor scale input is non-finite");
  }
  auto result = Tensor::create(value.shape(), 0.0, limits);
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  const auto inputValues = elements(value);
  auto resultValues = elements(result.value());
  for (Extent index = 0U; index < inputValues.size(); ++index) {
    const Scalar scaled = inputValues[index] * factor;
    if (options.requireFinite && !isFinite(scaled)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "tensor scale produced a non-finite result");
    }
    resultValues[index] = scaled;
  }
  return result;
}

Result<Tensor> outer(const Tensor& lhs, const Tensor& rhs, TensorLimits limits,
                     OperationOptions options) {
  const Status optionStatus = validateOptions(options);
  if (!optionStatus.ok()) {
    return fromStatus<Tensor>(optionStatus);
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor outer product input is non-finite");
  }

  std::vector<Extent> resultExtents;
  resultExtents.reserve(lhs.shape().rank() + rhs.shape().rank());
  resultExtents.insert(resultExtents.end(), lhs.shape().extents().begin(),
                       lhs.shape().extents().end());
  resultExtents.insert(resultExtents.end(), rhs.shape().extents().begin(),
                       rhs.shape().extents().end());
  const auto shape = TensorShape::create(resultExtents);
  if (!shape.hasValue()) {
    return fromStatus<Tensor>(shape.status());
  }
  auto result = Tensor::create(shape.value(), 0.0, limits);
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }

  const auto lhsValues = elements(lhs);
  const auto rhsValues = elements(rhs);
  auto resultValues = elements(result.value());
  for (Extent lhsIndex = 0U; lhsIndex < lhsValues.size(); ++lhsIndex) {
    for (Extent rhsIndex = 0U; rhsIndex < rhsValues.size(); ++rhsIndex) {
      const Scalar value = lhsValues[lhsIndex] * rhsValues[rhsIndex];
      if (options.requireFinite && !isFinite(value)) {
        return failure<Tensor>(ErrorCode::NonFiniteResult,
                               "tensor outer product is non-finite");
      }
      resultValues[lhsIndex * rhsValues.size() + rhsIndex] = value;
    }
  }
  return result;
}

Result<Tensor> contract(const Tensor& lhs, const Tensor& rhs,
                        std::span<const Index> lhsAxes,
                        std::span<const Index> rhsAxes, TensorLimits limits,
                        OperationOptions options) {
  const Status optionStatus = validateOptions(options);
  if (!optionStatus.ok()) {
    return fromStatus<Tensor>(optionStatus);
  }
  if (lhsAxes.size() != rhsAxes.size()) {
    return failure<Tensor>(ErrorCode::DimensionMismatch,
                           "contracted axis lists have different lengths");
  }

  std::vector<unsigned char> lhsContracted(lhs.shape().rank(), 0U);
  std::vector<unsigned char> rhsContracted(rhs.shape().rank(), 0U);
  std::vector<Extent> contractionExtents;
  contractionExtents.reserve(lhsAxes.size());
  for (Index pair = 0U; pair < lhsAxes.size(); ++pair) {
    const Index lhsAxis = lhsAxes[pair];
    const Index rhsAxis = rhsAxes[pair];
    if (lhsAxis >= lhs.shape().rank() || rhsAxis >= rhs.shape().rank()) {
      return failure<Tensor>(ErrorCode::IndexOutOfRange,
                             "contracted tensor axis is out of range");
    }
    if (lhsContracted[lhsAxis] != 0U || rhsContracted[rhsAxis] != 0U) {
      return failure<Tensor>(ErrorCode::InvalidExtent,
                             "contracted tensor axes must be unique");
    }
    const Extent lhsExtent = lhs.shape().extent(lhsAxis);
    const Extent rhsExtent = rhs.shape().extent(rhsAxis);
    if (lhsExtent != rhsExtent) {
      return failure<Tensor>(ErrorCode::DimensionMismatch,
                             "paired contracted tensor extents differ");
    }
    lhsContracted[lhsAxis] = 1U;
    rhsContracted[rhsAxis] = 1U;
    contractionExtents.push_back(lhsExtent);
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor contraction input is non-finite");
  }

  std::vector<Extent> resultExtents;
  resultExtents.reserve(lhs.shape().rank() + rhs.shape().rank() -
                        lhsAxes.size() - rhsAxes.size());
  for (Index axis = 0U; axis < lhs.shape().rank(); ++axis) {
    if (lhsContracted[axis] == 0U) {
      resultExtents.push_back(lhs.shape().extent(axis));
    }
  }
  for (Index axis = 0U; axis < rhs.shape().rank(); ++axis) {
    if (rhsContracted[axis] == 0U) {
      resultExtents.push_back(rhs.shape().extent(axis));
    }
  }
  const auto resultShape = TensorShape::create(resultExtents);
  if (!resultShape.hasValue()) {
    return fromStatus<Tensor>(resultShape.status());
  }
  auto result = Tensor::create(resultShape.value(), 0.0, limits);
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  if (result.value().size() == 0U) {
    return result;
  }

  Extent contractionCount = 1U;
  for (const Extent extent : contractionExtents) {
    if (extent != 0U &&
        contractionCount > std::numeric_limits<Extent>::max() / extent) {
      return failure<Tensor>(ErrorCode::SizeOverflow,
                             "tensor contraction size overflows Extent");
    }
    contractionCount *= extent;
  }

  std::vector<Index> resultCoordinates(resultShape.value().rank(), 0U);
  std::vector<Index> lhsCoordinates(lhs.shape().rank(), 0U);
  std::vector<Index> rhsCoordinates(rhs.shape().rank(), 0U);
  std::vector<Index> contractionCoordinates(lhsAxes.size(), 0U);
  for (Extent resultLinear = 0U; resultLinear < result.value().size();
       ++resultLinear) {
    Extent remainder = resultLinear;
    for (Index axis = resultCoordinates.size(); axis > 0U; --axis) {
      const Extent extent = resultExtents[axis - 1U];
      resultCoordinates[axis - 1U] = remainder % extent;
      remainder /= extent;
    }
    Index resultAxis = 0U;
    for (Index axis = 0U; axis < lhs.shape().rank(); ++axis) {
      if (lhsContracted[axis] == 0U) {
        lhsCoordinates[axis] = resultCoordinates[resultAxis++];
      }
    }
    for (Index axis = 0U; axis < rhs.shape().rank(); ++axis) {
      if (rhsContracted[axis] == 0U) {
        rhsCoordinates[axis] = resultCoordinates[resultAxis++];
      }
    }

    Scalar sum = 0.0;
    for (Extent contractionLinear = 0U; contractionLinear < contractionCount;
         ++contractionLinear) {
      remainder = contractionLinear;
      for (Index axis = contractionCoordinates.size(); axis > 0U; --axis) {
        const Extent extent = contractionExtents[axis - 1U];
        contractionCoordinates[axis - 1U] = remainder % extent;
        remainder /= extent;
      }
      for (Index pair = 0U; pair < lhsAxes.size(); ++pair) {
        lhsCoordinates[lhsAxes[pair]] = contractionCoordinates[pair];
        rhsCoordinates[rhsAxes[pair]] = contractionCoordinates[pair];
      }
      const Scalar term = lhs.at(lhsCoordinates) * rhs.at(rhsCoordinates);
      if (options.requireFinite && !isFinite(term)) {
        return failure<Tensor>(ErrorCode::NonFiniteResult,
                               "tensor contraction product is non-finite");
      }
      sum += term;
      if (options.requireFinite && !isFinite(sum)) {
        return failure<Tensor>(ErrorCode::NonFiniteResult,
                               "tensor contraction sum is non-finite");
      }
    }
    result.value().at(resultCoordinates) = sum;
  }
  return result;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  return unwrap(add(lhs, rhs));
}
Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  return unwrap(subtract(lhs, rhs));
}
Tensor operator*(const Tensor& value, Scalar factor) {
  return unwrap(scale(value, factor));
}
Tensor operator*(Scalar factor, const Tensor& value) { return value * factor; }

} // namespace vss::math
