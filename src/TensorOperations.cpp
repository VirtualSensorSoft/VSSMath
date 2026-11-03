#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/NumericalError.hpp>
#include <vss/math/Operations.hpp>
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

bool finite(const Tensor& tensor) {
  for (Extent index = 0U; index < tensor.size(); ++index) {
    if (!isFinite(tensor.data()[index])) {
      return false;
    }
  }
  return true;
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
  for (Extent index = 0U; index < lhs.size(); ++index) {
    const Scalar value = operation(lhs.data()[index], rhs.data()[index]);
    if (options.requireFinite && !isFinite(value)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult, label);
    }
    result.value().data()[index] = value;
  }
  return result;
}

template <class T> T unwrap(Result<T> result) {
  if (result.hasValue()) {
    return std::move(result).value();
  }
  throw NumericalError(result.status().code(),
                       std::string(result.status().message()));
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
  for (Extent index = 0U; index < value.size(); ++index) {
    const Scalar scaled = value.data()[index] * factor;
    if (options.requireFinite && !isFinite(scaled)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "tensor scale produced a non-finite result");
    }
    result.value().data()[index] = scaled;
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
