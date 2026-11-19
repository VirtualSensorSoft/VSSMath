#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/NumericalError.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/TensorViews.hpp>
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

Status preflightTensorOutput(const TensorShape& shape, const Limits& limits) {
  const Extent elementsCount = shape.elementCount();
  if (shape.rank() > std::numeric_limits<Extent>::max() / sizeof(Extent) ||
      elementsCount > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "SizeOverflow: tensor output byte count");
  }
  const Extent shapeBytes = shape.rank() * sizeof(Extent);
  const Extent valueBytes = elementsCount * sizeof(Scalar);
  if (shapeBytes > std::numeric_limits<Extent>::max() - valueBytes) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "SizeOverflow: tensor output byte count");
  }
  if (elementsCount > limits.elements) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "tensor output exceeds configured element limit");
  }
  if (shapeBytes + valueBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "tensor output exceeds configured byte limit");
  }
  return Status::success();
}

bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs > std::numeric_limits<Extent>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

void decodeRowMajor(Extent linear, std::span<const Extent> shape,
                    std::span<Extent> coordinates) noexcept {
  for (Extent axis = shape.size(); axis > 0U; --axis) {
    const Extent current = axis - 1U;
    coordinates[current] = linear % shape[current];
    linear /= shape[current];
  }
}

Status validateTensorView(const TensorView& view) {
  if (view.extents.size() != view.strides.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "tensor view extents and strides must match");
  }
  Extent count = 1U;
  bool empty = false;
  for (const Extent extent : view.extents) {
    if (extent == 0U) {
      empty = true;
      count = 0U;
      continue;
    }
    if (!empty && !checkedMultiply(count, extent, count)) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "tensor view element count overflows");
    }
  }
  if (count == 0U) {
    return Status::success();
  }
  Extent maximumOffset = 0U;
  for (Extent axis = 0U; axis < view.extents.size(); ++axis) {
    Extent axisOffset = 0U;
    if (!checkedMultiply(view.extents[axis] - 1U, view.strides[axis],
                         axisOffset) ||
        !checkedAdd(maximumOffset, axisOffset, maximumOffset)) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "tensor view stride offset overflows");
    }
  }
  if (maximumOffset >= view.data.size()) {
    return Status::failure(ErrorCode::InvalidExtent,
                           "tensor view strides exceed supplied storage");
  }
  return Status::success();
}

bool tensorViewFinite(const TensorView& view) {
  Extent count = 1U;
  for (const Extent extent : view.extents) {
    if (extent == 0U) {
      return true;
    }
    count *= extent;
  }
  for (Extent linear = 0U; linear < count; ++linear) {
    Extent remainder = linear;
    Extent offset = 0U;
    for (Extent axis = view.extents.size(); axis > 0U; --axis) {
      const Extent current = axis - 1U;
      const Extent coordinate = remainder % view.extents[current];
      remainder /= view.extents[current];
      offset += coordinate * view.strides[current];
    }
    if (!isFinite(view.data[offset])) {
      return false;
    }
  }
  return true;
}

} // namespace

Result<Tensor> hadamard(const Tensor& lhs, const Tensor& rhs, Limits limits) {
  if (!sameShape(lhs, rhs)) {
    return failure<Tensor>(ErrorCode::DimensionMismatch,
                           "tensor dimensions do not match");
  }
  const Status capacity = preflightTensorOutput(lhs.shape(), limits);
  if (!capacity.ok()) {
    return fromStatus<Tensor>(capacity);
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "Hadamard tensor input is non-finite");
  }
  auto result = Tensor::create(lhs.shape(), 0.0, TensorLimits{limits.elements});
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  const auto lhsValues = elements(lhs);
  const auto rhsValues = elements(rhs);
  auto resultValues = elements(result.value());
  for (Extent i = 0U; i < lhsValues.size(); ++i) {
    const Scalar product = lhsValues[i] * rhsValues[i];
    if (!isFinite(product)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "Hadamard tensor result is non-finite");
    }
    resultValues[i] = product;
  }
  return result;
}

Result<Tensor> divide(const Tensor& value, Scalar divisor, Limits limits) {
  if (!isFinite(divisor)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor divisor is non-finite");
  }
  if (divisor == 0.0) {
    return failure<Tensor>(ErrorCode::InvalidDomain,
                           "tensor divisor must be nonzero");
  }
  if (!finite(value)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor division input is non-finite");
  }
  const Status capacity = preflightTensorOutput(value.shape(), limits);
  if (!capacity.ok()) {
    return fromStatus<Tensor>(capacity);
  }
  auto result =
      Tensor::create(value.shape(), 0.0, TensorLimits{limits.elements});
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  const auto inputValues = elements(value);
  auto resultValues = elements(result.value());
  for (Extent i = 0U; i < inputValues.size(); ++i) {
    const Scalar quotient = inputValues[i] / divisor;
    if (!isFinite(quotient)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "tensor division result is non-finite");
    }
    resultValues[i] = quotient;
  }
  return result;
}

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
      resultValues[(lhsIndex * rhsValues.size()) + rhsIndex] = value;
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

Result<Tensor> reduceTensor(const Tensor& input, std::span<const Index> axes,
                            ReductionKind kind, Limits limits) {
  switch (kind) {
  case ReductionKind::sum:
  case ReductionKind::product:
  case ReductionKind::mean:
  case ReductionKind::minimum:
  case ReductionKind::maximum:
  case ReductionKind::l2:
    break;
  default:
    return failure<Tensor>(ErrorCode::InvalidDomain,
                           "unknown tensor reduction kind");
  }

  const Extent rank = input.shape().rank();
  Extent rankScratchBytes = 0U;
  if (!checkedMultiply(rank, 3U * sizeof(Extent), rankScratchBytes)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "tensor reduction coordinate workspace overflows");
  }
  if (rankScratchBytes > limits.bytes) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "tensor reduction exceeds workspace byte limit");
  }
  std::vector<bool> reduced(rank, false);
  for (const Index axis : axes) {
    if (axis >= rank) {
      return failure<Tensor>(ErrorCode::IndexOutOfRange,
                             "tensor reduction axis is out of range");
    }
    if (reduced[axis]) {
      return failure<Tensor>(ErrorCode::InvalidDomain,
                             "tensor reduction axes must be unique");
    }
    reduced[axis] = true;
  }
  if (!finite(input)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor reduction input is non-finite");
  }
  if (input.size() > limits.evaluations) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "tensor reduction exceeds evaluation limit");
  }

  std::vector<Extent> outputExtents;
  outputExtents.reserve(rank - axes.size());
  Extent reductionCount = 1U;
  for (Index axis = 0U; axis < rank; ++axis) {
    const Extent extent = input.shape().extent(axis);
    if (reduced[axis]) {
      if (!checkedMultiply(reductionCount, extent, reductionCount)) {
        return failure<Tensor>(ErrorCode::SizeOverflow,
                               "tensor reduction extent product overflows");
      }
    } else {
      outputExtents.push_back(extent);
    }
  }
  const auto shape = TensorShape::create(outputExtents);
  if (!shape.hasValue()) {
    return fromStatus<Tensor>(shape.status());
  }
  const Status outputStatus = preflightTensorOutput(shape.value(), limits);
  if (!outputStatus.ok()) {
    return fromStatus<Tensor>(outputStatus);
  }
  const Extent outputCount = shape.value().elementCount();
  Extent shapeBytes = 0U;
  Extent valueBytes = 0U;
  Extent baseOutputBytes = 0U;
  if (!checkedMultiply(shape.value().rank(), sizeof(Extent), shapeBytes) ||
      !checkedMultiply(outputCount, sizeof(Scalar), valueBytes) ||
      !checkedAdd(shapeBytes, valueBytes, baseOutputBytes)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "tensor reduction output byte size overflows");
  }
  if (baseOutputBytes > limits.bytes ||
      rankScratchBytes > limits.bytes - baseOutputBytes) {
    return failure<Tensor>(
        ErrorCode::ResourceLimit,
        "tensor reduction exceeds combined workspace limits");
  }
  if (outputCount == 0U) {
    return Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
  }
  if (reductionCount == 0U &&
      (kind == ReductionKind::mean || kind == ReductionKind::minimum ||
       kind == ReductionKind::maximum)) {
    return failure<Tensor>(ErrorCode::InsufficientSamples,
                           "empty reduction has no mean or extrema");
  }

  const bool needsAccumulation = kind == ReductionKind::sum ||
                                 kind == ReductionKind::product ||
                                 kind == ReductionKind::mean;
  Extent scratchBytes = 0U;
  Extent accumulatorBytes = 0U;
  if (needsAccumulation &&
      (!checkedMultiply(outputCount, sizeof(long double), accumulatorBytes) ||
       !checkedMultiply(accumulatorBytes, 2U, scratchBytes))) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "tensor reduction workspace size overflows");
  }
  if (scratchBytes > limits.bytes - baseOutputBytes - rankScratchBytes) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "tensor reduction exceeds workspace byte limit");
  }
  auto output =
      Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
  if (!output.hasValue()) {
    return output;
  }
  if (axes.empty()) {
    if (input.size() > 0U) {
      std::copy_n(input.data(), input.size(), output.value().data());
    }
    return output;
  }
  if (reductionCount == 0U) {
    if (kind == ReductionKind::product) {
      std::fill_n(output.value().data(), outputCount, Scalar{1});
    }
    return output;
  }

  std::vector<long double> accumulator;
  std::vector<long double> correction;
  if (needsAccumulation) {
    accumulator.assign(outputCount, 0.0L);
    correction.assign(outputCount, 0.0L);
    if (kind == ReductionKind::product) {
      std::ranges::fill(accumulator, 1.0L);
    }
  } else if (kind == ReductionKind::minimum) {
    std::fill_n(output.value().data(), outputCount,
                std::numeric_limits<Scalar>::infinity());
  } else if (kind == ReductionKind::maximum) {
    std::fill_n(output.value().data(), outputCount,
                -std::numeric_limits<Scalar>::infinity());
  }

  std::vector<Extent> inputCoordinates(rank, 0U);
  const auto inputShape = input.shape().extents();
  const std::span<const Scalar> inputValues(input.data(), input.size());
  const std::span<Scalar> outputValues(output.value().data(), outputCount);
  for (Extent linear = 0U; linear < input.size(); ++linear) {
    decodeRowMajor(linear, inputShape, inputCoordinates);
    Extent outputLinear = 0U;
    for (Index axis = 0U, outAxis = 0U; axis < rank; ++axis) {
      if (!reduced[axis]) {
        outputLinear =
            (outputLinear * outputExtents[outAxis]) + inputCoordinates[axis];
        ++outAxis;
      }
    }
    const Scalar value = inputValues[linear];
    switch (kind) {
    case ReductionKind::sum:
    case ReductionKind::mean: {
      const long double term = value;
      const long double adjusted = term - correction[outputLinear];
      const long double next = accumulator[outputLinear] + adjusted;
      correction[outputLinear] = (next - accumulator[outputLinear]) - adjusted;
      accumulator[outputLinear] = next;
      break;
    }
    case ReductionKind::product:
      if (value == 0.0) {
        accumulator[outputLinear] = 0.0L;
      } else if (accumulator[outputLinear] != 0.0L) {
        accumulator[outputLinear] *= static_cast<long double>(value);
      }
      break;
    case ReductionKind::minimum:
      outputValues[outputLinear] = std::min(outputValues[outputLinear], value);
      break;
    case ReductionKind::maximum:
      outputValues[outputLinear] = std::max(outputValues[outputLinear], value);
      break;
    case ReductionKind::l2:
      outputValues[outputLinear] =
          std::hypot(outputValues[outputLinear], value);
      if (!isFinite(outputValues[outputLinear])) {
        return failure<Tensor>(ErrorCode::NonFiniteResult,
                               "tensor l2 reduction result is non-finite");
      }
      break;
    }
  }
  if (needsAccumulation) {
    for (Extent i = 0U; i < outputCount; ++i) {
      long double value = accumulator[i];
      if (kind == ReductionKind::mean) {
        value /= static_cast<long double>(reductionCount);
      }
      outputValues[i] = static_cast<Scalar>(value);
      if (!isFinite(outputValues[i])) {
        return failure<Tensor>(ErrorCode::NonFiniteResult,
                               "tensor reduction result is non-finite");
      }
    }
  }
  return output;
}

Result<Tensor> einsum(std::span<const TensorView> terms,
                      std::span<const IndexList> termLabels,
                      std::span<const Index> outputLabels, Limits limits) {
  if (terms.empty() || terms.size() != termLabels.size()) {
    return failure<Tensor>(ErrorCode::InvalidDomain,
                           "einsum needs matching nonempty terms and labels");
  }
  std::map<Index, Extent> labelExtents;
  Extent totalInputElements = 0U;
  Extent totalAxes = 0U;
  for (Extent term = 0U; term < terms.size(); ++term) {
    const Status viewStatus = validateTensorView(terms[term]);
    if (!viewStatus.ok()) {
      return fromStatus<Tensor>(viewStatus);
    }
    if (termLabels[term].size() != terms[term].extents.size()) {
      return failure<Tensor>(ErrorCode::DimensionMismatch,
                             "einsum label rank must match tensor rank");
    }
    if (!checkedAdd(totalAxes, termLabels[term].size(), totalAxes)) {
      return failure<Tensor>(ErrorCode::SizeOverflow,
                             "einsum axis count overflows");
    }
    Extent elementCount = 1U;
    for (Index axis = 0U; axis < terms[term].extents.size(); ++axis) {
      if (!checkedMultiply(elementCount, terms[term].extents[axis],
                           elementCount)) {
        return failure<Tensor>(ErrorCode::SizeOverflow,
                               "einsum tensor element count overflows");
      }
      const Index label = termLabels[term][axis];
      const auto [position, inserted] =
          labelExtents.emplace(label, terms[term].extents[axis]);
      if (!inserted && position->second != terms[term].extents[axis]) {
        return failure<Tensor>(ErrorCode::DimensionMismatch,
                               "einsum shared label extents differ");
      }
    }
    if (!checkedAdd(totalInputElements, elementCount, totalInputElements)) {
      return failure<Tensor>(ErrorCode::SizeOverflow,
                             "einsum input validation work overflows");
    }
    if (!tensorViewFinite(terms[term])) {
      return failure<Tensor>(ErrorCode::NonFiniteInput,
                             "einsum input contains a non-finite value");
    }
  }

  std::map<Index, Index> labelPositions;
  Extent labelPosition = 0U;
  for (const auto& [label, extent] : labelExtents) {
    static_cast<void>(extent);
    labelPositions.emplace(label, labelPosition++);
  }
  std::vector<bool> outputSeen(labelExtents.size(), false);
  std::vector<Extent> outputExtents;
  outputExtents.reserve(outputLabels.size());
  for (const Index label : outputLabels) {
    const auto position = labelPositions.find(label);
    if (position == labelPositions.end()) {
      return failure<Tensor>(ErrorCode::IndexOutOfRange,
                             "einsum output label is absent from all inputs");
    }
    if (outputSeen[position->second]) {
      return failure<Tensor>(ErrorCode::InvalidDomain,
                             "einsum output labels must be unique");
    }
    outputSeen[position->second] = true;
    outputExtents.push_back(labelExtents.at(label));
  }
  const auto outputShape = TensorShape::create(outputExtents);
  if (!outputShape.hasValue()) {
    return fromStatus<Tensor>(outputShape.status());
  }
  const Status outputStatus =
      preflightTensorOutput(outputShape.value(), limits);
  if (!outputStatus.ok()) {
    return fromStatus<Tensor>(outputStatus);
  }
  const Extent outputCount = outputShape.value().elementCount();
  if (outputCount == 0U) {
    return Tensor::create(outputShape.value(), 0.0,
                          TensorLimits{limits.elements});
  }

  Extent assignments = 1U;
  std::vector<Extent> allExtents(labelExtents.size(), 0U);
  for (const auto& [label, extent] : labelExtents) {
    static_cast<void>(label);
    allExtents[labelPositions.at(label)] = extent;
    if (!checkedMultiply(assignments, extent, assignments)) {
      return failure<Tensor>(ErrorCode::SizeOverflow,
                             "einsum assignment count overflows");
    }
  }
  Extent contractionWork = 0U;
  Extent totalWork = 0U;
  if (!checkedMultiply(assignments, terms.size(), contractionWork) ||
      !checkedAdd(totalInputElements, contractionWork, totalWork)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "einsum work estimate overflows");
  }
  if (totalWork > limits.evaluations) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "einsum exceeds configured assignment work limit");
  }
  Extent accumulatorBytes = 0U;
  Extent scratchBytes = 0U;
  Extent axisBytes = 0U;
  Extent vectorBytes = 0U;
  if (!checkedMultiply(outputCount, sizeof(long double) * 2U,
                       accumulatorBytes) ||
      !checkedMultiply(labelExtents.size(), sizeof(Index) * 4U, scratchBytes) ||
      !checkedMultiply(totalAxes, sizeof(Index), axisBytes) ||
      !checkedMultiply(terms.size(), sizeof(std::vector<Index>), vectorBytes) ||
      !checkedAdd(scratchBytes, axisBytes, scratchBytes) ||
      !checkedAdd(scratchBytes, vectorBytes, scratchBytes) ||
      !checkedAdd(scratchBytes, accumulatorBytes, scratchBytes) ||
      !checkedMultiply(terms.size(), sizeof(TensorView), accumulatorBytes) ||
      !checkedAdd(scratchBytes, accumulatorBytes, scratchBytes)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "einsum workspace byte size overflows");
  }
  Extent outputDataBytes = 0U;
  Extent outputShapeBytes = 0U;
  Extent outputBytes = 0U;
  if (!checkedMultiply(outputCount, sizeof(Scalar), outputDataBytes) ||
      !checkedMultiply(outputShape.value().rank(), sizeof(Extent),
                       outputShapeBytes) ||
      !checkedAdd(outputDataBytes, outputShapeBytes, outputBytes)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "einsum output byte size overflows");
  }
  if (outputBytes > limits.bytes || scratchBytes > limits.bytes - outputBytes) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "einsum exceeds configured workspace byte limit");
  }
  auto output =
      Tensor::create(outputShape.value(), 0.0, TensorLimits{limits.elements});
  if (!output.hasValue()) {
    return output;
  }
  if (assignments == 0U) {
    return output;
  }

  std::vector<std::vector<Index>> termPositions;
  termPositions.reserve(terms.size());
  for (const auto labels : termLabels) {
    std::vector<Index> positions;
    positions.reserve(labels.size());
    for (const Index label : labels) {
      positions.push_back(labelPositions.at(label));
    }
    termPositions.push_back(std::move(positions));
  }
  std::vector<Index> coordinates(labelExtents.size(), 0U);
  std::vector<long double> sums(outputCount, 0.0L);
  std::vector<long double> corrections(outputCount, 0.0L);
  for (Extent assignment = 0U; assignment < assignments; ++assignment) {
    Extent remainder = assignment;
    for (Extent axis = allExtents.size(); axis > 0U; --axis) {
      const Extent current = axis - 1U;
      coordinates[current] = remainder % allExtents[current];
      remainder /= allExtents[current];
    }
    long double product = 1.0L;
    bool hasZero = false;
    for (Extent term = 0U; term < terms.size(); ++term) {
      Extent offset = 0U;
      for (Extent axis = 0U; axis < termPositions[term].size(); ++axis) {
        offset +=
            coordinates[termPositions[term][axis]] * terms[term].strides[axis];
      }
      const Scalar value = terms[term].data[offset];
      hasZero = hasZero || value == 0.0;
      product *= static_cast<long double>(value);
    }
    if (hasZero) {
      product = 0.0L;
    }
    if (!std::isfinite(product)) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "einsum term product is non-finite");
    }
    Extent outputLinear = 0U;
    for (Extent axis = 0U; axis < outputLabels.size(); ++axis) {
      outputLinear = (outputLinear * outputExtents[axis]) +
                     coordinates[labelPositions.at(outputLabels[axis])];
    }
    const long double adjusted = product - corrections[outputLinear];
    const long double next = sums[outputLinear] + adjusted;
    corrections[outputLinear] = (next - sums[outputLinear]) - adjusted;
    sums[outputLinear] = next;
  }
  const std::span<Scalar> outputValues(output.value().data(), outputCount);
  for (Extent i = 0U; i < outputCount; ++i) {
    outputValues[i] = static_cast<Scalar>(sums[i]);
    if (!isFinite(outputValues[i])) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "einsum result is non-finite");
    }
  }
  return output;
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
