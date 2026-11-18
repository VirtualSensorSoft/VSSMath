#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> fromStatus(const Status& status) {
  return Result<T>::failure(status.code(), std::string{status.message()});
}

bool finite(const Tensor& value) {
  for (Index i = 0U; i < value.size(); ++i) {
    if (!isFinite(value.data()[i])) {
      return false;
    }
  }
  return true;
}

Status preflight(std::span<const Extent> extents, const Limits& limits,
                 Extent& elementCount, Extent scratchBytes = 0U) {
  elementCount = 1U;
  for (const Extent extent : extents) {
    if (extent != 0U &&
        elementCount > std::numeric_limits<Extent>::max() / extent) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "tensor output element count overflows");
    }
    elementCount *= extent;
  }
  if (extents.size() > std::numeric_limits<Extent>::max() / sizeof(Extent) ||
      elementCount > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "tensor output byte count overflows");
  }
  const Extent shapeBytes = extents.size() * sizeof(Extent);
  const Extent valueBytes = elementCount * sizeof(Scalar);
  if (shapeBytes > std::numeric_limits<Extent>::max() - valueBytes ||
      scratchBytes >
          std::numeric_limits<Extent>::max() - shapeBytes - valueBytes) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "tensor output byte count overflows");
  }
  if (elementCount > limits.elements) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "tensor output exceeds element limit");
  }
  if (shapeBytes + valueBytes + scratchBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "tensor output exceeds byte limit");
  }
  return Status::success();
}

Result<Tensor> allocateTensor(std::span<const Extent> extents,
                              const Limits& limits) {
  const auto shape = TensorShape::create(extents);
  if (!shape.hasValue()) {
    return fromStatus<Tensor>(shape.status());
  }
  auto result =
      Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
  if (!result.hasValue()) {
    return fromStatus<Tensor>(result.status());
  }
  return result;
}

} // namespace

Result<Tensor> transformElements(const Tensor& value,
                                 const UnaryFunction& function, Limits limits) {
  Extent elementCount = 0U;
  const Status capacity =
      preflight(value.shape().extents(), limits, elementCount);
  if (!capacity.ok()) {
    return fromStatus<Tensor>(capacity);
  }
  if (elementCount > limits.evaluations) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "tensor transform exceeds callback limit");
  }
  if (!finite(value)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "tensor transform input is non-finite");
  }
  auto result = allocateTensor(value.shape().extents(), limits);
  if (!result.hasValue()) {
    return result;
  }
  for (Extent i = 0U; i < elementCount; ++i) {
    const auto invoke = [&function, &value, i]() -> Result<Scalar> {
      try {
        return function(value.data()[i]);
      } catch (...) {
        return Result<Scalar>::failure(
            ErrorCode::CallbackFailure,
            "tensor transform callback threw an exception");
      }
    };
    Result<Scalar> transformed = invoke();
    if (!transformed.hasValue()) {
      return Result<Tensor>::failure(
          transformed.status().code(),
          std::string{transformed.status().message()});
    }
    if (!isFinite(transformed.value())) {
      return failure<Tensor>(ErrorCode::NonFiniteResult,
                             "tensor transform produced a non-finite value");
    }
    result.value().data()[i] = transformed.value();
  }
  return result;
}

Result<Tensor> broadcastTo(const Tensor& value,
                           std::span<const Extent> destinationShape,
                           std::span<const Index> destinationAxisForSource,
                           Limits limits) {
  if (destinationAxisForSource.size() != value.shape().rank()) {
    return failure<Tensor>(ErrorCode::DimensionMismatch,
                           "broadcast axis map must cover each source axis");
  }
  for (Index sourceAxis = 0U; sourceAxis < value.shape().rank(); ++sourceAxis) {
    const Index destinationAxis = destinationAxisForSource[sourceAxis];
    if (destinationAxis >= destinationShape.size()) {
      return failure<Tensor>(ErrorCode::IndexOutOfRange,
                             "broadcast destination axis is out of range");
    }
    const Extent sourceExtent = value.shape().extent(sourceAxis);
    const Extent targetExtent = destinationShape[destinationAxis];
    if (sourceExtent != targetExtent && sourceExtent != 1U) {
      return failure<Tensor>(ErrorCode::DimensionMismatch,
                             "broadcast source and destination extents differ");
    }
  }
  Extent elementCount = 0U;
  if (destinationShape.size() > std::numeric_limits<Extent>::max() /
                                    (sizeof(Extent) + sizeof(unsigned char))) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "broadcast axis workspace size overflows");
  }
  const Extent mapScratchBytes =
      destinationShape.size() * (sizeof(Extent) + sizeof(unsigned char));
  const Status capacity =
      preflight(destinationShape, limits, elementCount, mapScratchBytes);
  if (!capacity.ok()) {
    return fromStatus<Tensor>(capacity);
  }
  std::vector<Extent> destinationStrides(destinationShape.size(), 1U);
  std::vector<unsigned char> mapped(destinationShape.size(), 0U);
  for (const Index destinationAxis : destinationAxisForSource) {
    if (mapped[destinationAxis] != 0U) {
      return failure<Tensor>(ErrorCode::InvalidDomain,
                             "broadcast destination axes must be unique");
    }
    mapped[destinationAxis] = 1U;
  }
  if (elementCount != 0U) {
    Extent stride = 1U;
    for (Index axis = destinationShape.size(); axis > 0U; --axis) {
      destinationStrides[axis - 1U] = stride;
      stride *= destinationShape[axis - 1U];
    }
  }
  if (!finite(value)) {
    return failure<Tensor>(ErrorCode::NonFiniteInput,
                           "broadcast tensor input is non-finite");
  }
  auto result = allocateTensor(destinationShape, limits);
  if (!result.hasValue()) {
    return result;
  }
  for (Extent linear = 0U; linear < elementCount; ++linear) {
    Extent sourceLinear = 0U;
    for (Index sourceAxis = 0U; sourceAxis < value.shape().rank();
         ++sourceAxis) {
      const Index destinationAxis = destinationAxisForSource[sourceAxis];
      const Extent targetExtent = destinationShape[destinationAxis];
      Extent sourceCoordinate = 0U;
      if (value.shape().extent(sourceAxis) != 1U) {
        sourceCoordinate =
            (linear / destinationStrides[destinationAxis]) % targetExtent;
      }
      sourceLinear =
          sourceLinear * value.shape().extent(sourceAxis) + sourceCoordinate;
    }
    result.value().data()[linear] = value.data()[sourceLinear];
  }
  return result;
}

} // namespace vss::math
