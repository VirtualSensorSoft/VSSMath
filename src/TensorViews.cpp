#include <algorithm>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/TensorViews.hpp>

namespace vss::math {
namespace {

Result<Tensor> failure(ErrorCode code, const char* message) {
  return Result<Tensor>::failure(code, message);
}

bool multiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool add(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs > std::numeric_limits<Extent>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

Result<Tensor> createOutput(std::span<const Extent> extent,
                            const Limits& limits,
                            Extent extraScratchBytes = 0U) {
  const auto shape = TensorShape::create(extent);
  if (!shape.hasValue()) {
    return Result<Tensor>::failure(shape.status().code(),
                                   std::string(shape.status().message()));
  }
  const Extent elements = shape.value().elementCount();
  Extent shapeBytes = 0U;
  Extent valueBytes = 0U;
  Extent totalBytes = 0U;
  if (!multiply(extent.size(), sizeof(Extent), shapeBytes) ||
      !multiply(shapeBytes, 2U, shapeBytes) ||
      !multiply(elements, sizeof(Scalar), valueBytes) ||
      !add(shapeBytes, valueBytes, totalBytes) ||
      !add(totalBytes, extraScratchBytes, totalBytes)) {
    return failure(ErrorCode::SizeOverflow, "tensor output byte size overflow");
  }
  if (elements > limits.elements || totalBytes > limits.bytes) {
    return failure(ErrorCode::ResourceLimit,
                   "tensor output exceeds configured resource limits");
  }
  return Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
}

Extent rowMajorOffset(std::span<const Extent> index,
                      std::span<const Extent> shape) noexcept {
  Extent offset = 0U;
  for (Extent axis = 0U; axis < shape.size(); ++axis) {
    offset = (offset * shape[axis]) + index[axis];
  }
  return offset;
}

void decodeRowMajor(Extent linear, std::span<const Extent> shape,
                    std::span<Extent> index) noexcept {
  for (Extent axis = shape.size(); axis > 0U; --axis) {
    const Extent current = axis - 1U;
    index[current] = linear % shape[current];
    linear /= shape[current];
  }
}

} // namespace

Result<Tensor> slice(const Tensor& input, std::span<const Extent> begin,
                     std::span<const Extent> extent,
                     std::span<const Extent> step, Limits limits) {
  const Extent rank = input.shape().rank();
  if (begin.size() != rank || extent.size() != rank || step.size() != rank) {
    return failure(ErrorCode::DimensionMismatch,
                   "slice index, extent, and step ranks must match input");
  }
  const auto inputShape = input.shape().extents();
  for (Extent axis = 0U; axis < rank; ++axis) {
    if (step[axis] == 0U) {
      return failure(ErrorCode::InvalidDomain, "slice steps must be positive");
    }
    if (extent[axis] == 0U) {
      if (begin[axis] > inputShape[axis]) {
        return failure(ErrorCode::IndexOutOfRange,
                       "empty slice begins beyond input extent");
      }
      continue;
    }
    Extent distance = 0U;
    Extent last = 0U;
    if (!multiply(extent[axis] - 1U, step[axis], distance) ||
        !add(begin[axis], distance, last)) {
      return failure(ErrorCode::SizeOverflow,
                     "slice bound arithmetic overflow");
    }
    if (last >= inputShape[axis]) {
      return failure(ErrorCode::IndexOutOfRange, "slice exceeds input extent");
    }
  }

  Extent scratchBytes = 0U;
  if (!multiply(rank, 2U * sizeof(Extent), scratchBytes)) {
    return failure(ErrorCode::SizeOverflow,
                   "slice coordinate workspace size overflows");
  }
  auto output = createOutput(extent, limits, scratchBytes);
  if (!output.hasValue()) {
    return output;
  }
  std::vector<Extent> outputIndex(rank, 0U);
  std::vector<Extent> inputIndex(rank, 0U);
  const std::span<Scalar> outputData(output.value().data(),
                                     output.value().size());
  const std::span<const Scalar> inputData(input.data(), input.size());
  const Extent outputSize = output.value().size();
  for (Extent linear = 0U; linear < outputSize; ++linear) {
    decodeRowMajor(linear, extent, outputIndex);
    for (Extent axis = 0U; axis < rank; ++axis) {
      inputIndex[axis] = begin[axis] + (outputIndex[axis] * step[axis]);
    }
    outputData[linear] = inputData[rowMajorOffset(inputIndex, inputShape)];
  }
  return output;
}

Result<Tensor> reshape(const Tensor& input, std::span<const Extent> extent,
                       Limits limits) {
  const auto requestedShape = TensorShape::create(extent);
  if (!requestedShape.hasValue()) {
    return Result<Tensor>::failure(
        requestedShape.status().code(),
        std::string(requestedShape.status().message()));
  }
  if (requestedShape.value().elementCount() != input.size()) {
    return failure(ErrorCode::DimensionMismatch,
                   "reshape must preserve the element count");
  }
  auto output = createOutput(extent, limits);
  if (!output.hasValue()) {
    return output;
  }
  if (input.size() > 0U) {
    std::copy_n(std::span<const Scalar>(input.data(), input.size()).begin(),
                input.size(),
                std::span<Scalar>(output.value().data(), output.value().size())
                    .begin());
  }
  return output;
}

Result<Tensor> permuteAxes(const Tensor& input, std::span<const Index> axes,
                           Limits limits) {
  const Extent rank = input.shape().rank();
  if (axes.size() != rank) {
    return failure(ErrorCode::DimensionMismatch,
                   "axis permutation rank must match input rank");
  }
  std::vector<bool> seen(rank, false);
  const auto inputShape = input.shape().extents();
  std::vector<Extent> outputShape(rank, 0U);
  for (Extent axis = 0U; axis < rank; ++axis) {
    if (axes[axis] >= rank) {
      return failure(ErrorCode::IndexOutOfRange,
                     "permutation input axis is out of range");
    }
    if (seen[axes[axis]]) {
      return failure(ErrorCode::InvalidDomain,
                     "axis permutation must be bijective");
    }
    seen[axes[axis]] = true;
    outputShape[axis] = inputShape[axes[axis]];
  }
  Extent scratchBytes = 0U;
  if (!multiply(rank, 3U * sizeof(Extent), scratchBytes)) {
    return failure(ErrorCode::SizeOverflow,
                   "permutation workspace size overflows");
  }
  auto output = createOutput(outputShape, limits, scratchBytes);
  if (!output.hasValue()) {
    return output;
  }
  std::vector<Extent> outputIndex(rank, 0U);
  std::vector<Extent> inputIndex(rank, 0U);
  const std::span<Scalar> outputData(output.value().data(),
                                     output.value().size());
  const std::span<const Scalar> inputData(input.data(), input.size());
  const Extent outputSize = output.value().size();
  for (Extent linear = 0U; linear < outputSize; ++linear) {
    decodeRowMajor(linear, outputShape, outputIndex);
    for (Extent axis = 0U; axis < rank; ++axis) {
      inputIndex[axes[axis]] = outputIndex[axis];
    }
    outputData[linear] = inputData[rowMajorOffset(inputIndex, inputShape)];
  }
  return output;
}

} // namespace vss::math
