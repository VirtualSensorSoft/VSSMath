#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>

namespace vss::math {
Tensor::Tensor(TensorShape shape, std::vector<Scalar> values) noexcept
    : shape_(std::move(shape)), values_(std::move(values)) {}

Result<Tensor> Tensor::create(const TensorShape& shape, Scalar fill,
                              const TensorLimits& limits) {
  const Status status = limits.validate(shape.elementCount());
  if (!status.ok()) {
    return Result<Tensor>::failure(status.code(),
                                   std::string{status.message()});
  }
  return Result<Tensor>::success(
      Tensor(shape, std::vector<Scalar>(shape.elementCount(), fill)));
}

const TensorShape& Tensor::shape() const noexcept { return shape_; }
Extent Tensor::size() const noexcept {
  return static_cast<Extent>(values_.size());
}
Scalar* Tensor::data() noexcept { return values_.data(); }
const Scalar* Tensor::data() const noexcept { return values_.data(); }

Index Tensor::linearIndex(std::span<const Index> indices) const {
  if (indices.size() != shape_.rank()) {
    throw std::invalid_argument("tensor index rank does not match shape rank");
  }
  Index offset = 0U;
  for (Index axis = 0U; axis < shape_.rank(); ++axis) {
    const Extent axisExtent = shape_.extent(axis);
    if (indices[axis] >= axisExtent) {
      throw std::out_of_range("IndexOutOfRange: tensor axis " +
                              std::to_string(axis));
    }
    offset = (offset * axisExtent) + indices[axis];
  }
  return offset;
}
Scalar& Tensor::at(std::span<const Index> indices) {
  return values_[linearIndex(indices)];
}
const Scalar& Tensor::at(std::span<const Index> indices) const {
  return values_[linearIndex(indices)];
}
} // namespace vss::math
