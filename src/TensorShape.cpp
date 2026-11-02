#include <vss/math/TensorShape.hpp>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vss::math {

TensorShape::TensorShape(std::vector<Extent> extents, Extent elementCount) noexcept
    : extents_(std::move(extents)), elementCount_(elementCount) {}

Result<TensorShape> TensorShape::create(std::span<const Extent> extents) {
  Extent count = 1U;
  for (const Extent extentValue : extents) {
    if (extentValue != 0U && count > std::numeric_limits<Extent>::max() / extentValue) {
      return Result<TensorShape>::failure(
          ErrorCode::SizeOverflow, "tensor shape element count overflows Extent");
    }
    count *= extentValue;
  }
  return Result<TensorShape>::success(
      TensorShape(std::vector<Extent>{extents.begin(), extents.end()}, count));
}

Extent TensorShape::rank() const noexcept {
  return static_cast<Extent>(extents_.size());
}

Extent TensorShape::extent(Index axis) const {
  if (axis >= extents_.size()) {
    throw std::out_of_range("IndexOutOfRange: tensor shape axis " +
                            std::to_string(axis));
  }
  return extents_[axis];
}

std::span<const Extent> TensorShape::extents() const noexcept {
  return extents_;
}

Extent TensorShape::elementCount() const noexcept { return elementCount_; }

} // namespace vss::math
