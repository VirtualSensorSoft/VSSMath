#ifndef VSS_MATH_TENSOR_SHAPE_HPP
#define VSS_MATH_TENSOR_SHAPE_HPP

#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

/// @brief Owned extents for a row-major, arbitrary-rank Tensor.
class TensorShape final {
public:
  /// @brief Copies extents and checks their product for size overflow.
  [[nodiscard]] static Result<TensorShape>
  create(std::span<const Extent> extents);
  /// @brief Returns the number of axes; a scalar has rank zero.
  [[nodiscard]] Extent rank() const noexcept;
  /// @brief Returns one extent, throwing std::out_of_range when invalid.
  [[nodiscard]] Extent extent(Index axis) const;
  /// @brief Borrows the owned extents until this shape changes or dies.
  [[nodiscard]] std::span<const Extent> extents() const noexcept;
  /// @brief Returns the checked product; rank zero has one element.
  [[nodiscard]] Extent elementCount() const noexcept;

private:
  TensorShape(std::vector<Extent> extents, Extent elementCount) noexcept;
  std::vector<Extent> extents_;
  Extent elementCount_;
};

} // namespace vss::math

#endif // VSS_MATH_TENSOR_SHAPE_HPP
