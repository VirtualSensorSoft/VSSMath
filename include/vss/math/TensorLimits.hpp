#ifndef VSS_MATH_TENSOR_LIMITS_HPP
#define VSS_MATH_TENSOR_LIMITS_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>

namespace vss::math {

/// @brief Maximum number of scalar elements permitted in one Tensor.
class TensorLimits final {
public:
  /// @brief Constructs an explicit element ceiling.
  explicit constexpr TensorLimits(Extent maxElements) noexcept;
  /// @brief Returns the configured element ceiling.
  [[nodiscard]] constexpr Extent maxElements() const noexcept;
  /// @brief Checks a requested element count against this ceiling.
  [[nodiscard]] Status validate(Extent requested) const;

private:
  Extent maxElements_;
};

constexpr TensorLimits::TensorLimits(Extent maxElements) noexcept
    : maxElements_(maxElements) {}
constexpr Extent TensorLimits::maxElements() const noexcept {
  return maxElements_;
}

} // namespace vss::math

#endif // VSS_MATH_TENSOR_LIMITS_HPP
