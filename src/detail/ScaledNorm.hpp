#ifndef VSS_MATH_DETAIL_SCALED_NORM_HPP
#define VSS_MATH_DETAIL_SCALED_NORM_HPP

#include <cmath>
#include <vss/math/Core.hpp>

namespace vss::math::detail {

/// @brief Fixed-storage scale-safe Euclidean norm accumulator.
/// @details The state stores the largest absolute value seen and a normalized
///          sum of squares. It does not validate finiteness; callers check
///          input and output boundaries. Each update is constant time and uses
///          constant auxiliary storage.
class ScaledNorm final {
public:
  /// @brief Adds one value to the norm in deterministic call order.
  /// @param value Value whose absolute magnitude contributes to the norm.
  void add(Scalar value) noexcept {
    const Scalar magnitude = std::abs(value);
    if (magnitude == 0.0) {
      return;
    }
    if (scale_ == 0.0) {
      scale_ = magnitude;
      sumSquares_ = 1.0;
      return;
    }
    if (magnitude > scale_) {
      const Scalar ratio = scale_ / magnitude;
      sumSquares_ = 1.0 + sumSquares_ * ratio * ratio;
      scale_ = magnitude;
    } else {
      const Scalar ratio = magnitude / scale_;
      sumSquares_ += ratio * ratio;
    }
  }

  /// @brief Returns the scale-safe Euclidean norm represented by the state.
  [[nodiscard]] Scalar value() const noexcept {
    return scale_ == 0.0 ? 0.0 : scale_ * std::sqrt(sumSquares_);
  }

private:
  Scalar scale_{0.0};
  Scalar sumSquares_{0.0};
};

} // namespace vss::math::detail

#endif // VSS_MATH_DETAIL_SCALED_NORM_HPP
