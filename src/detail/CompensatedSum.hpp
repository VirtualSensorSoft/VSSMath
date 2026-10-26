#ifndef VSS_MATH_DETAIL_COMPENSATED_SUM_HPP
#define VSS_MATH_DETAIL_COMPENSATED_SUM_HPP

#include <cmath>
#include <vss/math/Core.hpp>

namespace vss::math::detail {

/// @brief Fixed-storage Neumaier compensated sum for `Scalar` values.
/// @details The accumulator does not validate finiteness; callers must check
///          inputs, state, and the returned total at their algorithm boundary.
///          Updates are deterministic in call order and require constant time
///          and constant auxiliary storage.
class CompensatedSum final {
public:
  /// @brief Adds one value using the Neumaier correction.
  /// @param value Value to add in the caller-defined deterministic order.
  void add(Scalar value) noexcept {
    const Scalar total = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - total) + value;
    } else {
      correction_ += (value - total) + sum_;
    }
    sum_ = total;
  }

  /// @brief Returns the corrected accumulated total.
  [[nodiscard]] Scalar total() const noexcept { return sum_ + correction_; }

private:
  Scalar sum_{0.0};
  Scalar correction_{0.0};
};

} // namespace vss::math::detail

#endif // VSS_MATH_DETAIL_COMPENSATED_SUM_HPP
