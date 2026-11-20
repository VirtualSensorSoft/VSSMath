#ifndef VSS_MATH_SCALED_CONTRACTION_HPP
#define VSS_MATH_SCALED_CONTRACTION_HPP

#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vss/math/Core.hpp>

namespace vss::math::linear::detail {

class ScaledAccumulator final {
public:
  void add(Wide value) {
    const Wide next = sum_ + value;
    correction_ += std::abs(sum_) >= std::abs(value) ? (sum_ - next) + value
                                                     : (value - next) + sum_;
    sum_ = next;
  }

  [[nodiscard]] Wide value(int exponent) const {
    return std::scalbn(sum_ + correction_, exponent);
  }

private:
  Wide sum_{};
  Wide correction_{};
};

// Scaling the product mantissas in Wide preserves the full binary64 product
// exponent span. In particular, two cancelling large terms cannot flush a
// much smaller, representable survivor during normalization.
template <class PairAt> Wide scaledDot(Index count, const PairAt& pair_at) {
  int exponent = std::numeric_limits<int>::min();
  for (Index i = 0; i < count; ++i) {
    const auto [left, right] = pair_at(i);
    if (left == 0 || right == 0) {
      continue;
    }
    int left_exponent = 0;
    int right_exponent = 0;
    static_cast<void>(std::frexp(left, &left_exponent));
    static_cast<void>(std::frexp(right, &right_exponent));
    exponent = std::max(exponent, left_exponent + right_exponent);
  }
  if (exponent == std::numeric_limits<int>::min()) {
    return 0;
  }
  ScaledAccumulator sum;
  for (Index i = 0; i < count; ++i) {
    const auto [left, right] = pair_at(i);
    if (left == 0 || right == 0) {
      continue;
    }
    int left_exponent = 0;
    int right_exponent = 0;
    const Scalar left_fraction = std::frexp(left, &left_exponent);
    const Scalar right_fraction = std::frexp(right, &right_exponent);
    const Scalar product = left_fraction * right_fraction;
    const Scalar error = std::fma(left_fraction, right_fraction, -product);
    const int shift = left_exponent + right_exponent - exponent;
    sum.add(std::scalbn(static_cast<Wide>(product), shift));
    sum.add(std::scalbn(static_cast<Wide>(error), shift));
  }
  return sum.value(exponent);
}

} // namespace vss::math::linear::detail
#endif
