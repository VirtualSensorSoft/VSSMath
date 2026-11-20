#ifndef VSS_MATH_DENSE_WORKSPACE_HPP
#define VSS_MATH_DENSE_WORKSPACE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear::detail {

using Wide = long double;

class StorageBudget final {
public:
  explicit StorageBudget(Limits limits)
      : elements_(limits.elements), bytes_(limits.bytes) {}

  template <class T> [[nodiscard]] bool claim(Index rows, Index columns = 1) {
    if (rows == 0 || columns == 0) {
      return true;
    }
    const auto available = std::min(elements_, bytes_ / sizeof(T));
    if (rows > available / columns ||
        rows > (static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max()) /
                sizeof(T)) /
                   columns) {
      return false;
    }
    const auto count = rows * columns;
    elements_ -= count;
    bytes_ -= count * sizeof(T);
    return true;
  }

private:
  Index elements_;
  Index bytes_;
};

[[nodiscard]] inline bool validNumericalPolicy(const SolvePolicy& policy) {
  const auto nonnegative = [](Scalar x) { return std::isfinite(x) && x >= 0; };
  return nonnegative(policy.rank_abs) && nonnegative(policy.rank_rel) &&
         policy.rank_rel < 1 &&
         nonnegative(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor <= 1 &&
         nonnegative(policy.iteration.abs) && nonnegative(policy.iteration.rel);
}

[[nodiscard]] inline bool representable(Wide value) {
  return std::isfinite(value) &&
         std::abs(value) <=
             static_cast<Wide>(std::numeric_limits<Scalar>::max());
}

// The caller has independently established exact nonsingularity. Check
// selected spectral policies without converting singular values to Scalar.
[[nodiscard]] Status certifyNonsingularSpectrum(ConstMatrixView input,
                                                SolvePolicy policy,
                                                Limits limits);

} // namespace vss::math::linear::detail
#endif
