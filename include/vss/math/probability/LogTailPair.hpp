#ifndef VSS_MATH_PROBABILITY_LOG_TAIL_PAIR_HPP
#define VSS_MATH_PROBABILITY_LOG_TAIL_PAIR_HPP

#include <vss/math/Core.hpp>

namespace vss::math {

/// @brief Holds independently useful logarithms of complementary probability
///        tails.
/// @details A zero-probability tail is represented by negative infinity. The
///          values are logarithms and are not required to exponentiate to a
///          sum of exactly one after rounding.
struct LogTailPair final {
  /// Natural logarithm of the cumulative probability.
  Scalar log_cdf{};
  /// Natural logarithm of the complementary probability.
  Scalar log_survival{};
};

} // namespace vss::math

#endif // VSS_MATH_PROBABILITY_LOG_TAIL_PAIR_HPP
