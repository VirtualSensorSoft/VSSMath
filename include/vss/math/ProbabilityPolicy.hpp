#ifndef VSS_MATH_PROBABILITY_POLICY_HPP
#define VSS_MATH_PROBABILITY_POLICY_HPP

#include <vss/math/Core.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Accuracy and work limits for probability algorithms.
/// @details The iteration policy is borrowed by value for each call. The
///          draw limit is reserved for sampling operations and is not consumed
///          by deterministic special functions.
struct ProbabilityPolicy final {
  /// Iteration and convergence limits for iterative probability algorithms.
  IterPolicy iteration{};
  /// Maximum random draws permitted by sampling operations.
  Index max_draws{};
  /// Absolute error budget for log-domain probability calculations.
  Scalar abs_log_error{};
};

} // namespace vss::math

#endif // VSS_MATH_PROBABILITY_POLICY_HPP
