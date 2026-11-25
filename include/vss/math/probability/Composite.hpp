#ifndef VSS_MATH_PROBABILITY_COMPOSITE_HPP
#define VSS_MATH_PROBABILITY_COMPOSITE_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability {

/// @brief A possibly unbounded interval with explicit endpoint inclusion.
struct IntervalSupport final {
  /// Lower endpoint; negative infinity denotes an unbounded lower side.
  Scalar lower{0.0};
  /// Upper endpoint; positive infinity denotes an unbounded upper side.
  Scalar upper{0.0};
  /// Whether values equal to the lower endpoint belong to the interval.
  bool include_lower{true};
  /// Whether values equal to the upper endpoint belong to the interval.
  bool include_upper{true};
};

/// @brief Return the log density or mass of a normalized mixture at x.
[[nodiscard]] Result<Scalar> mixtureLogPdf(const Distribution& law, Scalar x,
                                           ProbabilityPolicy policy);

/// @brief Return the conditional law restricted to an interval.
[[nodiscard]] Result<Distribution>
truncatedDistribution(const Distribution& base, IntervalSupport interval,
                      ProbabilityPolicy policy);
/// @brief Return the conditional law restricted to an interval under limits.
[[nodiscard]] Result<Distribution>
truncatedDistribution(const Distribution& base, IntervalSupport interval,
                      ProbabilityPolicy policy, Limits limits);

/// @brief Evaluate the cumulative probability of an empirical law at x.
[[nodiscard]] Result<Scalar> empiricalCDF(const Distribution& empirical,
                                          Scalar x);

/// @brief Return the log probability of an observed interval event.
[[nodiscard]] Result<Scalar> censoredLogProbability(const Distribution& law,
                                                    IntervalSupport event,
                                                    ProbabilityPolicy policy);

/// @brief Return the log mass of the quantized interval bin.
[[nodiscard]] Result<Scalar> quantizedLikelihood(const Distribution& law,
                                                 IntervalSupport bin,
                                                 ProbabilityPolicy policy);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_COMPOSITE_HPP
