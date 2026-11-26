#ifndef VSS_MATH_PROBABILITY_COMPOSITE_HPP
#define VSS_MATH_PROBABILITY_COMPOSITE_HPP

#include <optional>
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

/// @brief Kind of information supplied by one nonideal observation.
enum class ObservationKind : std::uint8_t {
  exact,
  interval,
  below_threshold,
  above_threshold,
  quantized,
  unavailable
};

/// @brief Tagged scalar observation with fields validated against its kind.
/// @details Exact observations use `value`; interval, threshold and quantized
///          observations use `support`. Unavailable observations carry neither
///          and are accepted only when `omit_unavailable` is true.
struct ObservationEvent final {
  /// Event category that determines which optional payload is valid.
  ObservationKind kind{ObservationKind::exact};
  /// Exact measured value; present only when kind is `exact`.
  std::optional<Scalar> value{};
  /// Reported support or bin; present only for interval-like event kinds.
  std::optional<IntervalSupport> support{};
  /// Whether an unavailable event contributes no likelihood term.
  bool omit_unavailable{false};
};

/// @brief Optional log-likelihood contribution for one observation event.
/// @details A disengaged value denotes an explicitly omitted unavailable
///          observation. Negative infinity is a valid value for an impossible
///          event and is distinct from omission.
struct ObservationLikelihood final {
  /// Log density/mass, interval probability, or empty for an omitted event.
  std::optional<Scalar> log_probability{};
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

/// @brief Evaluates the log likelihood contribution of a tagged observation.
/// @details Exact events use density or mass according to the law's measure;
///          interval and quantized events use integrated probability. Dead-band
///          and hysteresis response models remain caller supplied.
[[nodiscard]] Result<ObservationLikelihood>
nonidealObservationLogProbability(const Distribution& law,
                                  const ObservationEvent& event,
                                  ProbabilityPolicy policy);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_COMPOSITE_HPP
