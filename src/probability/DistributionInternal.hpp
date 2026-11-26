#ifndef VSS_MATH_DISTRIBUTION_INTERNAL_HPP
#define VSS_MATH_DISTRIBUTION_INTERNAL_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <vss/math/Limits.hpp>
#include <vss/math/probability/Composite.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability::detail {

/// @brief Bounded private copy of the immutable expression needed by numerical
///        algorithms that must inspect composite-law structure.
struct DistributionSnapshot final {
  Family family{};
  Measure measure{};
  std::vector<Scalar> parameters;
  std::uint64_t integer_parameter{};
  std::vector<DistributionSnapshot> components;
  std::vector<Scalar> weights;
  std::vector<Scalar> atoms;
  std::vector<Scalar> masses;
};

class DistributionEngine final {
public:
  static std::shared_ptr<const Distribution::State>
  share(const Distribution& law) noexcept;

  static Result<std::vector<std::pair<Scalar, Scalar>>>
  finiteAtoms(const Distribution& law, Limits limits);

  static Result<DistributionSnapshot> snapshot(const Distribution& law,
                                               Limits limits);

  static Result<Distribution> mixture(std::span<const WeightedLaw> terms,
                                      ProbabilityPolicy policy, Limits limits);

  static Result<Distribution> convolution(std::span<const Distribution> terms,
                                          ProbabilityPolicy policy,
                                          Limits limits);

  static Result<TailValues> evaluate(const Distribution& law, Scalar x,
                                     ProbabilityPolicy policy);

  static Result<Scalar> rawMoment(const Distribution& law, Index order,
                                  ProbabilityPolicy policy);

  static Result<Scalar> quantile(const Distribution& law, Scalar probability,
                                 ProbabilityPolicy policy);

  static Result<Scalar> sample(const Distribution& law, Rng& rng,
                               ProbabilityPolicy policy);

  static Result<Scalar> variance(const Distribution& law,
                                 ProbabilityPolicy policy, Limits limits);

  static Result<Scalar> intervalLogProbability(const Distribution& law,
                                               IntervalSupport interval,
                                               ProbabilityPolicy policy);

  static Result<Distribution> truncated(const Distribution& base,
                                        IntervalSupport interval,
                                        ProbabilityPolicy policy,
                                        Limits limits);
};

} // namespace vss::math::probability::detail

#endif // VSS_MATH_DISTRIBUTION_INTERNAL_HPP
