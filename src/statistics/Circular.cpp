#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/VectorView.hpp"
#include "vss/math/covariance/PropagationResult.hpp"
#include "vss/math/statistics/Weighted.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vss/math/Tolerance.hpp>
#include <vss/math/statistics/Circular.hpp>

namespace vss::math::statistics {
namespace {
Result<CircularEstimate> fail(ErrorCode code, const char* message) {
  return Result<CircularEstimate>::failure(code, message);
}
class Sum final {
public:
  void add(Scalar x) {
    const Scalar y = x - correction_;
    const Scalar next = value_ + y;
    correction_ = (next - value_) - y;
    value_ = next;
  }

  [[nodiscard]] Scalar value() const noexcept { return value_; }

private:
  Scalar value_{};
  Scalar correction_{};
};
} // namespace

Result<CircularEstimate> circularStatistics(ConstVectorView angles,
                                            const WeightSpec& weights,
                                            Scalar period, Limits limits) {
  const Extent n = angles.size();
  if (n == 0U) {
    return fail(ErrorCode::InvalidExtent, "circular population is empty");
  }
  if (limits.elements < 1U || limits.bytes < sizeof(ErrorCode)) {
    return fail(ErrorCode::ResourceLimit, "circular workspace exceeds limits");
  }
  if (!isFinite(period)) {
    return fail(ErrorCode::NonFiniteInput, "period is non-finite");
  }
  if (!(period > 0.0)) {
    return fail(ErrorCode::InvalidDomain, "period must be positive");
  }
  const bool unweighted = weights.rule == WeightRule::unweighted &&
                          !weights.rowWeights && !weights.componentWeights &&
                          weights.frequencyCounts.empty();
  const bool rowWeighted =
      weights.rule == WeightRule::empirical && weights.rowWeights &&
      weights.rowWeights->size() == n && !weights.componentWeights &&
      weights.frequencyCounts.empty();
  if (!unweighted && !rowWeighted) {
    return fail(ErrorCode::InvalidDomain,
                "circular statistics supports unit or empirical row weights");
  }
  Scalar maxWeight = 0.0;
  for (Extent i = 0; i < n; ++i) {
    if (!isFinite(angles.at(i))) {
      return fail(ErrorCode::NonFiniteInput, "angle is non-finite");
    }
    const Scalar w = unweighted ? 1.0 : weights.rowWeights->at(i);
    if (!isFinite(w)) {
      return fail(ErrorCode::NonFiniteInput, "weight is non-finite");
    }
    if (w < 0.0) {
      return fail(ErrorCode::InvalidDomain, "weight is negative");
    }
    maxWeight = std::max(maxWeight, w);
  }
  if (maxWeight == 0.0) {
    return fail(ErrorCode::InvalidDomain, "all circular weights are zero");
  }
  constexpr Scalar tau = 6.283185307179586476925286766559005768;
  Sum massSum;
  Sum realSum;
  Sum imagSum;
  for (Extent i = 0; i < n; ++i) {
    const Scalar w = (unweighted ? 1.0 : weights.rowWeights->at(i)) / maxWeight;
    const Scalar phase = std::remainder(angles.at(i), period) / period * tau;
    massSum.add(w);
    realSum.add(w * std::cos(phase));
    imagSum.add(w * std::sin(phase));
  }
  const Scalar mass = massSum.value();
  const Scalar real = realSum.value();
  const Scalar imag = imagSum.value();
  if (!(mass > 0.0) || !isFinite(mass) || !isFinite(real) || !isFinite(imag)) {
    return fail(ErrorCode::NonFiniteResult,
                "circular resultant is not representable");
  }
  const Scalar resultant = std::clamp(std::hypot(real, imag) / mass, 0.0, 1.0);
  const Scalar threshold = 32.0 * std::numeric_limits<Scalar>::epsilon();
  std::optional<Scalar> direction;
  std::array<ErrorCode, 1U> warning{ErrorCode::IllConditioned};
  auto warningArray = Array<ErrorCode>::copy(
      resultant > threshold ? std::span<const ErrorCode>{}
                            : std::span<const ErrorCode>{warning},
      Limits{1U, sizeof(ErrorCode), 0U});
  if (!warningArray) {
    return fail(warningArray.status().code(),
                "circular diagnostic allocation failed");
  }
  if (resultant > threshold) {
    Scalar phase = std::atan2(imag, real);
    if (phase < 0.0) {
      phase += tau;
    }
    direction = phase / tau * period;
    if (*direction >= period) {
      *direction = 0.0;
    }
  }
  Support support{.input = n,
                  .valid = n,
                  .positive = 0U,
                  .frequency_mass = std::nullopt,
                  .weight_mass = std::nullopt,
                  .normalized_weight_powers = {},
                  .effective_count = std::nullopt,
                  .degrees_of_freedom = std::nullopt,
                  .squared_weight_mass = std::nullopt};
  for (Extent i = 0; i < n; ++i) {
    if ((unweighted ? 1.0 : weights.rowWeights->at(i)) > 0.0) {
      ++support.positive;
    }
  }
  EstimatorDiagnostics diagnostics{.estimand = Estimand::observed,
                                   .weight = weights.rule,
                                   .center = CenterRule::estimated,
                                   .approximation = Approximation::exact,
                                   .global = support,
                                   .coordinate_support = {},
                                   .tuple_support = {},
                                   .reciprocal_condition = std::nullopt,
                                   .numerical_error = std::nullopt,
                                   .sampling_error = std::nullopt,
                                   .rank = std::nullopt,
                                   .algorithm = "weighted-unit-phasor",
                                   .premises =
                                       "angles share one circular period; "
                                       "empirical row weights are nonnegative",
                                   .warnings = std::move(warningArray).value()};
  return Result<CircularEstimate>::success(
      CircularEstimate{.direction = direction,
                       .resultant_length = resultant,
                       .circular_variance = 1.0 - resultant,
                       .diagnostics = std::move(diagnostics)});
}
} // namespace vss::math::statistics
