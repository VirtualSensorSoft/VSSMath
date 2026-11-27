#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/signal/Series.hpp>

namespace vss::math {
namespace {

using Wide = long double;

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

struct Scaled final {
  Wide fraction{};
  std::int64_t exponent{};
};

bool addExponent(std::int64_t left, std::int64_t right,
                 std::int64_t& result) noexcept {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    return false;
  }
  result = left + right;
  return true;
}

bool makeScaled(Wide value, Scaled& result) noexcept {
  if (value == 0.0L) {
    result = {};
    return true;
  }
  int adjustment = 0;
  result.fraction = std::frexp(value, &adjustment);
  result.exponent = adjustment;
  return std::isfinite(result.fraction);
}

bool multiply(Scaled left, Scaled right, Scaled& result) noexcept {
  if (left.fraction == 0.0L || right.fraction == 0.0L) {
    result = {};
    return true;
  }
  std::int64_t exponent = 0;
  if (!addExponent(left.exponent, right.exponent, exponent)) {
    return false;
  }
  int adjustment = 0;
  result.fraction = std::frexp(left.fraction * right.fraction, &adjustment);
  if (!addExponent(exponent, adjustment, result.exponent)) {
    return false;
  }
  return std::isfinite(result.fraction);
}

bool scalarDifference(Scalar high, Scalar low, Scaled& result) noexcept {
  if constexpr (std::numeric_limits<Wide>::max_exponent >
                std::numeric_limits<Scalar>::max_exponent) {
    return makeScaled(static_cast<Wide>(high) - static_cast<Wide>(low), result);
  }
  const Scalar difference = high - low;
  if (std::isfinite(difference)) {
    return makeScaled(static_cast<Wide>(difference), result);
  }
  const Scalar halfDifference = (high * 0.5) - (low * 0.5);
  if (!makeScaled(static_cast<Wide>(halfDifference), result)) {
    return false;
  }
  return addExponent(result.exponent, 1, result.exponent);
}

bool scalarMidpoint(Scalar left, Scalar right, Scaled& result) noexcept {
  Wide midpoint = 0.0L;
  if constexpr (std::numeric_limits<Wide>::max_exponent >
                std::numeric_limits<Scalar>::max_exponent) {
    midpoint =
        (static_cast<Wide>(left) * 0.5L) + (static_cast<Wide>(right) * 0.5L);
  } else {
    midpoint = static_cast<Wide>(std::midpoint(left, right));
  }
  return makeScaled(midpoint, result);
}

class ScaledAccumulator final {
public:
  void add(Scaled value) noexcept {
    if (value.fraction == 0.0L) {
      return;
    }
    if (!initialized_) {
      sum_ = value.fraction;
      exponent_ = value.exponent;
      initialized_ = true;
      return;
    }
    if (value.exponent > exponent_) {
      const auto shift = exponent_ - value.exponent;
      sum_ = std::scalbn(sum_, boundedShift(shift));
      correction_ = std::scalbn(correction_, boundedShift(shift));
      exponent_ = value.exponent;
    }
    const auto shift = value.exponent - exponent_;
    const Wide term = std::scalbn(value.fraction, boundedShift(shift));
    const Wide next = sum_ + term;
    if (std::abs(sum_) >= std::abs(term)) {
      correction_ += (sum_ - next) + term;
    } else {
      correction_ += (term - next) + sum_;
    }
    sum_ = next;
    normalize();
  }

  [[nodiscard]] bool toScalar(Scalar& result) const noexcept {
    const Wide total = sum_ + correction_;
    if (total == 0.0L) {
      result = 0.0;
      return true;
    }
    int adjustment = 0;
    const Wide fraction = std::frexp(total, &adjustment);
    std::int64_t exponent = 0;
    if (!addExponent(exponent_, adjustment, exponent) ||
        exponent > std::numeric_limits<int>::max() ||
        exponent < std::numeric_limits<int>::min()) {
      return false;
    }
    result =
        std::scalbn(static_cast<Scalar>(fraction), static_cast<int>(exponent));
    return std::isfinite(result);
  }

private:
  static int boundedShift(std::int64_t shift) noexcept {
    if (shift > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::max();
    }
    if (shift < std::numeric_limits<int>::min()) {
      return std::numeric_limits<int>::min();
    }
    return static_cast<int>(shift);
  }

  void normalize() noexcept {
    const Wide total = sum_ + correction_;
    if (total == 0.0L || !std::isfinite(total)) {
      return;
    }
    int adjustment = 0;
    static_cast<void>(std::frexp(total, &adjustment));
    if (adjustment == 0) {
      return;
    }
    sum_ = std::scalbn(sum_, -adjustment);
    correction_ = std::scalbn(correction_, -adjustment);
    exponent_ += adjustment;
  }

  Wide sum_{};
  Wide correction_{};
  std::int64_t exponent_{};
  bool initialized_{};
};

bool cadenceMatches(const SeriesView& series) {
  const std::optional<Scalar> declaredStep = series.uniformStep();
  if (!declaredStep.has_value()) {
    return true;
  }
  const Scalar step = declaredStep.value();
  if (!std::isfinite(step) || step <= 0.0) {
    return false;
  }
  const Wide origin =
      series.times().empty() ? 0.0L : static_cast<Wide>(series.times().at(0U));
  const Wide wideStep = static_cast<Wide>(step);
  for (Index row = 1U; row < series.times().size(); ++row) {
    const Wide elapsed = static_cast<Wide>(series.times().at(row)) - origin;
    const Wide expected = static_cast<Wide>(row) * wideStep;
    const Wide scale = std::max({wideStep, std::abs(elapsed), expected});
    const Wide tolerance =
        8.0L * static_cast<Wide>(std::numeric_limits<Scalar>::epsilon()) *
        scale;
    if (!std::isfinite(elapsed) || !std::isfinite(expected) ||
        std::abs(elapsed - expected) > tolerance) {
      return false;
    }
  }
  return true;
}

ErrorCode validateSeries(const SeriesView& series, bool scalarOnly) {
  const Index count = series.times().size();
  if (series.rows().rows() != count || series.rows().columns() == 0U ||
      (scalarOnly && series.rows().columns() != 1U) ||
      (!series.valid().empty() && series.valid().size() != count)) {
    return ErrorCode::DimensionMismatch;
  }
  for (const std::uint8_t valid : series.valid()) {
    if (valid > 1U || valid == 0U) {
      return ErrorCode::InvalidDomain;
    }
  }
  for (Index row = 0U; row < count; ++row) {
    const Scalar time = series.times().at(row);
    if (!std::isfinite(time)) {
      return ErrorCode::NonFiniteInput;
    }
    if (row > 0U && !(time > series.times().at(row - 1U))) {
      return ErrorCode::InvalidDomain;
    }
    for (Index column = 0U; column < series.rows().columns(); ++column) {
      if (!std::isfinite(series.rows().at(row, column))) {
        return ErrorCode::NonFiniteInput;
      }
    }
  }
  return cadenceMatches(series) ? ErrorCode::None : ErrorCode::InvalidDomain;
}

bool checkedMultiply(Index left, Index right, Index& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

bool checkedAdd(Index left, Index right, Index& result) noexcept {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool weightPayloadValid(const WeightSpec& weights, Index count) noexcept {
  const bool hasRows = weights.rowWeights.has_value();
  const bool hasComponents = weights.componentWeights.has_value();
  const bool hasCounts = !weights.frequencyCounts.empty();
  switch (weights.rule) {
  case WeightRule::unweighted:
    return !hasRows && !hasComponents && !hasCounts;
  case WeightRule::empirical:
  case WeightRule::fixed_iid:
  case WeightRule::precision:
  case WeightRule::importance:
    return hasRows && !hasComponents && !hasCounts &&
           weights.rowWeights->size() == count;
  case WeightRule::frequency:
    return !hasRows && !hasComponents && (hasCounts || count == 0U) &&
           weights.frequencyCounts.size() == count;
  case WeightRule::componentwise:
  case WeightRule::dependent:
    return false;
  }
  return false;
}

ErrorCode weightPayloadError(const WeightSpec& weights, Index count) noexcept {
  if (weights.rowWeights.has_value() && weights.rowWeights->size() != count) {
    return ErrorCode::DimensionMismatch;
  }
  if (weights.componentWeights.has_value()) {
    return ErrorCode::InvalidDomain;
  }
  if (weights.rule == WeightRule::frequency &&
      weights.frequencyCounts.size() != count) {
    return ErrorCode::DimensionMismatch;
  }
  return ErrorCode::InvalidDomain;
}

Wide rowWeight(const WeightSpec& weights, Index row) {
  if (weights.rule == WeightRule::unweighted) {
    return 1.0L;
  }
  if (weights.rule == WeightRule::frequency) {
    return static_cast<Wide>(weights.frequencyCounts[row]);
  }
  if (!weights.rowWeights.has_value()) {
    return 0.0L;
  }
  return static_cast<Wide>(weights.rowWeights->at(row));
}

Scaled squareRootNormalizedWeight(Wide weight, Wide maximum,
                                  Wide scaledTotal) noexcept {
  int weightExponent = 0;
  int maximumExponent = 0;
  int totalExponent = 0;
  const Wide weightFraction = std::frexp(weight, &weightExponent);
  const Wide maximumFraction = std::frexp(maximum, &maximumExponent);
  const Wide totalFraction = std::frexp(scaledTotal, &totalExponent);
  Wide fraction = weightFraction / maximumFraction / totalFraction;
  std::int64_t exponent = static_cast<std::int64_t>(weightExponent) -
                          maximumExponent - totalExponent;
  if ((exponent % 2) != 0) {
    fraction *= 2.0L;
    --exponent;
  }
  Scaled result{};
  int adjustment = 0;
  result.fraction = std::frexp(std::sqrt(fraction), &adjustment);
  result.exponent = (exponent / 2) + adjustment;
  return result;
}

bool scaledToScalar(Scaled value, std::int64_t downshift,
                    Scalar& result) noexcept {
  if (value.fraction == 0.0L) {
    result = 0.0;
    return true;
  }
  std::int64_t exponent = 0;
  if (!addExponent(value.exponent, -downshift, exponent)) {
    return false;
  }
  if (exponent > std::numeric_limits<int>::max()) {
    return false;
  }
  if (exponent < std::numeric_limits<int>::min()) {
    return false;
  }
  result = std::scalbn(static_cast<Scalar>(value.fraction),
                       static_cast<int>(exponent));
  return std::isfinite(result) && result != 0.0;
}

} // namespace

Result<Vector> cumulativeIntegral(const SeriesView& series, Limits limits) {
  const ErrorCode validation = validateSeries(series, true);
  if (validation != ErrorCode::None) {
    return failure<Vector>(validation, "cumulative integral series is invalid");
  }
  const Index count = series.times().size();
  if (count > limits.elements || count > limits.bytes / sizeof(Scalar)) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative integral output exceeds limits");
  }
  try {
    Vector output(count, 0.0);
    ScaledAccumulator accumulator;
    for (Index row = 1U; row < count; ++row) {
      Scaled duration{};
      Scaled midpoint{};
      Scaled area{};
      if (!scalarDifference(series.times().at(row), series.times().at(row - 1U),
                            duration) ||
          !scalarMidpoint(series.rows().at(row - 1U, 0U),
                          series.rows().at(row, 0U), midpoint) ||
          !multiply(duration, midpoint, area)) {
        return failure<Vector>(ErrorCode::NonFiniteResult,
                               "trapezoid area is not representable");
      }
      accumulator.add(area);
      if (!accumulator.toScalar(output.at(row))) {
        return failure<Vector>(ErrorCode::NonFiniteResult,
                               "integral prefix is not representable");
      }
    }
    return Result<Vector>::success(std::move(output));
  } catch (const std::bad_alloc&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative integral allocation failed");
  } catch (const std::length_error&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative integral extent is unsupported");
  }
}

Result<LinearEstimate> trend(const SeriesView& series, Index degree,
                             const WeightSpec& weights, SolvePolicy policy,
                             Limits limits) {
  const ErrorCode validation = validateSeries(series, false);
  if (validation != ErrorCode::None) {
    return failure<LinearEstimate>(validation, "trend series is invalid");
  }
  const Index count = series.times().size();
  if (degree == std::numeric_limits<Index>::max()) {
    return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                   "trend basis extent overflows");
  }
  const Index columns = degree + 1U;
  if (!weightPayloadValid(weights, count)) {
    return failure<LinearEstimate>(weightPayloadError(weights, count),
                                   "trend weight rule or payload is invalid");
  }

  Wide maximumWeight = 0.0L;
  for (Index row = 0U; row < count; ++row) {
    const Wide weight = rowWeight(weights, row);
    if (!std::isfinite(weight) || weight < 0.0L) {
      return failure<LinearEstimate>(
          ErrorCode::InvalidDomain,
          "trend weights must be finite and nonnegative");
    }
    maximumWeight = std::max(maximumWeight, weight);
  }
  if (maximumWeight == 0.0L) {
    return failure<LinearEstimate>(ErrorCode::InsufficientSamples,
                                   "trend needs a positive row weight");
  }

  Wide scaledTotal = 0.0L;
  Wide correction = 0.0L;
  Index retained = 0U;
  for (Index row = 0U; row < count; ++row) {
    const Wide weight = rowWeight(weights, row);
    if (weight == 0.0L) {
      continue;
    }
    ++retained;
    const Wide term = weight / maximumWeight;
    const Wide adjusted = term - correction;
    const Wide next = scaledTotal + adjusted;
    correction = (next - scaledTotal) - adjusted;
    scaledTotal = next;
  }

  const Index channels = series.rows().columns();
  Index designElements = 0U;
  Index responseElements = 0U;
  Index ownElements = 0U;
  Index ownBytes = 0U;
  if (!checkedMultiply(retained, columns, designElements) ||
      !checkedMultiply(retained, channels, responseElements) ||
      !checkedAdd(designElements, responseElements, ownElements) ||
      !checkedMultiply(ownElements, sizeof(Scalar), ownBytes) ||
      ownElements > limits.elements || ownBytes > limits.bytes) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "trend design and response exceed limits");
  }
  const Limits solverLimits{.elements = limits.elements - ownElements,
                            .bytes = limits.bytes - ownBytes,
                            .evaluations = limits.evaluations};

  const Scalar origin = count == 0U ? 0.0 : series.times().at(0U);
  Scaled maximumEntry{};
  bool hasEntry = false;
  for (Index row = 0U; row < count; ++row) {
    const Wide weight = rowWeight(weights, row);
    if (weight == 0.0L) {
      continue;
    }
    const Scaled rootWeight =
        squareRootNormalizedWeight(weight, maximumWeight, scaledTotal);
    Scaled elapsed{};
    if (!scalarDifference(series.times().at(row), origin, elapsed)) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                     "elapsed trend time is not representable");
    }
    Scaled power{.fraction = 0.5L, .exponent = 1};
    for (Index column = 0U; column < columns; ++column) {
      Scaled entry{};
      if (!multiply(rootWeight, power, entry)) {
        return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                       "trend basis exponent overflows");
      }
      if (!hasEntry || entry.exponent > maximumEntry.exponent ||
          (entry.exponent == maximumEntry.exponent &&
           std::abs(entry.fraction) > std::abs(maximumEntry.fraction))) {
        maximumEntry = entry;
        hasEntry = true;
      }
      if (column + 1U < columns && !multiply(power, elapsed, power)) {
        return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                       "trend basis exponent overflows");
      }
    }
    for (Index channel = 0U; channel < channels; ++channel) {
      Scaled input{};
      Scaled weighted{};
      if (!makeScaled(static_cast<Wide>(series.rows().at(row, channel)),
                      input) ||
          !multiply(rootWeight, input, weighted)) {
        return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                       "weighted trend response overflows");
      }
      if (!hasEntry || weighted.exponent > maximumEntry.exponent ||
          (weighted.exponent == maximumEntry.exponent &&
           std::abs(weighted.fraction) > std::abs(maximumEntry.fraction))) {
        maximumEntry = weighted;
        hasEntry = true;
      }
    }
  }

  std::int64_t downshift = 0;
  if (hasEntry) {
    downshift = std::max<std::int64_t>(
        0, maximumEntry.exponent -
               (std::numeric_limits<Scalar>::max_exponent - 2));
  }
  if (downshift > std::numeric_limits<int>::max()) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "trend scaling exponent is unsupported");
  }
  const int scaleShift = static_cast<int>(downshift);
  policy.rank_abs = std::scalbn(policy.rank_abs, -scaleShift);

  try {
    Matrix design(retained, columns, 0.0);
    Matrix response(retained, channels, 0.0);
    Index outputRow = 0U;
    for (Index row = 0U; row < count; ++row) {
      const Wide weight = rowWeight(weights, row);
      if (weight == 0.0L) {
        continue;
      }
      const Scaled rootWeight =
          squareRootNormalizedWeight(weight, maximumWeight, scaledTotal);
      Scaled elapsed{};
      static_cast<void>(
          scalarDifference(series.times().at(row), origin, elapsed));
      Scaled power{.fraction = 0.5L, .exponent = 1};
      for (Index column = 0U; column < columns; ++column) {
        Scaled entry{};
        static_cast<void>(multiply(rootWeight, power, entry));
        Scalar converted = 0.0;
        if (!scaledToScalar(entry, downshift, converted)) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "trend design is not representable");
        }
        design.at(outputRow, column) = converted;
        if (column + 1U < columns) {
          static_cast<void>(multiply(power, elapsed, power));
        }
      }
      for (Index channel = 0U; channel < channels; ++channel) {
        Scaled input{};
        Scaled weighted{};
        static_cast<void>(makeScaled(
            static_cast<Wide>(series.rows().at(row, channel)), input));
        static_cast<void>(multiply(rootWeight, input, weighted));
        Scalar converted = 0.0;
        if (!scaledToScalar(weighted, downshift, converted)) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "trend response is not representable");
        }
        response.at(outputRow, channel) = converted;
      }
      ++outputRow;
    }

    auto estimate = linear::leastSquaresMinNorm(design.view(), response.view(),
                                                policy, solverLimits);
    if (!estimate.hasValue()) {
      return Result<LinearEstimate>::failure(
          estimate.status().code(), std::string(estimate.status().message()));
    }
    if (scaleShift != 0) {
      const Scalar cutoff =
          std::scalbn(estimate.value().rank_cutoff, scaleShift);
      if (!std::isfinite(cutoff)) {
        return failure<LinearEstimate>(
            ErrorCode::NonFiniteResult,
            "trend rank cutoff is not representable");
      }
      estimate.value().rank_cutoff = cutoff;
      for (Index channel = 0U; channel < estimate.value().residual_norms.size();
           ++channel) {
        const Scalar residual = std::scalbn(
            estimate.value().residual_norms.at(channel), scaleShift);
        if (!std::isfinite(residual)) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "trend residual is not representable");
        }
        estimate.value().residual_norms.at(channel) = residual;
      }
    }
    return estimate;
  } catch (const std::bad_alloc&) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "trend matrix allocation failed");
  } catch (const std::length_error&) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "trend matrix extent is unsupported");
  }
}

} // namespace vss::math
