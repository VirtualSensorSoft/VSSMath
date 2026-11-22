#include <algorithm>
#include <cmath>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <initializer_list>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedMultiply(Extent left, Extent right, Extent& result) noexcept {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool checkedAdd(Extent left, Extent right, Extent& result) noexcept {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

struct WeightedValue final {
  Scalar value;
  Scalar normalized_weight;
  std::uint64_t frequency;
};

Result<Scalar> finiteProduct(std::initializer_list<Scalar> factors,
                             const char* message) {
  Scalar fraction = 1.0;
  int exponent = 0;
  for (const Scalar factor : factors) {
    if (!std::isfinite(factor)) {
      return fail<Scalar>(ErrorCode::NonFiniteResult, message);
    }
    if (factor == 0.0) {
      return Result<Scalar>::success(0.0);
    }
    int factorExponent = 0;
    const Scalar factorFraction = std::frexp(factor, &factorExponent);
    int adjustment = 0;
    fraction = std::frexp(fraction * factorFraction, &adjustment);
    if ((factorExponent > 0 &&
         exponent > std::numeric_limits<int>::max() - factorExponent) ||
        (factorExponent < 0 &&
         exponent < std::numeric_limits<int>::min() - factorExponent)) {
      return fail<Scalar>(ErrorCode::NonFiniteResult, message);
    }
    exponent += factorExponent;
    if ((adjustment > 0 &&
         exponent > std::numeric_limits<int>::max() - adjustment) ||
        (adjustment < 0 &&
         exponent < std::numeric_limits<int>::min() - adjustment)) {
      return fail<Scalar>(ErrorCode::NonFiniteResult, message);
    }
    exponent += adjustment;
  }
  const Scalar result = std::scalbn(fraction, exponent);
  if (!std::isfinite(result)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult, message);
  }
  return Result<Scalar>::success(result);
}

Scalar interpolated(Scalar lower, Scalar upper, long double fraction) {
  return std::lerp(lower, upper, static_cast<Scalar>(fraction));
}

Result<Scalar> weightedQuantile(std::vector<WeightedValue>& values,
                                Scalar probability, WeightRule rule,
                                Scalar normalizedMass) {
  if (probability == 0.0) {
    return Result<Scalar>::success(values.front().value);
  }
  if (probability == 1.0) {
    return Result<Scalar>::success(values.back().value);
  }
  if (rule == WeightRule::unweighted) {
    const long double position =
        static_cast<long double>(values.size() - 1U) * probability;
    const auto lower = static_cast<Extent>(std::floor(position));
    const auto upper = std::min(lower + 1U, values.size() - 1U);
    return Result<Scalar>::success(interpolated(
        values[lower].value, values[upper].value, position - lower));
  }
  if (rule == WeightRule::frequency) {
    std::uint64_t total = 0U;
    for (const auto& entry : values) {
      if (entry.frequency > std::numeric_limits<std::uint64_t>::max() - total) {
        return fail<Scalar>(ErrorCode::SizeOverflow,
                            "frequency quantile count overflows");
      }
      total += entry.frequency;
    }
    const long double position =
        static_cast<long double>(total - 1U) * probability;
    const auto lowerRank = static_cast<std::uint64_t>(std::floor(position));
    const auto upperRank = std::min(lowerRank + 1U, total - 1U);
    auto atRank = [&](std::uint64_t rank) {
      std::uint64_t cumulative = 0U;
      for (const auto& entry : values) {
        cumulative += entry.frequency;
        if (rank < cumulative) {
          return entry.value;
        }
      }
      return values.back().value;
    };
    return Result<Scalar>::success(
        interpolated(atRank(lowerRank), atRank(upperRank),
                     position - static_cast<long double>(lowerRank)));
  }
  const long double target =
      static_cast<long double>(probability) * normalizedMass;
  detail::CompensatedSum cumulative;
  for (const auto& entry : values) {
    cumulative.add(entry.normalized_weight);
    if (cumulative.total() >= target) {
      return Result<Scalar>::success(entry.value);
    }
  }
  return Result<Scalar>::success(values.back().value);
}

} // namespace

Result<Summary> descriptiveSummary(const SampleView& sample,
                                   const WeightSpec& weights,
                                   const SummarySpec& spec, Limits limits) {
  const Extent rows = sample.rows.rows();
  const Extent dimension = sample.rows.columns();
  if (rows == 0U) {
    return fail<Summary>(ErrorCode::InsufficientSamples,
                         "descriptive summary requires at least one row");
  }
  if (dimension == 0U ||
      (sample.observation_shape.empty() && dimension != 1U)) {
    return fail<Summary>(ErrorCode::InvalidExtent,
                         "sample shape does not describe its coordinates");
  }
  if (!sample.observation_shape.empty()) {
    Extent product = 1U;
    for (const Index extent : sample.observation_shape) {
      if (extent == 0U || !checkedMultiply(product, extent, product)) {
        return fail<Summary>(ErrorCode::SizeOverflow,
                             "sample shape product is invalid or overflows");
      }
    }
    if (product != dimension) {
      return fail<Summary>(ErrorCode::DimensionMismatch,
                           "sample shape does not match its column count");
    }
  }
  Extent cells = 0U;
  if (!checkedMultiply(rows, dimension, cells)) {
    return fail<Summary>(ErrorCode::SizeOverflow,
                         "sample cell count overflows");
  }
  if (!sample.valid.empty()) {
    if (sample.valid.size() != cells) {
      return fail<Summary>(ErrorCode::DimensionMismatch,
                           "sample validity mask has the wrong size");
    }
    for (const auto cell : sample.valid) {
      if (cell > 1U || cell == 0U) {
        return fail<Summary>(ErrorCode::InvalidDomain,
                             "descriptive summary requires complete rows");
      }
    }
  }
  for (const Scalar probability : spec.quantile_probabilities) {
    if (!std::isfinite(probability)) {
      return fail<Summary>(ErrorCode::NonFiniteInput,
                           "quantile probability is non-finite");
    }
    if (probability < 0.0 || probability > 1.0) {
      return fail<Summary>(ErrorCode::InvalidDomain,
                           "quantile probability must lie in [0, 1]");
    }
  }
  if (spec.variance_mode != VarianceMode::empirical &&
      spec.variance_mode != VarianceMode::iid_sample &&
      spec.variance_mode != VarianceMode::known_center) {
    return fail<Summary>(ErrorCode::InvalidDomain,
                         "unsupported descriptive variance mode");
  }
  if (spec.variance_mode == VarianceMode::known_center) {
    return fail<Summary>(ErrorCode::InvalidDomain,
                         "summary has no known-center parameter");
  }

  if (weights.componentWeights.has_value()) {
    return fail<Summary>(ErrorCode::InvalidDomain,
                         "component weights are unsupported by summary");
  }
  if (weights.rule == WeightRule::unweighted) {
    if (weights.rowWeights.has_value() || !weights.frequencyCounts.empty()) {
      return fail<Summary>(ErrorCode::InvalidDomain,
                           "unweighted summary received weight data");
    }
  } else if (weights.rule == WeightRule::frequency) {
    if (weights.rowWeights.has_value() ||
        weights.frequencyCounts.size() != rows) {
      return fail<Summary>(ErrorCode::DimensionMismatch,
                           "frequency counts must match the row count");
    }
  } else if (weights.rule == WeightRule::empirical ||
             weights.rule == WeightRule::fixed_iid ||
             weights.rule == WeightRule::importance) {
    if (!weights.rowWeights.has_value() || weights.rowWeights->size() != rows ||
        !weights.frequencyCounts.empty()) {
      return fail<Summary>(ErrorCode::DimensionMismatch,
                           "row weights must match the row count");
    }
  } else {
    return fail<Summary>(ErrorCode::InvalidDomain,
                         "weight rule is unsupported by descriptive summary");
  }
  if (spec.variance_mode == VarianceMode::iid_sample &&
      weights.rule != WeightRule::unweighted &&
      weights.rule != WeightRule::frequency &&
      weights.rule != WeightRule::fixed_iid) {
    return fail<Summary>(ErrorCode::InvalidDomain,
                         "IID variance requires unweighted, frequency, or "
                         "fixed-IID weights");
  }

  std::uint64_t frequencyMass = 0U;
  Scalar maximumWeight = 0.0;
  Extent positiveRows = 0U;
  for (Index row = 0U; row < rows; ++row) {
    Scalar weight = 1.0;
    if (weights.rule == WeightRule::frequency) {
      const auto count = weights.frequencyCounts[row];
      if (count > std::numeric_limits<std::uint64_t>::max() - frequencyMass) {
        return fail<Summary>(ErrorCode::SizeOverflow,
                             "frequency mass overflows uint64");
      }
      frequencyMass += count;
      weight = static_cast<Scalar>(count);
    } else if (weights.rule != WeightRule::unweighted) {
      weight = weights.rowWeights.value().at(row);
      if (!std::isfinite(weight)) {
        return fail<Summary>(ErrorCode::NonFiniteInput,
                             "row weight is non-finite");
      }
      if (weight < 0.0) {
        return fail<Summary>(ErrorCode::InvalidDomain,
                             "row weight must be nonnegative");
      }
    }
    if (weight > 0.0) {
      ++positiveRows;
      maximumWeight = std::max(maximumWeight, weight);
    }
  }
  if (positiveRows == 0U) {
    return fail<Summary>(ErrorCode::InsufficientSamples,
                         "summary requires positive weight mass");
  }
  if (weights.rule == WeightRule::frequency && frequencyMass == 0U) {
    return fail<Summary>(ErrorCode::InsufficientSamples,
                         "frequency summary requires positive count mass");
  }

  for (Index row = 0U; row < rows; ++row) {
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      if (!std::isfinite(sample.rows.at(row, coordinate))) {
        return fail<Summary>(ErrorCode::NonFiniteInput,
                             "complete summary input contains a non-finite "
                             "value");
      }
    }
  }

  Extent scalarOutputs = 0U;
  Extent quantileOutputs = 0U;
  Extent scalarAndQuantileOutputs = 0U;
  Extent requiredElements = 0U;
  if (!checkedMultiply(dimension, 8U, scalarOutputs) ||
      !checkedMultiply(dimension, spec.quantile_probabilities.size(),
                       quantileOutputs) ||
      !checkedAdd(scalarOutputs, quantileOutputs, scalarAndQuantileOutputs)) {
    return fail<Summary>(ErrorCode::SizeOverflow,
                         "summary output or workspace size overflows");
  }
  requiredElements = scalarAndQuantileOutputs;
  Extent outputBytes = 0U;
  Extent workspaceBytes = 0U;
  Extent supportBytes = 0U;
  Extent statusBytes = 0U;
  Extent retainedMetadataCount = 0U;
  if (!checkedMultiply(dimension, 2U, retainedMetadataCount) ||
      !checkedMultiply(scalarAndQuantileOutputs, sizeof(Scalar), outputBytes) ||
      !checkedMultiply(retainedMetadataCount, sizeof(Support), supportBytes) ||
      !checkedMultiply(retainedMetadataCount, sizeof(ErrorCode), statusBytes) ||
      !checkedAdd(supportBytes, statusBytes, workspaceBytes) ||
      !checkedAdd(outputBytes, workspaceBytes, outputBytes) ||
      !checkedMultiply(rows, sizeof(WeightedValue), workspaceBytes) ||
      !checkedAdd(outputBytes, workspaceBytes, outputBytes)) {
    return fail<Summary>(ErrorCode::SizeOverflow,
                         "summary byte requirement overflows");
  }
  if (requiredElements > limits.elements || outputBytes > limits.bytes) {
    return fail<Summary>(ErrorCode::ResourceLimit,
                         "summary exceeds configured storage limits");
  }

  try {
    Vector sums(dimension);
    Vector minimums(dimension);
    Vector maximums(dimension);
    Vector rmsValues(dimension);
    Vector means(dimension);
    Vector variances(dimension);
    Vector ranges(dimension);
    Vector deviations(dimension);
    Matrix quantiles(dimension, spec.quantile_probabilities.size());
    std::vector<ErrorCode> varianceStatus(dimension, ErrorCode::None);
    std::vector<Support> coordinateSupport;
    coordinateSupport.reserve(dimension);

    detail::CompensatedSum normalizedMass;
    detail::CompensatedSum normalizedSquares;
    for (Index row = 0U; row < rows; ++row) {
      Scalar weight = 1.0;
      if (weights.rule == WeightRule::frequency) {
        weight = static_cast<Scalar>(weights.frequencyCounts[row]);
      } else if (weights.rule != WeightRule::unweighted) {
        weight = weights.rowWeights.value().at(row);
      }
      if (weight > 0.0) {
        const Scalar normalized = weight / maximumWeight;
        normalizedMass.add(normalized);
        normalizedSquares.add(normalized * normalized);
      }
    }
    const Scalar mass = normalizedMass.total();
    const Scalar squareMass = normalizedSquares.total();
    if (!(mass > 0.0) || !std::isfinite(mass) || !(squareMass > 0.0) ||
        !std::isfinite(squareMass)) {
      return fail<Summary>(ErrorCode::NonFiniteResult,
                           "normalized summary weights are not representable");
    }
    const long double actualMass =
        static_cast<long double>(maximumWeight) * mass;
    std::optional<Scalar> recordedMass;
    if (actualMass <=
        static_cast<long double>(std::numeric_limits<Scalar>::max())) {
      recordedMass = static_cast<Scalar>(actualMass);
    }
    Scalar effectiveCount =
        (mass / std::sqrt(squareMass)) * (mass / std::sqrt(squareMass));
    Scalar normalizedSquarePower = squareMass / (mass * mass);
    std::optional<Scalar> normalizedCubePower = std::nullopt;
    std::optional<Scalar> squaredWeightMass = std::nullopt;
    if (weights.rule == WeightRule::frequency) {
      effectiveCount = static_cast<Scalar>(frequencyMass);
      normalizedSquarePower = 1.0 / effectiveCount;
      normalizedCubePower = 1.0 / (effectiveCount * effectiveCount);
      squaredWeightMass = effectiveCount;
    }
    const Scalar normalizedSquarePowerForCorrection =
        squareMass / (mass * mass);
    Support global{.input = rows,
                   .valid = rows,
                   .positive = positiveRows,
                   .frequency_mass =
                       weights.rule == WeightRule::frequency
                           ? std::optional<std::uint64_t>{frequencyMass}
                           : std::nullopt,
                   .weight_mass = recordedMass,
                   .normalized_weight_powers = {1.0, normalizedSquarePower,
                                                normalizedCubePower},
                   .effective_count = effectiveCount,
                   .degrees_of_freedom = std::nullopt,
                   .squared_weight_mass = squaredWeightMass};
    if (spec.variance_mode == VarianceMode::iid_sample) {
      if (weights.rule == WeightRule::frequency && frequencyMass >= 2U) {
        global.degrees_of_freedom = static_cast<Scalar>(frequencyMass - 1U);
      } else if (weights.rule != WeightRule::frequency &&
                 1.0 - normalizedSquarePowerForCorrection > 0.0) {
        global.degrees_of_freedom =
            mass * (1.0 - normalizedSquarePowerForCorrection);
      }
    }

    std::vector<WeightedValue> ordered;
    ordered.reserve(rows);
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      ordered.clear();
      Scalar valueScale = 0.0;
      for (Index row = 0U; row < rows; ++row) {
        Scalar weight = 1.0;
        std::uint64_t count = 1U;
        if (weights.rule == WeightRule::frequency) {
          count = weights.frequencyCounts[row];
          weight = static_cast<Scalar>(count);
        } else if (weights.rule != WeightRule::unweighted) {
          weight = weights.rowWeights.value().at(row);
        }
        if (weight <= 0.0) {
          continue;
        }
        const Scalar value = sample.rows.at(row, coordinate);
        valueScale = std::max(valueScale, std::abs(value));
        ordered.push_back({.value = value,
                           .normalized_weight = weight / maximumWeight,
                           .frequency = count});
      }
      detail::CompensatedSum weightedNormalized;
      detail::CompensatedSum squareNormalized;
      Scalar minimum = ordered.front().value;
      Scalar maximum = minimum;
      for (const auto& entry : ordered) {
        minimum = std::min(minimum, entry.value);
        maximum = std::max(maximum, entry.value);
        const Scalar normalized =
            valueScale == 0.0 ? 0.0 : entry.value / valueScale;
        weightedNormalized.add(entry.normalized_weight * normalized);
        squareNormalized.add(entry.normalized_weight * normalized * normalized);
      }
      const Scalar meanNormalized = weightedNormalized.total() / mass;
      const Scalar rmsNormalized = std::sqrt(squareNormalized.total() / mass);
      const auto meanValue = finiteProduct({valueScale, meanNormalized},
                                           "summary mean is not representable");
      const auto rmsValue = finiteProduct({valueScale, rmsNormalized},
                                          "summary RMS is not representable");
      const auto sumValue =
          finiteProduct({maximumWeight, valueScale, weightedNormalized.total()},
                        "summary weighted sum is not representable");
      const auto rangeValue = [&]() -> Result<Scalar> {
        if (minimum < 0.0 &&
            maximum > std::numeric_limits<Scalar>::max() + minimum) {
          return fail<Scalar>(ErrorCode::NonFiniteResult,
                              "summary range is not representable");
        }
        return Result<Scalar>::success(maximum - minimum);
      }();
      if (!meanValue) {
        return fail<Summary>(meanValue.status().code(),
                             "descriptive summary result is not representable");
      }
      if (!rmsValue) {
        return fail<Summary>(rmsValue.status().code(),
                             "descriptive summary result is not representable");
      }
      if (!sumValue) {
        return fail<Summary>(sumValue.status().code(),
                             "descriptive summary result is not representable");
      }
      if (!rangeValue) {
        return fail<Summary>(rangeValue.status().code(),
                             "descriptive summary result is not representable");
      }
      means.at(coordinate) = meanValue.value();
      rmsValues.at(coordinate) = rmsValue.value();
      sums.at(coordinate) = sumValue.value();
      minimums.at(coordinate) = minimum;
      maximums.at(coordinate) = maximum;
      ranges.at(coordinate) = rangeValue.value();

      detail::CompensatedSum centered;
      for (const auto& entry : ordered) {
        const Scalar normalized =
            valueScale == 0.0 ? 0.0
                              : (entry.value / valueScale) - meanNormalized;
        centered.add(entry.normalized_weight * normalized * normalized);
      }
      std::ranges::sort(
          ordered, [](const WeightedValue& left, const WeightedValue& right) {
            return left.value < right.value;
          });
      Scalar varianceDenominator = mass;
      ErrorCode unavailable = ErrorCode::None;
      if (spec.variance_mode == VarianceMode::iid_sample) {
        if (weights.rule == WeightRule::frequency) {
          if (frequencyMass < 2U) {
            unavailable = ErrorCode::InsufficientSamples;
          } else {
            varianceDenominator =
                static_cast<Scalar>(frequencyMass - 1U) / maximumWeight;
          }
        } else if (!(1.0 - normalizedSquarePowerForCorrection >
                     64.0 * std::numeric_limits<Scalar>::epsilon())) {
          unavailable = ErrorCode::InsufficientSamples;
        } else {
          varianceDenominator =
              mass * (1.0 - normalizedSquarePowerForCorrection);
        }
      }
      varianceStatus[coordinate] = unavailable;
      if (unavailable == ErrorCode::None) {
        const Scalar varianceNormalized =
            centered.total() / varianceDenominator;
        const auto varianceValue =
            finiteProduct({valueScale, valueScale, varianceNormalized},
                          "summary variance is not representable");
        const auto deviationValue =
            finiteProduct({valueScale, std::sqrt(varianceNormalized)},
                          "summary standard deviation is not representable");
        if (!varianceValue || !deviationValue) {
          return fail<Summary>(ErrorCode::NonFiniteResult,
                               "summary variance or standard deviation is not "
                               "representable");
        }
        variances.at(coordinate) = varianceValue.value();
        deviations.at(coordinate) = deviationValue.value();
      }
      for (Index q = 0U; q < spec.quantile_probabilities.size(); ++q) {
        auto quantile = weightedQuantile(
            ordered, spec.quantile_probabilities[q], weights.rule, mass);
        if (!quantile) {
          return fail<Summary>(quantile.status().code(),
                               "summary quantile could not be computed");
        }
        quantiles.at(coordinate, q) = quantile.value();
      }
      coordinateSupport.push_back(global);
    }
    auto statuses = Array<ErrorCode>::copy(varianceStatus, limits);
    if (!statuses) {
      return fail<Summary>(statuses.status().code(),
                           "variance status exceeds configured limits");
    }
    auto supports = Array<Support>::copy(coordinateSupport, limits);
    if (!supports) {
      return fail<Summary>(supports.status().code(),
                           "coordinate support exceeds configured limits");
    }
    EstimatorDiagnostics diagnostics{
        .estimand = Estimand::observed,
        .weight = weights.rule,
        .center = CenterRule::estimated,
        .approximation = Approximation::exact,
        .global = global,
        .coordinate_support = std::move(supports).value(),
        .tuple_support = {},
        .reciprocal_condition = std::nullopt,
        .numerical_error = std::nullopt,
        .sampling_error = std::nullopt,
        .rank = std::nullopt,
        .algorithm = "weighted_descriptive_summary",
        .premises = weights.rule == WeightRule::importance
                        ? "normalized importance-weighted empirical measure; "
                          "no target-distribution unbiasedness claimed"
                        : "complete finite common-support weighted empirical "
                          "summary",
        .warnings = {}};
    return Result<Summary>::success(
        Summary{.sum = std::move(sums),
                .minimum = std::move(minimums),
                .maximum = std::move(maximums),
                .rms = std::move(rmsValues),
                .mean = std::move(means),
                .variance = std::move(variances),
                .quantiles = std::move(quantiles),
                .diagnostics = std::move(diagnostics),
                .range = std::move(ranges),
                .standard_deviation = std::move(deviations),
                .variance_status = std::move(statuses).value()});
  } catch (const std::bad_alloc&) {
    return fail<Summary>(ErrorCode::ResourceLimit,
                         "summary could not allocate bounded storage");
  }
}

} // namespace vss::math::statistics
