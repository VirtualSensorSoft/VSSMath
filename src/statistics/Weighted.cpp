#include <algorithm>
#include <cmath>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

Result<Extent> validateSample(const SampleView& sample) {
  const Extent rows = sample.rows.rows();
  const Extent columns = sample.rows.columns();
  if (rows == 0U || columns == 0U) {
    return fail<Extent>(
        ErrorCode::InvalidExtent,
        "weighted population must have nonempty rows and coordinates");
  }
  if (sample.observation_shape.empty()) {
    if (columns != 1U) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "scalar population must have exactly one coordinate");
    }
  } else {
    Extent product = 1U;
    for (const Extent extent : sample.observation_shape) {
      if (extent == 0U) {
        return fail<Extent>(ErrorCode::InvalidExtent,
                            "observation shape extents must be positive");
      }
      if (product > std::numeric_limits<Extent>::max() / extent) {
        return fail<Extent>(ErrorCode::SizeOverflow,
                            "observation shape product overflows");
      }
      product *= extent;
    }
    if (product != columns) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "observation shape does not match sample columns");
    }
  }
  if (rows > std::numeric_limits<Extent>::max() / columns) {
    return fail<Extent>(ErrorCode::SizeOverflow, "sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != rows * columns) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "validity mask must contain one byte per sample cell");
  }
  for (const std::uint8_t byte : sample.valid) {
    if (byte > 1U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "validity mask bytes must be zero or one");
    }
  }
  return Result<Extent>::success(columns);
}

bool validCell(const SampleView& sample, Index row, Index column) {
  return sample.valid.empty() ||
         sample.valid[(row * sample.rows.columns()) + column] != 0U;
}

bool completeRow(const SampleView& sample, Index row) {
  for (Index column = 0U; column < sample.rows.columns(); ++column) {
    if (!validCell(sample, row, column)) {
      return false;
    }
  }
  return true;
}

Status validateWeightSpec(const WeightSpec& weights, const SampleView& sample,
                          bool covariance) {
  if (weights.rule == WeightRule::unweighted &&
      !weights.rowWeights.has_value() &&
      !weights.componentWeights.has_value() &&
      weights.frequencyCounts.empty()) {
    return Status::success();
  }
  if (weights.rule == WeightRule::empirical && weights.rowWeights.has_value() &&
      !weights.componentWeights.has_value() &&
      weights.frequencyCounts.empty()) {
    if (weights.rowWeights->size() != sample.rows.rows()) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "row weights must match the observation count");
    }
    return Status::success();
  }
  if (!covariance && weights.rule == WeightRule::componentwise &&
      !weights.rowWeights.has_value() && weights.componentWeights.has_value() &&
      weights.frequencyCounts.empty()) {
    if (weights.componentWeights->rows() != sample.rows.rows() ||
        weights.componentWeights->columns() != sample.rows.columns()) {
      return Status::failure(
          ErrorCode::DimensionMismatch,
          "component weights must match sample rows and coordinates");
    }
    return Status::success();
  }
  return Status::failure(
      ErrorCode::InvalidDomain,
      covariance
          ? "empirical covariance requires unweighted or empirical row weights"
          : "weighted mean requires empirical row weights or componentwise "
            "weights");
}

Scalar rowWeightAt(const WeightSpec& weights, Index row) {
  if (weights.rule == WeightRule::unweighted) {
    (void)row;
    return 1.0;
  }
  if (weights.rowWeights.has_value()) {
    return weights.rowWeights.value().at(row);
  }
  throw std::logic_error("validated row weights are unexpectedly absent");
}

Scalar componentWeightAt(const WeightSpec& weights, Index row, Index column) {
  if (weights.componentWeights.has_value()) {
    return weights.componentWeights.value().at(row, column);
  }
  throw std::logic_error("validated component weights are unexpectedly absent");
}

Scalar weightAt(const WeightSpec& weights, Index row, Index column) {
  if (weights.rule == WeightRule::unweighted ||
      weights.rule == WeightRule::empirical ||
      weights.rule == WeightRule::fixed_iid) {
    return rowWeightAt(weights, row);
  }
  return componentWeightAt(weights, row, column);
}

Status validateWeightValues(const SampleView& sample,
                            const WeightSpec& weights) {
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    for (Index column = 0U; column < sample.rows.columns(); ++column) {
      if (weights.rule == WeightRule::componentwise &&
          !validCell(sample, row, column)) {
        continue;
      }
      const Scalar weight = weightAt(weights, row, column);
      if (!std::isfinite(weight)) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "weight input contains a non-finite value");
      }
      if (weight < 0.0) {
        return Status::failure(ErrorCode::InvalidDomain,
                               "weights must be nonnegative");
      }
      if (weights.rule == WeightRule::empirical ||
          weights.rule == WeightRule::fixed_iid) {
        break;
      }
    }
  }
  return Status::success();
}

bool withinLimits(Extent outputElements, Extent workspaceElements,
                  Extent outputBytes, Extent workspaceBytes, Limits limits) {
  return outputElements <= limits.elements &&
         workspaceElements <= limits.elements - outputElements &&
         outputBytes <= limits.bytes &&
         workspaceBytes <= limits.bytes - outputBytes;
}

Result<Array<Support>> makeSupports(const std::vector<Support>& supports,
                                    Limits limits) {
  return Array<Support>::copy(supports, limits);
}

Support supportFor(Index input, Index valid, Index positive, Scalar maximum,
                   Scalar scaledMass, Scalar power2, Scalar power3) {
  std::optional<Scalar> mass;
  const Scalar actualMass = maximum * scaledMass;
  if (std::isfinite(actualMass)) {
    mass = actualMass;
  }
  const Scalar effective = power2 > 0.0 ? 1.0 / power2 : 0.0;
  return Support{.input = input,
                 .valid = valid,
                 .positive = positive,
                 .frequency_mass = std::nullopt,
                 .weight_mass = mass,
                 .normalized_weight_powers = {1.0, power2, power3},
                 .effective_count = effective,
                 .degrees_of_freedom = std::nullopt,
                 .squared_weight_mass = std::nullopt};
}

EstimatorDiagnostics diagnosticsFor(WeightRule weightRule, Support support,
                                    Array<Support> coordinateSupport,
                                    const char* algorithm,
                                    const char* premises) {
  return EstimatorDiagnostics{.estimand = Estimand::observed,
                              .weight = weightRule,
                              .center = CenterRule::estimated,
                              .approximation = Approximation::exact,
                              .global = support,
                              .coordinate_support =
                                  std::move(coordinateSupport),
                              .tuple_support = {},
                              .reciprocal_condition = std::nullopt,
                              .numerical_error = std::nullopt,
                              .sampling_error = std::nullopt,
                              .rank = std::nullopt,
                              .algorithm = algorithm,
                              .premises = premises,
                              .warnings = {}};
}

Scalar scaledProduct(Scalar left, Scalar right, Scalar factor) {
  if (left == 0.0 || right == 0.0 || factor == 0.0) {
    return 0.0;
  }
  int leftExponent = 0;
  int rightExponent = 0;
  int factorExponent = 0;
  const Scalar leftMantissa = std::frexp(left, &leftExponent);
  const Scalar rightMantissa = std::frexp(right, &rightExponent);
  const Scalar factorMantissa = std::frexp(factor, &factorExponent);
  const Scalar mantissa = leftMantissa * rightMantissa * factorMantissa;
  return std::scalbn(mantissa, leftExponent + rightExponent + factorExponent);
}

} // namespace

Result<Estimate<Vector>> weightedMean(const SampleView& sample,
                                      const WeightSpec& weights,
                                      Limits limits) {
  const auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<Estimate<Vector>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  const auto weightStatus = validateWeightSpec(weights, sample, false);
  if (!weightStatus.ok()) {
    return fail<Estimate<Vector>>(weightStatus.code(),
                                  std::string(weightStatus.message()));
  }
  const auto valuesStatus = validateWeightValues(sample, weights);
  if (!valuesStatus.ok()) {
    return fail<Estimate<Vector>>(valuesStatus.code(),
                                  std::string(valuesStatus.message()));
  }

  const Extent columns = dimension.value();
  constexpr Extent scalarWorkspaceCount = 3U;
  constexpr Extent totalScalarCount = 4U;
  if (columns > std::numeric_limits<Extent>::max() / totalScalarCount) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "weighted mean workspace extent overflows");
  }
  const Extent workspaceElements = columns * scalarWorkspaceCount;
  if (columns > std::numeric_limits<Extent>::max() / sizeof(Scalar) ||
      workspaceElements > std::numeric_limits<Extent>::max() / sizeof(Scalar) ||
      columns > std::numeric_limits<Extent>::max() / sizeof(Support)) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "weighted mean workspace bytes overflow");
  }
  if (columns > std::numeric_limits<Extent>::max() / (2U * sizeof(Support))) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "weighted mean support bytes overflow");
  }
  const Extent supportBytes = 2U * columns * sizeof(Support);
  const Extent scalarBytes = workspaceElements * sizeof(Scalar);
  if (supportBytes > std::numeric_limits<Extent>::max() - scalarBytes ||
      !withinLimits(columns, workspaceElements, columns * sizeof(Scalar),
                    scalarBytes + supportBytes, limits)) {
    return fail<Estimate<Vector>>(ErrorCode::ResourceLimit,
                                  "weighted mean exceeds configured limits");
  }

  Vector mean(columns);
  std::vector<Support> coordinateSupports;
  coordinateSupports.reserve(columns);

  for (Index column = 0U; column < columns; ++column) {
    Scalar maximumWeight = 0.0;
    Index validCount = 0U;
    Index positiveCount = 0U;
    Scalar valueScale = 0.0;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (!validCell(sample, row, column)) {
        continue;
      }
      const Scalar value = sample.rows.at(row, column);
      if (!std::isfinite(value)) {
        return fail<Estimate<Vector>>(
            ErrorCode::NonFiniteInput,
            "a valid sample coordinate is non-finite");
      }
      const Scalar weight = weightAt(weights, row, column);
      ++validCount;
      if (weight > 0.0) {
        ++positiveCount;
        maximumWeight = std::max(maximumWeight, weight);
        valueScale = std::max(valueScale, std::abs(value));
      }
    }
    if (positiveCount == 0U) {
      return fail<Estimate<Vector>>(
          ErrorCode::InsufficientSamples,
          "each mean coordinate requires positive weight");
    }

    detail::CompensatedSum mass;
    detail::CompensatedSum power2;
    detail::CompensatedSum power3;
    detail::CompensatedSum weightedValue;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (!validCell(sample, row, column)) {
        continue;
      }
      const Scalar weight = weightAt(weights, row, column);
      if (weight == 0.0) {
        continue;
      }
      const Scalar scaledWeight = weight / maximumWeight;
      mass.add(scaledWeight);
      power2.add(scaledWeight * scaledWeight);
      power3.add(scaledWeight * scaledWeight * scaledWeight);
      const Scalar scaledValue =
          valueScale == 0.0 ? 0.0 : sample.rows.at(row, column) / valueScale;
      weightedValue.add(scaledWeight * scaledValue);
    }
    const Scalar scaledMass = mass.total();
    if (!(scaledMass > 0.0) || !std::isfinite(scaledMass)) {
      return fail<Estimate<Vector>>(ErrorCode::NonFiniteResult,
                                    "scaled weight mass is not representable");
    }
    const Scalar normalizedMean = weightedValue.total() / scaledMass;
    const Scalar estimate = valueScale * normalizedMean;
    if (!std::isfinite(estimate)) {
      return fail<Estimate<Vector>>(ErrorCode::NonFiniteResult,
                                    "weighted mean is non-finite");
    }
    mean.at(column) = estimate;
    coordinateSupports.push_back(
        supportFor(sample.rows.rows(), validCount, positiveCount, maximumWeight,
                   scaledMass, power2.total() / (scaledMass * scaledMass),
                   power3.total() / (scaledMass * scaledMass * scaledMass)));
  }

  auto supports = makeSupports(coordinateSupports, limits);
  if (!supports) {
    return fail<Estimate<Vector>>(supports.status().code(),
                                  std::string(supports.status().message()));
  }
  Index completeValid = 0U;
  Index completePositive = 0U;
  Index componentValidRows = 0U;
  Index componentPositiveRows = 0U;
  Scalar globalMaximum = 0.0;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    bool hasValidComponent = false;
    bool hasPositiveComponent = false;
    if (weights.rule == WeightRule::componentwise) {
      for (Index column = 0U; column < columns; ++column) {
        if (validCell(sample, row, column)) {
          hasValidComponent = true;
          hasPositiveComponent = hasPositiveComponent ||
                                 componentWeightAt(weights, row, column) > 0.0;
        }
      }
      componentValidRows += hasValidComponent ? 1U : 0U;
      componentPositiveRows += hasPositiveComponent ? 1U : 0U;
    }
    if (!completeRow(sample, row)) {
      continue;
    }
    ++completeValid;
    if (weights.rule == WeightRule::empirical ||
        weights.rule == WeightRule::unweighted) {
      const Scalar weight = rowWeightAt(weights, row);
      if (weight > 0.0) {
        ++completePositive;
        globalMaximum = std::max(globalMaximum, weight);
      }
    }
  }
  Support global{.input = sample.rows.rows(),
                 .valid = weights.rule == WeightRule::componentwise
                              ? componentValidRows
                              : completeValid,
                 .positive = weights.rule == WeightRule::componentwise
                                 ? componentPositiveRows
                                 : completePositive,
                 .frequency_mass = std::nullopt,
                 .weight_mass = std::nullopt,
                 .normalized_weight_powers = {},
                 .effective_count = std::nullopt,
                 .degrees_of_freedom = std::nullopt,
                 .squared_weight_mass = std::nullopt};
  if ((weights.rule == WeightRule::empirical ||
       weights.rule == WeightRule::unweighted) &&
      completePositive > 0U) {
    detail::CompensatedSum mass;
    detail::CompensatedSum power2;
    detail::CompensatedSum power3;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (completeRow(sample, row)) {
        const Scalar scaled = rowWeightAt(weights, row) / globalMaximum;
        mass.add(scaled);
        power2.add(scaled * scaled);
        power3.add(scaled * scaled * scaled);
      }
    }
    const Scalar total = mass.total();
    global = supportFor(sample.rows.rows(), completeValid, completePositive,
                        globalMaximum, total, power2.total() / (total * total),
                        power3.total() / (total * total * total));
  }
  auto diagnostics = diagnosticsFor(
      weights.rule, global, std::move(supports).value(),
      "maximum-scaled compensated weighted coordinate mean",
      "valid coordinates with positive nonnegative empirical weights");
  return Result<Estimate<Vector>>::success(
      Estimate<Vector>{.value = std::move(mean),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::nullopt});
}

Result<Estimate<Vector>> precisionWeightedMean(const SampleView& sample,
                                               ConstMatrixView variances,
                                               Limits limits) {
  const auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<Estimate<Vector>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  const Extent columns = dimension.value();
  const Extent rows = sample.rows.rows();
  if (columns == 0U) {
    return fail<Estimate<Vector>>(ErrorCode::InvalidExtent,
                                  "precision mean requires coordinates");
  }
  if (variances.rows() != rows || variances.columns() != columns) {
    return fail<Estimate<Vector>>(
        ErrorCode::DimensionMismatch,
        "known variances must match sample rows and coordinates");
  }
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      const Scalar variance = variances.at(row, column);
      if (!std::isfinite(variance)) {
        return fail<Estimate<Vector>>(
            ErrorCode::NonFiniteInput,
            "known variance input contains a non-finite value");
      }
      if (!(variance > 0.0)) {
        return fail<Estimate<Vector>>(
            ErrorCode::InvalidDomain,
            "known variances must be positive, including masked cells");
      }
    }
  }

  const Extent maximum = std::numeric_limits<Extent>::max();
  if (columns > maximum / columns) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "precision mean output extent overflows");
  }
  const Extent covarianceElements = columns * columns;
  if (covarianceElements > maximum - columns ||
      columns > maximum / sizeof(Support)) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "precision mean output extent overflows");
  }
  const Extent outputElements = covarianceElements + columns;
  const Extent supportBytes = columns * sizeof(Support);
  if (outputElements > maximum / sizeof(Scalar) ||
      supportBytes > maximum / 2U) {
    return fail<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                  "precision mean storage size overflows");
  }
  const Extent scalarBytes = outputElements * sizeof(Scalar);
  const Extent outputBytes = scalarBytes + supportBytes;
  const Extent workspaceBytes = supportBytes;
  if (outputBytes < scalarBytes ||
      !withinLimits(outputElements, 0U, outputBytes, workspaceBytes, limits)) {
    return fail<Estimate<Vector>>(ErrorCode::ResourceLimit,
                                  "precision mean exceeds configured limits");
  }

  Vector mean(columns);
  Matrix errorCovariance(columns, columns);
  std::vector<Support> coordinateSupports;
  coordinateSupports.reserve(columns);
  Index rowsWithAnyCell = 0U;
  for (Index row = 0U; row < rows; ++row) {
    bool admitted = false;
    for (Index column = 0U; column < columns; ++column) {
      admitted = admitted || validCell(sample, row, column);
    }
    rowsWithAnyCell += admitted ? 1U : 0U;
  }

  for (Index column = 0U; column < columns; ++column) {
    Scalar minimumVariance = std::numeric_limits<Scalar>::infinity();
    Scalar valueScale = 0.0;
    Index validCount = 0U;
    for (Index row = 0U; row < rows; ++row) {
      if (!validCell(sample, row, column)) {
        continue;
      }
      const Scalar value = sample.rows.at(row, column);
      if (!std::isfinite(value)) {
        return fail<Estimate<Vector>>(
            ErrorCode::NonFiniteInput,
            "a valid precision-weighted sample coordinate is non-finite");
      }
      minimumVariance = std::min(minimumVariance, variances.at(row, column));
      valueScale = std::max(valueScale, std::abs(value));
      ++validCount;
    }
    if (validCount == 0U) {
      return fail<Estimate<Vector>>(
          ErrorCode::InsufficientSamples,
          "each precision mean coordinate requires valid support");
    }

    detail::CompensatedSum precisionMass;
    detail::CompensatedSum weightedValue;
    for (Index row = 0U; row < rows; ++row) {
      if (!validCell(sample, row, column)) {
        continue;
      }
      const Scalar relativePrecision =
          minimumVariance / variances.at(row, column);
      if (!(relativePrecision > 0.0) || !std::isfinite(relativePrecision)) {
        return fail<Estimate<Vector>>(
            ErrorCode::NonFiniteResult,
            "a positive relative precision is not representable");
      }
      precisionMass.add(relativePrecision);
      const Scalar scaledValue =
          valueScale == 0.0 ? 0.0 : sample.rows.at(row, column) / valueScale;
      weightedValue.add(relativePrecision * scaledValue);
    }
    const Scalar scaledMass = precisionMass.total();
    if (!(scaledMass > 0.0) || !std::isfinite(scaledMass)) {
      return fail<Estimate<Vector>>(
          ErrorCode::NonFiniteResult,
          "normalized precision mass is not representable");
    }
    const Scalar normalizedMean =
        std::clamp(weightedValue.total() / scaledMass, -1.0, 1.0);
    const Scalar coordinateMean = valueScale * normalizedMean;
    if (!std::isfinite(coordinateMean)) {
      return fail<Estimate<Vector>>(
          ErrorCode::NonFiniteResult,
          "precision-weighted mean is outside the finite range");
    }
    const Scalar varianceOfMean = minimumVariance / scaledMass;
    if (!(varianceOfMean > 0.0) || !std::isfinite(varianceOfMean)) {
      return fail<Estimate<Vector>>(ErrorCode::NonFiniteResult,
                                    "precision-weighted mean variance is "
                                    "outside the representable range");
    }
    mean.at(column) = coordinateMean;
    errorCovariance.at(column, column) = varianceOfMean;
    coordinateSupports.push_back(Support{.input = rows,
                                         .valid = validCount,
                                         .positive = validCount,
                                         .frequency_mass = std::nullopt,
                                         .weight_mass = std::nullopt,
                                         .normalized_weight_powers = {},
                                         .effective_count = std::nullopt,
                                         .degrees_of_freedom = std::nullopt,
                                         .squared_weight_mass = std::nullopt});
  }

  auto supports = makeSupports(coordinateSupports, limits);
  if (!supports) {
    return fail<Estimate<Vector>>(supports.status().code(),
                                  std::string(supports.status().message()));
  }
  const Support global{.input = rows,
                       .valid = rowsWithAnyCell,
                       .positive = rowsWithAnyCell,
                       .frequency_mass = std::nullopt,
                       .weight_mass = std::nullopt,
                       .normalized_weight_powers = {},
                       .effective_count = std::nullopt,
                       .degrees_of_freedom = std::nullopt,
                       .squared_weight_mass = std::nullopt};
  auto diagnostics = diagnosticsFor(
      WeightRule::precision, global, std::move(supports).value(),
      "minimum-scaled compensated inverse-variance weighted coordinate mean",
      "componentwise common mean with mutually independent zero-mean errors "
      "and supplied positive known variances; no residual degrees of freedom");
  diagnostics.estimand = Estimand::estimate;
  return Result<Estimate<Vector>>::success(
      Estimate<Vector>{.value = std::move(mean),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::move(errorCovariance)});
}

Result<Estimate<Matrix>> weightedEmpiricalCovariance(const SampleView& sample,
                                                     const WeightSpec& weights,
                                                     Limits limits) {
  const auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<Estimate<Matrix>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  const auto weightStatus = validateWeightSpec(weights, sample, true);
  if (!weightStatus.ok()) {
    return fail<Estimate<Matrix>>(weightStatus.code(),
                                  std::string(weightStatus.message()));
  }
  const auto valuesStatus = validateWeightValues(sample, weights);
  if (!valuesStatus.ok()) {
    return fail<Estimate<Matrix>>(valuesStatus.code(),
                                  std::string(valuesStatus.message()));
  }

  const Extent columns = dimension.value();
  if (columns > std::numeric_limits<Extent>::max() / columns) {
    return fail<Estimate<Matrix>>(
        ErrorCode::SizeOverflow, "weighted covariance output extent overflows");
  }
  const Extent outputElements = columns * columns;
  constexpr Extent vectorWorkspaceCount = 3U;
  if (columns > std::numeric_limits<Extent>::max() / vectorWorkspaceCount ||
      outputElements > std::numeric_limits<Extent>::max() -
                           (columns * vectorWorkspaceCount)) {
    return fail<Estimate<Matrix>>(ErrorCode::SizeOverflow,
                                  "weighted covariance workspace overflows");
  }
  const Extent vectorElements = columns * vectorWorkspaceCount;
  if (outputElements > std::numeric_limits<Extent>::max() - vectorElements) {
    return fail<Estimate<Matrix>>(ErrorCode::SizeOverflow,
                                  "weighted covariance workspace overflows");
  }
  const Extent workspaceElements = outputElements + vectorElements;
  if (outputElements >
          std::numeric_limits<Extent>::max() / sizeof(detail::CompensatedSum) ||
      vectorElements > std::numeric_limits<Extent>::max() / sizeof(Scalar) ||
      columns > std::numeric_limits<Extent>::max() / sizeof(Support)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::SizeOverflow,
        "weighted covariance workspace bytes overflow");
  }
  const Extent outputBytes = outputElements * sizeof(Scalar);
  const Extent supportBytes = columns * sizeof(Support);
  const Extent accumulatorBytes =
      outputElements * sizeof(detail::CompensatedSum);
  const Extent vectorWorkspaceBytes = vectorElements * sizeof(Scalar);
  if (accumulatorBytes >
          std::numeric_limits<Extent>::max() - vectorWorkspaceBytes ||
      accumulatorBytes + vectorWorkspaceBytes >
          std::numeric_limits<Extent>::max() - supportBytes) {
    return fail<Estimate<Matrix>>(
        ErrorCode::SizeOverflow,
        "weighted covariance workspace bytes overflow");
  }
  const Extent workspaceBytes =
      accumulatorBytes + vectorWorkspaceBytes + supportBytes;
  if (workspaceBytes > std::numeric_limits<Extent>::max() - outputBytes ||
      !withinLimits(outputElements, workspaceElements, outputBytes,
                    workspaceBytes, limits)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::ResourceLimit,
        "weighted covariance exceeds configured limits");
  }

  Index validCount = 0U;
  Index positiveCount = 0U;
  Scalar maximumWeight = 0.0;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (!completeRow(sample, row)) {
      continue;
    }
    ++validCount;
    for (Index column = 0U; column < columns; ++column) {
      if (!std::isfinite(sample.rows.at(row, column))) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteInput,
            "a complete sample row contains a non-finite coordinate");
      }
    }
    const Scalar weight = rowWeightAt(weights, row);
    if (weight > 0.0) {
      ++positiveCount;
      maximumWeight = std::max(maximumWeight, weight);
    }
  }
  if (positiveCount == 0U) {
    return fail<Estimate<Matrix>>(ErrorCode::InsufficientSamples,
                                  "covariance requires positive row weight");
  }

  detail::CompensatedSum mass;
  detail::CompensatedSum power2;
  detail::CompensatedSum power3;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (!completeRow(sample, row)) {
      continue;
    }
    const Scalar weight = rowWeightAt(weights, row);
    if (weight == 0.0) {
      continue;
    }
    const Scalar scaled = weight / maximumWeight;
    mass.add(scaled);
    power2.add(scaled * scaled);
    power3.add(scaled * scaled * scaled);
  }
  const Scalar scaledMass = mass.total();
  if (!(scaledMass > 0.0) || !std::isfinite(scaledMass)) {
    return fail<Estimate<Matrix>>(ErrorCode::NonFiniteResult,
                                  "scaled weight mass is not representable");
  }

  Vector scales(columns, 0.0);
  Vector normalizedCenter(columns, 0.0);
  for (Index column = 0U; column < columns; ++column) {
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (completeRow(sample, row) && rowWeightAt(weights, row) > 0.0) {
        scales.at(column) =
            std::max(scales.at(column), std::abs(sample.rows.at(row, column)));
      }
    }
    if (scales.at(column) == 0.0) {
      continue;
    }
    detail::CompensatedSum weightedCoordinate;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (!completeRow(sample, row)) {
        continue;
      }
      const Scalar weight = rowWeightAt(weights, row);
      if (weight > 0.0) {
        weightedCoordinate.add(
            (weight / maximumWeight) *
            (sample.rows.at(row, column) / scales.at(column)));
      }
    }
    normalizedCenter.at(column) = weightedCoordinate.total() / scaledMass;
  }

  std::vector<detail::CompensatedSum> products(outputElements);
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (!completeRow(sample, row)) {
      continue;
    }
    const Scalar weight = rowWeightAt(weights, row);
    if (weight == 0.0) {
      continue;
    }
    for (Index left = 0U; left < columns; ++left) {
      const Scalar leftDelta =
          scales.at(left) == 0.0
              ? 0.0
              : (sample.rows.at(row, left) / scales.at(left)) -
                    normalizedCenter.at(left);
      for (Index right = left; right < columns; ++right) {
        const Scalar rightDelta =
            scales.at(right) == 0.0
                ? 0.0
                : (sample.rows.at(row, right) / scales.at(right)) -
                      normalizedCenter.at(right);
        products[(left * columns) + right].add((weight / maximumWeight) *
                                               leftDelta * rightDelta);
      }
    }
  }

  Matrix covariance(columns, columns, 0.0);
  for (Index left = 0U; left < columns; ++left) {
    for (Index right = left; right < columns; ++right) {
      const Scalar normalized =
          products[(left * columns) + right].total() / scaledMass;
      const Scalar value =
          scaledProduct(scales.at(left), scales.at(right), normalized);
      if (!std::isfinite(value)) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "weighted covariance is outside the finite scalar range");
      }
      covariance.at(left, right) = value;
      covariance.at(right, left) = value;
    }
  }

  const Support global =
      supportFor(sample.rows.rows(), validCount, positiveCount, maximumWeight,
                 scaledMass, power2.total() / (scaledMass * scaledMass),
                 power3.total() / (scaledMass * scaledMass * scaledMass));
  auto supports = makeSupports(std::vector<Support>{}, limits);
  if (!supports) {
    return fail<Estimate<Matrix>>(supports.status().code(),
                                  std::string(supports.status().message()));
  }
  auto diagnostics = diagnosticsFor(
      weights.rule, global, std::move(supports).value(),
      "maximum-scaled compensated weighted centered outer products",
      "complete rows with nonnegative empirical weights; descriptive empirical "
      "covariance");
  return Result<Estimate<Matrix>>::success(
      Estimate<Matrix>{.value = std::move(covariance),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::nullopt});
}

Result<Estimate<Matrix>>
unbiasedFixedWeightedCovariance(const SampleView& sample,
                                const WeightSpec& fixedWeights, Limits limits) {
  const auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<Estimate<Matrix>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  if (fixedWeights.rule != WeightRule::fixed_iid ||
      !fixedWeights.rowWeights.has_value() ||
      fixedWeights.componentWeights.has_value() ||
      !fixedWeights.frequencyCounts.empty()) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InvalidDomain,
        "unbiased covariance requires fixed-IID row weights only");
  }
  if (fixedWeights.rowWeights->size() != sample.rows.rows()) {
    return fail<Estimate<Matrix>>(
        ErrorCode::DimensionMismatch,
        "fixed-IID row weights must match the observation count");
  }

  const WeightSpec empirical{.rule = WeightRule::empirical,
                             .rowWeights = fixedWeights.rowWeights,
                             .componentWeights = std::nullopt,
                             .frequencyCounts = {}};
  const auto valuesStatus = validateWeightValues(sample, fixedWeights);
  if (!valuesStatus.ok()) {
    return fail<Estimate<Matrix>>(valuesStatus.code(),
                                  std::string(valuesStatus.message()));
  }

  Index positiveCount = 0U;
  Index firstPositiveRow = 0U;
  Index secondPositiveRow = 0U;
  Scalar maximumWeight = 0.0;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (!completeRow(sample, row)) {
      continue;
    }
    const Scalar weight = rowWeightAt(fixedWeights, row);
    if (weight > 0.0) {
      if (positiveCount == 0U) {
        firstPositiveRow = row;
      } else if (positiveCount == 1U) {
        secondPositiveRow = row;
      }
      ++positiveCount;
      maximumWeight = std::max(maximumWeight, weight);
    }
  }
  if (positiveCount < 2U) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InsufficientSamples,
        "fixed-IID covariance requires at least two positive complete rows");
  }

  detail::CompensatedSum mass;
  detail::CompensatedSum pairProducts;
  detail::CompensatedSum prefix;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (!completeRow(sample, row)) {
      continue;
    }
    const Scalar scaled = rowWeightAt(fixedWeights, row) / maximumWeight;
    if (scaled == 0.0) {
      continue;
    }
    pairProducts.add(scaled * prefix.total());
    prefix.add(scaled);
    mass.add(scaled);
  }
  const Scalar scaledMass = mass.total();
  const Scalar correction =
      (2.0 * pairProducts.total() / scaledMass) / scaledMass;
  const bool twoPointIdentity = positiveCount == 2U;
  if ((!std::isfinite(correction) || !(correction > 0.0) || correction > 1.0) &&
      (!twoPointIdentity || !std::isfinite(correction) || correction > 1.0)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InvalidDomain,
        "fixed-IID unbiased covariance correction is singular or invalid");
  }

  auto empiricalEstimate =
      weightedEmpiricalCovariance(sample, empirical, limits);
  if (!empiricalEstimate) {
    return fail<Estimate<Matrix>>(
        empiricalEstimate.status().code(),
        std::string(empiricalEstimate.status().message()));
  }
  Matrix& covariance = empiricalEstimate.value().value;
  for (Index row = 0U; row < covariance.rows(); ++row) {
    for (Index column = 0U; column < covariance.columns(); ++column) {
      if (twoPointIdentity) {
        const Scalar leftDifference = sample.rows.at(firstPositiveRow, row) -
                                      sample.rows.at(secondPositiveRow, row);
        const Scalar rightDifference =
            sample.rows.at(firstPositiveRow, column) -
            sample.rows.at(secondPositiveRow, column);
        covariance.at(row, column) =
            scaledProduct(leftDifference, rightDifference, 0.5);
      } else {
        covariance.at(row, column) /= correction;
      }
      if (!std::isfinite(covariance.at(row, column))) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "unbiased fixed-IID covariance is outside the finite range");
      }
    }
  }
  auto& diagnostics = empiricalEstimate.value().diagnostics;
  diagnostics.weight = WeightRule::fixed_iid;
  diagnostics.algorithm =
      twoPointIdentity
          ? "fixed-IID two-observation half-outer-product identity"
          : "maximum-scaled compensated weighted covariance with fixed-IID "
            "pair-sum correction";
  diagnostics.premises =
      "complete rows with nonnegative weights fixed independently of IID "
      "observations";
  return empiricalEstimate;
}

Result<Estimate<Matrix>> weightedMeanCovariance(const SampleView& sample,
                                                const WeightSpec& fixedWeights,
                                                Limits limits) {
  auto covariance =
      unbiasedFixedWeightedCovariance(sample, fixedWeights, limits);
  if (!covariance) {
    return fail<Estimate<Matrix>>(covariance.status().code(),
                                  std::string(covariance.status().message()));
  }
  const auto squaredWeightMass =
      covariance.value().diagnostics.global.normalized_weight_powers[1];
  if (!squaredWeightMass.has_value() ||
      !std::isfinite(squaredWeightMass.value()) ||
      !(squaredWeightMass.value() > 0.0)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::NonFiniteResult,
        "fixed-IID normalized squared weight mass is unavailable");
  }
  Matrix& estimateCovariance = covariance.value().value;
  for (Index row = 0U; row < estimateCovariance.rows(); ++row) {
    for (Index column = 0U; column < estimateCovariance.columns(); ++column) {
      estimateCovariance.at(row, column) *= squaredWeightMass.value();
      if (!std::isfinite(estimateCovariance.at(row, column))) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "weighted mean covariance is outside the finite range");
      }
    }
  }
  auto& diagnostics = covariance.value().diagnostics;
  diagnostics.estimand = Estimand::estimate;
  diagnostics.algorithm =
      "fixed-IID unbiased population covariance scaled by normalized "
      "squared weight mass";
  diagnostics.premises =
      "complete rows are independent and identically distributed; row weights "
      "are fixed independently of observations";
  return covariance;
}

Result<Estimate<Matrix>>
frequencyWeightedCovariance(const SampleView& sample,
                            std::span<const std::uint64_t> counts,
                            VarianceMode mode, Limits limits) {
  const auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<Estimate<Matrix>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  if (counts.size() != sample.rows.rows()) {
    return fail<Estimate<Matrix>>(
        ErrorCode::DimensionMismatch,
        "frequency counts must match the observation count");
  }
  if (mode == VarianceMode::known_center) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InvalidDomain,
        "frequency covariance does not accept known-center normalization");
  }
  if (mode != VarianceMode::empirical && mode != VarianceMode::iid_sample) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InvalidDomain,
        "frequency covariance received an unknown variance mode");
  }

  const Extent columns = dimension.value();
  const Extent maximum = std::numeric_limits<Extent>::max();
  if (columns > maximum / columns) {
    return fail<Estimate<Matrix>>(ErrorCode::SizeOverflow,
                                  "frequency covariance extent overflows");
  }
  const Extent outputElements = columns * columns;
  if (outputElements > maximum / sizeof(detail::CompensatedSum) ||
      columns > maximum / (2U * sizeof(Scalar))) {
    return fail<Estimate<Matrix>>(
        ErrorCode::SizeOverflow,
        "frequency covariance workspace bytes overflow");
  }
  const Extent outputBytes = outputElements * sizeof(Scalar);
  const Extent accumulatorBytes =
      outputElements * sizeof(detail::CompensatedSum);
  const Extent vectorWorkspaceBytes = 2U * columns * sizeof(Scalar);
  if (accumulatorBytes > maximum - vectorWorkspaceBytes) {
    return fail<Estimate<Matrix>>(
        ErrorCode::SizeOverflow,
        "frequency covariance workspace bytes overflow");
  }
  const Extent workspaceBytes = accumulatorBytes + vectorWorkspaceBytes;
  if (outputElements > maximum - (2U * columns) ||
      workspaceBytes > maximum - outputBytes ||
      !withinLimits(outputElements, outputElements + (2U * columns),
                    outputBytes, workspaceBytes, limits)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::ResourceLimit,
        "frequency covariance exceeds configured limits");
  }

  Index validRows = 0U;
  Index positiveRows = 0U;
  std::uint64_t frequencyMass = 0U;
  std::uint64_t maximumCount = 0U;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (counts[row] == 0U || !completeRow(sample, row)) {
      continue;
    }
    for (Index column = 0U; column < columns; ++column) {
      if (!std::isfinite(sample.rows.at(row, column))) {
        return fail<Estimate<Matrix>>(ErrorCode::NonFiniteInput,
                                      "a retained frequency-weighted row "
                                      "contains a non-finite coordinate");
      }
    }
    if (counts[row] >
        std::numeric_limits<std::uint64_t>::max() - frequencyMass) {
      return fail<Estimate<Matrix>>(
          ErrorCode::SizeOverflow,
          "retained frequency count total overflows uint64_t");
    }
    frequencyMass += counts[row];
    maximumCount = std::max(maximumCount, counts[row]);
    ++validRows;
    ++positiveRows;
  }
  const std::uint64_t minimumMass = mode == VarianceMode::iid_sample ? 2U : 1U;
  if (frequencyMass < minimumMass) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InsufficientSamples,
        mode == VarianceMode::iid_sample
            ? "IID frequency covariance requires at least two replications"
            : "empirical frequency covariance requires positive replication "
              "mass");
  }

  const auto countScale = static_cast<long double>(maximumCount);
  detail::CompensatedSum scaledMassAccumulator;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (counts[row] != 0U && completeRow(sample, row)) {
      const auto scaledCount = static_cast<Scalar>(
          static_cast<long double>(counts[row]) / countScale);
      if (scaledCount == 0.0 || !std::isfinite(scaledCount)) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "positive frequency count cannot be normalized representably");
      }
      scaledMassAccumulator.add(scaledCount);
    }
  }
  const Scalar scaledMass = scaledMassAccumulator.total();
  if (!(scaledMass > 0.0) || !std::isfinite(scaledMass)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::NonFiniteResult,
        "normalized frequency mass is not representable");
  }

  Vector scales(columns, 0.0);
  Vector normalizedCenter(columns, 0.0);
  for (Index column = 0U; column < columns; ++column) {
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (counts[row] != 0U && completeRow(sample, row)) {
        scales.at(column) =
            std::max(scales.at(column), std::abs(sample.rows.at(row, column)));
      }
    }
    if (scales.at(column) == 0.0) {
      continue;
    }
    detail::CompensatedSum coordinate;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (counts[row] == 0U || !completeRow(sample, row)) {
        continue;
      }
      const auto scaledCount = static_cast<Scalar>(
          static_cast<long double>(counts[row]) / countScale);
      coordinate.add(scaledCount *
                     (sample.rows.at(row, column) / scales.at(column)));
    }
    normalizedCenter.at(column) = coordinate.total() / scaledMass;
  }

  std::vector<detail::CompensatedSum> products(outputElements);
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (counts[row] == 0U || !completeRow(sample, row)) {
      continue;
    }
    const auto scaledCount =
        static_cast<Scalar>(static_cast<long double>(counts[row]) / countScale);
    for (Index left = 0U; left < columns; ++left) {
      const Scalar leftDelta =
          scales.at(left) == 0.0 ? 0.0
                                 : sample.rows.at(row, left) / scales.at(left) -
                                       normalizedCenter.at(left);
      for (Index right = left; right < columns; ++right) {
        const Scalar rightDelta =
            scales.at(right) == 0.0
                ? 0.0
                : sample.rows.at(row, right) / scales.at(right) -
                      normalizedCenter.at(right);
        products[(left * columns) + right].add(scaledCount * leftDelta *
                                               rightDelta);
      }
    }
  }

  Scalar denominator = scaledMass;
  if (mode == VarianceMode::iid_sample) {
    const auto total = static_cast<Scalar>(frequencyMass);
    denominator *= (total - 1.0) / total;
  }
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::NonFiniteResult,
        "frequency covariance denominator is not representable");
  }

  Matrix covariance(columns, columns, 0.0);
  for (Index left = 0U; left < columns; ++left) {
    for (Index right = left; right < columns; ++right) {
      const Scalar normalized =
          products[(left * columns) + right].total() / denominator;
      const Scalar value =
          scaledProduct(scales.at(left), scales.at(right), normalized);
      if (!std::isfinite(value)) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "frequency covariance is outside the finite scalar range");
      }
      covariance.at(left, right) = value;
      covariance.at(right, left) = value;
    }
  }

  const long double inverseMass =
      1.0L / static_cast<long double>(frequencyMass);
  const auto inverse = static_cast<Scalar>(inverseMass);
  const Scalar inverseSquared = inverse * inverse;
  const Support global{
      .input = sample.rows.rows(),
      .valid = validRows,
      .positive = positiveRows,
      .frequency_mass = frequencyMass,
      .weight_mass = std::nullopt,
      .normalized_weight_powers = {1.0, inverse, inverseSquared},
      .effective_count = static_cast<Scalar>(frequencyMass),
      .degrees_of_freedom =
          mode == VarianceMode::iid_sample
              ? std::optional<Scalar>{static_cast<Scalar>(frequencyMass - 1U)}
              : std::nullopt,
      .squared_weight_mass = std::nullopt};
  auto supports = makeSupports(std::vector<Support>{}, limits);
  if (!supports) {
    return fail<Estimate<Matrix>>(supports.status().code(),
                                  std::string(supports.status().message()));
  }
  auto diagnostics = diagnosticsFor(
      WeightRule::frequency, global, std::move(supports).value(),
      "checked-count-scaled compensated frequency covariance",
      mode == VarianceMode::iid_sample
          ? "complete rows represent replicated IID observations"
          : "descriptive empirical covariance over integer replication counts");
  return Result<Estimate<Matrix>>::success(
      Estimate<Matrix>{.value = std::move(covariance),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::nullopt});
}

Result<Matrix> linearEstimateCovariance(ConstMatrixView map,
                                        ConstMatrixView jointCovariance,
                                        Limits limits) {
  if (map.columns() != jointCovariance.rows() ||
      jointCovariance.rows() != jointCovariance.columns()) {
    return fail<Matrix>(ErrorCode::DimensionMismatch,
                        "linear map and joint covariance dimensions differ");
  }
  const Extent maximum = std::numeric_limits<Extent>::max();
  const Extent source = jointCovariance.rows();
  const Extent output = map.rows();
  const Extent input = map.columns();
  if ((source != 0U && source > maximum / source) ||
      (output != 0U && output > maximum / output) ||
      (output != 0U && input > maximum / output)) {
    return fail<Matrix>(ErrorCode::SizeOverflow,
                        "linear covariance workspace extent overflows");
  }
  const Extent covarianceElements = source * source;
  const Extent outputElements = output * output;
  const Extent mapElements = output * input;
  if (covarianceElements > maximum / 6U ||
      mapElements > maximum - (6U * covarianceElements)) {
    return fail<Matrix>(ErrorCode::SizeOverflow,
                        "linear covariance workspace extent overflows");
  }
  const Extent workspaceElements = mapElements + (6U * covarianceElements);
  if (outputElements > maximum / sizeof(Scalar) ||
      workspaceElements > maximum / sizeof(Scalar)) {
    return fail<Matrix>(ErrorCode::SizeOverflow,
                        "linear covariance workspace bytes overflow");
  }
  const Extent outputBytes = outputElements * sizeof(Scalar);
  const Extent workspaceBytes = workspaceElements * sizeof(Scalar);
  if (!withinLimits(outputElements, workspaceElements, outputBytes,
                    workspaceBytes, limits)) {
    return fail<Matrix>(ErrorCode::ResourceLimit,
                        "linear covariance exceeds configured limits");
  }
  return covariance::propagateCovariance(map, jointCovariance);
}

} // namespace vss::math::statistics
