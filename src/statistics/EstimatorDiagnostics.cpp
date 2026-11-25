#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/statistics/EstimatorDiagnostics.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> failure(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

bool knownEnum(Estimand value) {
  switch (value) {
  case Estimand::observed:
  case Estimand::measurement_error:
  case Estimand::estimate:
  case Estimand::calibration:
  case Estimand::process:
  case Estimand::fitted:
    return true;
  }
  return false;
}
bool knownEnum(WeightRule value) {
  switch (value) {
  case WeightRule::unweighted:
  case WeightRule::empirical:
  case WeightRule::fixed_iid:
  case WeightRule::frequency:
  case WeightRule::precision:
  case WeightRule::importance:
  case WeightRule::componentwise:
  case WeightRule::dependent:
    return true;
  }
  return false;
}
bool knownEnum(CenterRule value) {
  return value == CenterRule::estimated || value == CenterRule::known;
}
bool knownEnum(Approximation value) {
  switch (value) {
  case Approximation::exact:
  case Approximation::first_order:
  case Approximation::quadratic:
  case Approximation::fourth_amplitude:
  case Approximation::sigma_points:
  case Approximation::monte_carlo:
    return true;
  }
  return false;
}
bool nonfatalWarning(ErrorCode value) {
  return value == ErrorCode::RankDeficient ||
         value == ErrorCode::IllConditioned ||
         value == ErrorCode::NotPositiveSemidefinite;
}

Result<bool> validateSupport(const Support& support, WeightRule rule,
                             CenterRule center, std::string_view algorithm) {
  if (support.valid > support.input || support.positive > support.valid) {
    return Result<bool>::failure(ErrorCode::InvalidDomain,
                                 "support counts are contradictory");
  }
  for (const auto value :
       {support.weight_mass, support.squared_weight_mass,
        support.effective_count, support.degrees_of_freedom}) {
    if (value.has_value() && !std::isfinite(*value)) {
      return Result<bool>::failure(ErrorCode::NonFiniteInput,
                                   "support mass or count is non-finite");
    }
    if (value.has_value() && *value < 0.0) {
      return Result<bool>::failure(ErrorCode::InvalidDomain,
                                   "support mass or count is negative");
    }
  }
  if (support.frequency_mass.has_value() && rule != WeightRule::frequency) {
    return Result<bool>::failure(ErrorCode::InvalidDomain,
                                 "frequency mass requires frequency weights");
  }
  if (rule == WeightRule::frequency) {
    if (!support.frequency_mass.has_value() ||
        *support.frequency_mass < support.positive ||
        support.effective_count != std::optional<Scalar>{static_cast<Scalar>(
                                       *support.frequency_mass)}) {
      return Result<bool>::failure(
          ErrorCode::InvalidDomain,
          "frequency support must retain exact mass and matching count");
    }
  } else if (support.frequency_mass.has_value()) {
    return Result<bool>::failure(ErrorCode::InvalidDomain,
                                 "non-frequency support has frequency mass");
  }
  if (rule == WeightRule::precision || rule == WeightRule::importance ||
      rule == WeightRule::dependent) {
    if (support.effective_count.has_value() ||
        support.degrees_of_freedom.has_value()) {
      return Result<bool>::failure(ErrorCode::InvalidDomain,
                                   "this weight regime has no generic "
                                   "effective count or degrees of freedom");
    }
  }
  std::optional<Scalar> previous;
  bool powerGap = false;
  for (const auto& power : support.normalized_weight_powers) {
    if (!power.has_value()) {
      powerGap = true;
      continue;
    }
    if (powerGap) {
      return Result<bool>::failure(
          ErrorCode::InvalidDomain,
          "normalized weight powers must be a contiguous prefix");
    }
    if (!std::isfinite(*power)) {
      return Result<bool>::failure(ErrorCode::NonFiniteInput,
                                   "normalized weight power is non-finite");
    }
    if (*power < 0.0 || *power > 1.0 ||
        (previous.has_value() && *power > *previous)) {
      return Result<bool>::failure(ErrorCode::InvalidDomain,
                                   "normalized weight powers are out of range");
    }
    previous = power;
  }
  if (support.effective_count.has_value() &&
      !(*support.effective_count > 0.0)) {
    return Result<bool>::failure(
        ErrorCode::InvalidDomain,
        "effective count must be positive when present");
  }
  const bool iidCovariance =
      algorithm.find("covariance") != std::string_view::npos &&
      (algorithm.find("iid") != std::string_view::npos ||
       algorithm.find("IID") != std::string_view::npos ||
       algorithm.find("identically distributed") != std::string_view::npos);
  if (iidCovariance &&
      (rule == WeightRule::unweighted || rule == WeightRule::frequency)) {
    const Scalar expected =
        rule == WeightRule::frequency
            ? static_cast<Scalar>(*support.frequency_mass) -
                  (center == CenterRule::estimated ? 1.0 : 0.0)
            : static_cast<Scalar>(support.valid) -
                  (center == CenterRule::estimated ? 1.0 : 0.0);
    if (!support.degrees_of_freedom.has_value() ||
        *support.degrees_of_freedom != expected) {
      return Result<bool>::failure(
          ErrorCode::InvalidDomain,
          "IID covariance degrees of freedom must match support and center");
    }
  }
  return Result<bool>::success(true);
}

bool tupleLess(std::span<const Index> left, std::span<const Index> right) {
  return std::lexicographical_compare(left.begin(), left.end(), right.begin(),
                                      right.end());
}
bool checkedMultiply(Extent left, Extent right, Extent& output) {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}
bool checkedAdd(Extent left, Extent right, Extent& output) {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

Result<Extent> packedCount(Extent dimension, Index order) {
  Extent top = 0U;
  if (!checkedAdd(dimension, order - 1U, top)) {
    return failure<Extent>(ErrorCode::SizeOverflow,
                           "packed moment extent overflows");
  }
  Extent count = 1U;
  for (Index i = 1U; i <= order; ++i) {
    Extent numerator = top - order + i;
    Extent denominator = i;
    const Extent firstDivisor = std::gcd(numerator, denominator);
    numerator /= firstDivisor;
    denominator /= firstDivisor;
    const Extent secondDivisor = std::gcd(count, denominator);
    count /= secondDivisor;
    denominator /= secondDivisor;
    if (denominator != 1U || !checkedMultiply(count, numerator, count)) {
      return failure<Extent>(ErrorCode::SizeOverflow,
                             "packed moment count overflows");
    }
  }
  return Result<Extent>::success(count);
}

Result<Extent> expectedPerGroupValues(const MomentRequest& request,
                                      Extent dimension,
                                      Extent outputDimension) {
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U) {
    return failure<Extent>(ErrorCode::InvalidDomain,
                           "moment order mask is invalid");
  }
  Extent total = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Extent count = 0U;
    if (request.representation == MomentRepresentation::component) {
      count = dimension;
    } else if (request.representation == MomentRepresentation::selected) {
      if (request.selected.empty()) {
        return failure<Extent>(ErrorCode::InvalidExtent,
                               "selected moments require coordinate tuples");
      }
      for (const auto tuple : request.selected) {
        if (tuple.size() != order) {
          return failure<Extent>(ErrorCode::DimensionMismatch,
                                 "selected tuple order does not match request");
        }
      }
      count = request.selected.size();
    } else if (request.representation == MomentRepresentation::packed) {
      auto packed = packedCount(dimension, order);
      if (!packed) {
        return failure<Extent>(packed.status().code(),
                               std::string(packed.status().message()));
      }
      count = packed.value();
    } else if (request.representation == MomentRepresentation::dense) {
      count = 1U;
      for (Index axis = 0U; axis < order; ++axis) {
        if (!checkedMultiply(count, dimension, count)) {
          return failure<Extent>(ErrorCode::SizeOverflow,
                                 "dense moment extent overflows");
        }
      }
    } else if (request.representation == MomentRepresentation::contraction) {
      if (!request.contraction.has_value() ||
          request.contraction->columns() != dimension ||
          request.contraction->rows() != outputDimension) {
        return failure<Extent>(ErrorCode::DimensionMismatch,
                               "contraction map does not match observations");
      }
      count = 1U;
      for (Index axis = 0U; axis < order; ++axis) {
        if (!checkedMultiply(count, outputDimension, count)) {
          return failure<Extent>(ErrorCode::SizeOverflow,
                                 "contracted moment extent overflows");
        }
      }
    } else {
      return failure<Extent>(ErrorCode::InvalidDomain,
                             "moment representation is unsupported");
    }
    if (!checkedAdd(total, count, total)) {
      return failure<Extent>(ErrorCode::SizeOverflow,
                             "group moment output extent overflows");
    }
  }
  return Result<Extent>::success(total);
}

bool completeRow(const SampleView& sample, Index row) {
  if (sample.valid.empty()) {
    return true;
  }
  const Extent columns = sample.rows.columns();
  for (Index column = 0U; column < columns; ++column) {
    if (sample.valid[row * columns + column] == 0U) {
      return false;
    }
  }
  return true;
}

} // namespace

Result<EstimatorDiagnostics>
validateEstimatorDiagnostics(const EstimatorDiagnostics& diagnostics) {
  if (!knownEnum(diagnostics.estimand) || !knownEnum(diagnostics.weight) ||
      !knownEnum(diagnostics.center) || !knownEnum(diagnostics.approximation)) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain, "diagnostics contain an unknown enum value");
  }
  const auto global =
      validateSupport(diagnostics.global, diagnostics.weight,
                      diagnostics.center, diagnostics.algorithm);
  if (!global) {
    return failure<EstimatorDiagnostics>(
        global.status().code(), std::string(global.status().message()));
  }
  for (const Support& support : diagnostics.coordinate_support.view()) {
    const auto checked = validateSupport(
        support, diagnostics.weight, diagnostics.center, diagnostics.algorithm);
    if (!checked) {
      return failure<EstimatorDiagnostics>(
          checked.status().code(), std::string(checked.status().message()));
    }
  }
  const auto tuples = diagnostics.tuple_support.view();
  for (Index i = 0U; i < tuples.size(); ++i) {
    const auto coordinates = tuples[i].coordinates.view();
    const auto checked =
        validateSupport(tuples[i].support, diagnostics.weight,
                        diagnostics.center, diagnostics.algorithm);
    if (!checked) {
      return failure<EstimatorDiagnostics>(
          checked.status().code(), std::string(checked.status().message()));
    }
    if (!std::is_sorted(coordinates.begin(), coordinates.end()) ||
        coordinates.empty() ||
        (i > 0U &&
         !tupleLess(tuples[i - 1U].coordinates.view(), coordinates))) {
      return failure<EstimatorDiagnostics>(
          ErrorCode::InvalidDomain,
          "tuple support coordinates must be nonempty, sorted, and unique");
    }
  }
  for (const auto value :
       {diagnostics.reciprocal_condition, diagnostics.numerical_error,
        diagnostics.sampling_error}) {
    if (value.has_value() && !std::isfinite(*value)) {
      return failure<EstimatorDiagnostics>(
          ErrorCode::NonFiniteInput,
          "diagnostic error or condition is non-finite");
    }
  }
  if (diagnostics.reciprocal_condition.has_value() &&
      !diagnostics.rank.has_value()) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain,
        "a reported reciprocal condition requires the defining system rank");
  }
  if (diagnostics.reciprocal_condition.has_value() &&
      (*diagnostics.reciprocal_condition < 0.0 ||
       *diagnostics.reciprocal_condition > 1.0)) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain, "reciprocal condition must lie in [0,1]");
  }
  if (diagnostics.algorithm.empty() || diagnostics.premises.empty()) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain,
        "algorithm and estimator premises must be named");
  }
  if (diagnostics.sampling_error.has_value() &&
      diagnostics.premises.find("model") == std::string::npos &&
      diagnostics.premises.find("IID") == std::string::npos &&
      diagnostics.premises.find("iid") == std::string::npos) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain,
        "sampling error requires a named sampling-model premise");
  }
  if ((diagnostics.numerical_error.has_value() &&
       *diagnostics.numerical_error < 0.0) ||
      (diagnostics.sampling_error.has_value() &&
       *diagnostics.sampling_error < 0.0)) {
    return failure<EstimatorDiagnostics>(
        ErrorCode::InvalidDomain, "diagnostic errors must be nonnegative");
  }
  for (const ErrorCode warning : diagnostics.warnings.view()) {
    if (warning == ErrorCode::None || !nonfatalWarning(warning)) {
      return failure<EstimatorDiagnostics>(
          ErrorCode::InvalidDomain,
          "fatal or unknown status cannot be represented as a warning on a "
          "successful estimate");
    }
  }
  return Result<EstimatorDiagnostics>::success(diagnostics);
}

Result<Estimate<Vector>> groupedStatistics(const SampleView& sample,
                                           std::span<const Index> groupIds,
                                           const WeightSpec& weights,
                                           const MomentRequest& request,
                                           Limits limits) {
  const Extent rows = sample.rows.rows();
  const Extent dimension = sample.rows.columns();
  if (rows == 0U || dimension == 0U) {
    return failure<Estimate<Vector>>(
        ErrorCode::InvalidExtent,
        "grouped sample must have rows and coordinates");
  }
  if (groupIds.size() != rows) {
    return failure<Estimate<Vector>>(
        ErrorCode::DimensionMismatch,
        "group identifier count must match sample rows");
  }
  Extent sampleCells = 0U;
  if (!checkedMultiply(rows, dimension, sampleCells)) {
    return failure<Estimate<Vector>>(ErrorCode::SizeOverflow,
                                     "grouped sample cell extent overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != sampleCells) {
    return failure<Estimate<Vector>>(
        ErrorCode::DimensionMismatch,
        "sample validity mask has the wrong extent");
  }
  for (const std::uint8_t valid : sample.valid) {
    if (valid > 1U) {
      return failure<Estimate<Vector>>(
          ErrorCode::InvalidDomain,
          "sample validity mask must contain only zero or one");
    }
  }
  if (weights.rule == WeightRule::empirical &&
      (!weights.rowWeights.has_value() || weights.rowWeights->size() != rows)) {
    return failure<Estimate<Vector>>(ErrorCode::DimensionMismatch,
                                     "row weights must match sample rows");
  }
  if (weights.rule == WeightRule::componentwise &&
      (!weights.componentWeights.has_value() ||
       weights.componentWeights->rows() != rows ||
       weights.componentWeights->columns() != dimension)) {
    return failure<Estimate<Vector>>(
        ErrorCode::DimensionMismatch,
        "component weights must match sample shape");
  }
  if (weights.rule != WeightRule::unweighted &&
      weights.rule != WeightRule::empirical &&
      weights.rule != WeightRule::componentwise) {
    return failure<Estimate<Vector>>(ErrorCode::InvalidDomain,
                                     "grouped moments support unweighted, "
                                     "empirical, or componentwise weights");
  }
  if (rows > limits.elements || rows > limits.bytes / sizeof(Index)) {
    return failure<Estimate<Vector>>(ErrorCode::ResourceLimit,
                                     "group ordering workspace exceeds limits");
  }
  std::vector<Index> order(rows);
  std::iota(order.begin(), order.end(), 0U);
  std::sort(order.begin(), order.end(), [&](Index left, Index right) {
    return groupIds[left] < groupIds[right] ||
           (groupIds[left] == groupIds[right] && left < right);
  });
  Extent groups = 1U;
  Extent maximumGroupRows = 1U;
  Extent currentGroupRows = 1U;
  for (Index i = 1U; i < rows; ++i) {
    if (groupIds[order[i]] == groupIds[order[i - 1U]]) {
      ++currentGroupRows;
      maximumGroupRows = std::max(maximumGroupRows, currentGroupRows);
    } else {
      ++groups;
      currentGroupRows = 1U;
    }
  }
  const Extent outputDimension =
      request.representation == MomentRepresentation::contraction &&
              request.contraction.has_value()
          ? request.contraction->rows()
          : dimension;
  auto perGroup = expectedPerGroupValues(request, dimension, outputDimension);
  if (!perGroup) {
    return failure<Estimate<Vector>>(perGroup.status().code(),
                                     std::string(perGroup.status().message()));
  }
  Extent outputValues = 0U;
  Extent outputElements = 0U;
  Extent outputBytes = 0U;
  Extent groupCells = 0U;
  Extent groupWorkspace = 0U;
  Extent workspaceBytes = 0U;
  Extent workspaceElements = 0U;
  Extent rowIndexBytes = 0U;
  Extent groupValueBytes = 0U;
  Extent groupScalarSlots = 0U;
  Extent groupScalarBytes = 0U;
  Extent groupOutputBytes = 0U;
  Extent extraRows = 0U;
  Extent extraOutputs = 0U;
  if (!checkedMultiply(groups, perGroup.value(), outputValues) ||
      !checkedMultiply(outputValues, 3U, outputElements) ||
      !checkedMultiply(outputValues, sizeof(Scalar) + 2U * sizeof(Support),
                       outputBytes) ||
      !checkedMultiply(maximumGroupRows, dimension, groupCells) ||
      !checkedMultiply(groupCells, 2U, groupWorkspace) ||
      !checkedMultiply(maximumGroupRows, 3U, extraRows) ||
      !checkedAdd(groupWorkspace, rows, groupWorkspace) ||
      !checkedAdd(groupWorkspace, extraRows, groupWorkspace) ||
      !checkedMultiply(perGroup.value(), 2U, extraOutputs) ||
      !checkedAdd(groupWorkspace, extraOutputs, workspaceElements) ||
      !checkedMultiply(rows, sizeof(Index), rowIndexBytes) ||
      !checkedMultiply(groupCells, 2U * sizeof(Scalar), groupValueBytes) ||
      !checkedMultiply(maximumGroupRows, 3U, groupScalarSlots) ||
      !checkedMultiply(groupScalarSlots, sizeof(Scalar), groupScalarBytes) ||
      !checkedMultiply(perGroup.value(), sizeof(Scalar) + 2U * sizeof(Support),
                       groupOutputBytes) ||
      !checkedAdd(rowIndexBytes, groupValueBytes, workspaceBytes) ||
      !checkedAdd(workspaceBytes, groupScalarBytes, workspaceBytes) ||
      !checkedAdd(workspaceBytes, groupOutputBytes, workspaceBytes) ||
      outputElements > limits.elements || outputBytes > limits.bytes ||
      workspaceElements > limits.elements - outputElements ||
      workspaceBytes > limits.bytes - outputBytes) {
    return failure<Estimate<Vector>>(
        ErrorCode::ResourceLimit,
        "grouped outputs and working storage exceed limits");
  }

  Vector values(outputValues, 0.0);
  std::vector<Support> coordinateSupports;
  coordinateSupports.reserve(outputValues);
  Index outputCursor = 0U;
  Support global{.input = rows,
                 .valid = 0U,
                 .positive = 0U,
                 .frequency_mass = std::nullopt,
                 .weight_mass = std::nullopt,
                 .normalized_weight_powers = {},
                 .effective_count = std::nullopt,
                 .degrees_of_freedom = std::nullopt,
                 .squared_weight_mass = std::nullopt};
  std::vector<Index> shape;
  shape.assign(sample.observation_shape.begin(),
               sample.observation_shape.end());
  Index begin = 0U;
  while (begin < rows) {
    Index end = begin + 1U;
    while (end < rows && groupIds[order[end]] == groupIds[order[begin]]) {
      ++end;
    }
    std::vector<Index> retained;
    retained.reserve(end - begin);
    for (Index i = begin; i < end; ++i) {
      if (completeRow(sample, order[i])) {
        retained.push_back(order[i]);
      }
    }
    if (retained.empty()) {
      return failure<Estimate<Vector>>(ErrorCode::InsufficientSamples,
                                       "a group has no complete rows");
    }
    Matrix groupValues(retained.size(), dimension);
    std::vector<Scalar> rowWeights;
    std::vector<Scalar> componentWeights;
    if (weights.rule == WeightRule::empirical) {
      rowWeights.reserve(retained.size());
    }
    if (weights.rule == WeightRule::componentwise) {
      componentWeights.reserve(retained.size() * dimension);
    }
    for (Index local = 0U; local < retained.size(); ++local) {
      const Index source = retained[local];
      for (Index column = 0U; column < dimension; ++column) {
        const Scalar value = sample.rows.at(source, column);
        if (!std::isfinite(value)) {
          return failure<Estimate<Vector>>(
              ErrorCode::NonFiniteInput,
              "a retained grouped value is non-finite");
        }
        groupValues.at(local, column) = value;
        if (weights.rule == WeightRule::componentwise) {
          componentWeights.push_back(
              weights.componentWeights->at(source, column));
        }
      }
      if (weights.rule == WeightRule::empirical) {
        rowWeights.push_back(weights.rowWeights->at(source));
      }
    }
    WeightSpec groupWeights{.rule = weights.rule,
                            .rowWeights = std::nullopt,
                            .componentWeights = std::nullopt,
                            .frequencyCounts = {}};
    if (weights.rule == WeightRule::empirical) {
      auto view = ConstVectorView::create(rowWeights, rowWeights.size());
      if (!view) {
        return failure<Estimate<Vector>>(view.status().code(),
                                         std::string(view.status().message()));
      }
      groupWeights.rowWeights = view.value();
    } else if (weights.rule == WeightRule::componentwise) {
      auto view = ConstMatrixView::create(componentWeights, retained.size(),
                                          dimension, dimension);
      if (!view) {
        return failure<Estimate<Vector>>(view.status().code(),
                                         std::string(view.status().message()));
      }
      groupWeights.componentWeights = view.value();
    }
    const SampleView groupSample{groupValues.view(), shape, {}};
    auto estimate =
        estimateRawMoments(groupSample, groupWeights, request, limits);
    if (!estimate) {
      return failure<Estimate<Vector>>(
          estimate.status().code(), std::string(estimate.status().message()));
    }
    const auto terms = estimate.value().terms.view();
    const auto tupleSupports =
        estimate.value().diagnostics.tuple_support.view();
    for (const MomentTerm& term : terms) {
      std::vector<Index> packedTuple(term.order, 0U);
      for (Index i = 0U; i < term.values.size(); ++i) {
        values.at(outputCursor++) = term.values.at(i);
        Support outputSupport{};
        if (!tupleSupports.empty()) {
          std::vector<Index> coordinates(term.order, 0U);
          if (term.layout == MomentRepresentation::selected) {
            const auto flattenedTuples = term.flattened_tuples.view();
            for (Index axis = 0U; axis < term.order; ++axis) {
              coordinates[axis] = flattenedTuples[i * term.order + axis];
            }
          } else if (term.layout == MomentRepresentation::component) {
            std::ranges::fill(coordinates, i);
          } else if (term.layout == MomentRepresentation::packed) {
            coordinates = packedTuple;
            if (i + 1U < term.values.size()) {
              Index axis = term.order;
              while (axis > 0U && packedTuple[axis - 1U] + 1U >= dimension) {
                --axis;
              }
              if (axis > 0U) {
                --axis;
                ++packedTuple[axis];
                std::ranges::fill(
                    std::span<Index>{packedTuple}.subspan(axis + 1U),
                    packedTuple[axis]);
              }
            }
          } else {
            const Extent tupleDimension =
                request.representation == MomentRepresentation::contraction
                    ? outputDimension
                    : dimension;
            Index linear = i;
            for (Index axis = term.order; axis > 0U; --axis) {
              coordinates[axis - 1U] = linear % tupleDimension;
              linear /= tupleDimension;
            }
          }
          std::sort(coordinates.begin(), coordinates.end());
          const auto found = std::find_if(
              tupleSupports.begin(), tupleSupports.end(),
              [&](const TupleSupport& candidate) {
                return std::equal(candidate.coordinates.view().begin(),
                                  candidate.coordinates.view().end(),
                                  coordinates.begin(), coordinates.end());
              });
          if (found == tupleSupports.end()) {
            return failure<Estimate<Vector>>(
                ErrorCode::InvalidDomain,
                "moment output has no matching tuple support diagnostic");
          }
          outputSupport = found->support;
        } else if (estimate.value()
                       .diagnostics.coordinate_support.view()
                       .size() == term.values.size()) {
          outputSupport =
              estimate.value().diagnostics.coordinate_support.view()[i];
        } else {
          outputSupport = estimate.value().diagnostics.global;
        }
        outputSupport.input = end - begin;
        coordinateSupports.push_back(outputSupport);
      }
    }
    global.valid += estimate.value().diagnostics.global.valid;
    global.positive += estimate.value().diagnostics.global.positive;
    begin = end;
  }
  if (outputCursor != outputValues) {
    return failure<Estimate<Vector>>(
        ErrorCode::InvalidDomain,
        "grouped output size differs from its preflight");
  }
  auto supports = Array<Support>::copy(coordinateSupports, limits);
  if (!supports) {
    return failure<Estimate<Vector>>(supports.status().code(),
                                     std::string(supports.status().message()));
  }
  auto emptyTuples = Array<TupleSupport>::copy({}, limits);
  if (!emptyTuples) {
    return failure<Estimate<Vector>>(
        emptyTuples.status().code(),
        std::string(emptyTuples.status().message()));
  }
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = weights.rule,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = global,
      .coordinate_support = std::move(supports).value(),
      .tuple_support = std::move(emptyTuples).value(),
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "independent raw moments by sorted group identifier",
      .premises = "groups are separate supports; no common law is inferred",
      .warnings = {}};
  return Result<Estimate<Vector>>::success(
      Estimate<Vector>{.value = std::move(values),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::nullopt});
}

} // namespace vss::math::statistics
