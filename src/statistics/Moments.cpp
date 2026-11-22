#include "MomentInterval.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string_view message) {
  return Result<T>::failure(code, std::string(message));
}

struct Scaled final {
  Scalar fraction{0.0};
  int exponent{0};
};

Scaled scaled(Scalar value) {
  if (value == 0.0) {
    return {};
  }
  int exponent = 0;
  const Scalar fraction = std::frexp(value, &exponent);
  return {.fraction = fraction, .exponent = exponent};
}

Scaled multiply(Scaled left, Scaled right) {
  if (left.fraction == 0.0 || right.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(left.fraction * right.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = left.exponent + right.exponent + adjustment};
}

Scaled divide(Scaled numerator, Scaled denominator) {
  if (numerator.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(numerator.fraction / denominator.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = numerator.exponent - denominator.exponent + adjustment};
}

class ScaledAccumulator final {
public:
  void add(Scaled value) noexcept {
    if (value.fraction == 0.0) {
      return;
    }
    if (!initialized_) {
      exponent_ = value.exponent;
      initialized_ = true;
    } else if (value.exponent > exponent_) {
      const Scalar factor = std::scalbn(1.0, exponent_ - value.exponent);
      sum_ *= factor;
      correction_ *= factor;
      exponent_ = value.exponent;
    }
    const Scalar addend =
        std::scalbn(value.fraction, value.exponent - exponent_);
    const Scalar total = sum_ + addend;
    if (std::abs(sum_) >= std::abs(addend)) {
      correction_ += (sum_ - total) + addend;
    } else {
      correction_ += (addend - total) + sum_;
    }
    sum_ = total;
  }

  [[nodiscard]] Scaled value() const noexcept {
    const Scalar total = sum_ + correction_;
    if (!initialized_ || total == 0.0) {
      return {};
    }
    int adjustment = 0;
    const Scalar fraction = std::frexp(total, &adjustment);
    return {.fraction = fraction, .exponent = exponent_ + adjustment};
  }

private:
  Scalar sum_{0.0};
  Scalar correction_{0.0};
  int exponent_{0};
  bool initialized_{false};
};

struct ComponentSupport final {
  ScaledAccumulator mass;
  Index positive{0U};
  Index valid{0U};
  Index input{0U};
};

bool toScalar(Scaled value, Scalar& result) noexcept {
  result = std::scalbn(value.fraction, value.exponent);
  return std::isfinite(result);
}

Scaled difference(Scalar left, Scalar right) {
  ScaledAccumulator result;
  result.add(scaled(left));
  result.add(scaled(-right));
  return result.value();
}

bool checkedAdd(Extent left, Extent right, Extent& result) noexcept {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(Extent left, Extent right, Extent& result) noexcept {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

Result<Extent> validateSample(const SampleView& sample) {
  const Extent rows = sample.rows.rows();
  const Extent columns = sample.rows.columns();
  if (rows == 0U || columns == 0U) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "moment population must have rows and coordinates");
  }
  if (sample.observation_shape.empty()) {
    if (columns != 1U) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "scalar observations require exactly one column");
    }
  } else {
    Extent product = 1U;
    for (const Extent extent : sample.observation_shape) {
      if (extent == 0U) {
        return fail<Extent>(ErrorCode::InvalidExtent,
                            "observation shape extents must be positive");
      }
      if (!checkedMultiply(product, extent, product)) {
        return fail<Extent>(ErrorCode::SizeOverflow,
                            "observation shape product overflows");
      }
    }
    if (product != columns) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "observation shape does not match sample columns");
    }
  }
  Extent cells = 0U;
  if (!checkedMultiply(rows, columns, cells)) {
    return fail<Extent>(ErrorCode::SizeOverflow, "sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != cells) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "validity mask must match all sample cells");
  }
  for (const std::uint8_t valid : sample.valid) {
    if (valid > 1U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "validity mask entries must be zero or one");
    }
    if (valid == 0U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "moment estimators require complete observations");
    }
  }
  return Result<Extent>::success(columns);
}

Status validateWeightStructure(const WeightSpec& weights,
                               const SampleView& sample) {
  const bool hasRows = weights.rowWeights.has_value();
  const bool hasComponents = weights.componentWeights.has_value();
  const bool hasCounts = !weights.frequencyCounts.empty();
  if (weights.rule == WeightRule::unweighted && !hasRows && !hasComponents &&
      !hasCounts) {
    return Status::success();
  }
  if (weights.rule == WeightRule::empirical && hasRows && !hasComponents &&
      !hasCounts) {
    if (weights.rowWeights->size() != sample.rows.rows()) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "row weights must match the observation count");
    }
    return Status::success();
  }
  if (weights.rule == WeightRule::componentwise && !hasRows && hasComponents &&
      !hasCounts) {
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
      "moment weights require an unweighted, empirical-row, or compatible "
      "componentwise specification");
}

struct OrderPlan final {
  Index order;
  Extent count;
};

Result<Extent> denseCount(Extent dimension, Index order) {
  Extent count = 1U;
  for (Index axis = 0U; axis < order; ++axis) {
    if (!checkedMultiply(count, dimension, count)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "dense moment extent overflows");
    }
  }
  return Result<Extent>::success(count);
}

Result<Extent> packedCount(Extent dimension, Index order) {
  Extent n = 0U;
  if (!checkedAdd(dimension, order - 1U, n)) {
    return fail<Extent>(ErrorCode::SizeOverflow,
                        "packed moment extent overflows");
  }
  Extent value = 1U;
  for (Index i = 1U; i <= order; ++i) {
    Extent numerator = n - order + i;
    Extent denominator = i;
    const Extent common = std::gcd(numerator, denominator);
    numerator /= common;
    denominator /= common;
    const Extent commonWithValue = std::gcd(value, denominator);
    value /= commonWithValue;
    denominator /= commonWithValue;
    if (denominator != 1U || !checkedMultiply(value, numerator, value)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment count overflows");
    }
  }
  return Result<Extent>::success(value);
}

Status validateRequest(const MomentRequest& request, Extent dimension,
                       const WeightSpec& weights) {
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "moment order mask must select orders one through four");
  }
  if (request.representation != MomentRepresentation::component &&
      request.representation != MomentRepresentation::selected &&
      request.representation != MomentRepresentation::packed &&
      request.representation != MomentRepresentation::dense &&
      request.representation != MomentRepresentation::contraction) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "unknown moment representation");
  }
  const bool selected =
      request.representation == MomentRepresentation::selected;
  const bool contracted =
      request.representation == MomentRepresentation::contraction;
  if (!selected && !request.selected.empty()) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "selected tuples require the selected layout");
  }
  if (!contracted && request.contraction.has_value()) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "a contraction map requires contraction layout");
  }
  if (selected) {
    if (request.selected.empty()) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "selected moment layout requires at least one tuple");
    }
    if (std::popcount(request.orders) != 1) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "selected tuples require exactly one moment order");
    }
    const Index order =
        static_cast<Index>(std::countr_zero(request.orders)) + 1U;
    for (const auto tuple : request.selected) {
      if (tuple.size() != order) {
        return Status::failure(ErrorCode::DimensionMismatch,
                               "selected tuple length must equal its order");
      }
      for (const Index coordinate : tuple) {
        if (coordinate >= dimension) {
          return Status::failure(ErrorCode::IndexOutOfRange,
                                 "selected tuple coordinate is out of range");
        }
      }
      if (weights.rule == WeightRule::componentwise &&
          !std::ranges::all_of(tuple, [&](Index coordinate) {
            return coordinate == tuple.front();
          })) {
        return Status::failure(
            ErrorCode::InvalidDomain,
            "mixed moments require a common row-weight population");
      }
    }
  }
  if (contracted) {
    if (!request.contraction.has_value()) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "contraction layout requires a linear map");
    }
    const auto& map = request.contraction.value();
    if (map.rows() == 0U) {
      return Status::failure(ErrorCode::InvalidExtent,
                             "contraction map must have a positive row count");
    }
    if (map.columns() != dimension) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "contraction map must have d columns");
    }
    if (weights.rule == WeightRule::componentwise) {
      if (map.rows() != 1U) {
        return Status::failure(
            ErrorCode::InvalidDomain,
            "component weights cannot define joint contracted moments");
      }
    }
  } else if (weights.rule == WeightRule::componentwise &&
             request.representation != MomentRepresentation::component &&
             dimension > 1U) {
    for (Index order = 2U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      if (request.representation == MomentRepresentation::dense ||
          request.representation == MomentRepresentation::packed) {
        return Status::failure(
            ErrorCode::InvalidDomain,
            "dense or packed mixed moments require common row weights");
      }
    }
  }
  return Status::success();
}

Status validateCenter(const CenterSpec& center, Extent dimension,
                      bool validatePayload) {
  if (center.rule == CenterRule::estimated) {
    if (center.center.has_value()) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "estimated center must not supply coordinates");
    }
    return Status::success();
  }
  if (center.rule != CenterRule::known) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "unknown moment center rule");
  }
  if (!center.center.has_value() || center.center->size() != dimension) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "known center must contain exactly d coordinates");
  }
  if (validatePayload) {
    for (Index i = 0U; i < dimension; ++i) {
      if (!std::isfinite(center.center->at(i))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "known center contains a non-finite coordinate");
      }
    }
  }
  return Status::success();
}

Result<std::array<OrderPlan, 4U>>
makePlans(const MomentRequest& request, Extent dimension,
          Extent outputDimension, Extent& planCount, Extent& totalValues,
          Extent& maximumTupleItems) {
  std::array<OrderPlan, 4U> plans{};
  planCount = 0U;
  totalValues = 0U;
  maximumTupleItems = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Extent count = 0U;
    switch (request.representation) {
    case MomentRepresentation::component:
      count = dimension;
      break;
    case MomentRepresentation::selected:
      count = request.selected.size();
      break;
    case MomentRepresentation::packed: {
      auto packed = packedCount(dimension, order);
      if (!packed) {
        return fail<std::array<OrderPlan, 4U>>(packed.status().code(),
                                               packed.status().message());
      }
      count = packed.value();
      break;
    }
    case MomentRepresentation::dense:
      [[fallthrough]];
    case MomentRepresentation::contraction: {
      const Extent dim =
          request.representation == MomentRepresentation::contraction
              ? outputDimension
              : dimension;
      auto dense = denseCount(dim, order);
      if (!dense) {
        return fail<std::array<OrderPlan, 4U>>(dense.status().code(),
                                               dense.status().message());
      }
      count = dense.value();
      break;
    }
    }
    Extent tupleItems = 0U;
    if (!checkedMultiply(count, order, tupleItems)) {
      return fail<std::array<OrderPlan, 4U>>(
          ErrorCode::SizeOverflow, "moment tuple workspace overflows");
    }
    if (!checkedAdd(totalValues, count, totalValues)) {
      return fail<std::array<OrderPlan, 4U>>(ErrorCode::SizeOverflow,
                                             "moment output count overflows");
    }
    maximumTupleItems = std::max(maximumTupleItems, tupleItems);
    plans.at(planCount) = {.order = order, .count = count};
    ++planCount;
  }
  return Result<std::array<OrderPlan, 4U>>::success(plans);
}

bool withinLimits(Extent outputValues, Extent outputIndices,
                  Extent outputRecords, Extent workspaceValues,
                  Extent workspaceIndices, Extent workspaceRecords,
                  Limits limits) {
  Extent elements = 0U;
  Extent bytes = 0U;
  auto addPart = [&](Extent count, Extent size) {
    Extent partBytes = 0U;
    return checkedMultiply(count, size, partBytes) &&
           checkedAdd(bytes, partBytes, bytes) &&
           checkedAdd(elements, count, elements);
  };
  constexpr Extent outputRecordBytes =
      std::max({sizeof(MomentTerm), sizeof(MomentEstimate), sizeof(Support),
                sizeof(TupleSupport)});
  constexpr Extent workspaceRecordBytes =
      std::max(sizeof(ScaledAccumulator), sizeof(ComponentSupport));
  return addPart(outputValues, sizeof(Scalar)) &&
         addPart(outputIndices, sizeof(Index)) &&
         addPart(outputRecords, outputRecordBytes) &&
         addPart(workspaceValues, sizeof(Scaled)) &&
         addPart(workspaceIndices, sizeof(Index)) &&
         addPart(workspaceRecords, workspaceRecordBytes) &&
         elements <= limits.elements && bytes <= limits.bytes;
}

Scalar weightAt(const WeightSpec& weights, Index row, Index coordinate) {
  if (weights.rule == WeightRule::unweighted) {
    return 1.0;
  }
  if (weights.rule == WeightRule::empirical) {
    if (weights.rowWeights.has_value()) {
      return weights.rowWeights.value().at(row);
    }
    return 0.0;
  }
  if (weights.componentWeights.has_value()) {
    return weights.componentWeights.value().at(row, coordinate);
  }
  return 0.0;
}

Support
supportFrom(const ComponentSupport& source, bool includeMass,
            std::optional<std::array<Scalar, 2U>> powers = std::nullopt) {
  std::optional<Scalar> mass;
  std::optional<Scalar> power2;
  std::optional<Scalar> power3;
  std::optional<Scalar> effective;
  std::optional<Scalar> power1;
  if (includeMass) {
    Scalar converted = 0.0;
    if (toScalar(source.mass.value(), converted)) {
      mass = converted;
    }
    if (source.mass.value().fraction != 0.0 && powers.has_value()) {
      power1 = 1.0;
      power2 = powers->at(0U);
      power3 = powers->at(1U);
      effective = power2.value() > 0.0
                      ? std::optional<Scalar>{1.0 / power2.value()}
                      : std::nullopt;
    }
  }
  return Support{.input = source.input,
                 .valid = source.valid,
                 .positive = source.positive,
                 .frequency_mass = std::nullopt,
                 .weight_mass = mass,
                 .normalized_weight_powers = {power1, power2, power3},
                 .effective_count = effective,
                 .degrees_of_freedom = std::nullopt,
                 .squared_weight_mass = std::nullopt};
}

Result<std::array<Scalar, 2U>> normalizedPowers(const SampleView& sample,
                                                const WeightSpec& weights,
                                                Scaled denominator,
                                                Index coordinate = 0U) {
  vss::math::detail::CompensatedSum square;
  vss::math::detail::CompensatedSum cube;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar w = weightAt(weights, row, coordinate);
    if (w == 0.0) {
      continue;
    }
    const Scaled ratio = divide(scaled(w), denominator);
    Scalar normalized = 0.0;
    if (!toScalar(ratio, normalized)) {
      return fail<std::array<Scalar, 2U>>(ErrorCode::NonFiniteResult,
                                          "normalized weight is not finite");
    }
    square.add(normalized * normalized);
    cube.add(normalized * normalized * normalized);
  }
  return Result<std::array<Scalar, 2U>>::success(
      {square.total(), cube.total()});
}

Result<Scaled> dotRow(const SampleView& sample, Index row,
                      const ConstMatrixView& map, Index output,
                      const Vector* center) {
  ScaledAccumulator sum;
  for (Index coordinate = 0U; coordinate < sample.rows.columns();
       ++coordinate) {
    const Scalar coefficient = map.at(output, coordinate);
    if (coefficient == 0.0) {
      continue;
    }
    Scaled input = scaled(sample.rows.at(row, coordinate));
    if (center != nullptr) {
      input =
          difference(sample.rows.at(row, coordinate), center->at(coordinate));
    }
    sum.add(multiply(scaled(coefficient), input));
  }
  return Result<Scaled>::success(sum.value());
}

Result<MomentEstimate> estimate(const SampleView& sample,
                                const WeightSpec& weights,
                                const CenterSpec* center,
                                const MomentRequest& request, Limits limits,
                                bool central) {
  auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<MomentEstimate>(dimension.status().code(),
                                dimension.status().message());
  }
  const Extent d = dimension.value();
  const Status weightStructure = validateWeightStructure(weights, sample);
  if (weightStructure.code() != ErrorCode::None) {
    return fail<MomentEstimate>(weightStructure.code(),
                                weightStructure.message());
  }
  const Status requestStatus = validateRequest(request, d, weights);
  if (requestStatus.code() != ErrorCode::None) {
    return fail<MomentEstimate>(requestStatus.code(), requestStatus.message());
  }
  if (central) {
    const Status centerStatus = validateCenter(*center, d, false);
    if (centerStatus.code() != ErrorCode::None) {
      return fail<MomentEstimate>(centerStatus.code(), centerStatus.message());
    }
  }
  const bool contracted =
      request.representation == MomentRepresentation::contraction;
  const Extent outD = contracted ? request.contraction->rows() : d;
  Extent planCount = 0U;
  Extent totalValues = 0U;
  Extent tupleWorkspace = 0U;
  auto plansResult =
      makePlans(request, d, outD, planCount, totalValues, tupleWorkspace);
  if (!plansResult) {
    return fail<MomentEstimate>(plansResult.status().code(),
                                plansResult.status().message());
  }
  const auto plans = plansResult.value();
  Extent selectedItems = 0U;
  if (request.representation == MomentRepresentation::selected) {
    for (const auto tuple : request.selected) {
      if (!checkedAdd(selectedItems, tuple.size(), selectedItems)) {
        return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                    "selected tuple metadata overflows");
      }
    }
  }
  Extent diagnosticTupleItems = 0U;
  const Extent termCount = planCount;
  if (weights.rule == WeightRule::componentwise) {
    for (Extent i = 0U; i < planCount; ++i) {
      Extent items = 0U;
      if (!checkedMultiply(plans.at(i).count, plans.at(i).order, items) ||
          !checkedAdd(diagnosticTupleItems, items, diagnosticTupleItems)) {
        return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                    "tuple support metadata overflows");
      }
    }
  }
  const Extent outputShapeItems =
      contracted ? 1U : sample.observation_shape.size();
  Extent outputIndices = 0U;
  Extent workspaceIndices = 0U;
  Extent outputRecords = 2U;
  Extent workspaceRecords = 0U;
  Extent workspaceValues = 0U;
  if (!checkedAdd(outputShapeItems, selectedItems, outputIndices) ||
      !checkedAdd(outputIndices, diagnosticTupleItems, outputIndices) ||
      !checkedAdd(termCount, 1U + d, outputRecords) ||
      !checkedAdd(outputRecords,
                  weights.rule == WeightRule::componentwise ? totalValues : 0U,
                  outputRecords) ||
      !checkedAdd(tupleWorkspace, selectedItems, workspaceIndices) ||
      !checkedAdd(workspaceIndices, diagnosticTupleItems, workspaceIndices) ||
      !checkedAdd(d, outD, workspaceValues) ||
      !checkedAdd(workspaceValues, 8U, workspaceValues) ||
      !checkedAdd(d, termCount, workspaceRecords) ||
      !checkedAdd(workspaceRecords, totalValues, workspaceRecords) ||
      !checkedAdd(workspaceRecords, 2U, workspaceRecords)) {
    return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                "moment output/workspace accounting overflows");
  }
  Extent outputValueCopies = 0U;
  if (!checkedMultiply(totalValues, 2U, outputValueCopies) ||
      !checkedAdd(outputValueCopies, outD, outputValueCopies)) {
    return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                "moment output accounting overflows");
  }
  if (!withinLimits(outputValueCopies, outputIndices, outputRecords,
                    workspaceValues, workspaceIndices, workspaceRecords,
                    limits)) {
    return fail<MomentEstimate>(
        ErrorCode::ResourceLimit,
        "moment output/workspace exceeds configured limits");
  }

  if (central) {
    const Status centerPayload = validateCenter(*center, d, true);
    if (centerPayload.code() != ErrorCode::None) {
      return fail<MomentEstimate>(centerPayload.code(),
                                  centerPayload.message());
    }
  }
  if (contracted) {
    Index nonzero = 0U;
    for (Index row = 0U; row < outD; ++row) {
      for (Index column = 0U; column < d; ++column) {
        const Scalar coefficient = request.contraction->at(row, column);
        if (!std::isfinite(coefficient)) {
          return fail<MomentEstimate>(
              ErrorCode::NonFiniteInput,
              "contraction map contains a non-finite value");
        }
        if (row == 0U && coefficient != 0.0) {
          ++nonzero;
        }
      }
    }
    if (weights.rule == WeightRule::componentwise && nonzero != 1U) {
      return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                  "component weights cannot define a "
                                  "contraction combining coordinates");
    }
  }

  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    for (Index coordinate = 0U; coordinate < d; ++coordinate) {
      Scalar value = 0.0;
      if (weights.rule == WeightRule::componentwise) {
        value = weights.componentWeights->at(row, coordinate);
      } else if (coordinate == 0U && weights.rule == WeightRule::empirical) {
        value = weights.rowWeights->at(row);
      } else if (weights.rule == WeightRule::unweighted) {
        value = 1.0;
      } else {
        continue;
      }
      if (!std::isfinite(value)) {
        return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                    "weight input contains a non-finite value");
      }
      if (value < 0.0) {
        return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                    "moment weights must be nonnegative");
      }
    }
  }

  const bool componentWeights = weights.rule == WeightRule::componentwise;
  std::vector<ScaledAccumulator> numerators(d);
  std::vector<ComponentSupport> coordinateSupports(componentWeights ? d : 0U);
  for (auto& support : coordinateSupports) {
    support.input = sample.rows.rows();
    support.valid = sample.rows.rows();
  }
  ComponentSupport globalSupport{.mass = {},
                                 .positive = 0U,
                                 .valid = sample.rows.rows(),
                                 .input = sample.rows.rows()};
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    if (componentWeights) {
      bool rowPositive = false;
      for (Index coordinate = 0U; coordinate < d; ++coordinate) {
        const Scalar weight = weights.componentWeights->at(row, coordinate);
        auto& support = coordinateSupports[coordinate];
        if (weight == 0.0) {
          continue;
        }
        const Scalar value = sample.rows.at(row, coordinate);
        if (!std::isfinite(value)) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                      "positive-weight sample is non-finite");
        }
        support.mass.add(scaled(weight));
        support.positive += 1U;
        numerators[coordinate].add(multiply(scaled(weight), scaled(value)));
        rowPositive = true;
      }
      if (rowPositive) {
        globalSupport.positive += 1U;
      }
    } else {
      const Scalar weight = weightAt(weights, row, 0U);
      if (weight == 0.0) {
        continue;
      }
      globalSupport.mass.add(scaled(weight));
      globalSupport.positive += 1U;
      for (Index coordinate = 0U; coordinate < d; ++coordinate) {
        const Scalar value = sample.rows.at(row, coordinate);
        if (!std::isfinite(value)) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                      "positive-weight sample is non-finite");
        }
        numerators[coordinate].add(multiply(scaled(weight), scaled(value)));
      }
    }
  }
  if (!componentWeights && globalSupport.positive == 0U) {
    return fail<MomentEstimate>(ErrorCode::InsufficientSamples,
                                "moment population has no positive mass");
  }
  std::vector<Scalar> means(d, 0.0);
  std::vector<bool> meanAvailable(d, false);
  for (Index coordinate = 0U; coordinate < d; ++coordinate) {
    const Scaled mass = componentWeights
                            ? coordinateSupports[coordinate].mass.value()
                            : globalSupport.mass.value();
    if (mass.fraction == 0.0) {
      continue;
    }
    if (componentWeights && coordinateSupports[coordinate].positive == 0U) {
      continue;
    }
    if (!toScalar(divide(numerators[coordinate].value(), mass),
                  means[coordinate])) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                  "population mean is not representable");
    }
    meanAvailable[coordinate] = true;
  }

  for (Extent i = 0U; i < planCount; ++i) {
    if (central && center->rule == CenterRule::estimated) {
      if (request.representation == MomentRepresentation::contraction &&
          componentWeights) {
        for (Index coordinate = 0U; coordinate < d; ++coordinate) {
          if (request.contraction->at(0U, coordinate) != 0.0 &&
              !meanAvailable[coordinate]) {
            return fail<MomentEstimate>(
                ErrorCode::InsufficientSamples,
                "estimated center has no positive support");
          }
        }
      } else if (componentWeights &&
                 request.representation == MomentRepresentation::selected) {
        for (const auto tuple : request.selected) {
          if (!meanAvailable[tuple.front()]) {
            return fail<MomentEstimate>(
                ErrorCode::InsufficientSamples,
                "selected coordinate has no positive support");
          }
        }
      } else if (componentWeights &&
                 request.representation == MomentRepresentation::component) {
        for (Index coordinate = 0U; coordinate < d; ++coordinate) {
          if (!meanAvailable[coordinate]) {
            return fail<MomentEstimate>(ErrorCode::InsufficientSamples,
                                        "component has no positive support");
          }
        }
      } else {
        for (Index coordinate = 0U; coordinate < d; ++coordinate) {
          if (!meanAvailable[coordinate]) {
            return fail<MomentEstimate>(
                ErrorCode::InsufficientSamples,
                "estimated center has no positive support");
          }
        }
      }
    }
  }

  std::array<Scalar, 2U> commonPowers{0.0, 0.0};
  if (!componentWeights) {
    auto powers = normalizedPowers(sample, weights, globalSupport.mass.value());
    if (!powers) {
      return fail<MomentEstimate>(powers.status().code(),
                                  powers.status().message());
    }
    commonPowers = powers.value();
  }
  std::vector<Support> coordinateSupportValues;
  if (componentWeights) {
    coordinateSupportValues.reserve(d);
    for (Index coordinate = 0U; coordinate < d; ++coordinate) {
      std::optional<std::array<Scalar, 2U>> powers;
      if (coordinateSupports[coordinate].positive > 0U) {
        auto normalized = normalizedPowers(
            sample, weights, coordinateSupports[coordinate].mass.value(),
            coordinate);
        if (!normalized) {
          return fail<MomentEstimate>(normalized.status().code(),
                                      normalized.status().message());
        }
        powers = normalized.value();
      }
      coordinateSupportValues.push_back(
          supportFrom(coordinateSupports[coordinate], true, powers));
    }
  }
  auto coordinateSupport =
      Array<Support>::copy(coordinateSupportValues, limits);
  if (!coordinateSupport) {
    return fail<MomentEstimate>(coordinateSupport.status().code(),
                                coordinateSupport.status().message());
  }
  const Support global =
      supportFrom(globalSupport, !componentWeights, commonPowers);

  std::vector<Scalar> outputMean(outD, 0.0);
  bool outputMeanAvailable = true;
  if (outputMeanAvailable) {
    if (contracted) {
      for (Index output = 0U; output < outD; ++output) {
        ScaledAccumulator sum;
        for (Index coordinate = 0U; coordinate < d; ++coordinate) {
          const Scalar coefficient =
              request.contraction->at(output, coordinate);
          if (coefficient == 0.0) {
            continue;
          }
          if (!meanAvailable[coordinate]) {
            outputMeanAvailable = false;
            break;
          }
          sum.add(multiply(scaled(coefficient), scaled(means[coordinate])));
        }
        if (!outputMeanAvailable) {
          break;
        }
        if (!toScalar(sum.value(), outputMean[output])) {
          outputMeanAvailable = false;
          break;
        }
      }
    } else {
      for (Index coordinate = 0U; coordinate < d; ++coordinate) {
        if (!meanAvailable[coordinate]) {
          outputMeanAvailable = false;
          break;
        }
      }
      if (outputMeanAvailable) {
        outputMean = means;
      }
    }
  }

  std::vector<MomentTerm> terms;
  terms.reserve(planCount);
  std::vector<TupleSupport> tupleSupports;
  if (componentWeights) {
    tupleSupports.reserve(totalValues);
  }
  for (Extent planIndex = 0U; planIndex < planCount; ++planIndex) {
    const Index order = plans.at(planIndex).order;
    const Extent count = plans.at(planIndex).count;
    const Extent tupleItems = count * order;
    std::vector<Index> tuples(tupleItems, 0U);
    if (request.representation == MomentRepresentation::component) {
      const std::span<Index> tupleStorage{tuples};
      for (Extent term = 0U; term < count; ++term) {
        std::ranges::fill(tupleStorage.subspan(term * order, order), term);
      }
    } else if (request.representation == MomentRepresentation::selected) {
      const std::span<Index> tupleStorage{tuples};
      Extent cursor = 0U;
      for (const auto tuple : request.selected) {
        std::ranges::copy(tuple, tupleStorage.subspan(cursor, order).begin());
        cursor += order;
      }
    } else if (request.representation == MomentRepresentation::packed) {
      std::vector<Index> tuple(order, 0U);
      const std::span<Index> tupleStorage{tuples};
      for (Extent term = 0U; term < count; ++term) {
        std::ranges::copy(tuple,
                          tupleStorage.subspan(term * order, order).begin());
        if (term + 1U == count) {
          continue;
        }
        Index axis = order;
        while (axis > 0U && tuple[axis - 1U] + 1U >= d) {
          --axis;
        }
        --axis;
        ++tuple[axis];
        std::ranges::fill(std::span<Index>{tuple}.subspan(axis + 1U),
                          tuple[axis]);
      }
    } else {
      const Extent tupleDimension = contracted ? outD : d;
      for (Extent term = 0U; term < count; ++term) {
        Extent value = term;
        for (Index axis = order; axis > 0U; --axis) {
          tuples[(term * order) + axis - 1U] = value % tupleDimension;
          value /= tupleDimension;
        }
      }
    }

    const std::span<const Index> tupleStorage{tuples};
    std::vector<Scalar> values(count, 0.0);
    for (Extent term = 0U; term < count; ++term) {
      const std::span<const Index> tuple =
          tupleStorage.subspan(term * order, order);
      if (central && center->rule == CenterRule::estimated && order == 1U) {
        values[term] = 0.0;
      } else {
        const Index component = tuple.front();
        Index weightCoordinate = component;
        if (componentWeights && contracted) {
          for (Index input = 0U; input < d; ++input) {
            if (request.contraction->at(0U, input) != 0.0) {
              weightCoordinate = input;
              break;
            }
          }
        }
        const Scaled denominator =
            componentWeights ? coordinateSupports[weightCoordinate].mass.value()
                             : globalSupport.mass.value();
        ScaledAccumulator numerator;
        std::vector<Scaled> mapped(contracted ? outD : 0U);
        for (Index row = 0U; row < sample.rows.rows(); ++row) {
          const Scalar weight =
              weightAt(weights, row, componentWeights ? weightCoordinate : 0U);
          if (weight == 0.0) {
            continue;
          }
          if (contracted) {
            for (Index output = 0U; output < outD; ++output) {
              if (central) {
                if (center->rule == CenterRule::known) {
                  ScaledAccumulator dot;
                  for (Index input = 0U; input < d; ++input) {
                    const Scalar coefficient =
                        request.contraction->at(output, input);
                    if (coefficient == 0.0) {
                      continue;
                    }
                    dot.add(multiply(scaled(coefficient),
                                     difference(sample.rows.at(row, input),
                                                center->center->at(input))));
                  }
                  mapped[output] = dot.value();
                } else {
                  ScaledAccumulator dot;
                  for (Index input = 0U; input < d; ++input) {
                    const Scalar coefficient =
                        request.contraction->at(output, input);
                    if (coefficient == 0.0) {
                      continue;
                    }
                    dot.add(multiply(
                        scaled(coefficient),
                        difference(sample.rows.at(row, input), means[input])));
                  }
                  mapped[output] = dot.value();
                }
              } else {
                auto dot =
                    dotRow(sample, row, *request.contraction, output, nullptr);
                mapped[output] = dot.value();
              }
            }
          }
          Scaled product = scaled(1.0);
          for (Index axis = 0U; axis < order; ++axis) {
            const Index coordinate = tuple[axis];
            Scaled value;
            if (contracted) {
              value = mapped[coordinate];
            } else if (!central) {
              value = scaled(sample.rows.at(row, coordinate));
            } else if (center->rule == CenterRule::estimated) {
              value = difference(sample.rows.at(row, coordinate),
                                 means[coordinate]);
            } else {
              value = difference(sample.rows.at(row, coordinate),
                                 center->center->at(coordinate));
            }
            product = multiply(product, value);
          }
          numerator.add(multiply(scaled(weight), product));
        }
        if (!toScalar(divide(numerator.value(), denominator), values[term])) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                      "requested moment is not representable");
        }
      }
      if (componentWeights) {
        const Index coordinate = tuple.front();
        std::vector<Index> coordinates(tuple.begin(), tuple.end());
        auto storedCoordinates = Array<Index>::copy(coordinates, limits);
        if (!storedCoordinates) {
          return fail<MomentEstimate>(storedCoordinates.status().code(),
                                      storedCoordinates.status().message());
        }
        tupleSupports.push_back(
            {.coordinates = std::move(storedCoordinates).value(),
             .support = coordinateSupportValues[coordinate]});
      }
    }
    Array<Index> flattened;
    if (request.representation == MomentRepresentation::selected) {
      auto copy = Array<Index>::copy(tuples, limits);
      if (!copy) {
        return fail<MomentEstimate>(copy.status().code(),
                                    copy.status().message());
      }
      flattened = std::move(copy).value();
    }
    terms.push_back(
        MomentTerm{.order = order,
                   .layout = contracted ? MomentRepresentation::dense
                                        : request.representation,
                   .flattened_tuples = std::move(flattened),
                   .values = Vector(std::span<const Scalar>{values})});
  }

  auto shape = Array<Index>::copy(contracted ? std::span<const Index>{&outD, 1U}
                                             : sample.observation_shape,
                                  limits);
  if (!shape) {
    return fail<MomentEstimate>(shape.status().code(),
                                shape.status().message());
  }
  auto ownedTerms = Array<MomentTerm>::copy(terms, limits);
  if (!ownedTerms) {
    return fail<MomentEstimate>(ownedTerms.status().code(),
                                ownedTerms.status().message());
  }
  auto ownedTupleSupports = Array<TupleSupport>::copy(tupleSupports, limits);
  if (!ownedTupleSupports) {
    return fail<MomentEstimate>(ownedTupleSupports.status().code(),
                                ownedTupleSupports.status().message());
  }
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = weights.rule,
      .center = central ? center->rule : CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = global,
      .coordinate_support = std::move(coordinateSupport).value(),
      .tuple_support = std::move(ownedTupleSupports).value(),
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm =
          central ? "empirical_central_moments" : "empirical_raw_moments",
      .premises =
          "empirical population ratio; no IID correction or sampling error",
      .warnings = {}};
  std::optional<Vector> populationMean;
  if (outputMeanAvailable) {
    populationMean.emplace(std::span<const Scalar>{outputMean});
  }
  return Result<MomentEstimate>::success(
      MomentEstimate{.observation_shape = std::move(shape).value(),
                     .terms = std::move(ownedTerms).value(),
                     .diagnostics = std::move(diagnostics),
                     .population_mean = std::move(populationMean)});
}

using detail::BinaryEndpoint;
using detail::MomentCertificate;
using detail::MomentInterval;

bool checkedStorage(Extent outputValues, Extent outputIndices,
                    Extent outputRecords, Extent workspaceValues,
                    Extent workspaceIndices, Extent workspaceRecords,
                    Limits limits) {
  Extent elements = 0U;
  Extent bytes = 0U;
  auto add = [&](Extent count, Extent size) {
    Extent part = 0U;
    return checkedMultiply(count, size, part) &&
           checkedAdd(bytes, part, bytes) &&
           checkedAdd(elements, count, elements);
  };
  return add(outputValues, sizeof(Scalar)) &&
         add(outputIndices, sizeof(Index)) &&
         add(outputRecords,
             std::max({sizeof(MomentTerm), sizeof(MomentEstimate),
                       sizeof(Support)})) &&
         add(workspaceValues, sizeof(MomentInterval)) &&
         add(workspaceIndices, sizeof(Index)) &&
         add(workspaceRecords, sizeof(MomentTerm)) &&
         elements <= limits.elements && bytes <= limits.bytes;
}

bool intervalCandidate(MomentInterval range, Scalar& value) {
  const Scalar lower = range.lower().rounded();
  const Scalar upper = range.upper().rounded();
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return false;
  }
  value = std::midpoint(lower, upper);
  return std::isfinite(value);
}

Result<MomentEstimate> estimateUnbiasedFixedWeightedThird(
    const SampleView& sample, const WeightSpec& fixedWeights,
    const MomentRequest& request, Limits limits) {
  if (!MomentInterval::supportsCurrentRounding()) {
    return fail<MomentEstimate>(
        ErrorCode::IllConditioned,
        "fixed-weight third moment requires IEC 60559 binary arithmetic in "
        "round-to-nearest mode");
  }
  auto dimension = validateSample(sample);
  if (!dimension) {
    return fail<MomentEstimate>(dimension.status().code(),
                                dimension.status().message());
  }
  const Extent d = dimension.value();
  if (fixedWeights.rule != WeightRule::fixed_iid ||
      !fixedWeights.rowWeights.has_value() ||
      fixedWeights.componentWeights.has_value() ||
      !fixedWeights.frequencyCounts.empty()) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "unbiased third moment requires fixed-IID row weights only");
  }
  if (fixedWeights.rowWeights->size() != sample.rows.rows()) {
    return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                "fixed-IID row weights must match sample rows");
  }
  if (request.orders != 0x04U) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "unbiased third moment requires order three only");
  }
  const Status requestStatus = validateRequest(request, d, fixedWeights);
  if (!requestStatus.ok()) {
    return fail<MomentEstimate>(requestStatus.code(), requestStatus.message());
  }

  const bool contracted =
      request.representation == MomentRepresentation::contraction;
  const Extent outD = contracted ? request.contraction->rows() : d;
  Extent planCount = 0U;
  Extent totalValues = 0U;
  Extent tupleItems = 0U;
  const auto plans =
      makePlans(request, d, outD, planCount, totalValues, tupleItems);
  if (!plans) {
    return fail<MomentEstimate>(plans.status().code(),
                                plans.status().message());
  }
  if (planCount != 1U || plans.value()[0U].order != 3U) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "fixed-weight third-moment request must produce one third-order term");
  }
  const Extent count = plans.value()[0U].count;
  Extent selectedItems = 0U;
  if (request.representation == MomentRepresentation::selected) {
    for (const auto tuple : request.selected) {
      if (!checkedAdd(selectedItems, tuple.size(), selectedItems)) {
        return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                    "selected tuple storage overflows");
      }
    }
  }
  const Extent shapeItems = contracted ? 1U : sample.observation_shape.size();
  Extent outputValues = 0U;
  Extent outputIndices = 0U;
  const Extent outputRecords = 2U;
  Extent workspaceValues = 0U;
  Extent workspaceRecords = 0U;
  Extent workspaceIndices = tupleItems;
  if (!checkedMultiply(count, 2U, outputValues) ||
      !checkedAdd(outputValues, outD, outputValues) ||
      !checkedAdd(shapeItems, selectedItems, outputIndices) ||
      !checkedMultiply(d, 3U, workspaceValues) ||
      !checkedAdd(workspaceValues, contracted ? outD : 0U, workspaceValues) ||
      !checkedAdd(workspaceValues, outD, workspaceValues) ||
      !checkedAdd(workspaceValues, count, workspaceValues) ||
      !checkedAdd(workspaceValues, count, workspaceValues) ||
      !checkedAdd(workspaceValues, 12U, workspaceValues) ||
      !checkedAdd(workspaceIndices, selectedItems, workspaceIndices) ||
      !checkedAdd(workspaceRecords, 1U, workspaceRecords)) {
    return fail<MomentEstimate>(
        ErrorCode::SizeOverflow,
        "fixed-weight third-moment storage accounting overflows");
  }
  if (!checkedStorage(outputValues, outputIndices, outputRecords,
                      workspaceValues, workspaceIndices, workspaceRecords,
                      limits)) {
    return fail<MomentEstimate>(
        ErrorCode::ResourceLimit,
        "fixed-weight third-moment output/workspace exceeds configured limits");
  }

  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar weight = fixedWeights.rowWeights->at(row);
    if (!std::isfinite(weight)) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                  "fixed-IID weight is not finite");
    }
    if (weight < 0.0) {
      return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                  "fixed-IID weights must be nonnegative");
    }
    for (Index coordinate = 0U; coordinate < d; ++coordinate) {
      if (!std::isfinite(sample.rows.at(row, coordinate))) {
        return fail<MomentEstimate>(
            ErrorCode::NonFiniteInput,
            "fixed-weight third moment requires finite complete rows");
      }
    }
  }
  if (contracted) {
    for (Index row = 0U; row < outD; ++row) {
      for (Index column = 0U; column < d; ++column) {
        if (!std::isfinite(request.contraction->at(row, column))) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                      "contraction map is not finite");
        }
      }
    }
  }

  Index positive = 0U;
  Index originRow = 0U;
  Scalar maximumWeight = 0.0;
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar weight = fixedWeights.rowWeights->at(row);
    if (weight > 0.0) {
      ++positive;
      if (weight > maximumWeight) {
        maximumWeight = weight;
        originRow = row;
      }
    }
  }
  if (positive < 3U) {
    return fail<MomentEstimate>(
        ErrorCode::InsufficientSamples,
        "fixed-weight third moment requires at least three positive weights");
  }

  MomentInterval mass{};
  MomentInterval e1{};
  MomentInterval e2{};
  MomentInterval e3{};
  std::vector<MomentInterval> shiftedSums(d);
  std::vector<MomentInterval> shiftedMean(d);
  const BinaryEndpoint zero{};
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar weight = fixedWeights.rowWeights->at(row);
    const MomentInterval w = MomentInterval::point(weight);
    mass = mass + w;
    e3 = e3 + w * e2;
    e2 = e2 + w * e1;
    e1 = e1 + w;
    if (weight == 0.0) {
      continue;
    }
    for (Index coordinate = 0U; coordinate < d; ++coordinate) {
      const auto shifted =
          MomentInterval::point(sample.rows.at(row, coordinate)) -
          MomentInterval::point(sample.rows.at(originRow, coordinate));
      shiftedSums[coordinate] = shiftedSums[coordinate] + w * shifted;
    }
  }
  if (!(zero < mass.lower()) || !(zero < e3.lower())) {
    return fail<MomentEstimate>(
        ErrorCode::IllConditioned,
        "cannot certify a positive third-moment correction");
  }
  for (Index coordinate = 0U; coordinate < d; ++coordinate) {
    shiftedMean[coordinate] = shiftedSums[coordinate].divideByPositive(mass);
  }

  std::vector<Index> tuples(tupleItems, 0U);
  if (request.representation == MomentRepresentation::component) {
    for (Extent term = 0U; term < count; ++term) {
      std::ranges::fill(std::span<Index>{tuples}.subspan(term * 3U, 3U), term);
    }
  } else if (request.representation == MomentRepresentation::selected) {
    Extent cursor = 0U;
    for (const auto tuple : request.selected) {
      std::ranges::copy(tuple,
                        std::span<Index>{tuples}.subspan(cursor, 3U).begin());
      cursor += 3U;
    }
  } else if (request.representation == MomentRepresentation::packed) {
    std::array<Index, 3U> tuple{};
    for (Extent term = 0U; term < count; ++term) {
      std::ranges::copy(
          tuple, std::span<Index>{tuples}.subspan(term * 3U, 3U).begin());
      if (term + 1U == count) {
        continue;
      }
      Index axis = 3U;
      while (axis > 0U && tuple.at(axis - 1U) + 1U >= d) {
        --axis;
      }
      --axis;
      ++tuple.at(axis);
      std::ranges::fill(std::span<Index>{tuple}.subspan(axis + 1U),
                        tuple.at(axis));
    }
  } else {
    const Extent tupleDimension = contracted ? outD : d;
    for (Extent term = 0U; term < count; ++term) {
      Extent value = term;
      for (Index axis = 3U; axis > 0U; --axis) {
        tuples.at((term * 3U) + axis - 1U) = value % tupleDimension;
        value /= tupleDimension;
      }
    }
  }

  std::vector<MomentInterval> centered(d);
  std::vector<MomentInterval> mapped(contracted ? outD : 0U);
  std::vector<MomentInterval> numerators(count);
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar weight = fixedWeights.rowWeights->at(row);
    if (weight == 0.0) {
      continue;
    }
    for (Index coordinate = 0U; coordinate < d; ++coordinate) {
      centered[coordinate] =
          (MomentInterval::point(sample.rows.at(row, coordinate)) -
           MomentInterval::point(sample.rows.at(originRow, coordinate))) -
          shiftedMean[coordinate];
    }
    if (contracted) {
      for (Index output = 0U; output < outD; ++output) {
        MomentInterval sum{};
        for (Index coordinate = 0U; coordinate < d; ++coordinate) {
          sum = sum + MomentInterval::point(
                          request.contraction->at(output, coordinate)) *
                          centered[coordinate];
        }
        mapped[output] = sum;
      }
    }
    const MomentInterval w = MomentInterval::point(weight);
    for (Extent term = 0U; term < count; ++term) {
      MomentInterval product = MomentInterval::point(1.0);
      for (Index axis = 0U; axis < 3U; ++axis) {
        const Index coordinate = tuples.at((term * 3U) + axis);
        product =
            product * (contracted ? mapped[coordinate] : centered[coordinate]);
      }
      numerators[term] = numerators[term] + w * product;
    }
  }

  const MomentInterval divisor = MomentInterval::point(6.0) * e3;
  std::vector<Scalar> values(count, 0.0);
  Scalar maximumError = 0.0;
  for (Extent term = 0U; term < count; ++term) {
    const MomentInterval estimateRange =
        (numerators[term] * mass * mass).divideByPositive(divisor);
    Scalar error = 0.0;
    const MomentCertificate certificate =
        detail::certifyMoment(estimateRange, values[term], error);
    if (certificate == MomentCertificate::overflow) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                  "fixed-weight third-moment result is outside "
                                  "the finite Scalar range");
    }
    if (certificate != MomentCertificate::accepted) {
      return fail<MomentEstimate>(ErrorCode::IllConditioned,
                                  "fixed-weight third-moment result cannot "
                                  "meet its forward-error bound");
    }
    maximumError = std::max(maximumError, error);
  }

  std::vector<Index> flattenedValues;
  Array<Index> flattened;
  if (request.representation == MomentRepresentation::selected) {
    flattenedValues = tuples;
    auto copy = Array<Index>::copy(flattenedValues, limits);
    if (!copy) {
      return fail<MomentEstimate>(copy.status().code(),
                                  copy.status().message());
    }
    flattened = std::move(copy).value();
  }
  std::vector<MomentTerm> terms;
  terms.push_back(
      MomentTerm{.order = 3U,
                 .layout = contracted ? MomentRepresentation::dense
                                      : request.representation,
                 .flattened_tuples = std::move(flattened),
                 .values = Vector(std::span<const Scalar>{values})});
  auto ownedTerms = Array<MomentTerm>::copy(terms, limits);
  if (!ownedTerms) {
    return fail<MomentEstimate>(ownedTerms.status().code(),
                                ownedTerms.status().message());
  }

  std::vector<Scalar> outputMean(outD, 0.0);
  bool outputMeanAvailable = true;
  for (Index coordinate = 0U; coordinate < (contracted ? outD : d);
       ++coordinate) {
    MomentInterval meanRange;
    if (contracted) {
      meanRange = MomentInterval{};
      for (Index input = 0U; input < d; ++input) {
        const auto shifted =
            MomentInterval::point(sample.rows.at(originRow, input)) +
            shiftedMean[input];
        meanRange =
            meanRange +
            MomentInterval::point(request.contraction->at(coordinate, input)) *
                shifted;
      }
    } else {
      meanRange = MomentInterval::point(sample.rows.at(originRow, coordinate)) +
                  shiftedMean[coordinate];
    }
    if (!intervalCandidate(meanRange, outputMean[coordinate])) {
      outputMeanAvailable = false;
      break;
    }
  }
  auto shape = Array<Index>::copy(
      contracted ? std::span<const Index>{&outD, 1U}
                 : std::span<const Index>{sample.observation_shape},
      limits);
  if (!shape) {
    return fail<MomentEstimate>(shape.status().code(),
                                shape.status().message());
  }

  std::array<MomentInterval, 2U> powerSums{};
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    const Scalar weight = fixedWeights.rowWeights->at(row);
    if (weight == 0.0) {
      continue;
    }
    const auto normalized =
        MomentInterval::point(weight).divideByPositive(mass);
    powerSums[0U] = powerSums[0U] + normalized * normalized;
    powerSums[1U] = powerSums[1U] + normalized * normalized * normalized;
  }
  std::array<std::optional<Scalar>, 3U> powers{1.0, std::nullopt, std::nullopt};
  Scalar squaredPower = 0.0;
  Scalar cubedPower = 0.0;
  if (intervalCandidate(powerSums[0U], squaredPower)) {
    powers[1U] = squaredPower;
  }
  if (intervalCandidate(powerSums[1U], cubedPower)) {
    powers[2U] = cubedPower;
  }
  std::optional<Scalar> effectiveCount;
  if (squaredPower > 0.0 && powers[1U].has_value()) {
    const Scalar reciprocal = 1.0 / squaredPower;
    if (std::isfinite(reciprocal)) {
      effectiveCount = reciprocal;
    }
  }
  Scalar weightMass = 0.0;
  std::optional<Scalar> reportedMass;
  if (intervalCandidate(mass, weightMass)) {
    reportedMass = weightMass;
  }
  const Support support{.input = sample.rows.rows(),
                        .valid = sample.rows.rows(),
                        .positive = positive,
                        .frequency_mass = std::nullopt,
                        .weight_mass = reportedMass,
                        .normalized_weight_powers = powers,
                        .effective_count = effectiveCount,
                        .degrees_of_freedom = std::nullopt,
                        .squared_weight_mass = std::nullopt};
  auto coordinateSupport = Array<Support>::copy({}, limits);
  if (!coordinateSupport) {
    return fail<MomentEstimate>(coordinateSupport.status().code(),
                                coordinateSupport.status().message());
  }
  auto tupleSupport = Array<TupleSupport>::copy({}, limits);
  if (!tupleSupport) {
    return fail<MomentEstimate>(tupleSupport.status().code(),
                                tupleSupport.status().message());
  }
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = WeightRule::fixed_iid,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = support,
      .coordinate_support = std::move(coordinateSupport).value(),
      .tuple_support = std::move(tupleSupport).value(),
      .reciprocal_condition = std::nullopt,
      .numerical_error = maximumError,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "fixed_iid_weighted_third_central_moment_interval",
      .premises =
          "complete IID rows with deterministic nonnegative weights fixed "
          "independently of observations; no sampling covariance estimated",
      .warnings = {}};
  std::optional<Vector> populationMean;
  if (outputMeanAvailable) {
    populationMean.emplace(std::span<const Scalar>{outputMean});
  }
  return Result<MomentEstimate>::success(
      MomentEstimate{.observation_shape = std::move(shape).value(),
                     .terms = std::move(ownedTerms).value(),
                     .diagnostics = std::move(diagnostics),
                     .population_mean = std::move(populationMean)});
}

} // namespace

Result<MomentEstimate> estimateRawMoments(const SampleView& sample,
                                          const WeightSpec& weights,
                                          const MomentRequest& request,
                                          Limits limits) {
  return estimate(sample, weights, nullptr, request, limits, false);
}

Result<MomentEstimate> estimateCentralMoments(const SampleView& sample,
                                              const WeightSpec& weights,
                                              const CenterSpec& center,
                                              const MomentRequest& request,
                                              Limits limits) {
  return estimate(sample, weights, &center, request, limits, true);
}

Result<MomentEstimate>
unbiasedFixedWeightedThirdMoment(const SampleView& sample,
                                 const WeightSpec& fixedWeights,
                                 const MomentRequest& request, Limits limits) {
  return estimateUnbiasedFixedWeightedThird(sample, fixedWeights, request,
                                            limits);
}

} // namespace vss::math::statistics
