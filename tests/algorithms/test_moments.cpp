#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/TensorViews.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Extent row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Extent column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

Vector vector(std::initializer_list<Scalar> values) {
  const std::vector<Scalar> storage(values);
  return Vector(std::span<const Scalar>{storage});
}

SampleView sample(const Matrix& values, std::span<const Index> shape = {},
                  std::span<const std::uint8_t> valid = {}) {
  return {.rows = values.view(), .observation_shape = shape, .valid = valid};
}

Limits ampleLimits() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = 0U};
}

WeightSpec unweighted() {
  return {.rule = WeightRule::unweighted,
          .rowWeights = std::nullopt,
          .componentWeights = std::nullopt,
          .frequencyCounts = {}};
}

WeightSpec rowWeights(const Vector& weights) {
  return {.rule = WeightRule::empirical,
          .rowWeights = weights.view(),
          .componentWeights = std::nullopt,
          .frequencyCounts = {}};
}

MomentRequest request(std::uint8_t orders,
                      MomentRepresentation representation) {
  return {.orders = orders,
          .representation = representation,
          .selected = {},
          .contraction = std::nullopt};
}

void close(Scalar actual, Scalar expected) {
  if (!(std::abs(actual - expected) <=
        1.0e-12 + (1.0e-10 * std::max(std::abs(actual), std::abs(expected))))) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    VSS_CHECK(false);
  }
}

void scalarRawAndCentralOrders() {
  const Matrix values = matrix({{1.0}, {2.0}, {4.0}});
  const auto raw = estimateRawMoments(
      sample(values), unweighted(), request(0x0fU, MomentRepresentation::dense),
      ampleLimits());
  VSS_CHECK(raw);
  VSS_CHECK(raw.value().observation_shape.view().empty());
  VSS_CHECK(raw.value().terms.view().size() == 4U);
  close(raw.value().terms.view()[0].values.at(0U), 7.0 / 3.0);
  close(raw.value().terms.view()[1].values.at(0U), 7.0);
  close(raw.value().terms.view()[2].values.at(0U), 73.0 / 3.0);
  close(raw.value().terms.view()[3].values.at(0U), 91.0);
  const auto& populationMean = raw.value().population_mean;
  if (populationMean.has_value()) {
    close(populationMean.value().at(0U), 7.0 / 3.0);
  } else {
    VSS_CHECK(false);
  }

  const auto central = estimateCentralMoments(
      sample(values), unweighted(), CenterSpec{},
      request(0x0fU, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(central);
  close(central.value().terms.view()[0].values.at(0U), 0.0);
  close(central.value().terms.view()[1].values.at(0U), 14.0 / 9.0);
  close(central.value().terms.view()[2].values.at(0U), 20.0 / 27.0);
  close(central.value().terms.view()[3].values.at(0U), 98.0 / 27.0);
}

void layoutsAndOrdering() {
  const Matrix values = matrix(
      {{-2.0, 3.0, -5.0, -7.0}, {0.0, 0.0, 0.0, 0.0}, {2.0, -3.0, 5.0, 7.0}});
  const std::array<Index, 1> shape{4U};
  const CenterSpec center{};
  const auto dense = estimateCentralMoments(
      sample(values, shape), unweighted(), center,
      request(0x08U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(dense);
  close(dense.value().terms.view()[0].values.at((3U * 64U) + (1U * 16U) +
                                                (1U * 4U)),
        84.0);
  VSS_CHECK(dense.value().terms.view()[0].flattened_tuples.view().empty());

  const std::array<Index, 4> tuple{0U, 1U, 1U, 3U};
  const std::array<IndexList, 1> selected{IndexList{tuple}};
  auto selectedRequest = request(0x08U, MomentRepresentation::selected);
  selectedRequest.selected = selected;
  const auto selectedResult =
      estimateCentralMoments(sample(values, shape), unweighted(), center,
                             selectedRequest, ampleLimits());
  VSS_CHECK(selectedResult);
  close(selectedResult.value().terms.view()[0].values.at(0U), 84.0);
  VSS_CHECK(
      selectedResult.value().terms.view()[0].flattened_tuples.view().size() ==
      tuple.size());

  const auto packed = estimateCentralMoments(
      sample(values, shape), unweighted(), center,
      request(0x08U, MomentRepresentation::packed), ampleLimits());
  VSS_CHECK(packed);
  close(packed.value().terms.view()[0].values.at(12U), 84.0);

  const auto component = estimateCentralMoments(
      sample(values, shape), unweighted(), center,
      request(0x08U, MomentRepresentation::component), ampleLimits());
  VSS_CHECK(component);
  VSS_CHECK(component.value().terms.view()[0].values.size() == 4U);
}

void shapesAndDistinctThirdMoment() {
  const Matrix points = matrix(
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const std::array<Index, 1> vectorShape{3U};
  const std::array<Index, 3> tuple{0U, 1U, 2U};
  const std::array<IndexList, 1> selected{IndexList{tuple}};
  auto selectedRequest = request(0x04U, MomentRepresentation::selected);
  selectedRequest.selected = selected;
  const auto result =
      estimateCentralMoments(sample(points, vectorShape), unweighted(),
                             CenterSpec{}, selectedRequest, ampleLimits());
  VSS_CHECK(result);
  close(result.value().terms.view()[0].values.at(0U), 1.0 / 32.0);

  const Matrix shapeValues = matrix(
      {{-2.0, 3.0, -5.0, -7.0}, {0.0, 0.0, 0.0, 0.0}, {2.0, -3.0, 5.0, 7.0}});
  const std::array<Index, 4> fourthTuple{0U, 1U, 1U, 3U};
  const std::array<IndexList, 1> fourthSelection{IndexList{fourthTuple}};
  auto fourthRequest = request(0x08U, MomentRepresentation::selected);
  fourthRequest.selected = fourthSelection;
  const std::array<std::array<Index, 4>, 4> shapes{
      std::array<Index, 4>{4U, 0U, 0U, 0U},
      std::array<Index, 4>{2U, 2U, 0U, 0U},
      std::array<Index, 4>{2U, 1U, 1U, 2U},
      std::array<Index, 4>{1U, 2U, 2U, 0U}};
  const std::array<Extent, 4> shapeRanks{1U, 2U, 4U, 3U};
  for (Index i = 0U; i < shapes.size(); ++i) {
    const std::span<const Index> shape{shapes.at(i).data(), shapeRanks.at(i)};
    const auto fourth =
        estimateCentralMoments(sample(shapeValues, shape), unweighted(),
                               CenterSpec{}, fourthRequest, ampleLimits());
    VSS_CHECK(fourth);
    close(fourth.value().terms.view()[0].values.at(0U), 84.0);
  }

  const Matrix matrixValues = matrix(
      {{1.0, 2.0, 3.0, 4.0}, {2.0, 3.0, 4.0, 5.0}, {3.0, 4.0, 5.0, 6.0}});
  const std::array<Index, 2> matrixShape{2U, 2U};
  const auto matrixResult = estimateRawMoments(
      sample(matrixValues, matrixShape), unweighted(),
      request(0x01U, MomentRepresentation::component), ampleLimits());
  VSS_CHECK(matrixResult);
  VSS_CHECK(matrixResult.value().observation_shape.view().size() == 2U);
  VSS_CHECK(matrixResult.value().observation_shape.view()[0] == 2U);
  VSS_CHECK(matrixResult.value().observation_shape.view()[1] == 2U);

  const Matrix crossValues = matrix(
      {{-2.0, 3.0, -5.0, -7.0}, {0.0, 0.0, 0.0, 0.0}, {2.0, -3.0, 5.0, 7.0}});
  const auto cross = estimateCentralMoments(
      sample(crossValues, matrixShape), unweighted(), CenterSpec{},
      request(0x02U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(cross);
  close(cross.value().terms.view()[0].values.at(6U), -10.0);
}

void weightedContractionAndKnownCenter() {
  const Matrix values = matrix({{1.0, 2.0}, {3.0, 0.0}});
  const Vector weights = vector({1.0, 3.0});
  const Matrix map = matrix({{2.0, -1.0}});
  auto contracted = request(0x0fU, MomentRepresentation::contraction);
  contracted.contraction = map.view();
  const auto raw =
      estimateRawMoments(sample(values, std::array<Index, 1>{2U}),
                         rowWeights(weights), contracted, ampleLimits());
  VSS_CHECK(raw);
  VSS_CHECK(raw.value().observation_shape.view().size() == 1U);
  VSS_CHECK(raw.value().observation_shape.view()[0] == 1U);
  close(raw.value().terms.view()[0].values.at(0U), 9.0 / 2.0);
  close(raw.value().terms.view()[3].values.at(0U), 972.0);

  const auto central = estimateCentralMoments(
      sample(values, std::array<Index, 1>{2U}), rowWeights(weights),
      CenterSpec{}, contracted, ampleLimits());
  VSS_CHECK(central);
  close(central.value().terms.view()[0].values.at(0U), 0.0);
  close(central.value().terms.view()[1].values.at(0U), 27.0 / 4.0);
  close(central.value().terms.view()[2].values.at(0U), -81.0 / 4.0);
  close(central.value().terms.view()[3].values.at(0U), 1701.0 / 16.0);

  const Vector knownValue = vector({0.0, 0.0});
  const CenterSpec known{.rule = CenterRule::known,
                         .center = knownValue.view()};
  const auto aroundKnown = estimateCentralMoments(
      sample(values, std::array<Index, 1>{2U}), rowWeights(weights), known,
      contracted, ampleLimits());
  VSS_CHECK(aroundKnown);
  close(aroundKnown.value().terms.view()[0].values.at(0U), 9.0 / 2.0);

  const Matrix scalarValues = matrix({{-1.0}, {0.0}, {1.0}});
  const Vector ordinaryWeights = vector({1.0, 2.0, 1.0});
  const Vector scaledWeights = vector({1.0e200, 2.0e200, 1.0e200});
  const auto ordinary = estimateCentralMoments(
      sample(scalarValues), rowWeights(ordinaryWeights), CenterSpec{},
      request(0x02U, MomentRepresentation::dense), ampleLimits());
  const auto scaled = estimateCentralMoments(
      sample(scalarValues), rowWeights(scaledWeights), CenterSpec{},
      request(0x02U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(ordinary && scaled);
  close(ordinary.value().terms.view()[0].values.at(0U), 0.5);
  close(scaled.value().terms.view()[0].values.at(0U), 0.5);

  const Matrix largeValues = matrix({{2.0, 1.0}});
  const Matrix largeMap = matrix({{1.0e308, -1.0e308}});
  auto largeRequest = request(0x01U, MomentRepresentation::contraction);
  largeRequest.contraction = largeMap.view();
  const auto scaledContraction =
      estimateRawMoments(sample(largeValues, std::array<Index, 1>{2U}),
                         unweighted(), largeRequest, ampleLimits());
  VSS_CHECK(scaledContraction);
  close(scaledContraction.value().terms.view()[0].values.at(0U), 1.0e308);
}

void supportAndValidation() {
  const Matrix values = matrix({{1.0, 10.0}, {3.0, 20.0}, {5.0, 30.0}});
  const Matrix component = matrix({{1.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}});
  const WeightSpec componentWeights{.rule = WeightRule::componentwise,
                                    .rowWeights = std::nullopt,
                                    .componentWeights = component.view(),
                                    .frequencyCounts = {}};
  const auto diagonal = estimateRawMoments(
      sample(values, std::array<Index, 1>{2U}), componentWeights,
      request(0x01U, MomentRepresentation::component), ampleLimits());
  VSS_CHECK(diagonal);
  close(diagonal.value().terms.view()[0].values.at(0U), 1.0);
  close(diagonal.value().terms.view()[0].values.at(1U), 20.0);
  VSS_CHECK(diagonal.value().diagnostics.coordinate_support.view().size() ==
            2U);
  const auto& firstSupport =
      diagonal.value().diagnostics.coordinate_support.view()[0U];
  if (firstSupport.weight_mass.has_value()) {
    close(firstSupport.weight_mass.value(), 1.0);
  } else {
    VSS_CHECK(false);
  }
  if (firstSupport.effective_count.has_value()) {
    close(firstSupport.effective_count.value(), 1.0);
  } else {
    VSS_CHECK(false);
  }

  const std::array<Index, 2> mixedTuple{0U, 1U};
  const std::array<IndexList, 1> mixedSelected{IndexList{mixedTuple}};
  auto mixedRequest = request(0x02U, MomentRepresentation::selected);
  mixedRequest.selected = mixedSelected;
  const auto mixed =
      estimateRawMoments(sample(values, std::array<Index, 1>{2U}),
                         componentWeights, mixedRequest, ampleLimits());
  VSS_CHECK(!mixed);
  VSS_CHECK(mixed.status().code() == ErrorCode::InvalidDomain);

  const auto invalidOrders = estimateRawMoments(
      sample(values, std::array<Index, 1>{2U}), unweighted(),
      request(0U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(!invalidOrders);
  VSS_CHECK(invalidOrders.status().code() == ErrorCode::InvalidDomain);

  const std::array<Index, 1> wrongTuple{0U};
  const std::array<IndexList, 1> wrongSelected{IndexList{wrongTuple}};
  auto wrongRequest = request(0x04U, MomentRepresentation::selected);
  wrongRequest.selected = wrongSelected;
  const auto wrongLength = estimateCentralMoments(
      sample(values, std::array<Index, 1>{2U}), unweighted(), CenterSpec{},
      wrongRequest, ampleLimits());
  VSS_CHECK(!wrongLength);
  VSS_CHECK(wrongLength.status().code() == ErrorCode::DimensionMismatch);

  const std::array<Index, 2> outOfBounds{0U, 2U};
  const std::array<IndexList, 1> outSelected{IndexList{outOfBounds}};
  wrongRequest.orders = 0x02U;
  wrongRequest.selected = outSelected;
  const auto outOfRange = estimateCentralMoments(
      sample(values, std::array<Index, 1>{2U}), unweighted(), CenterSpec{},
      wrongRequest, ampleLimits());
  VSS_CHECK(!outOfRange);
  VSS_CHECK(outOfRange.status().code() == ErrorCode::IndexOutOfRange);

  const Matrix tooSmall = matrix({{1.0}, {2.0}, {3.0}});
  const auto limited = estimateRawMoments(
      sample(tooSmall), unweighted(),
      request(0x08U, MomentRepresentation::dense),
      Limits{.elements = 1U, .bytes = 8U, .evaluations = 0U});
  VSS_CHECK(!limited);
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);

  const Vector zeroWeights = vector({0.0, 0.0, 0.0});
  const auto noSupport = estimateRawMoments(
      sample(tooSmall), rowWeights(zeroWeights),
      request(0x01U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(!noSupport);
  VSS_CHECK(noSupport.status().code() == ErrorCode::InsufficientSamples);

  const Vector negativeWeights = vector({1.0, -1.0, 1.0});
  const auto negative = estimateRawMoments(
      sample(tooSmall), rowWeights(negativeWeights),
      request(0x01U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(!negative);
  VSS_CHECK(negative.status().code() == ErrorCode::InvalidDomain);

  const Vector nonfiniteWeights =
      vector({1.0, std::numeric_limits<Scalar>::infinity(), 1.0});
  const auto nonfiniteWeight = estimateRawMoments(
      sample(tooSmall), rowWeights(nonfiniteWeights),
      request(0x01U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(!nonfiniteWeight);
  VSS_CHECK(nonfiniteWeight.status().code() == ErrorCode::NonFiniteInput);

  const Matrix inactiveNan =
      matrix({{1.0}, {std::numeric_limits<Scalar>::quiet_NaN()}, {3.0}});
  const Vector inactiveWeight = vector({1.0, 0.0, 1.0});
  const auto inactiveValue = estimateRawMoments(
      sample(inactiveNan), rowWeights(inactiveWeight),
      request(0x01U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(inactiveValue);
  close(inactiveValue.value().terms.view()[0].values.at(0U), 2.0);

  const std::array<std::uint8_t, 3U> incomplete{1U, 0U, 1U};
  const auto masked = estimateRawMoments(
      sample(tooSmall, {}, incomplete), unweighted(),
      request(0x01U, MomentRepresentation::dense), ampleLimits());
  VSS_CHECK(!masked);
  VSS_CHECK(masked.status().code() == ErrorCode::InvalidDomain);

  const Matrix huge = matrix({{1.0e100}});
  const auto overflow = estimateRawMoments(
      sample(huge), unweighted(), request(0x08U, MomentRepresentation::dense),
      ampleLimits());
  VSS_CHECK(!overflow);
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() try {
  scalarRawAndCentralOrders();
  layoutsAndOrdering();
  shapesAndDistinctThirdMoment();
  weightedContractionAndKnownCenter();
  supportAndValidation();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
