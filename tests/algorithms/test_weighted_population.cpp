#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
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

SampleView sample(const Matrix& values,
                  std::span<const std::uint8_t> valid = {}) {
  static constexpr std::array<Extent, 1U> vectorShape{2U};
  return {.rows = values.view(),
          .observation_shape = values.columns() == 1U
                                   ? std::span<const Index>{}
                                   : std::span<const Index>{vectorShape},
          .valid = valid};
}

Limits limits() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = 0U};
}

void close(Scalar actual, Scalar expected) {
  if (!(std::abs(actual - expected) <=
        1.0e-12 + (1.0e-10 * std::max(std::abs(actual), std::abs(expected))))) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    VSS_CHECK(false);
  }
}

WeightSpec rowWeights(const Vector& values) {
  return {.rule = WeightRule::empirical,
          .rowWeights = values.view(),
          .componentWeights = std::nullopt,
          .frequencyCounts = {}};
}

void weightedScalarMeanAndEmpiricalVariance() {
  const Matrix values = matrix({{0.0}, {3.0}});
  const Vector weights = vector({1.0, 2.0});
  const auto estimate =
      weightedMean(sample(values), rowWeights(weights), limits());
  VSS_CHECK(estimate.hasValue());
  close(estimate.value().value.at(0U), 2.0);
  VSS_CHECK(estimate.value().diagnostics.weight == WeightRule::empirical);
  VSS_CHECK(estimate.value().diagnostics.global.positive == 2U);
  const auto effectiveCount =
      estimate.value().diagnostics.global.effective_count;
  VSS_CHECK(effectiveCount.has_value());
  close(effectiveCount.value_or(-1.0), 1.8);

  const auto covariance = weightedEmpiricalCovariance(
      sample(values), rowWeights(weights), limits());
  VSS_CHECK(covariance.hasValue());
  close(covariance.value().value.at(0U, 0U), 2.0);
  const auto weightMass = covariance.value().diagnostics.global.weight_mass;
  VSS_CHECK(weightMass.has_value());
  close(weightMass.value_or(-1.0), 3.0);
}

void rowWeightScalingAndVectorCovariance() {
  const Matrix values = matrix({{1.0, 2.0}, {3.0, 6.0}, {5.0, 10.0}});
  const Vector weights = vector({2.0, 4.0, 6.0});
  const Vector scaledWeights = vector({2.0e200, 4.0e200, 6.0e200});
  const auto mean = weightedMean(sample(values), rowWeights(weights), limits());
  const auto scaledMean =
      weightedMean(sample(values), rowWeights(scaledWeights), limits());
  VSS_CHECK(mean && scaledMean);
  for (Index i = 0U; i < 2U; ++i) {
    close(mean.value().value.at(i), scaledMean.value().value.at(i));
  }
  const auto covariance = weightedEmpiricalCovariance(
      sample(values), rowWeights(weights), limits());
  const auto scaledCovariance = weightedEmpiricalCovariance(
      sample(values), rowWeights(scaledWeights), limits());
  VSS_CHECK(covariance && scaledCovariance);
  for (Index row = 0U; row < 2U; ++row) {
    for (Index column = 0U; column < 2U; ++column) {
      close(covariance.value().value.at(row, column),
            scaledCovariance.value().value.at(row, column));
    }
  }
  close(covariance.value().value.at(0U, 1U), 40.0 / 9.0);
}

void componentWeightsAndPerCoordinateSupport() {
  const Matrix values = matrix({{1.0, 10.0}, {3.0, 20.0}, {5.0, 30.0}});
  const Matrix weights = matrix({{1.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}});
  const WeightSpec component{.rule = WeightRule::componentwise,
                             .rowWeights = std::nullopt,
                             .componentWeights = weights.view(),
                             .frequencyCounts = {}};
  const auto estimate = weightedMean(sample(values), component, limits());
  VSS_CHECK(estimate);
  close(estimate.value().value.at(0U), 1.0);
  close(estimate.value().value.at(1U), 20.0);
  VSS_CHECK(estimate.value().diagnostics.coordinate_support.view().size() ==
            2U);
  VSS_CHECK(
      estimate.value().diagnostics.coordinate_support.view()[0].positive == 1U);
  VSS_CHECK(
      estimate.value().diagnostics.coordinate_support.view()[1].positive == 1U);
  const auto unsupported =
      weightedEmpiricalCovariance(sample(values), component, limits());
  VSS_CHECK(!unsupported);
  VSS_CHECK(unsupported.status().code() == ErrorCode::InvalidDomain);
}

void masksAndInvalidWeights() {
  Matrix values = matrix({{1.0, 10.0}, {3.0, 20.0}, {5.0, 30.0}});
  values.at(1U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  const std::array<std::uint8_t, 6U> mask{1U, 1U, 0U, 1U, 1U, 1U};
  const Matrix weights = matrix({{1.0, 1.0}, {4.0, 1.0}, {1.0, 2.0}});
  const WeightSpec component{.rule = WeightRule::componentwise,
                             .rowWeights = std::nullopt,
                             .componentWeights = weights.view(),
                             .frequencyCounts = {}};
  const auto mean = weightedMean(sample(values, mask), component, limits());
  VSS_CHECK(mean);
  close(mean.value().value.at(0U), 3.0);
  close(mean.value().value.at(1U), 22.5);

  const Vector negative = vector({1.0, -1.0, 1.0});
  const Matrix scalarValues = matrix({{1.0}, {2.0}, {3.0}});
  const auto negativeResult =
      weightedMean(sample(scalarValues), rowWeights(negative), limits());
  VSS_CHECK(!negativeResult);
  VSS_CHECK(negativeResult.status().code() == ErrorCode::InvalidDomain);
  const Vector zero = vector({0.0, 0.0, 0.0});
  const auto zeroResult =
      weightedMean(sample(scalarValues), rowWeights(zero), limits());
  VSS_CHECK(!zeroResult);
  VSS_CHECK(zeroResult.status().code() == ErrorCode::InsufficientSamples);
}

void rejectsUnsupportedRulesAndLimits() {
  const Matrix values = matrix({{1.0}, {2.0}});
  const Vector weights = vector({1.0, 1.0});
  WeightSpec unsupported = rowWeights(weights);
  unsupported.rule = WeightRule::fixed_iid;
  const auto unsupportedResult =
      weightedMean(sample(values), unsupported, limits());
  VSS_CHECK(!unsupportedResult);
  VSS_CHECK(unsupportedResult.status().code() == ErrorCode::InvalidDomain);

  const WeightSpec empirical = rowWeights(weights);
  const auto limited =
      weightedMean(sample(values), empirical,
                   {.elements = 0U, .bytes = 0U, .evaluations = 0U});
  VSS_CHECK(!limited);
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
  const auto limitedCov = weightedEmpiricalCovariance(
      sample(values), empirical,
      {.elements = 1U, .bytes = 1U, .evaluations = 0U});
  VSS_CHECK(!limitedCov);
  VSS_CHECK(limitedCov.status().code() == ErrorCode::ResourceLimit);
}

void covarianceUsesCompleteRowsAndRejectsNonFiniteValues() {
  Matrix values = matrix({{0.0}, {3.0}, {6.0}});
  values.at(1U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  const std::array<std::uint8_t, 3U> mask{1U, 0U, 1U};
  const Vector weights = vector({1.0, 100.0, 1.0});
  const auto covariance = weightedEmpiricalCovariance(
      sample(values, mask), rowWeights(weights), limits());
  VSS_CHECK(covariance);
  close(covariance.value().value.at(0U, 0U), 9.0);
  VSS_CHECK(covariance.value().diagnostics.global.valid == 2U);
  VSS_CHECK(covariance.value().diagnostics.global.positive == 2U);

  const Matrix finite = matrix({{0.0}, {1.0}});
  const Vector invalidWeights =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  const auto invalid =
      weightedMean(sample(finite), rowWeights(invalidWeights), limits());
  VSS_CHECK(!invalid);
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);
}

void rejectsMalformedInputsAndUnrepresentableCovariance() {
  const Matrix values = matrix({{1.0}, {2.0}});
  const Vector oneWeight = vector({1.0});
  const auto wrongRows =
      weightedMean(sample(values), rowWeights(oneWeight), limits());
  VSS_CHECK(!wrongRows);
  VSS_CHECK(wrongRows.status().code() == ErrorCode::DimensionMismatch);

  const Matrix wrongComponentShape = matrix({{1.0, 2.0}, {3.0, 4.0}});
  const WeightSpec wrongComponents{.rule = WeightRule::componentwise,
                                   .rowWeights = std::nullopt,
                                   .componentWeights =
                                       wrongComponentShape.view(),
                                   .frequencyCounts = {}};
  const auto wrongShape =
      weightedMean(sample(values), wrongComponents, limits());
  VSS_CHECK(!wrongShape);
  VSS_CHECK(wrongShape.status().code() == ErrorCode::DimensionMismatch);

  const std::array<std::uint8_t, 1U> shortMask{1U};
  const Vector weights = vector({1.0, 1.0});
  const auto malformedMask =
      weightedMean(sample(values, shortMask), rowWeights(weights), limits());
  VSS_CHECK(!malformedMask);
  VSS_CHECK(malformedMask.status().code() == ErrorCode::DimensionMismatch);

  const Matrix extreme = matrix({{std::numeric_limits<Scalar>::max()},
                                 {-std::numeric_limits<Scalar>::max()}});
  const auto overflow = weightedEmpiricalCovariance(
      sample(extreme), rowWeights(weights), limits());
  VSS_CHECK(!overflow);
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  weightedScalarMeanAndEmpiricalVariance();
  rowWeightScalingAndVectorCovariance();
  componentWeightsAndPerCoordinateSupport();
  masksAndInvalidWeights();
  rejectsUnsupportedRulesAndLimits();
  covarianceUsesCompleteRowsAndRejectsNonFiniteValues();
  rejectsMalformedInputsAndUnrepresentableCovariance();
  std::cout << "weighted population estimators passed\n";
}
