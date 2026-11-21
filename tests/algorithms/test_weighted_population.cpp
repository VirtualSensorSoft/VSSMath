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

WeightSpec fixedWeights(const Vector& values) {
  return {.rule = WeightRule::fixed_iid,
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

void fixedWeightUnbiasedCovariance() {
  const Matrix values = matrix({{0.0}, {3.0}});
  const Vector weights = vector({1.0, 2.0});
  const Vector scaledWeights = vector({1.0e200, 2.0e200});
  const auto estimate = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(weights), limits());
  const auto scaled = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(scaledWeights), limits());
  VSS_CHECK(estimate && scaled);
  close(estimate.value().value.at(0U, 0U), 4.5);
  close(scaled.value().value.at(0U, 0U), 4.5);
  VSS_CHECK(estimate.value().diagnostics.weight == WeightRule::fixed_iid);

  const Matrix equalValues = matrix({{0.0}, {2.0}, {4.0}});
  const Vector equalWeights = vector({1.0, 1.0, 1.0});
  const auto equal = unbiasedFixedWeightedCovariance(
      sample(equalValues), fixedWeights(equalWeights), limits());
  VSS_CHECK(equal);
  close(equal.value().value.at(0U, 0U), 4.0);

  const Matrix vectorValues = matrix({{0.0, 0.0}, {1.0, 2.0}, {2.0, 4.0}});
  const Vector vectorWeights = vector({1.0, 2.0, 1.0});
  const auto vectorEstimate = unbiasedFixedWeightedCovariance(
      sample(vectorValues), fixedWeights(vectorWeights), limits());
  VSS_CHECK(vectorEstimate);
  close(vectorEstimate.value().value.at(0U, 0U), 0.8);
  close(vectorEstimate.value().value.at(0U, 1U), 1.6);
  close(vectorEstimate.value().value.at(1U, 1U), 3.2);
}

void fixedWeightCovarianceRejectsInvalidInputs() {
  const Matrix values = matrix({{0.0}, {3.0}});
  const Vector empirical = vector({1.0, 2.0});
  const auto wrongRule = unbiasedFixedWeightedCovariance(
      sample(values), rowWeights(empirical), limits());
  VSS_CHECK(!wrongRule);
  VSS_CHECK(wrongRule.status().code() == ErrorCode::InvalidDomain);

  const Vector onePositive = vector({1.0, 0.0});
  const auto insufficient = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(onePositive), limits());
  VSS_CHECK(!insufficient);
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);

  const Vector invalid = vector({1.0, -1.0});
  const auto negative = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(invalid), limits());
  VSS_CHECK(!negative);
  VSS_CHECK(negative.status().code() == ErrorCode::InvalidDomain);

  const Vector singular = vector({1.0, std::numeric_limits<Scalar>::min()});
  const auto nearSingular = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(singular), limits());
  VSS_CHECK(!nearSingular);
  VSS_CHECK(nearSingular.status().code() == ErrorCode::InvalidDomain);

  const Vector weights = vector({1.0, 1.0});
  const auto limited = unbiasedFixedWeightedCovariance(
      sample(values), fixedWeights(weights),
      {.elements = 1U, .bytes = 1U, .evaluations = 0U});
  VSS_CHECK(!limited);
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
}

void weightedMeanAndLinearEstimateCovariance() {
  const Matrix values = matrix({{1.0}, {2.0}, {4.0}, {8.0}, {16.0}});
  const Vector weights = vector({0.1, 0.15, 0.2, 0.25, 0.3});
  const auto estimate =
      weightedMeanCovariance(sample(values), fixedWeights(weights), limits());
  VSS_CHECK(estimate);
  close(estimate.value().value.at(0U, 0U), 2943.0 / 310.0);
  VSS_CHECK(estimate.value().diagnostics.weight == WeightRule::fixed_iid);
  VSS_CHECK(estimate.value().diagnostics.estimand == Estimand::estimate);

  const Matrix averageMap = matrix({{0.5, 0.5}});
  const Matrix correlated = matrix({{0.29, 0.25}, {0.25, 0.29}});
  const auto propagated =
      linearEstimateCovariance(averageMap.view(), correlated.view(), limits());
  VSS_CHECK(propagated);
  close(propagated.value().at(0U, 0U), 0.27);

  const Matrix incompatible = matrix({{1.0, 0.0, 0.0}});
  const auto dimensionError = linearEstimateCovariance(
      incompatible.view(), correlated.view(), limits());
  VSS_CHECK(!dimensionError);
  VSS_CHECK(dimensionError.status().code() == ErrorCode::DimensionMismatch);

  const Matrix indefinite = matrix({{1.0, 2.0}, {2.0, 1.0}});
  const auto invalidCovariance =
      linearEstimateCovariance(averageMap.view(), indefinite.view(), limits());
  VSS_CHECK(!invalidCovariance);
  VSS_CHECK(invalidCovariance.status().code() ==
            ErrorCode::NotPositiveSemidefinite);

  const auto limited = linearEstimateCovariance(
      averageMap.view(), correlated.view(),
      {.elements = 0U, .bytes = 0U, .evaluations = 0U});
  VSS_CHECK(!limited);
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
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

void frequencyWeightedCovarianceMatchesReplicatedObservations() {
  const Matrix compressedValues = matrix({{1.0}, {4.0}});
  const std::array<std::uint64_t, 2U> counts{2U, 1U};
  const auto compressedSample = sample(compressedValues);
  const auto empirical = frequencyWeightedCovariance(
      compressedSample, counts, VarianceMode::empirical, limits());
  const auto unbiased = frequencyWeightedCovariance(
      compressedSample, counts, VarianceMode::iid_sample, limits());
  VSS_CHECK(empirical && unbiased);
  close(empirical.value().value.at(0U, 0U), 2.0);
  close(unbiased.value().value.at(0U, 0U), 3.0);
  VSS_CHECK(unbiased.value().diagnostics.weight == WeightRule::frequency);
  VSS_CHECK(unbiased.value().diagnostics.global.frequency_mass == 3U);
  VSS_CHECK(unbiased.value().diagnostics.global.positive == 2U);

  const Matrix replicatedValues = matrix({{1.0}, {1.0}, {4.0}});
  const Vector replicatedWeights = vector({1.0, 1.0, 1.0});
  const auto replicated = unbiasedFixedWeightedCovariance(
      sample(replicatedValues), fixedWeights(replicatedWeights), limits());
  VSS_CHECK(replicated);
  close(unbiased.value().value.at(0U, 0U), replicated.value().value.at(0U, 0U));
}

void frequencyWeightedCovariancePreservesExactCountMass() {
  const Matrix values = matrix({{1.0}, {4.0}});
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::uint64_t, 2U> maximumCounts{maximum - 1U, 1U};
  const auto estimate = frequencyWeightedCovariance(
      sample(values), maximumCounts, VarianceMode::iid_sample, limits());
  VSS_CHECK(estimate);
  VSS_CHECK(estimate.value().diagnostics.global.frequency_mass == maximum);
  close(estimate.value().value.at(0U, 0U) * static_cast<Scalar>(maximum), 9.0);

  const std::array<std::uint64_t, 2U> overflowingCounts{maximum, 1U};
  const auto overflow = frequencyWeightedCovariance(
      sample(values), overflowingCounts, VarianceMode::iid_sample, limits());
  VSS_CHECK(!overflow);
  VSS_CHECK(overflow.status().code() == ErrorCode::SizeOverflow);
}

void frequencyWeightedCovarianceHandlesVectorObservations() {
  const Matrix values = matrix({{0.0, 0.0}, {2.0, 4.0}});
  const std::array<std::uint64_t, 2U> counts{2U, 1U};
  const auto estimate = frequencyWeightedCovariance(
      sample(values), counts, VarianceMode::iid_sample, limits());
  VSS_CHECK(estimate);
  close(estimate.value().value.at(0U, 0U), 4.0 / 3.0);
  close(estimate.value().value.at(0U, 1U), 8.0 / 3.0);
  close(estimate.value().value.at(1U, 1U), 16.0 / 3.0);
}

void frequencyWeightedCovarianceRejectsInvalidSupportAndMode() {
  const Matrix values = matrix({{1.0}, {4.0}});
  const std::array<std::uint64_t, 2U> counts{1U, 1U};
  const auto knownCenter = frequencyWeightedCovariance(
      sample(values), counts, VarianceMode::known_center, limits());
  VSS_CHECK(!knownCenter);
  VSS_CHECK(knownCenter.status().code() == ErrorCode::InvalidDomain);

  const std::array<std::uint64_t, 1U> wrongCountLength{1U};
  const auto wrongLength = frequencyWeightedCovariance(
      sample(values), wrongCountLength, VarianceMode::empirical, limits());
  VSS_CHECK(!wrongLength);
  VSS_CHECK(wrongLength.status().code() == ErrorCode::DimensionMismatch);

  const std::array<std::uint64_t, 2U> noReplication{1U, 0U};
  const auto insufficient = frequencyWeightedCovariance(
      sample(values), noReplication, VarianceMode::iid_sample, limits());
  VSS_CHECK(!insufficient);
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);

  const auto limited = frequencyWeightedCovariance(
      sample(values), counts, VarianceMode::empirical,
      {.elements = 0U, .bytes = 0U, .evaluations = 0U});
  VSS_CHECK(!limited);
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  weightedScalarMeanAndEmpiricalVariance();
  rowWeightScalingAndVectorCovariance();
  componentWeightsAndPerCoordinateSupport();
  masksAndInvalidWeights();
  rejectsUnsupportedRulesAndLimits();
  covarianceUsesCompleteRowsAndRejectsNonFiniteValues();
  fixedWeightUnbiasedCovariance();
  fixedWeightCovarianceRejectsInvalidInputs();
  weightedMeanAndLinearEstimateCovariance();
  rejectsMalformedInputsAndUnrepresentableCovariance();
  frequencyWeightedCovarianceMatchesReplicatedObservations();
  frequencyWeightedCovariancePreservesExactCountMass();
  frequencyWeightedCovarianceHandlesVectorObservations();
  frequencyWeightedCovarianceRejectsInvalidSupportAndMode();
  std::cout << "weighted population estimators passed\n";
}
