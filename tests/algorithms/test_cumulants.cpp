#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Cumulants.hpp>
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

SampleView sample(const Matrix& values, std::span<const Index> shape = {}) {
  return {.rows = values.view(), .observation_shape = shape, .valid = {}};
}

Limits ample() {
  return {.elements = std::numeric_limits<Extent>::max(),
          .bytes = std::numeric_limits<Extent>::max(),
          .evaluations = 0U};
}

WeightSpec unweighted() {
  return {.rule = WeightRule::unweighted,
          .rowWeights = std::nullopt,
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
  const Scalar tolerance =
      1.0e-12 + 1.0e-10 * std::max(std::abs(actual), std::abs(expected));
  if (!(std::abs(actual - expected) <= tolerance)) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    VSS_CHECK(false);
  }
}

MomentEstimate central(const Matrix& values, std::uint8_t orders,
                       MomentRepresentation representation) {
  const std::array<Index, 1U> vectorShape{values.columns()};
  const SampleView input =
      values.columns() == 1U ? sample(values) : sample(values, vectorShape);
  const auto result =
      estimateCentralMoments(input, unweighted(), CenterSpec{},
                             request(orders, representation), ample());
  VSS_CHECK(result);
  return result.value();
}

void symmetricTwoPointRoundTrip() {
  const Matrix observations = matrix({{3.0}, {5.0}});
  const auto moments =
      central(observations, 0x0fU, MomentRepresentation::dense);
  const auto cumulants = centralMomentsToCumulants(
      moments, request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(cumulants);
  const auto terms = cumulants.value().terms.view();
  close(terms[0U].values.at(0U), 4.0);
  close(terms[1U].values.at(0U), 1.0);
  close(terms[2U].values.at(0U), 0.0);
  close(terms[3U].values.at(0U), -2.0);

  const auto recovered = cumulantsToCentralMoments(
      cumulants.value(), request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(recovered);
  const auto recoveredTerms = recovered.value().terms.view();
  close(recoveredTerms[0U].values.at(0U), 0.0);
  close(recoveredTerms[1U].values.at(0U), 1.0);
  close(recoveredTerms[2U].values.at(0U), 0.0);
  close(recoveredTerms[3U].values.at(0U), 1.0);
  if (!recovered.value().population_mean.has_value()) {
    VSS_CHECK(false);
    return;
  }
  const auto recoveredPopulationMean =
      recovered.value().population_mean.value_or(Vector{});
  close(recoveredPopulationMean.at(0U), 4.0);

  auto cumulantsWithoutMean = cumulants.value();
  cumulantsWithoutMean.population_mean.reset();
  const auto recoveredMean = cumulantsToCentralMoments(
      cumulantsWithoutMean, request(0x0fU, MomentRepresentation::dense),
      ample());
  VSS_CHECK(recoveredMean);
  if (!recoveredMean.value().population_mean.has_value()) {
    VSS_CHECK(false);
    return;
  }
  const auto reconstructedMean =
      recoveredMean.value().population_mean.value_or(Vector{});
  close(reconstructedMean.at(0U), 4.0);

  auto conflicting = cumulants.value();
  VSS_CHECK(conflicting.population_mean.has_value());
  const std::array<Scalar, 1U> conflictingMean{4.5};
  conflicting.population_mean = Vector(conflictingMean);
  const auto mismatch = cumulantsToCentralMoments(
      conflicting, request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(!mismatch && mismatch.status().code() == ErrorCode::InvalidDomain);
}

void asymmetricMixedFourthPairingsAndPackedSource() {
  const Matrix observations =
      matrix({{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}, {1.0, 1.0}});
  const std::array<Index, 1U> shape{2U};
  const auto packed = estimateCentralMoments(
      sample(observations, shape), unweighted(), CenterSpec{},
      request(0x0aU, MomentRepresentation::packed), ample());
  VSS_CHECK(packed);

  const std::array<Index, 4U> tuple{0U, 0U, 1U, 1U};
  const std::array<std::span<const Index>, 1U> selectedTuples{
      std::span<const Index>{tuple}};
  auto selected = request(0x08U, MomentRepresentation::selected);
  selected.selected = selectedTuples;
  const auto cumulant =
      centralMomentsToCumulants(packed.value(), selected, ample());
  VSS_CHECK(cumulant);
  VSS_CHECK(cumulant.value().terms.view()[0U].layout ==
            MomentRepresentation::selected);
  close(cumulant.value().terms.view()[0U].values.at(0U), -1.0 / 32.0);

  const auto dense = estimateCentralMoments(
      sample(observations, shape), unweighted(), CenterSpec{},
      request(0x0aU, MomentRepresentation::dense), ample());
  VSS_CHECK(dense);
  const auto packedOutput = centralMomentsToCumulants(
      dense.value(), request(0x08U, MomentRepresentation::packed), ample());
  VSS_CHECK(packedOutput);
  close(packedOutput.value().terms.view()[0U].values.at(2U), -1.0 / 32.0);

  const auto denseCumulants = centralMomentsToCumulants(
      packed.value(), request(0x0aU, MomentRepresentation::dense), ample());
  VSS_CHECK(denseCumulants);
  const auto recoveredFourth =
      cumulantsToCentralMoments(denseCumulants.value(), selected, ample());
  VSS_CHECK(recoveredFourth);
  close(recoveredFourth.value().terms.view()[0U].values.at(0U), 0.171875);

  const auto recovered =
      cumulantsToCentralMoments(cumulant.value(), selected, ample());
  VSS_CHECK(!recovered &&
            recovered.status().code() == ErrorCode::InvalidDomain);
}

void missingMeanDependencyAndValidation() {
  const Matrix observations = matrix({{-1.0}, {1.0}});
  auto moments = central(observations, 0x0fU, MomentRepresentation::dense);
  moments.population_mean.reset();
  const auto missingMean = centralMomentsToCumulants(
      moments, request(0x01U, MomentRepresentation::dense), ample());
  VSS_CHECK(!missingMean &&
            missingMean.status().code() == ErrorCode::InvalidDomain);

  const auto onlyFourth =
      central(observations, 0x08U, MomentRepresentation::dense);
  const auto missingPairs = centralMomentsToCumulants(
      onlyFourth, request(0x08U, MomentRepresentation::dense), ample());
  VSS_CHECK(!missingPairs &&
            missingPairs.status().code() == ErrorCode::InvalidDomain);

  const auto limited = centralMomentsToCumulants(
      central(observations, 0x0fU, MomentRepresentation::dense),
      request(0x0fU, MomentRepresentation::dense), Limits{1U, 8U, 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);

  const auto unsupported = centralMomentsToCumulants(
      central(observations, 0x0fU, MomentRepresentation::dense),
      request(0x02U, MomentRepresentation::contraction), ample());
  VSS_CHECK(!unsupported &&
            unsupported.status().code() == ErrorCode::InvalidDomain);
}

void exactFallbackResolvesLargeCancellation() {
  const Matrix observations = matrix({{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}});
  const std::array<Index, 1U> shape{3U};
  auto result = estimateCentralMoments(
      sample(observations, shape), unweighted(), CenterSpec{},
      request(0x02U, MomentRepresentation::dense), ample());
  VSS_CHECK(result);
  auto moments = std::move(result).value();
  std::array<Scalar, 9U> covariance{};
  covariance[0U] = 1.0e200;
  covariance[4U] = 1.0e200;
  covariance[1U] = 1.0e200;
  covariance[3U] = 1.0e200;
  covariance[5U] = 1.0e200;
  covariance[7U] = 1.0e200;
  covariance[2U] = -0.5e200;
  covariance[6U] = -0.5e200;
  MomentTerm second{.order = 2U,
                    .layout = MomentRepresentation::dense,
                    .flattened_tuples = {},
                    .values = Vector(covariance)};
  const std::array<Index, 4U> tuple{0U, 0U, 1U, 2U};
  auto flattened = Array<Index>::copy(tuple, ample());
  VSS_CHECK(flattened);
  const std::array<Scalar, 1U> fourthValue{0.0};
  MomentTerm fourth{.order = 4U,
                    .layout = MomentRepresentation::selected,
                    .flattened_tuples = std::move(flattened).value(),
                    .values = Vector(fourthValue)};
  const std::array<MomentTerm, 2U> terms{std::move(second), std::move(fourth)};
  auto ownedTerms = Array<MomentTerm>::copy(terms, ample());
  VSS_CHECK(ownedTerms);
  moments.terms = std::move(ownedTerms).value();
  const std::array<std::span<const Index>, 1U> selectedTuples{
      std::span<const Index>{tuple}};
  auto selected = request(0x08U, MomentRepresentation::selected);
  selected.selected = selectedTuples;
  const auto converted = centralMomentsToCumulants(moments, selected, ample());
  VSS_CHECK(converted);
  close(converted.value().terms.view()[0U].values.at(0U), 0.0);
  const auto inverse = cumulantsToCentralMoments(moments, selected, ample());
  VSS_CHECK(inverse);
  close(inverse.value().terms.view()[0U].values.at(0U), 0.0);
}

void independentCumulantAndCentralMomentSums() {
  const Matrix rademacher = matrix({{-1.0}, {1.0}});
  const auto rademacherMoments =
      central(rademacher, 0x0fU, MomentRepresentation::dense);
  const auto rademacherCumulants = centralMomentsToCumulants(
      rademacherMoments, request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(rademacherCumulants);
  const std::array<MomentEstimate, 2U> symmetricInputs{
      rademacherCumulants.value(), rademacherCumulants.value()};
  const std::array<Scalar, 2U> coefficients{1.0, -2.0};
  const Vector coefficientVector(coefficients);
  const auto summedCumulants = cumulantsOfIndependentSum(
      symmetricInputs, coefficientVector.view(),
      request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(summedCumulants);
  const auto terms = summedCumulants.value().terms.view();
  close(terms[0U].values.at(0U), 0.0);
  close(terms[1U].values.at(0U), 5.0);
  close(terms[2U].values.at(0U), 0.0);
  close(terms[3U].values.at(0U), -34.0);

  const std::array<MomentEstimate, 2U> symmetricCentralInputs{
      rademacherMoments, rademacherMoments};
  const auto summedMoments =
      weightedSum(coefficientVector.view(), symmetricCentralInputs,
                  request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(summedMoments);
  close(summedMoments.value().terms.view()[1U].values.at(0U), 5.0);
  close(summedMoments.value().terms.view()[3U].values.at(0U), 41.0);

  const Matrix skewed = matrix({{-1.0}, {-1.0}, {2.0}});
  const auto skewedMoments =
      central(skewed, 0x0fU, MomentRepresentation::dense);
  const auto skewedCumulants = centralMomentsToCumulants(
      skewedMoments, request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(skewedCumulants);
  const std::array<MomentEstimate, 2U> skewedInputs{skewedCumulants.value(),
                                                    skewedCumulants.value()};
  const std::array<Scalar, 2U> signedCoefficients{2.0, -1.0};
  const Vector signedCoefficientVector(signedCoefficients);
  const auto signedSum = cumulantsOfIndependentSum(
      skewedInputs, signedCoefficientVector.view(),
      request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(signedSum);
  close(signedSum.value().terms.view()[1U].values.at(0U), 10.0);
  close(signedSum.value().terms.view()[2U].values.at(0U), 14.0);
  close(signedSum.value().terms.view()[3U].values.at(0U), -102.0);

  const std::array<MomentEstimate, 2U> skewedCentralInputs{skewedMoments,
                                                           skewedMoments};
  const auto signedCentralSum =
      weightedSum(signedCoefficientVector.view(), skewedCentralInputs,
                  request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(signedCentralSum);
  close(signedCentralSum.value().terms.view()[2U].values.at(0U), 14.0);
  close(signedCentralSum.value().terms.view()[3U].values.at(0U), 198.0);

  const std::array<Scalar, 2U> zeroCoefficients{0.0, -0.0};
  const Vector zeroCoefficientVector(zeroCoefficients);
  const auto zeroSum =
      weightedSum(zeroCoefficientVector.view(), skewedCentralInputs,
                  request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(zeroSum);
  for (const auto& term : zeroSum.value().terms.view()) {
    for (Index index = 0U; index < term.values.size(); ++index) {
      close(term.values.at(index), 0.0);
    }
  }
  VSS_CHECK(zeroSum.value().population_mean.has_value());
  close(zeroSum.value().population_mean.value_or(Vector{}).at(0U), 0.0);

  auto missingTerms = skewedCumulants.value();
  missingTerms.terms = Array<MomentTerm>{};
  missingTerms.population_mean.reset();
  const std::array<MomentEstimate, 2U> partiallyMissing{skewedCumulants.value(),
                                                        missingTerms};
  const std::array<Scalar, 2U> keepFirst{1.0, 0.0};
  const Vector keepFirstVector(keepFirst);
  const auto ignoresZeroOperand = cumulantsOfIndependentSum(
      partiallyMissing, keepFirstVector.view(),
      request(0x08U, MomentRepresentation::dense), ample());
  VSS_CHECK(ignoresZeroOperand);
  close(ignoresZeroOperand.value().terms.view()[0U].values.at(0U),
        skewedCumulants.value().terms.view()[3U].values.at(0U));

  auto differentlyShapedRecord = skewedCumulants.value();
  const std::array<Index, 1U> wrongShape{2U};
  auto wrongShapeCopy = Array<Index>::copy(wrongShape, ample());
  VSS_CHECK(wrongShapeCopy);
  differentlyShapedRecord.observation_shape = std::move(wrongShapeCopy).value();
  const std::array<MomentEstimate, 2U> differentlyShaped{
      skewedCumulants.value(), differentlyShapedRecord};
  const auto shapeMismatch = cumulantsOfIndependentSum(
      differentlyShaped, coefficientVector.view(),
      request(0x02U, MomentRepresentation::dense), ample());
  VSS_CHECK(!shapeMismatch &&
            shapeMismatch.status().code() == ErrorCode::DimensionMismatch);

  const auto mismatch = cumulantsOfIndependentSum(
      symmetricInputs, Vector(1U, 1.0).view(),
      request(0x02U, MomentRepresentation::dense), ample());
  VSS_CHECK(!mismatch &&
            mismatch.status().code() == ErrorCode::DimensionMismatch);
}

void weightedSumContractionAndSelectedFourth() {
  const Matrix observations = matrix({{-1.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}});
  const auto moments =
      central(observations, 0x0fU, MomentRepresentation::dense);
  const std::array<MomentEstimate, 2U> inputs{moments, moments};
  const std::array<Scalar, 2U> coefficients{1.0, 1.0};
  const Vector coefficientVector(coefficients);
  const auto vectorCumulants = centralMomentsToCumulants(
      moments, request(0x0fU, MomentRepresentation::dense), ample());
  VSS_CHECK(vectorCumulants);
  const std::array<MomentEstimate, 2U> independentVectorInputs{
      vectorCumulants.value(), vectorCumulants.value()};
  const std::array<Scalar, 2U> cancellingCoefficients{1.0, -1.0};
  const Vector cancellingCoefficientVector(cancellingCoefficients);
  const auto exactCancellation = cumulantsOfIndependentSum(
      independentVectorInputs, cancellingCoefficientVector.view(),
      request(0x01U, MomentRepresentation::dense), ample());
  VSS_CHECK(exactCancellation);
  for (Index index = 0U;
       index < exactCancellation.value().terms.view()[0U].values.size();
       ++index) {
    close(exactCancellation.value().terms.view()[0U].values.at(index), 0.0);
  }

  Matrix projection = matrix({{1.0, 0.0}});
  MomentRequest contracted = request(0x08U, MomentRepresentation::contraction);
  contracted.contraction = projection.view();
  const auto contractedFourth =
      weightedSum(coefficientVector.view(), inputs, contracted, ample());
  VSS_CHECK(contractedFourth);
  VSS_CHECK(contractedFourth.value().observation_shape.view().size() == 1U);
  VSS_CHECK(contractedFourth.value().observation_shape.view()[0U] == 1U);
  VSS_CHECK(contractedFourth.value().terms.view().size() == 1U);
  close(contractedFourth.value().terms.view()[0U].values.at(0U), 4.0);

  const std::array<Index, 4U> selectedTuple{0U, 0U, 0U, 0U};
  const std::array<std::span<const Index>, 1U> selectedTuples{
      std::span<const Index>(selectedTuple)};
  MomentRequest selected = request(0x08U, MomentRepresentation::selected);
  selected.selected = selectedTuples;
  const auto selectedFourth =
      weightedSum(coefficientVector.view(), inputs, selected, ample());
  VSS_CHECK(selectedFourth);
  VSS_CHECK(selectedFourth.value().terms.view().size() == 1U);
  VSS_CHECK(selectedFourth.value().terms.view()[0U].layout ==
            MomentRepresentation::selected);
  close(selectedFourth.value().terms.view()[0U].values.at(0U), 4.0);

  auto unavailable = moments;
  unavailable.terms = Array<MomentTerm>{};
  unavailable.population_mean.reset();
  const std::array<MomentEstimate, 2U> unavailableInputs{unavailable,
                                                         unavailable};
  Matrix zeroMap = matrix({{0.0, 0.0}});
  contracted.contraction = zeroMap.view();
  const auto zeroProjection = weightedSum(
      coefficientVector.view(), unavailableInputs, contracted, ample());
  VSS_CHECK(zeroProjection);
  close(zeroProjection.value().terms.view()[0U].values.at(0U), 0.0);
  VSS_CHECK(zeroProjection.value().population_mean.has_value());
  close(zeroProjection.value().population_mean.value_or(Vector{}).at(0U), 0.0);

  const MomentRequest missingMap =
      request(0x02U, MomentRepresentation::contraction);
  const auto rejectedMap = cumulantsOfIndependentSum(
      inputs, coefficientVector.view(), missingMap, ample());
  VSS_CHECK(!rejectedMap &&
            rejectedMap.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() try {
  symmetricTwoPointRoundTrip();
  asymmetricMixedFourthPairingsAndPackedSource();
  missingMeanDependencyAndValidation();
  exactFallbackResolvesLargeCancellation();
  independentCumulantAndCentralMomentSums();
  weightedSumContractionAndSelectedFourth();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
