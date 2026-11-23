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
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/statistics/MomentDiagnostics.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Extent row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Extent column = 0U;
    for (const Scalar value : values) {
      result.at(row, column) = value;
      ++column;
    }
    ++row;
  }
  return result;
}

SampleView sample(const Matrix& values, std::span<const Index> shape = {},
                  std::span<const std::uint8_t> valid = {}) {
  return {.rows = values.view(), .observation_shape = shape, .valid = valid};
}

Limits limits() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = 0U};
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 128U;
  result.rank = RankPolicy::reject;
  return result;
}

MomentRequest
allOrders(MomentRepresentation representation = MomentRepresentation::dense) {
  return {.orders = 0x0fU,
          .representation = representation,
          .selected = {},
          .contraction = std::nullopt};
}

MomentEstimate centralMoments(
    const Matrix& values, std::span<const Index> shape = {},
    MomentRepresentation representation = MomentRepresentation::dense) {
  auto result =
      estimateCentralMoments(sample(values, shape),
                             WeightSpec{.rule = WeightRule::unweighted,
                                        .rowWeights = std::nullopt,
                                        .componentWeights = std::nullopt,
                                        .frequencyCounts = {}},
                             CenterSpec{}, allOrders(representation), limits());
  VSS_CHECK(result);
  return result ? std::move(result).value() : MomentEstimate{};
}

void close(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-10) {
  VSS_CHECK(std::abs(actual - expected) <=
            tolerance * std::max({1.0, std::abs(actual), std::abs(expected)}));
}

void validCentralMomentsReportNecessaryChecksOnly() {
  const Matrix values = matrix({{-1.0}, {0.0}, {2.0}});
  const auto moments = centralMoments(values);
  const auto result = validateMoments(moments, policy(), limits());
  VSS_CHECK(result);
  if (result) {
    VSS_CHECK(result.value().symmetric);
    VSS_CHECK(result.value().second_psd);
    VSS_CHECK(result.value().degree_two_moment_gram_psd);
    VSS_CHECK(!result.value().sufficient_for_model);
    VSS_CHECK(result.value().second_scale > 0.0);
    VSS_CHECK(result.value().gram_scale > 0.0);
  }
}

void componentAndPackedMomentLayoutsAreValidated() {
  const Matrix scalar = matrix({{-1.0}, {0.0}, {2.0}});
  const auto componentResult = validateMoments(
      centralMoments(scalar, {}, MomentRepresentation::component), policy(),
      limits());
  VSS_CHECK(componentResult);
  if (componentResult) {
    VSS_CHECK(componentResult.value().symmetric);
    VSS_CHECK(componentResult.value().degree_two_moment_gram_psd);
  }

  const std::array<Index, 1U> shape{2U};
  const Matrix vectorValues =
      matrix({{-1.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}, {1.0, 2.0}, {-2.0, -2.0}});
  const auto packedResult = validateMoments(
      centralMoments(vectorValues, shape, MomentRepresentation::packed),
      policy(), limits());
  VSS_CHECK(packedResult);
  if (packedResult) {
    VSS_CHECK(packedResult.value().symmetric);
    VSS_CHECK(packedResult.value().second_psd);
    VSS_CHECK(packedResult.value().degree_two_moment_gram_psd);
  }
}

void completeSelectedFourthOrderLayoutIsValidated() {
  const std::array<Index, 1U> shape{2U};
  const Matrix values =
      matrix({{-1.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}, {1.0, 2.0}, {-2.0, -2.0}});
  auto moments = centralMoments(values, shape);
  std::array<std::array<Index, 4U>, 16U> tuples{};
  std::array<std::span<const Index>, 16U> selected{};
  for (Index index = 0U; index < tuples.size(); ++index) {
    for (Index axis = 0U; axis < 4U; ++axis) {
      tuples.at(index).at(axis) = (index >> axis) & 1U;
    }
    selected.at(index) = tuples.at(index);
  }
  const MomentRequest selectedRequest{.orders = 0x08U,
                                      .representation =
                                          MomentRepresentation::selected,
                                      .selected = selected,
                                      .contraction = std::nullopt};
  const auto selectedFourth =
      estimateCentralMoments(sample(values, shape),
                             WeightSpec{.rule = WeightRule::unweighted,
                                        .rowWeights = std::nullopt,
                                        .componentWeights = std::nullopt,
                                        .frequencyCounts = {}},
                             CenterSpec{}, selectedRequest, limits());
  VSS_CHECK(selectedFourth);
  if (!selectedFourth) {
    return;
  }
  std::vector<MomentTerm> terms(moments.terms.view().begin(),
                                moments.terms.view().end());
  terms.at(3U) = selectedFourth.value().terms.view()[0U];
  auto copied = Array<MomentTerm>::copy(terms, limits());
  VSS_CHECK(copied);
  if (!copied) {
    return;
  }
  moments.terms = std::move(copied).value();
  const auto result = validateMoments(moments, policy(), limits());
  VSS_CHECK(result);
  if (result) {
    VSS_CHECK(result.value().symmetric);
    VSS_CHECK(result.value().degree_two_moment_gram_psd);
  }
}

void asymmetricEntriesAreReportedWithoutDiscardingDiagnostics() {
  const std::array<Index, 1U> shape{2U};
  const Matrix values =
      matrix({{-1.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}, {1.0, 2.0}, {-2.0, -2.0}});
  auto moments = centralMoments(values, shape);
  std::vector<MomentTerm> terms(moments.terms.view().begin(),
                                moments.terms.view().end());
  terms.at(1U).values.at(1U) += 0.25;
  auto copied = Array<MomentTerm>::copy(terms, limits());
  VSS_CHECK(copied);
  if (copied) {
    moments.terms = std::move(copied).value();
  }
  const auto result = validateMoments(moments, policy(), limits());
  VSS_CHECK(result);
  if (result) {
    VSS_CHECK(!result.value().symmetric);
  }
}

void missingRequiredCrossMomentIsNotTreatedAsZero() {
  const std::array<Index, 1U> shape{2U};
  const Matrix values =
      matrix({{-1.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}, {1.0, 2.0}, {-2.0, -2.0}});
  auto moments = centralMoments(values, shape);
  std::vector<MomentTerm> terms(moments.terms.view().begin(),
                                moments.terms.view().end());
  const std::array<Index, 4U> tuple{0U, 0U, 0U, 0U};
  auto flattened = Array<Index>::copy(tuple, limits());
  VSS_CHECK(flattened);
  if (flattened) {
    terms.at(3U) = MomentTerm{.order = 4U,
                              .layout = MomentRepresentation::selected,
                              .flattened_tuples = std::move(flattened).value(),
                              .values = Vector(1U, 1.0)};
    auto copied = Array<MomentTerm>::copy(terms, limits());
    VSS_CHECK(copied);
    if (copied) {
      moments.terms = std::move(copied).value();
    }
  }
  const auto result = validateMoments(moments, policy(), limits());
  VSS_CHECK(!result);
  VSS_CHECK(result.status().code() == ErrorCode::InsufficientSamples);
}

void negativeVarianceIsAReportedFailedCheck() {
  const Matrix values = matrix({{-1.0}, {1.0}});
  auto moments = centralMoments(values);
  std::vector<MomentTerm> terms(moments.terms.view().begin(),
                                moments.terms.view().end());
  terms.at(1U).values.at(0U) = -1.0;
  auto copied = Array<MomentTerm>::copy(terms, limits());
  VSS_CHECK(copied);
  if (copied) {
    moments.terms = std::move(copied).value();
  }
  const auto result = validateMoments(moments, policy(), limits());
  VSS_CHECK(result);
  if (result) {
    VSS_CHECK(!result.value().second_psd);
    VSS_CHECK(!result.value().degree_two_moment_gram_psd);
    VSS_CHECK(!result.value().sufficient_for_model);
  }
}

void truncatedButConsistentMomentsNeverClaimModelSufficiency() {
  const Matrix values = matrix({{-1.0}, {1.0}});
  const auto result =
      validateMoments(centralMoments(values), policy(), limits());
  VSS_CHECK(result);
  if (result) {
    VSS_CHECK(result.value().symmetric);
    VSS_CHECK(result.value().second_psd);
    VSS_CHECK(result.value().degree_two_moment_gram_psd);
    VSS_CHECK(!result.value().sufficient_for_model);
  }
}

void gramWorkspaceLimitIsChecked() {
  const std::array<Index, 1U> shape{2U};
  const Matrix values =
      matrix({{-1.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}, {1.0, 2.0}, {-2.0, -2.0}});
  const auto result =
      validateMoments(centralMoments(values, shape), policy(),
                      {.elements = 8U, .bytes = 128U, .evaluations = 0U});
  VSS_CHECK(!result);
  VSS_CHECK(result.status().code() == ErrorCode::ResourceLimit);
}

void mardiaScalarReferences() {
  const Matrix symmetric = matrix({{-1.0}, {1.0}});
  const auto skew = mardiaSkewness(sample(symmetric), policy(), limits());
  const auto kurtosis = mardiaKurtosis(sample(symmetric), policy(), limits());
  VSS_CHECK(skew && kurtosis);
  if (skew && kurtosis) {
    close(skew.value(), 0.0);
    close(kurtosis.value(), 1.0);
  }
  const Matrix asymmetric = matrix({{-1.0}, {0.0}, {2.0}});
  const auto skewed = mardiaSkewness(sample(asymmetric), policy(), limits());
  VSS_CHECK(skewed);
  if (skewed) {
    close(skewed.value(), 50.0 / 343.0);
  }
  for (const Scalar sourceScale : {1.0e-150, 1.0e150}) {
    const Matrix scaled = matrix({{-sourceScale}, {0.0}, {2.0 * sourceScale}});
    const auto scaledSkewness =
        mardiaSkewness(sample(scaled), policy(), limits());
    const auto scaledKurtosis =
        mardiaKurtosis(sample(scaled), policy(), limits());
    VSS_CHECK(scaledSkewness && scaledKurtosis);
    if (scaledSkewness && scaledKurtosis) {
      close(scaledSkewness.value(), 50.0 / 343.0, 1.0e-8);
      close(scaledKurtosis.value(), 1.5, 1.0e-8);
    }
  }
}

void mardiaIsInvariantUnderNonsingularAffineMaps() {
  const Matrix values =
      matrix({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, -1.0}});
  Matrix transformed(values.rows(), 2U);
  for (Index row = 0U; row < values.rows(); ++row) {
    transformed.at(row, 0U) =
        2.0 * values.at(row, 0U) + 3.0 * values.at(row, 1U) + 5.0;
    transformed.at(row, 1U) =
        -values.at(row, 0U) + 4.0 * values.at(row, 1U) - 2.0;
  }
  const std::array<Index, 1U> shape{2U};
  const auto skew = mardiaSkewness(sample(values, shape), policy(), limits());
  const auto transformedSkew =
      mardiaSkewness(sample(transformed, shape), policy(), limits());
  const auto kurtosis =
      mardiaKurtosis(sample(values, shape), policy(), limits());
  const auto transformedKurtosis =
      mardiaKurtosis(sample(transformed, shape), policy(), limits());
  VSS_CHECK(skew && transformedSkew && kurtosis && transformedKurtosis);
  if (skew && transformedSkew && kurtosis && transformedKurtosis) {
    close(skew.value(), 13986.0 / 12167.0, 1.0e-8);
    close(kurtosis.value(), 2780.0 / 529.0, 1.0e-8);
    close(skew.value(), transformedSkew.value(), 1.0e-8);
    close(kurtosis.value(), transformedKurtosis.value(), 1.0e-8);
  }
}

void mardiaRejectsRankDeficientAndIncompleteInputs() {
  const std::array<Index, 1U> shape{2U};
  const Matrix collinear = matrix({{0.0, 0.0}, {1.0, 2.0}, {2.0, 4.0}});
  const auto rank =
      mardiaSkewness(sample(collinear, shape), policy(), limits());
  VSS_CHECK(!rank);
  VSS_CHECK(rank.status().code() == ErrorCode::RankDeficient);

  const Matrix tooFew = matrix({{0.0, 0.0}, {1.0, 0.0}});
  const auto insufficient =
      mardiaKurtosis(sample(tooFew, shape), policy(), limits());
  VSS_CHECK(!insufficient);
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);

  const std::array<std::uint8_t, 2U> invalidMask{1U, 0U};
  const Matrix incomplete = matrix({{-1.0}, {1.0}});
  const auto missing =
      mardiaKurtosis(sample(incomplete, {}, invalidMask), policy(), limits());
  VSS_CHECK(!missing);
  VSS_CHECK(missing.status().code() == ErrorCode::InvalidDomain);

  const Matrix nonFinite =
      matrix({{-1.0}, {std::numeric_limits<Scalar>::infinity()}});
  const auto rejected = mardiaSkewness(sample(nonFinite), policy(), limits());
  VSS_CHECK(!rejected);
  VSS_CHECK(rejected.status().code() == ErrorCode::NonFiniteInput);
}

} // namespace

int main() try {
  validCentralMomentsReportNecessaryChecksOnly();
  componentAndPackedMomentLayoutsAreValidated();
  completeSelectedFourthOrderLayoutIsValidated();
  asymmetricEntriesAreReportedWithoutDiscardingDiagnostics();
  missingRequiredCrossMomentIsNotTreatedAsZero();
  negativeVarianceIsAReportedFailedCheck();
  truncatedButConsistentMomentsNeverClaimModelSufficiency();
  gramWorkspaceLimitIsChecked();
  mardiaScalarReferences();
  mardiaIsInvariantUnderNonsingularAffineMaps();
  mardiaRejectsRankDeficientAndIncompleteInputs();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
