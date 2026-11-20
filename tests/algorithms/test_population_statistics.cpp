#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>
#include <vss/math/statistics/Population.hpp>

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

SampleView sample(const Matrix& values) { return {values.view(), {}, {}}; }

Limits ampleLimits() {
  return {std::numeric_limits<std::size_t>::max(),
          std::numeric_limits<std::size_t>::max(), 0U};
}

void close(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <=
            1.0e-12 + 1.0e-10 * std::max(std::abs(actual), std::abs(expected)));
}

void scalarPopulationAndSampleCovariance() {
  const Matrix values = matrix({{-1.0}, {0.0}, {1.0}});
  const CenterSpec estimated{CenterRule::estimated, std::nullopt};
  const auto population = covarianceMatrix(
      sample(values), estimated,
      {VarianceMode::empirical, MissingMode::reject, Estimand::observed},
      ampleLimits());
  VSS_CHECK(population.hasValue());
  close(population.value().value.at(0U, 0U), 2.0 / 3.0);

  const auto sampleCovariance = covarianceMatrix(
      sample(values), estimated,
      {VarianceMode::iid_sample, MissingMode::reject, Estimand::observed},
      ampleLimits());
  VSS_CHECK(sampleCovariance.hasValue());
  close(sampleCovariance.value().value.at(0U, 0U), 1.0);
}

void knownCenterAndCrossCovariance() {
  const Matrix left = matrix({{1.0}, {2.0}, {4.0}});
  const Matrix right = matrix({{2.0}, {1.0}, {5.0}});
  const CenterSpec leftCenter{CenterRule::estimated, std::nullopt};
  const CenterSpec rightCenter{CenterRule::estimated, std::nullopt};
  const PopulationPolicy empirical{VarianceMode::empirical, MissingMode::reject,
                                   Estimand::observed};
  const auto cross = crossCovariance(sample(left), sample(right), leftCenter,
                                     rightCenter, empirical, ampleLimits());
  VSS_CHECK(cross.hasValue());
  close(cross.value().value.at(0U, 0U), 16.0 / 9.0);
  const auto reverse = crossCovariance(sample(right), sample(left), rightCenter,
                                       leftCenter, empirical, ampleLimits());
  VSS_CHECK(reverse.hasValue());
  close(reverse.value().value.at(0U, 0U), cross.value().value.at(0U, 0U));

  const Vector centerValue(1U, 0.0);
  const CenterSpec known{CenterRule::known, centerValue.view()};
  const PopulationPolicy aroundKnown{VarianceMode::known_center,
                                     MissingMode::reject, Estimand::observed};
  const auto knownCovariance =
      covarianceMatrix(sample(left), known, aroundKnown, ampleLimits());
  VSS_CHECK(knownCovariance.hasValue());
  close(knownCovariance.value().value.at(0U, 0U), 7.0);

  const Matrix symmetricLeft = matrix({{-1.0}, {0.0}, {1.0}});
  const Matrix symmetricRight = matrix({{-2.0}, {0.0}, {2.0}});
  const PopulationPolicy sampleMode{VarianceMode::iid_sample,
                                    MissingMode::reject, Estimand::observed};
  const auto sampleCross =
      crossCovariance(sample(symmetricLeft), sample(symmetricRight), leftCenter,
                      rightCenter, sampleMode, ampleLimits());
  VSS_CHECK(sampleCross.hasValue());
  close(sampleCross.value().value.at(0U, 0U), 2.0);
}

void crossCovariancePreservesOrientationAndCommonRows() {
  const Matrix left = matrix({{0.0, 1.0}, {1.0, 0.0}, {2.0, 2.0}});
  const Matrix right = matrix({{2.0, 0.0}, {1.0, 4.0}, {5.0, 1.0}});
  const std::array<Extent, 1U> shape{2U};
  const SampleView leftView{left.view(), shape, {}};
  const SampleView rightView{right.view(), shape, {}};
  const CenterSpec estimated{CenterRule::estimated, std::nullopt};
  const PopulationPolicy empirical{VarianceMode::empirical, MissingMode::reject,
                                   Estimand::observed};
  const auto cross = crossCovariance(leftView, rightView, estimated, estimated,
                                     empirical, ampleLimits());
  VSS_CHECK(cross.hasValue());
  const Matrix& value = cross.value().value;
  VSS_CHECK(value.rows() == 2U && value.columns() == 2U);
  close(value.at(0U, 0U), 1.0);
  close(value.at(0U, 1U), 1.0 / 3.0);
  close(value.at(1U, 0U), 4.0 / 3.0);
  close(value.at(1U, 1U), -1.0);

  const auto reverse = crossCovariance(rightView, leftView, estimated,
                                       estimated, empirical, ampleLimits());
  VSS_CHECK(reverse.hasValue());
  for (Index row = 0U; row < 2U; ++row) {
    for (Index column = 0U; column < 2U; ++column) {
      close(reverse.value().value.at(row, column), value.at(column, row));
    }
  }

  const Matrix differentDimension =
      matrix({{2.0, 0.0, 3.0}, {1.0, 4.0, 4.0}, {5.0, 1.0, 7.0}});
  const std::array<Extent, 1U> threeCoordinates{3U};
  const SampleView threeDimensionalView{
      differentDimension.view(), threeCoordinates, {}};
  const auto rectangular =
      crossCovariance(leftView, threeDimensionalView, estimated, estimated,
                      empirical, ampleLimits());
  VSS_CHECK(rectangular.hasValue());
  VSS_CHECK(rectangular.value().value.rows() == 2U &&
            rectangular.value().value.columns() == 3U);
  const auto rectangularReverse =
      crossCovariance(threeDimensionalView, leftView, estimated, estimated,
                      empirical, ampleLimits());
  VSS_CHECK(rectangularReverse.hasValue());
  for (Index row = 0U; row < 2U; ++row) {
    for (Index column = 0U; column < 3U; ++column) {
      close(rectangularReverse.value().value.at(column, row),
            rectangular.value().value.at(row, column));
    }
  }
}

void rejectsMismatchedSupportAndInsufficientSample() {
  const Matrix left = matrix({{1.0}, {2.0}, {3.0}});
  const Matrix right = matrix({{1.0}, {2.0}});
  const CenterSpec estimated{CenterRule::estimated, std::nullopt};
  const PopulationPolicy empirical{VarianceMode::empirical, MissingMode::reject,
                                   Estimand::observed};
  const auto mismatch = crossCovariance(sample(left), sample(right), estimated,
                                        estimated, empirical, ampleLimits());
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);

  const Matrix one = matrix({{1.0}});
  const PopulationPolicy sampleMode{VarianceMode::iid_sample,
                                    MissingMode::reject, Estimand::observed};
  const auto insufficient =
      covarianceMatrix(sample(one), estimated, sampleMode, ampleLimits());
  VSS_CHECK(!insufficient.hasValue());
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);
}

void completeRowMaskUsesCommonSupport() {
  const Scalar missing = std::numeric_limits<Scalar>::quiet_NaN();
  const Matrix left = matrix({{1.0}, {missing}, {3.0}});
  const Matrix right = matrix({{2.0}, {missing}, {6.0}});
  const std::array<std::uint8_t, 3U> mask{1U, 0U, 1U};
  const SampleView leftView{left.view(), {}, mask};
  const SampleView rightView{right.view(), {}, mask};
  const CenterSpec estimated{CenterRule::estimated, std::nullopt};
  const Vector zeroCenter(1U, 0.0);
  const CenterSpec known{CenterRule::known, zeroCenter.view()};
  const PopulationPolicy empirical{
      VarianceMode::empirical, MissingMode::complete_row, Estimand::observed};
  const auto cross = crossCovariance(leftView, rightView, estimated, estimated,
                                     empirical, ampleLimits());
  VSS_CHECK(cross.hasValue());
  close(cross.value().value.at(0U, 0U), 2.0);
  VSS_CHECK(cross.value().diagnostics.global.valid == 2U);

  const PopulationPolicy aroundKnown{VarianceMode::known_center,
                                     MissingMode::complete_row,
                                     Estimand::observed};
  const auto knownCross = crossCovariance(leftView, rightView, known, known,
                                          aroundKnown, ampleLimits());
  VSS_CHECK(knownCross.hasValue());
  close(knownCross.value().value.at(0U, 0U), 10.0);

  const PopulationPolicy rejecting{VarianceMode::empirical, MissingMode::reject,
                                   Estimand::observed};
  const auto rejected = crossCovariance(leftView, rightView, estimated,
                                        estimated, rejecting, ampleLimits());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::InvalidDomain);
}

void enforcesOutputLimitsAndRejectsNonFiniteRetainedValues() {
  const Matrix values = matrix({{-1.0}, {0.0}, {1.0}});
  const CenterSpec estimated{CenterRule::estimated, std::nullopt};
  const PopulationPolicy empirical{VarianceMode::empirical, MissingMode::reject,
                                   Estimand::observed};
  const auto limited =
      covarianceMatrix(sample(values), estimated, empirical, {0U, 0U, 0U});
  VSS_CHECK(!limited.hasValue());
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);

  const Matrix nonFinite =
      matrix({{1.0}, {std::numeric_limits<Scalar>::infinity()}, {3.0}});
  const auto invalid =
      covarianceMatrix(sample(nonFinite), estimated, empirical, ampleLimits());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);
}

} // namespace

int main() {
  scalarPopulationAndSampleCovariance();
  knownCenterAndCrossCovariance();
  crossCovariancePreservesOrientationAndCommonRows();
  rejectsMismatchedSupportAndInsufficientSample();
  completeRowMaskUsesCommonSupport();
  enforcesOutputLimitsAndRejectsNonFiniteRetainedValues();
}
