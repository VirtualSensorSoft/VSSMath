#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
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

SolvePolicy spectralPolicy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 128U;
  result.rank = RankPolicy::supported_subspace;
  return result;
}

void correlationAndPartialCorrelationAcceptanceCases() {
  const auto policy = spectralPolicy();
  const Matrix scaled = matrix({{4.0, 3.0}, {3.0, 9.0}});
  const auto correlation =
      correlationMatrix(scaled.view(), policy, ampleLimits());
  VSS_CHECK(correlation.hasValue());
  close(correlation.value().covariance.at(0U, 0U), 1.0);
  close(correlation.value().covariance.at(0U, 1U), 0.5);
  close(correlation.value().covariance.at(1U, 1U), 1.0);
  auto conditionLimitedPolicy = policy;
  conditionLimitedPolicy.reciprocal_condition_floor = 0.5;
  const auto conditionLimited =
      correlationMatrix(scaled.view(), conditionLimitedPolicy, ampleLimits());
  VSS_CHECK(conditionLimited.hasValue());
  VSS_CHECK(conditionLimited.value().classification ==
            covariance::CovarianceKind::ill_conditioned);
  VSS_CHECK(conditionLimited.value().rank == 2U);
  close(*conditionLimited.value().reciprocal_condition, 1.0 / 3.0);
  const std::array<Index, 2U> both{0U, 1U};
  const std::array<Index, 1U> secondCoordinate{1U};
  const auto marginal =
      partialCorrelation(scaled.view(), both, {}, policy, ampleLimits());
  VSS_CHECK(marginal.hasValue());
  close(marginal.value().covariance.at(0U, 1U), 0.5);
  const auto overlap = partialCorrelation(scaled.view(), both, secondCoordinate,
                                          policy, ampleLimits());
  VSS_CHECK(!overlap.hasValue());
  VSS_CHECK(overlap.status().code() == ErrorCode::InvalidDomain);

  const Matrix joint =
      matrix({{4.0, 2.0, 2.0}, {2.0, 4.0, 2.0}, {2.0, 2.0, 4.0}});
  const std::array<Index, 2U> selected{0U, 1U};
  const std::array<Index, 1U> conditioned{2U};
  const auto partial = partialCorrelation(joint.view(), selected, conditioned,
                                          policy, ampleLimits());
  VSS_CHECK(partial.hasValue());
  close(partial.value().covariance.at(0U, 1U), 1.0 / 3.0);

  const Matrix redundant = matrix({{1.0, 0.5, 0.5, 0.5},
                                   {0.5, 1.0, 0.5, 0.5},
                                   {0.5, 0.5, 1.0, 1.0},
                                   {0.5, 0.5, 1.0, 1.0}});
  const std::array<Index, 2U> redundantConditioning{2U, 3U};
  auto strict = policy;
  strict.rank = RankPolicy::reject;
  const auto rejected = partialCorrelation(
      redundant.view(), selected, redundantConditioning, strict, ampleLimits());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::RankDeficient);
  const auto supported = partialCorrelation(
      redundant.view(), selected, redundantConditioning, policy, ampleLimits());
  VSS_CHECK(supported.hasValue());
  close(supported.value().covariance.at(0U, 1U), 1.0 / 3.0);
  auto minimumNormPolicy = policy;
  minimumNormPolicy.rank = RankPolicy::minimum_norm;
  const auto minimumNorm =
      partialCorrelation(redundant.view(), selected, redundantConditioning,
                         minimumNormPolicy, ampleLimits());
  VSS_CHECK(minimumNorm.hasValue());
  close(minimumNorm.value().covariance.at(0U, 1U), 1.0 / 3.0);

  Matrix illConditionedBlock = redundant;
  illConditionedBlock.at(2U, 3U) = 0.75;
  illConditionedBlock.at(3U, 2U) = 0.75;
  auto floorPolicy = policy;
  floorPolicy.reciprocal_condition_floor = 0.5;
  const auto illConditioned =
      partialCorrelation(illConditionedBlock.view(), selected,
                         redundantConditioning, floorPolicy, ampleLimits());
  VSS_CHECK(!illConditioned.hasValue());
  VSS_CHECK(illConditioned.status().code() == ErrorCode::IllConditioned);

  const Matrix singular = matrix({{1.0, -1.0}, {-1.0, 1.0}});
  const auto singularCorrelation =
      correlationMatrix(singular.view(), policy, ampleLimits());
  VSS_CHECK(singularCorrelation.hasValue());
  VSS_CHECK(singularCorrelation.value().rank == 1U);
  close(singularCorrelation.value().covariance.at(0U, 1U), -1.0);

  const Matrix identical = matrix({{1.0, 1.0}, {1.0, 1.0}});
  const std::array<Index, 1U> oneSelected{0U};
  const std::array<Index, 1U> oneConditioned{1U};
  const auto zeroResidual = partialCorrelation(
      identical.view(), oneSelected, oneConditioned, policy, ampleLimits());
  VSS_CHECK(!zeroResidual.hasValue());
  VSS_CHECK(zeroResidual.status().code() == ErrorCode::InvalidDomain);

  const Matrix zeroVariance = matrix({{0.0, 0.0}, {0.0, 1.0}});
  const auto undefined =
      correlationMatrix(zeroVariance.view(), policy, ampleLimits());
  VSS_CHECK(!undefined.hasValue());
  VSS_CHECK(undefined.status().code() == ErrorCode::InvalidDomain);
  Matrix separatedScales(2U, 2U, 0.0);
  separatedScales.at(0U, 0U) = std::numeric_limits<Scalar>::denorm_min();
  separatedScales.at(1U, 1U) = std::numeric_limits<Scalar>::max();
  const auto separated =
      correlationMatrix(separatedScales.view(), policy, ampleLimits());
  VSS_CHECK(separated.hasValue());
  VSS_CHECK(separated.value().rank == 2U);
  VSS_CHECK(separated.value().covariance.at(0U, 1U) == 0.0);

  const Matrix asymmetric = matrix({{1.0, 2.0}, {3.0, 1.0}});
  const auto notSymmetric =
      correlationMatrix(asymmetric.view(), policy, ampleLimits());
  VSS_CHECK(!notSymmetric.hasValue());
  VSS_CHECK(notSymmetric.status().code() == ErrorCode::NotSymmetric);
  const Matrix indefinite = matrix({{1.0, 2.0}, {2.0, 1.0}});
  const auto notPsd =
      correlationMatrix(indefinite.view(), policy, ampleLimits());
  VSS_CHECK(!notPsd.hasValue());
  VSS_CHECK(notPsd.status().code() == ErrorCode::NotPositiveSemidefinite);
  const Matrix invalidZeroVariance = matrix({{0.0, 1.0}, {1.0, 1.0}});
  const auto invalidZeroRow =
      correlationMatrix(invalidZeroVariance.view(), policy, ampleLimits());
  VSS_CHECK(!invalidZeroRow.hasValue());
  VSS_CHECK(invalidZeroRow.status().code() ==
            ErrorCode::NotPositiveSemidefinite);
  const Matrix nonfinite =
      matrix({{std::numeric_limits<Scalar>::infinity(), 0.0}, {0.0, 1.0}});
  const auto nonfiniteResult =
      correlationMatrix(nonfinite.view(), policy, ampleLimits());
  VSS_CHECK(!nonfiniteResult.hasValue());
  VSS_CHECK(nonfiniteResult.status().code() == ErrorCode::NonFiniteInput);
  const auto denied =
      correlationMatrix(singular.view(), policy, Limits{0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  const Matrix empty;
  const auto noSamples = correlationMatrix(empty.view(), policy, ampleLimits());
  VSS_CHECK(!noSamples.hasValue());
  VSS_CHECK(noSamples.status().code() == ErrorCode::InsufficientSamples);

  const Matrix zeroVarianceWithPositiveCoordinate =
      matrix({{0.0, 0.0}, {0.0, 1.0}});
  const std::array<Index, 1U> zeroCoordinate{0U};
  const std::array<Index, 1U> positiveCoordinate{1U};
  const auto zeroMarginal = partialCorrelation(
      zeroVarianceWithPositiveCoordinate.view(), zeroCoordinate,
      positiveCoordinate, policy, ampleLimits());
  VSS_CHECK(!zeroMarginal.hasValue());
  VSS_CHECK(zeroMarginal.status().code() == ErrorCode::InvalidDomain);
  auto supportedPolicy = policy;
  supportedPolicy.rank = RankPolicy::supported_subspace;
  const auto zeroRankCondition = partialCorrelation(
      zeroVarianceWithPositiveCoordinate.view(), positiveCoordinate,
      zeroCoordinate, supportedPolicy, ampleLimits());
  VSS_CHECK(zeroRankCondition.hasValue());
  VSS_CHECK(zeroRankCondition.value().rank == 1U);
  VSS_CHECK(zeroRankCondition.value().covariance.at(0U, 0U) == 1.0);

  const std::array<Index, 2U> duplicateSelection{0U, 0U};
  const std::array<Index, 1U> outOfRange{3U};
  const auto duplicate = partialCorrelation(joint.view(), duplicateSelection,
                                            conditioned, policy, ampleLimits());
  VSS_CHECK(!duplicate.hasValue());
  VSS_CHECK(duplicate.status().code() == ErrorCode::InvalidDomain);
  const auto invalidIndex = partialCorrelation(
      joint.view(), selected, outOfRange, policy, ampleLimits());
  VSS_CHECK(!invalidIndex.hasValue());
  VSS_CHECK(invalidIndex.status().code() == ErrorCode::IndexOutOfRange);
  const auto noSelection =
      partialCorrelation(joint.view(), {}, conditioned, policy, ampleLimits());
  VSS_CHECK(!noSelection.hasValue());
  VSS_CHECK(noSelection.status().code() == ErrorCode::InsufficientSamples);

  const std::array<Index, 2U> reverse{1U, 0U};
  Matrix reorderedInput = joint;
  reorderedInput.at(0U, 2U) = 1.0;
  reorderedInput.at(2U, 0U) = 1.0;
  const auto reordered = partialCorrelation(reorderedInput.view(), reverse,
                                            conditioned, policy, ampleLimits());
  VSS_CHECK(reordered.hasValue());
  close(reordered.value().covariance.at(0U, 1U), 1.5 / std::sqrt(3.0 * 3.75));

  auto noIterations = policy;
  noIterations.iteration.max_iterations = 0U;
  const auto noConvergence =
      correlationMatrix(joint.view(), noIterations, ampleLimits());
  VSS_CHECK(!noConvergence.hasValue());
  VSS_CHECK(noConvergence.status().code() == ErrorCode::NoConvergence);
  const auto partialNoConvergence = partialCorrelation(
      joint.view(), selected, conditioned, noIterations, ampleLimits());
  VSS_CHECK(!partialNoConvergence.hasValue());
  VSS_CHECK(partialNoConvergence.status().code() == ErrorCode::NoConvergence);

  auto invalidPolicy = policy;
  invalidPolicy.rank_rel = -1.0;
  const auto invalidTolerance =
      correlationMatrix(singular.view(), invalidPolicy, ampleLimits());
  VSS_CHECK(!invalidTolerance.hasValue());
  VSS_CHECK(invalidTolerance.status().code() == ErrorCode::InvalidTolerance);
  const auto invalidPartialTolerance = partialCorrelation(
      joint.view(), selected, conditioned, invalidPolicy, ampleLimits());
  VSS_CHECK(!invalidPartialTolerance.hasValue());
  VSS_CHECK(invalidPartialTolerance.status().code() ==
            ErrorCode::InvalidTolerance);

  const auto deniedPartial = partialCorrelation(
      joint.view(), selected, conditioned, policy, Limits{0U, 0U, 0U});
  VSS_CHECK(!deniedPartial.hasValue());
  VSS_CHECK(deniedPartial.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  scalarPopulationAndSampleCovariance();
  knownCenterAndCrossCovariance();
  crossCovariancePreservesOrientationAndCommonRows();
  rejectsMismatchedSupportAndInsufficientSample();
  completeRowMaskUsesCommonSupport();
  enforcesOutputLimitsAndRejectsNonFiniteRetainedValues();
  correlationAndPartialCorrelationAcceptanceCases();
}
