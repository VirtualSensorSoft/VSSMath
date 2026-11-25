#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vss/math/statistics/EstimatorDiagnostics.hpp>
#include <vss/math/statistics/Weighted.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

EstimatorDiagnostics diagnosticsWith(Support support) {
  return EstimatorDiagnostics{.estimand = Estimand::observed,
                              .weight = WeightRule::empirical,
                              .center = CenterRule::estimated,
                              .approximation = Approximation::exact,
                              .global = std::move(support),
                              .coordinate_support = {},
                              .tuple_support = {},
                              .reciprocal_condition = std::nullopt,
                              .numerical_error = std::nullopt,
                              .sampling_error = std::nullopt,
                              .rank = std::nullopt,
                              .algorithm = "weighted covariance",
                              .premises = "empirical row weights",
                              .warnings = {}};
}

Support unweightedSupport(Index count) {
  const Scalar n = static_cast<Scalar>(count);
  return Support{
      .input = count,
      .valid = count,
      .positive = count,
      .frequency_mass = std::nullopt,
      .weight_mass = n,
      .normalized_weight_powers =
          count == 0U
              ? std::array<std::optional<Scalar>, 3U>{}
              : std::array<std::optional<Scalar>, 3U>{1.0 / n, 1.0 / (n * n),
                                                      1.0 / (n * n * n)},
      .effective_count = count == 0U ? std::nullopt : std::optional<Scalar>{n},
      .degrees_of_freedom = std::nullopt,
      .squared_weight_mass = n};
}

void diagnosticsValidatorAcceptsExactWeightSupportAndOptionalFields() {
  Support support{
      .input = 3U,
      .valid = 3U,
      .positive = 2U,
      .frequency_mass = std::nullopt,
      .weight_mass = 3.0,
      .normalized_weight_powers = {5.0 / 9.0, 1.0 / 3.0, 17.0 / 81.0},
      .effective_count = 9.0 / 5.0,
      .degrees_of_freedom = std::nullopt,
      .squared_weight_mass = 5.0};
  auto diagnostics = diagnosticsWith(support);
  diagnostics.reciprocal_condition = 0.5;
  diagnostics.rank = 2U;
  diagnostics.numerical_error = 1.0e-12;
  diagnostics.sampling_error = 0.1;
  diagnostics.premises = "empirical weights under a calibrated sampling model";
  const auto result = validateEstimatorDiagnostics(diagnostics);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().global.normalized_weight_powers[0] == 5.0 / 9.0);
  VSS_CHECK(result.value().global.normalized_weight_powers[1] == 1.0 / 3.0);
  VSS_CHECK(result.value().global.normalized_weight_powers[2] == 17.0 / 81.0);
}

void diagnosticsValidatorRejectsContradictoryAndFatalRecords() {
  auto badCounts = diagnosticsWith(unweightedSupport(2U));
  badCounts.global.valid = 3U;
  const auto counts = validateEstimatorDiagnostics(badCounts);
  VSS_CHECK(!counts.hasValue());
  VSS_CHECK(counts.status().code() == ErrorCode::InvalidDomain);

  auto badPowers = diagnosticsWith(unweightedSupport(3U));
  badPowers.global.normalized_weight_powers = {0.4, 0.5, 0.2};
  const auto powers = validateEstimatorDiagnostics(badPowers);
  VSS_CHECK(!powers.hasValue());
  VSS_CHECK(powers.status().code() == ErrorCode::InvalidDomain);

  auto badDof = diagnosticsWith(unweightedSupport(3U));
  badDof.weight = WeightRule::unweighted;
  badDof.algorithm = "unweighted iid covariance";
  badDof.premises = "independent identically distributed rows";
  badDof.global.degrees_of_freedom = 3.0;
  const auto dof = validateEstimatorDiagnostics(badDof);
  VSS_CHECK(!dof.hasValue());
  VSS_CHECK(dof.status().code() == ErrorCode::InvalidDomain);

  auto badCondition = diagnosticsWith(unweightedSupport(3U));
  badCondition.reciprocal_condition = 1.01;
  const auto condition = validateEstimatorDiagnostics(badCondition);
  VSS_CHECK(!condition.hasValue());

  auto badError = diagnosticsWith(unweightedSupport(3U));
  badError.numerical_error = -1.0;
  const auto error = validateEstimatorDiagnostics(badError);
  VSS_CHECK(!error.hasValue());

  auto fatalWarning = diagnosticsWith(unweightedSupport(3U));
  const std::array<ErrorCode, 1U> singular{ErrorCode::Singular};
  fatalWarning.warnings =
      Array<ErrorCode>::copy(singular, {4U, 64U, 0U}).value();
  const auto warning = validateEstimatorDiagnostics(fatalWarning);
  VSS_CHECK(!warning.hasValue());
  VSS_CHECK(warning.status().code() == ErrorCode::InvalidDomain);
}

void diagnosticsValidatorRequiresSortedUniqueTupleSupport() {
  auto diagnostics = diagnosticsWith(unweightedSupport(4U));
  const std::array<Index, 2U> firstCoordinates{0U, 1U};
  const std::array<Index, 2U> secondCoordinates{1U, 2U};
  const std::array<TupleSupport, 2U> ordered{
      {{.coordinates =
            Array<Index>::copy(firstCoordinates, {4U, 64U, 0U}).value(),
        .support = unweightedSupport(3U)},
       {.coordinates =
            Array<Index>::copy(secondCoordinates, {4U, 64U, 0U}).value(),
        .support = unweightedSupport(2U)}}};
  diagnostics.tuple_support =
      Array<TupleSupport>::copy(ordered, {8U, 1024U, 0U}).value();
  VSS_CHECK(validateEstimatorDiagnostics(diagnostics).hasValue());
  const auto tupleView = diagnostics.tuple_support.view();
  const std::array<TupleSupport, 2U> reversed{tupleView[1U], tupleView[0U]};
  diagnostics.tuple_support =
      Array<TupleSupport>::copy(reversed, {8U, 1024U, 0U}).value();
  const auto unsorted = validateEstimatorDiagnostics(diagnostics);
  VSS_CHECK(!unsorted.hasValue());
}

MomentRequest firstComponentMoment() {
  return MomentRequest{.orders = 0x01U,
                       .representation = MomentRepresentation::component,
                       .selected = {},
                       .contraction = {}};
}

void extremeFiniteWeightsKeepNormalizedPowersWhenRawMassOverflows() {
  Matrix values(2U, 1U);
  values.at(0U, 0U) = 1.0;
  values.at(1U, 0U) = 3.0;
  const std::array<Scalar, 2U> rawWeights{std::numeric_limits<Scalar>::max(),
                                          std::numeric_limits<Scalar>::max()};
  const auto view = ConstVectorView::create(rawWeights, 2U);
  VSS_CHECK(view.hasValue());
  if (!view) {
    return;
  }
  const WeightSpec weights{.rule = WeightRule::empirical,
                           .rowWeights = view.value(),
                           .componentWeights = std::nullopt,
                           .frequencyCounts = {}};
  const auto result = weightedMean(SampleView{values.view(), {}, {}}, weights,
                                   Limits{128U, 16384U, 0U});
  VSS_CHECK(result.hasValue());
  if (result) {
    const Support& support = result.value().diagnostics.global;
    VSS_CHECK(!support.weight_mass.has_value());
    VSS_CHECK(!support.squared_weight_mass.has_value());
    VSS_CHECK(support.normalized_weight_powers[0] == 0.5);
    VSS_CHECK(support.normalized_weight_powers[1] == 0.25);
    VSS_CHECK(support.normalized_weight_powers[2] == 0.125);
    VSS_CHECK(support.effective_count == 2.0);
  }
}

void frequencyDiagnosticsDistinguishPhysicalRowsAndReplications() {
  Matrix values(3U, 1U);
  values.at(0U, 0U) = 1.0;
  values.at(1U, 0U) = 2.0;
  values.at(2U, 0U) = 4.0;
  const std::array<std::uint64_t, 3U> counts{1U, 2U, 0U};
  const auto estimate = frequencyWeightedCovariance(
      SampleView{values.view(), {}, {}}, counts, VarianceMode::empirical,
      Limits{256U, 32768U, 0U});
  VSS_CHECK(estimate.hasValue());
  if (estimate) {
    const Support& support = estimate.value().diagnostics.global;
    VSS_CHECK(support.input == 3U && support.valid == 3U &&
              support.positive == 2U);
    VSS_CHECK(support.frequency_mass == 3U);
    VSS_CHECK(support.weight_mass == 3.0);
    VSS_CHECK(support.squared_weight_mass == 5.0);
    VSS_CHECK(support.effective_count == 3.0);
    VSS_CHECK(support.normalized_weight_powers[0] == 1.0 / 3.0);
    VSS_CHECK(support.normalized_weight_powers[1] == 1.0 / 9.0);
    VSS_CHECK(support.normalized_weight_powers[2] == 1.0 / 27.0);
  }
}

void groupedStatisticsKeepsEachGroupsWeightsAndSupportSeparate() {
  Matrix values(4U, 1U);
  values.at(0U, 0U) = 1.0;
  values.at(1U, 0U) = 2.0;
  values.at(2U, 0U) = 4.0;
  values.at(3U, 0U) = 8.0;
  const std::array<Index, 4U> groupIds{8U, 3U, 8U, 3U};
  const std::array<Scalar, 4U> weights{1.0, 1.0, 2.0, 3.0};
  const WeightSpec weightSpec{.rule = WeightRule::empirical,
                              .rowWeights =
                                  ConstVectorView::create(weights, 4U).value(),
                              .componentWeights = std::nullopt,
                              .frequencyCounts = {}};
  const auto result =
      groupedStatistics(SampleView{values.view(), {}, {}}, groupIds, weightSpec,
                        firstComponentMoment(), Limits{256U, 32768U, 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().value.size() == 2U);
  VSS_CHECK(result.value().value.at(0U) == 6.5);
  VSS_CHECK(result.value().value.at(1U) == 3.0);
  VSS_CHECK(!result.value().diagnostics.global.weight_mass.has_value());
  VSS_CHECK(result.value().diagnostics.coordinate_support.view().size() == 2U);
  const auto supports = result.value().diagnostics.coordinate_support.view();
  VSS_CHECK(supports[0].input == 2U && supports[0].valid == 2U);
  VSS_CHECK(supports[0].weight_mass == 4.0);
  VSS_CHECK(supports[1].input == 2U && supports[1].valid == 2U);
  VSS_CHECK(supports[1].weight_mass == 3.0);
}

void groupedStatisticsMapsComponentSupportToFlattenedCoordinates() {
  Matrix values(2U, 2U);
  values.at(0U, 0U) = 1.0;
  values.at(0U, 1U) = 10.0;
  values.at(1U, 0U) = 3.0;
  values.at(1U, 1U) = 20.0;
  Matrix componentWeights(2U, 2U);
  componentWeights.at(0U, 0U) = 1.0;
  componentWeights.at(0U, 1U) = 3.0;
  componentWeights.at(1U, 0U) = 1.0;
  componentWeights.at(1U, 1U) = 1.0;
  const std::array<Index, 2U> groupIds{7U, 7U};
  const std::array<Index, 1U> shape{2U};
  const WeightSpec weights{.rule = WeightRule::componentwise,
                           .rowWeights = std::nullopt,
                           .componentWeights = componentWeights.view(),
                           .frequencyCounts = {}};
  const auto result =
      groupedStatistics(SampleView{values.view(), shape, {}}, groupIds, weights,
                        firstComponentMoment(), Limits{256U, 32768U, 0U});
  VSS_CHECK(result.hasValue());
  if (result) {
    VSS_CHECK(result.value().value.size() == 2U);
    VSS_CHECK(result.value().value.at(0U) == 2.0);
    VSS_CHECK(result.value().value.at(1U) == 12.5);
    const auto support = result.value().diagnostics.coordinate_support.view();
    VSS_CHECK(support.size() == 2U);
    VSS_CHECK(support[0].weight_mass == 2.0);
    VSS_CHECK(support[1].weight_mass == 4.0);
  }
}

void groupedStatisticsCountsCandidatesAndFailsEmptySupport() {
  Matrix values(3U, 1U);
  values.at(0U, 0U) = 2.0;
  values.at(1U, 0U) = 4.0;
  values.at(2U, 0U) = 8.0;
  const std::array<Index, 3U> groupIds{1U, 1U, 2U};
  const std::array<std::uint8_t, 3U> valid{1U, 0U, 0U};
  const WeightSpec weights{.rule = WeightRule::unweighted,
                           .rowWeights = std::nullopt,
                           .componentWeights = std::nullopt,
                           .frequencyCounts = {}};
  const auto result =
      groupedStatistics(SampleView{values.view(), {}, valid}, groupIds, weights,
                        firstComponentMoment(), Limits{256U, 32768U, 0U});
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::InsufficientSamples);

  const std::array<Index, 2U> wrongIds{1U, 1U};
  const auto mismatch =
      groupedStatistics(SampleView{values.view(), {}, {}}, wrongIds, weights,
                        firstComponentMoment(), Limits{256U, 32768U, 0U});
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);
}

} // namespace

int main() {
  diagnosticsValidatorAcceptsExactWeightSupportAndOptionalFields();
  diagnosticsValidatorRejectsContradictoryAndFatalRecords();
  diagnosticsValidatorRequiresSortedUniqueTupleSupport();
  frequencyDiagnosticsDistinguishPhysicalRowsAndReplications();
  extremeFiniteWeightsKeepNormalizedPowersWhenRawMassOverflows();
  groupedStatisticsKeepsEachGroupsWeightsAndSupportSeparate();
  groupedStatisticsMapsComponentSupportToFlattenedCoordinates();
  groupedStatisticsCountsCandidatesAndFailsEmptySupport();
}
