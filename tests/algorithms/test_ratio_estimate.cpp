#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/RatioEstimate.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

RatioModel scalarModel(Index rows) {
  return RatioModel{.rows = rows,
                    .observation_shape = {},
                    .parameter_count = 0U,
                    .observations =
                        [rows](ConstVectorView joint) {
                          Vector values(rows);
                          for (Index i = 0U; i < rows; ++i) {
                            values.at(i) = joint.at(i);
                          }
                          return Result<Vector>::success(std::move(values));
                        },
                    .weights =
                        [rows](ConstVectorView joint) {
                          Vector values(rows);
                          for (Index i = 0U; i < rows; ++i) {
                            values.at(i) = joint.at(rows + i);
                          }
                          return Result<Vector>::success(std::move(values));
                        }};
}

JointSampler scriptedSampler(std::span<const std::array<Scalar, 2U>> script) {
  return [script, next = Index{0U}](probability::Rng&) mutable {
    Vector joint(2U);
    const auto draw = script.subspan(next, 1U).front();
    joint.at(0U) = draw[0U];
    joint.at(1U) = draw[1U];
    ++next;
    return Result<Vector>::success(std::move(joint));
  };
}

RatioModel pairedScalarModel() {
  return RatioModel{.rows = 2U,
                    .observation_shape = {},
                    .parameter_count = 0U,
                    .observations =
                        [](ConstVectorView joint) {
                          Vector values(2U);
                          values.at(0U) = joint.at(0U);
                          values.at(1U) = joint.at(1U);
                          return Result<Vector>::success(std::move(values));
                        },
                    .weights =
                        [](ConstVectorView joint) {
                          Vector values(2U);
                          values.at(0U) = joint.at(2U);
                          values.at(1U) = joint.at(3U);
                          return Result<Vector>::success(std::move(values));
                        }};
}

JointSampler
scriptedJointSampler(std::span<const std::array<Scalar, 4U>> script) {
  return [script, next = Index{0U}](probability::Rng&) mutable {
    Vector joint(4U);
    const auto draw = script.subspan(next, 1U).front();
    joint.at(0U) = draw[0U];
    joint.at(1U) = draw[1U];
    joint.at(2U) = draw[2U];
    joint.at(3U) = draw[3U];
    ++next;
    return Result<Vector>::success(std::move(joint));
  };
}

void directDifferentialReturnsTheExactQuotientJacobianAndHessian() {
  Matrix values(2U, 1U);
  values.at(0U, 0U) = 1.0;
  values.at(1U, 0U) = 3.0;
  const std::array<Scalar, 2U> weights{1.0, 1.0};
  const SampleView sample{
      .rows = values.view(), .observation_shape = {}, .valid = {}};
  const auto result = weightedRatioDifferential(
      sample, ConstVectorView::create(weights, 2U).value(),
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 32U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().value.at(0U) == 2.0);
  const std::array<Scalar, 4U> jacobian{0.5, 0.5, -0.5, 0.5};
  for (Index i = 0U; i < jacobian.size(); ++i) {
    VSS_CHECK(result.value().jacobian.at(0U, i) == jacobian.at(i));
  }
  const auto& hessian = result.value().hessian;
  VSS_CHECK(hessian.has_value());
  if (hessian.has_value()) {
    const std::array<Index, 3U> x1w1{0U, 0U, 2U};
    const std::array<Index, 3U> x1w2{0U, 0U, 3U};
    const std::array<Index, 3U> w1w1{0U, 2U, 2U};
    const std::array<Index, 3U> w2w2{0U, 3U, 3U};
    VSS_CHECK(hessian.value().at(x1w1) == 0.25);
    VSS_CHECK(hessian.value().at(x1w2) == -0.25);
    VSS_CHECK(hessian.value().at(w1w1) == 0.5);
    VSS_CHECK(hessian.value().at(w2w2) == -0.5);
  }
}

void directDifferentialRejectsExtentWeightsAndZeroDenominators() {
  Matrix values(2U, 1U, 1.0);
  const SampleView sample{
      .rows = values.view(), .observation_shape = {}, .valid = {}};
  const std::array<Scalar, 1U> oneWeight{1.0};
  const std::array<Scalar, 2U> negative{1.0, -1.0};
  const std::array<Scalar, 2U> zero{0.0, 0.0};
  const auto extent = weightedRatioDifferential(
      sample, ConstVectorView::create(oneWeight, 1U).value(),
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 32U});
  VSS_CHECK(!extent.hasValue());
  VSS_CHECK(extent.status().code() == ErrorCode::DimensionMismatch);
  const auto invalid = weightedRatioDifferential(
      sample, ConstVectorView::create(negative, 2U).value(),
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 32U});
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::InvalidDomain);
  const auto empty = weightedRatioDifferential(
      sample, ConstVectorView::create(zero, 2U).value(),
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 32U});
  VSS_CHECK(!empty.hasValue());
  VSS_CHECK(empty.status().code() == ErrorCode::InvalidDomain);
}

void directDifferentialAcceptsAnExplicitAllValidMask() {
  Matrix values(2U, 1U);
  values.at(0U, 0U) = 1.0;
  values.at(1U, 0U) = 3.0;
  const std::array<std::uint8_t, 2U> valid{1U, 1U};
  const std::array<Scalar, 2U> weights{1.0, 1.0};
  const SampleView sample{
      .rows = values.view(), .observation_shape = {}, .valid = valid};
  const auto result = weightedRatioDifferential(
      sample, ConstVectorView::create(weights, 2U).value(),
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 32U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().value.at(0U) == 2.0);
}

void propagationRetainsJointCrossCovarianceAndStrictFloor() {
  const auto model = scalarModel(2U);
  Vector center(4U);
  center.at(0U) = 0.0;
  center.at(1U) = 2.0;
  center.at(2U) = 1.0;
  center.at(3U) = 1.0;
  const std::array<Scalar, 4U> direction{1.0, 0.0, 1.0, 0.0};
  Matrix covariance(4U, 4U);
  for (Index row = 0U; row < 4U; ++row) {
    for (Index column = 0U; column < 4U; ++column) {
      covariance.at(row, column) = direction.at(row) * direction.at(column);
    }
  }
  const JointUncertaintyInput input{center.view(), covariance.view(), {}};
  const PropagationPolicy policy{.method = Approximation::first_order,
                                 .denominator_floor = 1.0,
                                 .solve = {}};
  const auto result = propagateWeightedRatio(
      model, input, policy,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 512U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().mean.at(0U) == 1.0);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U)) < 1.0e-12);
  VSS_CHECK(!result.value().diagnostics.global.effective_count.has_value());
  VSS_CHECK(result.value().diagnostics.global.normalized_weight_powers[0] ==
            0.5);
  VSS_CHECK(result.value().diagnostics.global.normalized_weight_powers[1] ==
            0.25);
  VSS_CHECK(result.value().diagnostics.global.normalized_weight_powers[2] ==
            0.125);

  const PropagationPolicy equality{.method = Approximation::first_order,
                                   .denominator_floor = 2.0,
                                   .solve = {}};
  const auto rejected = propagateWeightedRatio(
      model, input, equality,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 512U});
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::InvalidDomain);
}

void propagationIncludesADeclaredValueWeightCrossTerm() {
  const auto model = scalarModel(2U);
  Vector center(4U);
  center.at(0U) = 0.0;
  center.at(1U) = 2.0;
  center.at(2U) = 1.0;
  center.at(3U) = 1.0;
  Matrix covariance(4U, 4U);
  covariance.at(0U, 0U) = 3.0;
  covariance.at(2U, 2U) = 1.0;
  covariance.at(0U, 2U) = 0.5;
  covariance.at(2U, 0U) = 0.5;
  const JointUncertaintyInput input{center.view(), covariance.view(), {}};
  const PropagationPolicy policy{.method = Approximation::first_order,
                                 .denominator_floor = 0.0,
                                 .solve = {}};
  const auto result = propagateWeightedRatio(
      model, input, policy,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 512U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 0.75) < 1.0e-6);
}

void propagationRejectsShapeMismatchAndInadmissibleDifferenceStencil() {
  const auto validModel = scalarModel(2U);
  const RatioModel mismatched{
      .rows = 2U,
      .observation_shape = {},
      .parameter_count = 0U,
      .observations =
          [](ConstVectorView) {
            return Result<Vector>::success(Vector(1U, 1.0));
          },
      .weights =
          [](ConstVectorView) {
            return Result<Vector>::success(Vector(2U, 1.0));
          }};
  Vector center(4U, 1.0);
  Matrix covariance(4U, 4U);
  const JointUncertaintyInput input{center.view(), covariance.view(), {}};
  const PropagationPolicy policy{.method = Approximation::first_order,
                                 .denominator_floor = 0.0,
                                 .solve = {}};
  const auto shape = propagateWeightedRatio(
      mismatched, input, policy,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 512U});
  VSS_CHECK(!shape.hasValue());
  VSS_CHECK(shape.status().code() == ErrorCode::DimensionMismatch);

  Vector nearFloor(4U);
  nearFloor.at(0U) = 0.0;
  nearFloor.at(1U) = 2.0;
  nearFloor.at(2U) = 1.0;
  nearFloor.at(3U) = std::nextafter(0.0, 1.0);
  // The positive floor leaves no admissible finite-difference stencil around
  // this center; the implementation must fail instead of moving the stencil.
  Matrix covariance2(4U, 4U);
  const JointUncertaintyInput nearInput{
      nearFloor.view(), covariance2.view(), {}};
  const PropagationPolicy strict{.method = Approximation::first_order,
                                 .denominator_floor = 1.0,
                                 .solve = {}};
  const auto stencil = propagateWeightedRatio(
      validModel, nearInput, strict,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 512U});
  VSS_CHECK(!stencil.hasValue());
  VSS_CHECK(stencil.status().code() == ErrorCode::InvalidDomain);
}

void simulationPreservesJointDrawsAndReportsConditionalSupport() {
  const auto model = scalarModel(1U);
  const std::array<std::array<Scalar, 2U>, 3U> script{
      {{{7.0, 0.0}}, {{1.0, 1.0}}, {{4.0 / 3.0, 1.0}}}};
  probability::Rng rng(12U);
  const SimPolicy policy{.draws = 3U,
                         .batches = 0U,
                         .quantiles = {},
                         .failure = InvalidDraw::report_conditional,
                         .output_central_moments = {},
                         .denominator_floor = 0.0};
  const auto result = simulateWeightedRatio(
      model, scriptedSampler(script), policy, rng,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().valid == 2U);
  VSS_CHECK(result.value().rejected == 1U);
  VSS_CHECK(std::abs(result.value().estimate.mean.at(0U) - (7.0 / 6.0)) <
            1.0e-12);
  VSS_CHECK(std::abs(result.value().estimate.covariance.at(0U, 0U) -
                     (1.0 / 36.0)) < 1.0e-12);
  VSS_CHECK(!result.value().central_moments.has_value());

  const std::array<std::array<Scalar, 2U>, 2U> emptyScript{
      {{{1.0, 0.0}}, {{2.0, 0.0}}}};
  const SimPolicy conditional{.draws = 2U,
                              .batches = 0U,
                              .quantiles = {},
                              .failure = InvalidDraw::report_conditional,
                              .output_central_moments = {},
                              .denominator_floor = 0.0};
  const auto empty = simulateWeightedRatio(
      model, scriptedSampler(emptyScript), conditional, rng,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 64U});
  VSS_CHECK(!empty.hasValue());
  VSS_CHECK(empty.status().code() == ErrorCode::InsufficientSamples);
}

void simulationUsesOneJointSamplerAndExactEmpiricalSummaries() {
  const auto model = scalarModel(1U);
  const std::array<std::array<Scalar, 2U>, 4U> script{
      {{{1.0, 1.0}}, {{2.0, 1.0}}, {{3.0, 1.0}}, {{4.0, 1.0}}}};
  const std::array<Scalar, 1U> quantiles{0.5};
  const MomentRequest secondMoment{.orders = 0x02U,
                                   .representation =
                                       MomentRepresentation::dense,
                                   .selected = {},
                                   .contraction = {}};
  const SimPolicy policy{.draws = 4U,
                         .batches = 2U,
                         .quantiles = quantiles,
                         .failure = InvalidDraw::reject,
                         .output_central_moments = secondMoment,
                         .denominator_floor = 0.0};
  probability::Rng rng(5U);
  const auto result = simulateWeightedRatio(
      model, scriptedSampler(script), policy, rng,
      Limits{.elements = 512U, .bytes = 65536U, .evaluations = 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().valid == 4U);
  VSS_CHECK(result.value().rejected == 0U);
  VSS_CHECK(result.value().estimate.mean.at(0U) == 2.5);
  VSS_CHECK(result.value().estimate.covariance.at(0U, 0U) == 1.25);
  VSS_CHECK(result.value().quantiles.rows() == 1U);
  VSS_CHECK(result.value().quantiles.columns() == 1U);
  VSS_CHECK(result.value().quantiles.at(0U, 0U) == 2.5);
  VSS_CHECK(result.value().monte_carlo_error.size() == 1U);
  VSS_CHECK(std::abs(result.value().monte_carlo_error.at(0U) - 1.0) < 1.0e-12);
  VSS_CHECK(result.value().central_moments.has_value());
}

void simulationEnforcesAdjacentRepresentableDenominatorCutoff() {
  const auto model = scalarModel(1U);
  const std::array<std::array<Scalar, 2U>, 1U> below{
      {{{2.0, std::nextafter(1.0, 0.0)}}}};
  const std::array<std::array<Scalar, 2U>, 1U> equal{{{{2.0, 1.0}}}};
  const std::array<std::array<Scalar, 2U>, 1U> above{
      {{{2.0, std::nextafter(1.0, std::numeric_limits<Scalar>::infinity())}}}};
  const SimPolicy policy{.draws = 1U,
                         .batches = 0U,
                         .quantiles = {},
                         .failure = InvalidDraw::reject,
                         .output_central_moments = {},
                         .denominator_floor = 1.0};
  probability::Rng rng(8U);
  const auto lower = simulateWeightedRatio(
      model, scriptedSampler(below), policy, rng,
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 16U});
  const auto same = simulateWeightedRatio(
      model, scriptedSampler(equal), policy, rng,
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 16U});
  const auto higher = simulateWeightedRatio(
      model, scriptedSampler(above), policy, rng,
      Limits{.elements = 128U, .bytes = 8192U, .evaluations = 16U});
  VSS_CHECK(!lower.hasValue() &&
            lower.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!same.hasValue() &&
            same.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(higher.hasValue());
  VSS_CHECK(higher.value().estimate.mean.at(0U) == 2.0);
}

void simulationPreservesTheJointPairingAcrossCoordinates() {
  const RatioModel model = pairedScalarModel();
  const std::array<Scalar, 4U> latent{-1.0, 0.0, 0.0, 2.0};
  std::array<std::array<Scalar, 4U>, 16U> paired{};
  std::array<std::array<Scalar, 4U>, 16U> swapped{};
  Index row = 0U;
  for (const Scalar left : latent) {
    for (const Scalar right : latent) {
      paired.at(row) = {1.0 + left, 1.0 + right, 2.0 + left, 2.0 + right};
      swapped.at(row) = {1.0 + left, 1.0 + right, 2.0 + right, 2.0 + left};
      ++row;
    }
  }
  const SimPolicy policy{.draws = 16U,
                         .batches = 0U,
                         .quantiles = {},
                         .failure = InvalidDraw::reject,
                         .output_central_moments = {},
                         .denominator_floor = 0.0};
  probability::Rng rng(10U);
  const auto result = simulateWeightedRatio(
      model, scriptedJointSampler(paired), policy, rng,
      Limits{.elements = 512U, .bytes = 65536U, .evaluations = 64U});
  const auto repaired = simulateWeightedRatio(
      model, scriptedJointSampler(swapped), policy, rng,
      Limits{.elements = 512U, .bytes = 65536U, .evaluations = 64U});
  VSS_CHECK(result.hasValue() && repaired.hasValue());
  VSS_CHECK(std::abs(result.value().estimate.mean.at(0U) - (119.0 / 80.0)) <
            1.0e-12);
  VSS_CHECK(std::abs(result.value().estimate.covariance.at(0U, 0U) -
                     (45623.0 / 57600.0)) < 1.0e-12);
  VSS_CHECK(std::abs(repaired.value().estimate.mean.at(0U) - (81.0 / 80.0)) <
            1.0e-12);
  VSS_CHECK(std::abs(repaired.value().estimate.covariance.at(0U, 0U) -
                     (31943.0 / 57600.0)) < 1.0e-12);
}

void simulationDistinguishesMeanOfRatiosFromRatioOfExpectations() {
  const auto model = pairedScalarModel();
  const std::array<std::array<Scalar, 4U>, 2U> script{{
      {1.0, 1.0, 1.0, 1.0},
      {4.0 / 3.0, 4.0 / 3.0, 1.5, 1.5},
  }};
  const SimPolicy policy{.draws = 2U,
                         .batches = 0U,
                         .quantiles = {},
                         .failure = InvalidDraw::reject,
                         .output_central_moments = {},
                         .denominator_floor = 0.0};
  probability::Rng rng(11U);
  const auto result = simulateWeightedRatio(
      model, scriptedJointSampler(script), policy, rng,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().estimate.mean.at(0U) - (7.0 / 6.0)) <
            1.0e-12);
  VSS_CHECK(std::abs(result.value().estimate.covariance.at(0U, 0U) -
                     (1.0 / 36.0)) < 1.0e-12);
  const Scalar expectedNumerator =
      (1.0 * 1.0 + 1.0 * 1.0 + 1.5 * (4.0 / 3.0) + 1.5 * (4.0 / 3.0)) / 2.0;
  const Scalar expectedDenominator = (2.0 + 3.0) / 2.0;
  VSS_CHECK(std::abs((expectedNumerator / expectedDenominator) - (6.0 / 5.0)) <
            1.0e-12);
  VSS_CHECK(std::abs(result.value().estimate.mean.at(0U) -
                     (expectedNumerator / expectedDenominator)) > 1.0e-2);
}

void conditionalSimulationUsesOnlyDrawsStrictlyAboveTheFloor() {
  const auto model = scalarModel(1U);
  const std::array<std::array<Scalar, 2U>, 3U> script{{
      {2.0, std::nextafter(1.0, 0.0)},
      {2.0, 1.0},
      {2.0, std::nextafter(1.0, std::numeric_limits<Scalar>::infinity())},
  }};
  const SimPolicy policy{.draws = 3U,
                         .batches = 0U,
                         .quantiles = {},
                         .failure = InvalidDraw::report_conditional,
                         .output_central_moments = {},
                         .denominator_floor = 1.0};
  probability::Rng rng(92U);
  const auto result = simulateWeightedRatio(
      model, scriptedSampler(script), policy, rng,
      Limits{.elements = 256U, .bytes = 32768U, .evaluations = 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().valid == 1U);
  VSS_CHECK(result.value().rejected == 2U);
  VSS_CHECK(result.value().estimate.mean.at(0U) == 2.0);
  VSS_CHECK(result.value().estimate.covariance.at(0U, 0U) == 0.0);
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
  directDifferentialReturnsTheExactQuotientJacobianAndHessian();
  directDifferentialRejectsExtentWeightsAndZeroDenominators();
  directDifferentialAcceptsAnExplicitAllValidMask();
  propagationRetainsJointCrossCovarianceAndStrictFloor();
  propagationIncludesADeclaredValueWeightCrossTerm();
  propagationRejectsShapeMismatchAndInadmissibleDifferenceStencil();
  simulationPreservesJointDrawsAndReportsConditionalSupport();
  simulationUsesOneJointSamplerAndExactEmpiricalSummaries();
  simulationEnforcesAdjacentRepresentableDenominatorCutoff();
  simulationPreservesTheJointPairingAcrossCoordinates();
  simulationDistinguishesMeanOfRatiosFromRatioOfExpectations();
  conditionalSimulationUsesOnlyDrawsStrictlyAboveTheFloor();
}
