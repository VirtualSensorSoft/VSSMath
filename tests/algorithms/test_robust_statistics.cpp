#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Robust.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Limits ampleLimits() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = std::numeric_limits<std::uint64_t>::max()};
}

Matrix scalarRows(std::span<const Scalar> values) {
  Matrix rows(values.size(), 1U);
  for (Extent i = 0; i < values.size(); ++i) {
    rows.at(i, 0U) = values[i];
  }
  return rows;
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> values) {
  const Extent rows = values.size();
  const Extent columns = rows == 0U ? 0U : values.begin()->size();
  Matrix result(rows, columns, 0.0);
  Extent i = 0U;
  for (const auto& row : values) {
    VSS_CHECK(row.size() == columns);
    Extent j = 0U;
    for (const Scalar value : row) {
      result.at(i, j) = value;
      ++j;
    }
    ++i;
  }
  return result;
}

SampleView sample(const Matrix& rows) {
  return {.rows = rows.view(), .observation_shape = {}, .valid = {}};
}

RobustPolicy policy(std::optional<ConstVectorView> fixedScale = std::nullopt,
                    std::optional<ConstMatrixView> metric = std::nullopt,
                    std::optional<ConstVectorView> center = std::nullopt,
                    Index iterations = 128U) {
  return {.huber_cutoff = 1.0,
          .scale_factor = 1.0,
          .fixed_scale = fixedScale,
          .metric = metric,
          .iteration = {.abs = 1.0e-12,
                        .rel = 1.0e-12,
                        .max_iterations = iterations,
                        .max_evaluations = 0U,
                        .bracket = std::nullopt},
          .center = center};
}

void close(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-10) {
  VSS_CHECK(std::abs(actual - expected) <=
            tolerance * std::max(Scalar{1.0}, std::abs(expected)));
}

void robustScaleUsesMedianAbsoluteDeviation() {
  const Matrix rows = scalarRows(std::array<Scalar, 4>{-1.0, 0.0, 1.0, 100.0});
  const auto result = robustScale(sample(rows), policy(), ampleLimits());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().value.at(0U), 1.0);
  }
}

void robustScaleAcceptsConstantCoordinates() {
  const Matrix rows = scalarRows(std::array<Scalar, 3>{4.0, 4.0, 4.0});
  const auto result = robustScale(sample(rows), policy(), ampleLimits());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().value.at(0U), 0.0);
  }
}

void robustLocationRejectsZeroMadForNonconstantSample() {
  // The exact zero-MAD fixture is tied but nonconstant: its linear-median
  // center is zero and at least half its absolute deviations are zero.
  const Matrix rows = scalarRows(std::array<Scalar, 4>{0.0, 0.0, 0.0, 1.0});
  const auto result = robustLocation(sample(rows), policy(), ampleLimits());
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
  const Vector fixed(std::array<Scalar, 1>{1.0});
  const auto withFixedScale =
      robustLocation(sample(rows), policy(fixed.view()), ampleLimits());
  VSS_CHECK(withFixedScale.hasValue());
  if (withFixedScale) {
    close(withFixedScale.value().value.at(0U), 0.25);
  }
}

void robustLocationAndExtremeResistance() {
  const Matrix clean = scalarRows(std::array<Scalar, 3>{-1.0, 0.0, 1.0});
  const Matrix contaminated =
      scalarRows(std::array<Scalar, 3>{-1.0, 0.0, 1.0e9});
  const Vector fixed(std::array<Scalar, 1>{1.0});
  auto cleanResult =
      robustLocation(sample(clean), policy(fixed.view()), ampleLimits());
  auto outlierResult =
      robustLocation(sample(contaminated), policy(fixed.view()), ampleLimits());
  VSS_CHECK(cleanResult.hasValue() && outlierResult.hasValue());
  if (cleanResult && outlierResult) {
    close(cleanResult.value().value.at(0U), 0.0);
    close(outlierResult.value().value.at(0U), 0.0);
  }
}

void robustLocationHonorsIterationCap() {
  const Matrix rows = scalarRows(std::array<Scalar, 3>{-10.0, 0.0, 10.0});
  const Vector fixed(std::array<Scalar, 1>{10.0});
  const auto result = robustLocation(
      sample(rows), policy(fixed.view(), std::nullopt, std::nullopt, 1U),
      ampleLimits());
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::NoConvergence);
}

void robustEstimatorsRejectMissingCells() {
  const Matrix rows = scalarRows(std::array<Scalar, 3>{1.0, 2.0, 3.0});
  const std::array<std::uint8_t, 3> mask{1U, 0U, 1U};
  const SampleView incomplete{
      .rows = rows.view(), .observation_shape = {}, .valid = mask};
  const auto result = robustScale(incomplete, policy(), ampleLimits());
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
}

void radialScatterUsesDeclaredCenterAndMetric() {
  const Matrix rows = scalarRows(std::array<Scalar, 3>{-10.0, 0.0, 10.0});
  const Vector center(std::array<Scalar, 1>{2.0});
  Matrix metric(1U, 1U, 1.0);
  const auto result = robustCovariance(
      sample(rows), policy(std::nullopt, metric.view(), center.view()),
      ampleLimits());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().value.at(0U, 0U), 1.0);
    VSS_CHECK(result.value().diagnostics.center == CenterRule::known);
  }
}

void radialScatterMatchesClippedVectorReference() {
  const Matrix rows = scalarRows(std::array<Scalar, 3>{-10.0, 0.0, 10.0});
  const Vector center(std::array<Scalar, 1>{0.0});
  Matrix metric(1U, 1U, 1.0);
  const auto result = robustCovariance(
      sample(rows), policy(std::nullopt, metric.view(), center.view()),
      ampleLimits());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().value.at(0U, 0U), 2.0 / 3.0);
  }
}

void radialScatterRejectsSingularMetric() {
  const Matrix rows = matrix({{0.0, 0.0}, {1.0, 1.0}, {-1.0, -1.0}});
  Matrix metric(2U, 2U, 0.0);
  metric.at(0U, 0U) = 1.0;
  const std::array<Extent, 1U> shape{2U};
  const SampleView shaped{
      .rows = rows.view(), .observation_shape = shape, .valid = {}};
  const auto result = robustCovariance(
      shaped, policy(std::nullopt, metric.view()), ampleLimits());
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
}

void covarianceShrinkagePreservesValidatedEndpoints() {
  Matrix source(2U, 2U, 0.0);
  source.at(0U, 0U) = 1.0;
  source.at(1U, 1U) = 3.0;
  Matrix target(2U, 2U, 0.0);
  target.at(0U, 0U) = 2.0;
  target.at(1U, 1U) = 2.0;
  const SolvePolicy solve{.iteration = {},
                          .rank_abs = 0.0,
                          .rank_rel = 1.0e-12,
                          .reciprocal_condition_floor = 0.0,
                          .rank = RankPolicy::reject,
                          .restart = 0U};
  const auto zero =
      shrinkCovariance(source.view(), target.view(), 0.0, solve, ampleLimits());
  const auto one =
      shrinkCovariance(source.view(), target.view(), 1.0, solve, ampleLimits());
  VSS_CHECK(zero.hasValue() && one.hasValue());
  if (zero && one) {
    VSS_CHECK(zero.value().covariance.at(0U, 0U) == 1.0);
    VSS_CHECK(zero.value().covariance.at(1U, 1U) == 3.0);
    VSS_CHECK(one.value().covariance.at(0U, 0U) == 2.0);
    VSS_CHECK(one.value().covariance.at(1U, 1U) == 2.0);
  }
}

void covarianceShrinkageValidatesUnusedEndpoint() {
  Matrix source(2U, 2U, 0.0);
  source.at(0U, 0U) = 1.0;
  source.at(1U, 1U) = 1.0;
  Matrix indefinite(2U, 2U, 0.0);
  indefinite.at(0U, 0U) = 1.0;
  indefinite.at(1U, 1U) = -1.0;
  const SolvePolicy solve{};
  const auto invalidTarget = shrinkCovariance(source.view(), indefinite.view(),
                                              0.0, solve, ampleLimits());
  const auto invalidWeight = shrinkCovariance(source.view(), source.view(),
                                              -0.01, solve, ampleLimits());
  VSS_CHECK(!invalidTarget.hasValue());
  VSS_CHECK(invalidTarget.status().code() ==
            ErrorCode::NotPositiveSemidefinite);
  VSS_CHECK(!invalidWeight.hasValue());
  VSS_CHECK(invalidWeight.status().code() == ErrorCode::InvalidDomain);
}

void covarianceShrinkageMatchesConvexReferenceAndCap() {
  Matrix source(2U, 2U, 0.0);
  source.at(0U, 0U) = 1.0;
  source.at(1U, 1U) = 3.0;
  Matrix target(2U, 2U, 0.0);
  target.at(0U, 0U) = 2.0;
  target.at(1U, 1U) = 2.0;
  const auto result = shrinkCovariance(source.view(), target.view(), 0.5,
                                       SolvePolicy{}, ampleLimits());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().covariance.at(0U, 0U), 1.5);
    close(result.value().covariance.at(1U, 1U), 2.5);
  }
  const auto capped =
      shrinkCovariance(source.view(), target.view(), 0.5, SolvePolicy{},
                       {.elements = 3U, .bytes = 1024U, .evaluations = 0U});
  VSS_CHECK(!capped.hasValue());
  VSS_CHECK(capped.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  robustScaleUsesMedianAbsoluteDeviation();
  robustScaleAcceptsConstantCoordinates();
  robustLocationRejectsZeroMadForNonconstantSample();
  robustLocationAndExtremeResistance();
  robustLocationHonorsIterationCap();
  robustEstimatorsRejectMissingCells();
  radialScatterUsesDeclaredCenterAndMetric();
  radialScatterMatchesClippedVectorReference();
  radialScatterRejectsSingularMetric();
  covarianceShrinkagePreservesValidatedEndpoints();
  covarianceShrinkageValidatesUnusedEndpoint();
  covarianceShrinkageMatchesConvexReferenceAndCap();
}

// VSS-MATH-TRACEABILITY: C037 CASE-C037.scale.mad CASE-C037.scale.constant
// VSS-MATH-TRACEABILITY: C037 CASE-C037.location-zero-mad-tie
// VSS-MATH-TRACEABILITY: C037 CASE-C037.location.huber
// VSS-MATH-TRACEABILITY: C037 CASE-C037.location.single-extreme-contamination
// VSS-MATH-TRACEABILITY: C037 CASE-C037.location-cap
// VSS-MATH-TRACEABILITY: C037 CASE-C037.scale-mask-policy
// VSS-MATH-TRACEABILITY: C037 CASE-C037.covariance-shifted-center
// VSS-MATH-TRACEABILITY: C037 CASE-C037.covariance-invalid-metric
// VSS-MATH-TRACEABILITY: C037 CASE-C037.covariance.radial
// VSS-MATH-TRACEABILITY: C038 CASE-C038.shrink-endpoints
// VSS-MATH-TRACEABILITY: C038 CASE-C038.shrink-invalid-target
// VSS-MATH-TRACEABILITY: C038 CASE-C038.shrink-shape-cap
// VSS-MATH-TRACEABILITY: C038 CASE-C038.shrink.nominal
