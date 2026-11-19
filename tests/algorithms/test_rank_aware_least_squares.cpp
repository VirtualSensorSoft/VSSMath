#include "Test.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

template <typename Enum> Enum invalidEnumValue() {
  volatile const std::uint8_t rawValue = 255U;
  // Deliberately construct an out-of-range value for invalid-policy tests.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(rawValue);
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount, 0.0);
  Index rowIndex = 0U;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == columnCount);
    Index columnIndex = 0U;
    for (const Scalar value : row) {
      result.at(rowIndex, columnIndex++) = value;
    }
    ++rowIndex;
  }
  return result;
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 128U;
  result.rank_rel = 1.0e-12;
  result.rank = RankPolicy::minimum_norm;
  return result;
}

Limits generousLimits() {
  return {
      .elements = 1'000'000U, .bytes = 64U * 1024U * 1024U, .evaluations = 0U};
}

template <class T> void checkFailure(const T& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkNear(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-10) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "actual=" << actual << " expected=" << expected
              << " tolerance=" << tolerance << '\n';
    VSS_CHECK(false);
  }
}

auto solve(const Matrix& design, const Matrix& observations,
           SolvePolicy settings = policy(), Limits limits = generousLimits()) {
  return linear::leastSquaresMinNorm(design.view(), observations.view(),
                                     settings, limits);
}

void wideSystemReturnsMinimumNormAndIdentifiedBasis() {
  const Matrix design = matrix({{1.0, 1.0}});
  const Matrix observations = matrix({{4.0}});
  const auto result = solve(design, observations);
  VSS_CHECK(result.hasValue());
  const auto& estimate = result.value();
  VSS_CHECK(estimate.parameters.rows() == 2U &&
            estimate.parameters.columns() == 1U);
  VSS_CHECK(estimate.rank == 1U);
  checkNear(estimate.rank_cutoff, std::sqrt(2.0) * 1.0e-12, 1.0e-24);
  VSS_CHECK(estimate.identified_basis.has_value());
  VSS_CHECK(estimate.identified_basis->rows() == 2U &&
            estimate.identified_basis->columns() == 1U);
  checkNear(estimate.parameters.at(0U, 0U), 2.0);
  checkNear(estimate.parameters.at(1U, 0U), 2.0);
  checkNear(estimate.residual_norms.at(0U), 0.0);
  const Scalar first = estimate.identified_basis->at(0U, 0U);
  const Scalar second = estimate.identified_basis->at(1U, 0U);
  checkNear(first * first, 0.5);
  checkNear(first * second, 0.5);
  VSS_CHECK(!estimate.covariance.has_value());
  // LSQ-01: wide rank-one system has the equal-split minimum-norm estimate.
}

void cutoffFitReportsOriginalSystemResidualsForEachRightHandSide() {
  Matrix design(3U, 4U, 0.0);
  design.at(0U, 0U) = 4.0;
  design.at(1U, 1U) = 1.0;
  design.at(2U, 2U) = 0.5;
  Matrix observations(3U, 2U, 0.0);
  observations.at(0U, 0U) = 8.0;
  observations.at(1U, 0U) = 3.0;
  observations.at(2U, 0U) = 4.0;
  observations.at(0U, 1U) = -4.0;
  observations.at(2U, 1U) = 2.0;
  auto settings = policy();
  settings.rank_abs = 0.0;
  settings.rank_rel = 0.25;

  const auto result = solve(design, observations, settings);
  VSS_CHECK(result.hasValue());
  const auto& estimate = result.value();
  VSS_CHECK(estimate.rank == 1U && estimate.rank_cutoff == 1.0);
  checkNear(estimate.parameters.at(0U, 0U), 2.0);
  checkNear(estimate.parameters.at(0U, 1U), -1.0);
  for (Index row = 1U; row < 4U; ++row) {
    VSS_CHECK(estimate.parameters.at(row, 0U) == 0.0);
    VSS_CHECK(estimate.parameters.at(row, 1U) == 0.0);
  }
  checkNear(estimate.residual_norms.at(0U), 5.0);
  checkNear(estimate.residual_norms.at(1U), 2.0);
  VSS_CHECK(estimate.identified_basis->rows() == 4U &&
            estimate.identified_basis->columns() == 1U);
  // LSQ-02: exact-cutoff modes are discarded and residuals use original B.
}

void scaledSystemsAndConditionFloorBoundaryRemainValid() {
  const Scalar scale = std::scalbn(1.0, 900);
  Matrix design(2U, 2U, 0.0);
  design.at(0U, 0U) = scale;
  design.at(1U, 1U) = scale * 0.5;
  Matrix observations(2U, 1U, 0.0);
  observations.at(0U, 0U) = scale * 3.0;
  observations.at(1U, 0U) = scale * -2.0;
  auto settings = policy();
  settings.rank_abs = 0.0;
  settings.rank_rel = 0.0;
  settings.reciprocal_condition_floor = 0.5;
  const auto result = solve(design, observations, settings);
  VSS_CHECK(result.hasValue());
  checkNear(result.value().parameters.at(0U, 0U), 3.0);
  checkNear(result.value().parameters.at(1U, 0U), -4.0);
  checkNear(result.value().residual_norms.at(0U), 0.0);
  // LSQ-08: common extreme scaling and an exact condition-floor boundary pass.
}

void rejectPolicyRejectsCutoffRankLoss() {
  Matrix design(3U, 4U, 0.0);
  design.at(0U, 0U) = 4.0;
  design.at(1U, 1U) = 1.0;
  design.at(2U, 2U) = 0.5;
  const Matrix observations(3U, 1U, 1.0);
  auto settings = policy();
  settings.rank_abs = 0.0;
  settings.rank_rel = 0.25;
  settings.rank = RankPolicy::reject;
  checkFailure(solve(design, observations, settings), ErrorCode::RankDeficient);
  // LSQ-03: strict rank policy rejects loss at the inclusive cutoff.
}

void zeroRankReturnsZeroEstimateAndObservationResiduals() {
  const Matrix design(2U, 3U, 0.0);
  const Matrix observations = matrix({{3.0, 0.0}, {3.0, 4.0}});
  const auto result = solve(design, observations);
  VSS_CHECK(result.hasValue());
  const auto& estimate = result.value();
  VSS_CHECK(estimate.rank == 0U);
  VSS_CHECK(estimate.parameters.rows() == 3U &&
            estimate.parameters.columns() == 2U);
  VSS_CHECK(estimate.identified_basis.has_value());
  VSS_CHECK(estimate.identified_basis->rows() == 3U &&
            estimate.identified_basis->columns() == 0U);
  for (Index row = 0U; row < 3U; ++row) {
    VSS_CHECK(estimate.parameters.at(row, 0U) == 0.0);
    VSS_CHECK(estimate.parameters.at(row, 1U) == 0.0);
  }
  checkNear(estimate.residual_norms.at(0U), 3.0 * std::sqrt(2.0));
  checkNear(estimate.residual_norms.at(1U), 4.0);
  // LSQ-04: zero numerical rank yields zero parameters and original residuals.
}

void emptyAndZeroRightHandSideShapesArePreserved() {
  const Matrix noEquations(0U, 3U);
  const Matrix emptyObservations(0U, 2U);
  const auto empty = solve(noEquations, emptyObservations);
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().parameters.rows() == 3U &&
            empty.value().parameters.columns() == 2U);
  VSS_CHECK(empty.value().rank == 0U);
  VSS_CHECK(empty.value().residual_norms.size() == 2U);
  checkNear(empty.value().residual_norms.at(0U), 0.0);
  checkNear(empty.value().residual_norms.at(1U), 0.0);

  const Matrix design = matrix({{4.0, 0.0}, {0.0, 2.0}});
  const Matrix noRightHandSides(2U, 0U);
  const auto emptyColumns = solve(design, noRightHandSides);
  VSS_CHECK(emptyColumns.hasValue());
  VSS_CHECK(emptyColumns.value().parameters.rows() == 2U &&
            emptyColumns.value().parameters.columns() == 0U);
  VSS_CHECK(emptyColumns.value().residual_norms.size() == 0U);
  // LSQ-05: empty observation and RHS axes preserve output shapes.
}

void invalidShapesInputsPoliciesAndLimitsAreRejected() {
  const Matrix design = matrix({{1.0, 0.0}, {0.0, 0.5}});
  checkFailure(solve(design, Matrix(3U, 1U, 1.0)),
               ErrorCode::DimensionMismatch);

  Matrix infiniteDesign = matrix({{1.0}});
  infiniteDesign.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkFailure(solve(infiniteDesign, matrix({{1.0}})),
               ErrorCode::NonFiniteInput);
  checkFailure(solve(matrix({{1.0}}),
                     matrix({{std::numeric_limits<Scalar>::quiet_NaN()}})),
               ErrorCode::NonFiniteInput);

  auto invalid = policy();
  invalid.rank_rel = 1.0;
  checkFailure(solve(design, matrix({{1.0}, {1.0}}), invalid),
               ErrorCode::InvalidTolerance);
  invalid = policy();
  invalid.reciprocal_condition_floor = 1.01;
  checkFailure(solve(design, matrix({{1.0}, {1.0}}), invalid),
               ErrorCode::InvalidTolerance);
  invalid = policy();
  invalid.rank = invalidEnumValue<RankPolicy>();
  checkFailure(solve(design, matrix({{1.0}, {1.0}}), invalid),
               ErrorCode::InvalidDomain);

  auto illConditioned = policy();
  illConditioned.rank_rel = 0.0;
  illConditioned.reciprocal_condition_floor = 0.75;
  checkFailure(solve(design, matrix({{1.0}, {1.0}}), illConditioned),
               ErrorCode::IllConditioned);

  checkFailure(solve(design, matrix({{1.0}, {1.0}}), policy(),
                     Limits{.elements = 1U, .bytes = 1U, .evaluations = 0U}),
               ErrorCode::ResourceLimit);
  // LSQ-06: malformed shapes, nonfinite data, policies, conditioning, limits.
}

void iterationAndUnrepresentableSolutionsFailWithoutPartialResults() {
  Matrix nondiagonal = matrix({{2.0, 1.0}, {1.0, 3.0}});
  auto exhausted = policy();
  exhausted.iteration.max_iterations = 0U;
  checkFailure(solve(nondiagonal, matrix({{1.0}, {2.0}}), exhausted),
               ErrorCode::NoConvergence);

  const Matrix tiny = matrix({{std::numeric_limits<Scalar>::min()}});
  auto noCutoff = policy();
  noCutoff.rank_abs = 0.0;
  noCutoff.rank_rel = 0.0;
  checkFailure(solve(tiny, matrix({{8.0}}), noCutoff),
               ErrorCode::NonFiniteResult);
  // LSQ-07: non-diagonal zero-iteration and unrepresentable solutions fail.
}

} // namespace

int main() {
  wideSystemReturnsMinimumNormAndIdentifiedBasis();
  cutoffFitReportsOriginalSystemResidualsForEachRightHandSide();
  scaledSystemsAndConditionFloorBoundaryRemainValid();
  rejectPolicyRejectsCutoffRankLoss();
  zeroRankReturnsZeroEstimateAndObservationResiduals();
  emptyAndZeroRightHandSideShapesArePreserved();
  invalidShapesInputsPoliciesAndLimitsAreRejected();
  iterationAndUnrepresentableSolutionsFailWithoutPartialResults();
  return 0;
}
