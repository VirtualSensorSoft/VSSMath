#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

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
  result.rank = RankPolicy::minimum_norm;
  return result;
}

Limits generousLimits() {
  return {.elements = 1'000'000U,
          .bytes = std::size_t{64} * 1024U * 1024U,
          .evaluations = 0U};
}

template <class T> void checkFailure(const T& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkFactors(const Matrix& input, const linear::FullQR& result,
                  Scalar tolerance = 1.0e-12) {
  VSS_CHECK(result.orthogonal.rows() == input.rows());
  VSS_CHECK(result.orthogonal.columns() == input.rows());
  VSS_CHECK(result.upper.rows() == input.rows());
  VSS_CHECK(result.upper.columns() == input.columns());
  VSS_CHECK(result.permutation.view().size() == input.columns());

  for (Index first = 0U; first < input.rows(); ++first) {
    for (Index second = 0U; second < input.rows(); ++second) {
      Scalar product = 0.0;
      for (Index row = 0U; row < input.rows(); ++row) {
        product += result.orthogonal.at(row, first) *
                   result.orthogonal.at(row, second);
      }
      VSS_CHECK(std::abs(product - (first == second ? 1.0 : 0.0)) <= tolerance);
    }
  }

  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      Scalar reconstructed = 0.0;
      for (Index inner = 0U; inner < input.rows(); ++inner) {
        reconstructed +=
            result.orthogonal.at(row, inner) * result.upper.at(inner, column);
      }
      const Scalar expected = input.at(
          row, result.permutation.view()[static_cast<std::size_t>(column)]);
      VSS_CHECK(std::abs(reconstructed - expected) <=
                tolerance * std::max(1.0, std::abs(expected)));
    }
  }

  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < std::min(row, input.columns()); ++column) {
      VSS_CHECK(result.upper.at(row, column) == 0.0);
    }
  }
}

void tallPivotedFactorHasFullOrthogonalCompletion() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 2.0}, {1.0, 1.0}});
  const auto result = linear::factorFullQR(
      input.view(), linear::QRColumnOrder::pivoted, policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2U);
  VSS_CHECK(result.value().permutation.view()[0U] == 1U);
  VSS_CHECK(result.value().permutation.view()[1U] == 0U);
  checkFactors(input, result.value());
  // CASE-C126.full.pivoted: pivot order and full 3x3 orthogonal completion.
}

void wideOriginalOrderPreservesUpperTrapezoidalShape() {
  const Matrix input = matrix({{1.0, 2.0, -1.0}, {3.0, -1.0, 2.0}});
  const auto result =
      linear::factorFullQR(input.view(), linear::QRColumnOrder::original,
                           policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2U);
  for (Index column = 0U; column < input.columns(); ++column) {
    VSS_CHECK(result.value().permutation.view()[column] == column);
  }
  checkFactors(input, result.value());
  // CASE-C126.full.wide: original order and full square Q.
}

void rankUsesStrictSingularCutoffAndIgnoresSolveRejection() {
  const Matrix input = matrix({{4.0, 0.0}, {0.0, 1.0}});
  auto settings = policy();
  settings.rank_abs = 0.0;
  settings.rank_rel = 0.25;
  settings.rank = RankPolicy::reject;
  settings.reciprocal_condition_floor = 1.0;
  settings.iteration.max_iterations = 0U;

  const auto result =
      linear::factorFullQR(input.view(), linear::QRColumnOrder::original,
                           settings, generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 1U);
  VSS_CHECK(result.value().cutoff == 1.0);
  checkFactors(input, result.value());
  // Equality at the C010 cutoff is excluded by the strict sigma > cutoff rule.
}

void zeroAndEmptyAxesProduceCorrectIdentities() {
  const Matrix noColumns(3U, 0U);
  const auto columns =
      linear::factorFullQR(noColumns.view(), linear::QRColumnOrder::original,
                           policy(), generousLimits());
  VSS_CHECK(columns.hasValue());
  VSS_CHECK(columns.value().rank == 0U);
  VSS_CHECK(columns.value().orthogonal.rows() == 3U &&
            columns.value().orthogonal.columns() == 3U);
  for (Index axis = 0U; axis < 3U; ++axis) {
    for (Index other = 0U; other < 3U; ++other) {
      VSS_CHECK(columns.value().orthogonal.at(axis, other) ==
                (axis == other ? 1.0 : 0.0));
    }
  }

  const Matrix noRows(0U, 3U);
  const auto rows =
      linear::factorFullQR(noRows.view(), linear::QRColumnOrder::pivoted,
                           policy(), generousLimits());
  VSS_CHECK(rows.hasValue());
  VSS_CHECK(rows.value().orthogonal.rows() == 0U &&
            rows.value().orthogonal.columns() == 0U);
  VSS_CHECK(rows.value().upper.rows() == 0U &&
            rows.value().upper.columns() == 3U);
  VSS_CHECK(rows.value().permutation.view().size() == 3U);
  for (Index column = 0U; column < 3U; ++column) {
    VSS_CHECK(rows.value().permutation.view()[column] == column);
  }
  // Empty axes keep the stated extents and identity column order.
}

void equalPivotNormsUseTheSmallestOriginalColumnIndex() {
  const Matrix input =
      matrix({{0.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const auto result = linear::factorFullQR(
      input.view(), linear::QRColumnOrder::pivoted, policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().permutation.view()[0U] == 1U);
  VSS_CHECK(result.value().permutation.view()[1U] == 2U);
  VSS_CHECK(result.value().permutation.view()[2U] == 0U);
  checkFactors(input, result.value());
  // Exact equal residual norms are resolved by original column index.
}

void allZeroInputKeepsFullFactorsAndIdentityPermutation() {
  const Matrix input(2U, 3U, 0.0);
  const auto result = linear::factorFullQR(
      input.view(), linear::QRColumnOrder::pivoted, policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 0U);
  for (Index column = 0U; column < input.columns(); ++column) {
    VSS_CHECK(result.value().permutation.view()[column] == column);
  }
  checkFactors(input, result.value());
  // Zero active tails use identity reflectors without shortening Q or R.
}

void invalidInputsAndResourceOrCertificationExhaustionReturnStatuses() {
  const Matrix valid = matrix({{1.0, 0.0}, {0.0, 2.0}});
  checkFailure(linear::factorFullQR(valid.view(),
                                    static_cast<linear::QRColumnOrder>(255U),
                                    policy(), generousLimits()),
               ErrorCode::InvalidDomain);

  auto invalidRankPolicy = policy();
  invalidRankPolicy.rank = static_cast<RankPolicy>(255U);
  checkFailure(linear::factorFullQR(valid.view(),
                                    linear::QRColumnOrder::original,
                                    invalidRankPolicy, generousLimits()),
               ErrorCode::InvalidDomain);

  auto invalidTolerance = policy();
  invalidTolerance.rank_rel = std::numeric_limits<Scalar>::quiet_NaN();
  checkFailure(linear::factorFullQR(valid.view(),
                                    linear::QRColumnOrder::original,
                                    invalidTolerance, generousLimits()),
               ErrorCode::InvalidTolerance);

  Matrix nonfinite = valid;
  nonfinite.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkFailure(linear::factorFullQR(nonfinite.view(),
                                    linear::QRColumnOrder::original, policy(),
                                    generousLimits()),
               ErrorCode::NonFiniteInput);

  checkFailure(linear::factorFullQR(
                   valid.view(), linear::QRColumnOrder::original, policy(),
                   Limits{.elements = 2U, .bytes = 16U, .evaluations = 0U}),
               ErrorCode::ResourceLimit);

  auto exhausted = policy();
  exhausted.iteration.max_iterations = 0U;
  const Matrix nondiagonal = matrix({{2.0, 1.0}, {1.0, 3.0}});
  checkFailure(linear::factorFullQR(nondiagonal.view(),
                                    linear::QRColumnOrder::original, exhausted,
                                    generousLimits()),
               ErrorCode::NoConvergence);
  // Invalid order, tolerance, input, storage and certification are explicit.
}

void unrepresentableFactorReturnsNonFiniteResult() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Matrix input = matrix({{maximum, 0.0}, {maximum, 1.0}});
  checkFailure(linear::factorFullQR(input.view(),
                                    linear::QRColumnOrder::original, policy(),
                                    generousLimits()),
               ErrorCode::NonFiniteResult);
  // Finite input whose Euclidean first-column norm cannot be represented.
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  tallPivotedFactorHasFullOrthogonalCompletion();
  wideOriginalOrderPreservesUpperTrapezoidalShape();
  rankUsesStrictSingularCutoffAndIgnoresSolveRejection();
  zeroAndEmptyAxesProduceCorrectIdentities();
  equalPivotNormsUseTheSmallestOriginalColumnIndex();
  allZeroInputKeepsFullFactorsAndIdentityPermutation();
  invalidInputsAndResourceOrCertificationExhaustionReturnStatuses();
  unrepresentableFactorReturnsNonFiniteResult();
  return 0;
}

// VSS-MATH-TRACEABILITY: QR-10 QR-11
