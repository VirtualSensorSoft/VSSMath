#include "Test.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
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

void checkNear(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-12) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
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

auto nullSpace(const Matrix& input, SolvePolicy settings = policy(),
               Limits limits = generousLimits()) {
  return linear::nullSpace(input.view(), settings, limits);
}

void normalizedConstraintDirectionIsReturned() {
  const Matrix constraint = matrix({{1.0, 1.0}});
  const auto result = nullSpace(constraint);
  VSS_CHECK(result.hasValue());
  const Matrix& basis = result.value();
  VSS_CHECK(basis.rows() == 2U && basis.columns() == 1U);
  checkNear(basis.at(0U, 0U) + basis.at(1U, 0U), 0.0);
  checkNear(std::hypot(basis.at(0U, 0U), basis.at(1U, 0U)), 1.0);
  // NUL-01: x+y=0 returns a unit right-null direction.
}

void cutoffProjectorIncludesDiscardedAndStructuralDirections() {
  Matrix input(3U, 4U, 0.0);
  input.at(0U, 0U) = 4.0;
  input.at(1U, 1U) = 1.0;
  input.at(2U, 2U) = 0.5;
  auto settings = policy();
  settings.rank_abs = 0.0;
  settings.rank_rel = 0.25;
  settings.rank = RankPolicy::reject;
  settings.reciprocal_condition_floor = 1.0;

  const auto result = nullSpace(input, settings);
  VSS_CHECK(result.hasValue());
  const Matrix& basis = result.value();
  VSS_CHECK(basis.rows() == 4U && basis.columns() == 3U);
  for (Index first = 0U; first < 4U; ++first) {
    for (Index second = 0U; second < 4U; ++second) {
      Scalar projector = 0.0;
      for (Index column = 0U; column < basis.columns(); ++column) {
        projector += basis.at(first, column) * basis.at(second, column);
      }
      checkNear(projector, first == second && first != 0U ? 1.0 : 0.0);
    }
  }
  Matrix conditioned(3U, 3U, 0.0);
  conditioned.at(0U, 0U) = 4.0;
  conditioned.at(1U, 1U) = 2.0;
  conditioned.at(2U, 2U) = 0.5;
  settings.rank = RankPolicy::minimum_norm;
  settings.rank_abs = 1.5;
  settings.rank_rel = 0.0;
  settings.reciprocal_condition_floor = 0.75;
  const auto belowFloor = nullSpace(conditioned, settings);
  VSS_CHECK(belowFloor.hasValue() && belowFloor.value().columns() == 1U);
  // NUL-02: equality at cutoff is null; RankPolicy and condition floor do not
  // reject a null-space query; the wide structural direction is included.
}

void exactZeroAndEmptyAxesReturnCompleteNullSpaces() {
  const Matrix zero(2U, 3U);
  const auto allDirections = nullSpace(zero);
  VSS_CHECK(allDirections.hasValue());
  VSS_CHECK(allDirections.value().rows() == 3U &&
            allDirections.value().columns() == 3U);
  for (Index row = 0U; row < 3U; ++row) {
    for (Index otherCoordinate = 0U; otherCoordinate < 3U; ++otherCoordinate) {
      Scalar projector = 0.0;
      for (Index basis = 0U; basis < 3U; ++basis) {
        projector += allDirections.value().at(row, basis) *
                     allDirections.value().at(otherCoordinate, basis);
      }
      checkNear(projector, row == otherCoordinate ? 1.0 : 0.0);
    }
  }

  const Matrix noEquations(0U, 3U);
  const auto emptyRowSpace = nullSpace(noEquations);
  VSS_CHECK(emptyRowSpace.hasValue());
  VSS_CHECK(emptyRowSpace.value().rows() == 3U &&
            emptyRowSpace.value().columns() == 3U);

  const Matrix noUnknowns(3U, 0U);
  const auto emptyColumnSpace = nullSpace(noUnknowns);
  VSS_CHECK(emptyColumnSpace.hasValue());
  VSS_CHECK(emptyColumnSpace.value().rows() == 0U &&
            emptyColumnSpace.value().columns() == 0U);
  // NUL-03: rank zero and an empty row axis expose every right-null direction.
}

void nontrivialWideNullBasisSatisfiesResidualAndOrthogonality() {
  const Matrix input = matrix({{1.0, 2.0, 0.0, 1.0}, {0.0, 1.0, 1.0, -1.0}});
  const auto result = nullSpace(input);
  VSS_CHECK(result.hasValue());
  const Matrix& basis = result.value();
  VSS_CHECK(basis.rows() == 4U && basis.columns() == 2U);
  for (Index direction = 0U; direction < basis.columns(); ++direction) {
    for (Index row = 0U; row < input.rows(); ++row) {
      Scalar residual = 0.0;
      for (Index inputCoordinate = 0U; inputCoordinate < input.columns();
           ++inputCoordinate) {
        residual += input.at(row, inputCoordinate) *
                    basis.at(inputCoordinate, direction);
      }
      checkNear(residual, 0.0);
    }
    for (Index other = 0U; other < basis.columns(); ++other) {
      Scalar dot = 0.0;
      for (Index coordinate = 0U; coordinate < basis.rows(); ++coordinate) {
        dot += basis.at(coordinate, direction) * basis.at(coordinate, other);
      }
      checkNear(dot, direction == other ? 1.0 : 0.0);
    }
  }
  // NUL-06: a nontrivial wide input has a residual-free orthonormal basis.
}

void fullColumnRankHasEmptyNullSpaceAndDiagonalCertificationWorks() {
  const Matrix fullColumnRank = matrix({{1.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}});
  const auto empty = nullSpace(fullColumnRank);
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().rows() == 2U && empty.value().columns() == 0U);

  Matrix diagonal(2U, 2U, 0.0);
  diagonal.at(0U, 0U) = 2.0;
  auto noIterations = policy();
  noIterations.iteration.max_iterations = 0U;
  const auto certified = nullSpace(diagonal, noIterations);
  VSS_CHECK(certified.hasValue());
  VSS_CHECK(certified.value().rows() == 2U &&
            certified.value().columns() == 1U);
  checkNear(certified.value().at(0U, 0U), 0.0);
  checkNear(certified.value().at(1U, 0U), 1.0);
  // NUL-04: full column rank has no null directions; diagonal rank is exact.
}

void invalidInputsAndInsufficientResourcesReturnStatuses() {
  auto invalid = policy();
  invalid.rank_rel = 1.0;
  checkFailure(nullSpace(matrix({{1.0}}), invalid),
               ErrorCode::InvalidTolerance);

  Matrix nonfinite = matrix({{1.0}});
  nonfinite.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkFailure(nullSpace(nonfinite), ErrorCode::NonFiniteInput);

  checkFailure(
      nullSpace(matrix({{1.0, 1.0}}), policy(),
                Limits{.elements = 1U, .bytes = 1U, .evaluations = 0U}),
      ErrorCode::ResourceLimit);

  auto exhausted = policy();
  exhausted.iteration.max_iterations = 0U;
  checkFailure(nullSpace(matrix({{2.0, 1.0}, {1.0, 3.0}}), exhausted),
               ErrorCode::NoConvergence);
  // NUL-05: tolerance, finiteness, bounded storage, and iteration failures.
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  normalizedConstraintDirectionIsReturned();
  cutoffProjectorIncludesDiscardedAndStructuralDirections();
  exactZeroAndEmptyAxesReturnCompleteNullSpaces();
  nontrivialWideNullBasisSatisfiesResidualAndOrthogonality();
  fullColumnRankHasEmptyNullSpaceAndDiagonalCertificationWorks();
  invalidInputsAndInsufficientResourcesReturnStatuses();
  return 0;
}

// VSS-MATH-TRACEABILITY: NUL-01 NUL-02 NUL-03 NUL-04 NUL-05 NUL-06
