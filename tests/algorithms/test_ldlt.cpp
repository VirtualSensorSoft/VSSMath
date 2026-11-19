#include "Test.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Index;
using vss::math::Limits;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::SolvePolicy;
using vss::math::linear::LDLT;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Index row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Index column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

SolvePolicy policy(Index maximumIterations = 64U) {
  SolvePolicy result{};
  result.iteration.max_iterations = maximumIterations;
  return result;
}

Limits generousLimits() {
  return {.elements = 1'000'000U,
          .bytes = std::size_t{64U} * 1024U * 1024U,
          .evaluations = 0U};
}

void checkNear(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <= 1.0e-12);
}

void checkReconstruction(const Matrix& input, const LDLT& factor) {
  const auto permutation = factor.permutation.view();
  const auto widths = factor.pivot_sizes.view();
  VSS_CHECK(permutation.size() == input.rows());
  Index widthSum = 0U;
  for (const auto width : widths) {
    VSS_CHECK(width == 1U || width == 2U);
    widthSum += width;
  }
  VSS_CHECK(widthSum == input.rows());

  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      Scalar reconstructed = 0.0;
      for (Index leftColumn = 0U; leftColumn < input.rows(); ++leftColumn) {
        for (Index rightColumn = 0U; rightColumn < input.rows();
             ++rightColumn) {
          reconstructed +=
              factor.unit_lower.at(row, leftColumn) *
              factor.block_d.at(leftColumn, rightColumn) *
              factor.unit_lower
                  .at( // NOLINT(readability-suspicious-call-argument):
                       // reconstructed output column uses the
                       // first matrix coordinate here
                      column, rightColumn);
        }
      }
      checkNear(reconstructed, input.at(permutation[row], permutation[column]));
    }
  }
}

void factorsIndefiniteExchangeWithOneTwoByTwoPivot() {
  const Matrix input = matrix({{0.0, 1.0}, {1.0, 0.0}});
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  const auto& factor = result.value();
  VSS_CHECK(factor.rank == 2U);
  VSS_CHECK(factor.pivot_sizes.view().size() == 1U);
  VSS_CHECK(factor.pivot_sizes.view()[0] == 2U);
  checkReconstruction(input, factor);
}

void supportsMixedBlocksAndNontrivialPermutation() {
  const Matrix input =
      matrix({{0.0, 0.0, 1.0}, {0.0, 3.0, 0.0}, {1.0, 0.0, 0.0}});
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  const auto& factor = result.value();
  VSS_CHECK(factor.rank == 3U);
  VSS_CHECK(factor.pivot_sizes.view().size() == 2U);
  VSS_CHECK(factor.pivot_sizes.view()[0] == 2U);
  VSS_CHECK(factor.pivot_sizes.view()[1] == 1U);
  VSS_CHECK(factor.permutation.view()[0] == 0U);
  VSS_CHECK(factor.permutation.view()[1] == 2U);
  VSS_CHECK(factor.permutation.view()[2] == 1U);
  checkReconstruction(input, factor);
}

void appliesTwoByTwoSchurUpdateBeforeFactoringTheRemainder() {
  const Matrix input =
      matrix({{0.0, 1.0, 1.0}, {1.0, 0.0, 2.0}, {1.0, 2.0, 5.0}});
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  const auto& factor = result.value();
  VSS_CHECK(factor.pivot_sizes.view().size() == 2U);
  VSS_CHECK(factor.pivot_sizes.view()[0] == 2U);
  VSS_CHECK(factor.pivot_sizes.view()[1] == 1U);
  VSS_CHECK(factor.permutation.view()[1] == 1U);
  checkNear(factor.unit_lower.at(2U, 0U), 2.0);
  checkNear(factor.unit_lower.at(2U, 1U), 1.0);
  checkNear(factor.block_d.at(2U, 2U), 1.0);
  checkReconstruction(input, factor);
}

void swapsTheSelectedSingletonAfterEarlierElimination() {
  const Matrix input = matrix({{10.0, 1.0, 2.0, 0.0},
                               {1.0, 0.0, 0.0, 1.0},
                               {2.0, 0.0, 0.0, 1.0},
                               {0.0, 1.0, 1.0, 3.0}});
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(), generousLimits());
  VSS_CHECK(result.hasValue());
  const auto& factor = result.value();
  VSS_CHECK(factor.permutation.view()[1] == 3U);
  checkReconstruction(input, factor);
}

void powerOfTwoScalingHandlesExtremeFiniteUnits() {
  const Scalar largest = std::numeric_limits<Scalar>::max();
  const Scalar smallest = std::numeric_limits<Scalar>::denorm_min();
  const Matrix large = matrix({{largest / 4.0, 0.0}, {0.0, -largest / 8.0}});
  const Matrix tiny = matrix({{smallest, 0.0}, {0.0, -smallest}});
  const auto largeResult =
      vss::math::linear::factorLDLT(large.view(), policy(0U), generousLimits());
  const auto tinyResult =
      vss::math::linear::factorLDLT(tiny.view(), policy(0U), generousLimits());
  VSS_CHECK(largeResult.hasValue());
  VSS_CHECK(tinyResult.hasValue());
  VSS_CHECK(largeResult.value().rank == 2U);
  VSS_CHECK(tinyResult.value().rank == 2U);
  checkReconstruction(large, largeResult.value());
  checkReconstruction(tiny, tinyResult.value());
}

void preservesSingularDirectionsWhileReportingNumericalRank() {
  const Matrix input =
      matrix({{4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.5}});
  auto numericalPolicy = policy(0U);
  numericalPolicy.rank_abs = 1.0;
  const auto result = vss::math::linear::factorLDLT(
      input.view(), numericalPolicy, generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 1U);
  checkNear(result.value().block_d.at(1U, 1U), 1.0);
  checkNear(result.value().block_d.at(2U, 2U), 0.5);
  checkReconstruction(input, result.value());
}

void preservesExactZeroPivotsInSingularFactors() {
  const Matrix input =
      matrix({{3.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}});
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(0U), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2U);
  VSS_CHECK(result.value().block_d.at(1U, 1U) == 0.0);
  VSS_CHECK(result.value().block_d.at(2U, 2U) == -1.0);
  checkReconstruction(input, result.value());
}

void acceptsAnEmptySquareMatrix() {
  const Matrix input(0U, 0U);
  const auto result =
      vss::math::linear::factorLDLT(input.view(), policy(0U), generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 0U);
  VSS_CHECK(result.value().unit_lower.rows() == 0U);
  VSS_CHECK(result.value().permutation.view().empty());
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void validatesShapeSymmetryFinitenessAndResources() {
  const Matrix nonsquare(2U, 1U);
  checkFailure(vss::math::linear::factorLDLT(nonsquare.view(), policy(),
                                             generousLimits()),
               ErrorCode::DimensionMismatch);

  const Matrix nonsymmetric = matrix({{1.0, 0.25}, {0.0, 1.0}});
  checkFailure(vss::math::linear::factorLDLT(nonsymmetric.view(), policy(),
                                             generousLimits()),
               ErrorCode::NotSymmetric);

  Matrix nonfinite = matrix({{1.0}});
  nonfinite.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkFailure(vss::math::linear::factorLDLT(nonfinite.view(), policy(),
                                             generousLimits()),
               ErrorCode::NonFiniteInput);

  const Matrix ordinary = matrix({{2.0, 0.0}, {0.0, 3.0}});
  checkFailure(vss::math::linear::factorLDLT(
                   ordinary.view(), policy(),
                   Limits{.elements = 1U, .bytes = 1U, .evaluations = 0U}),
               ErrorCode::ResourceLimit);

  const Matrix roundedSymmetry = matrix({{2.0, 1.0}, {1.0 + 1.0e-15, 3.0}});
  const auto accepted = vss::math::linear::factorLDLT(
      roundedSymmetry.view(), policy(), generousLimits());
  VSS_CHECK(accepted.hasValue());
  checkReconstruction(roundedSymmetry, accepted.value());
}

void zeroIterationCapOnlyCertifiesDiagonalInputs() {
  const Matrix input = matrix({{2.0, 1.0}, {1.0, 2.0}});
  checkFailure(
      vss::math::linear::factorLDLT(input.view(), policy(0U), generousLimits()),
      ErrorCode::NoConvergence);
}

void rejectsInvalidNumericalPolicy() {
  auto invalid = policy();
  invalid.rank_rel = 1.0;
  const Matrix input = matrix({{1.0}});
  checkFailure(
      vss::math::linear::factorLDLT(input.view(), invalid, generousLimits()),
      ErrorCode::InvalidTolerance);
}

void factorConstructionDoesNotApplySolveConditionFloor() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 4.0}});
  auto numericalPolicy = policy(0U);
  numericalPolicy.reciprocal_condition_floor = 1.0;
  const auto result = vss::math::linear::factorLDLT(
      input.view(), numericalPolicy, generousLimits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2U);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): test assertions deliberately
             // throw on failure
  factorsIndefiniteExchangeWithOneTwoByTwoPivot();
  supportsMixedBlocksAndNontrivialPermutation();
  appliesTwoByTwoSchurUpdateBeforeFactoringTheRemainder();
  swapsTheSelectedSingletonAfterEarlierElimination();
  powerOfTwoScalingHandlesExtremeFiniteUnits();
  preservesSingularDirectionsWhileReportingNumericalRank();
  preservesExactZeroPivotsInSingularFactors();
  acceptsAnEmptySquareMatrix();
  validatesShapeSymmetryFinitenessAndResources();
  zeroIterationCapOnlyCertifiesDiagonalInputs();
  rejectsInvalidNumericalPolicy();
  factorConstructionDoesNotApplySolveConditionFloor();
}
