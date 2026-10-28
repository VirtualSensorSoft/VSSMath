#include "Test.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace {

template <typename Enum> Enum invalidEnumValue() {
  volatile const std::uint8_t rawValue = 255U;
  // Deliberately construct an out-of-range value for invalid-enumerator tests.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(rawValue);
}

using vss::math::ErrorCode;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::linear::ConditionClassification;
using vss::math::linear::ConditionReport;
using vss::math::linear::FactorizationOptions;
using vss::math::linear::IllConditionedSolvePolicy;
using vss::math::linear::LUFactorization;
using vss::math::linear::SolveOptions;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const auto rowCount = rows.size();
  const auto columnCount =
      rows.begin() == rows.end() ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  std::size_t row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    std::size_t column = 0U;
    for (const Scalar value : values) {
      result.at(row, column) = value;
      ++column;
    }
    ++row;
  }
  return result;
}

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void checkNear(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-12) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
}

void checkMatrixNear(const Matrix& actual, const Matrix& expected,
                     Scalar tolerance = 1.0e-12) {
  VSS_CHECK(actual.rows() == expected.rows());
  VSS_CHECK(actual.columns() == expected.columns());
  for (std::size_t row = 0U; row < actual.rows(); ++row) {
    for (std::size_t column = 0U; column < actual.columns(); ++column) {
      checkNear(actual.at(row, column), expected.at(row, column), tolerance);
    }
  }
}

void checkVectorNear(const Vector& actual, const Vector& expected,
                     Scalar tolerance = 1.0e-12) {
  VSS_CHECK(actual.size() == expected.size());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    checkNear(actual.at(index), expected.at(index), tolerance);
  }
}

template <class Result>
void checkFailure(const Result& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkConditionReport(const std::optional<ConditionReport>& conditionReport,
                          const ConditionReport& expected) {
  VSS_CHECK(conditionReport.has_value());
  const auto observed = conditionReport.value_or(ConditionReport{
      .matrixOneNorm = 0.0,
      .estimatedConditionNumber1 = 0.0,
      .conditionLimit = 0.0,
      .classification = ConditionClassification::IllConditioned});
  VSS_CHECK(observed.matrixOneNorm == expected.matrixOneNorm);
  VSS_CHECK(observed.estimatedConditionNumber1 ==
            expected.estimatedConditionNumber1);
  VSS_CHECK(observed.conditionLimit == expected.conditionLimit);
  VSS_CHECK(observed.classification == expected.classification);
}

Matrix multiply(const Matrix& left, const Matrix& right) {
  VSS_CHECK(left.columns() == right.rows());
  Matrix result(left.rows(), right.columns());
  for (std::size_t row = 0U; row < left.rows(); ++row) {
    for (std::size_t column = 0U; column < right.columns(); ++column) {
      Scalar sum = 0.0;
      for (std::size_t index = 0U; index < left.columns(); ++index) {
        sum += left.at(row, index) * right.at(index, column);
      }
      result.at(row, column) = sum;
    }
  }
  return result;
}

Matrix permutedRows(const Matrix& input,
                    std::span<const std::size_t> permutation) {
  VSS_CHECK(input.rows() == permutation.size());
  Matrix result(input.rows(), input.columns());
  for (std::size_t row = 0U; row < input.rows(); ++row) {
    for (std::size_t column = 0U; column < input.columns(); ++column) {
      result.at(row, column) = input.at(permutation[row], column);
    }
  }
  return result;
}

void pivotPermutationReconstructionSolve() {
  const Matrix input =
      matrix({{0.0, 2.0, 1.0}, {1.0, -2.0, -3.0}, {2.0, 3.0, 1.0}});
  const Vector rightHandSide = vector({7.0, -12.0, 11.0});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.size() == 3U);
  VSS_CHECK(factorization.permutation().size() == 3U);
  VSS_CHECK(factorization.permutation()[0U] == 2U);
  VSS_CHECK(factorization.permutation()[1U] == 1U);
  VSS_CHECK(factorization.permutation()[2U] == 0U);
  VSS_CHECK(!factorization.isSingular());
  VSS_CHECK(!factorization.firstUnusablePivot().has_value());

  const Matrix expectedLower =
      matrix({{1.0, 0.0, 0.0}, {0.5, 1.0, 0.0}, {0.0, -4.0 / 7.0, 1.0}});
  const Matrix expectedUpper =
      matrix({{2.0, 3.0, 1.0}, {0.0, -3.5, -3.5}, {0.0, 0.0, -1.0}});
  checkMatrixNear(factorization.lowerFactor(), expectedLower);
  checkMatrixNear(factorization.upperFactor(), expectedUpper);

  const Matrix reconstructed =
      multiply(factorization.lowerFactor(), factorization.upperFactor());
  const Matrix permuted = permutedRows(input, factorization.permutation());
  checkMatrixNear(reconstructed, permuted);

  const auto solveResult = factorization.solve(rightHandSide.view());
  VSS_CHECK(solveResult.hasValue());
  checkVectorNear(solveResult.value().solution(), vector({1.0, 2.0, 3.0}));
  checkConditionReport(solveResult.value().conditionReport(),
                       factorization.conditionReport());

  auto copied = factorization;
  checkMatrixNear(copied.lowerFactor(), factorization.lowerFactor());
  checkMatrixNear(copied.upperFactor(), factorization.upperFactor());
  static_assert(std::is_nothrow_move_constructible_v<LUFactorization>);
  auto moved = LUFactorization(std::move(copied));
  VSS_CHECK(copied.size() == 0U); // NOLINT(clang-analyzer-cplusplus.Move):
                                  // the moved-from extent is a public contract.
  const auto movedFromSolve =
      copied // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move):
             // moved-from solve behavior is tested.
          .solve(rightHandSide.view());
  checkFailure(movedFromSolve, ErrorCode::InvalidExtent);
  VSS_CHECK(moved.size() == 3U);
}

void repeatedMatrixRightHandSides() {
  const Matrix input =
      matrix({{0.0, 2.0, 1.0}, {1.0, -2.0, -3.0}, {2.0, 3.0, 1.0}});
  const Matrix rightHandSides =
      matrix({{7.0, 2.0}, {-12.0, -7.0}, {11.0, 0.0}});
  const Matrix expected = matrix({{1.0, -1.0}, {2.0, 0.0}, {3.0, 2.0}});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  const auto solved = factorization.solve(rightHandSides.view());
  VSS_CHECK(solved.hasValue());
  checkMatrixNear(solved.value().solution(), expected);
  checkConditionReport(solved.value().conditionReport(),
                       factorization.conditionReport());
  const auto repeated = factorization.solve(rightHandSides.view());
  VSS_CHECK(repeated.hasValue());
  checkMatrixNear(repeated.value().solution(), expected);
  checkConditionReport(repeated.value().conditionReport(),
                       factorization.conditionReport());
  checkMatrixNear(rightHandSides,
                  matrix({{7.0, 2.0}, {-12.0, -7.0}, {11.0, 0.0}}));
}

void determinantDiagnostics() {
  const Matrix input =
      matrix({{0.0, 2.0, 1.0}, {1.0, -2.0, -3.0}, {2.0, 3.0, 1.0}});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().determinantSign() == -1);
  checkNear(result.value().logAbsDeterminant(), std::log(7.0));
}

void singularMatrix() {
  const Matrix input = matrix({{1.0, 2.0}, {2.0, 4.0}});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.isSingular());
  VSS_CHECK(factorization.firstUnusablePivot().has_value());
  const auto firstUnusablePivot = factorization.firstUnusablePivot();
  VSS_CHECK(firstUnusablePivot.has_value());
  VSS_CHECK(firstUnusablePivot.value_or(std::numeric_limits<Index>::max()) ==
            1U);
  checkNear(factorization.rankThreshold(), 4.0e-12);
  VSS_CHECK(factorization.determinantSign() == 0);
  VSS_CHECK(std::isinf(factorization.logAbsDeterminant()));
  VSS_CHECK(std::signbit(factorization.logAbsDeterminant()));
  VSS_CHECK(
      std::isinf(factorization.conditionReport().estimatedConditionNumber1));
  VSS_CHECK(factorization.conditionReport().estimatedConditionNumber1 > 0.0);
  checkFailure(factorization.solve(vector({1.0, 1.0}).view()),
               ErrorCode::Singular);

  const Matrix zeroPivot = matrix({{0.0, 1.0}, {0.0, 0.0}});
  const auto zeroPivotResult = vss::math::linear::factorizeLU(zeroPivot.view());
  VSS_CHECK(zeroPivotResult.hasValue());
  const auto zeroPivotFirst = zeroPivotResult.value().firstUnusablePivot();
  VSS_CHECK(zeroPivotFirst.has_value());
  VSS_CHECK(zeroPivotFirst.value_or(std::numeric_limits<Index>::max()) == 0U);
  checkNear(zeroPivotResult.value().rankThreshold(), 1.0e-12);
}

void illConditionedPolicyBoundary() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 1.0e-11}});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.conditionReport().estimatedConditionNumber1 > 1.0e10);
  VSS_CHECK(factorization.conditionReport().classification ==
            ConditionClassification::IllConditioned);
  checkFailure(factorization.solve(vector({2.0, 3.0e-11}).view()),
               ErrorCode::IllConditioned);

  SolveOptions allow;
  allow.illConditionedPolicy = IllConditionedSolvePolicy::AllowWithDiagnostic;
  const auto allowed =
      factorization.solve(vector({2.0, 3.0e-11}).view(), allow);
  VSS_CHECK(allowed.hasValue());
  checkVectorNear(allowed.value().solution(), vector({2.0, 3.0}));
  checkConditionReport(allowed.value().conditionReport(),
                       factorization.conditionReport());

  const Matrix exactBoundary = matrix({{1.0, 0.0}, {0.0, 1.0e-10}});
  FactorizationOptions permissive;
  permissive.conditionLimit = 1.0e300;
  const auto permissiveResult =
      vss::math::linear::factorizeLU(exactBoundary.view(), permissive);
  VSS_CHECK(permissiveResult.hasValue());
  FactorizationOptions boundary;
  boundary.conditionLimit =
      permissiveResult.value().conditionReport().estimatedConditionNumber1;
  const auto boundaryResult =
      vss::math::linear::factorizeLU(exactBoundary.view(), boundary);
  VSS_CHECK(boundaryResult.hasValue());
  VSS_CHECK(boundaryResult.value().conditionReport().classification ==
            ConditionClassification::IllConditioned);
  checkFailure(boundaryResult.value().solve(vector({1.0, 1.0e-10}).view()),
               ErrorCode::IllConditioned);

  Matrix empty;
  for (const Scalar conditionLimit :
       {std::numeric_limits<Scalar>::quiet_NaN(),
        std::numeric_limits<Scalar>::infinity(), 1.0, 0.0, -1.0}) {
    FactorizationOptions options;
    options.conditionLimit = conditionLimit;
    checkFailure(vss::math::linear::factorizeLU(empty.view(), options),
                 ErrorCode::InvalidTolerance);
  }
}

void validationFailuresAndEmptyRightHandSide() {
  Matrix empty;
  checkFailure(vss::math::linear::factorizeLU(empty.view()),
               ErrorCode::InvalidExtent);

  const Matrix nonSquare(2U, 3U);
  checkFailure(vss::math::linear::factorizeLU(nonSquare.view()),
               ErrorCode::DimensionMismatch);

  const Matrix nonFinite =
      matrix({{1.0, std::numeric_limits<Scalar>::infinity()}, {0.0, 1.0}});
  checkFailure(vss::math::linear::factorizeLU(nonFinite.view()),
               ErrorCode::NonFiniteInput);

  const Matrix input = matrix({{2.0, 0.0}, {0.0, 3.0}});
  const auto factorized = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(factorized.hasValue());
  const auto& factorization = factorized.value();
  checkFailure(factorization.solve(vector({1.0}).view()),
               ErrorCode::DimensionMismatch);
  checkFailure(
      factorization.solve(
          vector({1.0, std::numeric_limits<Scalar>::infinity()}).view()),
      ErrorCode::NonFiniteInput);

  const Matrix wrongRows(1U, 2U);
  checkFailure(factorization.solve(wrongRows.view()),
               ErrorCode::DimensionMismatch);

  const Matrix nonFiniteRightHandSides =
      matrix({{1.0, std::numeric_limits<Scalar>::infinity()}, {2.0, 3.0}});
  checkFailure(factorization.solve(nonFiniteRightHandSides.view()),
               ErrorCode::NonFiniteInput);

  const Matrix emptyRightHandSides(2U, 0U);
  checkFailure(factorization.solve(emptyRightHandSides.view()),
               ErrorCode::InvalidExtent);
}

void invalidRightHandSidePrecedesFactorClassification() {
  const Matrix singularInput = matrix({{1.0, 2.0}, {2.0, 4.0}});
  const auto singularResult =
      vss::math::linear::factorizeLU(singularInput.view());
  VSS_CHECK(singularResult.hasValue());
  const auto& singular = singularResult.value();

  checkFailure(
      singular.solve(
          vector({std::numeric_limits<Scalar>::quiet_NaN(), 1.0}).view()),
      ErrorCode::NonFiniteInput);
  checkFailure(
      singular.solve(
          vector({1.0, std::numeric_limits<Scalar>::infinity()}).view()),
      ErrorCode::NonFiniteInput);
  const Matrix nanRightHandSides =
      matrix({{1.0, std::numeric_limits<Scalar>::quiet_NaN()}, {2.0, 3.0}});
  checkFailure(singular.solve(nanRightHandSides.view()),
               ErrorCode::NonFiniteInput);
  const Matrix infiniteRightHandSides =
      matrix({{1.0, std::numeric_limits<Scalar>::infinity()}, {2.0, 3.0}});
  checkFailure(singular.solve(infiniteRightHandSides.view()),
               ErrorCode::NonFiniteInput);

  const Matrix emptyColumns(2U, 0U);
  checkFailure(singular.solve(emptyColumns.view()), ErrorCode::InvalidExtent);
  const Matrix wrongRows(1U, 2U);
  checkFailure(singular.solve(wrongRows.view()), ErrorCode::DimensionMismatch);
  const Matrix wrongRowsWithNan =
      matrix({{1.0, std::numeric_limits<Scalar>::quiet_NaN()}});
  checkFailure(singular.solve(wrongRowsWithNan.view()),
               ErrorCode::DimensionMismatch);

  SolveOptions invalidOptions;
  invalidOptions.illConditionedPolicy =
      invalidEnumValue<IllConditionedSolvePolicy>();
  checkFailure(
      singular.solve(
          vector({std::numeric_limits<Scalar>::quiet_NaN(), 1.0}).view(),
          invalidOptions),
      ErrorCode::InvalidDomain);

  const Matrix illConditionedInput = matrix({{1.0, 0.0}, {0.0, 1.0e-11}});
  const auto illConditionedResult =
      vss::math::linear::factorizeLU(illConditionedInput.view());
  VSS_CHECK(illConditionedResult.hasValue());
  const auto& illConditioned = illConditionedResult.value();
  VSS_CHECK(illConditioned.conditionReport().classification ==
            ConditionClassification::IllConditioned);
  checkFailure(
      illConditioned.solve(
          vector({std::numeric_limits<Scalar>::quiet_NaN(), 1.0}).view()),
      ErrorCode::NonFiniteInput);
  checkFailure(illConditioned.solve(infiniteRightHandSides.view()),
               ErrorCode::NonFiniteInput);
  checkFailure(illConditioned.solve(emptyColumns.view()),
               ErrorCode::InvalidExtent);
}

void pivotTieUsesLowestOriginalRow() {
  const Matrix input = matrix({{1.0, 1.0}, {1.0, 2.0}});
  const auto result = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().permutation()[0U] == 0U);
}

template <class Type>
concept HasRankQuery = requires(const Type& value) { value.rank(); };

static_assert(!HasRankQuery<LUFactorization>);

void invalidSolvePolicy() {
  const auto result =
      vss::math::linear::factorizeLU(matrix({{1.0, 0.0}, {0.0, 1.0}}).view());
  VSS_CHECK(result.hasValue());
  SolveOptions invalid;
  invalid.illConditionedPolicy = invalidEnumValue<IllConditionedSolvePolicy>();
  checkFailure(result.value().solve(vector({1.0, 2.0}).view(), invalid),
               ErrorCode::InvalidDomain);
  checkFailure(result.value().solve(matrix({{1.0}, {2.0}}).view(), invalid),
               ErrorCode::InvalidDomain);
  const Vector invalidInput{
      std::vector<Scalar>{std::numeric_limits<Scalar>::quiet_NaN()}};
  checkFailure(result.value().solve(invalidInput.view(), invalid),
               ErrorCode::InvalidDomain);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): test failures use exit
             // diagnostics.
  pivotPermutationReconstructionSolve();
  repeatedMatrixRightHandSides();
  determinantDiagnostics();
  singularMatrix();
  illConditionedPolicyBoundary();
  invalidSolvePolicy();
  validationFailuresAndEmptyRightHandSide();
  invalidRightHandSidePrecedesFactorClassification();
  pivotTieUsesLowestOriginalRow();
}

// VSS-MATH-TRACEABILITY: LU-01 LU-02 LU-03 LU-04 LU-05
// VSS-MATH-TRACEABILITY: SUP-LU-01
