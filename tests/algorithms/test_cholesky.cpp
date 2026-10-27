#include "Test.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace {

template <typename Enum> Enum invalidEnumValue() {
  volatile const std::uint8_t rawValue = 255U;
  // Deliberately construct an out-of-range value for invalid-enumerator tests.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(rawValue);
}

using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::linear::CholeskyFactorization;
using vss::math::linear::ConditionClassification;
using vss::math::linear::ConditionReport;
using vss::math::linear::IllConditionedSolvePolicy;
using vss::math::linear::SolveOptions;
using vss::math::linear::SymmetricFactorizationOptions;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount =
      rows.begin() == rows.end() ? 0U : rows.begin()->size();
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

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void checkNear(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-12) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
}

void checkVectorNear(const Vector& actual, const Vector& expected,
                     Scalar tolerance = 1.0e-12) {
  VSS_CHECK(actual.size() == expected.size());
  for (Index index = 0U; index < actual.size(); ++index) {
    checkNear(actual.at(index), expected.at(index), tolerance);
  }
}

void checkMatrixNear(const Matrix& actual, const Matrix& expected,
                     Scalar tolerance = 1.0e-12) {
  VSS_CHECK(actual.rows() == expected.rows());
  VSS_CHECK(actual.columns() == expected.columns());
  for (Index row = 0U; row < actual.rows(); ++row) {
    for (Index column = 0U; column < actual.columns(); ++column) {
      checkNear(actual.at(row, column), expected.at(row, column), tolerance);
    }
  }
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkConditionReport(const ConditionReport& observed, Scalar expectedNorm,
                          Scalar expectedCondition, Scalar expectedLimit,
                          ConditionClassification expectedClassification) {
  checkNear(observed.matrixOneNorm, expectedNorm);
  checkNear(observed.estimatedConditionNumber1, expectedCondition, 1.0e-2);
  checkNear(observed.conditionLimit, expectedLimit);
  VSS_CHECK(observed.classification == expectedClassification);
}

void spdReconstruction() {
  const Matrix input = matrix({{4.0, 2.0}, {2.0, 3.0}});
  const auto result = vss::math::linear::factorizeCholesky(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.size() == 2U);
  VSS_CHECK(factorization.rank() == 2U);
  checkNear(factorization.rankThreshold(), 4.0e-12);
  VSS_CHECK(factorization.symmetryTolerance().absolute == 0.0);
  VSS_CHECK(factorization.symmetryTolerance().relative == 1.0e-12);
  checkMatrixNear(factorization.lowerFactor(),
                  matrix({{2.0, 0.0}, {1.0, std::numbers::sqrt2}}));
  checkNear(factorization.logDeterminant(), std::log(8.0));
  checkConditionReport(factorization.conditionReport(), 6.0, 4.5, 1.0e10,
                       ConditionClassification::WithinLimit);
  checkMatrixNear(input, matrix({{4.0, 2.0}, {2.0, 3.0}}));
}

void solveAndLogDeterminant() {
  const auto result = vss::math::linear::factorizeCholesky(
      matrix({{4.0, 2.0}, {2.0, 3.0}}).view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();

  const auto solved = factorization.solve(vector({8.0, 8.0}).view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({1.0, 2.0}));
  const auto solveCondition = solved.value().conditionReport();
  VSS_CHECK(solveCondition.has_value());
  const auto solveReport = solveCondition.value_or(ConditionReport{
      .matrixOneNorm = 0.0,
      .estimatedConditionNumber1 = 0.0,
      .conditionLimit = 0.0,
      .classification = ConditionClassification::IllConditioned});
  checkConditionReport(solveReport, 6.0, 4.5, 1.0e10,
                       ConditionClassification::WithinLimit);

  const auto matrixSolved =
      factorization.solve(matrix({{8.0, 6.0}, {8.0, 1.0}}).view());
  VSS_CHECK(matrixSolved.hasValue());
  checkMatrixNear(matrixSolved.value().solution(),
                  matrix({{1.0, 2.0}, {2.0, -1.0}}));

  const Scalar rootTwo = std::numbers::sqrt2;
  const auto whitened =
      factorization.whiten(vector({6.0, 3.0 - (2.0 * rootTwo)}).view());
  VSS_CHECK(whitened.hasValue());
  checkVectorNear(whitened.value().solution(), vector({3.0, -2.0}));
  VSS_CHECK(whitened.value().conditionReport().has_value());
}

void semidefiniteRejection() {
  const auto result = vss::math::linear::factorizeCholesky(
      matrix({{1.0, 1.0}, {1.0, 1.0}}).view());
  checkFailure(result, ErrorCode::NotPositiveDefinite);
}

void indefiniteRejection() {
  const auto result = vss::math::linear::factorizeCholesky(
      matrix({{1.0, 2.0}, {2.0, 1.0}}).view());
  checkFailure(result, ErrorCode::NotPositiveDefinite);
}

void symmetryBoundary() {
  const auto accepted = vss::math::linear::factorizeCholesky(
      matrix({{4.0, 2.0000000000001}, {2.0, 3.0}}).view());
  VSS_CHECK(accepted.hasValue());

  const auto rejected = vss::math::linear::factorizeCholesky(
      matrix({{4.0, 2.00000000001}, {2.0, 3.0}}).view());
  checkFailure(rejected, ErrorCode::NotSymmetric);

  const auto precedence = vss::math::linear::factorizeCholesky(
      matrix({{1.0, 2.0}, {0.0, 1.0}}).view());
  checkFailure(precedence, ErrorCode::NotSymmetric);

  SymmetricFactorizationOptions invalidOptions;
  invalidOptions.symmetryTolerance.absolute = -1.0;
  checkFailure(
      vss::math::linear::factorizeCholesky(
          matrix({{std::numeric_limits<Scalar>::quiet_NaN(), 0.0}, {0.0, 1.0}})
              .view(),
          invalidOptions),
      ErrorCode::InvalidTolerance);

  invalidOptions = {};
  invalidOptions.factorization.conditionLimit = 1.0;
  checkFailure(
      vss::math::linear::factorizeCholesky(
          matrix({{std::numeric_limits<Scalar>::quiet_NaN(), 0.0}, {0.0, 1.0}})
              .view(),
          invalidOptions),
      ErrorCode::InvalidTolerance);
}

void illConditionedSolve() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 1.0e-11}});
  const auto result = vss::math::linear::factorizeCholesky(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.rank() == 2U);
  checkConditionReport(factorization.conditionReport(), 1.0, 1.0e11, 1.0e10,
                       ConditionClassification::IllConditioned);

  const Vector rightHandSide = vector({2.0, 3.0e-11});
  checkFailure(factorization.solve(rightHandSide.view()),
               ErrorCode::IllConditioned);

  SolveOptions options;
  options.illConditionedPolicy = IllConditionedSolvePolicy::AllowWithDiagnostic;
  const auto allowed = factorization.solve(rightHandSide.view(), options);
  VSS_CHECK(allowed.hasValue());
  checkVectorNear(allowed.value().solution(), vector({2.0, 3.0}));
  VSS_CHECK(allowed.value().conditionReport().has_value());
  const auto allowedReport = allowed.value().conditionReport();
  VSS_CHECK(allowedReport.has_value());
  if (allowedReport.has_value()) {
    VSS_CHECK(allowedReport->classification ==
              ConditionClassification::IllConditioned);
  }

  const Vector residual = vector({2.0, 3.0 * std::sqrt(1.0e-11)});
  const auto whitened = factorization.whiten(residual.view(), options);
  VSS_CHECK(whitened.hasValue());
  checkVectorNear(whitened.value().solution(), vector({2.0, 3.0}), 1.0e-10);
}

void validationAndPrecedence() {
  SymmetricFactorizationOptions options;
  options.factorization.rankTolerance.absolute = -1.0;
  checkFailure(vss::math::linear::factorizeCholesky(Matrix{}.view(), options),
               ErrorCode::InvalidTolerance);

  options = {};
  const auto empty =
      vss::math::linear::factorizeCholesky(Matrix{}.view(), options);
  checkFailure(empty, ErrorCode::InvalidExtent);

  const Matrix nonFinite =
      matrix({{std::numeric_limits<Scalar>::infinity(), 0.0}, {0.0, 1.0}});
  checkFailure(vss::math::linear::factorizeCholesky(nonFinite.view()),
               ErrorCode::NonFiniteInput);

  const auto result = vss::math::linear::factorizeCholesky(
      matrix({{4.0, 2.0}, {2.0, 3.0}}).view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  checkFailure(factorization.solve(vector({1.0}).view()),
               ErrorCode::DimensionMismatch);
  checkFailure(factorization.solve(Matrix(2U, 0U).view()),
               ErrorCode::InvalidExtent);
  checkFailure(
      factorization.solve(
          vector({1.0, std::numeric_limits<Scalar>::quiet_NaN()}).view()),
      ErrorCode::NonFiniteInput);
}

void stridedViewsAndLifecycle() {
  std::vector<Scalar> storage{4.0, 99.0, 2.0, 99.0, 2.0, 99.0, 3.0, 99.0};
  const auto input =
      vss::math::MatrixView::create(std::span<Scalar>(storage), 2U, 2U, 4U, 2U);
  VSS_CHECK(input.hasValue());
  const auto result = vss::math::linear::factorizeCholesky(input.value());
  VSS_CHECK(result.hasValue());

  static_assert(!std::is_default_constructible_v<CholeskyFactorization>);
  static_assert(std::is_nothrow_move_constructible_v<CholeskyFactorization>);
  static_assert(std::is_nothrow_move_assignable_v<CholeskyFactorization>);
  CholeskyFactorization source = result.value();
  const CholeskyFactorization moved(std::move(source));
  VSS_CHECK(source.size() == 0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.rank() == 0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.lowerFactor().rows() ==
            0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.conditionReport().matrixOneNorm ==
            0.0); // NOLINT(clang-analyzer-cplusplus.Move)
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  checkFailure(source.solve(vector({1.0, 1.0}).view()),
               ErrorCode::InvalidExtent);
  VSS_CHECK(moved.size() == 2U);

  CholeskyFactorization copy = moved;
  VSS_CHECK(copy.lowerFactor().data() != moved.lowerFactor().data());
  copy = moved;
  checkMatrixNear(copy.lowerFactor(), moved.lowerFactor());
}

void invalidSolvePolicy() {
  const auto result = vss::math::linear::factorizeCholesky(
      matrix({{1.0, 0.0}, {0.0, 1.0}}).view());
  VSS_CHECK(result.hasValue());
  SolveOptions invalid;
  invalid.illConditionedPolicy = invalidEnumValue<IllConditionedSolvePolicy>();
  checkFailure(result.value().solve(vector({1.0, 2.0}).view(), invalid),
               ErrorCode::InvalidDomain);
  checkFailure(result.value().solve(matrix({{1.0}, {2.0}}).view(), invalid),
               ErrorCode::InvalidDomain);
  checkFailure(result.value().whiten(vector({1.0, 2.0}).view(), invalid),
               ErrorCode::InvalidDomain);
  const Vector invalidInput{
      std::vector<Scalar>{std::numeric_limits<Scalar>::quiet_NaN()}};
  checkFailure(result.value().solve(invalidInput.view(), invalid),
               ErrorCode::InvalidDomain);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  spdReconstruction();
  solveAndLogDeterminant();
  semidefiniteRejection();
  indefiniteRejection();
  symmetryBoundary();
  illConditionedSolve();
  invalidSolvePolicy();
  validationAndPrecedence();
  stridedViewsAndLifecycle();
  return 0;
}

// VSS-MATH-TRACEABILITY: CHO-01 CHO-02 CHO-03 CHO-04 CHO-05 CHO-06
