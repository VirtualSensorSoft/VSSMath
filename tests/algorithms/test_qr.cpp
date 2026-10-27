#include "Test.hpp"

#include <algorithm>
#include <cmath>
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
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/QR.hpp>

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
using vss::math::linear::ConditionClassification;
using vss::math::linear::ConditionReport;
using vss::math::linear::FactorizationOptions;
using vss::math::linear::IllConditionedSolvePolicy;
using vss::math::linear::QRFactorization;
using vss::math::linear::SolveOptions;

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

void checkMatrixExact(const Matrix& actual, const Matrix& expected) {
  VSS_CHECK(actual.rows() == expected.rows());
  VSS_CHECK(actual.columns() == expected.columns());
  for (Index row = 0U; row < actual.rows(); ++row) {
    for (Index column = 0U; column < actual.columns(); ++column) {
      VSS_CHECK(actual.at(row, column) == expected.at(row, column));
    }
  }
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkConditionReport(const std::optional<ConditionReport>& actual,
                          const ConditionReport& expected) {
  VSS_CHECK(actual.has_value());
  const auto observed = actual.value_or(ConditionReport{
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

ConditionReport
requireConditionReport(const std::optional<ConditionReport>& actual) {
  VSS_CHECK(actual.has_value());
  return actual.value_or(ConditionReport{
      .matrixOneNorm = 0.0,
      .estimatedConditionNumber1 = 0.0,
      .conditionLimit = 0.0,
      .classification = ConditionClassification::IllConditioned});
}

Matrix multiply(const Matrix& left, const Matrix& right) {
  VSS_CHECK(left.columns() == right.rows());
  Matrix result(left.rows(), right.columns());
  for (Index row = 0U; row < left.rows(); ++row) {
    for (Index column = 0U; column < right.columns(); ++column) {
      Scalar sum = 0.0;
      for (Index index = 0U; index < left.columns(); ++index) {
        sum += left.at(row, index) * right.at(index, column);
      }
      result.at(row, column) = sum;
    }
  }
  return result;
}

Matrix transposeMultiply(const Matrix& left, const Matrix& right) {
  VSS_CHECK(left.rows() == right.rows());
  Matrix result(left.columns(), right.columns());
  for (Index resultRow = 0U; resultRow < left.columns(); ++resultRow) {
    for (Index resultColumn = 0U; resultColumn < right.columns();
         ++resultColumn) {
      Scalar sum = 0.0;
      for (Index leftRow = 0U; leftRow < left.rows(); ++leftRow) {
        sum += left.at(leftRow, resultRow) * right.at(leftRow, resultColumn);
      }
      result.at(resultRow, resultColumn) = sum;
    }
  }
  return result;
}

Vector transposeMultiply(const Matrix& left, const Vector& right) {
  VSS_CHECK(left.rows() == right.size());
  Vector result(left.columns());
  for (Index column = 0U; column < left.columns(); ++column) {
    Scalar sum = 0.0;
    for (Index row = 0U; row < left.rows(); ++row) {
      sum += left.at(row, column) * right.at(row);
    }
    result.at(column) = sum;
  }
  return result;
}

Matrix permutedColumns(const Matrix& input,
                       std::span<const Index> permutation) {
  VSS_CHECK(input.columns() == permutation.size());
  Matrix result(input.rows(), input.columns());
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      result.at(row, column) = input.at(row, permutation[column]);
    }
  }
  return result;
}

Scalar frobeniusNorm(const Matrix& input) {
  Scalar scale = 0.0;
  Scalar sumSquares = 0.0;
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      const Scalar magnitude = std::abs(input.at(row, column));
      if (magnitude == 0.0) {
        continue;
      }
      if (scale == 0.0) {
        scale = magnitude;
        sumSquares = 1.0;
      } else if (magnitude > scale) {
        sumSquares =
            1.0 + (sumSquares * (scale / magnitude) * (scale / magnitude));
        scale = magnitude;
      } else {
        sumSquares += (magnitude / scale) * (magnitude / scale);
      }
    }
  }
  return scale == 0.0 ? 0.0 : scale * std::sqrt(sumSquares);
}

Scalar gamma(Extent count) {
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  const Scalar scaled = static_cast<Scalar>(count) * epsilon;
  return scaled / (1.0 - scaled);
}

void checkQrEvidence(const Matrix& input,
                     const QRFactorization& factorization) {
  const Matrix q = factorization.economyQ();
  const Matrix& r = factorization.upperFactor();
  VSS_CHECK(q.rows() == input.rows());
  VSS_CHECK(q.columns() == input.columns());
  VSS_CHECK(r.rows() == input.columns());
  VSS_CHECK(r.columns() == input.columns());

  const Matrix ap = permutedColumns(input, factorization.permutation());
  const Matrix qr = multiply(q, r);
  Matrix reconstructionError(input.rows(), input.columns());
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      reconstructionError.at(row, column) =
          ap.at(row, column) - qr.at(row, column);
    }
  }
  const Scalar scale =
      std::max(frobeniusNorm(input), std::numeric_limits<Scalar>::min());
  const Scalar reconstructionBound =
      200.0 * gamma(std::max(input.rows(), input.columns()));
  VSS_CHECK(frobeniusNorm(reconstructionError) / scale <= reconstructionBound);

  const Matrix qtq = transposeMultiply(q, q);
  Matrix orthogonalityError(qtq.rows(), qtq.columns());
  for (Index row = 0U; row < qtq.rows(); ++row) {
    for (Index column = 0U; column < qtq.columns(); ++column) {
      orthogonalityError.at(row, column) =
          qtq.at(row, column) - (row == column ? 1.0 : 0.0);
    }
  }
  VSS_CHECK(frobeniusNorm(orthogonalityError) <= reconstructionBound);
}

void exactOverdeterminedLeastSquares() {
  const Matrix input = matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.rows() == 3U);
  VSS_CHECK(factorization.columns() == 2U);
  VSS_CHECK(factorization.rank() == 2U);
  VSS_CHECK(factorization.hasFullColumnRank());
  VSS_CHECK(!factorization.isSquare());
  VSS_CHECK(factorization.permutation().size() == 2U);
  VSS_CHECK(factorization.permutation()[0U] == 1U);
  VSS_CHECK(factorization.permutation()[1U] == 0U);
  VSS_CHECK(!factorization.conditionReport().has_value());
  checkQrEvidence(input, factorization);
  checkMatrixExact(input, matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}));

  const auto solved =
      factorization.leastSquares(vector({1.0, 3.0, 5.0}).view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({1.0, 2.0}));
  VSS_CHECK(!solved.value().conditionReport().has_value());
}

void permutationOrientation() {
  const Matrix input = matrix({{1.0, 10.0}, {0.0, 0.0}, {0.0, 1.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.permutation()[0U] == 1U);
  VSS_CHECK(factorization.permutation()[1U] == 0U);
  checkQrEvidence(input, factorization);
  const auto solved =
      factorization.leastSquares(vector({-8.0, 0.0, -1.0}).view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({2.0, -1.0}));
}

void normTieUsesLowestOriginalColumn() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().permutation()[0U] == 0U);
  VSS_CHECK(result.value().permutation()[1U] == 1U);
  checkQrEvidence(input, result.value());
}

void laterNormTieUsesLowestOriginalColumn() {
  const Matrix input =
      matrix({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto permutation = result.value().permutation();
  VSS_CHECK(permutation.size() == 3U);
  VSS_CHECK(permutation[0U] == 2U);
  VSS_CHECK(permutation[1U] == 0U);
  VSS_CHECK(permutation[2U] == 1U);
  checkQrEvidence(input, result.value());
}

void publicDiagonalIsNonNegative() {
  const Matrix input = matrix({{-2.0, 0.0}, {0.0, 1.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const Matrix& r = result.value().upperFactor();
  VSS_CHECK(r.at(0U, 0U) >= 0.0);
  VSS_CHECK(r.at(1U, 1U) >= 0.0);
  const Scalar firstDiagonal = r.at(0U, 0U);
  const Scalar secondDiagonal = r.at(1U, 1U);
  VSS_CHECK(firstDiagonal != 0.0 || !std::signbit(firstDiagonal));
  VSS_CHECK(secondDiagonal != 0.0 || !std::signbit(secondDiagonal));
  checkQrEvidence(input, result.value());
}

void normalizedSignsReachEconomyQAndSolve() {
  const Matrix input = matrix({{2.0, 0.0}, {0.0, 1.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  checkMatrixNear(result.value().economyQ(), matrix({{1.0, 0.0}, {0.0, 1.0}}));
  checkMatrixNear(result.value().upperFactor(),
                  matrix({{2.0, 0.0}, {0.0, 1.0}}));
  const auto solved = result.value().leastSquares(vector({4.0, 3.0}).view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({2.0, 3.0}));
  checkQrEvidence(input, result.value());
}

void zeroTrailingNormIsRankDeficient() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 0.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.rank() == 1U);
  VSS_CHECK(factorization.upperFactor().at(1U, 1U) == 0.0);
  VSS_CHECK(!std::signbit(factorization.upperFactor().at(1U, 1U)));
  const ConditionReport condition =
      requireConditionReport(factorization.conditionReport());
  VSS_CHECK(std::isinf(condition.estimatedConditionNumber1));
  VSS_CHECK(condition.classification ==
            ConditionClassification::IllConditioned);
  checkFailure(factorization.leastSquares(vector({1.0, 0.0}).view()),
               ErrorCode::RankDeficient);
  checkQrEvidence(input, factorization);
}

void maximumFiniteScaleDoesNotOverflowReflector() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Matrix input = matrix({{maximum}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(std::isfinite(factorization.upperFactor().at(0U, 0U)));
  checkMatrixNear(factorization.economyQ(), matrix({{1.0}}));
  checkNear(factorization.upperFactor().at(0U, 0U), maximum, 0.0);
  VSS_CHECK(factorization.rank() == 1U);
  const ConditionReport condition =
      requireConditionReport(factorization.conditionReport());
  VSS_CHECK(std::isfinite(condition.estimatedConditionNumber1));
  checkNear(condition.estimatedConditionNumber1, 1.0, 1.0e-12);
  checkQrEvidence(input, factorization);
}

void rankThresholdEqualityIsUnusable() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 1.0e-12}});
  FactorizationOptions options{};
  options.rankTolerance.absolute = 0.0;
  options.rankTolerance.relative = 1.0e-12;
  const auto result = vss::math::linear::factorizeQR(input.view(), options);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rankThreshold() == 1.0e-12);
  VSS_CHECK(result.value().rank() == 1U);
  checkQrEvidence(input, result.value());
}

void inconsistentLeastSquares() {
  const Matrix input = matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}});
  const Vector rightHandSide = vector({1.0, 2.0, 2.0, 4.0});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto solved = result.value().leastSquares(rightHandSide.view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({0.9, 0.9}), 1.0e-11);

  const Vector residual = vector({0.1, 0.2, -0.7, 0.4});
  checkNear(std::sqrt(0.7),
            std::sqrt((0.1 * 0.1) + (0.2 * 0.2) + (0.7 * 0.7) + (0.4 * 0.4)));
  const Vector gradient = transposeMultiply(input, residual);
  Scalar gradientInfinity = 0.0;
  for (Index index = 0U; index < gradient.size(); ++index) {
    gradientInfinity = std::max(gradientInfinity, std::abs(gradient.at(index)));
  }
  Scalar matrixOne = 0.0;
  for (Index column = 0U; column < input.columns(); ++column) {
    Scalar columnSum = 0.0;
    for (Index row = 0U; row < input.rows(); ++row) {
      columnSum += std::abs(input.at(row, column));
    }
    matrixOne = std::max(matrixOne, columnSum);
  }
  Scalar residualInfinity = 0.0;
  for (Index index = 0U; index < residual.size(); ++index) {
    residualInfinity = std::max(residualInfinity, std::abs(residual.at(index)));
  }
  const Scalar scaledGradient =
      gradientInfinity / std::max(matrixOne * residualInfinity,
                                  std::numeric_limits<Scalar>::min());
  VSS_CHECK(scaledGradient <=
            200.0 * gamma(std::max(input.rows(), input.columns())));
  checkQrEvidence(input, result.value());
}

void repeatedMatrixRightHandSides() {
  const Matrix input = matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}});
  const Matrix rightHandSides = matrix({{1.0, -1.0}, {3.0, 2.0}, {5.0, 5.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto solved = result.value().leastSquares(rightHandSides.view());
  VSS_CHECK(solved.hasValue());
  checkMatrixNear(solved.value().solution(), matrix({{1.0, -1.0}, {2.0, 3.0}}));
  VSS_CHECK(!solved.value().conditionReport().has_value());
  checkMatrixExact(rightHandSides,
                   matrix({{1.0, -1.0}, {3.0, 2.0}, {5.0, 5.0}}));
}

void rankDeficientColumns() {
  const Matrix input = matrix({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.rank() == 1U);
  VSS_CHECK(!factorization.conditionReport().has_value());
  checkFailure(factorization.leastSquares(vector({1.0, 2.0, 3.0}).view()),
               ErrorCode::RankDeficient);
  checkFailure(factorization.leastSquares(
                   matrix({{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}).view()),
               ErrorCode::RankDeficient);
  checkQrEvidence(input, factorization);
}

void symmetricIndefiniteSquareSystem() {
  const Matrix input =
      matrix({{2.0, 0.0, 1.0}, {0.0, 2.0, 1.0}, {1.0, 1.0, 0.0}});
  const auto result = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.matrixOneNorm() == 3.0);
  const ConditionReport condition =
      requireConditionReport(factorization.conditionReport());
  const Scalar estimate = condition.estimatedConditionNumber1;
  VSS_CHECK(estimate >= 3.0 && estimate <= 12.0);
  const auto solved =
      factorization.leastSquares(vector({5.0, 7.0, 3.0}).view());
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({1.0, 2.0, 3.0}));
  checkConditionReport(solved.value().conditionReport(), condition);
  checkQrEvidence(input, factorization);
}

void squareConditionPolicyBoundary() {
  const Matrix input = matrix({{1.0, 0.0}, {0.0, 1.0e-11}});
  FactorizationOptions permissive{};
  permissive.conditionLimit = 1.0e300;
  const auto first = vss::math::linear::factorizeQR(input.view(), permissive);
  VSS_CHECK(first.hasValue());
  const ConditionReport firstCondition =
      requireConditionReport(first.value().conditionReport());
  const Scalar estimate = firstCondition.estimatedConditionNumber1;
  FactorizationOptions equality = permissive;
  equality.conditionLimit = estimate;
  const auto second = vss::math::linear::factorizeQR(input.view(), equality);
  VSS_CHECK(second.hasValue());
  const ConditionReport secondCondition =
      requireConditionReport(second.value().conditionReport());
  VSS_CHECK(secondCondition.classification ==
            ConditionClassification::IllConditioned);
  checkFailure(second.value().leastSquares(vector({2.0, 3.0e-11}).view()),
               ErrorCode::IllConditioned);
  const SolveOptions allow{.illConditionedPolicy =
                               IllConditionedSolvePolicy::AllowWithDiagnostic};
  const auto solved =
      second.value().leastSquares(vector({2.0, 3.0e-11}).view(), allow);
  VSS_CHECK(solved.hasValue());
  checkVectorNear(solved.value().solution(), vector({2.0, 3.0}), 1.0e-10);
  checkConditionReport(solved.value().conditionReport(), secondCondition);
  checkQrEvidence(input, second.value());
}

void squareConditionOverflowIsDiagnostic() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Scalar smallest = std::numeric_limits<Scalar>::min();
  const Matrix input = matrix({{maximum, 0.0}, {0.0, smallest}});
  FactorizationOptions options{};
  options.rankTolerance.absolute = 0.0;
  options.rankTolerance.relative = 0.0;
  const auto result = vss::math::linear::factorizeQR(input.view(), options);
  VSS_CHECK(result.hasValue());
  const auto& factorization = result.value();
  VSS_CHECK(factorization.hasFullColumnRank());
  const ConditionReport condition =
      requireConditionReport(factorization.conditionReport());
  VSS_CHECK(std::isinf(condition.estimatedConditionNumber1));
  VSS_CHECK(condition.classification ==
            ConditionClassification::IllConditioned);
  checkFailure(factorization.leastSquares(vector({1.0, smallest}).view()),
               ErrorCode::IllConditioned);
  const SolveOptions allow{.illConditionedPolicy =
                               IllConditionedSolvePolicy::AllowWithDiagnostic};
  const auto solved =
      factorization.leastSquares(vector({1.0, smallest}).view(), allow);
  VSS_CHECK(solved.hasValue());
  VSS_CHECK(std::isfinite(solved.value().solution().at(0U)));
  VSS_CHECK(std::isfinite(solved.value().solution().at(1U)));
  const ConditionReport solvedCondition =
      requireConditionReport(solved.value().conditionReport());
  VSS_CHECK(std::isinf(solvedCondition.estimatedConditionNumber1));
  checkQrEvidence(input, factorization);
}

void validationFailures() {
  const Matrix empty;
  for (const Scalar invalid : {std::numeric_limits<Scalar>::quiet_NaN(),
                               std::numeric_limits<Scalar>::infinity(), -1.0}) {
    FactorizationOptions options{};
    options.rankTolerance.absolute = invalid;
    checkFailure(vss::math::linear::factorizeQR(empty.view(), options),
                 ErrorCode::InvalidTolerance);
    options = FactorizationOptions{};
    options.rankTolerance.relative = invalid;
    checkFailure(vss::math::linear::factorizeQR(empty.view(), options),
                 ErrorCode::InvalidTolerance);
  }
  for (const Scalar invalid :
       {std::numeric_limits<Scalar>::quiet_NaN(),
        std::numeric_limits<Scalar>::infinity(), 1.0, 0.0, -1.0}) {
    FactorizationOptions options{};
    options.conditionLimit = invalid;
    checkFailure(vss::math::linear::factorizeQR(empty.view(), options),
                 ErrorCode::InvalidTolerance);
  }
  checkFailure(vss::math::linear::factorizeQR(Matrix(0U, 0U).view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::linear::factorizeQR(Matrix(2U, 0U).view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::linear::factorizeQR(Matrix(0U, 2U).view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::linear::factorizeQR(Matrix(2U, 3U).view()),
               ErrorCode::InvalidExtent);

  Matrix nonFinite = matrix({{std::numeric_limits<Scalar>::quiet_NaN(), 0.0},
                             {1.0, 1.0},
                             {2.0, 2.0}});
  checkFailure(vss::math::linear::factorizeQR(nonFinite.view()),
               ErrorCode::NonFiniteInput);

  FactorizationOptions overflowing{};
  overflowing.rankTolerance.relative = std::numeric_limits<Scalar>::max();
  checkFailure(vss::math::linear::factorizeQR(
                   matrix({{2.0, 0.0}, {0.0, 1.0}}).view(), overflowing),
               ErrorCode::NonFiniteResult);

  const auto valid = vss::math::linear::factorizeQR(
      matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}).view());
  VSS_CHECK(valid.hasValue());
  checkFailure(valid.value().leastSquares(vector({1.0, 2.0}).view()),
               ErrorCode::DimensionMismatch);
  checkFailure(valid.value().leastSquares(matrix({{1.0}, {2.0}}).view()),
               ErrorCode::DimensionMismatch);
  checkFailure(valid.value().leastSquares(Matrix(3U, 0U).view()),
               ErrorCode::InvalidExtent);
  checkFailure(
      valid.value().leastSquares(
          vector({1.0, std::numeric_limits<Scalar>::quiet_NaN(), 3.0}).view()),
      ErrorCode::NonFiniteInput);
  checkFailure(
      valid.value().leastSquares(
          matrix({{1.0}, {std::numeric_limits<Scalar>::quiet_NaN()}, {3.0}})
              .view()),
      ErrorCode::NonFiniteInput);
}

void rightHandSideValidationPrecedence() {
  const auto rankDeficient = vss::math::linear::factorizeQR(
      matrix({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}}).view());
  VSS_CHECK(rankDeficient.hasValue());
  const Vector nanVector =
      vector({1.0, std::numeric_limits<Scalar>::quiet_NaN(), 3.0});
  checkFailure(rankDeficient.value().leastSquares(nanVector.view()),
               ErrorCode::NonFiniteInput);
  checkFailure(
      rankDeficient.value().leastSquares(
          matrix({{1.0}, {std::numeric_limits<Scalar>::quiet_NaN()}, {3.0}})
              .view()),
      ErrorCode::NonFiniteInput);
  const SolveOptions allow{.illConditionedPolicy =
                               IllConditionedSolvePolicy::AllowWithDiagnostic};
  checkFailure(
      rankDeficient.value().leastSquares(vector({1.0, 2.0, 3.0}).view(), allow),
      ErrorCode::RankDeficient);

  const auto illConditioned = vss::math::linear::factorizeQR(
      matrix({{1.0, 0.0}, {0.0, 1.0e-11}}).view());
  VSS_CHECK(illConditioned.hasValue());
  checkFailure(illConditioned.value().leastSquares(nanVector.view()),
               ErrorCode::DimensionMismatch);
  const Vector illNan = vector({1.0, std::numeric_limits<Scalar>::quiet_NaN()});
  checkFailure(illConditioned.value().leastSquares(illNan.view()),
               ErrorCode::NonFiniteInput);
  checkFailure(illConditioned.value().leastSquares(vector({1.0, 1.0}).view()),
               ErrorCode::IllConditioned);
}

void stridedInputAndRightHandSideViews() {
  std::vector<Scalar> matrixStorage{1.0,  0.0, 99.0, 1.0, 1.0,
                                    99.0, 1.0, 2.0,  99.0};
  const auto matrixView = vss::math::MatrixView::create(
      std::span<Scalar>(matrixStorage), 3U, 2U, 3U);
  VSS_CHECK(matrixView.hasValue());
  const auto factorization = vss::math::linear::factorizeQR(matrixView.value());
  VSS_CHECK(factorization.hasValue());

  std::vector<Scalar> rhsStorage{1.0, 99.0, 3.0, 99.0, 5.0};
  const auto rhsView =
      vss::math::MatrixView::create(std::span<Scalar>(rhsStorage), 3U, 1U, 2U);
  VSS_CHECK(rhsView.hasValue());
  const auto solved = factorization.value().leastSquares(rhsView.value());
  VSS_CHECK(solved.hasValue());
  checkMatrixNear(solved.value().solution(), matrix({{1.0}, {2.0}}));
}

void copyIndependence() {
  const auto originalResult = vss::math::linear::factorizeQR(
      matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}).view());
  const auto otherResult =
      vss::math::linear::factorizeQR(matrix({{2.0, 0.0}, {0.0, 1.0}}).view());
  VSS_CHECK(originalResult.hasValue());
  VSS_CHECK(otherResult.hasValue());
  const auto& original = originalResult.value();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const QRFactorization firstCopy = original;
  QRFactorization assigned = original;
  assigned = otherResult.value();
  checkMatrixNear(firstCopy.upperFactor(), original.upperFactor());
  checkVectorNear(
      firstCopy.leastSquares(vector({1.0, 3.0, 5.0}).view()).value().solution(),
      vector({1.0, 2.0}));
  checkMatrixNear(assigned.economyQ(), otherResult.value().economyQ());
}

void movedFromStateIsEmpty() {
  static_assert(!std::is_default_constructible_v<QRFactorization>);
  static_assert(std::is_nothrow_move_constructible_v<QRFactorization>);
  static_assert(std::is_nothrow_move_assignable_v<QRFactorization>);

  const auto result = vss::math::linear::factorizeQR(
      matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}).view());
  VSS_CHECK(result.hasValue());
  QRFactorization source = result.value();
  const QRFactorization moved(std::move(source));
  VSS_CHECK(source.rows() == 0U);    // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.columns() == 0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.rank() == 0U);    // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.rankThreshold() ==
            0.0); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.matrixOneNorm() ==
            0.0); // NOLINT(clang-analyzer-cplusplus.Move)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(!source.hasFullColumnRank());
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(!source.isSquare());
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.permutation().empty());
  VSS_CHECK(source.upperFactor().rows() ==
            0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(source.upperFactor().columns() ==
            0U); // NOLINT(clang-analyzer-cplusplus.Move)
  VSS_CHECK(!source.conditionReport()
                 .has_value()); // NOLINT(clang-analyzer-cplusplus.Move)
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  const Matrix emptyQ = source.economyQ();
  VSS_CHECK(emptyQ.rows() == 0U);
  VSS_CHECK(emptyQ.columns() == 0U);

  const Vector rightHandSide = vector({1.0, 3.0, 5.0});
  const Matrix matrixRightHandSide = matrix({{1.0}, {3.0}, {5.0}});
  checkFailure(
      source.leastSquares(rightHandSide.view()),
      ErrorCode::
          InvalidExtent); // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  checkFailure(
      source.leastSquares(matrixRightHandSide.view()),
      ErrorCode::
          InvalidExtent); // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  checkVectorNear(moved.leastSquares(rightHandSide.view()).value().solution(),
                  vector({1.0, 2.0}));
  checkQrEvidence(matrix({{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}), moved);
}

void invalidSolvePolicy() {
  const auto result =
      vss::math::linear::factorizeQR(matrix({{1.0, 0.0}, {0.0, 1.0}}).view());
  VSS_CHECK(result.hasValue());
  SolveOptions invalid;
  invalid.illConditionedPolicy = invalidEnumValue<IllConditionedSolvePolicy>();
  checkFailure(result.value().leastSquares(vector({1.0, 2.0}).view(), invalid),
               ErrorCode::InvalidDomain);
  checkFailure(
      result.value().leastSquares(matrix({{1.0}, {2.0}}).view(), invalid),
      ErrorCode::InvalidDomain);
  const Vector invalidInput{
      std::vector<Scalar>{std::numeric_limits<Scalar>::quiet_NaN()}};
  checkFailure(result.value().leastSquares(invalidInput.view(), invalid),
               ErrorCode::InvalidDomain);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  exactOverdeterminedLeastSquares();
  permutationOrientation();
  normTieUsesLowestOriginalColumn();
  laterNormTieUsesLowestOriginalColumn();
  publicDiagonalIsNonNegative();
  normalizedSignsReachEconomyQAndSolve();
  zeroTrailingNormIsRankDeficient();
  maximumFiniteScaleDoesNotOverflowReflector();
  rankThresholdEqualityIsUnusable();
  inconsistentLeastSquares();
  repeatedMatrixRightHandSides();
  rankDeficientColumns();
  symmetricIndefiniteSquareSystem();
  squareConditionPolicyBoundary();
  squareConditionOverflowIsDiagnostic();
  validationFailures();
  rightHandSideValidationPrecedence();
  invalidSolvePolicy();
  stridedInputAndRightHandSideViews();
  copyIndependence();
  movedFromStateIsEmpty();
  return 0;
}

// VSS-MATH-TRACEABILITY: QR-01 QR-02 QR-03 QR-04 QR-05 QR-06
// VSS-MATH-TRACEABILITY: SUP-QR-01 SUP-QR-02 SUP-QR-03
// VSS-MATH-TRACEABILITY: SUP-QR-04 SUP-QR-05
