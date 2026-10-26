#include "Test.hpp"

#include <cmath>
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
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace {

using vss::math::ConstVectorView;
using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::linear::PositiveDefiniteness;
using vss::math::linear::SymmetricEigenDecomposition;
using vss::math::linear::SymmetricEigenOptions;

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

Vector copyVector(ConstVectorView view) {
  Vector result(view.size(), 0.0);
  for (Index index = 0U; index < view.size(); ++index) {
    result.at(index) = view.at(index);
  }
  return result;
}

void checkEigenvalues(const SymmetricEigenDecomposition& decomposition,
                      std::initializer_list<Scalar> expected,
                      Scalar tolerance = 1.0e-12) {
  const auto values = decomposition.eigenvalues();
  VSS_CHECK(values.size() == expected.size());
  Index index = 0U;
  for (const Scalar value : expected) {
    checkNear(values.at(index), value, tolerance);
    ++index;
  }
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

Matrix reconstruct(const SymmetricEigenDecomposition& decomposition) {
  const auto vectors = decomposition.eigenvectors();
  const auto values = decomposition.eigenvalues();
  Matrix result(decomposition.size(), decomposition.size(), 0.0);
  for (Index matrixRow = 0U; matrixRow < decomposition.size(); ++matrixRow) {
    for (Index matrixColumn = 0U; matrixColumn < decomposition.size();
         ++matrixColumn) {
      Scalar sum = 0.0;
      for (Index index = 0U; index < decomposition.size(); ++index) {
        const Index vectorRow = matrixColumn;
        const Index vectorColumn = index;
        sum += vectors.at(matrixRow, index) * values.at(index) *
               vectors.at(vectorRow, vectorColumn);
      }
      result.at(matrixRow, matrixColumn) = sum;
    }
  }
  return result;
}

Matrix projector(const SymmetricEigenDecomposition& decomposition, Index first,
                 Index count) {
  const auto vectors = decomposition.eigenvectors();
  Matrix result(decomposition.size(), decomposition.size(), 0.0);
  for (Index matrixRow = 0U; matrixRow < decomposition.size(); ++matrixRow) {
    for (Index matrixColumn = 0U; matrixColumn < decomposition.size();
         ++matrixColumn) {
      for (Index index = first; index < first + count; ++index) {
        const Index vectorRow = matrixColumn;
        const Index vectorColumn = index;
        result.at(matrixRow, matrixColumn) +=
            vectors.at(matrixRow, index) * vectors.at(vectorRow, vectorColumn);
      }
    }
  }
  return result;
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

void checkOrthonormal(const SymmetricEigenDecomposition& decomposition) {
  const auto vectors = decomposition.eigenvectors();
  for (Index first = 0U; first < decomposition.size(); ++first) {
    for (Index second = 0U; second < decomposition.size(); ++second) {
      Scalar dot = 0.0;
      for (Index row = 0U; row < decomposition.size(); ++row) {
        dot += vectors.at(row, first) * vectors.at(row, second);
      }
      checkNear(dot, first == second ? 1.0 : 0.0, 1.0e-12);
    }
  }
}

void distinctEigenpairs() {
  const Matrix input = matrix({{2.0, 1.0}, {1.0, 2.0}});
  const auto result = vss::math::linear::decomposeSymmetricEigen(input.view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  checkEigenvalues(decomposition, {1.0, 3.0});
  checkMatrixNear(reconstruct(decomposition), input, 1.0e-12);
  checkOrthonormal(decomposition);
  const Scalar inverseRootTwo = 1.0 / std::numbers::sqrt2;
  checkNear(decomposition.eigenvectors().at(0U, 0U), inverseRootTwo);
  checkNear(decomposition.eigenvectors().at(1U, 0U), -inverseRootTwo);
  checkNear(decomposition.eigenvectors().at(0U, 1U), inverseRootTwo);
  checkNear(decomposition.eigenvectors().at(1U, 1U), inverseRootTwo);
  VSS_CHECK(decomposition.completedSweeps() > 0U);
  VSS_CHECK(decomposition.finalMaximumOffDiagonal() <=
            decomposition.convergenceThreshold());
}

void repeatedEigenvalueSubspace() {
  const auto result = vss::math::linear::decomposeSymmetricEigen(
      matrix({{2.0, 0.0}, {0.0, 2.0}}).view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  checkEigenvalues(decomposition, {2.0, 2.0});
  checkMatrixNear(projector(decomposition, 0U, 2U),
                  matrix({{1.0, 0.0}, {0.0, 1.0}}));
  VSS_CHECK(decomposition.completedSweeps() == 0U);
}

void psdRankLoss() {
  const auto result = vss::math::linear::decomposeSymmetricEigen(
      matrix({{4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}).view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  checkEigenvalues(decomposition, {0.0, 1.0, 4.0});
  VSS_CHECK(decomposition.rank() == 2U);
  VSS_CHECK(decomposition.definiteness() ==
            PositiveDefiniteness::PositiveSemidefinite);
  checkNear(decomposition.minimumEigenvalue(), 0.0);
  checkNear(decomposition.maximumEigenvalue(), 4.0);
  checkMatrixNear(reconstruct(decomposition),
                  matrix({{4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}));
}

void indefiniteMatrix() {
  const auto result = vss::math::linear::decomposeSymmetricEigen(
      matrix({{1.0, 0.0}, {0.0, -1.0}}).view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  checkEigenvalues(decomposition, {-1.0, 1.0});
  VSS_CHECK(decomposition.rank() == 2U);
  VSS_CHECK(decomposition.definiteness() ==
            PositiveDefiniteness::NotPositiveSemidefinite);
}

void symmetryBoundary() {
  const auto accepted = vss::math::linear::decomposeSymmetricEigen(
      matrix({{4.0, 2.0000000000001}, {2.0, 3.0}}).view());
  VSS_CHECK(accepted.hasValue());
  const auto rejected = vss::math::linear::decomposeSymmetricEigen(
      matrix({{4.0, 2.00000000001}, {2.0, 3.0}}).view());
  checkFailure(rejected, ErrorCode::NotSymmetric);
}

void thresholdBand() {
  const auto result = vss::math::linear::decomposeSymmetricEigen(
      matrix({{1.0, 0.0}, {0.0, 1.0e-13}}).view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  checkEigenvalues(decomposition, {1.0e-13, 1.0});
  VSS_CHECK(decomposition.rank() == 1U);
  VSS_CHECK(decomposition.definiteness() ==
            PositiveDefiniteness::PositiveSemidefinite);
  VSS_CHECK(decomposition.eigenvalues().at(0U) == 1.0e-13);
}

void maximumFiniteDiagonalScaleDoesNotOverflowRotation() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Matrix input =
      matrix({{-maximum / 2.0, maximum / 4.0}, {maximum / 4.0, maximum / 2.0}});
  const auto result = vss::math::linear::decomposeSymmetricEigen(input.view());
  VSS_CHECK(result.hasValue());
  const auto& decomposition = result.value();
  const Scalar expectedMagnitude = maximum * (std::sqrt(5.0) / 4.0);
  checkNear(decomposition.eigenvalues().at(0U), -expectedMagnitude,
            maximum * 1.0e-14);
  checkNear(decomposition.eigenvalues().at(1U), expectedMagnitude,
            maximum * 1.0e-14);
  checkOrthonormal(decomposition);
}

void sweepLimitReturnsNoConvergence() {
  SymmetricEigenOptions options;
  options.maximumSweeps = 1U;
  const auto result = vss::math::linear::decomposeSymmetricEigen(
      matrix({{1.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 4.0}}).view(),
      options);
  checkFailure(result, ErrorCode::NoConvergence);
}

void optionAndValidationFailures() {
  SymmetricEigenOptions options;
  options.maximumSweeps = 0U;
  checkFailure(
      vss::math::linear::decomposeSymmetricEigen(Matrix{}.view(), options),
      ErrorCode::InvalidDomain);
  options.maximumSweeps = 1025U;
  checkFailure(
      vss::math::linear::decomposeSymmetricEigen(Matrix{}.view(), options),
      ErrorCode::InvalidDomain);
  options = {};
  options.rankTolerance.relative = -1.0;
  checkFailure(
      vss::math::linear::decomposeSymmetricEigen(Matrix{}.view(), options),
      ErrorCode::InvalidTolerance);
  options = {};
  options.symmetryTolerance.relative = -1.0;
  checkFailure(
      vss::math::linear::decomposeSymmetricEigen(Matrix{}.view(), options),
      ErrorCode::InvalidTolerance);
  options = {};
  const auto empty =
      vss::math::linear::decomposeSymmetricEigen(Matrix{}.view(), options);
  checkFailure(empty, ErrorCode::InvalidExtent);
  checkFailure(
      vss::math::linear::decomposeSymmetricEigen(
          matrix({{std::numeric_limits<Scalar>::quiet_NaN(), 0.0}, {0.0, 1.0}})
              .view(),
          options),
      ErrorCode::NonFiniteInput);
}

void stridedInputAndLifecycle() {
  std::vector<Scalar> storage{2.0, 99.0, 1.0, 99.0, 1.0, 99.0, 2.0, 99.0};
  const auto input =
      vss::math::MatrixView::create(std::span<Scalar>(storage), 2U, 2U, 4U, 2U);
  VSS_CHECK(input.hasValue());
  const auto result = vss::math::linear::decomposeSymmetricEigen(input.value());
  VSS_CHECK(result.hasValue());

  static_assert(!std::is_default_constructible_v<SymmetricEigenDecomposition>);
  static_assert(
      std::is_nothrow_move_constructible_v<SymmetricEigenDecomposition>);
  static_assert(std::is_nothrow_move_assignable_v<SymmetricEigenDecomposition>);
  SymmetricEigenDecomposition source = result.value();
  const SymmetricEigenDecomposition moved(std::move(source));
  const Extent movedFromSize = source.size(); // NOLINT(bugprone-use-after-move,
                                              // clang-analyzer-cplusplus.Move)
  const Extent movedFromRank = source.rank(); // NOLINT(bugprone-use-after-move,
                                              // clang-analyzer-cplusplus.Move)
  const bool movedFromValuesEmpty =
      source.eigenvalues().empty(); // NOLINT(bugprone-use-after-move,
                                    // clang-analyzer-cplusplus.Move)
  const bool movedFromVectorsEmpty =
      source.eigenvectors().empty(); // NOLINT(bugprone-use-after-move,
                                     // clang-analyzer-cplusplus.Move)
  VSS_CHECK(movedFromSize == 0U);
  VSS_CHECK(movedFromRank == 0U);
  VSS_CHECK(movedFromValuesEmpty);
  VSS_CHECK(movedFromVectorsEmpty);
  VSS_CHECK(moved.size() == 2U);

  SymmetricEigenDecomposition copy = moved;
  const SymmetricEigenDecomposition assigned = std::move(copy);
  checkVectorNear(copyVector(assigned.eigenvalues()),
                  copyVector(moved.eigenvalues()));
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  distinctEigenpairs();
  repeatedEigenvalueSubspace();
  psdRankLoss();
  indefiniteMatrix();
  symmetryBoundary();
  thresholdBand();
  maximumFiniteDiagonalScaleDoesNotOverflowRotation();
  sweepLimitReturnsNoConvergence();
  optionAndValidationFailures();
  stridedInputAndLifecycle();
  return 0;
}

// VSS-MATH-TRACEABILITY: EIG-01 EIG-02 EIG-03 EIG-04 EIG-05 EIG-06
// VSS-MATH-TRACEABILITY: SUP-EIG-01 SUP-EIG-02
