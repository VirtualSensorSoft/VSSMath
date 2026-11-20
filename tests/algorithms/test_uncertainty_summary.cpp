#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>

namespace {

using namespace vss::math;

Limits limits() { return {4096U, 262144U, 0U}; }

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows);

void scalarAndRectangularTransforms() {
  const Matrix left = Matrix(1U, 1U, 2.0);
  const Matrix cross = Matrix(1U, 1U, 0.5);
  const Matrix right = Matrix(1U, 1U, 3.0);
  const auto scalar = covariance::crossCovarianceFromMaps(
      left.view(), cross.view(), right.view(), limits());
  VSS_CHECK(scalar.hasValue());
  VSS_CHECK(scalar.value().at(0U, 0U) == 3.0);

  const Matrix left2 = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const Matrix block = matrix({{1.0, 1.0}, {1.0, -1.0}});
  const Matrix right1 = matrix({{-1.0, -1.0}});
  const auto rectangular = covariance::crossCovarianceFromMaps(
      left2.view(), block.view(), right1.view(), limits());
  VSS_CHECK(rectangular.hasValue());
  VSS_CHECK(rectangular.value().rows() == 2U);
  VSS_CHECK(rectangular.value().columns() == 1U);
  VSS_CHECK(rectangular.value().at(0U, 0U) == -2.0);
  VSS_CHECK(rectangular.value().at(1U, 0U) == 0.0);

  const Matrix reversedLeft = right1;
  const Matrix reversedBlock = matrix({{1.0, 1.0}, {1.0, -1.0}});
  const Matrix reversedRight = left2;
  const auto reversed = covariance::crossCovarianceFromMaps(
      reversedLeft.view(), reversedBlock.view(), reversedRight.view(),
      limits());
  VSS_CHECK(reversed.hasValue());
  VSS_CHECK(reversed.value().rows() == 1U && reversed.value().columns() == 2U);
  VSS_CHECK(reversed.value().at(0U, 0U) == -2.0);
  VSS_CHECK(reversed.value().at(0U, 1U) == 0.0);
}

void emptyScaledAndInvalidInputs() {
  const Matrix left(2U, 0U, 0.0);
  const Matrix cross(0U, 1U, 0.0);
  const Matrix right(1U, 1U, 2.0);
  const auto empty = covariance::crossCovarianceFromMaps(
      left.view(), cross.view(), right.view(), limits());
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().rows() == 2U && empty.value().columns() == 1U);
  VSS_CHECK(empty.value().at(0U, 0U) == 0.0);
  VSS_CHECK(empty.value().at(1U, 0U) == 0.0);

  const Matrix scaleLeft(1U, 1U, 2.0);
  const Matrix maximum(1U, 1U, std::numeric_limits<Scalar>::max());
  const Matrix quarter(1U, 1U, 0.25);
  const auto scaled = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), maximum.view(), quarter.view(), limits());
  VSS_CHECK(scaled.hasValue());
  VSS_CHECK(scaled.value().at(0U, 0U) ==
            std::numeric_limits<Scalar>::max() / 2.0);

  const Matrix badShape(2U, 2U, 0.0);
  const auto mismatch = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), badShape.view(), quarter.view(), limits());
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);

  const Matrix nonFinite(1U, 1U, std::numeric_limits<Scalar>::infinity());
  const auto invalid = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), nonFinite.view(), quarter.view(), limits());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);

  const auto denied = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), maximum.view(), quarter.view(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount, 0.0);
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

} // namespace

int main() {
  scalarAndRectangularTransforms();
  emptyScaledAndInvalidInputs();
  return 0;
}
