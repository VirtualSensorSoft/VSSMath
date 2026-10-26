#include "Test.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/TriangularSolve.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::linear::DiagonalMode;
using vss::math::linear::TransposeMode;
using vss::math::linear::Triangle;
using vss::math::linear::TriangularSolveOptions;

std::vector<std::byte> bytes(const Matrix& matrix) {
  std::vector<std::byte> result(matrix.size() * sizeof(Scalar));
  std::memcpy(result.data(), matrix.data(), result.size());
  return result;
}

std::vector<std::byte> bytes(const Vector& vector) {
  std::vector<std::byte> result(vector.size() * sizeof(Scalar));
  std::memcpy(result.data(), vector.data(), result.size());
  return result;
}

void checkUnchanged(const std::vector<std::byte>& before,
                    const Matrix& matrix) {
  VSS_CHECK(before.size() == matrix.size() * sizeof(Scalar));
  VSS_CHECK(std::memcmp(before.data(), matrix.data(), before.size()) == 0);
}

void checkUnchanged(const std::vector<std::byte>& before,
                    const Vector& vector) {
  VSS_CHECK(before.size() == vector.size() * sizeof(Scalar));
  VSS_CHECK(std::memcmp(before.data(), vector.data(), before.size()) == 0);
}

Matrix lowerFactor() {
  Matrix factor(2U, 2U);
  factor.at(0U, 0U) = 2.0;
  factor.at(0U, 1U) = std::numeric_limits<Scalar>::quiet_NaN();
  factor.at(1U, 0U) = 3.0;
  factor.at(1U, 1U) = 1.0;
  return factor;
}

Matrix upperFactor() {
  Matrix factor(2U, 2U);
  factor.at(0U, 0U) = 2.0;
  factor.at(0U, 1U) = 1.0;
  factor.at(1U, 0U) = std::numeric_limits<Scalar>::infinity();
  factor.at(1U, 1U) = 3.0;
  return factor;
}

void lowerVectorSolve() {
  auto factor = lowerFactor();
  Vector rightHandSide(std::vector<Scalar>{2.0, 5.0});
  const auto factorBefore = bytes(factor);
  const auto rightHandSideBefore = bytes(rightHandSide);

  const auto result = vss::math::linear::solveTriangular(
      factor.view(), rightHandSide.view(),
      TriangularSolveOptions{Triangle::Lower});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().solution().size() == 2U);
  VSS_CHECK(result.value().solution().at(0U) == 1.0);
  VSS_CHECK(result.value().solution().at(1U) == 2.0);
  VSS_CHECK(!result.value().conditionReport().has_value());
  checkUnchanged(factorBefore, factor);
  checkUnchanged(rightHandSideBefore, rightHandSide);
}

void upperVectorSolve() {
  auto factor = upperFactor();
  Vector rightHandSide(std::vector<Scalar>{4.0, 6.0});
  const auto factorBefore = bytes(factor);
  const auto rightHandSideBefore = bytes(rightHandSide);

  const auto result = vss::math::linear::solveTriangular(
      factor.view(), rightHandSide.view(),
      TriangularSolveOptions{Triangle::Upper});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().solution().size() == 2U);
  VSS_CHECK(result.value().solution().at(0U) == 1.0);
  VSS_CHECK(result.value().solution().at(1U) == 2.0);
  VSS_CHECK(!result.value().conditionReport().has_value());
  checkUnchanged(factorBefore, factor);
  checkUnchanged(rightHandSideBefore, rightHandSide);
}

void transposedLowerSolve() {
  auto factor = lowerFactor();
  Vector rightHandSide(std::vector<Scalar>{8.0, 2.0});

  TriangularSolveOptions options{Triangle::Lower};
  options.transpose = TransposeMode::Transpose;
  const auto result = vss::math::linear::solveTriangular(
      factor.view(), rightHandSide.view(), options);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().solution().at(0U) == 1.0);
  VSS_CHECK(result.value().solution().at(1U) == 2.0);
  VSS_CHECK(!result.value().conditionReport().has_value());
}

void multipleRightHandSides() {
  auto factor = lowerFactor();
  Matrix rightHandSides(2U, 2U);
  rightHandSides.at(0U, 0U) = 2.0;
  rightHandSides.at(0U, 1U) = -2.0;
  rightHandSides.at(1U, 0U) = 5.0;
  rightHandSides.at(1U, 1U) = 0.0;
  const auto factorBefore = bytes(factor);
  const auto rightHandSidesBefore = bytes(rightHandSides);

  const auto result = vss::math::linear::solveTriangular(
      factor.view(), rightHandSides.view(),
      TriangularSolveOptions{Triangle::Lower});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().solution().rows() == 2U);
  VSS_CHECK(result.value().solution().columns() == 2U);
  VSS_CHECK(result.value().solution().at(0U, 0U) == 1.0);
  VSS_CHECK(result.value().solution().at(0U, 1U) == -1.0);
  VSS_CHECK(result.value().solution().at(1U, 0U) == 2.0);
  VSS_CHECK(result.value().solution().at(1U, 1U) == 3.0);
  VSS_CHECK(!result.value().conditionReport().has_value());
  checkUnchanged(factorBefore, factor);
  checkUnchanged(rightHandSidesBefore, rightHandSides);

  Matrix unitFactor(2U, 2U);
  unitFactor.at(0U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  unitFactor.at(0U, 1U) = std::numeric_limits<Scalar>::quiet_NaN();
  unitFactor.at(1U, 0U) = 3.0;
  unitFactor.at(1U, 1U) = std::numeric_limits<Scalar>::infinity();
  TriangularSolveOptions unitOptions{Triangle::Lower};
  unitOptions.diagonal = DiagonalMode::Unit;
  const auto unitResult = vss::math::linear::solveTriangular(
      unitFactor.view(), rightHandSides.view(), unitOptions);
  VSS_CHECK(unitResult.hasValue());
  VSS_CHECK(unitResult.value().solution().at(0U, 0U) == 2.0);
  VSS_CHECK(unitResult.value().solution().at(0U, 1U) == -2.0);
  VSS_CHECK(unitResult.value().solution().at(1U, 0U) == -1.0);
  VSS_CHECK(unitResult.value().solution().at(1U, 1U) == 6.0);
}

template <class Value>
void checkFailure(const Value& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void failureClassification() {
  Matrix singular(2U, 2U);
  singular.at(0U, 0U) = 1.0;
  singular.at(0U, 1U) = 0.0;
  singular.at(1U, 0U) = 0.0;
  singular.at(1U, 1U) = 0.0;
  Vector twoValues(std::vector<Scalar>{1.0, 1.0});
  checkFailure(vss::math::linear::solveTriangular(
                   singular.view(), twoValues.view(),
                   TriangularSolveOptions{Triangle::Lower}),
               ErrorCode::Singular);

  Vector threeValues(std::vector<Scalar>{1.0, 1.0, 1.0});
  checkFailure(vss::math::linear::solveTriangular(
                   singular.view(), threeValues.view(),
                   TriangularSolveOptions{Triangle::Lower}),
               ErrorCode::DimensionMismatch);

  Matrix nonFinite(2U, 2U);
  nonFinite.at(0U, 0U) = 1.0;
  nonFinite.at(0U, 1U) = 0.0;
  nonFinite.at(1U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  nonFinite.at(1U, 1U) = 1.0;
  checkFailure(vss::math::linear::solveTriangular(
                   nonFinite.view(), twoValues.view(),
                   TriangularSolveOptions{Triangle::Lower}),
               ErrorCode::NonFiniteInput);

  Matrix empty;
  Vector emptyRightHandSide;
  TriangularSolveOptions nanTolerance{Triangle::Lower};
  nanTolerance.rankTolerance.relative =
      std::numeric_limits<Scalar>::quiet_NaN();
  checkFailure(vss::math::linear::solveTriangular(
                   empty.view(), emptyRightHandSide.view(), nanTolerance),
               ErrorCode::InvalidTolerance);
  nanTolerance.rankTolerance.relative = std::numeric_limits<Scalar>::infinity();
  checkFailure(vss::math::linear::solveTriangular(
                   empty.view(), emptyRightHandSide.view(), nanTolerance),
               ErrorCode::InvalidTolerance);
  nanTolerance.rankTolerance.relative = -1.0;
  checkFailure(vss::math::linear::solveTriangular(
                   empty.view(), emptyRightHandSide.view(), nanTolerance),
               ErrorCode::InvalidTolerance);

  Matrix finiteMaximum(1U, 1U);
  finiteMaximum.at(0U, 0U) = std::numeric_limits<Scalar>::max();
  Vector maximumRightHandSide(std::vector<Scalar>{1.0});
  TriangularSolveOptions overflowingThreshold{Triangle::Lower};
  overflowingThreshold.rankTolerance.relative =
      std::numeric_limits<Scalar>::max();
  checkFailure(vss::math::linear::solveTriangular(finiteMaximum.view(),
                                                  maximumRightHandSide.view(),
                                                  overflowingThreshold),
               ErrorCode::NonFiniteResult);
}

void thresholdEqualityIsSingular() {
  Matrix factor(2U, 2U);
  factor.at(0U, 0U) = 1.0;
  factor.at(0U, 1U) = 0.0;
  factor.at(1U, 0U) = 0.0;
  factor.at(1U, 1U) = 1.0e-12;
  Vector rightHandSide(std::vector<Scalar>{1.0, 1.0});

  TriangularSolveOptions options{Triangle::Lower};
  options.rankTolerance.absolute = 0.0;
  options.rankTolerance.relative = 1.0e-12;
  checkFailure(vss::math::linear::solveTriangular(
                   factor.view(), rightHandSide.view(), options),
               ErrorCode::Singular);
}

} // namespace

int main() {
  lowerVectorSolve();
  upperVectorSolve();
  transposedLowerSolve();
  multipleRightHandSides();
  failureClassification();
  thresholdEqualityIsSingular();
}

// VSS-MATH-TRACEABILITY: TRI-01 TRI-02 TRI-03 TRI-04 TRI-05
// VSS-MATH-TRACEABILITY: SUP-TRI-01
