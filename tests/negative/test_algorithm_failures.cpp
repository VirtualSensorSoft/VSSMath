#include "Test.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/linear/TriangularSolve.hpp>
#include <vss/math/probability/ChiSquare.hpp>
#include <vss/math/statistics/Descriptive.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;

template <class Result>
void expectFailure(const Result& result, ErrorCode code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == code);
}

Matrix twoByTwo() {
  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 2.0;
  matrix.at(0U, 1U) = 0.0;
  matrix.at(1U, 0U) = 0.0;
  matrix.at(1U, 1U) = 3.0;
  return matrix;
}

void emptyAndShapeFailures() {
  Matrix emptyMatrix;
  Vector emptyVector;
  const vss::math::linear::TriangularSolveOptions triangularOptions{
      vss::math::linear::Triangle::Lower};
  expectFailure(vss::math::linear::solveTriangular(
                    emptyMatrix.view(), emptyVector.view(), triangularOptions),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::linear::factorizeLU(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::linear::factorizeQR(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::linear::factorizeCholesky(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::linear::decomposeSymmetricEigen(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::covariance::analyzeCovariance(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::covariance::toCorrelation(emptyMatrix.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::covariance::fromCorrelation(emptyMatrix.view(),
                                                       emptyVector.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::statistics::mean(emptyVector.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::statistics::populationVariance(emptyVector.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::statistics::sampleVariance(emptyVector.view()),
                ErrorCode::InvalidExtent);
  expectFailure(vss::math::probability::chiSquareSurvival(-1.0, 1U),
                ErrorCode::InvalidDomain);

  Matrix one(1U, 1U, 1.0);
  Matrix two(2U, 2U, 1.0);
  Vector wrongSize(2U, 1.0);
  expectFailure(vss::math::multiply(one.view(), two.view()),
                ErrorCode::DimensionMismatch);
  expectFailure(vss::math::multiply(one.view(), wrongSize.view()),
                ErrorCode::DimensionMismatch);
  expectFailure(vss::math::linear::solveTriangular(
                    two.view(), Vector(1U, 1.0).view(), triangularOptions),
                ErrorCode::DimensionMismatch);
  expectFailure(
      vss::math::covariance::propagateCovariance(one.view(), two.view()),
      ErrorCode::DimensionMismatch);
}

void nonFiniteAndDomainFailures() {
  Matrix nonFinite = twoByTwo();
  nonFinite.at(0U, 1U) = std::numeric_limits<Scalar>::quiet_NaN();
  Vector finite(2U, 1.0);
  expectFailure(vss::math::linear::factorizeLU(nonFinite.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::linear::factorizeQR(nonFinite.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::linear::factorizeCholesky(nonFinite.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::linear::decomposeSymmetricEigen(nonFinite.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::covariance::analyzeCovariance(nonFinite.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::multiply(nonFinite.view(), finite.view()),
                ErrorCode::NonFiniteInput);
  Vector nonFiniteVector(
      std::vector<Scalar>{1.0, std::numeric_limits<Scalar>::infinity()});
  expectFailure(vss::math::statistics::mean(nonFiniteVector.view()),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::probability::chiSquareSurvival(
                    std::numeric_limits<Scalar>::quiet_NaN(), 2U),
                ErrorCode::NonFiniteInput);
  expectFailure(vss::math::probability::chiSquareSurvival(1.0, 0U),
                ErrorCode::InvalidDomain);

  std::vector<Scalar> storage(1U, 0.0);
  const auto vectorOverflow = vss::math::VectorView::create(
      std::span<Scalar>(storage), 3U, std::numeric_limits<std::size_t>::max());
  expectFailure(vectorOverflow, ErrorCode::SizeOverflow);
  const auto matrixOverflow = vss::math::MatrixView::create(
      std::span<Scalar>(storage), 2U, 2U,
      std::numeric_limits<std::size_t>::max(), 1U);
  expectFailure(matrixOverflow, ErrorCode::SizeOverflow);
}

void differentiationFailures() {
  Vector parameters(2U, 1.0);
  std::vector<vss::math::differentiation::ParameterStep> steps;
  steps.emplace_back(1.0);
  const auto function = [](vss::math::ConstVectorView) {
    return vss::math::Result<Scalar>::success(1.0);
  };
  const auto callbackFailure = [](vss::math::ConstVectorView) {
    return vss::math::Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                              "callback failure");
  };
  expectFailure(vss::math::differentiation::numericalGradient(
                    function, parameters.view(), steps),
                ErrorCode::DimensionMismatch);
  steps.emplace_back(1.0);
  expectFailure(vss::math::differentiation::numericalGradient(
                    callbackFailure, parameters.view(), steps),
                ErrorCode::CallbackFailure);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): VSS_CHECK exits on failure.
  emptyAndShapeFailures();
  nonFiniteAndDomainFailures();
  differentiationFailures();
}
