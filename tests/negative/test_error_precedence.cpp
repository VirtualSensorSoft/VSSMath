#include "Test.hpp"

#include <limits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/TriangularSolve.hpp>
#include <vss/math/probability/ChiSquare.hpp>

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

void invalidTolerancePrecedesEmpty() {
  Matrix emptyMatrix;
  Vector emptyVector;
  vss::math::linear::TriangularSolveOptions options{
      vss::math::linear::Triangle::Lower};
  options.rankTolerance.relative = std::numeric_limits<Scalar>::quiet_NaN();
  expectFailure(vss::math::linear::solveTriangular(emptyMatrix.view(),
                                                   emptyVector.view(), options),
                ErrorCode::InvalidTolerance);
}

void shapePrecedesNonFinite() {
  Matrix matrix(1U, 1U, std::numeric_limits<Scalar>::quiet_NaN());
  Vector wrongSize(2U, 1.0);
  expectFailure(vss::math::multiply(matrix.view(), wrongSize.view()),
                ErrorCode::DimensionMismatch);
}

void nonFinitePrecedesNonsymmetry() {
  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = std::numeric_limits<Scalar>::quiet_NaN();
  matrix.at(1U, 0U) = 2.0;
  matrix.at(1U, 1U) = 1.0;
  expectFailure(vss::math::covariance::analyzeCovariance(matrix.view()),
                ErrorCode::NonFiniteInput);
}

void callbackConfigurationPrecedesCallbackFailure() {
  Vector parameters(1U, 1.0);
  std::vector<vss::math::differentiation::ParameterStep> steps;
  const vss::math::differentiation::ParameterStep invalid(0.0);
  steps.push_back(invalid);
  const auto callback = [](vss::math::ConstVectorView) {
    return vss::math::Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                              "callback failure");
  };
  expectFailure(vss::math::differentiation::numericalGradient(
                    callback, parameters.view(), steps),
                ErrorCode::InvalidDomain);
}

void probabilityInputPrecedence() {
  expectFailure(vss::math::probability::chiSquareSurvival(
                    std::numeric_limits<Scalar>::quiet_NaN(), 0U),
                ErrorCode::NonFiniteInput);
}

void singularPrecedesPolicy() {
  Matrix singular(2U, 2U);
  singular.at(0U, 0U) = 1.0;
  singular.at(0U, 1U) = 1.0;
  singular.at(1U, 0U) = 2.0;
  singular.at(1U, 1U) = 2.0;
  const auto result = vss::math::linear::factorizeLU(singular.view());
  VSS_CHECK(result.hasValue());
  Vector rightHandSide(2U, 1.0);
  vss::math::linear::SolveOptions options;
  options.illConditionedPolicy =
      vss::math::linear::IllConditionedSolvePolicy::Reject;
  const auto solve = result.value().solve(rightHandSide.view(), options);
  expectFailure(solve, ErrorCode::Singular);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): VSS_CHECK exits on failure.
  invalidTolerancePrecedesEmpty();
  shapePrecedesNonFinite();
  nonFinitePrecedesNonsymmetry();
  callbackConfigurationPrecedesCallbackFailure();
  probabilityInputPrecedence();
  singularPrecedesPolicy();
}
