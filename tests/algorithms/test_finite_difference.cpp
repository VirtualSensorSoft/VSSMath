#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>

namespace {

using vss::math::ConstVectorView;
using vss::math::ErrorCode;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::differentiation::DerivativeCheckResult;
using vss::math::differentiation::DifferenceFormula;
using vss::math::differentiation::GradientResult;
using vss::math::differentiation::JacobianResult;
using vss::math::differentiation::ParameterStep;
using vss::math::differentiation::ScalarFunction;
using vss::math::differentiation::VectorFunction;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void checkNear(Scalar actual, Scalar expected, Scalar absolute = 1.0e-12,
               Scalar relative = 1.0e-10) {
  const Scalar scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  VSS_CHECK(std::abs(actual - expected) <= absolute + (relative * scale));
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void publicContractAndDefaults() {
  static_assert(!std::is_default_constructible_v<GradientResult>);
  static_assert(std::is_copy_constructible_v<GradientResult>);
  static_assert(std::is_nothrow_move_constructible_v<GradientResult>);
  static_assert(std::is_nothrow_move_assignable_v<GradientResult>);
  static_assert(!std::is_default_constructible_v<JacobianResult>);
  static_assert(std::is_copy_constructible_v<JacobianResult>);
  static_assert(std::is_nothrow_move_constructible_v<JacobianResult>);
  static_assert(std::is_nothrow_move_assignable_v<JacobianResult>);
  static_assert(!std::is_default_constructible_v<DerivativeCheckResult>);
  static_assert(std::is_copy_constructible_v<DerivativeCheckResult>);
  static_assert(std::is_nothrow_move_constructible_v<DerivativeCheckResult>);
  static_assert(std::is_nothrow_move_assignable_v<DerivativeCheckResult>);

  using GradientSignature = vss::math::Result<GradientResult> (*)(
      const ScalarFunction&, ConstVectorView, std::span<const ParameterStep>);
  using JacobianSignature = vss::math::Result<JacobianResult> (*)(
      const VectorFunction&, ConstVectorView, std::span<const ParameterStep>);
  using GradientCheckSignature = vss::math::Result<DerivativeCheckResult> (*)(
      ConstVectorView, const GradientResult&, vss::math::Tolerance);
  using JacobianCheckSignature = vss::math::Result<DerivativeCheckResult> (*)(
      vss::math::ConstMatrixView, const JacobianResult&, vss::math::Tolerance);
  static_assert(
      std::is_same_v<decltype(static_cast<GradientSignature>(
                         &vss::math::differentiation::numericalGradient)),
                     GradientSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<JacobianSignature>(
                         &vss::math::differentiation::numericalJacobian)),
                     JacobianSignature>);
  static_assert(std::is_same_v<decltype(static_cast<GradientCheckSignature>(
                                   &vss::math::differentiation::checkGradient)),
                               GradientCheckSignature>);
  static_assert(std::is_same_v<decltype(static_cast<JacobianCheckSignature>(
                                   &vss::math::differentiation::checkJacobian)),
                               JacobianCheckSignature>);

  const ParameterStep defaults(2.0);
  VSS_CHECK(defaults.scale == 2.0);
  VSS_CHECK(!defaults.explicitStep.has_value());
  VSS_CHECK(!defaults.lowerBound.has_value());
  VSS_CHECK(!defaults.upperBound.has_value());
}

void jacobianOrientation() {
  Vector parameters = vector({2.0, 1.0});
  std::vector<Vector> callbackInputs;
  const VectorFunction function = [&](ConstVectorView input) {
    callbackInputs.emplace_back(
        std::vector<Scalar>{input.at(0U), input.at(1U)});
    return vss::math::Result<Vector>::success(
        vector({(input.at(0U) * input.at(0U)) + (3.0 * input.at(1U)),
                std::sin(input.at(0U))}));
  };
  const std::array<ParameterStep, 2> steps{ParameterStep(1.0),
                                           ParameterStep(1.0)};

  const auto result = vss::math::differentiation::numericalJacobian(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  VSS_CHECK(result.value().values().rows() == 2U);
  VSS_CHECK(result.value().values().columns() == 2U);
  checkNear(result.value().values().at(0U, 0U), 4.0, 1.0e-8, 1.0e-8);
  checkNear(result.value().values().at(0U, 1U), 3.0, 1.0e-8, 1.0e-8);
  checkNear(result.value().values().at(1U, 0U), std::cos(2.0), 1.0e-8, 1.0e-8);
  checkNear(result.value().values().at(1U, 1U), 0.0, 1.0e-8, 1.0e-8);
  VSS_CHECK(callbackInputs.size() == 5U);
  VSS_CHECK(callbackInputs.at(0U).at(0U) == 2.0);
  VSS_CHECK(callbackInputs.at(0U).at(1U) == 1.0);
  VSS_CHECK(callbackInputs.at(1U).at(0U) > 2.0);
  VSS_CHECK(callbackInputs.at(2U).at(0U) < 2.0);
  VSS_CHECK(callbackInputs.at(3U).at(1U) > 1.0);
  VSS_CHECK(callbackInputs.at(4U).at(1U) < 1.0);
  VSS_CHECK(parameters.at(0U) == 2.0);
  VSS_CHECK(parameters.at(1U) == 1.0);
  VSS_CHECK(result.value().formulas().size() == 2U);
  VSS_CHECK(result.value().formulas()[0U] == DifferenceFormula::Central);
  VSS_CHECK(result.value().formulas()[1U] == DifferenceFormula::Central);
}

void scalarCentralDifference() {
  const Vector parameters = vector({2.0});
  const ScalarFunction function = [](ConstVectorView input) {
    return vss::math::Result<Scalar>::success(std::sin(input.at(0U)));
  };
  const std::array<ParameterStep, 1> steps{ParameterStep(1.0)};
  const auto result = vss::math::differentiation::numericalGradient(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  checkNear(result.value().values().at(0U), std::cos(2.0), 1.0e-8, 1.0e-8);
  VSS_CHECK(result.value().formulas()[0U] == DifferenceFormula::Central);
}

void lowerBoundForwardFormula() {
  const Vector parameters = vector({0.0});
  const ScalarFunction function = [](ConstVectorView input) {
    return vss::math::Result<Scalar>::success(input.at(0U) * input.at(0U));
  };
  ParameterStep step(1.0);
  step.lowerBound = 0.0;
  const std::array<ParameterStep, 1> steps{step};
  const auto result = vss::math::differentiation::numericalGradient(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  checkNear(result.value().values().at(0U), 0.0, 1.0e-10, 1.0e-10);
  VSS_CHECK(result.value().formulas()[0U] == DifferenceFormula::Forward);
}

void upperBoundBackwardFormula() {
  const Vector parameters = vector({1.0});
  const ScalarFunction function = [](ConstVectorView input) {
    return vss::math::Result<Scalar>::success(input.at(0U) * input.at(0U));
  };
  ParameterStep step(1.0);
  step.upperBound = 1.0;
  const std::array<ParameterStep, 1> steps{step};
  const auto result = vss::math::differentiation::numericalGradient(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  checkNear(result.value().values().at(0U), 2.0, 1.0e-8, 1.0e-8);
  VSS_CHECK(result.value().formulas()[0U] == DifferenceFormula::Backward);
}

void explicitStep() {
  const Vector parameters = vector({2.0});
  const ScalarFunction function = [](ConstVectorView input) {
    const Scalar x = input.at(0U);
    return vss::math::Result<Scalar>::success(x * x * x);
  };
  ParameterStep step(1.0);
  step.explicitStep = 1.0e-4;
  const std::array<ParameterStep, 1> steps{step};
  const auto result = vss::math::differentiation::numericalGradient(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  checkNear(result.value().values().at(0U), 12.00000001, 1.0e-12, 1.0e-12);
  checkNear(result.value().effectiveSteps().at(0U), 1.0e-4, 1.0e-16, 1.0e-12);
}

void callbackFailure() {
  const Vector parameters = vector({0.0, 0.0});
  std::vector<Vector> callbackInputs;
  const VectorFunction function = [&](ConstVectorView input) {
    callbackInputs.emplace_back(
        std::vector<Scalar>{input.at(0U), input.at(1U)});
    if (input.at(1U) > 0.0) {
      return vss::math::Result<Vector>::failure(ErrorCode::InvalidDomain,
                                                "DIF-06 positive index 1");
    }
    return vss::math::Result<Vector>::success(vector(
        {(input.at(0U) + (2.0 * input.at(1U))), input.at(0U) - input.at(1U)}));
  };
  const std::array<ParameterStep, 2> steps{ParameterStep(1.0),
                                           ParameterStep(1.0)};
  const auto result = vss::math::differentiation::numericalJacobian(
      function, parameters.view(), steps);
  checkFailure(result, ErrorCode::CallbackFailure);
  VSS_CHECK(result.status().message().find("InvalidDomain") !=
            std::string_view::npos);
  VSS_CHECK(result.status().message().find("parameter 1") !=
            std::string_view::npos);
  VSS_CHECK(result.status().message().find("positive") !=
            std::string_view::npos);
  VSS_CHECK(callbackInputs.size() == 4U);
  VSS_CHECK(callbackInputs.at(0U).at(0U) == 0.0);
  VSS_CHECK(callbackInputs.at(0U).at(1U) == 0.0);
  VSS_CHECK(callbackInputs.at(1U).at(0U) > 0.0);
  VSS_CHECK(callbackInputs.at(2U).at(0U) < 0.0);
  VSS_CHECK(callbackInputs.at(3U).at(1U) > 0.0);
}

void callbackContractViolations() {
  const Vector parameters = vector({0.0});
  ParameterStep step(1.0);
  step.explicitStep = 1.0e-4;
  const std::array<ParameterStep, 1> steps{step};

  std::vector<Vector> changedExtentInputs;
  const VectorFunction changedExtent = [&](ConstVectorView input) {
    changedExtentInputs.emplace_back(std::vector<Scalar>{input.at(0U)});
    if (input.at(0U) > 0.0) {
      return vss::math::Result<Vector>::success(vector({0.0}));
    }
    return vss::math::Result<Vector>::success(vector({0.0, 1.0}));
  };
  const auto extentResult = vss::math::differentiation::numericalJacobian(
      changedExtent, parameters.view(), steps);
  checkFailure(extentResult, ErrorCode::DimensionMismatch);
  VSS_CHECK(changedExtentInputs.size() == 2U);

  std::vector<Vector> nonFiniteInputs;
  const VectorFunction nonFinite = [&](ConstVectorView input) {
    nonFiniteInputs.emplace_back(std::vector<Scalar>{input.at(0U)});
    if (input.at(0U) > 0.0) {
      return vss::math::Result<Vector>::success(
          vector({std::numeric_limits<Scalar>::quiet_NaN(), 1.0}));
    }
    return vss::math::Result<Vector>::success(vector({0.0, 1.0}));
  };
  const auto nonFiniteResult = vss::math::differentiation::numericalJacobian(
      nonFinite, parameters.view(), steps);
  checkFailure(nonFiniteResult, ErrorCode::NonFiniteResult);
  VSS_CHECK(nonFiniteInputs.size() == 2U);
}

void adjacentValueStepIsEnlarged() {
  const Scalar denormMin = std::numeric_limits<Scalar>::denorm_min();
  const Vector parameters = vector({denormMin});
  const ScalarFunction function = [](ConstVectorView input) {
    return vss::math::Result<Scalar>::success(input.at(0U));
  };
  const std::array<ParameterStep, 1> steps{ParameterStep(denormMin)};
  const auto result = vss::math::differentiation::numericalGradient(
      function, parameters.view(), steps);
  VSS_CHECK(result.hasValue());
  if (!result.hasValue()) {
    return;
  }
  checkNear(result.value().values().at(0U), 1.0, 1.0e-12, 1.0e-12);
  VSS_CHECK(result.value().effectiveSteps().at(0U) > 0.0);
  VSS_CHECK(parameters.at(0U) + result.value().effectiveSteps().at(0U) !=
            parameters.at(0U));
}

void invalidDifferentiationOptions() {
  const Vector parameters = vector({0.0});
  std::size_t callbackCount = 0U;
  const ScalarFunction function = [&](ConstVectorView) {
    ++callbackCount;
    return vss::math::Result<Scalar>::success(0.0);
  };

  const auto expectInvalid = [&](ParameterStep step, const Vector& input) {
    const std::array<ParameterStep, 1> steps{step};
    const auto result = vss::math::differentiation::numericalGradient(
        function, input.view(), steps);
    checkFailure(result, ErrorCode::InvalidDomain);
    VSS_CHECK(callbackCount == 0U);
  };

  const ParameterStep nanScale(std::numeric_limits<Scalar>::quiet_NaN());
  expectInvalid(nanScale, parameters);
  const ParameterStep infiniteScale(std::numeric_limits<Scalar>::infinity());
  expectInvalid(infiniteScale, parameters);
  const ParameterStep zeroScale(0.0);
  expectInvalid(zeroScale, parameters);
  ParameterStep invalidStep(1.0);
  invalidStep.explicitStep = 0.0;
  expectInvalid(invalidStep, parameters);
  invalidStep.explicitStep = std::numeric_limits<Scalar>::infinity();
  expectInvalid(invalidStep, parameters);
  invalidStep.explicitStep = std::numeric_limits<Scalar>::quiet_NaN();
  expectInvalid(invalidStep, parameters);
  ParameterStep invalidBounds(1.0);
  invalidBounds.lowerBound = std::numeric_limits<Scalar>::quiet_NaN();
  expectInvalid(invalidBounds, parameters);
  invalidBounds.lowerBound = 2.0;
  invalidBounds.upperBound = 1.0;
  expectInvalid(invalidBounds, parameters);
  invalidBounds.lowerBound = 1.0;
  invalidBounds.upperBound = 2.0;
  expectInvalid(invalidBounds, parameters);
  const Vector outside = vector({3.0});
  expectInvalid(invalidBounds, outside);

  const Vector analyticGradient = vector({1.0});
  const std::array<ParameterStep, 1> validSteps{ParameterStep(1.0)};
  const auto numerical = vss::math::differentiation::numericalGradient(
      function, parameters.view(), validSteps);
  VSS_CHECK(numerical.hasValue());
  if (!numerical.hasValue()) {
    return;
  }
  const auto invalidGradientTolerance =
      vss::math::differentiation::checkGradient(
          analyticGradient.view(), numerical.value(),
          {.absolute = std::numeric_limits<Scalar>::quiet_NaN(),
           .relative = 0.0});
  checkFailure(invalidGradientTolerance, ErrorCode::InvalidTolerance);

  const Matrix analyticJacobian(1U, 1U, 1.0);
  const auto numericalJacobian = vss::math::differentiation::numericalJacobian(
      [](ConstVectorView input) {
        return vss::math::Result<Vector>::success(vector({input.at(0U)}));
      },
      parameters.view(), validSteps);
  VSS_CHECK(numericalJacobian.hasValue());
  if (!numericalJacobian.hasValue()) {
    return;
  }
  const auto invalidJacobianTolerance =
      vss::math::differentiation::checkJacobian(
          analyticJacobian.view(), numericalJacobian.value(),
          {.absolute = 0.0, .relative = -1.0});
  checkFailure(invalidJacobianTolerance, ErrorCode::InvalidTolerance);
}

void arbitraryCallbackExceptionsPropagate() {
  const Vector parameters = vector({0.0});
  const ScalarFunction function =
      [](ConstVectorView) -> vss::math::Result<Scalar> {
    throw std::runtime_error("DIF exception");
  };
  const std::array<ParameterStep, 1> steps{ParameterStep(1.0)};
  VSS_CHECK_THROWS_AS(vss::math::differentiation::numericalGradient(
                          function, parameters.view(), steps),
                      std::runtime_error);
}

} // namespace

int main() {
  publicContractAndDefaults();
  jacobianOrientation();
  scalarCentralDifference();
  lowerBoundForwardFormula();
  upperBoundBackwardFormula();
  explicitStep();
  callbackFailure();
  callbackContractViolations();
  adjacentValueStepIsEnlarged();
  invalidDifferentiationOptions();
  arbitraryCallbackExceptionsPropagate();
  return EXIT_SUCCESS;
}

// VSS-MATH-TRACEABILITY: DIF-01 DIF-02 DIF-03 DIF-04 DIF-05 DIF-06 DIF-07
// VSS-MATH-TRACEABILITY: SUP-DIF-01
