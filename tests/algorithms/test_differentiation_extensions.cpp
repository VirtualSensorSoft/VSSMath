#include "Test.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/differentiation/FiniteDifference.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vss/math/Array.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/differentiation/ForwardAD.hpp>
#include <vss/math/differentiation/Hessian.hpp>

namespace {

using namespace vss::math;
using namespace vss::math::differentiation;

void close(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <=
            (1.0e-12 + (1.0e-10 * std::abs(expected))));
}

void hessianQuadratic() {
  const Vector x(std::array<Scalar, 2>{1.0, 2.0});
  const std::array<ParameterStep, 2> steps{ParameterStep(1.0),
                                           ParameterStep(1.0)};
  auto configured = steps;
  configured[0].explicitStep = 0.01;
  configured[1].explicitStep = 0.01;
  const HessianPolicy policy{.steps = configured,
                             .check_tolerance = 1.0e-8,
                             .halvings = 3U,
                             .one_sided = false};
  const ScalarFunction f = [](ConstVectorView v) {
    const Scalar a = v.at(0U);
    const Scalar b = v.at(1U);
    return Result<Scalar>::success((a * a) + (3.0 * a * b) + (2.0 * b * b));
  };
  const auto result = numericalHessian(
      f, x.view(), policy,
      Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 1000U});
  VSS_CHECK(result.hasValue());
  if (!result) {
    return;
  }
  close(result.value().at(0U, 0U), 2.0);
  close(result.value().at(0U, 1U), 3.0);
  close(result.value().at(1U, 0U), 3.0);
  close(result.value().at(1U, 1U), 4.0);
}

void outputHessianAndResourceLimit() {
  const Vector x(std::array<Scalar, 2>{1.0, 2.0});
  const std::array<ParameterStep, 2> steps{ParameterStep(1.0),
                                           ParameterStep(1.0)};
  const HessianPolicy policy{.steps = steps,
                             .check_tolerance = 1.0e-7,
                             .halvings = 3U,
                             .one_sided = false};
  const VectorFunction f = [](ConstVectorView v) {
    const Scalar a = v.at(0U);
    const Scalar b = v.at(1U);
    const std::array<Scalar, 2> values{(3.0 * a * a) + (2.0 * a * b) + (b * b),
                                       7.0};
    return Result<Vector>::success(Vector(values));
  };
  const auto result = numericalOutputHessians(
      f, x.view(), policy,
      Limits{.elements = 10000U, .bytes = 1000000U, .evaluations = 10000U});
  VSS_CHECK(result.hasValue());
  if (result) {
    VSS_CHECK(result.value().shape().rank() == 3U);
    VSS_CHECK(result.value().shape().extent(0U) == 2U);
    VSS_CHECK(result.value().shape().extent(1U) == 2U);
    VSS_CHECK(result.value().shape().extent(2U) == 2U);
    close(result.value().at(std::array<Index, 3>{0U, 0U, 0U}), 6.0);
    close(result.value().at(std::array<Index, 3>{0U, 0U, 1U}), 2.0);
    close(result.value().at(std::array<Index, 3>{1U, 0U, 0U}), 0.0);
    close(result.value().at(std::array<Index, 3>{1U, 0U, 1U}), 0.0);
    close(result.value().at(std::array<Index, 3>{1U, 1U, 0U}), 0.0);
    close(result.value().at(std::array<Index, 3>{1U, 1U, 1U}), 0.0);
  }
  const auto limited = numericalHessian(
      [](ConstVectorView v) {
        return Result<Scalar>::success(v.at(0U) * v.at(0U));
      },
      x.view(), policy, Limits{.elements = 1U, .bytes = 1U, .evaluations = 1U});
  VSS_CHECK(!limited.hasValue());
  if (!limited) {
    VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
  }
}

void hessianVectorProduct() {
  const Vector x(std::array<Scalar, 2>{1.0, 2.0});
  const std::array<ParameterStep, 2> steps{ParameterStep(1.0),
                                           ParameterStep(1.0)};
  const HessianPolicy policy{.steps = steps,
                             .check_tolerance = 1.0e-7,
                             .halvings = 3U,
                             .one_sided = false};
  const auto result = hessianVectorProduct(
      [](ConstVectorView v) {
        const Scalar a = v.at(0U);
        const Scalar b = v.at(1U);
        return Result<Scalar>::success((3.0 * a * a) + (2.0 * a * b) + (b * b));
      },
      x.view(), Vector(std::array<Scalar, 2>{1.0, -1.0}).view(), policy,
      Limits{.elements = 10000U, .bytes = 1000000U, .evaluations = 10000U});
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().at(0U), 4.0);
    close(result.value().at(1U), 0.0);
  }
  std::size_t calls = 0U;
  const Vector zero(2U);
  const auto zeroProduct = hessianVectorProduct(
      [&calls](ConstVectorView v) {
        ++calls;
        return Result<Scalar>::success(v.at(0U));
      },
      x.view(), zero.view(), policy,
      Limits{.elements = 10U, .bytes = 1000U, .evaluations = 10U});
  VSS_CHECK(zeroProduct.hasValue());
  VSS_CHECK(calls == 0U);

  const auto badDirection = hessianVectorProduct(
      [](ConstVectorView v) { return Result<Scalar>::success(v.at(0U)); },
      x.view(), Vector(std::array<Scalar, 1>{1.0}).view(), policy,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 100U});
  VSS_CHECK(!badDirection.hasValue());
  if (!badDirection) {
    VSS_CHECK(badDirection.status().code() == ErrorCode::DimensionMismatch);
  }
  const auto limitedProduct = hessianVectorProduct(
      [](ConstVectorView v) {
        return Result<Scalar>::success(v.at(0U) * v.at(0U));
      },
      x.view(), Vector(std::array<Scalar, 2>{1.0, 0.0}).view(), policy,
      Limits{.elements = 12U, .bytes = 10000U, .evaluations = 100U});
  VSS_CHECK(!limitedProduct.hasValue());
  if (!limitedProduct) {
    VSS_CHECK(limitedProduct.status().code() == ErrorCode::ResourceLimit);
  }
}

void authorizedOneSidedHessianAndBoundedProduct() {
  const Vector x(std::array<Scalar, 2>{0.0, 0.0});
  auto steps = std::array{ParameterStep(1.0), ParameterStep(1.0)};
  steps[0].lowerBound = 0.0;
  steps[1].upperBound = 0.0;
  const HessianPolicy policy{.steps = steps,
                             .check_tolerance = 1.0e-7,
                             .halvings = 3U,
                             .one_sided = true};
  std::size_t calls = 0U;
  const ScalarFunction f = [&calls](ConstVectorView v) {
    ++calls;
    VSS_CHECK(v.at(0U) >= 0.0);
    VSS_CHECK(v.at(1U) <= 0.0);
    const Scalar a = v.at(0U);
    const Scalar b = v.at(1U);
    return Result<Scalar>::success((3.0 * a * a) + (2.0 * a * b) + (b * b));
  };
  const auto result = numericalHessian(
      f, x.view(), policy,
      Limits{.elements = 10000U, .bytes = 1000000U, .evaluations = 10000U});
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().at(0U, 0U), 6.0);
    close(result.value().at(0U, 1U), 2.0);
    close(result.value().at(1U, 0U), 2.0);
    close(result.value().at(1U, 1U), 2.0);
  }
  const auto product = hessianVectorProduct(
      f, x.view(), Vector(std::array<Scalar, 2>{1.0, -1.0}).view(), policy,
      Limits{.elements = 10000U, .bytes = 1000000U, .evaluations = 10000U});
  VSS_CHECK(product.hasValue());
  if (product) {
    close(product.value().at(0U), 4.0);
    close(product.value().at(1U), 0.0);
  }
  VSS_CHECK(calls > 0U);
}

void hessianRefinementAndShapeFailures() {
  const Vector x(std::array<Scalar, 1>{0.0});
  const std::array<ParameterStep, 1> steps{ParameterStep(1.0)};
  const HessianPolicy policy{.steps = steps,
                             .check_tolerance = 1.0e-7,
                             .halvings = 2U,
                             .one_sided = false};
  const auto discontinuous = numericalHessian(
      [](ConstVectorView v) {
        return Result<Scalar>::success(v.at(0U) > 0.0 ? 1.0 : 0.0);
      },
      x.view(), policy,
      Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 1000U});
  VSS_CHECK(!discontinuous.hasValue());
  if (!discontinuous) {
    VSS_CHECK(discontinuous.status().code() == ErrorCode::NoConvergence);
  }

  std::size_t calls = 0U;
  const VectorFunction changingShape = [&calls](ConstVectorView v) {
    ++calls;
    if (v.at(0U) > 0.0) {
      return Result<Vector>::success(Vector{});
    }
    return Result<Vector>::success(
        Vector(std::array<Scalar, 1>{v.at(0U) * v.at(0U)}));
  };
  const auto shape = numericalOutputHessians(
      changingShape, x.view(), policy,
      Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 1000U});
  VSS_CHECK(!shape.hasValue());
  if (!shape) {
    VSS_CHECK(shape.status().code() == ErrorCode::DimensionMismatch);
  }
  VSS_CHECK(calls > 1U);

  const Vector boundary(std::array<Scalar, 1>{0.0});
  auto boundedStep = ParameterStep(1.0);
  boundedStep.lowerBound = 0.0;
  const std::array<ParameterStep, 1> boundedSteps{boundedStep};
  const HessianPolicy centralOnly{.steps = boundedSteps,
                                  .check_tolerance = 1.0e-7,
                                  .halvings = 2U,
                                  .one_sided = false};
  const auto unauthorized = numericalHessian(
      [](ConstVectorView v) {
        return Result<Scalar>::success(v.at(0U) * v.at(0U));
      },
      boundary.view(), centralOnly,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 100U});
  VSS_CHECK(!unauthorized.hasValue());
  if (!unauthorized) {
    VSS_CHECK(unauthorized.status().code() == ErrorCode::NoConvergence);
  }

  std::size_t budgetCalls = 0U;
  const auto budgeted = numericalHessian(
      [&budgetCalls](ConstVectorView v) {
        ++budgetCalls;
        return Result<Scalar>::success(v.at(0U) * v.at(0U));
      },
      x.view(), policy,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(!budgeted.hasValue());
  if (!budgeted) {
    VSS_CHECK(budgeted.status().code() == ErrorCode::ResourceLimit);
  }
  VSS_CHECK(budgetCalls == 1U);
}

void scaleConditionedHessian() {
  for (const int exponent : std::array<int, 2>{-300, 300}) {
    const Scalar scale = std::ldexp(1.0, exponent);
    const Vector x(std::array<Scalar, 1>{scale / 4.0});
    ParameterStep step(scale);
    step.explicitStep = scale / 64.0;
    const std::array<ParameterStep, 1> steps{step};
    const HessianPolicy policy{.steps = steps,
                               .check_tolerance = 1.0e-8,
                               .halvings = 3U,
                               .one_sided = false};
    const auto result = numericalHessian(
        [scale](ConstVectorView v) {
          const Scalar normalized = v.at(0U) / scale;
          return Result<Scalar>::success(scale * normalized * normalized);
        },
        x.view(), policy,
        Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 1000U});
    VSS_CHECK(result.hasValue());
    if (result) {
      close(result.value().at(0U, 0U) * scale, 2.0);
    }
  }
}

void forwardAutomaticDifferentiation() {
  const DualFunction function = [](std::span<const Dual> x) {
    const Scalar value = (x[0U].primal * x[1U].primal) + std::sin(x[0U].primal);
    Vector tangent(2U);
    tangent.at(0U) = x[1U].primal + std::cos(x[0U].primal);
    tangent.at(1U) = x[0U].primal;
    const std::array<Dual, 1> output{
        Dual{.primal = value, .tangent = std::move(tangent)}};
    return Array<Dual>::copy(
        output, Limits{.elements = 10U, .bytes = 1000U, .evaluations = 10U});
  };
  const Vector x(std::array<Scalar, 2>{0.0, 2.0});
  const auto result = forwardAutomaticDerivative(
      function, x.view(), 2U,
      Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 10U});
  VSS_CHECK(result.hasValue());
  if (result) {
    close(result.value().value.at(0U), 0.0);
    close(result.value().jacobian.at(0U, 0U), 3.0);
    close(result.value().jacobian.at(0U, 1U), 0.0);
  }
}

void dualOperations() {
  const Dual x{.primal = 2.0,
               .tangent = Vector(std::array<Scalar, 2>{1.0, 0.0})};
  const Dual y{.primal = 3.0,
               .tangent = Vector(std::array<Scalar, 2>{0.0, 1.0})};
  const auto product =
      dualBinary(DualBinaryOp::multiply, x, y,
                 Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(product.hasValue());
  if (product) {
    close(product.value().primal, 6.0);
    close(product.value().tangent.at(0U), 3.0);
    close(product.value().tangent.at(1U), 2.0);
  }
  const auto sum =
      dualBinary(DualBinaryOp::add, x, y,
                 Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(sum.hasValue());
  if (sum) {
    close(sum.value().primal, 5.0);
    close(sum.value().tangent.at(0U), 1.0);
    close(sum.value().tangent.at(1U), 1.0);
  }
  const auto difference =
      dualBinary(DualBinaryOp::subtract, x, y,
                 Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(difference.hasValue());
  if (difference) {
    close(difference.value().primal, -1.0);
    close(difference.value().tangent.at(0U), 1.0);
    close(difference.value().tangent.at(1U), -1.0);
  }
  const auto sine =
      dualUnary(DualUnaryOp::sin, x,
                Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(sine.hasValue());
  if (sine) {
    close(sine.value().tangent.at(0U), std::cos(2.0));
  }
  const auto quotient =
      dualBinary(DualBinaryOp::divide, x, y,
                 Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(quotient.hasValue());
  if (quotient) {
    close(quotient.value().tangent.at(0U), 1.0 / 3.0);
  }
  const auto power =
      dualBinary(DualBinaryOp::power, x, y,
                 Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(power.hasValue());
  if (power) {
    close(power.value().primal, 8.0);
    close(power.value().tangent.at(0U), 12.0);
    close(power.value().tangent.at(1U), 8.0 * std::numbers::ln2);
  }
  const auto zeroAbs =
      dualUnary(DualUnaryOp::abs, Dual{.primal = 0.0, .tangent = Vector(2U)},
                Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(!zeroAbs.hasValue());
  if (!zeroAbs) {
    VSS_CHECK(zeroAbs.status().code() == ErrorCode::InvalidDomain);
  }
  const auto zeroDivisor = dualBinary(
      DualBinaryOp::divide, x, Dual{.primal = 0.0, .tangent = Vector(2U)},
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(!zeroDivisor.hasValue());
  if (!zeroDivisor) {
    VSS_CHECK(zeroDivisor.status().code() == ErrorCode::InvalidDomain);
  }
  const auto widthMismatch = dualBinary(
      DualBinaryOp::add, x, Dual{.primal = 3.0, .tangent = Vector(1U)},
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(!widthMismatch.hasValue());
  if (!widthMismatch) {
    VSS_CHECK(widthMismatch.status().code() == ErrorCode::DimensionMismatch);
  }
  const auto insufficient =
      dualUnary(DualUnaryOp::sin, x,
                Limits{.elements = 2U, .bytes = 8U, .evaluations = 1U});
  VSS_CHECK(!insufficient.hasValue());
  if (!insufficient) {
    VSS_CHECK(insufficient.status().code() == ErrorCode::ResourceLimit);
  }

  const std::array<std::pair<DualUnaryOp, std::pair<Scalar, Scalar>>, 10> cases{
      {
          {DualUnaryOp::negate, {-2.0, -1.0}},
          {DualUnaryOp::exp, {std::exp(0.5), std::exp(0.5)}},
          {DualUnaryOp::log, {std::numbers::ln2, 0.5}},
          {DualUnaryOp::log1p, {std::log1p(0.5), 2.0 / 3.0}},
          {DualUnaryOp::expm1, {std::expm1(0.5), std::exp(0.5)}},
          {DualUnaryOp::sqrt, {2.0, 0.25}},
          {DualUnaryOp::sin, {std::sin(0.5), std::cos(0.5)}},
          {DualUnaryOp::cos, {std::cos(0.5), -std::sin(0.5)}},
          {DualUnaryOp::tanh,
           {std::tanh(0.5), 1.0 - (std::tanh(0.5) * std::tanh(0.5))}},
          {DualUnaryOp::abs, {2.0, -1.0}},
      }};
  for (const auto& [operation, expected] : cases) {
    Scalar primal = 0.5;
    switch (operation) {
    case DualUnaryOp::negate:
      primal = 2.0;
      break;
    case DualUnaryOp::abs:
      primal = -2.0;
      break;
    case DualUnaryOp::log:
      primal = 2.0;
      break;
    case DualUnaryOp::sqrt:
      primal = 4.0;
      break;
    default:
      break;
    }
    const Dual input{.primal = primal,
                     .tangent = Vector(std::array<Scalar, 1>{1.0})};
    const auto evaluated =
        dualUnary(operation, input,
                  Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
    VSS_CHECK(evaluated.hasValue());
    if (evaluated) {
      close(evaluated.value().primal, expected.first);
      close(evaluated.value().tangent.at(0U), expected.second);
    }
  }
}

void forwardAdBlocksAndFailures() {
  const Vector x(std::array<Scalar, 2>{2.0, 3.0});
  const DualFunction function = [](std::span<const Dual> input) {
    const Scalar a = input[0U].primal;
    const Scalar b = input[1U].primal;
    Vector first(input[0U].tangent.size());
    for (Index i = 0U; i < first.size(); ++i) {
      first.at(i) =
          (input[0U].tangent.at(i) * b) + (a * input[1U].tangent.at(i));
    }
    Vector second(first.size());
    for (Index i = 0U; i < second.size(); ++i) {
      second.at(i) =
          input[0U].tangent.at(i) + (2.0 * b * input[1U].tangent.at(i));
    }
    const std::array<Dual, 2> output{
        Dual{.primal = a * b, .tangent = std::move(first)},
        Dual{.primal = a + (b * b), .tangent = std::move(second)}};
    return Array<Dual>::copy(
        output, Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  };
  for (const Index width : std::array<Index, 2>{1U, 2U}) {
    const auto result = forwardAutomaticDerivative(
        function, x.view(), width,
        Limits{.elements = 1000U, .bytes = 100000U, .evaluations = 2U});
    VSS_CHECK(result.hasValue());
    if (result) {
      close(result.value().value.at(0U), 6.0);
      close(result.value().value.at(1U), 11.0);
      close(result.value().jacobian.at(0U, 0U), 3.0);
      close(result.value().jacobian.at(0U, 1U), 2.0);
      close(result.value().jacobian.at(1U, 0U), 1.0);
      close(result.value().jacobian.at(1U, 1U), 6.0);
    }
  }
  const DualFunction empty = [](std::span<const Dual> input) {
    VSS_CHECK(input.empty());
    const std::array<Dual, 1> output{Dual{.primal = 7.0, .tangent = Vector{}}};
    return Array<Dual>::copy(
        output, Limits{.elements = 10U, .bytes = 1000U, .evaluations = 1U});
  };
  const auto emptyResult = forwardAutomaticDerivative(
      empty, Vector{}.view(), 1U,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 1U});
  VSS_CHECK(emptyResult.hasValue());
  if (emptyResult) {
    close(emptyResult.value().value.at(0U), 7.0);
    VSS_CHECK(emptyResult.value().jacobian.rows() == 1U);
    VSS_CHECK(emptyResult.value().jacobian.columns() == 0U);
  }
  const DualFunction noOutputs = [](std::span<const Dual>) {
    return Result<Array<Dual>>::success(Array<Dual>{});
  };
  const auto emptyOutput = forwardAutomaticDerivative(
      noOutputs, x.view(), 1U,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 2U});
  VSS_CHECK(emptyOutput.hasValue());
  if (emptyOutput) {
    VSS_CHECK(emptyOutput.value().value.empty());
    VSS_CHECK(emptyOutput.value().jacobian.rows() == 0U);
    VSS_CHECK(emptyOutput.value().jacobian.columns() == x.size());
  }
  const DualFunction throws = [](std::span<const Dual>) -> Result<Array<Dual>> {
    throw std::runtime_error("callback failure");
  };
  const auto callbackFailure = forwardAutomaticDerivative(
      throws, x.view(), 1U,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 2U});
  VSS_CHECK(!callbackFailure.hasValue());
  if (!callbackFailure) {
    VSS_CHECK(callbackFailure.status().code() == ErrorCode::CallbackFailure);
  }
  const DualFunction drifting = [](std::span<const Dual> input) {
    const std::array<Dual, 1> output{Dual{
        .primal = input[0U].primal +
                  (input[0U].tangent.empty() ? 0.0 : input[0U].tangent.at(0U)),
        .tangent = Vector(input[0U].tangent.size())}};
    return Array<Dual>::copy(
        output, Limits{.elements = 10U, .bytes = 1000U, .evaluations = 1U});
  };
  const auto changed = forwardAutomaticDerivative(
      drifting, x.view(), 1U,
      Limits{.elements = 100U, .bytes = 10000U, .evaluations = 2U});
  VSS_CHECK(!changed.hasValue());
  if (!changed) {
    VSS_CHECK(changed.status().code() == ErrorCode::InvalidDomain);
  }
}

} // namespace

int main() {
  hessianQuadratic();
  outputHessianAndResourceLimit();
  hessianVectorProduct();
  authorizedOneSidedHessianAndBoundedProduct();
  hessianRefinementAndShapeFailures();
  scaleConditionedHessian();
  forwardAutomaticDifferentiation();
  dualOperations();
  forwardAdBlocksAndFailures();
  return EXIT_SUCCESS;
}

// VSS-MATH-TRACEABILITY: CASE-C065.scalar CASE-C065.output CASE-C065.product
// VSS-MATH-TRACEABILITY: CASE-C065.boundary CASE-C065.refinement
// VSS-MATH-TRACEABILITY: CASE-C065.hessian-authorized-one-sided
// VSS-MATH-TRACEABILITY: CASE-C065.hessian.scale-conditioned-quadratic
// VSS-MATH-TRACEABILITY: CASE-C066.block.axes CASE-C066.binary
// VSS-MATH-TRACEABILITY: CASE-C066.unary CASE-C066.empty
// VSS-MATH-TRACEABILITY: CASE-C066.vector-output-jacobian
