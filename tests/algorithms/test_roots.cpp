#include "Test.hpp"

#include <cmath>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Roots.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

IterPolicy policy() {
  return {.abs = 1.0e-12,
          .rel = 1.0e-12,
          .max_iterations = 128U,
          .max_evaluations = 512U,
          .bracket = std::nullopt};
}

Vector vec(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void scalarBracketing() {
  const UnaryFunction squareRoot = [](Scalar x) {
    return Result<Scalar>::success(x * x - 2.0);
  };
  const auto bisected = rootBisection(squareRoot, {1.0, 2.0}, policy());
  VSS_CHECK(bisected && bisected.value().termination == Termination::converged);
  VSS_CHECK(std::abs(bisected.value().value - std::sqrt(2.0)) <= 3.0e-10);
  VSS_CHECK(bisected.value().enclosure.has_value());
  VSS_CHECK(bisected.value().function_evaluations > 2U);

  const auto brent = rootBrent(squareRoot, {1.0, 2.0}, policy());
  VSS_CHECK(brent && brent.value().termination == Termination::converged);
  VSS_CHECK(std::abs(brent.value().value - std::sqrt(2.0)) <= 3.0e-10);
  const auto exactEndpoint =
      rootBisection([](Scalar x) { return Result<Scalar>::success(x); },
                    {0.0, 2.0}, policy());
  VSS_CHECK(exactEndpoint && exactEndpoint.value().value == 0.0);
  VSS_CHECK(exactEndpoint.value().function_evaluations == 1U);
  const auto endpointEnclosure = exactEndpoint.value().enclosure;
  VSS_CHECK(endpointEnclosure.has_value());
  VSS_CHECK(endpointEnclosure.value_or(Interval{}).lower == 0.0);
}

void newtonAndCaps() {
  const UnaryFunction f = [](Scalar x) {
    return Result<Scalar>::success(x * x - 2.0);
  };
  const UnaryFunction df = [](Scalar x) {
    return Result<Scalar>::success(2.0 * x);
  };
  const auto newton = rootNewton(f, df, 1.0, policy());
  VSS_CHECK(newton && newton.value().termination == Termination::converged);
  VSS_CHECK(std::abs(newton.value().value - std::sqrt(2.0)) <= 3.0e-10);

  IterPolicy capped = policy();
  capped.max_iterations = 0U;
  const auto cappedResult = rootBisection(f, {1.0, 2.0}, capped);
  VSS_CHECK(cappedResult &&
            cappedResult.value().termination == Termination::cap);
  VSS_CHECK(cappedResult.value().enclosure.has_value());

  IterPolicy noEvaluations = policy();
  noEvaluations.max_evaluations = 0U;
  const auto noCalls = rootBisection(f, {1.0, 2.0}, noEvaluations);
  VSS_CHECK(!noCalls && noCalls.status().code() == ErrorCode::ResourceLimit);

  const auto sameSign = rootBisection(
      [](Scalar x) { return Result<Scalar>::success(x * x + 1.0); },
      {-1.0, 1.0}, policy());
  VSS_CHECK(!sameSign && sameSign.status().code() == ErrorCode::InvalidDomain);
  const auto zeroSlope = rootNewton(
      [](Scalar x) { return Result<Scalar>::success(x + 1.0); },
      [](Scalar) { return Result<Scalar>::success(0.0); }, 0.0, policy());
  VSS_CHECK(!zeroSlope &&
            zeroSlope.status().code() == ErrorCode::RankDeficient);
  const auto escaped = rootNewton(
      [](Scalar) -> Result<Scalar> { throw std::runtime_error("failure"); }, df,
      1.0, policy());
  VSS_CHECK(!escaped && escaped.status().code() == ErrorCode::CallbackFailure);
}

void vectorRootContracts() {
  Index functionCalls = 0U;
  Index jacobianCallsTotal = 0U;
  const VectorFunction function = [&functionCalls](ConstVectorView x) {
    ++functionCalls;
    Vector value(2U);
    value.at(0U) = x.at(0U) * x.at(0U) - 2.0;
    value.at(1U) = x.at(1U) - 3.0;
    return Result<Vector>::success(std::move(value));
  };
  const JacobianFunction jacobian = [&jacobianCallsTotal](ConstVectorView x) {
    ++jacobianCallsTotal;
    Matrix value(2U, 2U);
    value.at(0U, 0U) = 2.0 * x.at(0U);
    value.at(0U, 1U) = 0.0;
    value.at(1U, 0U) = 0.0;
    value.at(1U, 1U) = 1.0;
    return Result<Matrix>::success(std::move(value));
  };
  const auto root = findVectorRoot(function, jacobian, vec({1.0, 0.0}),
                                   policy(), {100U, 10000U, 100U});
  VSS_CHECK(root && root.value().termination == Termination::converged);
  VSS_CHECK(std::abs(root.value().value.at(0U) - std::sqrt(2.0)) < 1.0e-11);
  VSS_CHECK(std::abs(root.value().value.at(1U) - 3.0) < 1.0e-11);
  VSS_CHECK(root.value().function_evaluations == functionCalls);
  VSS_CHECK(root.value().jacobian_evaluations > 0U);
  VSS_CHECK(root.value().jacobian_evaluations == jacobianCallsTotal);
  VSS_CHECK(root.value().iterations == root.value().jacobian_evaluations);

  const VectorFunction steep = [](ConstVectorView x) {
    return Result<Vector>::success(vec({std::exp(x.at(0U)) - 100.0}));
  };
  const JacobianFunction steepJacobian = [](ConstVectorView x) {
    return Result<Matrix>::success(Matrix(1U, 1U, std::exp(x.at(0U))));
  };
  const auto backtracked = findVectorRoot(steep, steepJacobian, vec({0.0}),
                                          policy(), {100U, 10000U, 100U});
  VSS_CHECK(backtracked &&
            backtracked.value().termination == Termination::converged);
  VSS_CHECK(std::abs(backtracked.value().value.at(0U) - std::log(100.0)) <
            1e-10);
  VSS_CHECK(backtracked.value().function_evaluations >
            backtracked.value().iterations + 1U);

  Index jacobianCalls = 0U;
  const VectorFunction exact = [](ConstVectorView) {
    return Result<Vector>::success(vec({0.0, 0.0}));
  };
  const JacobianFunction countedJacobian = [&jacobianCalls](ConstVectorView) {
    ++jacobianCalls;
    return Result<Matrix>::success(Matrix(2U, 2U, 0.0));
  };
  const auto exactInitial = findVectorRoot(
      exact, countedJacobian, vec({1.0, 2.0}), policy(), {100U, 10000U, 100U});
  VSS_CHECK(exactInitial && jacobianCalls == 0U &&
            exactInitial.value().function_evaluations == 1U &&
            exactInitial.value().jacobian_evaluations == 0U);

  const JacobianFunction singular = [](ConstVectorView) {
    Matrix value(2U, 2U, 0.0);
    value.at(0U, 0U) = 1.0;
    return Result<Matrix>::success(std::move(value));
  };
  const auto deficient = findVectorRoot(function, singular, vec({1.0, 0.0}),
                                        policy(), {100U, 10000U, 100U});
  VSS_CHECK(!deficient &&
            deficient.status().code() == ErrorCode::RankDeficient);
  const JacobianFunction wrongShape = [](ConstVectorView) {
    return Result<Matrix>::success(Matrix(1U, 2U, 0.0));
  };
  const auto shape = findVectorRoot(function, wrongShape, vec({1.0, 0.0}),
                                    policy(), {100U, 10000U, 100U});
  VSS_CHECK(!shape && shape.status().code() == ErrorCode::DimensionMismatch);
  IterPolicy noIterations = policy();
  noIterations.max_iterations = 0U;
  const auto capped = findVectorRoot(function, jacobian, vec({1.0, 0.0}),
                                     noIterations, {100U, 10000U, 100U});
  VSS_CHECK(capped && capped.value().termination == Termination::cap &&
            capped.value().function_evaluations == 1U &&
            capped.value().jacobian_evaluations == 0U);
}

} // namespace

int main() try {
  scalarBracketing();
  newtonAndCaps();
  vectorRootContracts();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
