#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/ODE.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/Vector.hpp>

namespace {

using namespace vss::math;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

OdePolicy policy(OdeMethod method = OdeMethod::dormand_prince_54) {
  return {.method = method, .min_step = 1.0e-8, .initial_step = 0.1,
          .max_step = 0.25,
          .iteration = {.abs = 1.0e-10, .rel = 1.0e-10,
                        .max_iterations = 10'000U,
                        .max_evaluations = 100'000U,
                        .bracket = std::nullopt},
          .parameters = std::nullopt,
          .initial_sensitivity = std::nullopt};
}

void near(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-7) {
  VSS_CHECK(std::abs(actual - expected) <=
            tolerance + tolerance * std::max(std::abs(actual), std::abs(expected)));
}

void explicitIntegration() {
  const OdeFunction exponential = [](Scalar, ConstVectorView state,
                                      ConstVectorView) {
    Vector derivative(state.size());
    derivative.at(0U) = state.at(0U);
    return Result<Vector>::success(std::move(derivative));
  };
  const auto trajectory = odeIntegrate(exponential, vector({1.0}),
                                       vector({0.0, 0.5, 1.0}), policy(),
                                       {100'000U, 32U * 1024U * 1024U,
                                        100'000U});
  VSS_CHECK(trajectory &&
            trajectory.value().termination == Termination::converged);
  VSS_CHECK(trajectory.value().states.rows() == 3U);
  near(trajectory.value().states.at(2U, 0U), std::exp(1.0));
  VSS_CHECK(trajectory.value().accepted_steps > 0U);

  Index calls = 0U;
  const OdeFunction counted = [&calls](Scalar, ConstVectorView state,
                                       ConstVectorView) {
    ++calls;
    return Result<Vector>::success(Vector(state.size(), 1.0));
  };
  const auto single = odeIntegrate(counted, vector({2.0}), vector({4.0}),
                                  policy(),
                                  {100U, 10000U, 100U});
  VSS_CHECK(single && calls == 0U && single.value().attempts == 0U);
  near(single.value().states.at(0U, 0U), 2.0, 0.0);
}

void sensitivityAndValidation() {
  const OdeFunction f = [](Scalar, ConstVectorView y, ConstVectorView p) {
    Vector derivative(1U);
    derivative.at(0U) = p.at(0U) * y.at(0U);
    return Result<Vector>::success(std::move(derivative));
  };
  const OdeJacobians jacobians{
      .state = [](Scalar, ConstVectorView, ConstVectorView p) {
        Matrix value(1U, 1U);
        value.at(0U, 0U) = p.at(0U);
        return Result<Matrix>::success(std::move(value));
      },
      .parameters = [](Scalar, ConstVectorView y, ConstVectorView) {
        Matrix value(1U, 1U);
        value.at(0U, 0U) = y.at(0U);
        return Result<Matrix>::success(std::move(value));
      }};
  const auto sensitivity = odeSensitivity(
      f, jacobians, vector({1.0}), vector({2.0}), vector({0.0, 0.5, 1.0}),
      policy(), {100'000U, 32U * 1024U * 1024U, 100'000U});
  VSS_CHECK(sensitivity &&
            sensitivity.value().sensitivities.shape().rank() == 3U);
  near(sensitivity.value().states.at(2U, 0U), std::exp(2.0));
  const std::array<Index, 3> index{2U, 0U, 0U};
  near(sensitivity.value().sensitivities.at(index), std::exp(2.0), 2.0e-6);

  const Vector empty;
  const auto noTime = odeIntegrate(f, vector({1.0}), empty, policy(),
                                   {100U, 10000U, 100U});
  VSS_CHECK(!noTime && noTime.status().code() == ErrorCode::InsufficientSamples);
  const auto invalid = odeIntegrate(
      f, vector({1.0}), vector({1.0, 0.0}), policy(),
      {100U, 10000U, 100U});
  VSS_CHECK(!invalid && invalid.status().code() == ErrorCode::InvalidDomain);
}

void implicitConstantFlow() {
  const OdeFunction constant = [](Scalar, ConstVectorView state,
                                  ConstVectorView) {
    return Result<Vector>::success(Vector(state.size(), 1.0));
  };
  const auto trajectory = odeIntegrate(
      constant, vector({0.0}), vector({0.0, 1.0}),
      policy(OdeMethod::backward_euler),
      {100'000U, 32U * 1024U * 1024U, 100'000U});
  VSS_CHECK(trajectory && trajectory.value().termination == Termination::converged);
  near(trajectory.value().states.at(1U, 0U), 1.0, 1.0e-8);

  const OdeFunction parameterFlow = [](Scalar, ConstVectorView,
                                       ConstVectorView p) {
    return Result<Vector>::success(Vector(1U, p.at(0U)));
  };
  const OdeJacobians jacobians{
      .state = [](Scalar, ConstVectorView, ConstVectorView) {
        return Result<Matrix>::success(Matrix(1U, 1U, 0.0));
      },
      .parameters = [](Scalar, ConstVectorView, ConstVectorView) {
        return Result<Matrix>::success(Matrix(1U, 1U, 1.0));
      }};
  const auto sensitivity = odeSensitivity(
      parameterFlow, jacobians, vector({0.0}), vector({2.0}),
      vector({0.0, 1.0}), policy(OdeMethod::backward_euler),
      {100'000U, 32U * 1024U * 1024U, 100'000U});
  VSS_CHECK(sensitivity);
  near(sensitivity.value().states.at(1U, 0U), 2.0, 1.0e-8);
  const std::array<Index, 3> index{1U, 0U, 0U};
  near(sensitivity.value().sensitivities.at(index), 1.0, 1.0e-8);
}

void callbackAndCapBehavior() {
  Index calls = 0U;
  const OdeFunction counted = [&calls](Scalar, ConstVectorView state,
                                       ConstVectorView) {
    ++calls;
    return Result<Vector>::success(Vector(state.size(), 1.0));
  };
  const auto denied = odeIntegrate(counted, vector({0.0}),
      vector({0.0, 1.0}), policy(), {10'000U, 1U, 10'000U});
  VSS_CHECK(!denied && denied.status().code() == ErrorCode::ResourceLimit &&
            calls == 0U);
  OdePolicy oneAttempt = policy();
  oneAttempt.iteration.max_iterations = 1U;
  oneAttempt.min_step = 0.01;
  oneAttempt.initial_step = 0.01;
  oneAttempt.max_step = 0.01;
  const auto capped = odeIntegrate(counted, vector({0.0}),
      vector({0.0, 1.0}), oneAttempt,
      {100'000U, 32U * 1024U * 1024U, 100'000U});
  VSS_CHECK(capped && capped.value().termination == Termination::cap &&
            capped.value().attempts == 1U && capped.value().times.size() == 1U);
}

} // namespace

int main() try {
  explicitIntegration();
  sensitivityAndValidation();
  implicitConstantFlow();
  callbackAndCapBehavior();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
