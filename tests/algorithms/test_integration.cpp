#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Integration.hpp>
#include <vss/math/Vector.hpp>

namespace {

using namespace vss::math;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

IterPolicy policy(Index evaluations = 100'000U, Index iterations = 128U) {
  return {.abs = 1.0e-12,
          .rel = 1.0e-12,
          .max_iterations = iterations,
          .max_evaluations = evaluations,
          .bracket = std::nullopt};
}

void near(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <=
            1.0e-12 + 1.0e-10 * std::max(std::abs(actual), std::abs(expected)));
}

void adaptiveCases() {
  const UnaryFunction square = [](Scalar x) {
    return Result<Scalar>::success(x * x);
  };
  const Limits resources{100'000U, 4U * 1024U * 1024U, 100'000U};
  const auto result =
      quadratureAdaptive(square, {0.0, 1.0}, policy(), resources);
  VSS_CHECK(result && result.value().termination == Termination::converged);
  near(result.value().value, 1.0 / 3.0);
  VSS_CHECK(result.value().evaluations == 15U);

  const UnaryFunction exponential = [](Scalar x) {
    return Result<Scalar>::success(std::exp(x));
  };
  near(quadratureAdaptive(exponential, {0.0, 1.0}, policy(), resources)
           .value()
           .value,
       std::exp(1.0) - 1.0);

  const UnaryFunction kink = [](Scalar x) {
    return Result<Scalar>::success(std::abs(x - 0.3));
  };
  const auto subdivided =
      quadratureAdaptive(kink, {0.0, 1.0}, policy(), resources);
  VSS_CHECK(subdivided && subdivided.value().panels > 1U);
  near(subdivided.value().value, 0.29);

  const auto reversed =
      quadratureAdaptive(square, {1.0, 0.0}, policy(), resources);
  VSS_CHECK(reversed);
  near(reversed.value().value, -1.0 / 3.0);
  const auto zero = quadratureAdaptive(square, {2.0, 2.0}, policy(), resources);
  VSS_CHECK(zero && zero.value().value == 0.0 &&
            !std::signbit(zero.value().value) &&
            zero.value().evaluations == 0U);
}

void adaptiveFailuresAndCaps() {
  const Limits resources{100U, 4096U, 100U};
  const UnaryFunction f = [](Scalar x) {
    return Result<Scalar>::success(x * x);
  };
  const auto tooFew = quadratureAdaptive(f, {0.0, 1.0}, policy(14U), resources);
  VSS_CHECK(!tooFew && tooFew.status().code() == ErrorCode::ResourceLimit);
  const UnaryFunction kink = [](Scalar x) {
    return Result<Scalar>::success(std::abs(x - 0.3));
  };
  const auto cap =
      quadratureAdaptive(kink, {0.0, 1.0}, policy(1000U, 0U), resources);
  VSS_CHECK(cap && cap.value().termination == Termination::cap);
  const auto bracket = quadratureAdaptive(f, {0.0, 1.0},
                                          {.abs = 1.0e-6,
                                           .rel = 1.0e-6,
                                           .max_iterations = 2U,
                                           .max_evaluations = 100U,
                                           .bracket = Interval{0.0, 1.0}},
                                          resources);
  VSS_CHECK(!bracket && bracket.status().code() == ErrorCode::InvalidDomain);
  const auto error = quadratureAdaptive(
      [](Scalar) {
        return Result<Scalar>::failure(ErrorCode::CallbackFailure,
                                       "callback error");
      },
      {0.0, 1.0}, policy(), resources);
  VSS_CHECK(!error && error.status().code() == ErrorCode::CallbackFailure);
}

void sampledRules() {
  const Vector nodes = vector({0.0, 1.0, 3.0});
  const Vector values = vector({0.0, 1.0, 3.0});
  near(integrateSamples(nodes.view(), values.view(), SampleRule::trapezoid)
           .value(),
       4.5);
  const Vector evenNodes = vector({0.0, 0.5, 1.0});
  const Vector squares = vector({0.0, 0.25, 1.0});
  near(integrateSamples(evenNodes.view(), squares.view(), SampleRule::simpson)
           .value(),
       1.0 / 3.0);
  const Vector tooFew = vector({0.0, 1.0});
  const Vector two = vector({0.0, 1.0});
  const auto insufficient =
      integrateSamples(tooFew.view(), two.view(), SampleRule::simpson);
  VSS_CHECK(!insufficient &&
            insufficient.status().code() == ErrorCode::InsufficientSamples);
  const Vector oddPanels = vector({0.0, 0.5, 1.0, 1.5});
  const Vector same = vector({0.0, 0.5, 1.0, 1.5});
  const auto parity =
      integrateSamples(oddPanels.view(), same.view(), SampleRule::simpson);
  VSS_CHECK(!parity && parity.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() try {
  adaptiveCases();
  adaptiveFailuresAndCaps();
  sampledRules();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
