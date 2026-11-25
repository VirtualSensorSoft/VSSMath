#include "Test.hpp"

#include <cmath>
#include <stdexcept>
#include <vss/math/Roots.hpp>

namespace {

using namespace vss::math;

IterPolicy policy() { return {.abs = 1.0e-12, .rel = 1.0e-12,
                              .max_iterations = 128U,
                              .max_evaluations = 512U,
                              .bracket = std::nullopt}; }

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
  const auto exactEndpoint = rootBisection(
      [](Scalar x) { return Result<Scalar>::success(x); }, {0.0, 2.0},
      policy());
  VSS_CHECK(exactEndpoint && exactEndpoint.value().value == 0.0);
  VSS_CHECK(exactEndpoint.value().function_evaluations == 1U);
  VSS_CHECK(exactEndpoint.value().enclosure->lower == 0.0);
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
  VSS_CHECK(cappedResult && cappedResult.value().termination == Termination::cap);
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
  VSS_CHECK(!zeroSlope && zeroSlope.status().code() == ErrorCode::RankDeficient);
  const auto escaped = rootNewton(
      [](Scalar) -> Result<Scalar> { throw std::runtime_error("failure"); },
      df, 1.0, policy());
  VSS_CHECK(!escaped && escaped.status().code() == ErrorCode::CallbackFailure);
}

} // namespace

int main() try {
  scalarBracketing();
  newtonAndCaps();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
