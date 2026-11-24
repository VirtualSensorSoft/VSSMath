#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/probability/MomentBounds.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

void exactAndConservativeBounds() {
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  const auto chebyshev = chebyshevBound(1.0, 2.0);
  const auto cantelli = cantelliBound(1.0, 1.0);
  VSS_CHECK(chebyshev.hasValue() && cantelli.hasValue());
  if (chebyshev && cantelli) {
    VSS_CHECK(chebyshev.value() >= 0.25);
    VSS_CHECK(chebyshev.value() <= 0.25 * (1.0 + 16.0 * epsilon));
    VSS_CHECK(cantelli.value() >= 0.5);
    VSS_CHECK(cantelli.value() <= 0.5 * (1.0 + 16.0 * epsilon));
  }
  const auto zero = chebyshevBound(0.0, 3.0);
  const auto capped = chebyshevBound(9.0, 1.0);
  const auto midpointChebyshev = chebyshevBound(4.0, 2.0);
  const auto tinyChebyshev = chebyshevBound(
      std::numeric_limits<Scalar>::denorm_min(),
      std::numeric_limits<Scalar>::max());
  const auto tinyCantelli = cantelliBound(
      std::numeric_limits<Scalar>::denorm_min(),
      std::numeric_limits<Scalar>::max());
  VSS_CHECK(zero.hasValue() && zero.value() == 0.0);
  VSS_CHECK(capped.hasValue() && capped.value() == 1.0);
  VSS_CHECK(midpointChebyshev.hasValue() && midpointChebyshev.value() == 1.0);
  VSS_CHECK(tinyChebyshev.hasValue() && tinyChebyshev.value() > 0.0);
  VSS_CHECK(tinyCantelli.hasValue() && tinyCantelli.value() > 0.0);
}

void invalidInputs() {
  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  VSS_CHECK(chebyshevBound(-1.0, 1.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(cantelliBound(1.0, 0.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(chebyshevBound(nan, 1.0).status().code() ==
            ErrorCode::NonFiniteInput);
  VSS_CHECK(cantelliBound(1.0, infinity).status().code() ==
            ErrorCode::NonFiniteInput);
}
} // namespace

int main() {
  exactAndConservativeBounds();
  invalidInputs();
}
