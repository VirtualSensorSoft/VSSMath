#include <iostream>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/version.hpp>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "math_consumer: " << message << '\n';
    return false;
  }
  return true;
}

} // namespace

int main() {
  using namespace vss::math;

  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = 2.0;
  matrix.at(1U, 0U) = 3.0;
  matrix.at(1U, 1U) = 4.0;

  Vector vector(2U);
  vector.at(0U) = 5.0;
  vector.at(1U) = 6.0;

  const auto result = multiply(matrix.view(), vector.view());
  if (!require(result.hasValue(), "matrix-vector multiplication failed")) {
    return 1;
  }
  if (!require(result.value().size() == 2U, "unexpected result size")) {
    return 1;
  }
  if (!require(result.value().at(0U) == 17.0 && result.value().at(1U) == 39.0,
               "unexpected matrix-vector result")) {
    return 1;
  }

  Matrix diagonal(2U, 2U);
  diagonal.at(0U, 0U) = 2.0;
  diagonal.at(1U, 1U) = 3.0;
  Vector rightHandSide(2U);
  rightHandSide.at(0U) = 4.0;
  rightHandSide.at(1U) = 9.0;
  const auto factorized = vss::math::linear::factorizeLU(diagonal.view());
  if (!require(factorized.hasValue(), "LU factorization failed")) {
    return 1;
  }
  const auto solved = factorized.value().solve(rightHandSide.view());
  if (!require(solved.hasValue(), "LU solve failed")) {
    return 1;
  }
  if (!require(solved.value().solution().at(0U) == 2.0 &&
                   solved.value().solution().at(1U) == 3.0,
               "unexpected LU solution")) {
    return 1;
  }
  if (!require(solved.value().conditionReport().has_value(),
               "LU solve did not return condition diagnostics")) {
    return 1;
  }

  std::cout << "VirtualSensorMath " << version::string << " consumer passed\n";
  return 0;
}
