#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Polynomial.hpp>
#include <vss/math/Vector.hpp>

namespace {

using namespace vss::math;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void near(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <=
            1.0e-12 + 1.0e-10 * std::max(std::abs(actual), std::abs(expected)));
}

void fail(const auto& result, ErrorCode expected) {
  VSS_CHECK(!result);
  VSS_CHECK(result.status().code() == expected);
}

constexpr Limits generous{100'000U, 8U * 1024U * 1024U, 0U};

void powerBasisAndEvaluation() {
  const Vector x = vector({0.0, 1.0, 2.0});
  const auto matrix = basisMatrix(x.view(), 2U, {}, generous);
  VSS_CHECK(matrix);
  VSS_CHECK(matrix.value().rows() == 3U && matrix.value().columns() == 3U);
  near(matrix.value().at(2U, 0U), 1.0);
  near(matrix.value().at(2U, 1U), 2.0);
  near(matrix.value().at(2U, 2U), 4.0);
  const Vector coefficients = vector({1.0, 2.0, 3.0});
  const auto value = evaluatePolynomial(coefficients.view(), 2.0, {});
  VSS_CHECK(value);
  near(value.value(), 17.0);
}

void chebyshevAndAffineScaling() {
  const Vector x = vector({-1.0, 0.5, 2.0});
  const BasisSpec chebyshev{BasisKind::chebyshev_first, 0.0, 1.0};
  const auto matrix = basisMatrix(x.view(), 3U, chebyshev, generous);
  VSS_CHECK(matrix);
  near(matrix.value().at(0U, 3U), -1.0);
  near(matrix.value().at(1U, 2U), -0.5);
  near(matrix.value().at(2U, 2U), 7.0);
  const Vector coefficients = vector({1.0, 2.0, 3.0});
  near(evaluatePolynomial(coefficients.view(), 2.0, chebyshev).value(), 26.0);

  const BasisSpec reversed{BasisKind::power, 1.0, -2.0};
  const Vector one = vector({3.0});
  near(evaluatePolynomial(coefficients.view(), 3.0, reversed).value(), 2.0);
  const auto reversed_matrix = basisMatrix(one.view(), 2U, reversed, generous);
  VSS_CHECK(reversed_matrix);
  near(reversed_matrix.value().at(0U, 1U), -1.0);
  near(reversed_matrix.value().at(0U, 2U), 1.0);
}

void dimensionsLimitsAndErrors() {
  const Vector empty;
  const auto empty_matrix = basisMatrix(empty.view(), 3U, {}, generous);
  VSS_CHECK(empty_matrix);
  VSS_CHECK(empty_matrix.value().rows() == 0U);
  VSS_CHECK(empty_matrix.value().columns() == 4U);
  const auto constants = basisMatrix(empty.view(), 0U, {}, generous);
  VSS_CHECK(constants && constants.value().columns() == 1U);
  const Vector x = vector({1.0});
  fail(basisMatrix(x.view(), 2U, {},
                   {.elements = 2U, .bytes = 1024U, .evaluations = 0U}),
       ErrorCode::ResourceLimit);
  fail(basisMatrix(x.view(), 1U, {},
                   {.elements = 8U, .bytes = 8U, .evaluations = 0U}),
       ErrorCode::ResourceLimit);
  fail(basisMatrix(x.view(), 1U, {.scale = 0.0}, generous),
       ErrorCode::InvalidDomain);
  fail(basisMatrix(x.view(), 1U, {static_cast<BasisKind>(255U), 0.0, 1.0},
                   generous),
       ErrorCode::InvalidDomain);
  const Vector nonfinite = vector({std::numeric_limits<Scalar>::infinity()});
  fail(basisMatrix(nonfinite.view(), 0U, {}, generous),
       ErrorCode::NonFiniteInput);
  const Vector no_coefficients;
  fail(evaluatePolynomial(no_coefficients.view(), 0.0, {}),
       ErrorCode::InsufficientSamples);
  const Vector huge = vector(
      {std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max()});
  fail(evaluatePolynomial(huge.view(), 1.0, {}), ErrorCode::NonFiniteResult);
}

void scaledAffineDifferenceAndCancellation() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const BasisSpec scaled{BasisKind::power, -maximum, maximum};
  const Vector x = vector({maximum});
  const auto matrix = basisMatrix(x.view(), 1U, scaled, generous);
  VSS_CHECK(matrix);
  near(matrix.value().at(0U, 1U), 2.0);

  const BasisSpec chebyshev{BasisKind::chebyshev_first, 0.0, 1.0};
  const Vector cancellation = vector({1.0, 0.0, 1.0});
  const auto value =
      evaluatePolynomial(cancellation.view(), maximum, chebyshev);
  fail(value, ErrorCode::NonFiniteResult);
}

} // namespace

int main() try {
  powerBasisAndEvaluation();
  chebyshevAndAffineScaling();
  dimensionsLimitsAndErrors();
  scaledAffineDifferenceAndCancellation();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
