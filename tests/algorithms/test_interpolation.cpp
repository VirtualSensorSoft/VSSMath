#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Interpolation.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Vector.hpp>

namespace {

using namespace vss::math;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void near(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-10) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance +
            1.0e-10 * std::max(std::abs(actual), std::abs(expected)));
}

void fail(const auto& result, ErrorCode expected) {
  VSS_CHECK(!result);
  VSS_CHECK(result.status().code() == expected);
}

constexpr Limits generous{100'000U, 8U * 1024U * 1024U, 0U};

void linearInterpolationAndGap() {
  const Vector nodes = vector({0.0, 1.0, 2.0});
  const Vector values = vector({0.0, 2.0, 4.0});
  near(interpolate(nodes.view(), values.view(), 0.5, {}).value(), 1.0);
  near(interpolate(nodes.view(), values.view(), 1.0, {}).value(), 2.0);
  const InterpolateSpec maxGap{Extrapolation::reject, 1.0};
  near(interpolate(nodes.view(), values.view(), 0.5, maxGap).value(), 1.0);

  const Vector wideNodes = vector({0.0, 2.0});
  const Vector wideValues = vector({0.0, 4.0});
  fail(interpolate(wideNodes.view(), wideValues.view(), 1.0, maxGap),
       ErrorCode::InvalidDomain);
  const InterpolateSpec noGap{Extrapolation::reject, 2.0};
  near(interpolate(wideNodes.view(), wideValues.view(), 1.0, noGap).value(), 2.0);
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector extremeValues = vector({-maximum, maximum});
  near(interpolate(wideNodes.view(), extremeValues.view(), 1.0, {}).value(),
       0.0);
}

void extrapolationAndValidation() {
  const Vector nodes = vector({0.0, 1.0});
  const Vector values = vector({2.0, 4.0});
  fail(interpolate(nodes.view(), values.view(), -1.0, {}),
       ErrorCode::InvalidDomain);
  near(interpolate(nodes.view(), values.view(), -1.0,
                   {Extrapolation::hold, std::nullopt}).value(), 2.0);
  near(interpolate(nodes.view(), values.view(), -1.0,
                   {Extrapolation::linear, std::nullopt}).value(), 0.0);
  near(interpolate(nodes.view(), values.view(), 2.0,
                   {Extrapolation::linear, std::nullopt}).value(), 6.0);

  const Vector one = vector({1.0});
  fail(interpolate(one.view(), one.view(), 1.0, {}),
       ErrorCode::InsufficientSamples);
  const Vector other = vector({1.0, 2.0, 3.0});
  fail(interpolate(nodes.view(), other.view(), 0.5, {}),
       ErrorCode::DimensionMismatch);
  const Vector unordered = vector({0.0, 0.0});
  const Vector pair = vector({1.0, 2.0});
  fail(interpolate(unordered.view(), pair.view(), 0.5, {}),
       ErrorCode::InvalidDomain);
  fail(interpolate(nodes.view(), values.view(), 0.5,
                   {Extrapolation::reject, 0.0}),
       ErrorCode::InvalidTolerance);
  fail(interpolate(nodes.view(), values.view(), 0.5,
                   {static_cast<Extrapolation>(255U), std::nullopt}),
       ErrorCode::InvalidDomain);
  fail(interpolate(nodes.view(), values.view(),
                   std::numeric_limits<Scalar>::quiet_NaN(), {}),
       ErrorCode::NonFiniteInput);
}

void naturalAndClampedSplines() {
  const Vector nodes = vector({0.0, 1.0, 2.0});
  const Vector line = vector({0.0, 2.0, 4.0});
  const auto natural = makeSpline(nodes.view(), line.view(), {}, generous);
  VSS_CHECK(natural);
  VSS_CHECK(natural.value().coefficients.rows() == 2U);
  VSS_CHECK(natural.value().coefficients.columns() == 4U);
  near(evaluate(natural.value(), 0.5, Extrapolation::reject).value(), 1.0);
  near(evaluate(natural.value(), 1.0, Extrapolation::reject).value(), 2.0);
  near(evaluate(natural.value(), 2.0, Extrapolation::reject).value(), 4.0);
  near(evaluate(natural.value(), -1.0, Extrapolation::linear).value(), -2.0);
  near(evaluate(natural.value(), -1.0, Extrapolation::hold).value(), 0.0);

  const Vector squareValues = vector({0.0, 1.0, 4.0});
  const SplineSpec clamped{SplineBoundary::clamped, 0.0, 4.0};
  const auto cubic = makeSpline(nodes.view(), squareValues.view(), clamped,
                                generous);
  VSS_CHECK(cubic);
  near(evaluate(cubic.value(), 0.5, Extrapolation::reject).value(), 0.25);
  near(evaluate(cubic.value(), 1.5, Extrapolation::reject).value(), 2.25);
  near(evaluate(cubic.value(), 3.0, Extrapolation::linear).value(), 8.0);

  const Vector twoNodes = vector({0.0, 1.0});
  const Vector cube = vector({0.0, 1.0});
  const SplineSpec cubeClamped{SplineBoundary::clamped, 0.0, 3.0};
  const auto hermite = makeSpline(twoNodes.view(), cube.view(), cubeClamped,
                                  generous);
  VSS_CHECK(hermite);
  near(evaluate(hermite.value(), 0.5, Extrapolation::reject).value(), 0.125);
}

void splineErrorsAndCallerValidation() {
  const Vector nodes = vector({0.0, 1.0});
  const Vector values = vector({0.0, 1.0});
  fail(makeSpline(nodes.view(), values.view(),
                  {SplineBoundary::clamped, std::nullopt, 1.0}, generous),
       ErrorCode::InvalidDomain);
  fail(makeSpline(nodes.view(), values.view(),
                  {static_cast<SplineBoundary>(255U), std::nullopt,
                   std::nullopt}, generous),
       ErrorCode::InvalidDomain);
  fail(makeSpline(nodes.view(), values.view(),
                  {SplineBoundary::clamped,
                   std::numeric_limits<Scalar>::infinity(), 1.0}, generous),
       ErrorCode::NonFiniteInput);
  fail(makeSpline(nodes.view(), values.view(), {},
                  {.elements = 2U, .bytes = 4096U, .evaluations = 0U}),
       ErrorCode::ResourceLimit);
  const auto spline = makeSpline(nodes.view(), values.view(), {}, generous);
  VSS_CHECK(spline);
  fail(evaluate(spline.value(), -1.0, Extrapolation::reject),
       ErrorCode::InvalidDomain);
  fail(evaluate(spline.value(), 0.5,
                static_cast<Extrapolation>(255U)), ErrorCode::InvalidDomain);
  Spline malformed{nodes, Matrix(2U, 4U)};
  fail(evaluate(malformed, 0.5, Extrapolation::reject),
       ErrorCode::DimensionMismatch);
}

} // namespace

int main() try {
  linearInterpolationAndGap();
  extrapolationAndValidation();
  naturalAndClampedSplines();
  splineErrorsAndCallerValidation();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
