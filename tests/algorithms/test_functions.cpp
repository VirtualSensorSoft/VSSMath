#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Functions.hpp>

namespace {
using namespace vss::math;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void recurrenceAndBoundaries() {
  const auto result = exponentialSmooth(vector({2.0, 4.0, 8.0}).view(), 0.5,
                                        0.0, {3U, 3U * sizeof(Scalar), 0U});
  VSS_CHECK(result && result.value().size() == 3U);
  VSS_CHECK(result.value().at(0U) == 1.0);
  VSS_CHECK(result.value().at(1U) == 2.5);
  VSS_CHECK(result.value().at(2U) == 5.25);

  const auto alphaZero = exponentialSmooth(vector({3.0, -2.0}).view(), 0.0,
                                           7.0, {2U, 16U, 0U});
  VSS_CHECK(alphaZero && alphaZero.value().at(0U) == 7.0 &&
            alphaZero.value().at(1U) == 7.0);
  const auto alphaOne = exponentialSmooth(vector({3.0, -2.0}).view(), 1.0,
                                           7.0, {2U, 16U, 0U});
  VSS_CHECK(alphaOne && alphaOne.value().at(0U) == 3.0 &&
            alphaOne.value().at(1U) == -2.0);
  const Vector empty;
  const auto none = exponentialSmooth(empty.view(), 0.5, 1.0, {0U, 0U, 0U});
  VSS_CHECK(none && none.value().empty());
}

void splitReplayAndValidation() {
  const Vector input = vector({1.0, 3.0, -2.0, 6.0});
  const auto whole = exponentialSmooth(input.view(), 0.3, 2.0,
                                       {4U, 32U, 0U});
  VSS_CHECK(whole);
  const auto firstView = input.view().subview(0U, 2U);
  VSS_CHECK(firstView);
  const auto first = exponentialSmooth(firstView.value(), 0.3, 2.0,
                                       {2U, 16U, 0U});
  VSS_CHECK(first);
  const auto secondView = input.view().subview(2U, 2U);
  VSS_CHECK(secondView);
  const auto second = exponentialSmooth(secondView.value(), 0.3,
                                        first.value().at(1U), {2U, 16U, 0U});
  VSS_CHECK(second);
  VSS_CHECK(std::abs(second.value().at(1U) - whole.value().at(3U)) < 1e-14);

  const auto badAlpha = exponentialSmooth(input.view(), 1.01, 0.0,
                                          {4U, 32U, 0U});
  VSS_CHECK(!badAlpha && badAlpha.status().code() == ErrorCode::InvalidDomain);
  Vector nonFinite = vector({1.0, 2.0});
  nonFinite.at(1U) = std::numeric_limits<Scalar>::quiet_NaN();
  const auto badInput = exponentialSmooth(nonFinite.view(), 0.5, 0.0,
                                          {2U, 16U, 0U});
  VSS_CHECK(!badInput && badInput.status().code() == ErrorCode::NonFiniteInput);
  const auto cap = exponentialSmooth(input.view(), 0.5, 0.0, {4U, 1U, 0U});
  VSS_CHECK(!cap && cap.status().code() == ErrorCode::ResourceLimit);
}

void extremeConvexCombination() {
  const Vector input = vector({std::numeric_limits<Scalar>::max(),
                               -std::numeric_limits<Scalar>::max()});
  const auto result = exponentialSmooth(input.view(), 0.5,
                                        -std::numeric_limits<Scalar>::max(),
                                        {2U, 16U, 0U});
  VSS_CHECK(result && std::isfinite(result.value().at(0U)) &&
            std::isfinite(result.value().at(1U)));
  VSS_CHECK(result.value().at(0U) == 0.0);
  VSS_CHECK(result.value().at(1U) < 0.0);
}
} // namespace

int main() try {
  recurrenceAndBoundaries();
  splitReplayAndValidation();
  extremeConvexCombination();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
