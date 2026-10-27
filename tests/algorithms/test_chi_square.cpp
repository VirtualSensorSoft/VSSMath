#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <detail/NumericalClassification.hpp>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/ChiSquare.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Scalar;

void checkNear(Scalar actual, Scalar expected, Scalar absolute = 1.0e-14,
               Scalar relative = 1.0e-10) {
  const Scalar scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  VSS_CHECK(std::abs(actual - expected) <= (absolute + (relative * scale)));
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void publicContract() {
  using Signature = vss::math::Result<Scalar> (*)(Scalar, Extent);
  static_assert(std::is_same_v<decltype(static_cast<Signature>(
                                   &vss::math::probability::chiSquareSurvival)),
                               Signature>);
  static_assert(
      std::is_same_v<decltype(static_cast<Signature>(
                         &vss::math::probability::chiSquareLogSurvival)),
                     Signature>);
}

void zeroChiSquare() {
  for (Extent degreesOfFreedom = 1U; degreesOfFreedom <= 10000U;
       ++degreesOfFreedom) {
    const auto survival =
        vss::math::probability::chiSquareSurvival(0.0, degreesOfFreedom);
    const auto logSurvival =
        vss::math::probability::chiSquareLogSurvival(0.0, degreesOfFreedom);
    VSS_CHECK(survival.hasValue());
    VSS_CHECK(logSurvival.hasValue());
    if (survival.hasValue() && logSurvival.hasValue()) {
      VSS_CHECK(survival.value() == 1.0);
      VSS_CHECK(logSurvival.value() == 0.0);
    }
  }
}

void twoDegreesOfFreedom() {
  const Scalar expected = std::exp(-1.0);
  const auto survival = vss::math::probability::chiSquareSurvival(2.0, 2U);
  const auto logSurvival =
      vss::math::probability::chiSquareLogSurvival(2.0, 2U);
  VSS_CHECK(survival.hasValue());
  VSS_CHECK(logSurvival.hasValue());
  if (survival.hasValue() && logSurvival.hasValue()) {
    checkNear(survival.value(), expected);
    checkNear(logSurvival.value(), -1.0);
  }
}

void fourDegreesOfFreedom() {
  const Scalar expected = 2.0 * std::exp(-1.0);
  const auto survival = vss::math::probability::chiSquareSurvival(2.0, 4U);
  const auto logSurvival =
      vss::math::probability::chiSquareLogSurvival(2.0, 4U);
  VSS_CHECK(survival.hasValue());
  VSS_CHECK(logSurvival.hasValue());
  if (survival.hasValue() && logSurvival.hasValue()) {
    checkNear(survival.value(), expected);
    checkNear(logSurvival.value(), std::log(expected));
  }
}

void oneDegreeFivePercentTail() {
  const auto result =
      vss::math::probability::chiSquareSurvival(3.841458820694124, 1U);
  VSS_CHECK(result.hasValue());
  if (result.hasValue()) {
    checkNear(result.value(), 0.0500000000000000574353696875729153);
    VSS_CHECK(result.value() > 0.0);
    VSS_CHECK(result.value() <= 1.0);
  }
}

void tenDegreeFivePercentTail() {
  const auto result =
      vss::math::probability::chiSquareSurvival(18.307038053275146, 10U);
  VSS_CHECK(result.hasValue());
  if (result.hasValue()) {
    checkNear(result.value(), 0.0500000000000000067056857232555);
    VSS_CHECK(result.value() > 0.0);
    VSS_CHECK(result.value() <= 1.0);
  }
}

void extremeLogTail() {
  const auto survival = vss::math::probability::chiSquareSurvival(2000.0, 2U);
  const auto logSurvival =
      vss::math::probability::chiSquareLogSurvival(2000.0, 2U);
  VSS_CHECK(survival.hasValue());
  VSS_CHECK(logSurvival.hasValue());
  if (survival.hasValue() && logSurvival.hasValue()) {
    VSS_CHECK(survival.value() >= 0.0);
    VSS_CHECK(survival.value() <= 1.0);
    VSS_CHECK(std::isfinite(logSurvival.value()));
    checkNear(logSurvival.value(), -1000.0, 1.0e-10, 0.0);
  }
}

void domainFailures() {
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  const std::vector<std::pair<Scalar, ErrorCode>> invalidValues{
      {-1.0, ErrorCode::InvalidDomain},
      {nan, ErrorCode::NonFiniteInput},
      {infinity, ErrorCode::NonFiniteInput}};
  for (const auto& [value, code] : invalidValues) {
    checkFailure(vss::math::probability::chiSquareSurvival(value, 1U), code);
    checkFailure(vss::math::probability::chiSquareLogSurvival(value, 1U), code);
  }
  checkFailure(vss::math::probability::chiSquareSurvival(1.0, 0U),
               ErrorCode::InvalidDomain);
  checkFailure(vss::math::probability::chiSquareLogSurvival(1.0, 0U),
               ErrorCode::InvalidDomain);
  checkFailure(vss::math::probability::chiSquareSurvival(1.0, 10001U),
               ErrorCode::InvalidDomain);
  checkFailure(vss::math::probability::chiSquareLogSurvival(1.0, 10001U),
               ErrorCode::InvalidDomain);
}

void probabilityRangeBoundary() {
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  const auto nearUpper =
      vss::math::detail::classifyUnitProbability(1.0 + (16.0 * epsilon));
  const auto farUpper =
      vss::math::detail::classifyUnitProbability(1.0 + (64.0 * epsilon));
  const auto nearLower =
      vss::math::detail::classifyUnitProbability(-16.0 * epsilon);
  const auto farLower =
      vss::math::detail::classifyUnitProbability(-64.0 * epsilon);
  VSS_CHECK(nearUpper.hasValue());
  VSS_CHECK(nearLower.hasValue());
  if (nearUpper.hasValue() && nearLower.hasValue()) {
    VSS_CHECK(nearUpper.value() == 1.0);
    VSS_CHECK(nearLower.value() == 0.0);
  }
  checkFailure(farUpper, ErrorCode::NonFiniteResult);
  checkFailure(farLower, ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  publicContract();
  zeroChiSquare();
  twoDegreesOfFreedom();
  fourDegreesOfFreedom();
  oneDegreeFivePercentTail();
  tenDegreeFivePercentTail();
  extremeLogTail();
  domainFailures();
  probabilityRangeBoundary();
}

// VSS-MATH-TRACEABILITY: PRO-01 PRO-02 PRO-03 PRO-04 PRO-05 PRO-06 PRO-07
// VSS-MATH-TRACEABILITY: SUP-PRO-01
