#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Order.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Limits;
using vss::math::QuantileRule;
using vss::math::Scalar;
using vss::math::Vector;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

constexpr Limits unlimited{std::numeric_limits<std::size_t>::max(),
                           std::numeric_limits<std::size_t>::max(),
                           std::numeric_limits<std::uint64_t>::max()};

void checkNear(Scalar actual, Scalar expected) {
  constexpr Scalar tolerance = 1.0e-12;
  VSS_CHECK(std::abs(actual - expected) <=
            tolerance * std::max({1.0, std::abs(actual), std::abs(expected)}));
}

template <class T>
void checkFailure(const vss::math::Result<T>& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void publicSignatures() {
  using QuantileSignature = vss::math::Result<Scalar> (*)(
      vss::math::ConstVectorView, Scalar, QuantileRule, Limits);
  using SampleStatisticSignature =
      vss::math::Result<Scalar> (*)(vss::math::ConstVectorView, Limits);
  using TailMeanSignature =
      vss::math::Result<Scalar> (*)(vss::math::ConstVectorView, Scalar, Limits);
  using PositiveMeanSignature =
      vss::math::Result<Scalar> (*)(vss::math::ConstVectorView);
  using MadSignature =
      vss::math::Result<Scalar> (*)(vss::math::ConstVectorView, Scalar, Limits);
  using IqrSignature = vss::math::Result<Scalar> (*)(vss::math::ConstVectorView,
                                                     QuantileRule, Limits);
  static_assert(std::is_same_v<decltype(static_cast<QuantileSignature>(
                                   &vss::math::statistics::quantile)),
                               QuantileSignature>);
  static_assert(std::is_same_v<decltype(static_cast<SampleStatisticSignature>(
                                   &vss::math::statistics::median)),
                               SampleStatisticSignature>);
  static_assert(std::is_same_v<decltype(static_cast<TailMeanSignature>(
                                   &vss::math::statistics::trimmedMean)),
                               TailMeanSignature>);
  static_assert(std::is_same_v<decltype(static_cast<TailMeanSignature>(
                                   &vss::math::statistics::winsorizedMean)),
                               TailMeanSignature>);
  static_assert(std::is_same_v<decltype(static_cast<PositiveMeanSignature>(
                                   &vss::math::statistics::geometricMean)),
                               PositiveMeanSignature>);
  static_assert(std::is_same_v<decltype(static_cast<PositiveMeanSignature>(
                                   &vss::math::statistics::harmonicMean)),
                               PositiveMeanSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<MadSignature>(
                         &vss::math::statistics::medianAbsoluteDeviation)),
                     MadSignature>);
  static_assert(std::is_same_v<decltype(static_cast<IqrSignature>(
                                   &vss::math::statistics::interquartileRange)),
                               IqrSignature>);
}

void robustOrderSummaryCases() {
  const Vector values = vector({1.0, 2.0, 3.0, 5.0});
  checkNear(vss::math::statistics::medianAbsoluteDeviation(values.view(), 1.0,
                                                           unlimited)
                .value(),
            1.0);
  checkNear(vss::math::statistics::medianAbsoluteDeviation(values.view(), 0.0,
                                                           unlimited)
                .value(),
            0.0);
  checkNear(vss::math::statistics::interquartileRange(
                values.view(), QuantileRule::linear, unlimited)
                .value(),
            1.75);
  checkNear(vss::math::statistics::interquartileRange(
                values.view(), QuantileRule::inverse_empirical, unlimited)
                .value(),
            2.0);

  const Vector empty;
  checkFailure(vss::math::statistics::medianAbsoluteDeviation(empty.view(), 1.0,
                                                              unlimited),
               ErrorCode::InsufficientSamples);
  checkFailure(vss::math::statistics::interquartileRange(
                   empty.view(), QuantileRule::linear, unlimited),
               ErrorCode::InsufficientSamples);
  checkFailure(vss::math::statistics::medianAbsoluteDeviation(values.view(),
                                                              -1.0, unlimited),
               ErrorCode::InvalidDomain);
  checkFailure(
      vss::math::statistics::medianAbsoluteDeviation(
          values.view(), std::numeric_limits<Scalar>::infinity(), unlimited),
      ErrorCode::InvalidDomain);
  checkFailure(vss::math::statistics::interquartileRange(
                   values.view(), static_cast<QuantileRule>(255U), unlimited),
               ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(vss::math::statistics::medianAbsoluteDeviation(nonfinite.view(),
                                                              1.0, unlimited),
               ErrorCode::NonFiniteInput);
  checkFailure(vss::math::statistics::interquartileRange(
                   nonfinite.view(), QuantileRule::linear, unlimited),
               ErrorCode::NonFiniteInput);
  const Limits tooSmall{1U, std::numeric_limits<std::size_t>::max(), 0U};
  checkFailure(vss::math::statistics::medianAbsoluteDeviation(values.view(),
                                                              1.0, tooSmall),
               ErrorCode::ResourceLimit);
  checkFailure(vss::math::statistics::interquartileRange(
                   values.view(), QuantileRule::linear, tooSmall),
               ErrorCode::ResourceLimit);

  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector largeMad = vector({-1.0, 1.0, 3.0, 5.0});
  checkFailure(vss::math::statistics::medianAbsoluteDeviation(
                   largeMad.view(), maximum, unlimited),
               ErrorCode::NonFiniteResult);
  const Vector wideIqr = vector({-maximum, -maximum, maximum, maximum});
  checkFailure(vss::math::statistics::interquartileRange(
                   wideIqr.view(), QuantileRule::linear, unlimited),
               ErrorCode::NonFiniteResult);
}

void quantileCases() {
  const Vector values = vector({1.0, 2.0, 4.0, 8.0});
  checkNear(vss::math::statistics::quantile(values.view(), 0.25,
                                            QuantileRule::linear, unlimited)
                .value(),
            1.75);
  checkNear(vss::math::statistics::quantile(
                values.view(), 0.25, QuantileRule::inverse_empirical, unlimited)
                .value(),
            1.0);
  checkNear(vss::math::statistics::quantile(values.view(), 0.0,
                                            QuantileRule::linear, unlimited)
                .value(),
            1.0);
  checkNear(vss::math::statistics::quantile(values.view(), 1.0,
                                            QuantileRule::linear, unlimited)
                .value(),
            8.0);

  const Vector empty;
  checkFailure(vss::math::statistics::quantile(empty.view(), 0.5,
                                               QuantileRule::linear, unlimited),
               ErrorCode::InsufficientSamples);
  checkFailure(vss::math::statistics::quantile(values.view(), -0.01,
                                               QuantileRule::linear, unlimited),
               ErrorCode::InvalidDomain);
  checkFailure(vss::math::statistics::quantile(values.view(), 1.01,
                                               QuantileRule::linear, unlimited),
               ErrorCode::InvalidDomain);
  checkFailure(vss::math::statistics::quantile(values.view(), 0.5,
                                               static_cast<QuantileRule>(255U),
                                               unlimited),
               ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(vss::math::statistics::quantile(nonfinite.view(), 0.5,
                                               QuantileRule::linear, unlimited),
               ErrorCode::NonFiniteInput);
  checkFailure(vss::math::statistics::quantile(nonfinite.view(), -0.1,
                                               QuantileRule::linear, unlimited),
               ErrorCode::InvalidDomain);
  const Limits tooSmall{1U, std::numeric_limits<std::size_t>::max(), 0U};
  checkFailure(vss::math::statistics::quantile(values.view(), 0.5,
                                               QuantileRule::linear, tooSmall),
               ErrorCode::ResourceLimit);
}

void medianCases() {
  const Vector even = vector({1.0, 2.0, 4.0, 8.0});
  const Vector nominal = vector({1.0, 2.0, 3.0, 5.0});
  checkNear(vss::math::statistics::median(even.view(), unlimited).value(), 3.0);
  checkNear(vss::math::statistics::median(nominal.view(), unlimited).value(),
            2.5);
  const Vector empty;
  checkFailure(vss::math::statistics::median(empty.view(), unlimited),
               ErrorCode::InsufficientSamples);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(vss::math::statistics::median(nonfinite.view(), unlimited),
               ErrorCode::NonFiniteInput);
  const Limits tooSmall{1U, std::numeric_limits<std::size_t>::max(), 0U};
  checkFailure(vss::math::statistics::median(even.view(), tooSmall),
               ErrorCode::ResourceLimit);
}

void trimmedMeanCases() {
  const Vector values = vector({1.0, 2.0, 4.0, 8.0});
  const Vector nominal = vector({1.0, 2.0, 3.0, 5.0});
  checkNear(vss::math::statistics::trimmedMean(values.view(), 0.25, unlimited)
                .value(),
            3.0);
  checkNear(vss::math::statistics::trimmedMean(nominal.view(), 0.25, unlimited)
                .value(),
            2.5);
  checkNear(
      vss::math::statistics::trimmedMean(values.view(), 0.0, unlimited).value(),
      3.75);
  const Vector empty;
  checkFailure(vss::math::statistics::trimmedMean(empty.view(), 0.0, unlimited),
               ErrorCode::InsufficientSamples);
  checkFailure(
      vss::math::statistics::trimmedMean(values.view(), -0.1, unlimited),
      ErrorCode::InvalidDomain);
  checkFailure(
      vss::math::statistics::trimmedMean(values.view(), 0.5, unlimited),
      ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(
      vss::math::statistics::trimmedMean(nonfinite.view(), 0.0, unlimited),
      ErrorCode::NonFiniteInput);
  checkFailure(
      vss::math::statistics::trimmedMean(nonfinite.view(), 0.5, unlimited),
      ErrorCode::InvalidDomain);
  const Limits tooSmall{1U, std::numeric_limits<std::size_t>::max(), 0U};
  checkFailure(vss::math::statistics::trimmedMean(values.view(), 0.0, tooSmall),
               ErrorCode::ResourceLimit);
}

void winsorizedMeanCases() {
  const Vector values = vector({1.0, 2.0, 4.0, 8.0});
  const Vector nominal = vector({1.0, 2.0, 3.0, 5.0});
  checkNear(
      vss::math::statistics::winsorizedMean(values.view(), 0.25, unlimited)
          .value(),
      3.0);
  checkNear(
      vss::math::statistics::winsorizedMean(nominal.view(), 0.25, unlimited)
          .value(),
      2.5);
  checkNear(vss::math::statistics::winsorizedMean(values.view(), 0.0, unlimited)
                .value(),
            3.75);
  const Vector empty;
  checkFailure(
      vss::math::statistics::winsorizedMean(empty.view(), 0.0, unlimited),
      ErrorCode::InsufficientSamples);
  checkFailure(
      vss::math::statistics::winsorizedMean(values.view(), -0.1, unlimited),
      ErrorCode::InvalidDomain);
  checkFailure(
      vss::math::statistics::winsorizedMean(values.view(), 0.5, unlimited),
      ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(
      vss::math::statistics::winsorizedMean(nonfinite.view(), 0.0, unlimited),
      ErrorCode::NonFiniteInput);
  checkFailure(
      vss::math::statistics::winsorizedMean(nonfinite.view(), 0.5, unlimited),
      ErrorCode::InvalidDomain);
  const Limits tooSmall{1U, std::numeric_limits<std::size_t>::max(), 0U};
  checkFailure(
      vss::math::statistics::winsorizedMean(values.view(), 0.0, tooSmall),
      ErrorCode::ResourceLimit);
}

void extremeFiniteValues() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector oppositeExtremes = vector({-maximum, maximum});
  checkNear(vss::math::statistics::quantile(oppositeExtremes.view(), 0.5,
                                            QuantileRule::linear, unlimited)
                .value(),
            0.0);
  checkNear(vss::math::statistics::trimmedMean(oppositeExtremes.view(), 0.0,
                                               unlimited)
                .value(),
            0.0);
  checkNear(vss::math::statistics::winsorizedMean(oppositeExtremes.view(), 0.0,
                                                  unlimited)
                .value(),
            0.0);
  const Vector repeatedMaximum = vector({maximum, maximum, maximum, maximum});
  VSS_CHECK(vss::math::statistics::trimmedMean(repeatedMaximum.view(), 0.25,
                                               unlimited)
                .value() == maximum);
  VSS_CHECK(vss::math::statistics::winsorizedMean(repeatedMaximum.view(), 0.25,
                                                  unlimited)
                .value() == maximum);
}

void geometricMeanCases() {
  const Vector values = vector({1.0, 4.0});
  checkNear(vss::math::statistics::geometricMean(values.view()).value(), 2.0);
  const Vector scaled = vector({8.0, 32.0});
  checkNear(vss::math::statistics::geometricMean(scaled.view()).value(), 16.0);
  const Vector empty;
  checkFailure(vss::math::statistics::geometricMean(empty.view()),
               ErrorCode::InsufficientSamples);
  const Vector zero = vector({0.0, 1.0});
  checkFailure(vss::math::statistics::geometricMean(zero.view()),
               ErrorCode::InvalidDomain);
  const Vector negative = vector({-1.0, 2.0});
  checkFailure(vss::math::statistics::geometricMean(negative.view()),
               ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(vss::math::statistics::geometricMean(nonfinite.view()),
               ErrorCode::NonFiniteInput);
}

void harmonicMeanCases() {
  const Vector values = vector({1.0, 3.0});
  checkNear(vss::math::statistics::harmonicMean(values.view()).value(), 1.5);
  const Vector conventional = vector({1.0, 4.0});
  checkNear(vss::math::statistics::harmonicMean(conventional.view()).value(),
            1.6);
  const Vector empty;
  checkFailure(vss::math::statistics::harmonicMean(empty.view()),
               ErrorCode::InsufficientSamples);
  const Vector zero = vector({0.0, 1.0});
  checkFailure(vss::math::statistics::harmonicMean(zero.view()),
               ErrorCode::InvalidDomain);
  const Vector negative = vector({-1.0, 2.0});
  checkFailure(vss::math::statistics::harmonicMean(negative.view()),
               ErrorCode::InvalidDomain);
  const Vector nonfinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  checkFailure(vss::math::statistics::harmonicMean(nonfinite.view()),
               ErrorCode::NonFiniteInput);
}

void positiveMeanExtremes() {
  const Scalar smallest = std::numeric_limits<Scalar>::denorm_min();
  const Scalar largest = std::numeric_limits<Scalar>::max();
  const Vector smallestOnly = vector({smallest});
  const Vector largestOnly = vector({largest});
  VSS_CHECK(vss::math::statistics::geometricMean(smallestOnly.view()).value() ==
            smallest);
  VSS_CHECK(vss::math::statistics::harmonicMean(smallestOnly.view()).value() ==
            smallest);
  VSS_CHECK(vss::math::statistics::geometricMean(largestOnly.view()).value() ==
            largest);
  VSS_CHECK(vss::math::statistics::harmonicMean(largestOnly.view()).value() ==
            largest);

  const Vector extremes = vector({smallest, largest});
  const auto geometric = vss::math::statistics::geometricMean(extremes.view());
  const auto harmonic = vss::math::statistics::harmonicMean(extremes.view());
  VSS_CHECK(geometric.hasValue() && std::isfinite(geometric.value()));
  VSS_CHECK(harmonic.hasValue() && harmonic.value() == 2.0 * smallest);
  if (geometric.hasValue()) {
    const Scalar expected = std::scalbn(1.0, -25);
    VSS_CHECK(std::abs(std::log(geometric.value()) - std::log(expected)) <=
              2.0e-12);
  }
}

} // namespace

int main() {
  publicSignatures();
  quantileCases();
  medianCases();
  trimmedMeanCases();
  winsorizedMeanCases();
  extremeFiniteValues();
  geometricMeanCases();
  harmonicMeanCases();
  positiveMeanExtremes();
  robustOrderSummaryCases();
}

// VSS-MATH-TRACEABILITY: STAT-ORDER-01
