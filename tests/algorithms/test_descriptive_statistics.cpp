#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <detail/NumericalClassification.hpp>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Descriptive.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Scalar;
using vss::math::Vector;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void checkNear(Scalar actual, Scalar expected, Scalar absolute = 1.0e-12,
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
  using StatisticSignature =
      vss::math::Result<Scalar> (*)(vss::math::ConstVectorView);
  static_assert(std::is_same_v<decltype(static_cast<StatisticSignature>(
                                   &vss::math::statistics::mean)),
                               StatisticSignature>);
  static_assert(std::is_same_v<decltype(static_cast<StatisticSignature>(
                                   &vss::math::statistics::populationVariance)),
                               StatisticSignature>);
  static_assert(std::is_same_v<decltype(static_cast<StatisticSignature>(
                                   &vss::math::statistics::sampleVariance)),
                               StatisticSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<StatisticSignature>(
                         &vss::math::statistics::populationStandardDeviation)),
                     StatisticSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<StatisticSignature>(
                         &vss::math::statistics::sampleStandardDeviation)),
                     StatisticSignature>);
  static_assert(std::is_same_v<decltype(static_cast<StatisticSignature>(
                                   &vss::math::statistics::rootMeanSquare)),
                               StatisticSignature>);
}

void ordinaryFiniteSample() {
  const Vector values = vector({1.0, 2.0, 3.0, 4.0});
  checkNear(vss::math::statistics::mean(values.view()).value(), 2.5);
  checkNear(vss::math::statistics::populationVariance(values.view()).value(),
            1.25);
  checkNear(vss::math::statistics::sampleVariance(values.view()).value(),
            5.0 / 3.0);
  checkNear(
      vss::math::statistics::populationStandardDeviation(values.view()).value(),
      std::sqrt(1.25));
  checkNear(
      vss::math::statistics::sampleStandardDeviation(values.view()).value(),
      std::sqrt(5.0 / 3.0));
  checkNear(vss::math::statistics::rootMeanSquare(values.view()).value(),
            std::sqrt(7.5));
}

void offsetStability() {
  const Vector values =
      vector({1000000000001.0, 1000000000002.0, 1000000000003.0});
  checkNear(vss::math::statistics::mean(values.view()).value(), 1000000000002.0,
            1.0e-9, 1.0e-15);
  checkNear(vss::math::statistics::populationVariance(values.view()).value(),
            2.0 / 3.0);
  checkNear(vss::math::statistics::sampleVariance(values.view()).value(), 1.0);
}

void rootMeanSquare() {
  const Vector values = vector({3.0, 4.0});
  checkNear(vss::math::statistics::rootMeanSquare(values.view()).value(),
            std::sqrt(12.5));
}

void emptyInput() {
  const Vector values;
  checkFailure(vss::math::statistics::mean(values.view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::statistics::populationVariance(values.view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::statistics::sampleVariance(values.view()),
               ErrorCode::InvalidExtent);
  checkFailure(
      vss::math::statistics::populationStandardDeviation(values.view()),
      ErrorCode::InvalidExtent);
  checkFailure(vss::math::statistics::sampleStandardDeviation(values.view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::statistics::rootMeanSquare(values.view()),
               ErrorCode::InvalidExtent);
}

void oneElementSample() {
  const Vector values = vector({7.0});
  checkNear(vss::math::statistics::mean(values.view()).value(), 7.0);
  checkNear(vss::math::statistics::populationVariance(values.view()).value(),
            0.0);
  checkFailure(vss::math::statistics::sampleVariance(values.view()),
               ErrorCode::InsufficientSamples);
  checkNear(
      vss::math::statistics::populationStandardDeviation(values.view()).value(),
      0.0);
  checkFailure(vss::math::statistics::sampleStandardDeviation(values.view()),
               ErrorCode::InsufficientSamples);
  checkNear(vss::math::statistics::rootMeanSquare(values.view()).value(), 7.0);
}

void nonFiniteInputOrder() {
  const Vector values = vector({1.0, std::numeric_limits<Scalar>::quiet_NaN(),
                                std::numeric_limits<Scalar>::infinity()});
  const auto result = vss::math::statistics::mean(values.view());
  checkFailure(result, ErrorCode::NonFiniteInput);
  VSS_CHECK(result.status().message().find("index 1") !=
            std::string_view::npos);
}

void negativeRoundoffBoundary() {
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  const auto accepted =
      vss::math::detail::classifyVarianceRoundoff(16.0 * epsilon, 1.0);
  const auto rejected =
      vss::math::detail::classifyVarianceRoundoff(64.0 * epsilon, 1.0);
  VSS_CHECK(accepted == ErrorCode::None);
  VSS_CHECK(rejected == ErrorCode::NonFiniteResult);
}

void largeFiniteVarianceDoesNotOverflow() {
  std::vector<Scalar> storage(4096U, 0.0);
  storage.front() = 1.0e155;
  const auto view = vss::math::ConstVectorView::create(storage, storage.size());
  VSS_CHECK(view.hasValue());
  if (!view.hasValue()) {
    return;
  }
  const auto population =
      vss::math::statistics::populationVariance(view.value());
  const auto sample = vss::math::statistics::sampleVariance(view.value());
  VSS_CHECK(population.hasValue());
  VSS_CHECK(sample.hasValue());
  if (population.hasValue() && sample.hasValue()) {
    VSS_CHECK(std::isfinite(population.value()));
    VSS_CHECK(std::isfinite(sample.value()));
    checkNear(population.value(), 2.440810203552246e306, 1.0e294, 1.0e-12);
    checkNear(sample.value(), 2.44140625e306, 1.0e294, 1.0e-12);
  }
}

} // namespace

int main() {
  publicContract();
  ordinaryFiniteSample();
  offsetStability();
  rootMeanSquare();
  emptyInput();
  oneElementSample();
  nonFiniteInputOrder();
  negativeRoundoffBoundary();
  largeFiniteVarianceDoesNotOverflow();
}

// VSS-MATH-TRACEABILITY: STA-01 STA-02 STA-03 STA-04 STA-05 STA-06
// VSS-MATH-TRACEABILITY: SUP-STA-01 SUP-STA-02
