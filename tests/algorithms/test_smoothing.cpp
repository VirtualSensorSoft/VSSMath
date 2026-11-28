#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/signal/Smoothing.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::signal;

constexpr Limits ample{
    .elements = 4096U, .bytes = 1U << 20U, .evaluations = 0U};

Vector vector(std::span<const Scalar> values) { return Vector(values); }

void quadraticSmoothingAndDerivativeReproducePolynomial() {
  const std::array<Scalar, 5U> samples{0.0, 1.0, 4.0, 9.0, 16.0};
  const Vector x = vector(samples);
  const auto smoothed = savitzkyGolay(x.view(), 3U, 2U, 0U, 0.5, ample);
  const auto derivative = savitzkyGolay(x.view(), 3U, 2U, 1U, 0.5, ample);
  VSS_CHECK(smoothed && derivative);
  for (Index i = 0U; i < samples.size(); ++i) {
    VSS_CHECK(std::abs(smoothed.value().at(i) - samples[i]) < 1.0e-11);
    VSS_CHECK(std::abs(derivative.value().at(i) -
                       4.0 * static_cast<Scalar>(i)) < 1.0e-10);
  }
}

void edgeWindowsUseFullOneSidedFits() {
  const std::array<Scalar, 4U> samples{0.0, 0.0, 3.0, 9.0};
  const Vector x = vector(samples);
  const auto smoothed = savitzkyGolay(x.view(), 3U, 1U, 0U, 1.0, ample);
  const auto derivative = savitzkyGolay(x.view(), 3U, 1U, 1U, 1.0, ample);
  VSS_CHECK(smoothed && derivative);
  VSS_CHECK(std::abs(smoothed.value().at(0U) - (-0.5)) < 1.0e-12);
  VSS_CHECK(std::abs(derivative.value().at(0U) - 1.5) < 1.0e-12);
  VSS_CHECK(std::abs(smoothed.value().at(3U) - 8.5) < 1.0e-12);
  VSS_CHECK(std::abs(derivative.value().at(3U) - 4.5) < 1.0e-12);
}

void singletonWindowIsIdentity() {
  const std::array<Scalar, 3U> samples{-2.0, 3.0, 8.0};
  const Vector x = vector(samples);
  const auto result = savitzkyGolay(x.view(), 1U, 0U, 0U, 2.0, ample);
  VSS_CHECK(result && result.value().size() == samples.size());
  for (Index i = 0U; i < samples.size(); ++i) {
    VSS_CHECK(result.value().at(i) == samples[i]);
  }
}

void higherDerivativeUsesPhysicalStepScaling() {
  const std::array<Scalar, 5U> samples{0.0, 1.0, 8.0, 27.0, 64.0};
  const Vector x = vector(samples);
  const auto second = savitzkyGolay(x.view(), 5U, 3U, 2U, 2.0, ample);
  VSS_CHECK(second);
  for (Index i = 0U; i < samples.size(); ++i) {
    VSS_CHECK(std::abs(second.value().at(i) - 1.5 * static_cast<Scalar>(i)) <
              1.0e-10);
  }
}

void invalidFiniteParametersAndInsufficientSamplesAreDistinct() {
  const std::array<Scalar, 4U> samples{0.0, 1.0, 2.0, 3.0};
  const Vector x = vector(samples);
  const auto even = savitzkyGolay(x.view(), 2U, 1U, 0U, 1.0, ample);
  const auto zeroWindow = savitzkyGolay(x.view(), 0U, 0U, 0U, 1.0, ample);
  const auto degreeNotBelowWindow =
      savitzkyGolay(x.view(), 3U, 3U, 0U, 1.0, ample);
  const auto derivativeAboveDegree =
      savitzkyGolay(x.view(), 3U, 1U, 2U, 1.0, ample);
  const auto zeroStep = savitzkyGolay(x.view(), 3U, 1U, 0U, 0.0, ample);
  const auto negativeStep = savitzkyGolay(x.view(), 3U, 1U, 0U, -1.0, ample);
  const auto noSamples = savitzkyGolay(x.view(), 5U, 2U, 0U, 1.0, ample);
  VSS_CHECK(!even && even.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!zeroWindow &&
            zeroWindow.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!degreeNotBelowWindow &&
            degreeNotBelowWindow.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!derivativeAboveDegree &&
            derivativeAboveDegree.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!zeroStep && zeroStep.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!negativeStep &&
            negativeStep.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!noSamples &&
            noSamples.status().code() == ErrorCode::InsufficientSamples);
}

void numericallyRankDeficientDesignIsRejected() {
  std::array<Scalar, 21U> samples{};
  for (Index i = 0U; i < samples.size(); ++i) {
    samples[i] = static_cast<Scalar>(i);
  }
  const Vector x = vector(samples);
  const auto result = savitzkyGolay(x.view(), 21U, 20U, 0U, 1.0, ample);
  VSS_CHECK(!result && result.status().code() == ErrorCode::RankDeficient);
}

void nonFiniteInputsAndUnrepresentableDerivativeAreReported() {
  const std::array<Scalar, 3U> samples{0.0, 1.0, 2.0};
  const Vector x = vector(samples);
  const auto nanStep = savitzkyGolay(
      x.view(), 3U, 1U, 0U, std::numeric_limits<Scalar>::quiet_NaN(), ample);
  const auto infiniteStep = savitzkyGolay(
      x.view(), 3U, 1U, 0U, std::numeric_limits<Scalar>::infinity(), ample);
  VSS_CHECK(!nanStep && nanStep.status().code() == ErrorCode::NonFiniteInput);
  VSS_CHECK(!infiniteStep &&
            infiniteStep.status().code() == ErrorCode::NonFiniteInput);

  const std::array<Scalar, 3U> nonFiniteSamples{
      0.0, std::numeric_limits<Scalar>::infinity(), 2.0};
  const Vector bad = vector(nonFiniteSamples);
  const auto badData = savitzkyGolay(bad.view(), 3U, 1U, 0U, 1.0, ample);
  VSS_CHECK(!badData && badData.status().code() == ErrorCode::NonFiniteInput);
  const std::array<Scalar, 3U> nanSamples{
      0.0, std::numeric_limits<Scalar>::quiet_NaN(), 2.0};
  const Vector nanData = vector(nanSamples);
  const auto nanResult = savitzkyGolay(nanData.view(), 3U, 1U, 0U, 1.0, ample);
  VSS_CHECK(!nanResult &&
            nanResult.status().code() == ErrorCode::NonFiniteInput);

  const auto overflow = savitzkyGolay(
      x.view(), 3U, 1U, 1U, std::numeric_limits<Scalar>::denorm_min(), ample);
  VSS_CHECK(!overflow &&
            overflow.status().code() == ErrorCode::NonFiniteResult);
}

void resourceLimitsAndStridedInputsAreHandled() {
  const std::array<Scalar, 5U> interleaved{0.0, 99.0, 1.0, 99.0, 4.0};
  const auto strided = ConstVectorView::create(interleaved, 3U, 2U);
  VSS_CHECK(strided);
  const auto smoothed = savitzkyGolay(strided.value(), 3U, 2U, 0U, 1.0, ample);
  VSS_CHECK(smoothed && smoothed.value().at(1U) == 1.0);

  const std::array<Scalar, 3U> samples{0.0, 1.0, 2.0};
  const Vector x = vector(samples);
  const Limits tiny{.elements = 1U, .bytes = 1U, .evaluations = 1U};
  const auto limited = savitzkyGolay(x.view(), 3U, 1U, 0U, 1.0, tiny);
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const Limits byteLimited{.elements = 3U, .bytes = 1U, .evaluations = 0U};
  const auto noWorkspace =
      savitzkyGolay(x.view(), 3U, 1U, 0U, 1.0, byteLimited);
  VSS_CHECK(!noWorkspace &&
            noWorkspace.status().code() == ErrorCode::ResourceLimit);
}
} // namespace

int main() try {
  quadraticSmoothingAndDerivativeReproducePolynomial();
  edgeWindowsUseFullOneSidedFits();
  singletonWindowIsIdentity();
  higherDerivativeUsesPhysicalStepScaling();
  invalidFiniteParametersAndInsufficientSamplesAreDistinct();
  numericallyRankDeficientDesignIsRejected();
  nonFiniteInputsAndUnrepresentableDerivativeAreReported();
  resourceLimitsAndStridedInputsAreHandled();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
