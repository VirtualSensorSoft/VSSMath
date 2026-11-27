#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/signal/Dependence.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::signal;

constexpr Limits ample{
    .elements = 4096U, .bytes = 1U << 20U, .evaluations = 0U};

template <class Enum> Enum invalidEnumValue() {
  volatile const std::uint8_t rawValue = 255U;
  // Deliberately construct an out-of-range value for invalid-enumerator tests.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(rawValue);
}

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>{values.begin(), values.size()});
}

void autocovariancePreservesLagOrderAndNormalization() {
  const Vector samples = vector({0.0, 1.0, 2.0});
  const std::array<Index, 4U> lags{2U, 0U, 2U, 1U};
  const auto full =
      autocovariance(samples.view(), lags, LagNormalization::full_count, ample);
  VSS_CHECK(full && full.value().lags().size() == lags.size());
  const std::array<std::int64_t, 4U> expectedLags{2, 0, 2, 1};
  const std::array<Index, 4U> expectedCounts{1U, 3U, 1U, 2U};
  const std::array<Scalar, 4U> expectedFull{-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0,
                                            0.0};
  VSS_CHECK(std::ranges::equal(full.value().lags(), expectedLags));
  VSS_CHECK(std::ranges::equal(full.value().counts(), expectedCounts));
  for (Index i = 0U; i < expectedFull.size(); ++i) {
    VSS_CHECK(std::abs(full.value().values().at(i) - expectedFull.at(i)) <
              1.0e-12);
  }

  const auto overlap = autocovariance(samples.view(), lags,
                                      LagNormalization::overlap_count, ample);
  const std::array<Scalar, 4U> expectedOverlap{-1.0, 2.0 / 3.0, -1.0, 0.0};
  VSS_CHECK(overlap);
  for (Index i = 0U; i < expectedOverlap.size(); ++i) {
    VSS_CHECK(std::abs(overlap.value().values().at(i) - expectedOverlap.at(i)) <
              1.0e-12);
  }

  const std::array<Scalar, 5U> interleaved{0.0, 99.0, 1.0, 99.0, 2.0};
  const auto strided = ConstVectorView::create(interleaved, 3U, 2U);
  VSS_CHECK(strided);
  const auto stridedCovariance =
      autocovariance(strided.value(), std::array<Index, 1U>{0U},
                     LagNormalization::full_count, ample);
  VSS_CHECK(stridedCovariance &&
            std::abs(stridedCovariance.value().values().at(0U) - (2.0 / 3.0)) <
                1.0e-12);

  constexpr Scalar largeOffset = 1.0e16;
  const Vector offsetSamples =
      vector({largeOffset, largeOffset + 2.0, largeOffset + 4.0});
  const auto offsetCovariance =
      autocovariance(offsetSamples.view(), std::array<Index, 1U>{0U},
                     LagNormalization::full_count, ample);
  VSS_CHECK(offsetCovariance &&
            std::abs(offsetCovariance.value().values().at(0U) - (8.0 / 3.0)) <
                1.0e-12);
}

void autocorrelationUsesFullVarianceAndAllowsOverlapAboveOne() {
  const Vector samples = vector({0.0, 1.0, 2.0});
  const std::array<Index, 3U> lags{0U, 1U, 2U};
  const auto overlap = autocorrelation(samples.view(), lags,
                                       LagNormalization::overlap_count, ample);
  VSS_CHECK(overlap);
  const std::array<Scalar, 3U> expected{1.0, 0.0, -1.5};
  for (Index i = 0U; i < expected.size(); ++i) {
    VSS_CHECK(std::abs(overlap.value().values().at(i) - expected.at(i)) <
              1.0e-12);
  }

  const std::array<Index, 0U> noLags{};
  const auto empty = autocovariance(samples.view(), noLags,
                                    LagNormalization::full_count, ample);
  VSS_CHECK(empty && empty.value().lags().empty() &&
            empty.value().values().empty() && empty.value().counts().empty());
}

void dependenceRejectsInvalidInputsAndResourceRequests() {
  const Vector samples = vector({0.0, 1.0, 2.0});
  const std::array<Index, 1U> outside{3U};
  const auto badLag = autocovariance(samples.view(), outside,
                                     LagNormalization::full_count, ample);
  VSS_CHECK(!badLag && badLag.status().code() == ErrorCode::IndexOutOfRange);

  const std::array<Index, 1U> lag{0U};
  const auto badMode = autocovariance(
      samples.view(), lag, invalidEnumValue<LagNormalization>(), ample);
  VSS_CHECK(!badMode && badMode.status().code() == ErrorCode::InvalidDomain);

  const Vector empty;
  const auto noSamples =
      autocovariance(empty.view(), lag, LagNormalization::full_count, ample);
  VSS_CHECK(!noSamples &&
            noSamples.status().code() == ErrorCode::InsufficientSamples);

  const Vector nonFinite =
      vector({0.0, std::numeric_limits<Scalar>::infinity()});
  const auto badValue = autocovariance(nonFinite.view(), lag,
                                       LagNormalization::full_count, ample);
  VSS_CHECK(!badValue && badValue.status().code() == ErrorCode::NonFiniteInput);

  const auto limited =
      autocovariance(samples.view(), std::array<Index, 2U>{0U, 1U},
                     LagNormalization::full_count,
                     {.elements = 5U, .bytes = 1U << 20U, .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const auto byteLimited =
      autocovariance(samples.view(), lag, LagNormalization::full_count,
                     {.elements = 4096U,
                      .bytes = (3U * sizeof(std::int64_t)) - 1U,
                      .evaluations = 0U});
  VSS_CHECK(!byteLimited &&
            byteLimited.status().code() == ErrorCode::ResourceLimit);
}

void dependenceHandlesZeroVarianceAndUnrepresentableCovariance() {
  const Vector constant = vector({4.0, 4.0, 4.0});
  const std::array<Index, 2U> lags{0U, 1U};
  const auto covariance = autocovariance(
      constant.view(), lags, LagNormalization::overlap_count, ample);
  VSS_CHECK(covariance && covariance.value().values().at(0U) == 0.0 &&
            covariance.value().values().at(1U) == 0.0);
  const auto undefinedCorrelation = autocorrelation(
      constant.view(), lags, LagNormalization::full_count, ample);
  VSS_CHECK(!undefinedCorrelation &&
            undefinedCorrelation.status().code() == ErrorCode::InvalidDomain);

  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector extreme = vector({-maximum, maximum});
  const std::array<Index, 1U> lagZero{0U};
  const auto overflowingCovariance = autocovariance(
      extreme.view(), lagZero, LagNormalization::full_count, ample);
  VSS_CHECK(!overflowingCovariance && overflowingCovariance.status().code() ==
                                          ErrorCode::NonFiniteResult);
  const std::array<Index, 2U> extremeLags{0U, 1U};
  const auto finiteCorrelation = autocorrelation(
      extreme.view(), extremeLags, LagNormalization::overlap_count, ample);
  VSS_CHECK(finiteCorrelation &&
            finiteCorrelation.value().values().at(0U) == 1.0 &&
            finiteCorrelation.value().values().at(1U) == -1.0);
}

} // namespace

int main() try {
  autocovariancePreservesLagOrderAndNormalization();
  autocorrelationUsesFullVarianceAndAllowsOverlapAboveOne();
  dependenceRejectsInvalidInputsAndResourceRequests();
  dependenceHandlesZeroVarianceAndUnrepresentableCovariance();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
