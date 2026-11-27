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
#include <vss/math/linear/SolvePolicy.hpp>
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

SolvePolicy solvePolicy(Index maximumIterations = 64U) {
  SolvePolicy result{};
  result.iteration.max_iterations = maximumIterations;
  return result;
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

void crossDependencePreservesSignedLagOrientationAndCounts() {
  const Vector x = vector({1.0, 0.0, 0.0});
  const Vector y = vector({0.0, 1.0, 0.0});
  const std::array<std::int64_t, 4U> lags{1, -1, 1, 0};
  const auto result = crossCorrelation(x.view(), y.view(), lags,
                                       LagNormalization::full_count, ample);
  VSS_CHECK(result);
  const std::array<std::int64_t, 4U> expectedLags{1, -1, 1, 0};
  const std::array<Index, 4U> expectedCounts{2U, 2U, 2U, 3U};
  const std::array<Scalar, 4U> expectedValues{-1.0 / 6.0, 5.0 / 6.0, -1.0 / 6.0,
                                              -0.5};
  VSS_CHECK(std::ranges::equal(result.value().lags(), expectedLags));
  VSS_CHECK(std::ranges::equal(result.value().counts(), expectedCounts));
  for (Index i = 0U; i < expectedValues.size(); ++i) {
    VSS_CHECK(std::abs(result.value().values().at(i) - expectedValues.at(i)) <
              1.0e-12);
  }

  const std::array<std::int64_t, 3U> signedLags{2, 0, -2};
  const Vector scaleY = vector({0.0, 2.0, 4.0});
  const auto covariance =
      crossCovariance(vector({0.0, 1.0, 2.0}).view(), scaleY.view(), signedLags,
                      LagNormalization::full_count, ample);
  VSS_CHECK(covariance);
  const std::array<Scalar, 3U> expectedCovariance{-2.0 / 3.0, 4.0 / 3.0,
                                                  -2.0 / 3.0};
  for (Index i = 0U; i < expectedCovariance.size(); ++i) {
    VSS_CHECK(std::abs(covariance.value().values().at(i) -
                       expectedCovariance.at(i)) < 1.0e-12);
  }
  const auto overlapCovariance =
      crossCovariance(vector({0.0, 1.0, 2.0}).view(), scaleY.view(), signedLags,
                      LagNormalization::overlap_count, ample);
  VSS_CHECK(overlapCovariance);
  const std::array<Scalar, 3U> expectedOverlapCovariance{-2.0, 4.0 / 3.0, -2.0};
  for (Index i = 0U; i < expectedOverlapCovariance.size(); ++i) {
    VSS_CHECK(std::abs(overlapCovariance.value().values().at(i) -
                       expectedOverlapCovariance.at(i)) < 1.0e-12);
  }

  const Vector base = vector({0.0, 1.0, 2.0});
  const Vector doubled = vector({0.0, 2.0, 4.0});
  const std::array<std::int64_t, 1U> zeroLag{0};
  const auto perfectlyCorrelated =
      crossCorrelation(base.view(), doubled.view(), zeroLag,
                       LagNormalization::full_count, ample);
  VSS_CHECK(perfectlyCorrelated &&
            std::abs(perfectlyCorrelated.value().values().at(0U) - 1.0) <
                1.0e-12);

  const std::array<std::int64_t, 5U> allLags{-2, -1, 0, 1, 2};
  const std::array<std::int64_t, 5U> reverseLags{2, 1, 0, -1, -2};
  const auto xy = crossCorrelation(base.view(), doubled.view(), allLags,
                                   LagNormalization::full_count, ample);
  const auto yx = crossCorrelation(doubled.view(), base.view(), reverseLags,
                                   LagNormalization::full_count, ample);
  VSS_CHECK(xy && yx);
  for (Index i = 0U; i < allLags.size(); ++i) {
    VSS_CHECK(std::abs(xy.value().values().at(i) -
                       yx.value().values().at(allLags.size() - 1U - i)) <
              1.0e-12);
  }
  const auto xyCovariance =
      crossCovariance(base.view(), doubled.view(), allLags,
                      LagNormalization::full_count, ample);
  const auto yxCovariance =
      crossCovariance(doubled.view(), base.view(), reverseLags,
                      LagNormalization::full_count, ample);
  VSS_CHECK(xyCovariance && yxCovariance);
  for (Index i = 0U; i < allLags.size(); ++i) {
    VSS_CHECK(std::abs(xyCovariance.value().values().at(i) -
                       yxCovariance.value().values().at(allLags.size() - 1U -
                                                        i)) < 1.0e-12);
  }

  constexpr Scalar largeOffset = 1.0e16;
  const Vector offsetX =
      vector({largeOffset, largeOffset + 2.0, largeOffset + 4.0});
  const Vector offsetY =
      vector({-largeOffset, -largeOffset + 4.0, -largeOffset + 8.0});
  const auto offsetCovariance =
      crossCovariance(offsetX.view(), offsetY.view(), zeroLag,
                      LagNormalization::full_count, ample);
  VSS_CHECK(offsetCovariance &&
            std::abs(offsetCovariance.value().values().at(0U) - 16.0 / 3.0) <
                1.0e-12);
}

void crossDependenceRejectsMismatchesAndUnusableLags() {
  const Vector x = vector({0.0, 1.0, 2.0});
  const Vector y = vector({0.0, 1.0});
  const std::array<std::int64_t, 1U> lag{0};
  const auto mismatched = crossCovariance(x.view(), y.view(), lag,
                                          LagNormalization::full_count, ample);
  VSS_CHECK(!mismatched &&
            mismatched.status().code() == ErrorCode::DimensionMismatch);

  const Vector empty;
  const auto noSamples = crossCovariance(empty.view(), empty.view(), lag,
                                         LagNormalization::full_count, ample);
  VSS_CHECK(!noSamples &&
            noSamples.status().code() == ErrorCode::InsufficientSamples);

  const std::array<std::int64_t, 2U> invalidLags{3, -3};
  const auto outside = crossCovariance(x.view(), x.view(), invalidLags,
                                       LagNormalization::full_count, ample);
  VSS_CHECK(!outside && outside.status().code() == ErrorCode::IndexOutOfRange);
  const std::array<std::int64_t, 1U> minimumLag{
      std::numeric_limits<std::int64_t>::min()};
  const auto minimumOutside = crossCovariance(
      x.view(), x.view(), minimumLag, LagNormalization::full_count, ample);
  VSS_CHECK(!minimumOutside &&
            minimumOutside.status().code() == ErrorCode::IndexOutOfRange);

  const Vector nonFinite =
      vector({0.0, std::numeric_limits<Scalar>::quiet_NaN(), 2.0});
  const auto invalidSamples = crossCorrelation(
      x.view(), nonFinite.view(), lag, LagNormalization::full_count, ample);
  VSS_CHECK(!invalidSamples &&
            invalidSamples.status().code() == ErrorCode::NonFiniteInput);

  const Vector constant = vector({4.0, 4.0, 4.0});
  const auto zeroVariance = crossCorrelation(
      x.view(), constant.view(), lag, LagNormalization::full_count, ample);
  VSS_CHECK(!zeroVariance &&
            zeroVariance.status().code() == ErrorCode::InvalidDomain);

  const auto invalidPolicy = crossCovariance(
      x.view(), x.view(), lag, invalidEnumValue<LagNormalization>(), ample);
  VSS_CHECK(!invalidPolicy &&
            invalidPolicy.status().code() == ErrorCode::InvalidDomain);

  const auto limitedElements =
      crossCovariance(x.view(), x.view(), lag, LagNormalization::full_count,
                      {.elements = 2U, .bytes = 1U << 20U, .evaluations = 0U});
  VSS_CHECK(!limitedElements &&
            limitedElements.status().code() == ErrorCode::ResourceLimit);
  const auto limitedBytes =
      crossCovariance(x.view(), x.view(), lag, LagNormalization::full_count,
                      {.elements = 4096U,
                       .bytes = (3U * sizeof(std::int64_t)) - 1U,
                       .evaluations = 0U});
  VSS_CHECK(!limitedBytes &&
            limitedBytes.status().code() == ErrorCode::ResourceLimit);
}

void crossDependenceScalesExtremeFiniteInputs() {
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector x = vector({-maximum, maximum});
  const Vector y = vector({maximum, -maximum});
  const std::array<std::int64_t, 3U> lags{0, 1, -1};
  const auto covariance =
      crossCovariance(x.view(), y.view(), std::array<std::int64_t, 1U>{0},
                      LagNormalization::full_count, ample);
  VSS_CHECK(!covariance &&
            covariance.status().code() == ErrorCode::NonFiniteResult);

  const auto correlation = crossCorrelation(
      x.view(), y.view(), lags, LagNormalization::full_count, ample);
  VSS_CHECK(correlation);
  const std::array<Scalar, 3U> expected{-1.0, 0.5, 0.5};
  for (Index i = 0U; i < expected.size(); ++i) {
    VSS_CHECK(std::abs(correlation.value().values().at(i) - expected.at(i)) <
              1.0e-12);
  }
}

void partialAutocorrelationMatchesLevinsonDurbinOracles() {
  const Vector ar1 = vector({1.0, 0.5, 0.25});
  const auto ar1Result =
      partialAutocorrelation(ar1.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(ar1Result && ar1Result.value().size() == 2U);
  VSS_CHECK(std::abs(ar1Result.value().at(0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(ar1Result.value().at(1U)) < 1.0e-12);

  // These normalized autocovariances come from an AR(2) with coefficients
  // (0.6, -0.2): gamma1/gamma0=0.5 and gamma2/gamma0=0.1. Independently solve
  // its order-two Yule-Walker system by the closed-form 2x2 determinant.
  const Vector ar2 = vector({1.0, 0.5, 0.1});
  const Scalar determinant = 1.0 - 0.5 * 0.5;
  const Scalar expectedSecond = (0.1 - 0.5 * 0.5) / determinant;
  const Scalar expectedFirst = 0.5 * (1.0 - expectedSecond);
  const auto ar2Result =
      partialAutocorrelation(ar2.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(ar2Result && ar2Result.value().size() == 2U);
  VSS_CHECK(std::abs(ar2Result.value().at(0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(ar2Result.value().at(1U) - expectedSecond) < 1.0e-12);
  VSS_CHECK(std::abs(expectedFirst - 0.6) < 1.0e-12);

  // A second independent AR(3) fixture exercises the simultaneous in-place
  // coefficient update and the third reflection coefficient.
  const Vector ar3 = vector({1.0, 0.5, 0.1, 0.032});
  const auto ar3Result =
      partialAutocorrelation(ar3.view(), 3U, solvePolicy(), ample);
  VSS_CHECK(ar3Result && ar3Result.value().size() == 3U);
  VSS_CHECK(std::abs(ar3Result.value().at(0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(ar3Result.value().at(1U) + 0.2) < 1.0e-12);
  VSS_CHECK(std::abs(ar3Result.value().at(2U) - 0.1) < 1.0e-12);

  const Vector extremeScale = vector({1.0e300, 5.0e299, 2.5e299});
  const auto scaledResult =
      partialAutocorrelation(extremeScale.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(scaledResult &&
            std::abs(scaledResult.value().at(0U) - 0.5) < 1.0e-12 &&
            std::abs(scaledResult.value().at(1U)) < 1.0e-12);

  const Vector extraLag =
      vector({1.0, 0.5, 0.25, std::numeric_limits<Scalar>::quiet_NaN()});
  const auto ignoresExtra =
      partialAutocorrelation(extraLag.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(ignoresExtra &&
            std::abs(ignoresExtra.value().at(0U) - 0.5) < 1.0e-12 &&
            std::abs(ignoresExtra.value().at(1U)) < 1.0e-12);

  const Vector zeroOrderInput = vector({2.0});
  auto zeroOrderPolicy = solvePolicy(0U);
  zeroOrderPolicy.rank_abs = 3.0;
  const auto zeroOrder =
      partialAutocorrelation(zeroOrderInput.view(), 0U, zeroOrderPolicy, ample);
  VSS_CHECK(zeroOrder && zeroOrder.value().empty());
}

void partialAutocorrelationRejectsInvalidAndUnstableInputs() {
  const Vector empty;
  const auto noLagZero =
      partialAutocorrelation(empty.view(), 0U, solvePolicy(), ample);
  VSS_CHECK(!noLagZero &&
            noLagZero.status().code() == ErrorCode::InsufficientSamples);

  const Vector tooFew = vector({1.0, 0.5});
  const auto insufficient =
      partialAutocorrelation(tooFew.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(!insufficient &&
            insufficient.status().code() == ErrorCode::InsufficientSamples);

  const Vector nonFinite =
      vector({1.0, 0.5, std::numeric_limits<Scalar>::infinity()});
  const auto invalidValue =
      partialAutocorrelation(nonFinite.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(!invalidValue &&
            invalidValue.status().code() == ErrorCode::NonFiniteInput);

  const Vector zeroVariance = vector({0.0, 0.0});
  const auto invalidVariance =
      partialAutocorrelation(zeroVariance.view(), 1U, solvePolicy(), ample);
  VSS_CHECK(!invalidVariance &&
            invalidVariance.status().code() == ErrorCode::InvalidDomain);
  const Vector negativeVariance = vector({-1.0, 0.0});
  const auto invalidNegativeVariance =
      partialAutocorrelation(negativeVariance.view(), 1U, solvePolicy(), ample);
  VSS_CHECK(!invalidNegativeVariance &&
            invalidNegativeVariance.status().code() ==
                ErrorCode::InvalidDomain);

  const Vector singular = vector({1.0, 1.0});
  const auto rankDeficient =
      partialAutocorrelation(singular.view(), 1U, solvePolicy(), ample);
  VSS_CHECK(!rankDeficient &&
            rankDeficient.status().code() == ErrorCode::RankDeficient);
  auto permissiveRankPolicy = solvePolicy();
  permissiveRankPolicy.rank = RankPolicy::minimum_norm;
  const auto stillRankDeficient =
      partialAutocorrelation(singular.view(), 1U, permissiveRankPolicy, ample);
  VSS_CHECK(!stillRankDeficient &&
            stillRankDeficient.status().code() == ErrorCode::RankDeficient);

  const Vector indefinite = vector({1.0, 2.0});
  const auto notPositiveSemidefinite =
      partialAutocorrelation(indefinite.view(), 1U, solvePolicy(), ample);
  VSS_CHECK(!notPositiveSemidefinite &&
            notPositiveSemidefinite.status().code() ==
                ErrorCode::NotPositiveSemidefinite);

  const Vector indefiniteToeplitz = vector({1.0, 0.9, -0.9});
  const auto negativeInnovation = partialAutocorrelation(
      indefiniteToeplitz.view(), 2U, solvePolicy(), ample);
  VSS_CHECK(!negativeInnovation && negativeInnovation.status().code() ==
                                       ErrorCode::NotPositiveSemidefinite);

  const Vector illConditioned = vector({1.0, 0.95});
  auto conditionedPolicy = solvePolicy();
  conditionedPolicy.reciprocal_condition_floor = 0.1;
  const auto belowConditionFloor = partialAutocorrelation(
      illConditioned.view(), 1U, conditionedPolicy, ample);
  VSS_CHECK(!belowConditionFloor &&
            belowConditionFloor.status().code() == ErrorCode::IllConditioned);

  auto rankLimitedPolicy = solvePolicy();
  rankLimitedPolicy.rank_abs = 0.1;
  const auto belowRankThreshold = partialAutocorrelation(
      illConditioned.view(), 1U, rankLimitedPolicy, ample);
  VSS_CHECK(!belowRankThreshold &&
            belowRankThreshold.status().code() == ErrorCode::RankDeficient);
  auto relativeRankPolicy = solvePolicy();
  relativeRankPolicy.rank_rel = 0.1;
  const auto belowRelativeRankThreshold = partialAutocorrelation(
      illConditioned.view(), 1U, relativeRankPolicy, ample);
  VSS_CHECK(!belowRelativeRankThreshold &&
            belowRelativeRankThreshold.status().code() ==
                ErrorCode::RankDeficient);

  auto invalidPolicy = solvePolicy();
  invalidPolicy.rank_rel = 1.0;
  const auto rejectedPolicy =
      partialAutocorrelation(tooFew.view(), 1U, invalidPolicy, ample);
  VSS_CHECK(!rejectedPolicy &&
            rejectedPolicy.status().code() == ErrorCode::InvalidTolerance);

  auto invalidRankPolicy = solvePolicy();
  invalidRankPolicy.rank = invalidEnumValue<RankPolicy>();
  const auto rejectedRankPolicy =
      partialAutocorrelation(singular.view(), 1U, invalidRankPolicy, ample);
  VSS_CHECK(!rejectedRankPolicy &&
            rejectedRankPolicy.status().code() == ErrorCode::InvalidDomain);

  const Vector ar1 = vector({1.0, 0.5, 0.25});
  const auto insufficientIterations =
      partialAutocorrelation(ar1.view(), 2U, solvePolicy(1U), ample);
  VSS_CHECK(!insufficientIterations &&
            insufficientIterations.status().code() == ErrorCode::NoConvergence);

  const auto limited = partialAutocorrelation(
      ar1.view(), 1U, solvePolicy(),
      {.elements = 2U, .bytes = 1U << 20U, .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const auto byteLimited =
      partialAutocorrelation(ar1.view(), 1U, solvePolicy(),
                             {.elements = 64U, .bytes = 1U, .evaluations = 0U});
  VSS_CHECK(!byteLimited &&
            byteLimited.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() try {
  autocovariancePreservesLagOrderAndNormalization();
  autocorrelationUsesFullVarianceAndAllowsOverlapAboveOne();
  dependenceRejectsInvalidInputsAndResourceRequests();
  dependenceHandlesZeroVarianceAndUnrepresentableCovariance();
  crossDependencePreservesSignedLagOrientationAndCounts();
  crossDependenceRejectsMismatchesAndUnusableLags();
  crossDependenceScalesExtremeFiniteInputs();
  partialAutocorrelationMatchesLevinsonDurbinOracles();
  partialAutocorrelationRejectsInvalidAndUnstableInputs();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
