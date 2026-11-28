#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/TensorViews.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/signal/LaggedMoments.hpp>
#include <vss/math/signal/Series.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::signal;
using namespace vss::math::statistics;

constexpr Limits ample{
    .elements = 4096U, .bytes = 1U << 20U, .evaluations = 0U};

Matrix column(const std::array<Scalar, 4U>& values) {
  Matrix result(values.size(), 1U);
  for (Index row = 0U; row < values.size(); ++row) {
    result.at(row, 0U) = values.at(row);
  }
  return result;
}

MomentRequest selectedCovariance(std::span<const IndexList> tuples) {
  return {.orders = 0b0010U,
          .representation = MomentRepresentation::selected,
          .selected = tuples,
          .contraction = std::nullopt};
}

MomentRequest selectedOrder(std::uint8_t orderMask,
                            std::span<const IndexList> tuples) {
  return {.orders = orderMask,
          .representation = MomentRepresentation::selected,
          .selected = tuples,
          .contraction = std::nullopt};
}

LagMomentSpec lagSpec(std::span<const std::int64_t> lags,
                      std::span<const IndexList> components, CenterRule center,
                      const MomentRequest& request) {
  return {.lags = lags,
          .components = components,
          .center = center,
          .request = request};
}

void zeroLagMatchesBatchCrossMoment() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix x = column({1.0, 2.0, 3.0, 4.0});
  const Matrix y = column({-1.0, 0.0, 1.0, 2.0});
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView right{times.view(), y.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> lags{0};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto result = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::estimated, request), ample);
  VSS_CHECK(result);
  VSS_CHECK(result.value().terms.view().size() == 1U);
  VSS_CHECK(std::abs(result.value().terms.view()[0U].values.at(0U) - 1.25) <
            1.0e-12);
  VSS_CHECK(result.value().diagnostics.global.valid == 4U);
}

void positiveLagUsesMatchedPairsAndWholeRowMask() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix x = column({1.0, 2.0, 3.0, 4.0});
  const Matrix y = column({-1.0, 0.0, 1.0, 2.0});
  const std::array<std::uint8_t, 4U> mask{1U, 0U, 1U, 1U};
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView right{times.view(), y.view(), mask, 1.0};
  const std::array<std::int64_t, 1U> lags{1};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto result = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::estimated, request), ample);
  VSS_CHECK(result);
  VSS_CHECK(std::abs(result.value().terms.view()[0U].values.at(0U) - 1.0) <
            1.0e-12);
  VSS_CHECK(result.value().diagnostics.global.input == 3U);
  VSS_CHECK(result.value().diagnostics.global.valid == 2U);
}

void negativeLagPreservesSignedOrientation() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix x = column({0.0, 1.0, 4.0, 10.0});
  const Matrix y = column({0.0, 3.0, 10.0, 2.0});
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView right{times.view(), y.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> lags{-1};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto result = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::estimated, request), ample);
  VSS_CHECK(result);
  VSS_CHECK(std::abs(result.value().terms.view()[0U].values.at(0U) +
                     (7.0 / 3.0)) < 1.0e-12);
}

void knownCenterUsesZeroAndReturnsRawCrossMoment() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix x = column({1.0, 2.0, 3.0, 4.0});
  const Matrix y = column({-1.0, 0.0, 1.0, 2.0});
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView right{times.view(), y.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> lags{0};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto result = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::known, request), ample);
  VSS_CHECK(result);
  VSS_CHECK(std::abs(result.value().terms.view()[0U].values.at(0U) - 2.5) <
            1.0e-12);
  VSS_CHECK(result.value().diagnostics.center == CenterRule::known);
}

void knownCenterSupportsRepeatedFourthOrderTuples() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix values = column({1.0, 2.0, 3.0, 4.0});
  const SeriesView series{times.view(), values.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> lags{0};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> components{IndexList{pair}};
  const std::array<Index, 4U> tuple{0U, 0U, 1U, 1U};
  const std::array<IndexList, 1U> selected{IndexList{tuple}};
  const auto result =
      laggedHigherMoments(series, series,
                          lagSpec(lags, components, CenterRule::known,
                                  selectedOrder(0b1000U, selected)),
                          ample);
  VSS_CHECK(result);
  VSS_CHECK(std::abs(result.value().terms.view()[0U].values.at(0U) - 88.5) <
            1.0e-12);
}

void constantCenteredMomentsAndMultipleLagsAreExact() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Matrix constantX = column({4.0, 4.0, 4.0, 4.0});
  const Matrix constantY = column({-2.0, -2.0, -2.0, -2.0});
  const SeriesView x{times.view(), constantX.view(), {}, 1.0};
  const SeriesView y{times.view(), constantY.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> zeroLag{0};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> pairComponents{IndexList{pair}};
  const MomentRequest allOrders{.orders = 0b1111U,
                                .representation =
                                    MomentRepresentation::component,
                                .selected = {},
                                .contraction = std::nullopt};
  const auto constant = laggedHigherMoments(
      x, y, lagSpec(zeroLag, pairComponents, CenterRule::estimated, allOrders),
      ample);
  VSS_CHECK(constant);
  for (const auto& term : constant.value().terms.view()) {
    for (Index i = 0U; i < term.values.size(); ++i) {
      VSS_CHECK(term.values.at(i) == 0.0);
    }
  }

  const Matrix ramp = column({1.0, 2.0, 3.0, 4.0});
  const SeriesView rampSeries{times.view(), ramp.view(), {}, 1.0};
  const std::array<std::int64_t, 2U> lags{0, 1};
  const std::array<IndexList, 2U> components{IndexList{pair}, IndexList{pair}};
  const std::array<Index, 2U> crossLagTuple{0U, 2U};
  const std::array<IndexList, 1U> selected{IndexList{crossLagTuple}};
  const auto multiLag =
      laggedHigherMoments(rampSeries, rampSeries,
                          lagSpec(lags, components, CenterRule::estimated,
                                  selectedCovariance(selected)),
                          ample);
  VSS_CHECK(multiLag);
  VSS_CHECK(multiLag.value().diagnostics.global.input == 3U);
  VSS_CHECK(multiLag.value().diagnostics.global.valid == 3U);
  VSS_CHECK(std::abs(multiLag.value().terms.view()[0U].values.at(0U) -
                     (2.0 / 3.0)) < 1.0e-12);
}

void rejectsMisalignedAxesMasksComponentsAndUnsupportedLag() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  const Vector differentTimes{std::array<Scalar, 4U>{0.0, 1.0, 2.1, 3.0}};
  const Matrix x = column({1.0, 2.0, 3.0, 4.0});
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView misaligned{differentTimes.view(), x.view(), {}, 1.0};
  const std::array<std::int64_t, 1U> zero{0};
  const std::array<std::int64_t, 1U> noPairs{4};
  const std::array<std::int64_t, 1U> minimumLag{
      std::numeric_limits<std::int64_t>::min()};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto axisMismatch = laggedHigherMoments(
      left, misaligned, lagSpec(zero, lists, CenterRule::estimated, request),
      ample);
  VSS_CHECK(!axisMismatch &&
            axisMismatch.status().code() == ErrorCode::InvalidDomain);

  const auto unsupported = laggedHigherMoments(
      left, left, lagSpec(noPairs, lists, CenterRule::estimated, request),
      ample);
  VSS_CHECK(!unsupported &&
            unsupported.status().code() == ErrorCode::InsufficientSamples);
  const auto minimumUnsupported = laggedHigherMoments(
      left, left, lagSpec(minimumLag, lists, CenterRule::estimated, request),
      ample);
  VSS_CHECK(!minimumUnsupported && minimumUnsupported.status().code() ==
                                       ErrorCode::InsufficientSamples);

  const std::array<std::uint8_t, 4U> allMissing{0U, 0U, 0U, 0U};
  const SeriesView missing{times.view(), x.view(), allMissing, 1.0};
  const auto emptySupport = laggedHigherMoments(
      missing, left, lagSpec(zero, lists, CenterRule::estimated, request),
      ample);
  VSS_CHECK(!emptySupport &&
            emptySupport.status().code() == ErrorCode::InsufficientSamples);

  const std::array<std::uint8_t, 4U> invalidMask{1U, 2U, 1U, 1U};
  const SeriesView malformedMask{times.view(), x.view(), invalidMask, 1.0};
  const auto badMask = laggedHigherMoments(
      malformedMask, left, lagSpec(zero, lists, CenterRule::estimated, request),
      ample);
  VSS_CHECK(!badMask && badMask.status().code() == ErrorCode::InvalidDomain);

  const std::array<Index, 2U> badPair{0U, 2U};
  const std::array<IndexList, 1U> badLists{IndexList{badPair}};
  const auto badComponent =
      laggedHigherMoments(left, left,
                          lagSpec(zero, badLists, CenterRule::estimated,
                                  selectedCovariance(badLists)),
                          ample);
  VSS_CHECK(!badComponent &&
            badComponent.status().code() == ErrorCode::IndexOutOfRange);

  const auto noCadence = laggedHigherMoments(
      {times.view(), x.view(), {}, std::nullopt},
      {times.view(), x.view(), {}, std::nullopt},
      lagSpec(zero, lists, CenterRule::estimated, request), ample);
  VSS_CHECK(!noCadence &&
            noCadence.status().code() == ErrorCode::InvalidDomain);
}

void maskedPayloadIsNotReadAndResourceLimitsAreApplied() {
  const Vector times{std::array<Scalar, 4U>{0.0, 1.0, 2.0, 3.0}};
  Matrix x = column({1.0, 2.0, 3.0, 4.0});
  Matrix y = column({-1.0, 0.0, 1.0, 2.0});
  y.at(1U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  const std::array<std::uint8_t, 4U> mask{1U, 0U, 1U, 1U};
  const SeriesView left{times.view(), x.view(), {}, 1.0};
  const SeriesView right{times.view(), y.view(), mask, 1.0};
  const std::array<std::int64_t, 1U> lags{1};
  const std::array<Index, 2U> pair{0U, 1U};
  const std::array<IndexList, 1U> lists{IndexList{pair}};
  const MomentRequest request = selectedCovariance(lists);

  const auto result = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::estimated, request), ample);
  VSS_CHECK(result && result.value().diagnostics.global.valid == 2U);

  const auto limited = laggedHigherMoments(
      left, right, lagSpec(lags, lists, CenterRule::estimated, request),
      {.elements = 1U, .bytes = 8U, .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() try {
  zeroLagMatchesBatchCrossMoment();
  positiveLagUsesMatchedPairsAndWholeRowMask();
  negativeLagPreservesSignedOrientation();
  knownCenterUsesZeroAndReturnsRawCrossMoment();
  knownCenterSupportsRepeatedFourthOrderTuples();
  constantCenteredMomentsAndMultipleLagsAreExact();
  rejectsMisalignedAxesMasksComponentsAndUnsupportedLag();
  maskedPayloadIsNotReadAndResourceLimitsAreApplied();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
