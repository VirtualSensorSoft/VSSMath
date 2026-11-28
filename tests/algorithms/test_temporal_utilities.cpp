#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <span>
#include <vss/math/Error.hpp>
#include <vss/math/Interpolation.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/signal/Dependence.hpp>
#include <vss/math/signal/Series.hpp>
#include <vss/math/signal/TemporalUtilities.hpp>

namespace {
using namespace vss::math;

constexpr Limits ample{
    .elements = 4096U, .bytes = 1U << 20U, .evaluations = 0U};

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>{values.begin(), values.size()});
}

void cumulativeSumPreservesEveryRepresentablePrefix() {
  const Vector values = vector({0.0, 1.0, 2.0, 3.0});
  const auto sum = signal::cumulativeSum(values.view(), ample);
  VSS_CHECK(sum && sum.value().size() == 4U);
  VSS_CHECK(sum.value().at(0U) == 0.0 && !std::signbit(sum.value().at(0U)));
  VSS_CHECK(sum.value().at(1U) == 1.0 && sum.value().at(2U) == 3.0 &&
            sum.value().at(3U) == 6.0);

  const Scalar largest = std::numeric_limits<Scalar>::max();
  const Vector cancellation = vector({largest, 1.0, -largest});
  const auto exactPrefix = signal::cumulativeSum(cancellation.view(), ample);
  VSS_CHECK(exactPrefix && exactPrefix.value().at(2U) == 1.0);
  const Scalar halfUlp = std::numeric_limits<Scalar>::epsilon() * 0.5;
  const Vector halfway = vector({1.0, halfUlp});
  const auto tiesToEven = signal::cumulativeSum(halfway.view(), ample);
  VSS_CHECK(tiesToEven && tiesToEven.value().at(1U) == 1.0);
  const Scalar tiny = std::numeric_limits<Scalar>::denorm_min();
  const Vector subnormals = vector({tiny, tiny});
  const auto exactSubnormal = signal::cumulativeSum(subnormals.view(), ample);
  VSS_CHECK(exactSubnormal && exactSubnormal.value().at(1U) == 2.0 * tiny);
  const Vector negativeZero = vector({-0.0});
  const auto normalizedZero = signal::cumulativeSum(negativeZero.view(), ample);
  VSS_CHECK(normalizedZero && normalizedZero.value().at(0U) == 0.0 &&
            !std::signbit(normalizedZero.value().at(0U)));
  const Vector overflow = vector({largest, largest});
  const auto outOfRange = signal::cumulativeSum(overflow.view(), ample);
  VSS_CHECK(!outOfRange &&
            outOfRange.status().code() == ErrorCode::NonFiniteResult);
  const Vector empty(0U);
  VSS_CHECK(signal::cumulativeSum(empty.view(), ample).value().empty());
  const auto limited = signal::cumulativeSum(
      values.view(),
      {.elements = 3U, .bytes = 3U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const Vector nonFinite =
      vector({1.0, std::numeric_limits<Scalar>::infinity()});
  const auto rejected = signal::cumulativeSum(nonFinite.view(), ample);
  VSS_CHECK(!rejected && rejected.status().code() == ErrorCode::NonFiniteInput);
}

void lagSummaryUsesSignedTieBreakAndExplicitCutoff() {
  const Scalar second = (4.0 - std::sqrt(7.0)) / 3.0;
  const Vector samples = vector({1.0, second, -1.0, -second});
  const std::array<std::int64_t, 4U> requested{-2, 0, 2, 1};
  const auto correlations =
      signal::crossCorrelation(samples.view(), samples.view(), requested,
                               signal::LagNormalization::overlap_count, ample);
  VSS_CHECK(correlations);
  const auto summary =
      signal::summarizeLags(correlations.value(), 2.0, 1U, ample);
  VSS_CHECK(summary && summary.value().peak_lag == -2 &&
            summary.value().peak_value == -1.0 && summary.value().cutoff == 1U);
  VSS_CHECK(std::abs(summary.value().correlation_time - 3.0) < 1.0e-12);

  const std::array<Index, 2U> incompleteLags{0U, 2U};
  const auto incomplete =
      signal::autocorrelation(samples.view(), incompleteLags,
                              signal::LagNormalization::full_count, ample);
  VSS_CHECK(incomplete);
  const auto missingCutoff =
      signal::summarizeLags(incomplete.value(), 1.0, 1U, ample);
  VSS_CHECK(!missingCutoff &&
            missingCutoff.status().code() == ErrorCode::InsufficientSamples);

  const std::array<Index, 2U> adjacentLags{0U, 1U};
  const Vector alternatingPair = vector({1.0, -1.0});
  const auto zeroTimeSeries =
      signal::autocorrelation(alternatingPair.view(), adjacentLags,
                              signal::LagNormalization::full_count, ample);
  VSS_CHECK(zeroTimeSeries);
  const auto zeroTime =
      signal::summarizeLags(zeroTimeSeries.value(), 2.0, 1U, ample);
  VSS_CHECK(zeroTime && zeroTime.value().correlation_time == 0.0 &&
            !std::signbit(zeroTime.value().correlation_time));

  const Vector alternatingFour = vector({1.0, -1.0, 1.0, -1.0});
  const auto negativeTimeSeries =
      signal::autocorrelation(alternatingFour.view(), adjacentLags,
                              signal::LagNormalization::full_count, ample);
  VSS_CHECK(negativeTimeSeries);
  const auto negativeTime =
      signal::summarizeLags(negativeTimeSeries.value(), 1.0, 1U, ample);
  VSS_CHECK(!negativeTime &&
            negativeTime.status().code() == ErrorCode::InvalidDomain);

  const std::array<std::int64_t, 3U> duplicateLags{0, 1, 1};
  const auto duplicates =
      signal::crossCorrelation(samples.view(), samples.view(), duplicateLags,
                               signal::LagNormalization::full_count, ample);
  VSS_CHECK(duplicates);
  const auto duplicateSummary =
      signal::summarizeLags(duplicates.value(), 1.0, 1U, ample);
  VSS_CHECK(!duplicateSummary &&
            duplicateSummary.status().code() == ErrorCode::InvalidDomain);

  const auto badStep =
      signal::summarizeLags(correlations.value(), 0.0, 1U, ample);
  VSS_CHECK(!badStep && badStep.status().code() == ErrorCode::InvalidDomain);
  const auto denied =
      signal::summarizeLags(correlations.value(), 1.0, 1U,
                            {.elements = 1U, .bytes = 1U, .evaluations = 0U});
  VSS_CHECK(!denied && denied.status().code() == ErrorCode::ResourceLimit);
}

void alignSeriesPreservesSourceMasksAndExactKnots() {
  const Vector sourceTimes = vector({0.0, 2.0});
  Matrix sourceRows(2U, 1U);
  sourceRows.at(0U, 0U) = 4.0;
  sourceRows.at(1U, 0U) = 8.0;
  const SeriesView source{sourceTimes.view(), sourceRows.view()};
  const Vector target = vector({0.0, 1.0, 2.0});
  const std::array<SeriesView, 1U> sources{source};
  const auto aligned = signal::alignSeries(sources, target.view(),
                                           {Extrapolation::reject, 1.0}, ample);
  VSS_CHECK(aligned && aligned.value().targetTimes().size() == 3U);
  VSS_CHECK(aligned.value().sources().size() == 1U);
  const auto& result = aligned.value().sources().front();
  VSS_CHECK(result.times().size() == 3U && result.rows().rows() == 3U &&
            result.valid().size() == 3U);
  VSS_CHECK(!result.uniformStep().has_value());
  VSS_CHECK(result.valid()[0U] == 1U && result.valid()[1U] == 0U &&
            result.valid()[2U] == 1U);
  VSS_CHECK(result.rows().at(1U, 0U) == 0.0 &&
            !std::signbit(result.rows().at(1U, 0U)));
  VSS_CHECK(result.rows().at(0U, 0U) == 4.0 && result.rows().at(2U, 0U) == 8.0);
}

void alignSeriesKeepsIndependentSourcesAndValidatesLimits() {
  const Vector firstTimes = vector({0.0, 2.0});
  Matrix firstRows(2U, 1U);
  firstRows.at(0U, 0U) = 0.0;
  firstRows.at(1U, 0U) = 2.0;
  const Vector secondTimes = vector({1.0, 2.0});
  Matrix secondRows(2U, 2U);
  secondRows.at(0U, 0U) = 7.0;
  secondRows.at(0U, 1U) = 8.0;
  secondRows.at(1U, 0U) = 9.0;
  secondRows.at(1U, 1U) = 10.0;
  const std::array<SeriesView, 2U> inputs{
      SeriesView{firstTimes.view(), firstRows.view()},
      SeriesView{secondTimes.view(), secondRows.view()}};
  const Vector target = vector({0.0, 1.0, 2.0});
  const auto aligned = signal::alignSeries(inputs, target.view(),
                                           {Extrapolation::reject, 2.0}, ample);
  VSS_CHECK(aligned && aligned.value().sources().size() == 2U);
  const auto result = aligned.value().sources();
  VSS_CHECK(result[0U].rows().columns() == 1U &&
            result[1U].rows().columns() == 2U);
  VSS_CHECK(result[0U].valid()[0U] == 1U && result[1U].valid()[0U] == 0U);
  VSS_CHECK(result[0U].valid()[1U] == 1U && result[1U].valid()[1U] == 1U);
  VSS_CHECK(result[1U].rows().at(1U, 0U) == 7.0 &&
            result[1U].rows().at(1U, 1U) == 8.0);
  VSS_CHECK(result[1U].times().at(0U) == target.at(0U));

  const auto limited =
      signal::alignSeries(inputs, target.view(), {Extrapolation::reject, 2.0},
                          {.elements = 8U, .bytes = 64U, .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const Vector unorderedTarget = vector({1.0, 1.0});
  const auto invalid =
      signal::alignSeries(inputs, unorderedTarget.view(), {}, ample);
  VSS_CHECK(!invalid && invalid.status().code() == ErrorCode::InvalidDomain);
}

void alignSeriesSupportsExtrapolationAndEmptySources() {
  const Vector times = vector({1.0, 2.0});
  Matrix rows(2U, 1U);
  rows.at(0U, 0U) = 10.0;
  rows.at(1U, 0U) = 20.0;
  const SeriesView source{times.view(), rows.view()};
  const std::array<SeriesView, 1U> inputs{source};
  const Vector target = vector({0.0, 1.0, 3.0, 4.0});

  const auto held = signal::alignSeries(
      inputs, target.view(), {Extrapolation::hold, std::nullopt}, ample);
  VSS_CHECK(held && held.value().sources()[0U].valid()[0U] == 1U &&
            held.value().sources()[0U].rows().at(0U, 0U) == 10.0 &&
            held.value().sources()[0U].rows().at(3U, 0U) == 20.0);
  const auto extended = signal::alignSeries(
      inputs, target.view(), {Extrapolation::linear, std::nullopt}, ample);
  VSS_CHECK(extended &&
            extended.value().sources()[0U].rows().at(0U, 0U) == 0.0 &&
            extended.value().sources()[0U].rows().at(2U, 0U) == 30.0 &&
            extended.value().sources()[0U].rows().at(3U, 0U) == 40.0);

  const Vector emptyTimes(0U);
  const Matrix emptyRows(0U, 2U);
  const std::array<SeriesView, 1U> emptyInput{
      SeriesView{emptyTimes.view(), emptyRows.view()}};
  const auto empty = signal::alignSeries(emptyInput, target.view(), {}, ample);
  VSS_CHECK(empty && empty.value().sources().size() == 1U &&
            empty.value().sources()[0U].rows().columns() == 2U &&
            empty.value().sources()[0U].valid().size() == target.size());
  for (std::uint8_t valid : empty.value().sources()[0U].valid()) {
    VSS_CHECK(valid == 0U);
  }
  const std::span<const SeriesView> noInputs;
  const auto noSources =
      signal::alignSeries(noInputs, target.view(), {}, ample);
  VSS_CHECK(noSources && noSources.value().sources().empty() &&
            noSources.value().targetTimes().size() == target.size());
}

void alignSeriesPropagatesMissingnessWithoutGapBridging() {
  const Vector times = vector({0.0, 1.0, 2.0});
  Matrix rows(3U, 1U);
  rows.at(0U, 0U) = 2.0;
  rows.at(1U, 0U) = std::numeric_limits<Scalar>::quiet_NaN();
  rows.at(2U, 0U) = 6.0;
  const std::array<std::uint8_t, 3U> validity{1U, 0U, 1U};
  const SeriesView source{times.view(), rows.view(), validity};
  const std::array<SeriesView, 1U> inputs{source};
  const Vector target = vector({0.0, 0.5, 1.0, 1.5, 2.0});
  const auto aligned = signal::alignSeries(inputs, target.view(), {}, ample);
  VSS_CHECK(aligned);
  const auto& result = aligned.value().sources()[0U];
  VSS_CHECK(result.valid()[0U] == 1U && result.valid()[1U] == 0U &&
            result.valid()[2U] == 0U && result.valid()[3U] == 0U &&
            result.valid()[4U] == 1U);
  VSS_CHECK(result.rows().at(1U, 0U) == 0.0 &&
            result.rows().at(2U, 0U) == 0.0 && result.rows().at(3U, 0U) == 0.0);
}
} // namespace

int main() try {
  cumulativeSumPreservesEveryRepresentablePrefix();
  lagSummaryUsesSignedTieBreakAndExplicitCutoff();
  alignSeriesPreservesSourceMasksAndExactKnots();
  alignSeriesKeepsIndependentSourcesAndValidatesLimits();
  alignSeriesSupportsExtrapolationAndEmptySources();
  alignSeriesPropagatesMissingnessWithoutGapBridging();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
