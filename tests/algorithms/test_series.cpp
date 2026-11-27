#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/signal/Series.hpp>

namespace {
using namespace vss::math;

constexpr Limits ample{
    .elements = 4096U, .bytes = 1U << 20U, .evaluations = 0U};
constexpr SolvePolicy solve{.iteration = {.abs = 0.0,
                                          .rel = 0.0,
                                          .max_iterations = 100U,
                                          .max_evaluations = 0U,
                                          .bracket = std::nullopt},
                            .rank_abs = 0.0,
                            .rank_rel = 1.0e-12,
                            .reciprocal_condition_floor = 0.0,
                            .rank = RankPolicy::minimum_norm,
                            .restart = 0U};

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>{values.begin(), values.size()});
}

Matrix column(std::initializer_list<Scalar> values) {
  Matrix result(values.size(), 1U);
  Index row = 0U;
  std::ranges::for_each(values, [&](Scalar value) {
    result.at(row, 0U) = value;
    ++row;
  });
  return result;
}

void integralUsesIrregularTimeAndCompleteRows() {
  const Vector times = vector({0.0, 1.0, 3.0});
  const Matrix values = column({0.0, 2.0, 6.0});
  const SeriesView series{times.view(), values.view(), {}, std::nullopt};
  const auto result = cumulativeIntegral(series, ample);
  VSS_CHECK(result && result.value().size() == 3U);
  VSS_CHECK(result.value().at(0U) == 0.0 &&
            !std::signbit(result.value().at(0U)));
  VSS_CHECK(std::abs(result.value().at(1U) - 1.0) < 1.0e-14);
  VSS_CHECK(std::abs(result.value().at(2U) - 9.0) < 1.0e-14);

  const std::array<std::uint8_t, 3U> missing{1U, 0U, 1U};
  const auto gapped = cumulativeIntegral(
      {times.view(), values.view(), missing, std::nullopt}, ample);
  VSS_CHECK(!gapped && gapped.status().code() == ErrorCode::InvalidDomain);
  const auto limited = cumulativeIntegral(
      series, {.elements = 3U, .bytes = sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
}

void integralValidatesShapeCadenceAndRange() {
  const Vector times = vector({0.0, 1.0, 1.0});
  const Matrix values = column({1.0, 1.0, 1.0});
  const SeriesView duplicate{times.view(), values.view(), {}, std::nullopt};
  const auto unordered = cumulativeIntegral(duplicate, ample);
  VSS_CHECK(!unordered &&
            unordered.status().code() == ErrorCode::InvalidDomain);

  const Vector wrongTimes = vector({0.0, 1.0});
  const auto mismatch = cumulativeIntegral(
      {wrongTimes.view(), values.view(), {}, std::nullopt}, ample);
  VSS_CHECK(!mismatch &&
            mismatch.status().code() == ErrorCode::DimensionMismatch);

  const Vector regularTimes = vector({0.0, 1.0, 2.0});
  const auto badCadence =
      cumulativeIntegral({regularTimes.view(), values.view(), {}, 2.0}, ample);
  VSS_CHECK(!badCadence &&
            badCadence.status().code() == ErrorCode::InvalidDomain);

  const Vector extremeTimes = vector({0.0, 2.0});
  const Matrix extremeValues = column({-std::numeric_limits<Scalar>::max(),
                                       std::numeric_limits<Scalar>::max()});
  const auto cancellation = cumulativeIntegral(
      {extremeTimes.view(), extremeValues.view(), {}, std::nullopt}, ample);
  VSS_CHECK(cancellation && cancellation.value().at(1U) == 0.0);

  const Vector emptyTimes(0U);
  const Matrix emptyValues(0U, 1U);
  const auto empty = cumulativeIntegral(
      {emptyTimes.view(), emptyValues.view(), {}, std::nullopt}, ample);
  VSS_CHECK(empty && empty.value().empty());

  const Vector oneTime = vector({7.0});
  const Matrix oneValue = column({std::numeric_limits<Scalar>::max()});
  const auto singleton = cumulativeIntegral(
      {oneTime.view(), oneValue.view(), {}, std::nullopt}, ample);
  VSS_CHECK(singleton && singleton.value().size() == 1U &&
            singleton.value().at(0U) == 0.0 &&
            !std::signbit(singleton.value().at(0U)));

  const Vector wideTimes = vector({0.0, std::numeric_limits<Scalar>::max()});
  const Matrix wideValues = column(
      {std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max()});
  const auto overflow = cumulativeIntegral(
      {wideTimes.view(), wideValues.view(), {}, std::nullopt}, ample);
  VSS_CHECK(!overflow &&
            overflow.status().code() == ErrorCode::NonFiniteResult);

  const std::array<std::uint8_t, 2U> invalidMask{1U, 2U};
  const Vector twoTimes = vector({0.0, 1.0});
  const Matrix twoValues = column({0.0, 1.0});
  const auto badMask = cumulativeIntegral(
      {twoTimes.view(), twoValues.view(), invalidMask, std::nullopt}, ample);
  VSS_CHECK(!badMask && badMask.status().code() == ErrorCode::InvalidDomain);
}

void trendUsesElapsedTimeAndSharedWeights() {
  const Vector times = vector({1.0e9, 1.0e9 + 1.0, 1.0e9 + 2.0});
  Matrix values(3U, 2U);
  values.at(0U, 0U) = 3.0;
  values.at(1U, 0U) = 999.0;
  values.at(2U, 0U) = 7.0;
  values.at(0U, 1U) = 5.0;
  values.at(1U, 1U) = -999.0;
  values.at(2U, 1U) = 9.0;
  const std::array<Scalar, 3U> rowWeights{1.0, 0.0, 1.0};
  const WeightSpec weights{.rule = WeightRule::empirical,
                           .rowWeights =
                               ConstVectorView::create(rowWeights, 3U).value(),
                           .componentWeights = std::nullopt,
                           .frequencyCounts = {}};
  const auto result = trend({times.view(), values.view(), {}, std::nullopt}, 1U,
                            weights, solve, ample);
  VSS_CHECK(result && result.value().parameters.rows() == 2U &&
            result.value().parameters.columns() == 2U);
  VSS_CHECK(std::abs(result.value().parameters.at(0U, 0U) - 3.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().parameters.at(1U, 0U) - 2.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().parameters.at(0U, 1U) - 5.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().parameters.at(1U, 1U) - 2.0) < 1.0e-12);
  VSS_CHECK(!result.value().covariance.has_value());
  VSS_CHECK(result.value().residual_norms.size() == 2U);
  VSS_CHECK(result.value().residual_norms.at(0U) < 1.0e-12);
  VSS_CHECK(result.value().residual_norms.at(1U) < 1.0e-12);
}

void trendRejectsUnsupportedAndUnidentifiableInputs() {
  const Vector times = vector({0.0, 1.0, 2.0});
  const Matrix values = column({1.0, 3.0, 5.0});
  const SeriesView series{times.view(), values.view(), {}, std::nullopt};
  const WeightSpec unit{.rule = WeightRule::unweighted,
                        .rowWeights = std::nullopt,
                        .componentWeights = std::nullopt,
                        .frequencyCounts = {}};
  const WeightSpec unsupported{.rule = WeightRule::componentwise,
                               .rowWeights = std::nullopt,
                               .componentWeights = std::nullopt,
                               .frequencyCounts = {}};
  const auto badWeights = trend(series, 1U, unsupported, solve, ample);
  VSS_CHECK(!badWeights &&
            badWeights.status().code() == ErrorCode::InvalidDomain);

  const std::array<Scalar, 3U> noPositiveWeights{0.0, 0.0, 0.0};
  const auto noSupport = trend(
      series, 1U,
      {.rule = WeightRule::empirical,
       .rowWeights = ConstVectorView::create(noPositiveWeights, 3U).value(),
       .componentWeights = std::nullopt,
       .frequencyCounts = {}},
      solve, ample);
  VSS_CHECK(!noSupport &&
            noSupport.status().code() == ErrorCode::InsufficientSamples);

  SolvePolicy strict = solve;
  strict.rank_rel = 0.9;
  strict.rank = RankPolicy::reject;
  const auto deficient = trend(series, 2U,
                               {.rule = WeightRule::unweighted,
                                .rowWeights = std::nullopt,
                                .componentWeights = std::nullopt,
                                .frequencyCounts = {}},
                               strict, ample);
  VSS_CHECK(!deficient &&
            deficient.status().code() == ErrorCode::RankDeficient);

  const std::array<std::uint64_t, 3U> counts{1U, 2U, 1U};
  const WeightSpec frequency{.rule = WeightRule::frequency,
                             .rowWeights = std::nullopt,
                             .componentWeights = std::nullopt,
                             .frequencyCounts = counts};
  const auto replicated = trend(series, 1U, frequency, solve, ample);
  VSS_CHECK(replicated &&
            std::abs(replicated.value().parameters.at(0U, 0U) - 1.0) <
                1.0e-12 &&
            std::abs(replicated.value().parameters.at(1U, 0U) - 2.0) < 1.0e-12);

  const std::array<Scalar, 2U> shortWeights{1.0, 1.0};
  const WeightSpec wrongExtent{
      .rule = WeightRule::empirical,
      .rowWeights = ConstVectorView::create(shortWeights, 2U).value(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto wrongWeightSize = trend(series, 1U, wrongExtent, solve, ample);
  VSS_CHECK(!wrongWeightSize &&
            wrongWeightSize.status().code() == ErrorCode::DimensionMismatch);

  const auto tooLarge = trend(series, 2U, unit, solve,
                              {.elements = 2U, .bytes = 2U, .evaluations = 0U});
  VSS_CHECK(!tooLarge && tooLarge.status().code() == ErrorCode::ResourceLimit);

  const auto badCadence =
      trend({times.view(), values.view(), {}, 2.0}, 1U, unit, solve, ample);
  VSS_CHECK(!badCadence &&
            badCadence.status().code() == ErrorCode::InvalidDomain);
}
} // namespace

int main() try {
  integralUsesIrregularTimeAndCompleteRows();
  integralValidatesShapeCadenceAndRange();
  trendUsesElapsedTimeAndSharedWeights();
  trendRejectsUnsupportedAndUnidentifiableInputs();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
