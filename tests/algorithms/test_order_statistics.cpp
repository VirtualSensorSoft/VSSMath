#include "Test.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Limits;
using vss::math::Matrix;
using vss::math::QuantileRule;
using vss::math::SampleView;
using vss::math::Scalar;
using vss::math::Vector;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

QuantileRule unsupportedRuleForTest() {
  return std::bit_cast<QuantileRule>(std::uint8_t{255U});
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
                   values.view(), unsupportedRuleForTest(), unlimited),
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
  checkFailure(vss::math::statistics::quantile(
                   values.view(), 0.5, unsupportedRuleForTest(), unlimited),
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

void descriptiveSummaryCases() {
  const Matrix values(3U, 1U);
  Matrix sampleValues(values);
  sampleValues.at(0U, 0U) = 1.0;
  sampleValues.at(1U, 0U) = 2.0;
  sampleValues.at(2U, 0U) = 4.0;
  const SampleView population{
      .rows = sampleValues.view(), .observation_shape = {}, .valid = {}};
  const std::array<Scalar, 3U> probabilities{0.25, 0.5, 0.75};
  const vss::math::statistics::WeightSpec noWeights{
      .rule = vss::math::WeightRule::unweighted,
      .rowWeights = std::nullopt,
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto summary = vss::math::statistics::descriptiveSummary(
      population, noWeights,
      {.quantile_probabilities = probabilities,
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  VSS_CHECK(summary);
  if (summary) {
    checkNear(summary.value().range.at(0U), 3.0);
    checkNear(summary.value().sum.at(0U), 7.0);
    checkNear(summary.value().mean.at(0U), 7.0 / 3.0);
    checkNear(summary.value().rms.at(0U), std::sqrt(7.0));
    checkNear(summary.value().variance.at(0U), 14.0 / 9.0);
    checkNear(summary.value().quantiles.at(0U, 0U), 1.5);
    checkNear(summary.value().quantiles.at(0U, 1U), 2.0);
    checkNear(summary.value().quantiles.at(0U, 2U), 3.0);
    VSS_CHECK(summary.value().variance_status.view()[0U] == ErrorCode::None);
  }

  const Matrix singleton(1U, 1U);
  Matrix oneValue(singleton);
  oneValue.at(0U, 0U) = 5.0;
  const SampleView oneSample{
      .rows = oneValue.view(), .observation_shape = {}, .valid = {}};
  const auto iid = vss::math::statistics::descriptiveSummary(
      oneSample, noWeights,
      {.quantile_probabilities = {},
       .variance_mode = vss::math::statistics::VarianceMode::iid_sample},
      unlimited);
  VSS_CHECK(iid);
  if (iid) {
    VSS_CHECK(iid.value().variance_status.view()[0U] ==
              ErrorCode::InsufficientSamples);
    VSS_CHECK(iid.value().variance.at(0U) == 0.0);
    VSS_CHECK(iid.value().mean.at(0U) == 5.0);
  }

  Matrix pairValues(2U, 1U);
  pairValues.at(0U, 0U) = 1.0;
  pairValues.at(1U, 0U) = 4.0;
  const SampleView pairSample{
      .rows = pairValues.view(), .observation_shape = {}, .valid = {}};
  const std::array<std::uint64_t, 2U> counts{2U, 1U};
  const vss::math::statistics::WeightSpec frequency{
      .rule = vss::math::WeightRule::frequency,
      .rowWeights = std::nullopt,
      .componentWeights = std::nullopt,
      .frequencyCounts = counts};
  const std::array<Scalar, 1U> medianProbability{0.5};
  const auto replicated = vss::math::statistics::descriptiveSummary(
      pairSample, frequency,
      {.quantile_probabilities = medianProbability,
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  VSS_CHECK(replicated);
  if (replicated) {
    checkNear(replicated.value().sum.at(0U), 6.0);
    checkNear(replicated.value().mean.at(0U), 2.0);
    checkNear(replicated.value().variance.at(0U), 2.0);
    checkNear(replicated.value().quantiles.at(0U, 0U), 1.0);
    VSS_CHECK(replicated.value().diagnostics.global.frequency_mass == 3U);
    const auto& diagnostics = replicated.value().diagnostics.global;
    VSS_CHECK(diagnostics.effective_count.has_value());
    checkNear(diagnostics.effective_count.value_or(0.0), 3.0);
    VSS_CHECK(diagnostics.normalized_weight_powers[1U].has_value());
    checkNear(diagnostics.normalized_weight_powers[1U].value_or(0.0),
              1.0 / 3.0);
  }
  const auto frequencyIid = vss::math::statistics::descriptiveSummary(
      pairSample, frequency,
      {.quantile_probabilities = {},
       .variance_mode = vss::math::statistics::VarianceMode::iid_sample},
      unlimited);
  VSS_CHECK(frequencyIid);
  if (frequencyIid) {
    checkNear(frequencyIid.value().variance.at(0U), 3.0);
  }

  const Vector empiricalWeights = vector({1.0, 3.0});
  const Vector scaledWeights = vector({10.0, 30.0});
  const vss::math::statistics::WeightSpec empirical{
      .rule = vss::math::WeightRule::empirical,
      .rowWeights = empiricalWeights.view(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const vss::math::statistics::WeightSpec scaled{
      .rule = vss::math::WeightRule::empirical,
      .rowWeights = scaledWeights.view(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto weighted = vss::math::statistics::descriptiveSummary(
      pairSample, empirical,
      {.quantile_probabilities = medianProbability,
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  const auto rescaled = vss::math::statistics::descriptiveSummary(
      pairSample, scaled,
      {.quantile_probabilities = medianProbability,
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  VSS_CHECK(weighted && rescaled);
  if (weighted && rescaled) {
    checkNear(weighted.value().sum.at(0U), 13.0);
    checkNear(weighted.value().mean.at(0U), 3.25);
    checkNear(rescaled.value().sum.at(0U), 130.0);
    checkNear(rescaled.value().mean.at(0U), weighted.value().mean.at(0U));
    checkNear(rescaled.value().quantiles.at(0U, 0U),
              weighted.value().quantiles.at(0U, 0U));
    checkNear(weighted.value().quantiles.at(0U, 0U), 4.0);
  }

  Matrix importanceValues(3U, 1U);
  importanceValues.at(0U, 0U) = 0.0;
  importanceValues.at(1U, 0U) = 2.0;
  importanceValues.at(2U, 0U) = 6.0;
  const SampleView importanceSample{
      .rows = importanceValues.view(), .observation_shape = {}, .valid = {}};
  const Vector importanceWeights = vector({1.0, 2.0, 1.0});
  const Vector rescaledImportanceWeights = vector({3.0, 6.0, 3.0});
  const vss::math::statistics::WeightSpec importance{
      .rule = vss::math::WeightRule::importance,
      .rowWeights = importanceWeights.view(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const vss::math::statistics::WeightSpec rescaledImportance{
      .rule = vss::math::WeightRule::importance,
      .rowWeights = rescaledImportanceWeights.view(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto importanceSummary = vss::math::statistics::descriptiveSummary(
      importanceSample, importance,
      {.quantile_probabilities = medianProbability,
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  const auto rescaledImportanceSummary =
      vss::math::statistics::descriptiveSummary(
          importanceSample, rescaledImportance,
          {.quantile_probabilities = medianProbability,
           .variance_mode = vss::math::statistics::VarianceMode::empirical},
          unlimited);
  VSS_CHECK(importanceSummary && rescaledImportanceSummary);
  if (importanceSummary && rescaledImportanceSummary) {
    checkNear(importanceSummary.value().sum.at(0U), 10.0);
    checkNear(rescaledImportanceSummary.value().sum.at(0U), 30.0);
    checkNear(importanceSummary.value().mean.at(0U), 2.5);
    checkNear(importanceSummary.value().rms.at(0U), std::sqrt(11.0));
    checkNear(importanceSummary.value().variance.at(0U), 19.0 / 4.0);
    checkNear(importanceSummary.value().quantiles.at(0U, 0U), 2.0);
    checkNear(rescaledImportanceSummary.value().mean.at(0U), 2.5);
    VSS_CHECK(importanceSummary.value().diagnostics.premises.find(
                  "no target-distribution unbiasedness claimed") !=
              std::string::npos);
  }

  const auto resourceLimited = vss::math::statistics::descriptiveSummary(
      pairSample, noWeights,
      {.quantile_probabilities = {},
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      {.elements = 7U, .bytes = unlimited.bytes, .evaluations = 0U});
  VSS_CHECK(!resourceLimited);
  VSS_CHECK(resourceLimited.status().code() == ErrorCode::ResourceLimit);

  const vss::math::statistics::WeightSpec precision{
      .rule = vss::math::WeightRule::precision,
      .rowWeights = empiricalWeights.view(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto rejected = vss::math::statistics::descriptiveSummary(
      pairSample, precision,
      {.quantile_probabilities = {},
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  VSS_CHECK(!rejected);
  VSS_CHECK(rejected.status().code() == ErrorCode::InvalidDomain);

  Matrix extremeValues(2U, 1U);
  extremeValues.at(0U, 0U) = -std::numeric_limits<Scalar>::max();
  extremeValues.at(1U, 0U) = std::numeric_limits<Scalar>::max();
  const SampleView extremeSample{
      .rows = extremeValues.view(), .observation_shape = {}, .valid = {}};
  const auto unrepresentableRange = vss::math::statistics::descriptiveSummary(
      extremeSample, noWeights,
      {.quantile_probabilities = {},
       .variance_mode = vss::math::statistics::VarianceMode::empirical},
      unlimited);
  VSS_CHECK(!unrepresentableRange);
  VSS_CHECK(unrepresentableRange.status().code() == ErrorCode::NonFiniteResult);
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
  descriptiveSummaryCases();
}

// VSS-MATH-TRACEABILITY: STAT-ORDER-01
