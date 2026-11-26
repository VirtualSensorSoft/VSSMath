#include "Test.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vss/math/Error.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/probability/Composite.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

ProbabilityPolicy policy() {
  return {{1.0e-12, 1.0e-12, 4096U, 100000U, std::nullopt}, 100000U, 1.0e-10};
}

void mixedObservationKindsHaveTheSpecifiedLikelihood() {
  const auto normalLaw = normal(0.0, 1.0);
  VSS_CHECK(normalLaw.hasValue());
  if (!normalLaw) {
    return;
  }

  const auto exact = nonidealObservationLogProbability(
      normalLaw.value(), {ObservationKind::exact, 0.0, std::nullopt, false},
      policy());
  const auto interval = nonidealObservationLogProbability(
      normalLaw.value(),
      {ObservationKind::interval, std::nullopt,
       IntervalSupport{0.0, 1.0, true, true}, false},
      policy());
  const auto below = nonidealObservationLogProbability(
      normalLaw.value(),
      {ObservationKind::below_threshold, std::nullopt,
       IntervalSupport{-std::numeric_limits<Scalar>::infinity(), 0.0, false,
                       true},
       false},
      policy());
  const auto above = nonidealObservationLogProbability(
      normalLaw.value(),
      {ObservationKind::above_threshold, std::nullopt,
       IntervalSupport{0.0, std::numeric_limits<Scalar>::infinity(), false,
                       true},
       false},
      policy());
  const auto quantized = nonidealObservationLogProbability(
      normalLaw.value(),
      {ObservationKind::quantized, std::nullopt,
       IntervalSupport{0.0, 1.0, true, false}, false},
      policy());
  const auto unavailable = nonidealObservationLogProbability(
      normalLaw.value(),
      {ObservationKind::unavailable, std::nullopt, std::nullopt, true},
      policy());
  VSS_CHECK(exact && interval && below && above && quantized && unavailable);
  if (exact && interval && below && above && quantized && unavailable) {
    VSS_CHECK(exact.value().log_probability.has_value());
    VSS_CHECK(std::abs(*exact.value().log_probability +
                       0.5 * std::log(2.0 * std::numbers::pi)) < 2.0e-12);
    VSS_CHECK(interval.value().log_probability.has_value());
    VSS_CHECK(std::abs(*interval.value().log_probability -
                       std::log(std::erf(1.0 / std::sqrt(2.0)) / 2.0)) <
              2.0e-10);
    VSS_CHECK(below.value().log_probability.has_value());
    VSS_CHECK(std::abs(*below.value().log_probability - std::log(0.5)) <
              2.0e-12);
    VSS_CHECK(above.value().log_probability.has_value());
    VSS_CHECK(std::abs(*above.value().log_probability - std::log(0.5)) <
              2.0e-12);
    VSS_CHECK(quantized.value().log_probability.has_value());
    VSS_CHECK(std::abs(*quantized.value().log_probability -
                       std::log(std::erf(1.0 / std::sqrt(2.0)) / 2.0)) <
              2.0e-10);
    VSS_CHECK(!unavailable.value().log_probability.has_value());
  }

  const auto coin = bernoulli(0.25);
  VSS_CHECK(coin.hasValue());
  if (coin) {
    const auto exactAtom = nonidealObservationLogProbability(
        coin.value(), {ObservationKind::exact, 1.0, std::nullopt, false},
        policy());
    const auto inclusiveBelow = nonidealObservationLogProbability(
        coin.value(),
        {ObservationKind::below_threshold, std::nullopt,
         IntervalSupport{-std::numeric_limits<Scalar>::infinity(), 0.0, true,
                         true},
         false},
        policy());
    const auto exclusiveAbove = nonidealObservationLogProbability(
        coin.value(),
        {ObservationKind::above_threshold, std::nullopt,
         IntervalSupport{0.0, std::numeric_limits<Scalar>::infinity(), false,
                         false},
         false},
        policy());
    const auto allValues = nonidealObservationLogProbability(
        coin.value(),
        {ObservationKind::interval, std::nullopt,
         IntervalSupport{0.0, 1.0, true, true}, false},
        policy());
    const auto noAtoms = nonidealObservationLogProbability(
        coin.value(),
        {ObservationKind::interval, std::nullopt,
         IntervalSupport{0.0, 1.0, false, false}, false},
        policy());
    const auto quantizedAtom = nonidealObservationLogProbability(
        coin.value(),
        {ObservationKind::quantized, std::nullopt,
         IntervalSupport{0.5, 1.5, true, false}, false},
        policy());
    VSS_CHECK(exactAtom && inclusiveBelow && exclusiveAbove && allValues &&
              noAtoms && quantizedAtom);
    if (exactAtom && inclusiveBelow && exclusiveAbove && allValues && noAtoms &&
        quantizedAtom) {
      VSS_CHECK(std::abs(*exactAtom.value().log_probability - std::log(0.25)) <
                1.0e-14);
      VSS_CHECK(std::abs(*inclusiveBelow.value().log_probability -
                         std::log(0.75)) < 1.0e-14);
      VSS_CHECK(std::abs(*exclusiveAbove.value().log_probability -
                         std::log(0.25)) < 1.0e-14);
      VSS_CHECK(*allValues.value().log_probability == 0.0);
      VSS_CHECK(std::isinf(*noAtoms.value().log_probability) &&
                *noAtoms.value().log_probability < 0.0);
      VSS_CHECK(std::abs(*quantizedAtom.value().log_probability -
                         std::log(0.25)) < 1.0e-14);
    }
  }
}

void contradictoryAndUnapprovedUnavailableEventsFail() {
  const auto law = normal(0.0, 1.0);
  VSS_CHECK(law.hasValue());
  if (!law) {
    return;
  }
  const auto contradictory = nonidealObservationLogProbability(
      law.value(),
      {ObservationKind::exact, 0.0, IntervalSupport{0.0, 1.0, true, true},
       false},
      policy());
  const auto unavailable = nonidealObservationLogProbability(
      law.value(),
      {ObservationKind::unavailable, std::nullopt, std::nullopt, false},
      policy());
  VSS_CHECK(!contradictory);
  VSS_CHECK(contradictory.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!unavailable);
  VSS_CHECK(unavailable.status().code() == ErrorCode::InvalidDomain);
}
} // namespace

int main() {
  mixedObservationKindsHaveTheSpecifiedLikelihood();
  contradictoryAndUnapprovedUnavailableEventsFail();
}

// VSS-MATH-TRACEABILITY: C103 CASE-C103.nonideal-observation-events
// VSS-MATH-TRACEABILITY: C103 CASE-C103.S04.events
