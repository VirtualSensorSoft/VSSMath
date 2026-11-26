#include "Test.hpp"
#include "vss/math/Core.hpp"

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
  return {.iteration = {.abs = 1.0e-12,
                        .rel = 1.0e-12,
                        .max_iterations = 4096U,
                        .max_evaluations = 100000U,
                        .bracket = std::nullopt},
          .max_draws = 100000U,
          .abs_log_error = 1.0e-10};
}

Scalar logProbabilityOrNaN(const ObservationLikelihood& likelihood) {
  return likelihood.log_probability.value_or(
      std::numeric_limits<Scalar>::quiet_NaN());
}

void mixedObservationKindsHaveTheSpecifiedLikelihood() {
  const auto normalLaw = normal(0.0, 1.0);
  VSS_CHECK(normalLaw.hasValue());
  if (!normalLaw) {
    return;
  }

  const auto exact =
      nonidealObservationLogProbability(normalLaw.value(),
                                        {.kind = ObservationKind::exact,
                                         .value = 0.0,
                                         .support = std::nullopt,
                                         .omit_unavailable = false},
                                        policy());
  const auto interval = nonidealObservationLogProbability(
      normalLaw.value(),
      {.kind = ObservationKind::interval,
       .value = std::nullopt,
       .support = IntervalSupport{.lower = 0.0,
                                  .upper = 1.0,
                                  .include_lower = true,
                                  .include_upper = true},
       .omit_unavailable = false},
      policy());
  const auto below = nonidealObservationLogProbability(
      normalLaw.value(),
      {.kind = ObservationKind::below_threshold,
       .value = std::nullopt,
       .support =
           IntervalSupport{.lower = -std::numeric_limits<Scalar>::infinity(),
                           .upper = 0.0,
                           .include_lower = false,
                           .include_upper = true},
       .omit_unavailable = false},
      policy());
  const auto above = nonidealObservationLogProbability(
      normalLaw.value(),
      {.kind = ObservationKind::above_threshold,
       .value = std::nullopt,
       .support =
           IntervalSupport{.lower = 0.0,
                           .upper = std::numeric_limits<Scalar>::infinity(),
                           .include_lower = false,
                           .include_upper = true},
       .omit_unavailable = false},
      policy());
  const auto quantized = nonidealObservationLogProbability(
      normalLaw.value(),
      {.kind = ObservationKind::quantized,
       .value = std::nullopt,
       .support = IntervalSupport{.lower = 0.0,
                                  .upper = 1.0,
                                  .include_lower = true,
                                  .include_upper = false},
       .omit_unavailable = false},
      policy());
  const auto unavailable =
      nonidealObservationLogProbability(normalLaw.value(),
                                        {.kind = ObservationKind::unavailable,
                                         .value = std::nullopt,
                                         .support = std::nullopt,
                                         .omit_unavailable = true},
                                        policy());
  VSS_CHECK(exact && interval && below && above && quantized && unavailable);
  if (exact && interval && below && above && quantized && unavailable) {
    VSS_CHECK(exact.value().log_probability.has_value());
    VSS_CHECK(std::abs(logProbabilityOrNaN(exact.value()) +
                       (0.5 * std::log(2.0 * std::numbers::pi))) < 2.0e-12);
    VSS_CHECK(interval.value().log_probability.has_value());
    VSS_CHECK(std::abs(logProbabilityOrNaN(interval.value()) -
                       std::log(std::erf(1.0 / std::sqrt(2.0)) / 2.0)) <
              2.0e-10);
    VSS_CHECK(below.value().log_probability.has_value());
    VSS_CHECK(std::abs(logProbabilityOrNaN(below.value()) - std::log(0.5)) <
              2.0e-12);
    VSS_CHECK(above.value().log_probability.has_value());
    VSS_CHECK(std::abs(logProbabilityOrNaN(above.value()) - std::log(0.5)) <
              2.0e-12);
    VSS_CHECK(quantized.value().log_probability.has_value());
    VSS_CHECK(std::abs(logProbabilityOrNaN(quantized.value()) -
                       std::log(std::erf(1.0 / std::sqrt(2.0)) / 2.0)) <
              2.0e-10);
    VSS_CHECK(!unavailable.value().log_probability.has_value());
  }

  const auto coin = bernoulli(0.25);
  VSS_CHECK(coin.hasValue());
  if (coin) {
    const auto exactAtom =
        nonidealObservationLogProbability(coin.value(),
                                          {.kind = ObservationKind::exact,
                                           .value = 1.0,
                                           .support = std::nullopt,
                                           .omit_unavailable = false},
                                          policy());
    const auto inclusiveBelow = nonidealObservationLogProbability(
        coin.value(),
        {.kind = ObservationKind::below_threshold,
         .value = std::nullopt,
         .support =
             IntervalSupport{.lower = -std::numeric_limits<Scalar>::infinity(),
                             .upper = 0.0,
                             .include_lower = true,
                             .include_upper = true},
         .omit_unavailable = false},
        policy());
    const auto exclusiveAbove = nonidealObservationLogProbability(
        coin.value(),
        {.kind = ObservationKind::above_threshold,
         .value = std::nullopt,
         .support =
             IntervalSupport{.lower = 0.0,
                             .upper = std::numeric_limits<Scalar>::infinity(),
                             .include_lower = false,
                             .include_upper = false},
         .omit_unavailable = false},
        policy());
    const auto allValues = nonidealObservationLogProbability(
        coin.value(),
        {.kind = ObservationKind::interval,
         .value = std::nullopt,
         .support = IntervalSupport{.lower = 0.0,
                                    .upper = 1.0,
                                    .include_lower = true,
                                    .include_upper = true},
         .omit_unavailable = false},
        policy());
    const auto noAtoms = nonidealObservationLogProbability(
        coin.value(),
        {.kind = ObservationKind::interval,
         .value = std::nullopt,
         .support = IntervalSupport{.lower = 0.0,
                                    .upper = 1.0,
                                    .include_lower = false,
                                    .include_upper = false},
         .omit_unavailable = false},
        policy());
    const auto quantizedAtom = nonidealObservationLogProbability(
        coin.value(),
        {.kind = ObservationKind::quantized,
         .value = std::nullopt,
         .support = IntervalSupport{.lower = 0.5,
                                    .upper = 1.5,
                                    .include_lower = true,
                                    .include_upper = false},
         .omit_unavailable = false},
        policy());
    VSS_CHECK(exactAtom && inclusiveBelow && exclusiveAbove && allValues &&
              noAtoms && quantizedAtom);
    if (exactAtom && inclusiveBelow && exclusiveAbove && allValues && noAtoms &&
        quantizedAtom) {
      VSS_CHECK(std::abs(logProbabilityOrNaN(exactAtom.value()) -
                         std::log(0.25)) < 1.0e-14);
      VSS_CHECK(std::abs(logProbabilityOrNaN(inclusiveBelow.value()) -
                         std::log(0.75)) < 1.0e-14);
      VSS_CHECK(std::abs(logProbabilityOrNaN(exclusiveAbove.value()) -
                         std::log(0.25)) < 1.0e-14);
      VSS_CHECK(logProbabilityOrNaN(allValues.value()) == 0.0);
      VSS_CHECK(std::isinf(logProbabilityOrNaN(noAtoms.value())) &&
                logProbabilityOrNaN(noAtoms.value()) < 0.0);
      VSS_CHECK(std::abs(logProbabilityOrNaN(quantizedAtom.value()) -
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
      {.kind = ObservationKind::exact,
       .value = 0.0,
       .support = IntervalSupport{.lower = 0.0,
                                  .upper = 1.0,
                                  .include_lower = true,
                                  .include_upper = true},
       .omit_unavailable = false},
      policy());
  const auto unavailable =
      nonidealObservationLogProbability(law.value(),
                                        {.kind = ObservationKind::unavailable,
                                         .value = std::nullopt,
                                         .support = std::nullopt,
                                         .omit_unavailable = false},
                                        policy());
  VSS_CHECK(!contradictory);
  VSS_CHECK(contradictory.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(!unavailable);
  VSS_CHECK(unavailable.status().code() == ErrorCode::InvalidDomain);
}
} // namespace

int main() try {
  mixedObservationKindsHaveTheSpecifiedLikelihood();
  contradictoryAndUnapprovedUnavailableEventsFail();
} catch (...) {
  return 1;
}

// VSS-MATH-TRACEABILITY: C103 CASE-C103.nonideal-observation-events
// VSS-MATH-TRACEABILITY: C103 CASE-C103.S04.events
