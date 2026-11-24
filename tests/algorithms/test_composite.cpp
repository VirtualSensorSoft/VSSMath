#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vss/math/Error.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Composite.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

void publicSignatures() {
  using Truncate = Result<Distribution> (*)(const Distribution&,
                                            IntervalSupport, ProbabilityPolicy);
  using TruncateLimited = Result<Distribution> (*)(
      const Distribution&, IntervalSupport, ProbabilityPolicy, Limits);
  using IntervalLog = Result<Scalar> (*)(const Distribution&, IntervalSupport,
                                         ProbabilityPolicy);
  static_assert(
      std::is_same_v<decltype(static_cast<Truncate>(&truncatedDistribution)),
                     Truncate>);
  static_assert(std::is_same_v<decltype(static_cast<TruncateLimited>(
                                   &truncatedDistribution)),
                               TruncateLimited>);
  static_assert(std::is_same_v<decltype(static_cast<IntervalLog>(
                                   &censoredLogProbability)),
                               IntervalLog>);
  static_assert(
      std::is_same_v<decltype(static_cast<IntervalLog>(&quantizedLikelihood)),
                     IntervalLog>);
}

ProbabilityPolicy policy() {
  return {{1.0e-12, 1.0e-12, 4096U, 100000U, std::nullopt}, 100000U, 1.0e-10};
}

void empiricalCdfAndMixtureLogDensity() {
  const std::array<Scalar, 2> values{0.0, 2.0};
  const std::array<Scalar, 2> weights{1.0, 3.0};
  const auto valueView = ConstVectorView::create(values, values.size());
  const auto weightView = ConstVectorView::create(weights, weights.size());
  const auto empiricalLaw =
      empirical(valueView.value(), weightView.value(), {32U, 4096U, 256U});
  VSS_CHECK(empiricalLaw.hasValue());
  if (empiricalLaw) {
    VSS_CHECK(std::abs(empiricalCDF(empiricalLaw.value(), 0.0).value() - 0.25) <
              1.0e-15);
    const auto notEmpirical = empiricalCDF(normal(0.0, 1.0).value(), 0.0);
    VSS_CHECK(!notEmpirical.hasValue());
    VSS_CHECK(notEmpirical.status().code() == ErrorCode::InvalidDomain);
    VSS_CHECK(std::abs(empiricalCDF(empiricalLaw.value(), 1.0).value() - 0.25) <
              1.0e-15);
  }
  const auto normalLaw = normal(0.0, 1.0);
  VSS_CHECK(normalLaw.hasValue());
  if (normalLaw) {
    const std::array<WeightedLaw, 2> same{
        {{normalLaw.value(), 1.0}, {normalLaw.value(), 1.0}}};
    const auto mixed = mixture(same, policy());
    VSS_CHECK(mixed.hasValue());
    if (mixed) {
      const auto actual = mixtureLogPdf(mixed.value(), 0.0, policy());
      VSS_CHECK(actual.hasValue());
      if (actual) {
        VSS_CHECK(std::abs(actual.value() +
                           0.5 * std::log(2.0 * std::numbers::pi)) < 2.0e-12);
      }
    }
  }
}

void truncationAndIntervalLikelihoods() {
  const auto uniformLaw = uniform(0.0, 2.0);
  VSS_CHECK(uniformLaw.hasValue());
  if (uniformLaw) {
    const auto truncated = truncatedDistribution(
        uniformLaw.value(), IntervalSupport{0.0, 1.0, true, true}, policy());
    VSS_CHECK(truncated.hasValue());
    if (truncated) {
      const auto median = quantile(truncated.value(), 0.5, policy());
      const auto atHalf = evaluate(truncated.value(), 0.5, policy());
      const auto secondMoment = rawMoment(truncated.value(), 2U, policy());
      VSS_CHECK(truncated.value().family() == Family::truncated);
      VSS_CHECK(median.hasValue());
      VSS_CHECK(atHalf.hasValue());
      VSS_CHECK(secondMoment.hasValue());
      if (median && atHalf && secondMoment) {
        VSS_CHECK(std::abs(median.value() - 0.5) < 2.0e-10);
        VSS_CHECK(std::abs(atHalf.value().density_or_mass - 1.0) < 1.0e-14);
        VSS_CHECK(std::abs(secondMoment.value() - (1.0 / 3.0)) < 1.0e-11);
      }
    }
    const auto zeroMass = truncatedDistribution(
        uniformLaw.value(), IntervalSupport{3.0, 4.0, true, true}, policy());
    VSS_CHECK(!zeroMass.hasValue());
    VSS_CHECK(zeroMass.status().code() == ErrorCode::InvalidDomain);
    const auto limited = truncatedDistribution(
        uniformLaw.value(), IntervalSupport{0.0, 1.0, true, true}, policy(),
        Limits{0U, 0U, 0U});
    VSS_CHECK(!limited.hasValue());
    VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
  }

  const std::array<Scalar, 3> atomValues{0.0, 1.0, 2.0};
  const std::array<Scalar, 3> atomWeights{1.0, 2.0, 1.0};
  const auto atomValueView =
      ConstVectorView::create(atomValues, atomValues.size());
  const auto atomWeightView =
      ConstVectorView::create(atomWeights, atomWeights.size());
  const auto empiricalLaw = empirical(
      atomValueView.value(), atomWeightView.value(), {32U, 4096U, 256U});
  if (empiricalLaw) {
    const auto atomTruncation = truncatedDistribution(
        empiricalLaw.value(), IntervalSupport{1.0, 2.0, true, false}, policy());
    VSS_CHECK(atomTruncation.hasValue());
    if (atomTruncation) {
      const auto atOne = evaluate(atomTruncation.value(), 1.0, policy());
      VSS_CHECK(atOne.hasValue());
      if (atOne) {
        VSS_CHECK(atOne.value().density_or_mass == 1.0);
        VSS_CHECK(atOne.value().cdf == 1.0);
      }
      VSS_CHECK(
          std::abs(rawMoment(atomTruncation.value(), 1U, policy()).value() -
                   1.0) < 1.0e-14);
    }
  }

  const auto coin = bernoulli(0.25);
  const auto normalLaw = normal(0.0, 1.0);
  VSS_CHECK(coin.hasValue() && normalLaw.hasValue());
  if (coin && normalLaw) {
    const auto atom = censoredLogProbability(
        coin.value(), IntervalSupport{1.0, 1.0, true, true}, policy());
    const auto halfNormal = censoredLogProbability(
        normalLaw.value(),
        IntervalSupport{-std::numeric_limits<Scalar>::infinity(), 0.0, false,
                        true},
        policy());
    const auto empty = censoredLogProbability(
        normalLaw.value(), IntervalSupport{1.0, 1.0, false, false}, policy());
    const auto bin = quantizedLikelihood(
        normalLaw.value(), IntervalSupport{-0.5, 0.5, true, true}, policy());
    const auto lowerOnly = quantizedLikelihood(
        coin.value(), IntervalSupport{0.0, 1.0, true, false}, policy());
    const auto upperOnly = censoredLogProbability(
        coin.value(), IntervalSupport{0.0, 1.0, false, true}, policy());
    VSS_CHECK(atom.hasValue() && halfNormal.hasValue() && empty.hasValue() &&
              bin.hasValue());
    VSS_CHECK(lowerOnly.hasValue() && upperOnly.hasValue());
    if (atom && halfNormal && empty && bin && lowerOnly && upperOnly) {
      VSS_CHECK(std::abs(atom.value() - std::log(0.25)) < 1.0e-14);
      VSS_CHECK(std::abs(halfNormal.value() - std::log(0.5)) < 1.0e-14);
      VSS_CHECK(std::abs(lowerOnly.value() - std::log(0.75)) < 1.0e-14);
      VSS_CHECK(std::abs(upperOnly.value() - std::log(0.25)) < 1.0e-14);
      VSS_CHECK(std::isinf(empty.value()) && empty.value() < 0.0);
      VSS_CHECK(std::abs(bin.value() -
                         std::log(std::erf(0.5 / std::sqrt(2.0)))) < 2.0e-10);
    }
    const auto reversed = quantizedLikelihood(
        normalLaw.value(), IntervalSupport{2.0, 1.0, true, true}, policy());
    VSS_CHECK(!reversed.hasValue());
    VSS_CHECK(reversed.status().code() == ErrorCode::InvalidDomain);
  }
}
} // namespace

int main() {
  publicSignatures();
  empiricalCdfAndMixtureLogDensity();
  truncationAndIntervalLikelihoods();
}

// VSS-MATH-TRACEABILITY: C062 CASE-C062.empirical-cdf
// VSS-MATH-TRACEABILITY: C062 CASE-C062.mixture-logpdf
// VSS-MATH-TRACEABILITY: C062 CASE-C062.truncated-wrapper
// VSS-MATH-TRACEABILITY: C062 CASE-P09.truncation.mass
// VSS-MATH-TRACEABILITY: C062 CASE-P09.truncation.zero-mass
// VSS-MATH-TRACEABILITY: C062 CASE-P09.truncation.limit
// VSS-MATH-TRACEABILITY: C090 CASE-C090.censored-log-probability
// VSS-MATH-TRACEABILITY: C090 CASE-C090.quantized-likelihood
