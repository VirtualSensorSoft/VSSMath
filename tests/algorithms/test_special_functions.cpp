#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Special.hpp>

namespace {

using namespace vss::math;
using namespace vss::math::probability;

ProbabilityPolicy policy() {
  return {{1.0e-13, 1.0e-13, 1000U, 10000U, std::nullopt}, 10000U, 2.0e-9};
}

void close(Scalar actual, Scalar expected, Scalar tolerance = 2.0e-12) {
  VSS_CHECK(std::abs(actual - expected) <=
            tolerance * std::max(Scalar{1}, std::abs(expected)));
}

void logGammaReference() {
  const std::array<std::pair<Scalar, Scalar>, 4> references{{
      {0.5, 0.572364942924700087},
      {1.0, 0.0},
      {5.0, std::log(24.0)},
      {10.0, 12.8018274800814696},
  }};
  for (const auto& [argument, expected] : references) {
    const auto result = logGamma(argument, policy());
    VSS_CHECK(result.hasValue());
    if (result) {
      close(result.value(), expected);
    }
  }
}

void incompleteGammaPairedTails() {
  const auto result = incompleteGamma(1.0, 2.0, policy());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(std::exp(result.value().log_survival), std::exp(-2.0));
    close(std::exp(result.value().log_cdf) +
              std::exp(result.value().log_survival),
          1.0);
  }
  const auto integerShape = incompleteGamma(2.0, 1.0, policy());
  VSS_CHECK(integerShape.hasValue());
  if (integerShape) {
    close(std::exp(integerShape.value().log_cdf), 1.0 - 2.0 / std::numbers::e);
    close(std::exp(integerShape.value().log_survival), 2.0 / std::numbers::e);
  }
  const auto halfShape = incompleteGamma(0.5, 0.5, policy());
  VSS_CHECK(halfShape.hasValue());
  if (halfShape) {
    close(std::exp(halfShape.value().log_cdf), std::erf(std::sqrt(0.5)));
    close(std::exp(halfShape.value().log_survival), std::erfc(std::sqrt(0.5)));
  }
  const auto extremeTail = incompleteGamma(1.0, 100.0, policy());
  VSS_CHECK(extremeTail.hasValue());
  if (extremeTail) {
    close(extremeTail.value().log_survival, -100.0);
  }
  const auto lowerExtreme = incompleteGamma(1000.0, 900.0, policy());
  VSS_CHECK(lowerExtreme.hasValue());
  if (lowerExtreme) {
    close(lowerExtreme.value().log_cdf, -7.5057699942338924, 2.0e-11);
    close(lowerExtreme.value().log_survival, -0.0005500535174143447, 2.0e-11);
  }
  const auto upperExtreme = incompleteGamma(1000.0, 1100.0, policy());
  VSS_CHECK(upperExtreme.hasValue());
  if (upperExtreme) {
    close(upperExtreme.value().log_cdf, -0.0010598847333686436, 2.0e-11);
    close(upperExtreme.value().log_survival, -6.8501250144362653, 2.0e-11);
  }
}

void incompleteBetaPairedTails() {
  const auto result = incompleteBeta(1.0, 1.0, 0.25, policy());
  VSS_CHECK(result.hasValue());
  if (result) {
    close(std::exp(result.value().log_cdf), 0.25);
    close(std::exp(result.value().log_survival), 0.75);
  }
  const auto asymmetric = incompleteBeta(2.0, 3.0, 0.4, policy());
  VSS_CHECK(asymmetric.hasValue());
  if (asymmetric) {
    close(std::exp(asymmetric.value().log_cdf), 0.5248);
    close(std::exp(asymmetric.value().log_survival), 0.4752);
  }
  const auto lowerTail = incompleteBeta(0.1, 0.3, 0.01, policy());
  VSS_CHECK(lowerTail.hasValue());
  if (lowerTail) {
    close(lowerTail.value().log_cdf, -0.7091257443001208, 2.0e-11);
    close(lowerTail.value().log_survival, -0.6774199211001502, 2.0e-11);
  }
  const auto largeShapes = incompleteBeta(100.0, 120.0, 0.45, policy());
  VSS_CHECK(largeShapes.hasValue());
  if (largeShapes) {
    close(largeShapes.value().log_cdf, -0.8034058277294568, 2.0e-11);
    close(largeShapes.value().log_survival, -0.5938472072098748, 2.0e-11);
  }
}

void logSumExpStableAndBoundaryCases() {
  const std::array<Scalar, 3> stable{1000.0, 1000.0,
                                     -std::numeric_limits<Scalar>::infinity()};
  const auto stableView = ConstVectorView::create(stable, stable.size());
  VSS_CHECK(stableView.hasValue());
  if (stableView) {
    const auto result = logSumExp(stableView.value());
    VSS_CHECK(result.hasValue());
    if (result) {
      close(result.value(), 1000.0 + std::numbers::ln2);
    }
  }

  const std::array<Scalar, 2> largest{std::numeric_limits<Scalar>::max(),
                                      std::numeric_limits<Scalar>::max()};
  const auto largestView = ConstVectorView::create(largest, largest.size());
  VSS_CHECK(largestView.hasValue());
  if (largestView) {
    const auto result = logSumExp(largestView.value());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(result.value() == std::numeric_limits<Scalar>::max());
    }
  }

  const std::array<Scalar, 2> zeroMass{
      -std::numeric_limits<Scalar>::infinity(),
      -std::numeric_limits<Scalar>::infinity()};
  const auto zeroView = ConstVectorView::create(zeroMass, zeroMass.size());
  VSS_CHECK(zeroView.hasValue());
  if (zeroView) {
    const auto result = logSumExp(zeroView.value());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(result.value() == -std::numeric_limits<Scalar>::infinity());
    }
  }

  const auto empty = ConstVectorView::create({}, 0U);
  VSS_CHECK(empty.hasValue());
  if (empty) {
    const auto result = logSumExp(empty.value());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(result.value() == -std::numeric_limits<Scalar>::infinity());
    }
  }
}

void endpointsDomainsAndWorkLimits() {
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  const auto gammaZero = incompleteGamma(2.0, 0.0, policy());
  VSS_CHECK(gammaZero.hasValue());
  if (gammaZero) {
    VSS_CHECK(gammaZero.value().log_cdf == negativeInfinity);
    VSS_CHECK(gammaZero.value().log_survival == 0.0);
  }
  const auto gammaInfinity =
      incompleteGamma(2.0, std::numeric_limits<Scalar>::infinity(), policy());
  VSS_CHECK(gammaInfinity.hasValue());
  if (gammaInfinity) {
    VSS_CHECK(gammaInfinity.value().log_cdf == 0.0);
    VSS_CHECK(gammaInfinity.value().log_survival == negativeInfinity);
  }
  VSS_CHECK(logGamma(0.0, policy()).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(incompleteGamma(0.0, 1.0, policy()).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(incompleteGamma(1.0, -1.0, policy()).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(incompleteBeta(1.0, 1.0, 0.0, policy()).value().log_cdf ==
            negativeInfinity);
  VSS_CHECK(incompleteBeta(1.0, 1.0, 1.0, policy()).value().log_survival ==
            negativeInfinity);
  VSS_CHECK(incompleteBeta(1.0, 1.0, 1.1, policy()).status().code() ==
            ErrorCode::InvalidDomain);

  auto noIterations = policy();
  noIterations.iteration.max_iterations = 0U;
  VSS_CHECK(incompleteGamma(1.0, 2.0, noIterations).status().code() ==
            ErrorCode::InvalidDomain);
  noIterations.iteration.max_iterations = 1000U;
  noIterations.max_draws = 0U;
  VSS_CHECK(logGamma(5.0, noIterations).hasValue());
}

void logSumExpRejectsPositiveInfinity() {
  const std::array<Scalar, 1> terms{std::numeric_limits<Scalar>::infinity()};
  const auto view = ConstVectorView::create(terms, terms.size());
  VSS_CHECK(view.hasValue());
  if (view) {
    VSS_CHECK(logSumExp(view.value()).status().code() ==
              ErrorCode::NonFiniteInput);
  }
}

void publicSignatures() {
  using GammaSignature =
      Result<LogTailPair> (*)(Scalar, Scalar, ProbabilityPolicy);
  using BetaSignature =
      Result<LogTailPair> (*)(Scalar, Scalar, Scalar, ProbabilityPolicy);
  using LogGammaSignature = Result<Scalar> (*)(Scalar, ProbabilityPolicy);
  using LogSumExpSignature = Result<Scalar> (*)(ConstVectorView);
  static_assert(
      std::is_same_v<decltype(static_cast<GammaSignature>(&incompleteGamma)),
                     GammaSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<BetaSignature>(&incompleteBeta)),
                     BetaSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<LogGammaSignature>(&logGamma)),
                     LogGammaSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<LogSumExpSignature>(&logSumExp)),
                     LogSumExpSignature>);
}

} // namespace

int main() {
  publicSignatures();
  logGammaReference();
  incompleteGammaPairedTails();
  incompleteBetaPairedTails();
  logSumExpStableAndBoundaryCases();
  endpointsDomainsAndWorkLimits();
  logSumExpRejectsPositiveInfinity();
}

// VSS-MATH-TRACEABILITY: C063 CASE-C063.log-gamma
// VSS-MATH-TRACEABILITY: C063 CASE-C063.incomplete-gamma
// VSS-MATH-TRACEABILITY: C063 CASE-C063.incomplete-beta
// VSS-MATH-TRACEABILITY: C063 CASE-C063.logsumexp.empty
// VSS-MATH-TRACEABILITY: C063 CASE-C063.logsumexp.maxfinite
// VSS-MATH-TRACEABILITY: C063 CASE-C063.logsumexp.stable
// VSS-MATH-TRACEABILITY: C063 CASE-C063.logsumexp.zero-mass
