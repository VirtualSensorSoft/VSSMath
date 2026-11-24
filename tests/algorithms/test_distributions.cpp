#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/DistributionMoments.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

ProbabilityPolicy policy() {
  return {{1.0e-13, 1.0e-13, 1000U, 10000U, std::nullopt}, 10000U, 2.0e-9};
}

void factoriesAcceptNominalFamilies() {
  VSS_CHECK(normal(0.0, 1.0).hasValue());
  VSS_CHECK(studentT(3.0, 0.0, 1.0).hasValue());
  VSS_CHECK(laplace(0.0, 1.0).hasValue());
  VSS_CHECK(logistic(0.0, 1.0).hasValue());
  VSS_CHECK(lognormal(0.0, 1.0).hasValue());
  VSS_CHECK(gamma(2.0, 1.0).hasValue());
  VSS_CHECK(beta(2.0, 3.0).hasValue());
  VSS_CHECK(weibull(2.0, 1.0).hasValue());
  VSS_CHECK(skewNormal(0.0, 1.0, 1.0).hasValue());
  VSS_CHECK(uniform(-1.0, 2.0).hasValue());
  VSS_CHECK(triangular(0.0, 1.0, 2.0).hasValue());
  VSS_CHECK(
      truncatedNormal(0.0, 1.0, 0.0, std::numeric_limits<Scalar>::infinity())
          .hasValue());
  VSS_CHECK(bernoulli(0.25).hasValue());
  VSS_CHECK(binomial(2U, 0.25).hasValue());
  VSS_CHECK(poisson(1.0).hasValue());
  VSS_CHECK(negativeBinomial(1.0, 0.5).hasValue());
  VSS_CHECK(chiSquare(2.0).hasValue());
  VSS_CHECK(fisherF(2.0, 4.0).hasValue());
}

void factoriesRejectInvalidParameters() {
  VSS_CHECK(normal(0.0, 0.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(studentT(0.0, 0.0, 1.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(laplace(0.0, -1.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(logistic(0.0, 0.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(lognormal(0.0, 0.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(gamma(0.0, 1.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(beta(0.0, 1.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(weibull(2.0, 0.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(skewNormal(0.0, 0.0, 1.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(uniform(1.0, 1.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(triangular(0.0, 3.0, 2.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(bernoulli(1.1).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(binomial(2U, -0.1).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(poisson(-1.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(negativeBinomial(1.0, 0.0).status().code() ==
            ErrorCode::InvalidDomain);
  VSS_CHECK(chiSquare(0.0).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(fisherF(2.0, 0.0).status().code() == ErrorCode::InvalidDomain);
}

void evaluateNominalLawAnchors() {
  const auto gaussian = evaluate(normal(0.0, 1.0).value(), 0.0, policy());
  VSS_CHECK(gaussian.hasValue());
  if (gaussian) {
    VSS_CHECK(std::abs(gaussian.value().density_or_mass - 0.3989422804014327) <
              1.0e-14);
    VSS_CHECK(gaussian.value().cdf == 0.5);
  }
  const auto bernoulliLaw = bernoulli(0.25);
  VSS_CHECK(bernoulliLaw.hasValue());
  if (bernoulliLaw) {
    const auto atZero = evaluate(bernoulliLaw.value(), 0.0, policy());
    VSS_CHECK(atZero.hasValue());
    if (atZero) {
      VSS_CHECK(atZero.value().density_or_mass == 0.75);
      VSS_CHECK(atZero.value().cdf == 0.75);
      VSS_CHECK(atZero.value().on_support);
    }
  }
  const auto chiSquareLaw = chiSquare(2.0);
  VSS_CHECK(chiSquareLaw.hasValue());
  if (chiSquareLaw) {
    const auto result = evaluate(chiSquareLaw.value(), 4.0, policy());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(std::abs(result.value().survival - std::exp(-2.0)) < 1.0e-13);
    }
  }

  const auto checkLaw = [](Result<Distribution> law, Scalar x,
                           Scalar expectedDensity, Scalar expectedCdf,
                           Scalar expectedSurvival) {
    VSS_CHECK(law.hasValue());
    if (law) {
      const auto result = evaluate(law.value(), x, policy());
      VSS_CHECK(result.hasValue());
      if (result) {
        VSS_CHECK(std::abs(result.value().density_or_mass - expectedDensity) <
                  2.0e-12);
        VSS_CHECK(std::abs(result.value().cdf - expectedCdf) < 2.0e-12);
        VSS_CHECK(std::abs(result.value().survival - expectedSurvival) <
                  2.0e-12);
      }
    }
  };
  const Scalar normalDensity = 1.0 / std::sqrt(2.0 * 3.14159265358979323846);
  const Scalar expMinusOne = std::exp(-1.0);
  checkLaw(studentT(1.0, 0.0, 1.0), 0.0, 1.0 / 3.14159265358979323846, 0.5,
           0.5);
  checkLaw(laplace(0.0, 1.0), 0.0, 0.5, 0.5, 0.5);
  checkLaw(logistic(0.0, 1.0), 0.0, 0.25, 0.5, 0.5);
  checkLaw(lognormal(0.0, 1.0), 1.0, normalDensity, 0.5, 0.5);
  checkLaw(gamma(2.0, 1.0), 1.0, expMinusOne, 1.0 - 2.0 * expMinusOne,
           2.0 * expMinusOne);
  checkLaw(beta(2.0, 3.0), 0.4, 1.728, 0.5248, 0.4752);
  checkLaw(weibull(1.0, 2.0), 2.0, expMinusOne / 2.0, 1.0 - expMinusOne,
           expMinusOne);
  checkLaw(skewNormal(0.0, 1.0, 0.0), 0.0, normalDensity, 0.5, 0.5);
  checkLaw(uniform(-1.0, 2.0), 0.5, 1.0 / 3.0, 0.5, 0.5);
  checkLaw(triangular(0.0, 1.0, 2.0), 1.0, 1.0, 0.5, 0.5);
  checkLaw(
      truncatedNormal(0.0, 1.0, 0.0, std::numeric_limits<Scalar>::infinity()),
      0.0, std::sqrt(2.0 / 3.14159265358979323846), 0.0, 1.0);
  checkLaw(binomial(2U, 0.25), 0.0, 0.5625, 0.5625, 0.4375);
  checkLaw(poisson(1.0), 1.0, expMinusOne, 2.0 * expMinusOne,
           1.0 - 2.0 * expMinusOne);
  checkLaw(negativeBinomial(1.0, 0.5), 1.0, 0.25, 0.75, 0.25);
  checkLaw(fisherF(2.0, 2.0), 2.0, 1.0 / 9.0, 2.0 / 3.0, 1.0 / 3.0);
}

void quantilesMomentsSamplingAndComposites() {
  auto gaussian = normal(0.0, 1.0);
  auto coin = bernoulli(0.25);
  VSS_CHECK(gaussian.hasValue());
  VSS_CHECK(coin.hasValue());
  if (gaussian && coin) {
    const auto median = quantile(gaussian.value(), 0.5, policy());
    VSS_CHECK(median.hasValue());
    if (median) {
      VSS_CHECK(std::abs(median.value()) < 1.0e-8);
    }
    const auto discreteMedian = quantile(coin.value(), 0.5, policy());
    VSS_CHECK(discreteMedian.hasValue());
    if (discreteMedian) {
      VSS_CHECK(discreteMedian.value() == 0.0);
    }
    VSS_CHECK(std::abs(rawMoment(gaussian.value(), 0U, policy()).value() -
                       1.0) < 1.0e-14);
    VSS_CHECK(std::abs(rawMoment(gaussian.value(), 2U, policy()).value() -
                       1.0) < 1.0e-12);
    VSS_CHECK(std::abs(mean(gaussian.value(), policy(), {16U, 4096U, 1000U})
                           .value()) < 1.0e-12);
    VSS_CHECK(
        std::abs(
            variance(gaussian.value(), policy(), {16U, 4096U, 1000U}).value() -
            1.0) < 1.0e-12);
    VSS_CHECK(rawMoment(studentT(1.0, 0.0, 1.0).value(), 1U, policy())
                  .status()
                  .code() == ErrorCode::InvalidDomain);

    Rng first(12345U);
    Rng replay(12345U);
    const auto firstDraw = sample(coin.value(), first, policy());
    const auto replayDraw = sample(coin.value(), replay, policy());
    VSS_CHECK(firstDraw.hasValue() && replayDraw.hasValue());
    if (firstDraw && replayDraw) {
      VSS_CHECK(firstDraw.value() == replayDraw.value());
      VSS_CHECK(firstDraw.value() == 0.0 || firstDraw.value() == 1.0);
    }
  }

  const auto left = bernoulli(0.5);
  const auto right = bernoulli(0.5);
  VSS_CHECK(left.hasValue() && right.hasValue());
  if (left && right) {
    const std::array<Distribution, 2> terms{left.value(), right.value()};
    const auto convolved = finiteConvolution(terms, policy());
    VSS_CHECK(convolved.hasValue());
    if (convolved) {
      const auto atOne = evaluate(convolved.value(), 1.0, policy());
      VSS_CHECK(atOne.hasValue());
      if (atOne) {
        VSS_CHECK(std::abs(atOne.value().density_or_mass - 0.5) < 1.0e-14);
      }
    }
    const std::array<WeightedLaw, 2> components{
        {{left.value(), 0.25}, {right.value(), 0.75}}};
    const auto mixed = mixture(components, policy());
    VSS_CHECK(mixed.hasValue());
    if (mixed) {
      const auto atOne = evaluate(mixed.value(), 1.0, policy());
      VSS_CHECK(atOne.hasValue());
      if (atOne) {
        VSS_CHECK(std::abs(atOne.value().density_or_mass - 0.5) < 1.0e-14);
      }
    }
  }

  const std::array<Scalar, 3> observations{0.0, 2.0, 2.0};
  const std::array<Scalar, 3> masses{1.0, 1.0, 2.0};
  const auto observationView =
      ConstVectorView::create(observations, observations.size());
  const auto massView = ConstVectorView::create(masses, masses.size());
  VSS_CHECK(observationView.hasValue() && massView.hasValue());
  if (observationView && massView) {
    const auto law = empirical(observationView.value(), massView.value(),
                               {1024U, 16384U, 10000U});
    VSS_CHECK(law.hasValue());
    if (law) {
      const auto atZero = evaluate(law.value(), 0.0, policy());
      VSS_CHECK(atZero.hasValue());
      if (atZero) {
        VSS_CHECK(std::abs(atZero.value().density_or_mass - 0.25) < 1.0e-14);
        VSS_CHECK(atZero.value().cdf == 0.25);
      }
    }
  }
}

void convolutionRegimes() {
  const std::array<Distribution, 0> noTerms{};
  const auto empty = finiteConvolution(noTerms, policy());
  VSS_CHECK(empty.hasValue());
  if (empty) {
    const auto atZero = evaluate(empty.value(), 0.0, policy());
    VSS_CHECK(atZero.hasValue());
    if (atZero) {
      VSS_CHECK(atZero.value().density_or_mass == 1.0);
    }
  }

  const auto normalA = normal(1.0, 2.0);
  const auto normalB = normal(-2.0, 1.0);
  const std::array<Distribution, 2> normals{normalA.value(), normalB.value()};
  const auto normalSum = finiteConvolution(normals, policy());
  VSS_CHECK(normalSum.hasValue());
  if (normalSum) {
    VSS_CHECK(normalSum.value().family() == Family::normal);
    VSS_CHECK(
        std::abs(mean(normalSum.value(), policy(), {8U, 128U, 100U}).value() -
                 -1.0) < 1.0e-14);
    VSS_CHECK(
        std::abs(
            variance(normalSum.value(), policy(), {8U, 128U, 100U}).value() -
            5.0) < 1.0e-14);
  }

  const auto gaussian = normal(0.0, 1.0);
  const auto coin = bernoulli(0.5);
  const std::array<Distribution, 2> normalAndCoin{gaussian.value(),
                                                  coin.value()};
  const auto shiftedMixture = finiteConvolution(normalAndCoin, policy());
  VSS_CHECK(shiftedMixture.hasValue());
  if (shiftedMixture) {
    const auto atHalf = evaluate(shiftedMixture.value(), 0.5, policy());
    VSS_CHECK(atHalf.hasValue());
    if (atHalf) {
      const Scalar phi0 = 1.0 / std::sqrt(2.0 * 3.14159265358979323846);
      const Scalar phiHalf = phi0 * std::exp(-0.125);
      VSS_CHECK(std::abs(atHalf.value().density_or_mass -
                         0.5 * (phiHalf + phiHalf)) < 1.0e-12);
    }
    VSS_CHECK(
        std::abs(
            mean(shiftedMixture.value(), policy(), {8U, 128U, 100U}).value() -
            0.5) < 1.0e-12);
    VSS_CHECK(
        std::abs(variance(shiftedMixture.value(), policy(), {8U, 128U, 100U})
                     .value() -
                 1.25) < 1.0e-12);
  }

  const auto coinA = bernoulli(0.5);
  const auto coinB = bernoulli(0.25);
  const std::array<Distribution, 3> normalAndTwoCoins{
      gaussian.value(), coinA.value(), coinB.value()};
  const auto multiShift = finiteConvolution(normalAndTwoCoins, policy());
  VSS_CHECK(multiShift.hasValue());
  if (multiShift) {
    const auto atOne = evaluate(multiShift.value(), 1.0, policy());
    const auto atTwo = evaluate(multiShift.value(), 2.0, policy());
    const auto phi = [](Scalar x) {
      return std::exp(-0.5 * x * x) / std::sqrt(2.0 * 3.14159265358979323846);
    };
    VSS_CHECK(atOne.hasValue() && atTwo.hasValue());
    if (atOne && atTwo) {
      const Scalar expected =
          0.375 * phi(1.0) + 0.5 * phi(0.0) + 0.125 * phi(-1.0);
      VSS_CHECK(std::abs(atOne.value().density_or_mass - expected) < 1.0e-12);
      VSS_CHECK(std::abs(atTwo.value().density_or_mass -
                         (0.375 * phi(2.0) + 0.5 * phi(1.0) +
                          0.125 * phi(0.0))) < 1.0e-12);
      VSS_CHECK(
          std::abs(
              mean(multiShift.value(), policy(), {8U, 128U, 100U}).value() -
              0.75) < 1.0e-12);
      VSS_CHECK(
          std::abs(
              variance(multiShift.value(), policy(), {8U, 128U, 100U}).value() -
              1.4375) < 1.0e-12);
    }
  }

  const std::array<Distribution, 2> normalAndPoisson{gaussian.value(),
                                                     poisson(1.0).value()};
  const auto countMixture = finiteConvolution(normalAndPoisson, policy());
  VSS_CHECK(countMixture.hasValue());
  if (countMixture) {
    const auto atZero = evaluate(countMixture.value(), 0.0, policy());
    VSS_CHECK(atZero.hasValue());
    if (atZero) {
      long double mass = std::exp(-1.0L);
      long double density = 0.0L;
      long double cdf = 0.0L;
      long double survival = 0.0L;
      for (Index count = 0U; count <= 20U; ++count) {
        const long double coordinate = -static_cast<long double>(count);
        const long double normalDensity =
            std::exp(-coordinate * coordinate / 2.0L) /
            std::sqrt(2.0L * std::numbers::pi_v<long double>);
        const long double normalCdf =
            0.5L * std::erfc(-coordinate / std::sqrt(2.0L));
        density += mass * normalDensity;
        cdf += mass * normalCdf;
        survival += mass * (1.0L - normalCdf);
        mass /= static_cast<long double>(count + 1U);
      }
      VSS_CHECK(std::abs(atZero.value().density_or_mass - density) < 1.0e-12);
      VSS_CHECK(std::abs(atZero.value().cdf - cdf) < 1.0e-12);
      VSS_CHECK(std::abs(atZero.value().survival - survival) < 1.0e-12);
    }
  }

  const std::array<Distribution, 2> commonScaleGammas{gamma(2.0, 2.0).value(),
                                                      chiSquare(4.0).value()};
  const auto gammaSum = finiteConvolution(commonScaleGammas, policy());
  VSS_CHECK(gammaSum.hasValue());
  if (gammaSum) {
    VSS_CHECK(gammaSum.value().family() == Family::gamma);
    const auto expected = gamma(4.0, 2.0);
    const auto actualAtFive = evaluate(gammaSum.value(), 5.0, policy());
    const auto expectedAtFive = evaluate(expected.value(), 5.0, policy());
    VSS_CHECK(actualAtFive.hasValue() && expectedAtFive.hasValue());
    if (actualAtFive && expectedAtFive) {
      VSS_CHECK(std::abs(actualAtFive.value().cdf -
                         expectedAtFive.value().cdf) < 1.0e-13);
      VSS_CHECK(std::abs(actualAtFive.value().density_or_mass -
                         expectedAtFive.value().density_or_mass) < 1.0e-13);
    }
  }

  const auto uniformA = uniform(0.0, 1.0);
  const auto uniformB = uniform(0.0, 1.0);
  const std::array<Distribution, 2> uniforms{uniformA.value(),
                                             uniformB.value()};
  const auto triangular = finiteConvolution(uniforms, policy());
  VSS_CHECK(triangular.hasValue());
  if (triangular) {
    const auto atOne = evaluate(triangular.value(), 1.0, policy());
    VSS_CHECK(atOne.hasValue());
    if (atOne) {
      VSS_CHECK(std::abs(atOne.value().density_or_mass - 1.0) < 1.0e-14);
      VSS_CHECK(std::abs(atOne.value().cdf - 0.5) < 1.0e-14);
      VSS_CHECK(std::abs(atOne.value().survival - 0.5) < 1.0e-14);
    }
  }
}

void familyQuantileMomentAndReplayGrid() {
  const std::array<Distribution, 18> laws{
      normal(1.0, 2.0).value(),
      studentT(5.0, 0.0, 1.0).value(),
      laplace(0.5, 1.5).value(),
      logistic(-1.0, 0.75).value(),
      lognormal(0.0, 0.5).value(),
      gamma(2.0, 1.5).value(),
      beta(2.0, 3.0).value(),
      weibull(2.0, 1.5).value(),
      skewNormal(0.0, 1.0, 1.5).value(),
      uniform(-2.0, 3.0).value(),
      triangular(-1.0, 0.25, 2.0).value(),
      truncatedNormal(0.0, 1.0, -2.0, 3.0).value(),
      bernoulli(0.25).value(),
      binomial(10U, 0.4).value(),
      poisson(3.0).value(),
      negativeBinomial(3.0, 0.5).value(),
      chiSquare(4.0).value(),
      fisherF(5.0, 10.0).value()};
  for (const auto& law : laws) {
    const auto median = quantile(law, 0.5, policy());
    VSS_CHECK(median.hasValue());
    if (!median) {
      continue;
    }
    const auto atMedian = evaluate(law, median.value(), policy());
    VSS_CHECK(atMedian.hasValue());
    if (atMedian) {
      VSS_CHECK(atMedian.value().cdf >= 0.5);
      if (law.measure() == Measure::integer_mass) {
        const auto below = evaluate(law, median.value() - 1.0, policy());
        VSS_CHECK(below.hasValue());
        if (below) {
          VSS_CHECK(below.value().cdf < 0.5);
        }
      }
    }
    Rng first(0xA123456789ABCDEFULL);
    Rng second(0xA123456789ABCDEFULL);
    const auto draw1 = sample(law, first, policy());
    const auto draw2 = sample(law, second, policy());
    VSS_CHECK(draw1.hasValue() && draw2.hasValue());
    if (draw1 && draw2) {
      VSS_CHECK(std::isfinite(draw1.value()));
      VSS_CHECK(draw1.value() == draw2.value());
    }
    const auto firstMoment = rawMoment(law, 1U, policy());
    const auto secondMoment = rawMoment(law, 2U, policy());
    const auto lawMean = mean(law, policy(), {32U, 1024U, 10000U});
    const auto lawVariance = variance(law, policy(), {32U, 1024U, 10000U});
    VSS_CHECK(firstMoment.hasValue() && secondMoment.hasValue());
    VSS_CHECK(lawMean.hasValue() && lawVariance.hasValue());
    if (firstMoment && lawMean) {
      VSS_CHECK(std::abs(firstMoment.value() - lawMean.value()) < 1.0e-10);
    }
    if (firstMoment && secondMoment && lawVariance) {
      VSS_CHECK(std::abs(secondMoment.value() -
                         firstMoment.value() * firstMoment.value() -
                         lawVariance.value()) < 1.0e-8);
    }
  }
}

} // namespace

int main() {
  factoriesAcceptNominalFamilies();
  factoriesRejectInvalidParameters();
  evaluateNominalLawAnchors();
  quantilesMomentsSamplingAndComposites();
  convolutionRegimes();
  familyQuantileMomentAndReplayGrid();
}

// VSS-MATH-TRACEABILITY: C059 CASE-C059.normal.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.normal.factory.invalid
// VSS-MATH-TRACEABILITY: C059 CASE-C059.bernoulli.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.bernoulli.factory.invalid
// VSS-MATH-TRACEABILITY: C059 CASE-C059.chi-square.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.chi-square.factory.invalid
