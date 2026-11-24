#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
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

  const std::array<Scalar, 2> tailObservations{0.0, 1.0};
  const std::array<Scalar, 2> tailMasses{1.0e200, 1.0e188};
  const auto tailObservationView =
      ConstVectorView::create(tailObservations, tailObservations.size());
  const auto tailMassView =
      ConstVectorView::create(tailMasses, tailMasses.size());
  const auto tailLaw = empirical(tailObservationView.value(),
                                 tailMassView.value(), {32U, 1024U, 100U});
  VSS_CHECK(tailLaw.hasValue());
  if (tailLaw) {
    const auto atFirstAtom = evaluate(tailLaw.value(), 0.0, policy());
    VSS_CHECK(atFirstAtom.hasValue());
    if (atFirstAtom) {
      VSS_CHECK(std::abs(atFirstAtom.value().survival - 1.0e-12) < 3.0e-24);
    }
    VSS_CHECK(quantile(tailLaw.value(), 1.0 - 5.0e-13, policy()).value() ==
              1.0);
  }
}

void compositeInputLimitsPreflight() {
  const auto first = normal(0.0, 1.0);
  const auto second = normal(1.0, 1.0);
  VSS_CHECK(first.hasValue() && second.hasValue());
  if (first && second) {
    const std::array<WeightedLaw, 2> terms{WeightedLaw{first.value(), 0.5},
                                           WeightedLaw{second.value(), 0.5}};
    const auto limited = mixture(terms, policy(), {1U, 4096U, 100U});
    VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
  }

  const std::array<Scalar, 2> observations{1.0, 2.0};
  const std::array<Scalar, 2> masses{1.0, 1.0};
  const auto observationView = ConstVectorView::create(observations, 2U);
  const auto massView = ConstVectorView::create(masses, 2U);
  VSS_CHECK(observationView.hasValue() && massView.hasValue());
  if (observationView && massView) {
    const auto limited =
        empirical(observationView.value(), massView.value(), {1U, 4096U, 100U});
    VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);

    const auto oneByteShort = empirical(
        observationView.value(), massView.value(),
        {8U, 4U * 2U * sizeof(Scalar) - 1U, 100U});
    VSS_CHECK(oneByteShort.status().code() == ErrorCode::ResourceLimit);
    const auto exactStorage = empirical(
        observationView.value(), massView.value(),
        {8U, 4U * 2U * sizeof(Scalar), 100U});
    VSS_CHECK(exactStorage.hasValue());
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
    auto cappedPolicy = policy();
    cappedPolicy.iteration.max_evaluations = 1U;
    VSS_CHECK(
        evaluate(countMixture.value(), 0.0, cappedPolicy).status().code() ==
        ErrorCode::ResourceLimit);
  }

  const std::array<Distribution, 2> poissonAndGamma{
      gamma(0.5, 1.0).value(), poisson(1.0).value()};
  const auto poleMixture = finiteConvolution(poissonAndGamma, policy());
  VSS_CHECK(poleMixture.hasValue());
  if (poleMixture) {
    const Scalar sqrtPi = std::sqrt(std::numbers::pi_v<Scalar>);
    const auto referenceAt = [sqrtPi](Scalar x) {
      Scalar mass = std::exp(-1.0);
      Scalar density = 0.0;
      Scalar cdf = 0.0;
      Scalar survival = 0.0;
      Scalar cumulativeCountMass = 0.0;
      const auto lastCount = static_cast<Index>(std::floor(x));
      for (Index count = 0U; count <= lastCount; ++count) {
        const Scalar residual = x - static_cast<Scalar>(count);
        if (residual == 0.0) {
          density = std::numeric_limits<Scalar>::infinity();
          survival += mass;
        } else {
          density += mass * std::exp(-residual) /
                     (sqrtPi * std::sqrt(residual));
          cdf += mass * std::erf(std::sqrt(residual));
          survival += mass * std::erfc(std::sqrt(residual));
        }
        cumulativeCountMass += mass;
        mass /= static_cast<Scalar>(count + 1U);
      }
      survival += std::max(0.0, 1.0 - cumulativeCountMass);
      return std::array<Scalar, 3>{density, cdf, survival};
    };
    for (const Scalar x : {0.0, 1.0, 2.0, 1.0 - 0x1p-16, 1.0 + 0x1p-16,
                           2.0 - 0x1p-16, 2.0 + 0x1p-16}) {
      const auto actual = evaluate(poleMixture.value(), x, policy());
      const auto expected = referenceAt(x);
      VSS_CHECK(actual.hasValue());
      if (actual) {
        if (std::isinf(expected[0])) {
          VSS_CHECK(actual.value().density_or_mass == expected[0]);
        } else {
          VSS_CHECK(std::abs(actual.value().density_or_mass - expected[0]) <=
                    2.0e-12 * expected[0]);
        }
        VSS_CHECK(std::abs(actual.value().cdf - expected[1]) < 2.0e-14);
        VSS_CHECK(std::abs(actual.value().survival - expected[2]) < 2.0e-14);
      }
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

  for (const auto [firstShape, secondShape] :
       {std::pair{0.25, 0.25}, std::pair{0.5, 0.5}, std::pair{0.75, 0.75},
        std::pair{0.25, 2.0}}) {
    const std::array<Distribution, 2> terms{gamma(firstShape, 1.0).value(),
                                            gamma(secondShape, 1.0).value()};
    const auto sum = finiteConvolution(terms, policy());
    VSS_CHECK(sum.hasValue());
    if (!sum) {
      continue;
    }
    const Scalar shape = firstShape + secondShape;
    const auto atZero = evaluate(sum.value(), 0.0, policy());
    VSS_CHECK(atZero.hasValue());
    if (atZero) {
      const Scalar expectedEndpoint =
          shape < 1.0    ? std::numeric_limits<Scalar>::infinity()
          : shape == 1.0 ? 1.0
                         : 0.0;
      VSS_CHECK(atZero.value().density_or_mass == expectedEndpoint);
    }
    const auto nearZero = evaluate(sum.value(), 1.0e-6, policy());
    const auto interior = evaluate(sum.value(), 2.0, policy());
    VSS_CHECK(nearZero.hasValue() && interior.hasValue());
    if (nearZero && interior) {
      const auto reference = gamma(shape, 1.0);
      VSS_CHECK(reference.hasValue());
      if (reference) {
        const auto expectedNearZero =
            evaluate(reference.value(), 1.0e-6, policy());
        const auto expectedInterior =
            evaluate(reference.value(), 2.0, policy());
        VSS_CHECK(expectedNearZero.hasValue() && expectedInterior.hasValue());
        if (expectedNearZero && expectedInterior) {
          VSS_CHECK(std::abs(nearZero.value().density_or_mass -
                             expectedNearZero.value().density_or_mass) <
                    1.0e-12);
          VSS_CHECK(std::abs(interior.value().cdf -
                             expectedInterior.value().cdf) < 1.0e-12);
        }
      }
    }
  }

  const auto betaGammaDensityReference = [](Scalar a, Scalar b, Scalar x) {
    if (x == 0.0) {
      const Scalar exponent = a + b - 1.0;
      if (exponent < 0.0) {
        return std::numeric_limits<Scalar>::infinity();
      }
      return exponent == 0.0 ? std::tgamma(a + 1.0) : 0.0;
    }
    if (x == 1.0 && a + b <= 1.0) {
      return std::numeric_limits<Scalar>::infinity();
    }
    constexpr Index midpointCount = 131072U;
    const long double normalizer =
        static_cast<long double>(a) / std::tgamma(static_cast<long double>(b));
    const auto integrate = [&](Scalar lower, Scalar upper, Index power,
                               bool zeroLower, bool collision) {
      if (!(upper > lower)) {
        return 0.0L;
      }
      long double total = 0.0L;
      const long double width = upper - lower;
      for (Index index = 0U; index <= midpointCount; ++index) {
        if (index == 0U && zeroLower) {
          continue;
        }
        const long double z = static_cast<long double>(index) / midpointCount;
        const long double q = lower + width * std::pow(z, power);
        const long double jacobian =
            width * power * std::pow(z, static_cast<long double>(power - 1U));
        const long double betaGap = x < 1.0 ? (1.0L - x) + q : q;
        const long double gammaCoordinate = x < 1.0 ? q : (x - 1.0L) + q;
        const long double exponent =
            collision ? a + b - 2.0L : (x < 1.0 ? b - 1.0L : a - 1.0L);
        const long double integrand =
            normalizer * (collision ? 1.0L : std::pow(betaGap, a - 1.0L)) *
            (collision ? std::pow(q, exponent)
                       : std::pow(gammaCoordinate, b - 1.0L)) *
            std::exp(-gammaCoordinate) * jacobian;
        const long double weight = index == 0U || index == midpointCount ? 1.0L
                                   : index % 2U == 0U                    ? 2.0L
                                                                         : 4.0L;
        total += weight * integrand;
      }
      return total / (3.0L * midpointCount);
    };
    long double density = 0.0L;
    if (x < 1.0) {
      const Scalar split = std::min(x, 1.0 - x);
      const Index power = static_cast<Index>(std::ceil(2.0 / b));
      density += integrate(0.0, split, power, true, false);
      for (Scalar lower = split; lower < x;) {
        const Scalar upper = std::min(x, 2.0 * lower);
        density += integrate(lower, upper, 1U, false, false);
        lower = upper;
      }
    } else if (x > 1.0) {
      const Scalar split = std::min(1.0, x - 1.0);
      const Index power = static_cast<Index>(std::ceil(2.0 / a));
      density += integrate(0.0, split, power, true, false);
      for (Scalar lower = split; lower < 1.0;) {
        const Scalar upper = std::min(1.0, 2.0 * lower);
        density += integrate(lower, upper, 1U, false, false);
        lower = upper;
      }
    } else {
      const Scalar alpha = a + b - 1.0;
      const Index power = static_cast<Index>(std::ceil(2.0 / alpha));
      density += integrate(0.0, 1.0, power, true, true);
    }
    return static_cast<Scalar>(density);
  };
  const auto betaGammaCollisionTailReference = [](Scalar a, Scalar b) {
    constexpr Index intervalCount = 16384U;
    const Index power =
        std::max<Index>(2U, static_cast<Index>(std::ceil(2.0 / a)));
    const Distribution gammaLaw = gamma(b, 1.0).value();
    std::array<long double, 2> total{};
    for (Index index = 0U; index <= intervalCount; ++index) {
      if (index == 0U) {
        continue;
      }
      const long double z = static_cast<long double>(index) / intervalCount;
      const long double coordinate = std::pow(z, power);
      const long double jacobian = power * std::pow(z, power - 1U);
      const long double density = a * std::pow(coordinate, a - 1.0L) * jacobian;
      const auto gammaTails =
          evaluate(gammaLaw, static_cast<Scalar>(coordinate), policy());
      const long double weight = index == intervalCount ? 1.0L
                                 : index % 2U == 0U     ? 2.0L
                                                        : 4.0L;
      total[0] += weight * density * gammaTails.value().cdf;
      total[1] += weight * density * gammaTails.value().survival;
    }
    return std::array<Scalar, 2>{
        static_cast<Scalar>(total[0] / (3.0L * intervalCount)),
        static_cast<Scalar>(total[1] / (3.0L * intervalCount))};
  };
  for (const auto [betaShape, gammaShape] :
       {std::pair{0.25, 0.25}, std::pair{0.5, 0.5}, std::pair{0.75, 0.75},
        std::pair{0.25, 0.4}, std::pair{0.72, 0.81}}) {
    const std::array<Distribution, 2> terms{beta(1.0, betaShape).value(),
                                            gamma(gammaShape, 1.0).value()};
    const auto sum = finiteConvolution(terms, policy());
    VSS_CHECK(sum.hasValue());
    if (!sum) {
      continue;
    }
    auto compositePolicy = policy();
    compositePolicy.iteration.rel = 2.0e-11;
    compositePolicy.iteration.max_evaluations = 100000U;
    for (const Scalar x :
         {0.0, 0.5, 1.0 - 0x1p-8, 1.0 - 0x1p-16, 1.0 - 0x1p-24, 1.0,
          1.0 + 0x1p-24, 1.0 + 0x1p-16, 1.0 + 0x1p-8, 2.0, 10.0}) {
      const auto value = evaluate(sum.value(), x, compositePolicy);
      VSS_CHECK(value.hasValue());
      if (value) {
        const Scalar expected =
            betaGammaDensityReference(betaShape, gammaShape, x);
        if (std::isinf(expected)) {
          VSS_CHECK(value.value().density_or_mass == expected);
        } else {
          VSS_CHECK(std::abs(value.value().density_or_mass - expected) <=
                    5.0e-11 * std::max(1.0, expected));
        }
        VSS_CHECK(std::abs(value.value().cdf + value.value().survival - 1.0) <
                  2.0e-12);
        if (x == 1.0 && betaShape <= 0.5) {
          const auto expectedTails =
              betaGammaCollisionTailReference(betaShape, gammaShape);
          VSS_CHECK(std::abs(value.value().cdf - expectedTails[0]) < 5.0e-11);
          VSS_CHECK(std::abs(value.value().survival - expectedTails[1]) <
                    5.0e-11);
        }
      }
    }
  }
  const std::array<Distribution, 2> reversedBetaGammaTerms{
      gamma(0.72, 1.0).value(), beta(1.0, 0.81).value()};
  const auto reversedBetaGamma =
      finiteConvolution(reversedBetaGammaTerms, policy());
  const std::array<Distribution, 2> orderedBetaGammaTerms{
      beta(1.0, 0.81).value(), gamma(0.72, 1.0).value()};
  const auto orderedBetaGamma =
      finiteConvolution(orderedBetaGammaTerms, policy());
  VSS_CHECK(reversedBetaGamma.hasValue() && orderedBetaGamma.hasValue());
  if (reversedBetaGamma && orderedBetaGamma) {
    for (const Scalar x : {0.5, 1.0, 2.0}) {
      const auto reversed = evaluate(reversedBetaGamma.value(), x, policy());
      const auto ordered = evaluate(orderedBetaGamma.value(), x, policy());
      VSS_CHECK(reversed.hasValue() && ordered.hasValue());
      if (reversed && ordered) {
        VSS_CHECK(std::abs(reversed.value().density_or_mass -
                           ordered.value().density_or_mass) < 2.0e-12);
        VSS_CHECK(std::abs(reversed.value().cdf - ordered.value().cdf) <
                  2.0e-12);
        VSS_CHECK(std::abs(reversed.value().survival -
                           ordered.value().survival) < 2.0e-12);
      }
    }
  }

  const auto checkBetaGammaOriginDensity = [&](Scalar betaShape,
                                               Scalar gammaShape,
                                               Scalar expectedDensity) {
    const std::array<Distribution, 2> terms{beta(1.0, betaShape).value(),
                                            gamma(gammaShape, 1.0).value()};
    const auto sum = finiteConvolution(terms, policy());
    VSS_CHECK(sum.hasValue());
    if (sum) {
      const auto value = evaluate(sum.value(), 0.0, policy());
      VSS_CHECK(value.hasValue());
      if (value) {
        if (std::isinf(expectedDensity)) {
          VSS_CHECK(value.value().density_or_mass == expectedDensity);
        } else {
          VSS_CHECK(std::abs(value.value().density_or_mass - expectedDensity) <
                    5.0e-14 * std::max(1.0, expectedDensity));
        }
        VSS_CHECK(value.value().cdf == 0.0);
        VSS_CHECK(value.value().survival == 1.0);
      }
    }
  };
  checkBetaGammaOriginDensity(0.25, 0.25,
                              std::numeric_limits<Scalar>::infinity());
  checkBetaGammaOriginDensity(0.25, 0.75, std::tgamma(1.25));
  checkBetaGammaOriginDensity(0.75, 0.75, 0.0);

  const std::array<Distribution, 2> extremeBetaGammaTerms{
      beta(1.0, 0.72).value(), gamma(0.81, 1.0).value()};
  const auto extremeBetaGamma =
      finiteConvolution(extremeBetaGammaTerms, policy());
  VSS_CHECK(extremeBetaGamma.hasValue());
  if (extremeBetaGamma) {
    const auto extremeLowerTails =
        evaluate(extremeBetaGamma.value(), 1.0e-300, policy());
    VSS_CHECK(extremeLowerTails.hasValue());
    if (extremeLowerTails) {
      VSS_CHECK(extremeLowerTails.value().cdf == 0.0);
      VSS_CHECK(std::isfinite(extremeLowerTails.value().log_tails.log_cdf));
      VSS_CHECK(extremeLowerTails.value().log_tails.log_cdf < -700.0);
      VSS_CHECK(std::abs(extremeLowerTails.value().log_tails.log_survival) <
                1.0e-12);
    }
    const auto extremeTails =
        evaluate(extremeBetaGamma.value(), 1000.0, policy());
    VSS_CHECK(extremeTails.hasValue());
    if (extremeTails) {
      VSS_CHECK(extremeTails.value().survival == 0.0);
      VSS_CHECK(std::isfinite(extremeTails.value().log_density_or_mass));
      VSS_CHECK(extremeTails.value().log_density_or_mass < -700.0);
      VSS_CHECK(std::isfinite(extremeTails.value().log_tails.log_survival));
      VSS_CHECK(extremeTails.value().log_tails.log_survival < -700.0);
      VSS_CHECK(std::abs(extremeTails.value().log_tails.log_cdf) < 1.0e-12);
    }
    const auto farTails = evaluate(extremeBetaGamma.value(), 1.0e6, policy());
    VSS_CHECK(farTails.hasValue());
    if (farTails) {
      VSS_CHECK(farTails.value().survival == 0.0);
      VSS_CHECK(std::isfinite(farTails.value().log_tails.log_survival));
      VSS_CHECK(farTails.value().log_tails.log_survival < -1.0e6);
    }
    const auto representableLogTail =
        evaluate(extremeBetaGamma.value(), 1.0e300, policy());
    VSS_CHECK(representableLogTail.hasValue());
    if (representableLogTail) {
      VSS_CHECK(representableLogTail.value().survival == 0.0);
      VSS_CHECK(
          std::isfinite(representableLogTail.value().log_tails.log_survival));
      VSS_CHECK(representableLogTail.value().log_tails.log_survival < -1.0e299);
    }
  }

  const std::array<Distribution, 2> poissonTerms{poisson(1.25).value(),
                                                 poisson(2.75).value()};
  const auto poissonSum = finiteConvolution(poissonTerms, policy());
  VSS_CHECK(poissonSum.hasValue());
  if (poissonSum) {
    VSS_CHECK(poissonSum.value().family() == Family::poisson);
    const auto summedMass = evaluate(poissonSum.value(), 3.0, policy());
    const auto referenceMass = evaluate(poisson(4.0).value(), 3.0, policy());
    VSS_CHECK(summedMass.hasValue() && referenceMass.hasValue());
    if (summedMass && referenceMass) {
      VSS_CHECK(std::abs(summedMass.value().density_or_mass -
                         referenceMass.value().density_or_mass) < 1.0e-14);
    }
  }

  const std::array<Distribution, 2> binomialTerms{binomial(2U, 0.25).value(),
                                                  binomial(3U, 0.25).value()};
  const auto binomialSum = finiteConvolution(binomialTerms, policy());
  VSS_CHECK(binomialSum.hasValue());
  if (binomialSum) {
    VSS_CHECK(binomialSum.value().family() == Family::binomial);
    VSS_CHECK(std::abs(evaluate(binomialSum.value(), 2.0, policy())
                           .value()
                           .density_or_mass -
                       evaluate(binomial(5U, 0.25).value(), 2.0, policy())
                           .value()
                           .density_or_mass) < 1.0e-14);
  }

  const std::array<Distribution, 2> negativeBinomialTerms{
      negativeBinomial(1.25, 0.4).value(), negativeBinomial(2.75, 0.4).value()};
  const auto negativeBinomialSum =
      finiteConvolution(negativeBinomialTerms, policy());
  VSS_CHECK(negativeBinomialSum.hasValue());
  if (negativeBinomialSum) {
    VSS_CHECK(negativeBinomialSum.value().family() ==
              Family::negative_binomial);
    VSS_CHECK(
        std::abs(evaluate(negativeBinomialSum.value(), 3.0, policy())
                     .value()
                     .density_or_mass -
                 evaluate(negativeBinomial(4.0, 0.4).value(), 3.0, policy())
                     .value()
                     .density_or_mass) < 1.0e-14);
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
  compositeInputLimitsPreflight();
  convolutionRegimes();
  familyQuantileMomentAndReplayGrid();
}

// VSS-MATH-TRACEABILITY: C059 CASE-C059.normal.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.normal.factory.invalid
// VSS-MATH-TRACEABILITY: C059 CASE-C059.bernoulli.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.bernoulli.factory.invalid
// VSS-MATH-TRACEABILITY: C059 CASE-C059.chi-square.factory
// VSS-MATH-TRACEABILITY: C059 CASE-C059.chi-square.factory.invalid
