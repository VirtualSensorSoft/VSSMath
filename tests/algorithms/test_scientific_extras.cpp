#include "Test.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/ScientificExtras.hpp>

namespace {
using namespace vss::math;
Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void matrixExponentialIdentities() {
  const SolvePolicy solve{.iteration = {.abs = 1.0e-13,
                                        .rel = 1.0e-13,
                                        .max_iterations = 2000U,
                                        .max_evaluations = 0U,
                                        .bracket = std::nullopt},
                          .rank_abs = 0.0,
                          .rank_rel = 1.0e-14,
                          .reciprocal_condition_floor = 1.0e-15,
                          .rank = RankPolicy::reject,
                          .restart = 0U};
  const auto empty = matrixExponential(Matrix{}, solve, {0U, 0U, 0U});
  VSS_CHECK(empty && empty.value().rows() == 0U &&
            empty.value().columns() == 0U);
  Matrix diagonal(2U, 2U, 0.0);
  diagonal.at(1U, 1U) = std::log(2.0);
  const auto result =
      matrixExponential(std::move(diagonal), solve, {1000U, 100000U, 0U});
  VSS_CHECK(result);
  VSS_CHECK(std::abs(result.value().at(0U, 0U) - 1.0) < 1.0e-14);
  VSS_CHECK(std::abs(result.value().at(1U, 1U) - 2.0) < 1.0e-13);

  Matrix nilpotent(2U, 2U, 0.0);
  nilpotent.at(0U, 1U) = 100.0;
  const auto shear =
      matrixExponential(std::move(nilpotent), solve, {1000U, 100000U, 0U});
  VSS_CHECK(shear && shear.value().at(0U, 1U) == 100.0);

  Matrix rotation(2U, 2U, 0.0);
  rotation.at(0U, 1U) = -std::numbers::pi / 3.0;
  rotation.at(1U, 0U) = std::numbers::pi / 3.0;
  const auto rotated =
      matrixExponential(std::move(rotation), solve, {1000U, 100000U, 0U});
  VSS_CHECK(rotated);
  VSS_CHECK(std::abs(rotated.value().at(0U, 0U) - 0.5) < 2e-14);
  VSS_CHECK(std::abs(rotated.value().at(0U, 1U) + std::sqrt(3.0) / 2.0) <
            2e-14);

  Matrix jordan(2U, 2U, 0.0);
  jordan.at(0U, 0U) = 1.0;
  jordan.at(1U, 1U) = 1.0;
  jordan.at(0U, 1U) = 1.0;
  const auto jordanResult =
      matrixExponential(std::move(jordan), solve, {1000U, 100000U, 0U});
  VSS_CHECK(jordanResult &&
            std::abs(jordanResult.value().at(0U, 1U) - std::exp(1.0)) < 2e-14);

  const Scalar delta = std::ldexp(1.0, -40);
  Matrix nearJordan(2U, 2U, 0.0);
  nearJordan.at(0U, 0U) = 1.0;
  nearJordan.at(0U, 1U) = 1.0;
  nearJordan.at(1U, 1U) = 1.0 + delta;
  const auto nearResult =
      matrixExponential(std::move(nearJordan), solve, {1000U, 100000U, 0U});
  VSS_CHECK(nearResult);
  const Scalar exactOffDiagonal = std::exp(1.0) * std::expm1(delta) / delta;
  VSS_CHECK(std::abs(nearResult.value().at(0U, 1U) - exactOffDiagonal) < 1e-12);

  Matrix overflow(1U, 1U);
  overflow.at(0U, 0U) = 1000.0;
  const auto overflowResult =
      matrixExponential(std::move(overflow), solve, {100U, 100000U, 0U});
  VSS_CHECK(!overflowResult &&
            overflowResult.status().code() == ErrorCode::NonFiniteResult);
  Matrix underflow(1U, 1U);
  underflow.at(0U, 0U) = -1000.0;
  const auto underflowResult =
      matrixExponential(std::move(underflow), solve, {100U, 100000U, 0U});
  VSS_CHECK(underflowResult && underflowResult.value().at(0U, 0U) == 0.0);
  SolvePolicy shortSolve = solve;
  shortSolve.iteration.max_iterations = 1U;
  const auto cap = matrixExponential(Matrix(1U, 1U, 1000.0), shortSolve,
                                     {100U, 100000U, 1U});
  VSS_CHECK(!cap && cap.status().code() == ErrorCode::ResourceLimit);
  const auto nonsquare =
      matrixExponential(Matrix(1U, 2U), solve, {100U, 100000U, 0U});
  VSS_CHECK(!nonsquare &&
            nonsquare.status().code() == ErrorCode::DimensionMismatch);
}

void pValueAdjustment() {
  const Vector p = vector({0.01, 0.04, 0.03});
  const Limits cap{20U, 1024U, 0U};
  const auto bonferroni =
      adjustPValues(p.view(), MultiplicityRule::bonferroni, cap);
  const auto holm = adjustPValues(p.view(), MultiplicityRule::holm, cap);
  const auto bh =
      adjustPValues(p.view(), MultiplicityRule::benjamini_hochberg, cap);
  VSS_CHECK(bonferroni && holm && bh);
  VSS_CHECK(std::abs(bonferroni.value().at(1U) - 0.12) < 1e-15);
  VSS_CHECK(std::abs(holm.value().at(1U) - 0.06) < 1e-15);
  VSS_CHECK(std::abs(bh.value().at(1U) - 0.04) < 1e-15);
  const Vector empty;
  const auto emptyResult =
      adjustPValues(empty.view(), MultiplicityRule::holm, cap);
  VSS_CHECK(emptyResult && emptyResult.value().empty());
  const Vector invalid = vector({0.2, -0.1});
  const auto bad = adjustPValues(invalid.view(), MultiplicityRule::holm, cap);
  VSS_CHECK(!bad && bad.status().code() == ErrorCode::InvalidDomain);
  const auto denied =
      adjustPValues(p.view(), MultiplicityRule::holm, {2U, 1024U, 0U});
  VSS_CHECK(!denied && denied.status().code() == ErrorCode::ResourceLimit);
}

ProbabilityPolicy probabilityPolicy() {
  return {.iteration = {.abs = 2.0e-10,
                        .rel = 2.0e-10,
                        .max_iterations = 1000U,
                        .max_evaluations = 100'000U,
                        .bracket = std::nullopt},
          .max_draws = 0U,
          .abs_log_error = 2.0e-9};
}

ProbabilityPolicy deepProbabilityPolicy() {
  ProbabilityPolicy value = probabilityPolicy();
  value.iteration.abs = 2.0e-9;
  value.iteration.max_iterations = 1'000'000U;
  value.iteration.max_evaluations = 100'000'000U;
  return value;
}

void probabilityLogIntegrals() {
  const auto bernoulli = probability::bernoulli(0.5);
  const auto uniform = probability::uniform(0.0, 2.0);
  const probability::IntervalSupport allReal{
      -std::numeric_limits<Scalar>::infinity(),
      std::numeric_limits<Scalar>::infinity(), true, true};
  VSS_CHECK(bernoulli && uniform);
  const auto hBernoulli =
      numericalEntropy(bernoulli.value(), allReal, probabilityPolicy());
  const auto hUniform = numericalEntropy(
      uniform.value(), {0.0, 2.0, true, true}, probabilityPolicy());
  VSS_CHECK(hBernoulli && std::abs(hBernoulli.value() - std::log(2.0)) < 1e-14);
  VSS_CHECK(hUniform && std::abs(hUniform.value() - std::log(2.0)) < 1e-14);
  const auto n0 = probability::normal(0.0, 1.0);
  const auto n1 = probability::normal(1.0, 1.0);
  VSS_CHECK(n0 && n1);
  const auto klNormal = numericalKLDivergence(n1.value(), n0.value(), allReal,
                                              probabilityPolicy());
  VSS_CHECK(klNormal && std::abs(klNormal.value() - 0.5) < 1e-14);
  auto movedFrom = probability::normal(0.0, 1.0);
  VSS_CHECK(movedFrom);
  probability::Distribution movedLaw(std::move(movedFrom.value()));
  const auto movedFromKl = numericalKLDivergence(movedFrom.value(), movedLaw,
                                                 allReal, probabilityPolicy());
  VSS_CHECK(!movedFromKl &&
            movedFromKl.status().code() == ErrorCode::InvalidDomain);
  const auto b25 = probability::bernoulli(0.25);
  VSS_CHECK(b25);
  const auto klBernoulli =
      numericalKLDivergence(b25.value(), bernoulli.value(),
                            {0.0, 1.0, true, true}, probabilityPolicy());
  const Scalar expected = 0.25 * std::log(0.5) + 0.75 * std::log(1.5);
  VSS_CHECK(klBernoulli && std::abs(klBernoulli.value() - expected) < 1e-14);
  const auto uniform01 = probability::uniform(0.0, 1.0);
  const auto beta22 = probability::beta(2.0, 2.0);
  VSS_CHECK(uniform01 && beta22);
  const auto boundaryZeroKl =
      numericalKLDivergence(uniform01.value(), beta22.value(),
                            {0.0, 1.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(boundaryZeroKl &&
            std::abs(boundaryZeroKl.value() - (2.0 - std::log(6.0))) < 2.0e-9);
  const auto leftUniform = probability::uniform(0.0, 0.4);
  const auto rightUniform = probability::uniform(0.6, 1.0);
  VSS_CHECK(leftUniform && rightUniform);
  const std::array<probability::WeightedLaw, 2> separatedTerms{
      probability::WeightedLaw{leftUniform.value(), 0.5},
      probability::WeightedLaw{rightUniform.value(), 0.5}};
  const auto separated =
      probability::mixture(separatedTerms, deepProbabilityPolicy());
  VSS_CHECK(separated);
  const auto gapKl =
      numericalKLDivergence(uniform01.value(), separated.value(),
                            {0.0, 1.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(gapKl && std::isinf(gapKl.value()) && gapKl.value() > 0.0);
  const auto omittedKl =
      numericalKLDivergence(uniform01.value(), uniform.value(),
                            {0.2, 1.0, true, true}, probabilityPolicy());
  VSS_CHECK(!omittedKl &&
            omittedKl.status().code() == ErrorCode::InvalidDomain);
  const auto mixedMeasureKl =
      numericalKLDivergence(bernoulli.value(), uniform01.value(),
                            {0.0, 1.0, true, true}, probabilityPolicy());
  VSS_CHECK(!mixedMeasureKl &&
            mixedMeasureKl.status().code() == ErrorCode::InvalidDomain);
  const auto missing = probability::bernoulli(0.0);
  VSS_CHECK(missing);
  const auto infinite =
      numericalKLDivergence(bernoulli.value(), missing.value(),
                            {0.0, 1.0, true, true}, probabilityPolicy());
  VSS_CHECK(infinite && std::isinf(infinite.value()) && infinite.value() > 0.0);
  const auto omitted = numericalEntropy(uniform.value(), {0.0, 1.0, true, true},
                                        probabilityPolicy());
  VSS_CHECK(!omitted && omitted.status().code() == ErrorCode::InvalidDomain);
  const auto gamma = probability::gamma(0.5, 1.0);
  VSS_CHECK(gamma);
  const auto gammaEntropy = numericalEntropy(
      gamma.value(),
      {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
      probabilityPolicy());
  VSS_CHECK(gammaEntropy && std::isfinite(gammaEntropy.value()));
  const Scalar gammaHalfReference =
      0.5 + 0.5 * std::log(std::numbers::pi) +
      0.5 * (-0.5772156649015328606 - 2.0 * std::log(2.0));
  VSS_CHECK(std::abs(gammaEntropy.value() - gammaHalfReference) < 2e-8);
  const auto gammaQuarter = probability::gamma(0.25, 1.0);
  VSS_CHECK(gammaQuarter);
  const auto gammaQuarterEntropy = numericalEntropy(
      gammaQuarter.value(),
      {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
      probabilityPolicy());
  VSS_CHECK(gammaQuarterEntropy && std::abs(gammaQuarterEntropy.value() +
                                            1.6325676253341215) < 2.0e-9);
  const auto beta = probability::beta(0.25, 0.25);
  VSS_CHECK(beta);
  const auto betaEntropy = numericalEntropy(
      beta.value(), {0.0, 1.0, true, true}, probabilityPolicy());
  VSS_CHECK(betaEntropy &&
            std::abs(betaEntropy.value() + 1.392235154560808) < 2.0e-9);
  for (const auto [shapeA, shapeB, reference] :
       {std::tuple{0.25, 0.25, 0.0906099299139882}, std::tuple{0.5, 0.5, 1.0},
        std::tuple{0.75, 0.75, 1.3609727753754666},
        std::tuple{0.25, 2.0, 1.6591886316127282}}) {
    const auto first = probability::gamma(shapeA, 1.0);
    const auto second = probability::gamma(shapeB, 1.0);
    VSS_CHECK(first && second);
    const std::array<probability::Distribution, 2> terms{first.value(),
                                                         second.value()};
    const auto sum = probability::finiteConvolution(terms, probabilityPolicy());
    VSS_CHECK(sum);
    const auto entropy = numericalEntropy(
        sum.value(),
        {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
        probabilityPolicy());
    VSS_CHECK(entropy && std::abs(entropy.value() - reference) < 2.0e-9);
  }
  for (const auto [degrees, reference] : {std::pair{3.0, 0.645460961341382},
                                          std::pair{5.0, 0.1247691941236102}}) {
    const auto student = probability::studentT(degrees, 0.0, 1.0);
    VSS_CHECK(student);
    const auto studentNormal = numericalKLDivergence(
        student.value(), n0.value(), allReal, probabilityPolicy());
    VSS_CHECK(studentNormal && std::isfinite(studentNormal.value()));
    VSS_CHECK(std::abs(studentNormal.value() - reference) <
              2.0e-9 + 2.0e-9 * std::abs(reference));
  }
  for (const Scalar degrees : {0.5, 1.0, 2.0, 2.0 - std::ldexp(1.0, -20)}) {
    const auto infiniteStudent = probability::studentT(degrees, 0.0, 1.0);
    VSS_CHECK(infiniteStudent);
    const auto divergent = numericalKLDivergence(
        infiniteStudent.value(), n0.value(), allReal, probabilityPolicy());
    VSS_CHECK(divergent && std::isinf(divergent.value()) &&
              divergent.value() > 0.0);
  }
  const Scalar aboveTwo = 2.0 + std::ldexp(1.0, -20);
  const auto barelyFiniteStudent = probability::studentT(aboveTwo, 0.0, 1.0);
  VSS_CHECK(barelyFiniteStudent);
  const auto barelyFiniteKl =
      numericalKLDivergence(barelyFiniteStudent.value(), n0.value(), allReal,
                            deepProbabilityPolicy());
  VSS_CHECK(barelyFiniteKl && std::isfinite(barelyFiniteKl.value()));
  VSS_CHECK(std::abs(barelyFiniteKl.value() - 1048575.4586595736) < 0.01);
  for (const auto [degrees, reference] :
       {std::pair{0.5, 0.49917928622267915},
        std::pair{1.0, 0.25924453248886237},
        std::pair{2.0, 0.11722851713864935},
        std::pair{3.0, 0.06915159848517671}}) {
    const auto studentTail = probability::studentT(degrees, 0.0, 1.0);
    VSS_CHECK(studentTail);
    const auto reverse = numericalKLDivergence(
        n0.value(), studentTail.value(), allReal, deepProbabilityPolicy());
    VSS_CHECK(reverse && std::isfinite(reverse.value()));
    VSS_CHECK(std::abs(reverse.value() - reference) <
              2.0e-9 + 2.0e-9 * std::abs(reference));
  }
  const auto normalOne = probability::normal(0.0, 1.0);
  const auto normalTwo = probability::normal(1.0, 2.0);
  VSS_CHECK(normalOne && normalTwo);
  const std::array<probability::Distribution, 2> normalTerms{normalOne.value(),
                                                             normalTwo.value()};
  const auto normalSum =
      probability::finiteConvolution(normalTerms, probabilityPolicy());
  VSS_CHECK(normalSum);
  const auto normalSumEntropy =
      numericalEntropy(normalSum.value(), allReal, probabilityPolicy());
  const Scalar normalSumReference =
      0.5 * std::log(2.0 * std::numbers::pi * std::numbers::e * 5.0);
  VSS_CHECK(normalSumEntropy &&
            std::abs(normalSumEntropy.value() - normalSumReference) < 2e-14);
  const std::array<probability::WeightedLaw, 2> normalMixture{
      probability::WeightedLaw{normalOne.value(), 0.5},
      probability::WeightedLaw{normalTwo.value(), 0.5}};
  const auto mixture = probability::mixture(normalMixture, probabilityPolicy());
  VSS_CHECK(mixture);
  const auto mixtureEntropy =
      numericalEntropy(mixture.value(), allReal, probabilityPolicy());
  VSS_CHECK(mixtureEntropy && std::isfinite(mixtureEntropy.value()));
  const auto coin = probability::bernoulli(0.25);
  const auto count = probability::poisson(0.125);
  VSS_CHECK(coin && count);
  const std::array<probability::Distribution, 2> coinNormalTerms{
      coin.value(), normalOne.value()};
  const auto coinNormal =
      probability::finiteConvolution(coinNormalTerms, probabilityPolicy());
  VSS_CHECK(coinNormal);
  const auto coinNormalEntropy =
      numericalEntropy(coinNormal.value(), allReal, probabilityPolicy());
  VSS_CHECK(coinNormalEntropy && std::isfinite(coinNormalEntropy.value()));
  const std::array<probability::Distribution, 2> poissonNormalTerms{
      count.value(), normalOne.value()};
  const auto poissonNormal =
      probability::finiteConvolution(poissonNormalTerms, probabilityPolicy());
  VSS_CHECK(poissonNormal);
  const auto poissonNormalEntropy =
      numericalEntropy(poissonNormal.value(), allReal, deepProbabilityPolicy());
  VSS_CHECK(poissonNormalEntropy &&
            std::isfinite(poissonNormalEntropy.value()));
  VSS_CHECK(std::abs(poissonNormalEntropy.value() - 1.476841221530215) <
            4.0e-9);
  for (const Scalar rate : {1.0, 8.0}) {
    const auto countLaw = probability::poisson(rate);
    VSS_CHECK(countLaw);
    const std::array<probability::Distribution, 2> terms{countLaw.value(),
                                                         normalOne.value()};
    const auto sum =
        probability::finiteConvolution(terms, deepProbabilityPolicy());
    VSS_CHECK(sum);
    const auto entropy =
        numericalEntropy(sum.value(), allReal, deepProbabilityPolicy());
    const Scalar reference =
        rate == 1.0 ? 1.755392643469712 : 2.509974814992024;
    VSS_CHECK(entropy && std::abs(entropy.value() - reference) < 4.0e-9);
  }
  const auto gammaHalf = probability::gamma(0.5, 1.0);
  const auto poissonOne = probability::poisson(1.0);
  VSS_CHECK(gammaHalf && poissonOne);
  const std::array<probability::Distribution, 2> poissonGammaTerms{
      poissonOne.value(), gammaHalf.value()};
  const auto poissonGamma = probability::finiteConvolution(
      poissonGammaTerms, deepProbabilityPolicy());
  VSS_CHECK(poissonGamma);
  const auto poissonGammaEntropy = numericalEntropy(
      poissonGamma.value(),
      {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
      deepProbabilityPolicy());
  VSS_CHECK(poissonGammaEntropy && std::isfinite(poissonGammaEntropy.value()) &&
            std::abs(poissonGammaEntropy.value() - 0.987297799874736) < 4.0e-9);

  const std::array<std::array<Scalar, 3>, 6> betaGammaCases{
      {{0.25, 0.25, 0.311252338158},
       {0.5, 0.5, 0.887476400712},
       {0.75, 0.75, 1.094002759925},
       {0.25, 0.75, 1.054938086996},
       {0.25, 0.5, 0.796208979422},
       {0.5, 0.75, 1.098939348322}}};
  for (const auto& [betaShape, gammaShape, reference] : betaGammaCases) {
    const auto betaInterior = probability::beta(1.0, betaShape);
    const auto gammaInterior = probability::gamma(gammaShape, 1.0);
    VSS_CHECK(betaInterior && gammaInterior);
    const std::array<probability::Distribution, 2> betaGammaTerms{
        betaInterior.value(), gammaInterior.value()};
    const auto betaGamma =
        probability::finiteConvolution(betaGammaTerms, deepProbabilityPolicy());
    VSS_CHECK(betaGamma);
    const auto betaGammaEntropy = numericalEntropy(
        betaGamma.value(),
        {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
        deepProbabilityPolicy());
    if (!betaGammaEntropy) {
      std::cerr << "Beta-Gamma(" << betaShape << ',' << gammaShape
                << "): " << betaGammaEntropy.status().message() << '\n';
    }
    VSS_CHECK(betaGammaEntropy && std::isfinite(betaGammaEntropy.value()));
    VSS_CHECK(std::abs(betaGammaEntropy.value() - reference) <
              2.0e-9 + 2.0e-9 * std::abs(reference));
    if (betaShape == 0.25 && gammaShape == 0.25) {
      const auto collision = probability::evaluate(betaGamma.value(), 1.0,
                                                   deepProbabilityPolicy());
      VSS_CHECK(collision && std::isinf(collision.value().density_or_mass));
      ProbabilityPolicy shortBudget = deepProbabilityPolicy();
      shortBudget.iteration.max_evaluations = 31U;
      const auto denied = numericalEntropy(
          betaGamma.value(),
          {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
          shortBudget);
      VSS_CHECK(!denied && denied.status().code() == ErrorCode::ResourceLimit);
    }
    if (betaShape == 0.5 && gammaShape == 0.5) {
      const auto collision = probability::evaluate(betaGamma.value(), 1.0,
                                                   deepProbabilityPolicy());
      VSS_CHECK(collision && std::isinf(collision.value().density_or_mass));
    }
    if (betaShape == 0.75 && gammaShape == 0.75) {
      const auto collision = probability::evaluate(betaGamma.value(), 1.0,
                                                   deepProbabilityPolicy());
      const Scalar referenceDensity = 0.75 * std::sqrt(std::numbers::pi) *
                                      std::erf(1.0) / std::tgamma(0.75);
      VSS_CHECK(collision && std::abs(collision.value().density_or_mass -
                                      referenceDensity) < 2.0e-8);
    }
  }
  for (const Scalar gammaShape :
       {0.5 - std::ldexp(1.0, -12), 0.5 + std::ldexp(1.0, -12)}) {
    const auto betaThreshold = probability::beta(1.0, 0.5);
    const auto gammaThreshold = probability::gamma(gammaShape, 1.0);
    VSS_CHECK(betaThreshold && gammaThreshold);
    const std::array<probability::Distribution, 2> terms{
        betaThreshold.value(), gammaThreshold.value()};
    const auto law =
        probability::finiteConvolution(terms, deepProbabilityPolicy());
    VSS_CHECK(law);
    const auto entropy = numericalEntropy(
        law.value(),
        {0.0, std::numeric_limits<Scalar>::infinity(), true, false},
        deepProbabilityPolicy());
    VSS_CHECK(entropy && std::isfinite(entropy.value()));
    const Scalar reference =
        gammaShape < 0.5 ? 0.8872292742524058 : 0.8877234256850304;
    VSS_CHECK(std::abs(entropy.value() - reference) <
              2.0e-9 + 2.0e-9 * std::abs(reference));
  }

  const auto normalForNested = probability::normal(0.0, 1.0);
  const auto uniformForNested = probability::uniform(-1.0, 1.0);
  VSS_CHECK(normalForNested && uniformForNested);
  const std::array<probability::Distribution, 2> innerTerms{
      normalForNested.value(), uniformForNested.value()};
  const auto innerSum =
      probability::finiteConvolution(innerTerms, deepProbabilityPolicy());
  VSS_CHECK(innerSum);
  const auto innerTruncated = probability::truncatedDistribution(
      innerSum.value(), {-1.0, 2.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(innerTruncated);
  VSS_CHECK(std::abs(std::exp(innerTruncated.value().parameters()[2U]) -
                     0.7633075529723792) < 2.0e-9);
  const auto gammaOuter = probability::gamma(0.75, 1.0);
  VSS_CHECK(gammaOuter);
  const std::array<probability::Distribution, 2> gammaTerms{
      innerTruncated.value(), gammaOuter.value()};
  const auto gammaSum =
      probability::finiteConvolution(gammaTerms, deepProbabilityPolicy());
  VSS_CHECK(gammaSum);
  const auto nestedGamma = probability::truncatedDistribution(
      gammaSum.value(), {0.0, 3.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(nestedGamma);
  VSS_CHECK(std::abs(std::exp(nestedGamma.value().parameters()[2U]) -
                     0.7576918664596104) < 2.0e-9);
  const auto nestedGammaEntropy = numericalEntropy(
      nestedGamma.value(), {0.0, 3.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(nestedGammaEntropy &&
            std::abs(nestedGammaEntropy.value() - 1.000067421724897) < 4.0e-9);
  ProbabilityPolicy nestedEvaluationCap = deepProbabilityPolicy();
  nestedEvaluationCap.iteration.max_evaluations = 31U;
  const auto deniedNested = numericalEntropy(
      nestedGamma.value(), {0.0, 3.0, true, true}, nestedEvaluationCap);
  VSS_CHECK(!deniedNested &&
            deniedNested.status().code() == ErrorCode::ResourceLimit);
  const auto nestedPoissonOuter = probability::poisson(1.0);
  VSS_CHECK(nestedPoissonOuter);
  const std::array<probability::Distribution, 2> poissonTerms{
      innerTruncated.value(), nestedPoissonOuter.value()};
  const auto poissonSum =
      probability::finiteConvolution(poissonTerms, deepProbabilityPolicy());
  VSS_CHECK(poissonSum);
  const auto nestedPoisson = probability::truncatedDistribution(
      poissonSum.value(), {0.0, 3.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(nestedPoisson);
  VSS_CHECK(std::abs(std::exp(nestedPoisson.value().parameters()[2U]) -
                     0.7602400948478492) < 2.0e-9);
  const auto poissonEntropy = numericalEntropy(
      nestedPoisson.value(), {0.0, 3.0, true, true}, deepProbabilityPolicy());
  VSS_CHECK(poissonEntropy &&
            std::abs(poissonEntropy.value() - 1.065923226102197) < 4.0e-9);
}
} // namespace

int main() try {
  matrixExponentialIdentities();
  pValueAdjustment();
  probabilityLogIntegrals();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
