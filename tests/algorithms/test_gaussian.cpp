#include "Test.hpp"

#include <array>
#include <cmath>
#include <vss/math/probability/DistributionMoments.hpp>
#include <vss/math/probability/Gaussian.hpp>
#include <vss/math/probability/Likelihood.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

SolvePolicy solvePolicy(RankPolicy rank = RankPolicy::supported_subspace) {
  return {.iteration = {.abs = 1.0e-12,
                        .rel = 1.0e-12,
                        .max_iterations = 100U,
                        .max_evaluations = 1000U,
                        .bracket = std::nullopt},
          .rank_abs = 1.0e-14,
          .rank_rel = 1.0e-12,
          .reciprocal_condition_floor = 0.0,
          .rank = rank,
          .restart = 0U};
}

ProbabilityPolicy probabilityPolicy() {
  return {{1.0e-13, 1.0e-13, 1000U, 10000U, std::nullopt}, 10000U, 2.0e-9};
}

void likelihoodAtGaussianCenter() {
  const std::array<Scalar, 2> locationValues{0.0, 0.0};
  const auto location = ConstVectorView::create(locationValues, 2U);
  Matrix covarianceMatrix(2U, 2U, 0.0);
  covarianceMatrix.at(0U, 0U) = 1.0;
  covarianceMatrix.at(1U, 1U) = 4.0;
  const auto law = multivariateNormal(location.value(), covarianceMatrix.view(),
                                      solvePolicy(), {128U, 4096U, 10000U});
  VSS_CHECK(law.hasValue());
  if (law) {
    const auto result =
        logLikelihood(law.value(), location.value(), probabilityPolicy());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(std::abs(result.value() +
                         std::log(4.0 * 3.14159265358979323846)) < 1.0e-12);
    }
    auto invalidPolicy = probabilityPolicy();
    invalidPolicy.abs_log_error = 0.0;
    VSS_CHECK(logLikelihood(law.value(), location.value(), invalidPolicy)
                  .status()
                  .code() == ErrorCode::InvalidTolerance);
  }
}

void gaussianConditioningAndReplay() {
  const std::array<Scalar, 2> locationValues{0.0, 0.0};
  const auto location = ConstVectorView::create(locationValues, 2U);
  Matrix scale(2U, 2U, 0.0);
  scale.at(0U, 0U) = 4.0;
  scale.at(0U, 1U) = 2.0;
  scale.at(1U, 0U) = 2.0;
  scale.at(1U, 1U) = 4.0;
  auto law = multivariateNormal(location.value(), scale.view(), solvePolicy(),
                                {128U, 4096U, 10000U});
  VSS_CHECK(law.hasValue());
  if (law) {
    const std::array<Index, 1> keep{0U};
    const std::array<Index, 1> observed{1U};
    const std::array<Scalar, 1> observedValues{2.0};
    const auto values = ConstVectorView::create(observedValues, 1U);
    auto conditional =
        conditionalGaussian(law.value(), keep, observed, values.value(),
                            solvePolicy(), {128U, 4096U, 10000U});
    VSS_CHECK(conditional.hasValue());
    if (conditional) {
      VSS_CHECK(std::abs(conditional.value().location().at(0U) - 1.0) <
                1.0e-12);
      VSS_CHECK(std::abs(conditional.value().scale().at(0U, 0U) - 3.0) <
                1.0e-12);
    }
    Rng first(9876U);
    Rng replay(9876U);
    auto a = multivariateNormalSample(law.value(), first, probabilityPolicy(),
                                      {32U, 1024U, 1000U});
    auto b = multivariateNormalSample(law.value(), replay, probabilityPolicy(),
                                      {32U, 1024U, 1000U});
    VSS_CHECK(a.hasValue() && b.hasValue());
    if (a && b) {
      VSS_CHECK(a.value().size() == 2U && b.value().size() == 2U);
      VSS_CHECK(a.value().at(0U) == b.value().at(0U));
      VSS_CHECK(a.value().at(1U) == b.value().at(1U));
    }
  }
}

void singularSupportIsPreserved() {
  const std::array<Scalar, 2> center{0.0, 0.0};
  const auto location = ConstVectorView::create(center, 2U);
  Matrix scale(2U, 2U, 1.0);
  auto law = multivariateNormal(location.value(), scale.view(), solvePolicy(),
                                {128U, 4096U, 10000U});
  VSS_CHECK(law.hasValue());
  if (law) {
    const std::array<Scalar, 2> onSupport{1.0, 1.0};
    const std::array<Scalar, 2> offSupport{1.0, 2.0};
    const auto onView = ConstVectorView::create(onSupport, 2U);
    const auto offView = ConstVectorView::create(offSupport, 2U);
    VSS_CHECK(std::isfinite(
        logLikelihood(law.value(), onView.value(), probabilityPolicy())
            .value()));
    VSS_CHECK(logLikelihood(law.value(), offView.value(), probabilityPolicy())
                  .value() == -std::numeric_limits<Scalar>::infinity());
  }
}

void normalSamplingUsesOneWordPerRetainedCoordinate() {
  const std::array<Scalar, 1> center{3.0};
  const auto location = ConstVectorView::create(center, 1U);
  Matrix scale(1U, 1U, 4.0);
  const auto law = multivariateNormal(location.value(), scale.view(),
                                      solvePolicy(), {32U, 1024U, 1000U});
  VSS_CHECK(law.hasValue());
  if (!law) {
    return;
  }
  Rng actual(444U);
  Rng expected(444U);
  const auto draw = multivariateNormalSample(
      law.value(), actual, probabilityPolicy(), {32U, 1024U, 1000U});
  VSS_CHECK(draw.hasValue());
  (void)expected.next();
  VSS_CHECK(actual.next() == expected.next());
}

void jointMomentsAndRankPolicy() {
  const std::array<Scalar, 2> center{1.0, -2.0};
  const auto location = ConstVectorView::create(center, 2U);
  Matrix scale(2U, 2U, 0.0);
  scale.at(0U, 0U) = 2.0;
  scale.at(0U, 1U) = 0.5;
  scale.at(1U, 0U) = 0.5;
  scale.at(1U, 1U) = 4.0;
  const auto student = multivariateStudentT(location.value(), scale.view(), 3.0,
                                            solvePolicy(), {64U, 2048U, 1000U});
  VSS_CHECK(student.hasValue());
  if (student) {
    const auto studentMean =
        mean(student.value(), probabilityPolicy(), {16U, 512U, 1000U});
    const auto studentCovariance =
        covariance(student.value(), probabilityPolicy(), {32U, 1024U, 1000U});
    VSS_CHECK(studentMean.hasValue() && studentCovariance.hasValue());
    if (studentMean && studentCovariance) {
      VSS_CHECK(studentMean.value().at(0U) == 1.0);
      VSS_CHECK(studentMean.value().at(1U) == -2.0);
      VSS_CHECK(std::abs(studentCovariance.value().at(0U, 0U) - 6.0) < 1.0e-12);
      VSS_CHECK(std::abs(studentCovariance.value().at(0U, 1U) - 1.5) < 1.0e-12);
      VSS_CHECK(std::abs(studentCovariance.value().at(1U, 1U) - 12.0) <
                1.0e-12);
    }
  }

  const std::array<Scalar, 3> singularCenter{0.0, 0.0, 0.0};
  const auto singularLocation = ConstVectorView::create(singularCenter, 3U);
  Matrix singular(3U, 3U, 0.0);
  singular.at(0U, 0U) = 1.0;
  singular.at(0U, 1U) = 1.0;
  singular.at(1U, 0U) = 1.0;
  singular.at(1U, 1U) = 1.0;
  singular.at(2U, 2U) = 1.0;
  const auto rankOne =
      multivariateNormal(singularLocation.value(), singular.view(),
                         solvePolicy(), {64U, 2048U, 1000U});
  VSS_CHECK(rankOne.hasValue());
  if (rankOne) {
    const std::array<Index, 1> keep{2U};
    const std::array<Index, 2> observed{0U, 1U};
    const std::array<Scalar, 2> observedValue{0.0, 0.0};
    const auto values = ConstVectorView::create(observedValue, 2U);
    const auto strict = conditionalGaussian(
        rankOne.value(), keep, observed, values.value(),
        solvePolicy(RankPolicy::reject), {64U, 2048U, 1000U});
    VSS_CHECK(strict.status().code() == ErrorCode::RankDeficient);
  }
}

void singularConditioningAndEmptyMarginal() {
  const std::array<Scalar, 2> center{0.0, 0.0};
  const auto location = ConstVectorView::create(center, 2U);
  Matrix scale(2U, 2U, 1.0);
  const auto law = multivariateNormal(location.value(), scale.view(),
                                      solvePolicy(), {64U, 2048U, 1000U});
  VSS_CHECK(law.hasValue());
  if (!law) {
    return;
  }
  const std::array<Index, 1> keepFirst{0U};
  const std::array<Index, 1> observeSecond{1U};
  const std::array<Scalar, 1> observedValue{1.0};
  const auto valueView = ConstVectorView::create(observedValue, 1U);
  const auto conditional = conditionalGaussian(
      law.value(), keepFirst, observeSecond, valueView.value(), solvePolicy(),
      {64U, 2048U, 1000U});
  VSS_CHECK(conditional.hasValue());
  if (conditional) {
    VSS_CHECK(std::abs(conditional.value().location().at(0U) - 1.0) < 1.0e-12);
    VSS_CHECK(std::abs(conditional.value().scale().at(0U, 0U)) < 1.0e-12);
  }

  const std::array<Index, 2> observeBoth{0U, 1U};
  const std::array<Scalar, 2> incompatible{0.0, 1.0};
  const auto incompatibleView = ConstVectorView::create(incompatible, 2U);
  const std::array<Index, 0> keepNone{};
  const auto rejected = conditionalGaussian(law.value(), keepNone, observeBoth,
                                            incompatibleView.value(),
                                            solvePolicy(), {64U, 2048U, 1000U});
  VSS_CHECK(rejected.status().code() == ErrorCode::InvalidDomain);

  const std::array<Index, 0> observeNone{};
  const std::array<Scalar, 0> noValues{};
  const auto emptyView = ConstVectorView::create(noValues, 0U);
  const auto marginal = conditionalGaussian(law.value(), keepFirst, observeNone,
                                            emptyView.value(), solvePolicy(),
                                            {64U, 2048U, 1000U});
  VSS_CHECK(marginal.hasValue());
  if (marginal) {
    VSS_CHECK(marginal.value().dimension() == 1U);
    VSS_CHECK(marginal.value().scale().at(0U, 0U) == 1.0);
  }
}

} // namespace

int main() {
  likelihoodAtGaussianCenter();
  gaussianConditioningAndReplay();
  singularSupportIsPreserved();
  normalSamplingUsesOneWordPerRetainedCoordinate();
  jointMomentsAndRankPolicy();
  singularConditioningAndEmptyMarginal();
}

// VSS-MATH-TRACEABILITY: C060 CASE-C060.normal.factor
// VSS-MATH-TRACEABILITY: C061 CASE-C061.conditional-gaussian
// VSS-MATH-TRACEABILITY: C061 CASE-C061.multivariate-sample-replay
// VSS-MATH-TRACEABILITY: C059 CASE-C059.singular-conditional-correlated
