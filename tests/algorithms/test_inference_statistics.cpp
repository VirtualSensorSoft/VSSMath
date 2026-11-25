#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Diagnostics.hpp>
#include <vss/math/statistics/Inference.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

void varianceIntervalUsesExactNormalIidChiSquareQuantiles() {
  const auto interval =
      varianceConfidenceInterval(1.0, 3U, 0.5, NormalIID{true});
  VSS_CHECK(interval.hasValue());
  VSS_CHECK(std::abs(interval.value().lower - (1.0 / std::log(4.0))) < 1e-12);
  VSS_CHECK(std::abs(interval.value().upper - (1.0 / std::log(4.0 / 3.0))) <
            1e-12);

  const auto zero = varianceConfidenceInterval(0.0, 3U, 0.95, NormalIID{true});
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().lower == 0.0);
  VSS_CHECK(zero.value().upper == 0.0);
}

void varianceIntervalRejectsInvalidPremisesAndUnrepresentableEndpoints() {
  const auto premise =
      varianceConfidenceInterval(1.0, 3U, 0.5, NormalIID{false});
  VSS_CHECK(!premise.hasValue());
  VSS_CHECK(premise.status().code() == ErrorCode::InvalidDomain);

  const auto count = varianceConfidenceInterval(1.0, 1U, 0.5, NormalIID{true});
  VSS_CHECK(!count.hasValue());
  VSS_CHECK(count.status().code() == ErrorCode::InsufficientSamples);

  const auto confidence =
      varianceConfidenceInterval(1.0, 3U, 1.0, NormalIID{true});
  VSS_CHECK(!confidence.hasValue());
  VSS_CHECK(confidence.status().code() == ErrorCode::InvalidDomain);

  const auto negative =
      varianceConfidenceInterval(-1.0, 3U, 0.5, NormalIID{true});
  VSS_CHECK(!negative.hasValue());
  VSS_CHECK(negative.status().code() == ErrorCode::InvalidDomain);

  const auto overflow = varianceConfidenceInterval(
      std::numeric_limits<Scalar>::max(), 4U, 0.95, NormalIID{true});
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
}

void hotellingT2MatchesExactBivariateReference() {
  Matrix rows(4U, 2U);
  rows.at(0U, 0U) = 1.0;
  rows.at(1U, 0U) = -1.0;
  rows.at(2U, 1U) = 1.0;
  rows.at(3U, 1U) = -1.0;
  const std::array<Index, 1U> shape{2U};
  const SampleView sample{
      .rows = rows.view(), .observation_shape = shape, .valid = {}};
  Vector target(2U);
  target.at(0U) = 1.0;
  const SolvePolicy solvePolicy{.iteration = {},
                                .rank_abs = 0.0,
                                .rank_rel = 1.0e-12,
                                .reciprocal_condition_floor = 1.0e-12,
                                .rank = RankPolicy::reject,
                                .restart = 0U};
  const auto result =
      hotellingT2(sample, target.view(), NormalIID{true}, solvePolicy,
                  Limits{.elements = 64U, .bytes = 4096U, .evaluations = 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().statistic - 6.0) < 1.0e-12);
  VSS_CHECK(result.value().p_value.has_value());
  if (!result.value().p_value.has_value()) {
    VSS_CHECK(false);
    return;
  }
  VSS_CHECK(std::abs(result.value().p_value.value_or(
                         std::numeric_limits<Scalar>::quiet_NaN()) -
                     (1.0 / 3.0)) < 1.0e-12);
  VSS_CHECK(!result.value().effective_df.has_value());
  VSS_CHECK(!hotellingT2(sample, target.view(), NormalIID{false}, solvePolicy,
                         Limits{64U, 4096U, 64U})
                 .hasValue());
}

void hotellingT2HasDeclaredGaussianNullRejectionRate() {
  // A fixed seed makes the statistical acceptance case reproducible; this is
  // not a security RNG.
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937_64 generator(0x5a17c9e3ULL);
  const auto standardNormal = [&]() {
    const Scalar u1 =
        (static_cast<Scalar>(generator() >> 11U) + 0.5) / 9007199254740992.0;
    const Scalar u2 =
        (static_cast<Scalar>(generator() >> 11U) + 0.5) / 9007199254740992.0;
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * std::acos(-1.0) * u2);
  };
  const SolvePolicy solvePolicy{.iteration = {},
                                .rank_abs = 0.0,
                                .rank_rel = 1.0e-12,
                                .reciprocal_condition_floor = 1.0e-12,
                                .rank = RankPolicy::reject,
                                .restart = 0U};
  Index rejected = 0U;
  for (Index trial = 0U; trial < 512U; ++trial) {
    Matrix rows(4U, 2U);
    for (Index i = 0U; i < 4U; ++i) {
      rows.at(i, 0U) = standardNormal();
      rows.at(i, 1U) = standardNormal();
    }
    const std::array<Index, 1U> shape{2U};
    const SampleView sample{
        .rows = rows.view(), .observation_shape = shape, .valid = {}};
    Vector target(2U);
    const auto result = hotellingT2(
        sample, target.view(), NormalIID{true}, solvePolicy,
        Limits{.elements = 64U, .bytes = 4096U, .evaluations = 64U});
    VSS_CHECK(result.hasValue());
    if (!result.value().p_value.has_value()) {
      VSS_CHECK(false);
      continue;
    }
    rejected += result.value().p_value.value_or(
                    std::numeric_limits<Scalar>::quiet_NaN()) <= (1.0 / 3.0)
                    ? 1U
                    : 0U;
  }
  const Scalar rate = static_cast<Scalar>(rejected) / 512.0;
  VSS_CHECK(std::abs(rate - (1.0 / 3.0)) <= 0.13);
}

void ljungBoxReportsUncalibratedStatisticAndDeclaredTails() {
  const std::array<Scalar, 3U> zeroCorrelation{1.0, 0.0, -1.0};
  const auto zero =
      ljungBox(ConstVectorView::create(zeroCorrelation, 3U).value(), 1U, 0U,
               Limits{.elements = 8U, .bytes = 128U, .evaluations = 8U});
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().statistic == 0.0);
  VSS_CHECK(zero.value().effective_df == 1.0);
  VSS_CHECK(!zero.value().p_value.has_value());

  const std::array<Scalar, 3U> correlated{0.0, 1.0, 2.0};
  const auto uncalibrated =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 0U,
               Limits{.elements = 8U, .bytes = 128U, .evaluations = 8U});
  VSS_CHECK(uncalibrated.hasValue());
  VSS_CHECK(std::abs(uncalibrated.value().statistic - 3.75) < 1.0e-12);
  VSS_CHECK(!uncalibrated.value().p_value.has_value());
  const auto whiteNoise =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 0U,
               LjungBoxReference::white_noise,
               Limits{.elements = 8U, .bytes = 128U, .evaluations = 8U});
  VSS_CHECK(whiteNoise.hasValue());
  VSS_CHECK(std::abs(whiteNoise.value().p_value.value_or(
                         std::numeric_limits<Scalar>::quiet_NaN()) -
                     std::exp(-1.875)) < 1.0e-12);
  const auto fitted =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 1U,
               LjungBoxReference::fitted_arma,
               Limits{.elements = 8U, .bytes = 128U, .evaluations = 8U});
  VSS_CHECK(fitted.hasValue());
  VSS_CHECK(std::abs(fitted.value().p_value.value_or(
                         std::numeric_limits<Scalar>::quiet_NaN()) -
                     std::erfc(std::sqrt(1.875))) < 1.0e-12);
  const auto invalidReference =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 1U,
               LjungBoxReference::white_noise,
               Limits{.elements = 8U, .bytes = 128U, .evaluations = 8U});
  VSS_CHECK(!invalidReference.hasValue());
}

void covarianceEstimatorUncertaintyMatchesWickIdentityAndSelection() {
  Matrix scalar(1U, 1U, 4.0);
  const CovarianceUncertaintySpec allScalar{
      .premise = NormalIID{true}, .plug_in = false, .selected = {}};
  const auto scalarUncertainty = covarianceEstimatorUncertainty(
      scalar.view(), 5U, allScalar,
      Limits{.elements = 16U, .bytes = 1024U, .evaluations = 16U});
  VSS_CHECK(scalarUncertainty.hasValue());
  VSS_CHECK(scalarUncertainty.value().value.rows() == 1U);
  VSS_CHECK(scalarUncertainty.value().value.at(0U, 0U) == 8.0);
  VSS_CHECK(scalarUncertainty.value().diagnostics.estimand ==
            Estimand::estimate);

  Matrix diagonal(2U, 2U);
  diagonal.at(0U, 0U) = 1.0;
  diagonal.at(1U, 1U) = 2.0;
  const std::array<CovarianceEntry, 3U> selected{
      CovarianceEntry{.row = 0U, .column = 0U},
      CovarianceEntry{.row = 1U, .column = 1U},
      CovarianceEntry{.row = 0U, .column = 1U}};
  const CovarianceUncertaintySpec selection{
      .premise = NormalIID{true}, .plug_in = false, .selected = selected};
  const auto result = covarianceEstimatorUncertainty(
      diagonal.view(), 5U, selection,
      Limits{.elements = 16U, .bytes = 1024U, .evaluations = 16U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().value.rows() == 3U);
  VSS_CHECK(std::abs(result.value().value.at(0U, 0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().value.at(1U, 1U) - 2.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().value.at(2U, 2U) - 0.5) < 1.0e-12);
  VSS_CHECK(result.value().value.at(0U, 1U) == 0.0);
  const CovarianceUncertaintySpec noPremise{
      .premise = NormalIID{false}, .plug_in = false, .selected = selected};
  VSS_CHECK(!covarianceEstimatorUncertainty(diagonal.view(), 5U, noPremise,
                                            Limits{16U, 1024U, 16U})
                 .hasValue());
  const auto capped = covarianceEstimatorUncertainty(
      diagonal.view(), 5U, allScalar,
      Limits{.elements = 1U, .bytes = 8U, .evaluations = 1U});
  VSS_CHECK(!capped.hasValue());
  VSS_CHECK(capped.status().code() == ErrorCode::ResourceLimit);
}

void covarianceBootstrapResamplesWholeRows() {
  Matrix rows(3U, 2U);
  for (Index i = 0U; i < 3U; ++i) {
    rows.at(i, 0U) = static_cast<Scalar>(i + 1U);
    rows.at(i, 1U) = 2.0 * rows.at(i, 0U);
  }
  const std::array<Index, 1U> shape{2U};
  const SampleView sample{
      .rows = rows.view(), .observation_shape = shape, .valid = {}};
  const BootstrapPolicy policy{.kind = BootstrapKind::iid,
                               .block_length = 1U,
                               .draws = 32U,
                               .confidence = 0.8,
                               .interval = QuantileInterval::percentile,
                               .invalid = InvalidDraw::reject};
  probability::Rng rng(71U);
  const auto result = bootstrapCovarianceUncertainty(
      sample, policy, rng,
      Limits{.elements = 4096U, .bytes = 32768U, .evaluations = 4096U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().estimates.rows() == 32U);
  VSS_CHECK(result.value().estimates.columns() == 4U);
  VSS_CHECK(result.value().intervals.rows() == 4U);
  VSS_CHECK(result.value().intervals.columns() == 2U);
  for (Index i = 0U; i < result.value().estimates.rows(); ++i) {
    const Scalar variance = result.value().estimates.at(i, 0U);
    VSS_CHECK(std::abs(result.value().estimates.at(i, 1U) - (2.0 * variance)) <
              1.0e-12);
    VSS_CHECK(std::abs(result.value().estimates.at(i, 3U) - (4.0 * variance)) <
              1.0e-12);
  }
}

void informationCriteriaAndGoodnessStatisticsMatchReferences() {
  VSS_CHECK(aic(-10.0, 2U).value() == 24.0);
  VSS_CHECK(std::abs(bic(-10.0, 2U, 4U).value() -
                     (20.0 + 2.0 * std::log(4.0))) < 1.0e-12);
  VSS_CHECK(aic(std::numeric_limits<Scalar>::infinity(), 0U).status().code() ==
            ErrorCode::NonFiniteInput);
  VSS_CHECK(aic(std::numeric_limits<Scalar>::max(), 0U).status().code() ==
            ErrorCode::NonFiniteResult);
  VSS_CHECK(bic(-1.0, 1U, 0U).status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(bic(std::numeric_limits<Scalar>::max(), 0U, 1U).status().code() ==
            ErrorCode::NonFiniteResult);

  const auto uniform = probability::uniform(0.0, 1.0);
  VSS_CHECK(uniform.hasValue());
  Matrix rows(2U, 1U);
  rows.at(0U, 0U) = 0.25;
  rows.at(1U, 0U) = 0.75;
  const SampleView sample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  probability::Rng rng(22U);
  const GoodnessCalibration statisticOnly{
      .statistic_only = true, .replicates = 0U, .refit_law = {}};
  const auto ksResult =
      ks(sample, uniform.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(ksResult.hasValue());
  VSS_CHECK(ksResult.value().statistic == 0.25);
  VSS_CHECK(!ksResult.value().p_value.has_value());
  rows.at(0U, 0U) = 0.5;
  rows.at(1U, 0U) = 0.5;
  const auto tiedKs =
      ks(sample, uniform.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(tiedKs.hasValue());
  VSS_CHECK(tiedKs.value().statistic == 0.5);
  rows.at(0U, 0U) = 1.5;
  rows.at(1U, 0U) = 0.75;
  const auto outsideKs =
      ks(sample, uniform.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(outsideKs.hasValue());
  VSS_CHECK(std::isinf(outsideKs.value().statistic));
  rows.at(0U, 0U) = 0.25;
  rows.at(1U, 0U) = 0.75;
  const auto adResult = andersonDarling(
      sample, uniform.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(adResult.hasValue());
  const Scalar expectedAd =
      -2.0 - (0.5 * std::log(1.0 / 16.0)) - (1.5 * std::log(9.0 / 16.0));
  VSS_CHECK(std::abs(adResult.value().statistic - expectedAd) < 1.0e-12);

  rows.at(0U, 0U) = 0.0;
  const auto boundary = andersonDarling(
      sample, uniform.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(boundary.hasValue());
  VSS_CHECK(std::isinf(boundary.value().statistic));

  rows.at(0U, 0U) = 0.0;
  rows.at(1U, 0U) = 1.0;
  const SampleView bernoulliSample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto bernoulli = probability::bernoulli(0.5);
  VSS_CHECK(bernoulli.hasValue());
  const auto discreteAd = andersonDarling(
      bernoulliSample, bernoulli.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(discreteAd.hasValue());
  VSS_CHECK(discreteAd.value().statistic == 0.0);
  const auto discreteKs =
      ks(bernoulliSample, bernoulli.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(discreteKs.hasValue());
  VSS_CHECK(discreteKs.value().statistic == 0.0);

  rows.at(0U, 0U) = 1.0;
  rows.at(1U, 0U) = 1.0;
  const SampleView bernoulliTail{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto discreteTailAd = andersonDarling(
      bernoulliTail, bernoulli.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(discreteTailAd.hasValue());
  VSS_CHECK(discreteTailAd.value().statistic == 1.0);

  const std::array<Scalar, 2U> atomLocations{0.5, 1.5};
  const std::array<Scalar, 2U> atomMasses{0.5, 0.5};
  const auto empiricalLaw = probability::empirical(
      ConstVectorView::create(atomLocations, 2U).value(),
      ConstVectorView::create(atomMasses, 2U).value(),
      Limits{.elements = 16U, .bytes = 1024U, .evaluations = 16U});
  VSS_CHECK(empiricalLaw.hasValue());
  const auto finiteAtoms = empiricalLaw.value().finiteAtoms();
  VSS_CHECK(finiteAtoms.locations.size() == 2U);
  VSS_CHECK(finiteAtoms.masses.size() == 2U);
  VSS_CHECK(finiteAtoms.locations[0U] == 0.5);
  VSS_CHECK(finiteAtoms.masses[1U] == 0.5);
  rows.at(0U, 0U) = 0.5;
  rows.at(1U, 0U) = 1.5;
  const SampleView empiricalSample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto empiricalKs =
      ks(empiricalSample, empiricalLaw.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(empiricalKs.hasValue());
  VSS_CHECK(empiricalKs.value().statistic == 0.0);
  const auto empiricalAd = andersonDarling(
      empiricalSample, empiricalLaw.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(empiricalAd.hasValue());
  VSS_CHECK(empiricalAd.value().statistic == 0.0);

  const auto poisson = probability::poisson(1.0);
  VSS_CHECK(poisson.hasValue());
  rows.at(0U, 0U) = 0.0;
  rows.at(1U, 0U) = 1.0;
  const SampleView poissonSample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto poissonAd = andersonDarling(
      poissonSample, poisson.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 256U});
  VSS_CHECK(poissonAd.hasValue());
  VSS_CHECK(std::abs(poissonAd.value().statistic - 0.35408100975136214) <
            1.0e-12);

  const std::array<Scalar, 1U> singleAtom{4.5};
  const std::array<Scalar, 1U> unitMass{1.0};
  const auto pointMass = probability::empirical(
      ConstVectorView::create(singleAtom, 1U).value(),
      ConstVectorView::create(unitMass, 1U).value(),
      Limits{.elements = 8U, .bytes = 512U, .evaluations = 8U});
  VSS_CHECK(pointMass.hasValue());
  rows.at(0U, 0U) = 4.5;
  rows.at(1U, 0U) = 4.5;
  const SampleView pointSample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto pointKs =
      ks(pointSample, pointMass.value(), statisticOnly, rng,
         Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  const auto pointAd = andersonDarling(
      pointSample, pointMass.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(pointKs.hasValue() && pointKs.value().statistic == 0.0);
  VSS_CHECK(pointAd.hasValue() && pointAd.value().statistic == 0.0);
  rows.at(1U, 0U) = 4.0;
  const auto outsidePoint = andersonDarling(
      pointSample, pointMass.value(), statisticOnly, rng,
      Limits{.elements = 32U, .bytes = 4096U, .evaluations = 100U});
  VSS_CHECK(outsidePoint.hasValue());
  VSS_CHECK(std::isinf(outsidePoint.value().statistic));
}

void goodnessCalibrationRefitsObservedAndEveryReplicate() {
  const auto normal = probability::normal(0.0, 1.0);
  VSS_CHECK(normal.hasValue());
  Matrix rows(3U, 1U);
  rows.at(0U, 0U) = -1.0;
  rows.at(1U, 0U) = 0.0;
  rows.at(2U, 0U) = 1.0;
  const SampleView sample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  Index calls = 0U;
  const GoodnessCalibration calibration{
      .statistic_only = false,
      .replicates = 8U,
      .refit_law = [&calls](ConstVectorView values) {
        ++calls;
        Scalar mean = 0.0;
        for (Index i = 0U; i < values.size(); ++i) {
          mean += values.at(i) / static_cast<Scalar>(values.size());
        }
        return probability::normal(mean, 1.0);
      }};
  probability::Rng rng(551U);
  const auto result =
      ks(sample, normal.value(), calibration, rng,
         Limits{.elements = 128U, .bytes = 16384U, .evaluations = 1000U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(calls == 9U);
  VSS_CHECK(result.value().calibration_method ==
            "approximate_composite_bootstrap");
  VSS_CHECK(result.value().sampling_error.has_value());

  Index failedCalls = 0U;
  const GoodnessCalibration failedCalibration{
      .statistic_only = false,
      .replicates = 8U,
      .refit_law = [&failedCalls](ConstVectorView) {
        ++failedCalls;
        return Result<probability::Distribution>::failure(
            ErrorCode::InvalidDomain, "scripted refit failure");
      }};
  const auto failed =
      ks(sample, normal.value(), failedCalibration, rng,
         Limits{.elements = 128U, .bytes = 16384U, .evaluations = 1000U});
  VSS_CHECK(!failed.hasValue());
  VSS_CHECK(failed.status().code() == ErrorCode::CallbackFailure);
  VSS_CHECK(failedCalls == 1U);
}

} // namespace

int main() {
  varianceIntervalUsesExactNormalIidChiSquareQuantiles();
  varianceIntervalRejectsInvalidPremisesAndUnrepresentableEndpoints();
  hotellingT2MatchesExactBivariateReference();
  hotellingT2HasDeclaredGaussianNullRejectionRate();
  ljungBoxReportsUncalibratedStatisticAndDeclaredTails();
  covarianceEstimatorUncertaintyMatchesWickIdentityAndSelection();
  covarianceBootstrapResamplesWholeRows();
  informationCriteriaAndGoodnessStatisticsMatchReferences();
  goodnessCalibrationRefitsObservedAndEveryReplicate();
}
