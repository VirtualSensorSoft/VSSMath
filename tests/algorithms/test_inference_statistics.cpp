#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Inference.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

void varianceIntervalUsesExactNormalIidChiSquareQuantiles() {
  const auto interval =
      varianceConfidenceInterval(1.0, 3U, 0.5, NormalIID{true});
  VSS_CHECK(interval.hasValue());
  VSS_CHECK(std::abs(interval.value().lower - 1.0 / std::log(4.0)) < 1e-12);
  VSS_CHECK(std::abs(interval.value().upper - 1.0 / std::log(4.0 / 3.0)) <
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
  const SampleView sample{rows.view(), shape, {}};
  Vector target(2U);
  target.at(0U) = 1.0;
  const SolvePolicy solvePolicy{.iteration = {},
                                .rank_abs = 0.0,
                                .rank_rel = 1.0e-12,
                                .reciprocal_condition_floor = 1.0e-12,
                                .rank = RankPolicy::reject,
                                .restart = 0U};
  const auto result = hotellingT2(sample, target.view(), NormalIID{true},
                                  solvePolicy, Limits{64U, 4096U, 64U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().statistic - 6.0) < 1.0e-12);
  VSS_CHECK(result.value().p_value.has_value());
  VSS_CHECK(std::abs(*result.value().p_value - 1.0 / 3.0) < 1.0e-12);
  VSS_CHECK(!result.value().effective_df.has_value());
  VSS_CHECK(!hotellingT2(sample, target.view(), NormalIID{false}, solvePolicy,
                         Limits{64U, 4096U, 64U})
                 .hasValue());
}

void hotellingT2HasDeclaredGaussianNullRejectionRate() {
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
    const SampleView sample{rows.view(), shape, {}};
    Vector target(2U);
    const auto result = hotellingT2(sample, target.view(), NormalIID{true},
                                    solvePolicy, Limits{64U, 4096U, 64U});
    VSS_CHECK(result.hasValue());
    rejected += *result.value().p_value <= (1.0 / 3.0) ? 1U : 0U;
  }
  const Scalar rate = static_cast<Scalar>(rejected) / 512.0;
  VSS_CHECK(std::abs(rate - 1.0 / 3.0) <= 0.13);
}

void ljungBoxReportsUncalibratedStatisticAndDeclaredTails() {
  const std::array<Scalar, 3U> zeroCorrelation{1.0, 0.0, -1.0};
  const auto zero =
      ljungBox(ConstVectorView::create(zeroCorrelation, 3U).value(), 1U, 0U,
               Limits{8U, 128U, 8U});
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().statistic == 0.0);
  VSS_CHECK(zero.value().effective_df == 1.0);
  VSS_CHECK(!zero.value().p_value.has_value());

  const std::array<Scalar, 3U> correlated{0.0, 1.0, 2.0};
  const auto uncalibrated =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 0U,
               Limits{8U, 128U, 8U});
  VSS_CHECK(uncalibrated.hasValue());
  VSS_CHECK(std::abs(uncalibrated.value().statistic - 3.75) < 1.0e-12);
  VSS_CHECK(!uncalibrated.value().p_value.has_value());
  const auto whiteNoise =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 0U,
               LjungBoxReference::white_noise, Limits{8U, 128U, 8U});
  VSS_CHECK(whiteNoise.hasValue());
  VSS_CHECK(std::abs(*whiteNoise.value().p_value - std::exp(-1.875)) < 1.0e-12);
  const auto fitted =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 1U,
               LjungBoxReference::fitted_arma, Limits{8U, 128U, 8U});
  VSS_CHECK(fitted.hasValue());
  VSS_CHECK(std::abs(*fitted.value().p_value - std::erfc(std::sqrt(1.875))) <
            1.0e-12);
  const auto invalidReference =
      ljungBox(ConstVectorView::create(correlated, 3U).value(), 2U, 1U,
               LjungBoxReference::white_noise, Limits{8U, 128U, 8U});
  VSS_CHECK(!invalidReference.hasValue());
}

void covarianceEstimatorUncertaintyMatchesWickIdentityAndSelection() {
  Matrix scalar(1U, 1U, 4.0);
  const CovarianceUncertaintySpec allScalar{NormalIID{true}, false, {}};
  const auto scalarUncertainty = covarianceEstimatorUncertainty(
      scalar.view(), 5U, allScalar, Limits{16U, 1024U, 16U});
  VSS_CHECK(scalarUncertainty.hasValue());
  VSS_CHECK(scalarUncertainty.value().value.rows() == 1U);
  VSS_CHECK(scalarUncertainty.value().value.at(0U, 0U) == 8.0);
  VSS_CHECK(scalarUncertainty.value().diagnostics.estimand ==
            Estimand::estimate);

  Matrix diagonal(2U, 2U);
  diagonal.at(0U, 0U) = 1.0;
  diagonal.at(1U, 1U) = 2.0;
  const std::array<CovarianceEntry, 3U> selected{CovarianceEntry{0U, 0U},
                                                 CovarianceEntry{1U, 1U},
                                                 CovarianceEntry{0U, 1U}};
  const CovarianceUncertaintySpec selection{NormalIID{true}, false, selected};
  const auto result = covarianceEstimatorUncertainty(
      diagonal.view(), 5U, selection, Limits{16U, 1024U, 16U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().value.rows() == 3U);
  VSS_CHECK(std::abs(result.value().value.at(0U, 0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().value.at(1U, 1U) - 2.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().value.at(2U, 2U) - 0.5) < 1.0e-12);
  VSS_CHECK(result.value().value.at(0U, 1U) == 0.0);
  const CovarianceUncertaintySpec noPremise{NormalIID{false}, false, selected};
  VSS_CHECK(!covarianceEstimatorUncertainty(diagonal.view(), 5U, noPremise,
                                            Limits{16U, 1024U, 16U})
                 .hasValue());
  const auto capped = covarianceEstimatorUncertainty(
      diagonal.view(), 5U, allScalar, Limits{1U, 8U, 1U});
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
  const SampleView sample{rows.view(), shape, {}};
  const BootstrapPolicy policy{.kind = BootstrapKind::iid,
                               .block_length = 1U,
                               .draws = 32U,
                               .confidence = 0.8,
                               .interval = QuantileInterval::percentile,
                               .invalid = InvalidDraw::reject};
  probability::Rng rng(71U);
  const auto result = bootstrapCovarianceUncertainty(
      sample, policy, rng, Limits{4096U, 32768U, 4096U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().estimates.rows() == 32U);
  VSS_CHECK(result.value().estimates.columns() == 4U);
  VSS_CHECK(result.value().intervals.rows() == 4U);
  VSS_CHECK(result.value().intervals.columns() == 2U);
  for (Index i = 0U; i < result.value().estimates.rows(); ++i) {
    const Scalar variance = result.value().estimates.at(i, 0U);
    VSS_CHECK(std::abs(result.value().estimates.at(i, 1U) - 2.0 * variance) <
              1.0e-12);
    VSS_CHECK(std::abs(result.value().estimates.at(i, 3U) - 4.0 * variance) <
              1.0e-12);
  }
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
}
