#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/covariance/Structured.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/Reduction.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Sparse.hpp>
#include <vss/math/linear/Structured.hpp>
#include <vss/math/statistics/Cumulants.hpp>
#include <vss/math/statistics/Descriptive.hpp>
#include <vss/math/statistics/MomentDiagnostics.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>
#include <vss/math/version.hpp>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "math_consumer: " << message << '\n';
    return false;
  }
  return true;
}

template <class ResultType>
bool requireError(const ResultType& result, vss::math::ErrorCode code,
                  const char* message) {
  return require(!result.hasValue() && result.status().code() == code, message);
}

template <std::size_t N>
vss::math::Matrix
matrixFromArray(const std::array<vss::math::Scalar, N>& values,
                vss::math::Extent rows, vss::math::Extent columns) {
  vss::math::Matrix result(rows, columns);
  for (std::size_t index = 0U; index < N; ++index) {
    result.at(index / columns, index % columns) = values[index];
  }
  return result;
}

bool verifyRobustOrderStatisticContracts() {
  using namespace vss::math;

  constexpr Limits ample{4096U, 262144U, 0U};
  const Vector values{std::array<Scalar, 4U>{1.0, 2.0, 3.0, 5.0}};
  const auto mad =
      statistics::medianAbsoluteDeviation(values.view(), 1.0, ample);
  const auto linearIqr = statistics::interquartileRange(
      values.view(), QuantileRule::linear, ample);
  const auto inverseIqr = statistics::interquartileRange(
      values.view(), QuantileRule::inverse_empirical, ample);
  if (!require(mad.hasValue() && mad.value() == 1.0 && linearIqr.hasValue() &&
                   linearIqr.value() == 1.75 && inverseIqr.hasValue() &&
                   inverseIqr.value() == 2.0,
               "installed MAD/IQR rules disagree with their contracts")) {
    return false;
  }

  const Vector empty;
  if (!requireError(
          statistics::medianAbsoluteDeviation(empty.view(), 1.0, ample),
          ErrorCode::InsufficientSamples,
          "installed MAD accepted an empty sample") ||
      !requireError(statistics::interquartileRange(empty.view(),
                                                   QuantileRule::linear, ample),
                    ErrorCode::InsufficientSamples,
                    "installed IQR accepted an empty sample") ||
      !requireError(
          statistics::medianAbsoluteDeviation(values.view(), -1.0, ample),
          ErrorCode::InvalidDomain,
          "installed MAD accepted a negative scale")) {
    return false;
  }

  const auto zeroScale =
      statistics::medianAbsoluteDeviation(values.view(), 0.0, ample);
  if (!require(zeroScale.hasValue() && zeroScale.value() == 0.0,
               "installed MAD rejected the valid zero scale")) {
    return false;
  }

  const Vector nonfinite{
      std::array<Scalar, 2U>{1.0, std::numeric_limits<Scalar>::infinity()}};
  if (!requireError(
          statistics::medianAbsoluteDeviation(nonfinite.view(), 1.0, ample),
          ErrorCode::NonFiniteInput,
          "installed MAD accepted non-finite input") ||
      !requireError(statistics::interquartileRange(nonfinite.view(),
                                                   QuantileRule::linear, ample),
                    ErrorCode::NonFiniteInput,
                    "installed IQR accepted non-finite input")) {
    return false;
  }
  constexpr Limits tooSmall{1U, 262144U, 0U};
  if (!requireError(statistics::interquartileRange(
                        values.view(), QuantileRule::linear, tooSmall),
                    ErrorCode::ResourceLimit,
                    "installed IQR ignored its element limit")) {
    return false;
  }

  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Vector wideMad{std::array<Scalar, 4U>{-1.0, 1.0, 3.0, 5.0}};
  if (!requireError(
          statistics::medianAbsoluteDeviation(wideMad.view(), maximum, ample),
          ErrorCode::NonFiniteResult,
          "installed MAD returned an unrepresentable result")) {
    return false;
  }
  const Vector wideIqr{
      std::array<Scalar, 4U>{-maximum, -maximum, maximum, maximum}};
  return requireError(statistics::interquartileRange(
                          wideIqr.view(), QuantileRule::linear, ample),
                      ErrorCode::NonFiniteResult,
                      "installed IQR returned an unrepresentable range");
}

bool verifyPopulationCovarianceContracts() {
  using namespace vss::math;

  constexpr Limits ample{4096U, 262144U, 0U};
  constexpr statistics::PopulationPolicy empirical{
      .normalization = statistics::VarianceMode::empirical,
      .missingness = statistics::MissingMode::reject,
      .estimand = Estimand::observed};
  constexpr statistics::PopulationPolicy samplePolicy{
      .normalization = statistics::VarianceMode::iid_sample,
      .missingness = statistics::MissingMode::reject,
      .estimand = Estimand::observed};
  constexpr statistics::CenterSpec estimated{.rule = CenterRule::estimated,
                                             .center = std::nullopt};

  const Matrix left = matrixFromArray(
      std::array<Scalar, 6U>{0.0, 1.0, 1.0, 0.0, 2.0, 2.0}, 3U, 2U);
  const Matrix right = matrixFromArray(
      std::array<Scalar, 6U>{2.0, 0.0, 1.0, 4.0, 5.0, 1.0}, 3U, 2U);
  constexpr std::array<Index, 1U> shape{2U};
  const SampleView leftView{
      .rows = left.view(), .observation_shape = shape, .valid = {}};
  const SampleView rightView{
      .rows = right.view(), .observation_shape = shape, .valid = {}};
  const auto cross = statistics::crossCovariance(leftView, rightView, estimated,
                                                 estimated, empirical, ample);
  const auto reverse = statistics::crossCovariance(
      rightView, leftView, estimated, estimated, empirical, ample);
  if (!require(
          cross.hasValue() && reverse.hasValue() &&
              cross.value().value.rows() == 2U &&
              cross.value().value.columns() == 2U &&
              cross.value().value.at(0U, 0U) == 1.0 &&
              std::abs(cross.value().value.at(0U, 1U) - 1.0 / 3.0) < 1.0e-12 &&
              std::abs(cross.value().value.at(1U, 0U) - 4.0 / 3.0) < 1.0e-12 &&
              cross.value().value.at(1U, 1U) == -1.0 &&
              reverse.value().value.rows() == 2U &&
              reverse.value().value.columns() == 2U &&
              reverse.value().value.at(0U, 0U) ==
                  cross.value().value.at(0U, 0U) &&
              reverse.value().value.at(0U, 1U) ==
                  cross.value().value.at(1U, 0U) &&
              reverse.value().value.at(1U, 0U) ==
                  cross.value().value.at(0U, 1U) &&
              reverse.value().value.at(1U, 1U) ==
                  cross.value().value.at(1U, 1U),
          "installed cross-covariance lost shape, values, or orientation")) {
    return false;
  }

  const Matrix scalarLeft =
      matrixFromArray(std::array<Scalar, 3U>{1.0, 2.0, 4.0}, 3U, 1U);
  const Matrix scalarRight =
      matrixFromArray(std::array<Scalar, 3U>{2.0, 1.0, 5.0}, 3U, 1U);
  const auto scalarEmpirical =
      statistics::crossCovariance(SampleView{.rows = scalarLeft.view()},
                                  SampleView{.rows = scalarRight.view()},
                                  estimated, estimated, empirical, ample);
  const auto scalarSample =
      statistics::crossCovariance(SampleView{.rows = scalarLeft.view()},
                                  SampleView{.rows = scalarRight.view()},
                                  estimated, estimated, samplePolicy, ample);
  if (!require(
          scalarEmpirical.hasValue() && scalarSample.hasValue() &&
              std::abs(scalarEmpirical.value().value.at(0U, 0U) - 16.0 / 9.0) <
                  1.0e-12 &&
              std::abs(scalarSample.value().value.at(0U, 0U) - 8.0 / 3.0) <
                  1.0e-12,
          "installed scalar cross-covariance normalization is incorrect")) {
    return false;
  }

  Matrix retained = matrixFromArray(
      std::array<Scalar, 6U>{1.0, 2.0, 3.0, 6.0, 100.0, 200.0}, 3U, 2U);
  constexpr std::array<std::uint8_t, 6U> valid{1U, 1U, 1U, 1U, 0U, 0U};
  constexpr std::array<Index, 1U> vectorShape{2U};
  const SampleView masked{.rows = retained.view(),
                          .observation_shape = vectorShape,
                          .valid = valid};
  const statistics::PopulationPolicy commonSupport{
      .normalization = statistics::VarianceMode::empirical,
      .missingness = statistics::MissingMode::complete_row,
      .estimand = Estimand::observed};
  const statistics::PopulationPolicy commonSample{
      .normalization = statistics::VarianceMode::iid_sample,
      .missingness = statistics::MissingMode::complete_row,
      .estimand = Estimand::observed};
  const auto retainedCovariance =
      statistics::covarianceMatrix(masked, estimated, commonSupport, ample);
  const auto retainedSampleCovariance =
      statistics::covarianceMatrix(masked, estimated, commonSample, ample);
  Vector origin(2U, 0.0);
  const statistics::CenterSpec knownOrigin{.rule = CenterRule::known,
                                           .center = origin.view()};
  const statistics::PopulationPolicy knownCenter{
      .normalization = statistics::VarianceMode::known_center,
      .missingness = statistics::MissingMode::complete_row,
      .estimand = Estimand::observed};
  const auto knownCenterCovariance =
      statistics::covarianceMatrix(masked, knownOrigin, knownCenter, ample);
  if (!require(retainedCovariance.hasValue() &&
                   retainedCovariance.value().diagnostics.global.valid == 2U &&
                   retainedCovariance.value().value.at(0U, 1U) == 2.0 &&
                   retainedSampleCovariance.hasValue() &&
                   retainedSampleCovariance.value().value.at(0U, 1U) == 4.0 &&
                   knownCenterCovariance.hasValue() &&
                   knownCenterCovariance.value().value.at(0U, 1U) == 10.0,
               "installed covariance did not retain common complete rows")) {
    return false;
  }

  const Matrix oneRow = matrixFromArray(std::array<Scalar, 1U>{1.0}, 1U, 1U);
  const Matrix twoRows =
      matrixFromArray(std::array<Scalar, 2U>{1.0, 2.0}, 2U, 1U);
  const auto mismatch = statistics::crossCovariance(
      SampleView{.rows = oneRow.view()}, SampleView{.rows = twoRows.view()},
      estimated, estimated, empirical, ample);
  return requireError(mismatch, ErrorCode::DimensionMismatch,
                      "installed cross-covariance accepted mismatched rows");
}

bool verifyCorrelationContracts() {
  using namespace vss::math;

  constexpr Limits ample{4096U, 262144U, 0U};
  SolvePolicy policy{};
  policy.iteration.max_iterations = 128U;
  policy.rank = RankPolicy::supported_subspace;
  policy.reciprocal_condition_floor = 0.5;
  const Matrix covariance =
      matrixFromArray(std::array<Scalar, 4U>{4.0, 3.0, 3.0, 9.0}, 2U, 2U);
  const auto correlation =
      statistics::correlationMatrix(covariance.view(), policy, ample);
  if (!require(
          correlation.hasValue() && correlation.value().rank == 2U &&
              correlation.value().classification ==
                  covariance::CovarianceKind::ill_conditioned &&
              std::abs(correlation.value().covariance.at(0U, 1U) - 0.5) <
                  1.0e-12 &&
              std::abs(correlation.value().reciprocal_condition.value_or(0.0) -
                       1.0 / 3.0) < 1.0e-12,
          "installed correlation scaling/conditioning is incorrect")) {
    return false;
  }

  const Matrix singular =
      matrixFromArray(std::array<Scalar, 4U>{1.0, -1.0, -1.0, 1.0}, 2U, 2U);
  const auto singularCorrelation =
      statistics::correlationMatrix(singular.view(), policy, ample);
  if (!require(singularCorrelation.hasValue() &&
                   singularCorrelation.value().rank == 1U &&
                   singularCorrelation.value().covariance.at(0U, 1U) == -1.0,
               "installed correlation rejected a valid singular matrix")) {
    return false;
  }

  const Matrix redundant = matrixFromArray(
      std::array<Scalar, 16U>{1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,
                              1.0, 1.0, 0.5, 0.5, 1.0, 1.0},
      4U, 4U);
  constexpr std::array<Index, 2U> selected{0U, 1U};
  constexpr std::array<Index, 2U> conditioned{2U, 3U};
  auto strictPolicy = policy;
  strictPolicy.rank = RankPolicy::reject;
  strictPolicy.reciprocal_condition_floor = 0.0;
  const auto strictResult = statistics::partialCorrelation(
      redundant.view(), selected, conditioned, strictPolicy, ample);
  auto supportedPolicy = policy;
  supportedPolicy.reciprocal_condition_floor = 0.0;
  const auto supportedResult = statistics::partialCorrelation(
      redundant.view(), selected, conditioned, supportedPolicy, ample);
  if (!requireError(
          strictResult, ErrorCode::RankDeficient,
          "installed strict partial correlation accepted a rank loss") ||
      !require(supportedResult.hasValue() &&
                   std::abs(supportedResult.value().covariance.at(0U, 1U) -
                            1.0 / 3.0) < 1.0e-12,
               "installed partial correlation mishandled redundant support")) {
    return false;
  }

  const Matrix identical =
      matrixFromArray(std::array<Scalar, 4U>{1.0, 1.0, 1.0, 1.0}, 2U, 2U);
  constexpr std::array<Index, 1U> oneSelected{0U};
  constexpr std::array<Index, 1U> oneConditioned{1U};
  if (!requireError(
          statistics::partialCorrelation(identical.view(), oneSelected,
                                         oneConditioned, supportedPolicy,
                                         ample),
          ErrorCode::InvalidDomain,
          "installed partial correlation accepted zero residual variance")) {
    return false;
  }

  const Matrix zeroVariance =
      matrixFromArray(std::array<Scalar, 4U>{0.0, 0.0, 0.0, 1.0}, 2U, 2U);
  return requireError(
      statistics::correlationMatrix(zeroVariance.view(), policy, ample),
      ErrorCode::InvalidDomain,
      "installed correlation fabricated zero-variance units");
}

bool verifyWeightedPopulationContracts() {
  using namespace vss::math;

  constexpr Limits ample{4096U, 262144U, 0U};
  const Matrix values =
      matrixFromArray(std::array<Scalar, 5U>{1.0, 2.0, 4.0, 8.0, 16.0}, 5U, 1U);
  const Vector masses{std::array<Scalar, 5U>{0.1, 0.15, 0.2, 0.25, 0.3}};
  const SampleView sample{.rows = values.view()};
  const statistics::WeightSpec empirical{.rule = WeightRule::empirical,
                                         .rowWeights = masses.view(),
                                         .componentWeights = std::nullopt,
                                         .frequencyCounts = {}};
  const statistics::WeightSpec fixed{.rule = WeightRule::fixed_iid,
                                     .rowWeights = masses.view(),
                                     .componentWeights = std::nullopt,
                                     .frequencyCounts = {}};
  const auto mean = statistics::weightedMean(sample, empirical, ample);
  const auto covariance =
      statistics::weightedEmpiricalCovariance(sample, empirical, ample);
  const auto unbiasedCovariance =
      statistics::unbiasedFixedWeightedCovariance(sample, fixed, ample);
  if (!require(mean.hasValue() && mean.value().value.at(0U) == 8.0 &&
                   covariance.hasValue() &&
                   std::abs(covariance.value().value.at(0U, 0U) - 32.7) <
                       1.0e-12 &&
                   unbiasedCovariance.hasValue() &&
                   std::abs(unbiasedCovariance.value().value.at(0U, 0U) -
                            1308.0 / 31.0) < 1.0e-12,
               "installed weighted mean/covariance estimators are incorrect")) {
    return false;
  }

  const Matrix componentValues =
      matrixFromArray(std::array<Scalar, 4U>{2.0, 10.0, 8.0, 20.0}, 2U, 2U);
  const Matrix componentMasses =
      matrixFromArray(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 0.0}, 2U, 2U);
  constexpr std::array<Index, 1U> vectorShape{2U};
  const SampleView componentSample{.rows = componentValues.view(),
                                   .observation_shape = vectorShape};
  const statistics::WeightSpec componentwise{.rule = WeightRule::componentwise,
                                             .rowWeights = std::nullopt,
                                             .componentWeights =
                                                 componentMasses.view(),
                                             .frequencyCounts = {}};
  const auto componentMean =
      statistics::weightedMean(componentSample, componentwise, ample);
  if (!require(componentMean.hasValue() &&
                   componentMean.value().value.at(0U) == 6.5 &&
                   componentMean.value().value.at(1U) == 10.0 &&
                   componentMean.value()
                           .diagnostics.coordinate_support.view()
                           .size() == 2U &&
                   componentMean.value()
                           .diagnostics.coordinate_support.view()[1U]
                           .positive == 1U,
               "installed componentwise weighting lost channel support")) {
    return false;
  }

  const Matrix jointValues =
      matrixFromArray(std::array<Scalar, 4U>{0.0, 6.0, 6.0, 0.0}, 2U, 2U);
  const SampleView jointSample{.rows = jointValues.view(),
                               .observation_shape = vectorShape};
  const Vector jointMasses{std::array<Scalar, 2U>{0.25, 0.75}};
  const statistics::WeightSpec jointWeights{.rule = WeightRule::empirical,
                                            .rowWeights = jointMasses.view(),
                                            .componentWeights = std::nullopt,
                                            .frequencyCounts = {}};
  const auto jointCovariance =
      statistics::weightedEmpiricalCovariance(jointSample, jointWeights, ample);
  if (!require(
          jointCovariance.hasValue() &&
              jointCovariance.value().value.rows() == 2U &&
              jointCovariance.value().value.columns() == 2U &&
              jointCovariance.value().value.at(0U, 1U) == -27.0 / 4.0,
          "installed joint weighted covariance lost shared-row correlation")) {
    return false;
  }
  const auto componentCovariance = statistics::weightedEmpiricalCovariance(
      jointSample, componentwise, ample);
  if (!requireError(componentCovariance, ErrorCode::InvalidDomain,
                    "installed covariance accepted componentwise weights")) {
    return false;
  }

  const Matrix twoRows =
      matrixFromArray(std::array<Scalar, 4U>{0.0, 0.0, 2.0, -4.0}, 2U, 2U);
  const SampleView vectorSample{.rows = twoRows.view(),
                                .observation_shape = vectorShape};
  const Vector dominantMasses{std::array<Scalar, 2U>{1.0, 1.0e6}};
  const statistics::WeightSpec dominantWeights{.rule = WeightRule::fixed_iid,
                                               .rowWeights =
                                                   dominantMasses.view(),
                                               .componentWeights = std::nullopt,
                                               .frequencyCounts = {}};
  const auto dominant = statistics::unbiasedFixedWeightedCovariance(
      vectorSample, dominantWeights, ample);
  if (!require(dominant.hasValue() &&
                   dominant.value().value.at(0U, 0U) == 2.0 &&
                   dominant.value().value.at(0U, 1U) == -4.0 &&
                   dominant.value().value.at(1U, 1U) == 8.0,
               "installed fixed-weight covariance mishandled extreme ratios")) {
    return false;
  }

  const Vector negativeMasses{std::array<Scalar, 2U>{1.0, -1.0}};
  const statistics::WeightSpec negativeWeights{.rule = WeightRule::empirical,
                                               .rowWeights =
                                                   negativeMasses.view(),
                                               .componentWeights = std::nullopt,
                                               .frequencyCounts = {}};
  const Vector zeroMasses{std::array<Scalar, 2U>{0.0, 0.0}};
  const statistics::WeightSpec zeroWeights{.rule = WeightRule::empirical,
                                           .rowWeights = zeroMasses.view(),
                                           .componentWeights = std::nullopt,
                                           .frequencyCounts = {}};
  const Matrix scalarPair =
      matrixFromArray(std::array<Scalar, 2U>{1.0, 3.0}, 2U, 1U);
  const SampleView scalarSample{.rows = scalarPair.view()};
  if (!requireError(
          statistics::weightedMean(scalarSample, negativeWeights, ample),
          ErrorCode::InvalidDomain,
          "installed empirical mean accepted negative mass") ||
      !requireError(statistics::weightedMean(scalarSample, zeroWeights, ample),
                    ErrorCode::InsufficientSamples,
                    "installed empirical mean accepted zero support") ||
      !requireError(statistics::unbiasedFixedWeightedCovariance(
                        vectorSample,
                        statistics::WeightSpec{.rule = WeightRule::fixed_iid,
                                               .rowWeights = zeroMasses.view(),
                                               .componentWeights = std::nullopt,
                                               .frequencyCounts = {}},
                        ample),
                    ErrorCode::InsufficientSamples,
                    "installed fixed covariance accepted zero support")) {
    return false;
  }

  const Vector equalMasses{std::array<Scalar, 3U>{1.0, 1.0, 1.0}};
  const Matrix equalValues =
      matrixFromArray(std::array<Scalar, 3U>{1.0, 2.0, 4.0}, 3U, 1U);
  const statistics::WeightSpec equalWeights{.rule = WeightRule::empirical,
                                            .rowWeights = equalMasses.view(),
                                            .componentWeights = std::nullopt,
                                            .frequencyCounts = {}};
  const auto equalMean = statistics::weightedMean(
      SampleView{.rows = equalValues.view()}, equalWeights, ample);
  return require(equalMean.hasValue() &&
                     equalMean.value().diagnostics.global.effective_count ==
                         3.0,
                 "installed equal-mass effective sample count is incorrect");
}

} // namespace

int main() {
  using namespace vss::math;

  if (!verifyRobustOrderStatisticContracts() ||
      !verifyPopulationCovarianceContracts() || !verifyCorrelationContracts()) {
    return 1;
  }
  if (!verifyWeightedPopulationContracts()) {
    return 1;
  }

  {
    const Vector retainedSample(std::array{1.0, 2.0, 4.0});
    const auto retainedMean = statistics::mean(retainedSample.view());
    const auto retainedPopulationVariance =
        statistics::populationVariance(retainedSample.view());
    const auto retainedSampleVariance =
        statistics::sampleVariance(retainedSample.view());
    const auto retainedPopulationStandardDeviation =
        statistics::populationStandardDeviation(retainedSample.view());
    const auto retainedSampleStandardDeviation =
        statistics::sampleStandardDeviation(retainedSample.view());
    if (!require(retainedMean.hasValue() &&
                     std::abs(retainedMean.value() - 7.0 / 3.0) < 1.0e-12 &&
                     retainedPopulationVariance.hasValue() &&
                     std::abs(retainedPopulationVariance.value() - 14.0 / 9.0) <
                         1.0e-12 &&
                     retainedSampleVariance.hasValue() &&
                     std::abs(retainedSampleVariance.value() - 7.0 / 3.0) <
                         1.0e-12 &&
                     retainedPopulationStandardDeviation.hasValue() &&
                     std::abs(retainedPopulationStandardDeviation.value() -
                              std::sqrt(14.0) / 3.0) < 1.0e-12 &&
                     retainedSampleStandardDeviation.hasValue() &&
                     std::abs(retainedSampleStandardDeviation.value() -
                              std::sqrt(7.0 / 3.0)) < 1.0e-12,
                 "installed retained descriptive APIs failed")) {
      return 1;
    }

    const Vector emptySample;
    if (!requireError(statistics::mean(emptySample.view()),
                      ErrorCode::InvalidExtent,
                      "installed retained mean accepted an empty sample") ||
        !requireError(statistics::populationVariance(emptySample.view()),
                      ErrorCode::InvalidExtent,
                      "installed retained population variance accepted an "
                      "empty sample") ||
        !requireError(
            statistics::sampleVariance(emptySample.view()),
            ErrorCode::InvalidExtent,
            "installed retained sample variance accepted an empty sample")) {
      return 1;
    }

    const Vector singletonSample(std::array{5.0});
    if (!require(
            statistics::populationVariance(singletonSample.view()).hasValue() &&
                statistics::populationVariance(singletonSample.view())
                        .value() == 0.0,
            "installed retained population variance singleton failed") ||
        !requireError(
            statistics::sampleVariance(singletonSample.view()),
            ErrorCode::InsufficientSamples,
            "installed retained sample variance accepted a singleton")) {
      return 1;
    }
  }

  const std::array<Extent, 2> tensorExtents{2U, 2U};
  const auto tensorShape = TensorShape::create(tensorExtents);
  if (!require(tensorShape.hasValue(), "tensor shape construction failed")) {
    return 1;
  }
  auto tensor = Tensor::create(tensorShape.value(), 0.0, TensorLimits{4U});
  if (!require(tensor.hasValue(), "tensor allocation failed")) {
    return 1;
  }
  tensor.value().at(std::array<Index, 2>{1U, 1U}) = 5.0;
  if (!require(tensor.value().at(std::array<Index, 2>{1U, 1U}) == 5.0,
               "installed tensor access failed")) {
    return 1;
  }

  const Limits limits{.elements = 64U, .bytes = 4096U, .evaluations = 64U};
  {
    const Vector input(std::array{2.0, 3.0});
    const Vector diagonal(std::array{4.0, 5.0});
    const auto action =
        linear::apply(linear::DiagonalMatrix{diagonal}, input.view(), limits);
    const auto structured = covariance::structuredCovariance(
        {covariance::StructureKind::diagonal_low_rank,
         diagonal.view(),
         std::nullopt,
         {},
         std::nullopt,
         std::nullopt},
        SolvePolicy{}, limits);
    if (!require(
            action.hasValue() && action.value().at(0) == 8 &&
                structured.hasValue(),
            "installed structured construction or diagonal action failed")) {
      return 1;
    }
    const auto applied = covariance::applyStructuredCovariance(
        structured.value(), input.view(), limits);
    if (!require(applied.hasValue() && applied.value().at(1) == 15,
                 "installed structured covariance action failed")) {
      return 1;
    }
    const auto sparse = linear::sparseCSRFromTriplets(
        2, 2, std::array{linear::Triplet{0, 1, 4}}, limits);
    if (!require(sparse.hasValue(), "installed sparse construction failed")) {
      return 1;
    }
    const auto sparse_product =
        linear::apply(sparse.value(), input.view(), limits);
    if (!require(sparse_product.hasValue() &&
                     sparse_product.value().at(0) == 12,
                 "installed sparse action failed")) {
      return 1;
    }
    const Vector iterative_rhs(std::array{2.0, 8.0});
    const Vector iterative_start(2U);
    const linear::LinearMap iterative_map{
        2U,
        2U,
        [](ConstVectorView x) {
          return Result<Vector>::success(
              Vector{std::array{2.0 * x.at(0U), 4.0 * x.at(1U)}});
        },
        {}};
    SolvePolicy iterative_policy;
    iterative_policy.iteration.abs = 1e-12;
    iterative_policy.iteration.rel = 1e-12;
    iterative_policy.iteration.max_iterations = 4U;
    iterative_policy.iteration.max_evaluations = limits.evaluations;
    iterative_policy.restart = 2U;
    const auto iterative =
        linear::cg(iterative_map, iterative_rhs.view(), iterative_start.view(),
                   nullptr, iterative_policy, limits);
    if (!require(iterative.hasValue() &&
                     iterative.value().termination == Termination::converged &&
                     std::abs(iterative.value().value.at(0U) - 1.0) < 1e-12 &&
                     std::abs(iterative.value().value.at(1U) - 2.0) < 1e-12,
                 "installed iterative solver API failed")) {
      return 1;
    }
  }
  const auto tensorProduct = hadamard(tensor.value(), tensor.value(), limits);
  if (!require(tensorProduct.hasValue() &&
                   tensorProduct.value().at(std::array<Index, 2>{1U, 1U}) ==
                       25.0,
               "installed tensor Hadamard API failed")) {
    return 1;
  }
  const auto tensorQuotient = divide(tensor.value(), 5.0, limits);
  if (!require(tensorQuotient.hasValue() &&
                   tensorQuotient.value().at(std::array<Index, 2>{1U, 1U}) ==
                       1.0,
               "installed tensor division API failed")) {
    return 1;
  }
  const std::array<Extent, 2> broadcastShape{2U, 2U};
  const std::array<Index, 2> broadcastAxes{0U, 1U};
  const auto broadcast =
      broadcastTo(tensor.value(), broadcastShape, broadcastAxes, limits);
  if (!require(broadcast.hasValue() &&
                   broadcast.value().at(std::array<Index, 2>{1U, 1U}) == 5.0,
               "installed tensor broadcast API failed")) {
    return 1;
  }
  const auto transformed = transformElements(
      tensor.value(),
      [](Scalar value) -> Result<Scalar> {
        return Result<Scalar>::success(value + 1.0);
      },
      limits);
  if (!require(transformed.hasValue() &&
                   transformed.value().at(std::array<Index, 2>{1U, 1U}) == 6.0,
               "installed tensor transform API failed")) {
    return 1;
  }

  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = 2.0;
  matrix.at(1U, 0U) = 3.0;
  matrix.at(1U, 1U) = 4.0;
  const auto matrixProduct = hadamard(matrix.view(), matrix.view(), limits);
  if (!require(matrixProduct.hasValue() &&
                   matrixProduct.value().at(1U, 1U) == 16.0,
               "installed matrix Hadamard API failed")) {
    return 1;
  }
  const auto matrixNorm = frobenius(matrix.view());
  if (!require(matrixNorm.hasValue() && matrixNorm.value() > 5.47 &&
                   matrixNorm.value() < 5.48,
               "installed matrix Frobenius API failed")) {
    return 1;
  }
  const auto matrixInner = frobeniusInner(matrix.view(), matrix.view());
  if (!require(matrixInner.hasValue() && matrixInner.value() == 30.0,
               "installed matrix Frobenius inner-product API failed")) {
    return 1;
  }
  const auto tensorNorm = frobenius(tensor.value());
  const auto tensorInner = frobeniusInner(tensor.value(), tensor.value());
  if (!require(tensorNorm.hasValue() && tensorInner.hasValue() &&
                   tensorInner.value() == 25.0,
               "installed tensor Frobenius APIs failed")) {
    return 1;
  }

  Vector vector(2U);
  vector.at(0U) = 5.0;
  vector.at(1U) = 6.0;
  const auto vectorProduct = hadamard(vector.view(), vector.view(), limits);
  if (!require(vectorProduct.hasValue() &&
                   vectorProduct.value().at(0U) == 25.0 &&
                   vectorProduct.value().at(1U) == 36.0,
               "installed vector Hadamard API failed")) {
    return 1;
  }

  PrintOptions printOptions;
  printOptions.maxElements = 2U;
  std::ostringstream formatted;
  print(formatted, tensor.value(), printOptions);
  if (!require(formatted.str().find("2 omitted") != std::string::npos,
               "installed bounded tensor printing failed")) {
    return 1;
  }
  const std::array<Extent, 1> scalarExtents{1U};
  const auto scalarShape = TensorShape::create(scalarExtents);
  if (!require(scalarShape.hasValue(), "scalar shape construction failed")) {
    return 1;
  }
  auto scalarTensor =
      Tensor::create(scalarShape.value(), 12.0, TensorLimits{1U});
  if (!require(scalarTensor.hasValue(), "scalar tensor allocation failed")) {
    return 1;
  }
  auto firstMoment = FirstMoment::create(scalarTensor.value());
  auto moments = MultivariateMoments::create(1U);
  if (!require(firstMoment.hasValue() && moments.hasValue(),
               "installed moment API construction failed")) {
    return 1;
  }
  if (!require(moments.value().setFirstMoment(firstMoment.value()).ok(),
               "installed moment setter failed")) {
    return 1;
  }
  formatted << vector << matrix << tensor.value() << tensorShape.value()
            << firstMoment.value() << moments.value();
  if (!require(formatted.str().find("MultivariateMoments(") !=
                   std::string::npos,
               "installed stream insertion symbols failed to link")) {
    return 1;
  }

  const auto result = multiply(matrix.view(), vector.view());
  if (!require(result.hasValue(), "matrix-vector multiplication failed")) {
    return 1;
  }
  if (!require(result.value().size() == 2U, "unexpected result size")) {
    return 1;
  }
  if (!require(result.value().at(0U) == 17.0 && result.value().at(1U) == 39.0,
               "unexpected matrix-vector result")) {
    return 1;
  }

  Matrix diagonal(2U, 2U);
  diagonal.at(0U, 0U) = 2.0;
  diagonal.at(1U, 1U) = 3.0;
  Vector diagonalInput(2U);
  diagonalInput.at(0U) = 2.0;
  diagonalInput.at(1U) = 3.0;
  const auto diagonalResult = multiply(diagonal.view(), diagonalInput.view());
  if (!require(diagonalResult.hasValue(),
               "diagonal matrix-vector multiplication failed")) {
    return 1;
  }
  if (!require(diagonalResult.value().at(0U) == 4.0 &&
                   diagonalResult.value().at(1U) == 9.0,
               "unexpected diagonal matrix-vector result")) {
    return 1;
  }

  Vector rightHandSide(2U);
  rightHandSide.at(0U) = 4.0;
  rightHandSide.at(1U) = 9.0;
  const auto factorized = vss::math::linear::factorizeLU(diagonal.view());
  if (!require(factorized.hasValue(), "LU factorization failed")) {
    return 1;
  }
  const auto solved = factorized.value().solve(rightHandSide.view());
  if (!require(solved.hasValue(), "LU solve failed")) {
    return 1;
  }
  if (!require(solved.value().solution().at(0U) == 2.0 &&
                   solved.value().solution().at(1U) == 3.0,
               "unexpected LU solution")) {
    return 1;
  }
  if (!require(solved.value().conditionReport().has_value(),
               "LU solve did not return condition diagnostics")) {
    return 1;
  }

  Matrix generalizedA(2U, 2U);
  generalizedA.at(0U, 0U) = 2.0;
  generalizedA.at(1U, 1U) = 12.0;
  Matrix generalizedB(2U, 2U);
  generalizedB.at(0U, 0U) = 1.0;
  generalizedB.at(1U, 1U) = 3.0;
  SolvePolicy generalizedPolicy{};
  generalizedPolicy.iteration.max_iterations = 64U;
  const auto generalized = linear::factorGeneralizedSymmetric(
      generalizedA.view(), generalizedB.view(), generalizedPolicy,
      Limits{.elements = 100'000U,
             .bytes = std::size_t{8U} * 1024U * 1024U,
             .evaluations = 0U});
  if (!require(generalized.hasValue(),
               "installed generalized eigen API failed")) {
    return 1;
  }
  if (!require(std::abs(generalized.value().values.at(0U) - 2.0) < 1.0e-12 &&
                   std::abs(generalized.value().values.at(1U) - 4.0) < 1.0e-12,
               "installed generalized eigen values are incorrect")) {
    return 1;
  }

  const std::array<Index, 2> indices{4U, 9U};
  const auto ownedIndices = Array<Index>::copy(indices, limits);
  if (!require(ownedIndices.hasValue() &&
                   ownedIndices.value().view().size() == indices.size() &&
                   ownedIndices.value().view()[1] == 9U,
               "installed immutable Array copy failed")) {
    return 1;
  }

  Matrix exchange(2U, 2U);
  exchange.at(0U, 1U) = 1.0;
  exchange.at(1U, 0U) = 1.0;
  SolvePolicy decompositionPolicy;
  decompositionPolicy.iteration.max_iterations = 64U;
  const auto indefiniteFactor =
      linear::factorLDLT(exchange.view(), decompositionPolicy, limits);
  if (!require(indefiniteFactor.hasValue() &&
                   indefiniteFactor.value().rank == 2U &&
                   indefiniteFactor.value().pivot_sizes.view().size() == 1U &&
                   indefiniteFactor.value().pivot_sizes.view()[0] == 2U,
               "installed symmetric-indefinite LDLT factorization failed")) {
    return 1;
  }

  Matrix singular(2U, 2U);
  const auto singularFactor = linear::factorizeLU(singular.view());
  if (!require(singularFactor.hasValue() && singularFactor.value().isSingular(),
               "zero matrix did not produce a singular LU diagnostic")) {
    return 1;
  }
  Vector nanVector(2U);
  nanVector.at(0U) = std::numeric_limits<Scalar>::quiet_NaN();
  nanVector.at(1U) = 1.0;
  if (!requireError(singularFactor.value().solve(nanVector.view()),
                    ErrorCode::NonFiniteInput,
                    "singular vector solve did not prioritize NaN input")) {
    return 1;
  }
  Matrix infiniteMatrix(2U, 1U);
  infiniteMatrix.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  infiniteMatrix.at(1U, 0U) = 1.0;
  if (!requireError(singularFactor.value().solve(infiniteMatrix.view()),
                    ErrorCode::NonFiniteInput,
                    "singular matrix solve did not prioritize infinity")) {
    return 1;
  }
  Matrix emptyColumns(2U, 0U);
  if (!requireError(singularFactor.value().solve(emptyColumns.view()),
                    ErrorCode::InvalidExtent,
                    "singular matrix solve accepted an empty RHS")) {
    return 1;
  }

  Matrix illConditionedInput(2U, 2U);
  illConditionedInput.at(0U, 0U) = 1.0;
  illConditionedInput.at(1U, 1U) = 1.0e-8;
  linear::FactorizationOptions factorizationOptions;
  factorizationOptions.conditionLimit = 2.0;
  const auto illConditionedFactor =
      linear::factorizeLU(illConditionedInput.view(), factorizationOptions);
  if (!require(
          illConditionedFactor.hasValue() &&
              illConditionedFactor.value().conditionReport().classification ==
                  linear::ConditionClassification::IllConditioned,
          "configured ill-conditioned LU factorization did not succeed")) {
    return 1;
  }
  if (!requireError(
          illConditionedFactor.value().solve(nanVector.view()),
          ErrorCode::NonFiniteInput,
          "ill-conditioned vector solve did not prioritize NaN input")) {
    return 1;
  }
  if (!requireError(
          illConditionedFactor.value().solve(infiniteMatrix.view()),
          ErrorCode::NonFiniteInput,
          "ill-conditioned matrix solve did not prioritize infinity")) {
    return 1;
  }

  Matrix rankAwareDesign(1U, 2U, 1.0);
  Matrix rankAwareObservations(1U, 1U, 4.0);
  SolvePolicy rankAwarePolicy;
  rankAwarePolicy.iteration.max_iterations = 64U;
  rankAwarePolicy.rank = RankPolicy::minimum_norm;
  const auto rankAware = linear::leastSquaresMinNorm(
      rankAwareDesign.view(), rankAwareObservations.view(), rankAwarePolicy,
      limits);
  if (!require(rankAware.hasValue() && rankAware.value().rank == 1U &&
                   rankAware.value().identified_basis.has_value() &&
                   std::abs(rankAware.value().parameters.at(0U, 0U) - 2.0) <
                       1.0e-12 &&
                   std::abs(rankAware.value().parameters.at(1U, 0U) - 2.0) <
                       1.0e-12 &&
                   rankAware.value().residual_norms.at(0U) < 1.0e-12,
               "installed rank-aware minimum-norm least-squares API failed")) {
    return 1;
  }
  const auto nullBasis =
      linear::nullSpace(rankAwareDesign.view(), rankAwarePolicy, limits);
  if (!require(nullBasis.hasValue() && nullBasis.value().rows() == 2U &&
                   nullBasis.value().columns() == 1U &&
                   std::abs(nullBasis.value().at(0U, 0U) +
                            nullBasis.value().at(1U, 0U)) < 1.0e-12 &&
                   std::abs(std::hypot(nullBasis.value().at(0U, 0U),
                                       nullBasis.value().at(1U, 0U)) -
                            1.0) < 1.0e-12,
               "installed right-null-space API failed")) {
    return 1;
  }

  Matrix fullQrInput(3U, 2U);
  fullQrInput.at(0U, 0U) = 1.0;
  fullQrInput.at(1U, 1U) = 2.0;
  fullQrInput.at(2U, 0U) = 1.0;
  fullQrInput.at(2U, 1U) = 1.0;
  SolvePolicy fullQrPolicy{};
  fullQrPolicy.iteration.max_iterations = 128U;
  const auto fullQr = linear::factorFullQR(
      fullQrInput.view(), linear::QRColumnOrder::pivoted, fullQrPolicy, limits);
  if (!require(fullQr.hasValue() && fullQr.value().orthogonal.rows() == 3U &&
                   fullQr.value().orthogonal.columns() == 3U &&
                   fullQr.value().upper.rows() == 3U &&
                   fullQr.value().upper.columns() == 2U &&
                   fullQr.value().permutation.view()[0U] == 1U &&
                   fullQr.value().rank == 2U,
               "installed full QR API failed")) {
    return 1;
  }

  Matrix jointCovariance(2U, 2U);
  jointCovariance.at(0U, 0U) = 4.0;
  jointCovariance.at(0U, 1U) = 3.0;
  jointCovariance.at(1U, 0U) = 3.0;
  jointCovariance.at(1U, 1U) = 9.0;
  const std::array<Index, 1U> kept{0U};
  const std::array<Index, 1U> conditioned{1U};
  const auto conditional = covariance::conditionalCovariance(
      jointCovariance.view(), kept, conditioned, decompositionPolicy, limits);
  if (!require(conditional.hasValue() && conditional.value().rank == 1U &&
                   std::abs(conditional.value().covariance.at(0U, 0U) - 3.0) <
                       1.0e-12,
               "installed conditional covariance API failed")) {
    return 1;
  }
  const Vector whiteningInput{std::array<Scalar, 2U>{2.0, 3.0}};
  const auto whitening =
      covariance::whitenSupported(jointCovariance.view(), whiteningInput.view(),
                                  decompositionPolicy, limits);
  const auto mahalanobis =
      whitening.hasValue()
          ? covariance::mahalanobisSquared(whitening.value(),
                                           whiteningInput.view())
          : Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                    "whitening consumer setup failed");
  if (!require(whitening.hasValue() && mahalanobis.hasValue() &&
                   std::abs(mahalanobis.value() - (4.0 / 3.0)) < 1.0e-12,
               "installed covariance whitening APIs failed")) {
    return 1;
  }

  Matrix glsDesign(2U, 1U, 1.0);
  Matrix glsObservations(2U, 1U);
  glsObservations.at(0U, 0U) = 1.0;
  glsObservations.at(1U, 0U) = 3.0;
  Matrix glsCovariance(2U, 2U);
  glsCovariance.at(0U, 0U) = 4.0;
  glsCovariance.at(0U, 1U) = 1.0;
  glsCovariance.at(1U, 0U) = 1.0;
  glsCovariance.at(1U, 1U) = 1.0;
  SolvePolicy glsPolicy = decompositionPolicy;
  glsPolicy.rank = RankPolicy::supported_subspace;
  const auto gls = linear::generalizedLinearEstimate(
      glsDesign.view(), glsObservations.view(), glsCovariance.view(), glsPolicy,
      limits);
  if (!require(gls.hasValue() && gls.value().covariance.has_value() &&
                   std::abs(gls.value().parameters.at(0U, 0U) - 3.0) <
                       1.0e-12 &&
                   std::abs(gls.value().covariance->at(0U, 0U) - 1.0) < 1.0e-12,
               "installed known-covariance GLS API failed")) {
    return 1;
  }

  Matrix reductionInput(2U, 2U);
  reductionInput.at(0U, 0U) = 3.0;
  const Limits reductionLimits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 64U};
  const auto reduced = linear::lowRankApproximation(reductionInput.view(), 1U,
                                                    glsPolicy, reductionLimits);
  if (!require(reduced.hasValue() && reduced.value().singular.at(0U) == 3.0 &&
                   reduced.value().residual_frobenius == 0.0,
               "installed low-rank approximation API failed")) {
    return 1;
  }
  Matrix sampleRows(4U, 2U);
  sampleRows.at(0U, 0U) = 2.0;
  sampleRows.at(1U, 0U) = -2.0;
  sampleRows.at(2U, 1U) = 1.0;
  sampleRows.at(3U, 1U) = -1.0;
  const std::array<Index, 1U> sampleShape{2U};
  const SampleView sampleView{
      .rows = sampleRows.view(), .observation_shape = sampleShape, .valid = {}};
  const auto principal =
      linear::pca(sampleView, 1U, true, glsPolicy, reductionLimits);
  if (!require(principal.hasValue() &&
                   std::abs(principal.value().explained_variance.at(0U) -
                            (8.0 / 3.0)) < 1.0e-12,
               "installed centered PCA API failed")) {
    return 1;
  }

  Matrix propagatedCovariance(2U, 2U);
  propagatedCovariance.at(0U, 0U) = 4.0;
  propagatedCovariance.at(1U, 1U) = 9.0;
  const PropagationResult propagated{
      .mean = Vector{std::array<Scalar, 2U>{2.0, -3.0}},
      .covariance = propagatedCovariance,
      .method = Approximation::exact,
      .diagnostics = {},
      .covariance_check = PropagationCovarianceCheck::positive_semidefinite};
  const auto uncertainty = covariance::summarizeUncertainty(
      propagated, 2.0, std::nullopt, std::nullopt, reductionLimits);
  if (!require(uncertainty.hasValue() &&
                   uncertainty.value().standard.at(0U) == 2.0 &&
                   uncertainty.value().expanded.at(1U) == 6.0,
               "installed uncertainty summary API failed")) {
    return 1;
  }

  const std::array<Index, 1U> marginalIndex{1U};
  const auto marginal = covariance::marginalCovariance(
      jointCovariance.view(), marginalIndex, glsPolicy, reductionLimits);
  const auto inverseRoot = covariance::inverseSquareRoot(
      jointCovariance.view(), glsPolicy, reductionLimits);
  if (!require(marginal.hasValue() && marginal.value().at(0U, 0U) == 9.0 &&
                   inverseRoot.hasValue() && inverseRoot.value().rank == 2U,
               "installed covariance diagnostic APIs failed")) {
    return 1;
  }

  const auto jointDeterminant = covariance::determinant(
      jointCovariance.view(), glsPolicy, reductionLimits);
  const auto jointLogDeterminant = covariance::logAbsoluteDeterminant(
      jointCovariance.view(), glsPolicy, reductionLimits);
  if (!require(jointDeterminant.hasValue() && jointLogDeterminant.hasValue() &&
                   std::abs(jointDeterminant.value() - 27.0) < 1.0e-12 &&
                   std::abs(jointLogDeterminant.value() - std::log(27.0)) <
                       1.0e-12,
               "installed determinant diagnostic APIs failed")) {
    return 1;
  }

  Matrix leftMap(1U, 1U, 2.0);
  Matrix sourceCross(1U, 1U, 0.5);
  Matrix rightMap(1U, 1U, 3.0);
  const auto transformedCross = covariance::crossCovarianceFromMaps(
      leftMap.view(), sourceCross.view(), rightMap.view(), reductionLimits);
  if (!require(transformedCross.hasValue() &&
                   transformedCross.value().at(0U, 0U) == 3.0,
               "installed cross-covariance transformation API failed")) {
    return 1;
  }

  const Vector orderSample{std::array<Scalar, 4U>{1.0, 2.0, 4.0, 8.0}};
  constexpr Limits orderLimits{64U, 4096U, 0U};
  const auto installedQuantile = statistics::quantile(
      orderSample.view(), 0.25, QuantileRule::linear, orderLimits);
  const auto installedInverseQuantile = statistics::quantile(
      orderSample.view(), 0.25, QuantileRule::inverse_empirical, orderLimits);
  const auto installedLowerQuantile = statistics::quantile(
      orderSample.view(), 0.0, QuantileRule::linear, orderLimits);
  const auto installedUpperQuantile = statistics::quantile(
      orderSample.view(), 1.0, QuantileRule::linear, orderLimits);
  const auto installedMedian =
      statistics::median(orderSample.view(), orderLimits);
  const auto installedTrimmed =
      statistics::trimmedMean(orderSample.view(), 0.25, orderLimits);
  const auto installedWinsorized =
      statistics::winsorizedMean(orderSample.view(), 0.25, orderLimits);
  const Vector nominalOrderSample{std::array<Scalar, 4U>{1.0, 2.0, 3.0, 5.0}};
  const auto installedNominalMedian =
      statistics::median(nominalOrderSample.view(), orderLimits);
  const auto installedNominalTrimmed =
      statistics::trimmedMean(nominalOrderSample.view(), 0.25, orderLimits);
  const auto installedNominalWinsorized =
      statistics::winsorizedMean(nominalOrderSample.view(), 0.25, orderLimits);
  const auto installedGeometric = statistics::geometricMean(orderSample.view());
  const auto installedHarmonic = statistics::harmonicMean(orderSample.view());
  const auto installedMad =
      statistics::medianAbsoluteDeviation(orderSample.view(), 1.0, orderLimits);
  const auto installedIqr = statistics::interquartileRange(
      orderSample.view(), QuantileRule::linear, orderLimits);
  if (!require(
          installedQuantile.hasValue() && installedQuantile.value() == 1.75 &&
              installedInverseQuantile.hasValue() &&
              installedInverseQuantile.value() == 1.0 &&
              installedLowerQuantile.hasValue() &&
              installedLowerQuantile.value() == 1.0 &&
              installedUpperQuantile.hasValue() &&
              installedUpperQuantile.value() == 8.0 &&
              installedMedian.hasValue() && installedMedian.value() == 3.0 &&
              installedTrimmed.hasValue() && installedTrimmed.value() == 3.0 &&
              installedWinsorized.hasValue() &&
              installedWinsorized.value() == 3.0 &&
              installedNominalMedian.hasValue() &&
              installedNominalMedian.value() == 2.5 &&
              installedNominalTrimmed.hasValue() &&
              installedNominalTrimmed.value() == 2.5 &&
              installedNominalWinsorized.hasValue() &&
              installedNominalWinsorized.value() == 2.5 &&
              installedGeometric.hasValue() &&
              std::abs(installedGeometric.value() - std::sqrt(8.0)) < 1.0e-12 &&
              installedHarmonic.hasValue() &&
              std::abs(installedHarmonic.value() - 32.0 / 15.0) < 1.0e-12 &&
              installedMad.hasValue() && installedMad.value() == 1.5 &&
              installedIqr.hasValue() && installedIqr.value() == 3.25,
          "installed order-statistics API failed")) {
    return 1;
  }

  const Vector geometricMeanSample{std::array<Scalar, 2U>{1.0, 4.0}};
  const Vector harmonicMeanSample{std::array<Scalar, 2U>{1.0, 3.0}};
  const auto installedGeometricMean =
      statistics::geometricMean(geometricMeanSample.view());
  const auto installedHarmonicMean =
      statistics::harmonicMean(harmonicMeanSample.view());
  if (!require(installedGeometricMean.hasValue() &&
                   installedGeometricMean.value() == 2.0 &&
                   installedHarmonicMean.hasValue() &&
                   installedHarmonicMean.value() == 1.5,
               "installed geometric or harmonic mean API failed")) {
    return 1;
  }

  Matrix summaryRows(3U, 1U);
  summaryRows.at(0U, 0U) = 1.0;
  summaryRows.at(1U, 0U) = 2.0;
  summaryRows.at(2U, 0U) = 4.0;
  const SampleView summarySample{summaryRows.view(), {}, {}};
  const std::array<Scalar, 3U> summaryProbabilities{0.25, 0.5, 0.75};
  const statistics::WeightSpec noSummaryWeights{.rule = WeightRule::unweighted,
                                                .rowWeights = std::nullopt,
                                                .componentWeights =
                                                    std::nullopt,
                                                .frequencyCounts = {}};
  const auto installedSummary = statistics::descriptiveSummary(
      summarySample, noSummaryWeights,
      {.quantile_probabilities = summaryProbabilities,
       .variance_mode = statistics::VarianceMode::empirical},
      limits);
  if (!require(installedSummary.hasValue() &&
                   installedSummary.value().sum.at(0U) == 7.0 &&
                   installedSummary.value().range.at(0U) == 3.0 &&
                   std::abs(installedSummary.value().mean.at(0U) - 7.0 / 3.0) <
                       1.0e-12 &&
                   installedSummary.value().quantiles.at(0U, 1U) == 2.0,
               "installed weighted descriptive summary API failed")) {
    return 1;
  }

  Matrix transformRows(3U, 2U);
  transformRows.at(0U, 0U) = 0.0;
  transformRows.at(0U, 1U) = 5.0;
  transformRows.at(1U, 0U) = 1.0;
  transformRows.at(1U, 1U) = 5.0;
  transformRows.at(2U, 0U) = 2.0;
  transformRows.at(2U, 1U) = 5.0;
  const std::array<Index, 1U> transformShape{2U};
  const SampleView transformSample{transformRows.view(), transformShape, {}};
  constexpr Limits transformLimits{65536U, 16U * 1024U * 1024U, 64U};
  const statistics::MomentRequest transformSourceRequest{
      .orders = 0x0aU,
      .representation = statistics::MomentRepresentation::dense,
      .selected = {},
      .contraction = std::nullopt};
  const auto transformSource = statistics::estimateCentralMoments(
      transformSample, noSummaryWeights, statistics::CenterSpec{},
      transformSourceRequest, transformLimits);
  if (!transformSource.hasValue()) {
    std::cerr << "math_consumer: source moment code="
              << static_cast<int>(transformSource.status().code()) << ", "
              << transformSource.status().message() << '\n';
  }
  if (!require(transformSource.hasValue(),
               "installed source moment estimate API failed")) {
    return 1;
  }
  const std::array<Index, 1U> selectedCoordinate{0U};
  const auto installedMarginal =
      statistics::marginalMoments(transformSource.value(), selectedCoordinate,
                                  transformSourceRequest, transformLimits);
  const std::array<Index, 4U> selectedFourthTuple{0U, 0U, 0U, 0U};
  const std::array<std::span<const Index>, 1U> selectedFourthTuples{
      std::span<const Index>{selectedFourthTuple}};
  const statistics::MomentRequest selectedFourthRequest{
      .orders = 0x08U,
      .representation = statistics::MomentRepresentation::selected,
      .selected = selectedFourthTuples,
      .contraction = std::nullopt};
  const auto installedStandardized = statistics::standardizedMoments(
      transformSource.value(), selectedFourthRequest, transformLimits);
  if (!require(
          installedMarginal.hasValue() && installedStandardized.hasValue() &&
              installedMarginal.value().observation_shape.view()[0U] == 1U &&
              std::abs(
                  installedMarginal.value().terms.view()[0U].values.at(0U) -
                  2.0 / 3.0) < 1.0e-12 &&
              std::abs(
                  installedStandardized.value().terms.view()[0U].values.at(0U) -
                  1.5) < 1.0e-12,
          "installed marginal/standardized moment APIs failed")) {
    return 1;
  }

  Matrix populationRows(3U, 1U);
  populationRows.at(0U, 0U) = -1.0;
  populationRows.at(1U, 0U) = 0.0;
  populationRows.at(2U, 0U) = 1.0;
  const Matrix& immutablePopulationRows = populationRows;
  const SampleView populationSample{immutablePopulationRows.view(), {}, {}};
  const statistics::CenterSpec estimatedCenter{CenterRule::estimated,
                                               std::nullopt};
  const statistics::PopulationPolicy populationPolicy{
      statistics::VarianceMode::empirical, statistics::MissingMode::reject,
      Estimand::observed};
  const auto populationEstimate =
      statistics::covarianceMatrix(populationSample, estimatedCenter,
                                   populationPolicy, Limits{16U, 1024U, 0U});
  if (!require(populationEstimate.hasValue() &&
                   std::abs(populationEstimate.value().value.at(0U, 0U) -
                            2.0 / 3.0) < 1.0e-12,
               "installed population covariance API failed")) {
    return 1;
  }

  Matrix weightedRows(2U, 1U);
  weightedRows.at(0U, 0U) = 0.0;
  weightedRows.at(1U, 0U) = 3.0;
  const Vector rowWeights{std::array<Scalar, 2U>{1.0, 2.0}};
  const statistics::WeightSpec empiricalWeights{
      WeightRule::empirical, rowWeights.view(), std::nullopt, {}};
  const statistics::WeightSpec fixedWeights{
      WeightRule::fixed_iid, rowWeights.view(), std::nullopt, {}};
  const SampleView weightedSample{weightedRows.view(), {}, {}};
  const statistics::MomentRequest rawMomentRequest{
      .orders = 0x03U,
      .representation = statistics::MomentRepresentation::dense,
      .selected = {},
      .contraction = std::nullopt};
  const auto installedMoments =
      statistics::estimateRawMoments(weightedSample, empiricalWeights,
                                     rawMomentRequest, Limits{128U, 4096U, 0U});
  Matrix cumulantRows(2U, 1U);
  cumulantRows.at(0U, 0U) = 3.0;
  cumulantRows.at(1U, 0U) = 5.0;
  const SampleView cumulantSample{cumulantRows.view(), {}, {}};
  const statistics::MomentRequest cumulantRequest{
      .orders = 0x0fU,
      .representation = statistics::MomentRepresentation::dense,
      .selected = {},
      .contraction = std::nullopt};
  const statistics::WeightSpec unweightedSpec{.rule = WeightRule::unweighted,
                                              .rowWeights = std::nullopt,
                                              .componentWeights = std::nullopt,
                                              .frequencyCounts = {}};
  const auto centralMoments = statistics::estimateCentralMoments(
      cumulantSample, unweightedSpec, statistics::CenterSpec{}, cumulantRequest,
      Limits{128U, 4096U, 0U});
  if (!require(centralMoments.hasValue(),
               "installed central-moment API failed")) {
    return 1;
  }
  SolvePolicy diagnosticPolicy{};
  diagnosticPolicy.iteration.max_iterations = 128U;
  const auto momentValidity = statistics::validateMoments(
      centralMoments.value(), diagnosticPolicy, Limits{4096U, 1U << 20U, 0U});
  Matrix mardiaRows(3U, 1U);
  mardiaRows.at(0U, 0U) = -1.0;
  mardiaRows.at(1U, 0U) = 0.0;
  mardiaRows.at(2U, 0U) = 2.0;
  const SampleView mardiaSample{mardiaRows.view(), {}, {}};
  const auto installedMardiaSkewness = statistics::mardiaSkewness(
      mardiaSample, diagnosticPolicy, Limits{4096U, 1U << 20U, 0U});
  const auto installedMardiaKurtosis = statistics::mardiaKurtosis(
      mardiaSample, diagnosticPolicy, Limits{4096U, 1U << 20U, 0U});
  if (!require(momentValidity.hasValue() && momentValidity.value().symmetric &&
                   momentValidity.value().second_psd &&
                   momentValidity.value().degree_two_moment_gram_psd &&
                   !momentValidity.value().sufficient_for_model &&
                   installedMardiaSkewness.hasValue() &&
                   std::abs(installedMardiaSkewness.value() - 50.0 / 343.0) <
                       1.0e-12 &&
                   installedMardiaKurtosis.hasValue() &&
                   std::abs(installedMardiaKurtosis.value() - 1.5) < 1.0e-12,
               "installed moment diagnostics API failed")) {
    return 1;
  }
  const auto installedCumulants = statistics::centralMomentsToCumulants(
      centralMoments.value(), cumulantRequest, Limits{128U, 4096U, 0U});
  if (!require(
          installedCumulants.hasValue() &&
              installedCumulants.value().terms.view().size() == 4U &&
              std::abs(
                  installedCumulants.value().terms.view()[0U].values.at(0U) -
                  4.0) < 1.0e-12 &&
              std::abs(
                  installedCumulants.value().terms.view()[3U].values.at(0U) +
                  2.0) < 1.0e-12,
          "installed cumulant conversion failed")) {
    return 1;
  }
  const auto recoveredCentralMoments = statistics::cumulantsToCentralMoments(
      installedCumulants.value(), cumulantRequest, Limits{128U, 4096U, 0U});
  if (!require(
          recoveredCentralMoments.hasValue() &&
              std::abs(
                  recoveredCentralMoments.value().terms.view()[0U].values.at(
                      0U)) < 1.0e-12 &&
              std::abs(
                  recoveredCentralMoments.value().terms.view()[3U].values.at(
                      0U) -
                  1.0) < 1.0e-12,
          "installed inverse cumulant conversion failed")) {
    return 1;
  }
  const auto weightedAverage = statistics::weightedMean(
      weightedSample, empiricalWeights, Limits{16U, 1024U, 0U});
  const auto weightedSpread = statistics::weightedEmpiricalCovariance(
      weightedSample, empiricalWeights, Limits{16U, 1024U, 0U});
  const auto unbiasedSpread = statistics::unbiasedFixedWeightedCovariance(
      weightedSample, fixedWeights, Limits{16U, 1024U, 0U});
  const auto meanSpread = statistics::weightedMeanCovariance(
      weightedSample, fixedWeights, Limits{16U, 1024U, 0U});
  Matrix thirdMomentRows(3U, 1U);
  thirdMomentRows.at(0U, 0U) = 0.0;
  thirdMomentRows.at(1U, 0U) = 0.0;
  thirdMomentRows.at(2U, 0U) = 3.0;
  const Vector thirdMomentWeights{std::array<Scalar, 3U>{1.0, 1.0, 1.0}};
  const statistics::WeightSpec fixedThirdMomentWeights{
      WeightRule::fixed_iid, thirdMomentWeights.view(), std::nullopt, {}};
  const SampleView thirdMomentSample{thirdMomentRows.view(), {}, {}};
  const statistics::MomentRequest thirdMomentRequest{
      .orders = 0x04U,
      .representation = statistics::MomentRepresentation::dense,
      .selected = {},
      .contraction = std::nullopt};
  const auto unbiasedThirdMoment = statistics::unbiasedFixedWeightedThirdMoment(
      thirdMomentSample, fixedThirdMomentWeights, thirdMomentRequest,
      Limits{64U, 4096U, 0U});
  const std::array<std::uint64_t, 2U> frequencyCounts{1U, 2U};
  const auto frequencySpread = statistics::frequencyWeightedCovariance(
      weightedSample, frequencyCounts, statistics::VarianceMode::iid_sample,
      Limits{16U, 1024U, 0U});
  Matrix knownVariances(2U, 1U);
  knownVariances.at(0U, 0U) = 1.0;
  knownVariances.at(1U, 0U) = 3.0;
  const auto precisionAverage = statistics::precisionWeightedMean(
      weightedSample, knownVariances.view(), Limits{16U, 1024U, 0U});
  Matrix heterogeneousRows(2U, 2U);
  heterogeneousRows.at(0U, 0U) = 1.0;
  heterogeneousRows.at(1U, 0U) = 3.0;
  heterogeneousRows.at(0U, 1U) = 2.0;
  heterogeneousRows.at(1U, 1U) = 6.0;
  const std::array<Index, 1U> heterogeneousShape{2U};
  const SampleView heterogeneousSample{
      heterogeneousRows.view(), heterogeneousShape, {}};
  const Scalar tinyVariance = std::numeric_limits<Scalar>::min() / 8.0;
  Matrix heterogeneousVariances(2U, 2U);
  heterogeneousVariances.at(0U, 0U) = tinyVariance;
  heterogeneousVariances.at(1U, 0U) = tinyVariance;
  heterogeneousVariances.at(0U, 1U) = 3.0;
  heterogeneousVariances.at(1U, 1U) = 1.0;
  const auto heterogeneousPrecision = statistics::precisionWeightedMean(
      heterogeneousSample, heterogeneousVariances.view(),
      Limits{16U, 1024U, 0U});
  if (!require(
          installedMoments.hasValue() &&
              installedMoments.value().terms.view().size() == 2U &&
              std::abs(installedMoments.value().terms.view()[0].values.at(0U) -
                       2.0) < 1.0e-12 &&
              std::abs(installedMoments.value().terms.view()[1].values.at(0U) -
                       6.0) < 1.0e-12 &&
              weightedAverage.hasValue() && weightedSpread.hasValue() &&
              unbiasedSpread.hasValue() && meanSpread.hasValue() &&
              unbiasedThirdMoment.hasValue() &&
              std::abs(
                  unbiasedThirdMoment.value().terms.view()[0U].values.at(0U) -
                  9.0) < 1.0e-12 &&
              frequencySpread.hasValue() && precisionAverage.hasValue() &&
              weightedAverage.value().value.at(0U) == 2.0 &&
              weightedSpread.value().value.at(0U, 0U) == 2.0 &&
              std::abs(unbiasedSpread.value().value.at(0U, 0U) - 4.5) <
                  1.0e-12 &&
              std::abs(meanSpread.value().value.at(0U, 0U) - 2.5) < 1.0e-12 &&
              std::abs(frequencySpread.value().value.at(0U, 0U) - 3.0) <
                  1.0e-12 &&
              frequencySpread.value().diagnostics.global.frequency_mass == 3U &&
              std::abs(precisionAverage.value().value.at(0U) - 0.75) <
                  1.0e-12 &&
              precisionAverage.value().error_covariance.has_value() &&
              std::abs(precisionAverage.value().error_covariance->at(0U, 0U) -
                       0.75) < 1.0e-12 &&
              heterogeneousPrecision.hasValue() &&
              std::abs(heterogeneousPrecision.value().value.at(0U) - 2.0) <
                  1.0e-12 &&
              std::abs(heterogeneousPrecision.value().value.at(1U) - 5.0) <
                  1.0e-12 &&
              heterogeneousPrecision.value().error_covariance.has_value() &&
              std::abs(
                  heterogeneousPrecision.value().error_covariance->at(0U, 0U) /
                      tinyVariance -
                  0.5) < 1.0e-12 &&
              std::abs(
                  heterogeneousPrecision.value().error_covariance->at(1U, 1U) -
                  0.75) < 1.0e-12,
          "installed weighted-population APIs failed")) {
    return 1;
  }

  Matrix averageMap(1U, 2U);
  averageMap.at(0U, 0U) = 0.5;
  averageMap.at(0U, 1U) = 0.5;
  Matrix meanJointCovariance(2U, 2U);
  meanJointCovariance.at(0U, 0U) = 0.29;
  meanJointCovariance.at(0U, 1U) = 0.25;
  meanJointCovariance.at(1U, 0U) = 0.25;
  meanJointCovariance.at(1U, 1U) = 0.29;
  const auto mappedCovariance = statistics::linearEstimateCovariance(
      averageMap.view(), meanJointCovariance.view(), Limits{64U, 4096U, 0U});
  if (!require(mappedCovariance.hasValue() &&
                   std::abs(mappedCovariance.value().at(0U, 0U) - 0.27) <
                       1.0e-12,
               "installed linear estimate covariance API failed")) {
    return 1;
  }

  Matrix covarianceInput(3U, 3U, 4.0);
  covarianceInput.at(0U, 1U) = 2.0;
  covarianceInput.at(1U, 0U) = 2.0;
  covarianceInput.at(0U, 2U) = 2.0;
  covarianceInput.at(2U, 0U) = 2.0;
  covarianceInput.at(1U, 2U) = 2.0;
  covarianceInput.at(2U, 1U) = 2.0;
  SolvePolicy correlationPolicy{};
  correlationPolicy.iteration.max_iterations = 64U;
  correlationPolicy.rank = RankPolicy::supported_subspace;
  const auto installedCorrelation = statistics::correlationMatrix(
      covarianceInput.view(), correlationPolicy, Limits{4096U, 262144U, 0U});
  const std::array<Index, 2U> selectedCoordinates{0U, 1U};
  const std::array<Index, 1U> conditionedCoordinate{2U};
  const auto installedPartial = statistics::partialCorrelation(
      covarianceInput.view(), selectedCoordinates, conditionedCoordinate,
      correlationPolicy, Limits{4096U, 262144U, 0U});
  if (!require(installedCorrelation.hasValue() && installedPartial.hasValue() &&
                   std::abs(installedCorrelation.value().covariance.at(0U, 1U) -
                            0.5) < 1.0e-12 &&
                   std::abs(installedPartial.value().covariance.at(0U, 1U) -
                            1.0 / 3.0) < 1.0e-12,
               "installed correlation APIs failed")) {
    return 1;
  }

  std::cout << "VirtualSensorMath " << version::string << " consumer passed\n";
  return 0;
}
