#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/Special.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Inference.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {
namespace {

Result<Scalar> varianceEndpoint(Scalar sampleVariance, Scalar degreesOfFreedom,
                                Scalar chiSquareQuantile) {
  int varianceExponent = 0;
  int degreesExponent = 0;
  int quantileExponent = 0;
  const Scalar varianceMantissa = std::frexp(sampleVariance, &varianceExponent);
  const Scalar degreesMantissa = std::frexp(degreesOfFreedom, &degreesExponent);
  const Scalar quantileMantissa =
      std::frexp(chiSquareQuantile, &quantileExponent);
  const Scalar mantissa =
      (varianceMantissa * degreesMantissa) / quantileMantissa;
  int mantissaExponent = 0;
  const Scalar normalizedMantissa = std::frexp(mantissa, &mantissaExponent);
  const int exponent =
      varianceExponent + degreesExponent - quantileExponent + mantissaExponent;
  const Scalar endpoint = std::scalbn(normalizedMantissa, exponent);
  if (!std::isfinite(endpoint)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "variance confidence interval endpoint is not representable");
  }
  return Result<Scalar>::success(endpoint);
}

template <class T> Result<T> propagateFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

ProbabilityPolicy inferenceProbabilityPolicy() {
  return ProbabilityPolicy{.iteration = {.abs = 1.0e-13,
                                         .rel = 1.0e-13,
                                         .max_iterations = 256U,
                                         .max_evaluations = 1024U,
                                         .bracket = std::nullopt},
                           .max_draws = 0U,
                           .abs_log_error = 1.0e-13};
}

} // namespace

Result<Interval> varianceConfidenceInterval(Scalar sampleVariance, Index n,
                                            Scalar confidence,
                                            NormalIID premise) {
  if (!std::isfinite(sampleVariance) || !std::isfinite(confidence)) {
    return Result<Interval>::failure(
        ErrorCode::NonFiniteInput,
        "variance confidence interval input is non-finite");
  }
  if (!premise.acknowledged) {
    return Result<Interval>::failure(
        ErrorCode::InvalidDomain,
        "variance confidence interval requires acknowledged NormalIID premise");
  }
  if (n < 2U) {
    return Result<Interval>::failure(
        ErrorCode::InsufficientSamples,
        "variance confidence interval requires at least two observations");
  }
  if (sampleVariance < 0.0 || confidence <= 0.0 || confidence >= 1.0) {
    return Result<Interval>::failure(
        ErrorCode::InvalidDomain,
        "variance must be non-negative and confidence must be in (0,1)");
  }
  if (sampleVariance == 0.0) {
    return Result<Interval>::success(Interval{.lower = 0.0, .upper = 0.0});
  }

  const Scalar alpha = 1.0 - confidence;
  const Scalar lowerProbability = alpha * 0.5;
  const Scalar upperProbability = 1.0 - lowerProbability;
  const auto degreesOfFreedom = static_cast<Scalar>(n - 1U);
  const auto law = probability::chiSquare(degreesOfFreedom);
  if (!law.hasValue()) {
    return propagateFailure<Interval>(law.status());
  }
  const ProbabilityPolicy accuracy{.iteration = {.abs = 1.0e-13,
                                                 .rel = 1.0e-13,
                                                 .max_iterations = 256U,
                                                 .max_evaluations = 1024U,
                                                 .bracket = std::nullopt},
                                   .max_draws = 0U,
                                   .abs_log_error = 1.0e-13};
  const auto lowerQuantile =
      probability::quantile(law.value(), lowerProbability, accuracy);
  if (!lowerQuantile.hasValue()) {
    return propagateFailure<Interval>(lowerQuantile.status());
  }
  const auto upperQuantile =
      probability::quantile(law.value(), upperProbability, accuracy);
  if (!upperQuantile.hasValue()) {
    return propagateFailure<Interval>(upperQuantile.status());
  }
  if (!(lowerQuantile.value() > 0.0) ||
      !(upperQuantile.value() > lowerQuantile.value())) {
    return Result<Interval>::failure(
        ErrorCode::NoConvergence,
        "chi-square quantiles do not certify ordered positive endpoints");
  }
  const auto lower =
      varianceEndpoint(sampleVariance, degreesOfFreedom, upperQuantile.value());
  if (!lower.hasValue()) {
    return propagateFailure<Interval>(lower.status());
  }
  const auto upper =
      varianceEndpoint(sampleVariance, degreesOfFreedom, lowerQuantile.value());
  if (!upper.hasValue()) {
    return propagateFailure<Interval>(upper.status());
  }
  if (lower.value() > upper.value()) {
    return Result<Interval>::failure(
        ErrorCode::NoConvergence,
        "variance confidence interval endpoints are not ordered");
  }
  return Result<Interval>::success(
      Interval{.lower = lower.value(), .upper = upper.value()});
}

Result<TestResult> hotellingT2(const SampleView& sample, ConstVectorView center,
                               NormalIID premise, SolvePolicy policy,
                               Limits limits) {
  const Index n = sample.rows.rows();
  const Index d = sample.rows.columns();
  if (!premise.acknowledged) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "Hotelling T2 requires acknowledged NormalIID premise");
  }
  if (n == 0U || d == 0U) {
    return Result<TestResult>::failure(ErrorCode::InsufficientSamples,
                                       "Hotelling T2 requires nonempty data");
  }
  if (n <= d || center.size() != d) {
    return Result<TestResult>::failure(
        n <= d ? ErrorCode::InsufficientSamples : ErrorCode::DimensionMismatch,
        "Hotelling T2 requires n>d and a target matching data dimension");
  }
  if (sample.observation_shape.empty() && d != 1U) {
    return Result<TestResult>::failure(
        ErrorCode::DimensionMismatch,
        "scalar observation shape requires one flattened coordinate");
  }
  if (!sample.valid.empty()) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "Hotelling T2 requires complete observations");
  }
  if (d > limits.elements || d > limits.bytes / sizeof(Scalar)) {
    return Result<TestResult>::failure(ErrorCode::ResourceLimit,
                                       "Hotelling T2 workspace exceeds limits");
  }

  const auto covariance = covarianceMatrix(
      sample, CenterSpec{.rule = CenterRule::estimated, .center = std::nullopt},
      PopulationPolicy{.normalization = VarianceMode::iid_sample,
                       .missingness = MissingMode::reject,
                       .estimand = Estimand::observed},
      limits);
  if (!covariance.hasValue()) {
    return propagateFailure<TestResult>(covariance.status());
  }
  linear::SymmetricFactorizationOptions factorOptions{};
  factorOptions.factorization.rankTolerance = linear::RankTolerance{
      .absolute = policy.rank_abs, .relative = policy.rank_rel};
  factorOptions.factorization.conditionLimit =
      policy.reciprocal_condition_floor > 0.0
          ? 1.0 / policy.reciprocal_condition_floor
          : std::numeric_limits<Scalar>::max();
  const auto factor =
      linear::factorizeCholesky(covariance.value().value.view(), factorOptions);
  if (!factor.hasValue()) {
    return propagateFailure<TestResult>(factor.status());
  }

  Vector difference(d);
  for (Index j = 0U; j < d; ++j) {
    long double mean = 0.0L;
    for (Index i = 0U; i < n; ++i) {
      const Scalar value = sample.rows.at(i, j);
      if (!std::isfinite(value)) {
        return Result<TestResult>::failure(
            ErrorCode::NonFiniteInput,
            "Hotelling T2 observations must be finite");
      }
      mean += static_cast<long double>(value) / static_cast<long double>(n);
    }
    const long double delta = mean - static_cast<long double>(center.at(j));
    if (!std::isfinite(center.at(j)) ||
        std::abs(delta) > std::numeric_limits<Scalar>::max()) {
      return Result<TestResult>::failure(
          std::isfinite(center.at(j)) ? ErrorCode::NonFiniteResult
                                      : ErrorCode::NonFiniteInput,
          "Hotelling T2 mean difference is not representable");
    }
    difference.at(j) = static_cast<Scalar>(delta);
  }
  const auto solved = factor.value().solve(difference.view());
  if (!solved.hasValue()) {
    return propagateFailure<TestResult>(solved.status());
  }
  long double quadratic = 0.0L;
  for (Index j = 0U; j < d; ++j) {
    quadratic += static_cast<long double>(difference.at(j)) *
                 solved.value().solution().at(j);
  }
  const long double t2 = static_cast<long double>(n) * quadratic;
  if (!std::isfinite(t2) || t2 < 0.0L ||
      t2 > std::numeric_limits<Scalar>::max()) {
    return Result<TestResult>::failure(
        ErrorCode::NonFiniteResult,
        "Hotelling T2 statistic is not representable");
  }
  const auto statistic = static_cast<Scalar>(t2);
  Scalar pValue = 1.0;
  if (statistic > 0.0) {
    const Scalar ratio = statistic / static_cast<Scalar>(n - 1U);
    const Scalar z = std::isinf(ratio) ? 0.0 : 1.0 / (1.0 + ratio);
    const auto tail = probability::incompleteBeta(
        static_cast<Scalar>(n - d) * 0.5, static_cast<Scalar>(d) * 0.5, z,
        inferenceProbabilityPolicy());
    if (!tail.hasValue()) {
      return propagateFailure<TestResult>(tail.status());
    }
    pValue = std::exp(tail.value().log_cdf);
  }
  return Result<TestResult>::success(TestResult{
      .statistic = statistic,
      .p_value = pValue,
      .effective_df = std::nullopt,
      .sampling_error = std::nullopt,
      .calibration_method = "Hotelling T2 exact Gaussian-IID F reference"});
}

Result<TestResult> ljungBox(ConstVectorView residuals, Index lag,
                            Index fittedDegrees, Limits limits) {
  const Index n = residuals.size();
  if (n == 0U) {
    return Result<TestResult>::failure(ErrorCode::InsufficientSamples,
                                       "Ljung-Box requires residuals");
  }
  if (lag == 0U || lag >= n || fittedDegrees >= lag) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "Ljung-Box requires 1<=lag<n and fittedDegrees<lag");
  }
  if (limits.elements == 0U || limits.bytes < sizeof(Scalar)) {
    return Result<TestResult>::failure(ErrorCode::ResourceLimit,
                                       "Ljung-Box result exceeds limits");
  }
  Scalar scale = 0.0;
  for (Index i = 0U; i < n; ++i) {
    if (!std::isfinite(residuals.at(i))) {
      return Result<TestResult>::failure(ErrorCode::NonFiniteInput,
                                         "Ljung-Box residual is non-finite");
    }
    scale = std::max(scale, std::abs(residuals.at(i)));
  }
  if (scale == 0.0) {
    return Result<TestResult>::failure(ErrorCode::InvalidDomain,
                                       "Ljung-Box residual variance is zero");
  }
  long double mean = 0.0L;
  for (Index i = 0U; i < n; ++i) {
    mean += static_cast<long double>(residuals.at(i) / scale) /
            static_cast<long double>(n);
  }
  long double denominator = 0.0L;
  for (Index i = 0U; i < n; ++i) {
    const long double centered = (residuals.at(i) / scale) - mean;
    denominator += centered * centered;
  }
  if (!(denominator > 0.0L)) {
    return Result<TestResult>::failure(ErrorCode::InvalidDomain,
                                       "Ljung-Box residual variance is zero");
  }
  long double sum = 0.0L;
  for (Index k = 1U; k <= lag; ++k) {
    long double numerator = 0.0L;
    for (Index i = k; i < n; ++i) {
      numerator += (residuals.at(i) / scale - mean) *
                   (residuals.at(i - k) / scale - mean);
    }
    const long double rho = numerator / denominator;
    sum += rho * rho / static_cast<long double>(n - k);
  }
  const long double q = static_cast<long double>(n) * (n + 2.0L) * sum;
  if (!std::isfinite(q) || q > std::numeric_limits<Scalar>::max()) {
    return Result<TestResult>::failure(
        ErrorCode::NonFiniteResult, "Ljung-Box statistic is not representable");
  }
  return Result<TestResult>::success(
      TestResult{.statistic = static_cast<Scalar>(q),
                 .p_value = std::nullopt,
                 .effective_df = static_cast<Scalar>(lag - fittedDegrees),
                 .sampling_error = std::nullopt,
                 .calibration_method = "uncalibrated Ljung-Box statistic"});
}

Result<TestResult> ljungBox(ConstVectorView residuals, Index lag,
                            Index fittedDegrees, LjungBoxReference reference,
                            Limits limits) {
  if ((reference == LjungBoxReference::white_noise && fittedDegrees != 0U) ||
      (reference == LjungBoxReference::fitted_arma && fittedDegrees == 0U)) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "Ljung-Box reference tag conflicts with fitted degrees");
  }
  auto base = ljungBox(residuals, lag, fittedDegrees, limits);
  if (!base.hasValue()) {
    return base;
  }
  const auto& baseValue = base.value();
  if (!baseValue.effective_df) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "Ljung-Box result lacks effective degrees of freedom");
  }
  const Scalar degrees = baseValue.effective_df.value();
  const auto law = probability::chiSquare(degrees);
  if (!law.hasValue()) {
    return propagateFailure<TestResult>(law.status());
  }
  const auto tail = probability::evaluate(law.value(), baseValue.statistic,
                                          inferenceProbabilityPolicy());
  if (!tail.hasValue()) {
    return propagateFailure<TestResult>(tail.status());
  }
  TestResult result = baseValue;
  result.p_value = tail.value().survival;
  result.calibration_method =
      reference == LjungBoxReference::white_noise
          ? "asymptotic chi-square white-noise reference"
          : "asymptotic chi-square fitted-residual reference";
  return Result<TestResult>::success(std::move(result));
}

Result<Estimate<Matrix>>
covarianceEstimatorUncertainty(ConstMatrixView sampleCovariance, Index n,
                               const CovarianceUncertaintySpec& specification,
                               Limits limits) {
  if (!specification.premise.acknowledged) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::InvalidDomain,
        "covariance estimator uncertainty requires acknowledged NormalIID");
  }
  if (n < 2U) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::InvalidDomain,
        "covariance estimator uncertainty requires N>=2");
  }
  const Index d = sampleCovariance.rows();
  if (d == 0U) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::InvalidExtent, "sample covariance must be nonempty");
  }
  if (sampleCovariance.columns() != d) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::DimensionMismatch, "sample covariance must be square");
  }
  for (Index i = 0U; i < d; ++i) {
    for (Index j = 0U; j < d; ++j) {
      if (!std::isfinite(sampleCovariance.at(i, j))) {
        return Result<Estimate<Matrix>>::failure(
            ErrorCode::NonFiniteInput,
            "sample covariance contains non-finite values");
      }
    }
  }
  const auto eigen = linear::decomposeSymmetricEigen(sampleCovariance);
  if (!eigen.hasValue()) {
    return propagateFailure<Estimate<Matrix>>(eigen.status());
  }
  if (eigen.value().definiteness() ==
      linear::PositiveDefiniteness::NotPositiveSemidefinite) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::NotPositiveSemidefinite,
        "sample covariance is not positive semidefinite");
  }
  Index entryCount = specification.selected.size();
  if (entryCount == 0U) {
    if (d > std::numeric_limits<Index>::max() / d) {
      return Result<Estimate<Matrix>>::failure(
          ErrorCode::SizeOverflow, "covariance entry count overflows");
    }
    entryCount = d * d;
  }
  if (entryCount > std::numeric_limits<Index>::max() / entryCount) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::SizeOverflow, "covariance uncertainty extent overflows");
  }
  const Index outputElements = entryCount * entryCount;
  if (outputElements > limits.elements ||
      outputElements > limits.bytes / sizeof(Scalar)) {
    return Result<Estimate<Matrix>>::failure(
        ErrorCode::ResourceLimit,
        "covariance uncertainty output exceeds limits");
  }
  const auto entryAt = [&](Index index) {
    return specification.selected.empty()
               ? CovarianceEntry{.row = index / d, .column = index % d}
               : specification.selected[index];
  };
  for (Index p = 0U; p < entryCount; ++p) {
    const CovarianceEntry entry = entryAt(p);
    if (entry.row >= d || entry.column >= d) {
      return Result<Estimate<Matrix>>::failure(
          ErrorCode::IndexOutOfRange,
          "selected covariance entry is outside the matrix");
    }
  }
  const auto divisor = static_cast<long double>(n - 1U);
  Matrix uncertainty(entryCount, entryCount);
  for (Index a = 0U; a < entryCount; ++a) {
    const CovarianceEntry left = entryAt(a);
    for (Index b = 0U; b < entryCount; ++b) {
      const CovarianceEntry right = entryAt(b);
      const long double first =
          static_cast<long double>(sampleCovariance.at(left.row, right.row)) *
          sampleCovariance.at(left.column, right.column);
      const long double second = static_cast<long double>(sampleCovariance.at(
                                     left.row, right.column)) *
                                 sampleCovariance.at(left.column, right.row);
      const long double value = (first + second) / divisor;
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return Result<Estimate<Matrix>>::failure(
            ErrorCode::NonFiniteResult,
            "covariance estimator uncertainty is not representable");
      }
      uncertainty.at(a, b) = static_cast<Scalar>(value);
    }
  }
  const Support support{.input = n,
                        .valid = n,
                        .positive = n,
                        .frequency_mass = std::nullopt,
                        .weight_mass = static_cast<Scalar>(n),
                        .normalized_weight_powers = {},
                        .effective_count = static_cast<Scalar>(n),
                        .degrees_of_freedom = static_cast<Scalar>(n - 1U),
                        .squared_weight_mass = static_cast<Scalar>(n)};
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::estimate,
      .weight = WeightRule::fixed_iid,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = support,
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = specification.plug_in
                       ? "Gaussian-IID plug-in covariance of covariance"
                       : "Gaussian-IID model covariance of covariance",
      .premises =
          specification.plug_in
              ? "NormalIID; unknown mean estimated; supplied covariance "
                "interpreted as estimated S; plug-in, not finite-sample "
                "unbiased"
              : "NormalIID; unknown mean estimated; supplied covariance "
                "interpreted as true Sigma",
      .warnings = {}};
  return Result<Estimate<Matrix>>::success(
      Estimate<Matrix>{.value = std::move(uncertainty),
                       .diagnostics = std::move(diagnostics),
                       .error_covariance = std::nullopt});
}

Result<BootstrapResult>
bootstrapCovarianceUncertainty(const SampleView& sample,
                               const BootstrapPolicy& policy,
                               probability::Rng& rng, Limits limits) {
  if (sample.rows.columns() == 0U || sample.rows.rows() == 0U) {
    return Result<BootstrapResult>::failure(
        ErrorCode::InsufficientSamples,
        "covariance bootstrap requires nonempty observations");
  }
  if (!sample.valid.empty()) {
    return Result<BootstrapResult>::failure(
        ErrorCode::InvalidDomain,
        "covariance bootstrap requires complete observation rows");
  }
  const Index d = sample.rows.columns();
  if (d > std::numeric_limits<Index>::max() / d) {
    return Result<BootstrapResult>::failure(
        ErrorCode::SizeOverflow,
        "covariance bootstrap output extent overflows");
  }
  const Index covarianceElements = d * d;
  if (covarianceElements > limits.elements ||
      covarianceElements > limits.bytes / sizeof(Scalar)) {
    return Result<BootstrapResult>::failure(
        ErrorCode::ResourceLimit,
        "covariance bootstrap statistic exceeds limits");
  }
  const Statistic covarianceStatistic = [limits](const SampleView& rows) {
    const auto estimate = covarianceMatrix(
        rows, CenterSpec{.rule = CenterRule::estimated, .center = std::nullopt},
        PopulationPolicy{.normalization = VarianceMode::iid_sample,
                         .missingness = MissingMode::reject,
                         .estimand = Estimand::observed},
        limits);
    if (!estimate.hasValue()) {
      return propagateFailure<Vector>(estimate.status());
    }
    Vector flattened(estimate.value().value.size());
    const Matrix& covariance = estimate.value().value;
    for (Index i = 0U; i < covariance.rows(); ++i) {
      for (Index j = 0U; j < covariance.columns(); ++j) {
        flattened.at((i * covariance.columns()) + j) = covariance.at(i, j);
      }
    }
    return Result<Vector>::success(std::move(flattened));
  };
  return bootstrapEstimate(sample, covarianceStatistic, policy, rng, limits);
}

} // namespace vss::math::statistics
