#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/statistics/Missing.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

bool observed(const SampleView& sample, Extent row, Extent column) {
  return sample.valid.empty() ||
         sample.valid[row * sample.rows.columns() + column] == 1U;
}

Result<Extent> validateSample(const SampleView& sample, Limits limits) {
  const Extent n = sample.rows.rows();
  const Extent d = sample.rows.columns();
  if (n == 0U || d == 0U) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "missing-data operations require nonempty dimensions");
  }
  if (n > std::numeric_limits<Extent>::max() / d) {
    return fail<Extent>(ErrorCode::SizeOverflow, "sample cell count overflows");
  }
  if (!sample.observation_shape.empty()) {
    Extent product = 1U;
    for (const Extent extent : sample.observation_shape) {
      if (extent == 0U ||
          product > std::numeric_limits<Extent>::max() / extent) {
        return fail<Extent>(ErrorCode::InvalidExtent,
                            "invalid observation shape");
      }
      product *= extent;
    }
    if (product != d) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "observation shape does not match sample columns");
    }
  }
  if (!sample.valid.empty() && sample.valid.size() != n * d) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "validity mask must contain one byte per sample cell");
  }
  for (Extent i = 0U; i < n; ++i) {
    for (Extent j = 0U; j < d; ++j) {
      if (!observed(sample, i, j)) {
        continue;
      }
      if (!std::isfinite(sample.rows.at(i, j))) {
        return fail<Extent>(ErrorCode::NonFiniteInput,
                            "an observed sample cell is non-finite");
      }
    }
  }
  for (const std::uint8_t bit : sample.valid) {
    if (bit > 1U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "validity mask entries must be zero or one");
    }
  }
  static_cast<void>(limits);
  return Result<Extent>::success(d);
}

bool completeRow(const SampleView& sample, Extent row) {
  for (Extent j = 0U; j < sample.rows.columns(); ++j) {
    if (!observed(sample, row, j)) {
      return false;
    }
  }
  return true;
}

Result<EstimatorDiagnostics>
diagnostics(Extent input, Extent validCount, WeightRule weight,
            CenterRule center, std::string algorithm, std::string premises,
            std::vector<TupleSupport> tuples = {},
            std::vector<ErrorCode> warnings = {}, Limits limits = {}) {
  const auto n = static_cast<Scalar>(validCount);
  const Support support{
      .input = input,
      .valid = validCount,
      .positive = validCount,
      .frequency_mass = std::nullopt,
      .weight_mass = n,
      .normalized_weight_powers =
          validCount == 0U
              ? std::array<std::optional<Scalar>, 3>{}
              : std::array<std::optional<Scalar>, 3>{1.0, 1.0 / n,
                                                     1.0 / (n * n)},
      .effective_count =
          validCount == 0U ? std::nullopt : std::optional<Scalar>(n),
      .degrees_of_freedom = std::nullopt,
      .squared_weight_mass = std::nullopt};
  auto tupleArray = Array<TupleSupport>::copy(tuples, limits);
  auto warningArray = Array<ErrorCode>::copy(warnings, limits);
  if (!tupleArray || !warningArray) {
    return fail<EstimatorDiagnostics>(ErrorCode::ResourceLimit,
                                      "diagnostics allocation failed");
  }
  return Result<EstimatorDiagnostics>::success(
      EstimatorDiagnostics{.estimand = Estimand::observed,
                           .weight = weight,
                           .center = center,
                           .approximation = Approximation::exact,
                           .global = support,
                           .coordinate_support = {},
                           .tuple_support = std::move(tupleArray).value(),
                           .reciprocal_condition = std::nullopt,
                           .numerical_error = std::nullopt,
                           .sampling_error = std::nullopt,
                           .rank = std::nullopt,
                           .algorithm = std::move(algorithm),
                           .premises = std::move(premises),
                           .warnings = std::move(warningArray).value()});
}

Result<Matrix> covarianceOnRows(const SampleView& sample,
                                std::span<const Extent> rows, Limits limits) {
  const Extent d = sample.rows.columns();
  if (d > std::numeric_limits<Extent>::max() / d) {
    return fail<Matrix>(ErrorCode::SizeOverflow,
                        "covariance result extent overflows");
  }
  if (d * d > limits.elements || d * d > limits.bytes / sizeof(Scalar)) {
    return fail<Matrix>(ErrorCode::ResourceLimit,
                        "covariance result exceeds configured limits");
  }
  if (rows.empty()) {
    return fail<Matrix>(ErrorCode::InsufficientSamples,
                        "no complete rows remain for covariance");
  }
  std::vector<long double> mean(d, 0.0L);
  for (const Extent row : rows) {
    for (Extent j = 0U; j < d; ++j) {
      mean[j] += static_cast<long double>(sample.rows.at(row, j)) /
                 static_cast<long double>(rows.size());
    }
  }
  Matrix result(d, d, 0.0);
  for (Extent i = 0U; i < d; ++i) {
    for (Extent j = i; j < d; ++j) {
      long double sum = 0.0L;
      for (const Extent row : rows) {
        const long double left = sample.rows.at(row, i) - mean[i];
        const long double right = sample.rows.at(row, j) - mean[j];
        sum += left * right / static_cast<long double>(rows.size());
      }
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max()) {
        return fail<Matrix>(ErrorCode::NonFiniteResult,
                            "covariance is not representable");
      }
      result.at(i, j) = static_cast<Scalar>(sum);
      result.at(j, i) = static_cast<Scalar>(sum);
    }
  }
  return Result<Matrix>::success(std::move(result));
}

struct SpectralSystem final {
  Matrix inverse;
  Extent rank{};
  Scalar log_pseudodeterminant{};
  Scalar cutoff{};
};

Result<SpectralSystem> factorPsd(ConstMatrixView matrix,
                                 const SolvePolicy& policy) {
  if (matrix.rows() != matrix.columns()) {
    return fail<SpectralSystem>(ErrorCode::DimensionMismatch,
                                "Gaussian covariance block must be square");
  }
  if (matrix.empty()) {
    return Result<SpectralSystem>::success(
        SpectralSystem{.inverse = Matrix{},
                       .rank = 0U,
                       .log_pseudodeterminant = 0.0,
                       .cutoff = 0.0});
  }
  linear::SymmetricEigenOptions options;
  options.rankTolerance = {.absolute = policy.rank_abs,
                           .relative = policy.rank_rel};
  if (policy.iteration.max_iterations != 0U) {
    options.maximumSweeps = policy.iteration.max_iterations;
  }
  auto eigen = linear::decomposeSymmetricEigen(matrix, options);
  if (!eigen) {
    return fail<SpectralSystem>(eigen.status().code(),
                                std::string(eigen.status().message()));
  }
  if (eigen.value().definiteness() ==
      linear::PositiveDefiniteness::NotPositiveSemidefinite) {
    return fail<SpectralSystem>(
        ErrorCode::NotPositiveSemidefinite,
        "Gaussian covariance block is not positive semidefinite");
  }
  const Extent d = matrix.rows();
  Matrix inverse(d, d, 0.0);
  long double logDet = 0.0L;
  Extent rank = 0U;
  const auto values = eigen.value().eigenvalues();
  const auto vectors = eigen.value().eigenvectors();
  for (Extent component = 0U; component < d; ++component) {
    const Scalar value = values.at(component);
    if (value <= eigen.value().rankThreshold()) {
      continue;
    }
    ++rank;
    logDet += std::log(static_cast<long double>(value));
    for (Extent i = 0U; i < d; ++i) {
      for (Extent j = 0U; j < d; ++j) {
        inverse.at(i, j) +=
            vectors.at(i, component) * vectors.at(j, component) / value;
      }
    }
  }
  if (!std::isfinite(logDet) ||
      std::abs(logDet) > std::numeric_limits<Scalar>::max()) {
    return fail<SpectralSystem>(
        ErrorCode::NonFiniteResult,
        "Gaussian covariance log-pseudodeterminant is not finite");
  }
  return Result<SpectralSystem>::success(
      SpectralSystem{.inverse = std::move(inverse),
                     .rank = rank,
                     .log_pseudodeterminant = static_cast<Scalar>(logDet),
                     .cutoff = eigen.value().rankThreshold()});
}

struct ObservedBlock final {
  std::vector<Extent> indices;
  std::vector<long double> residual;
  SpectralSystem system;
};

Result<ObservedBlock> observedBlock(const SampleView& sample, Extent row,
                                    ConstVectorView mean,
                                    ConstMatrixView covariance,
                                    const SolvePolicy& policy,
                                    std::uint64_t& evaluations, Limits limits) {
  if (evaluations >= limits.evaluations) {
    return fail<ObservedBlock>(ErrorCode::ResourceLimit,
                               "Gaussian EM evaluation budget exhausted");
  }
  ++evaluations;
  const Extent d = sample.rows.columns();
  std::vector<Extent> indices;
  for (Extent j = 0U; j < d; ++j) {
    if (observed(sample, row, j)) {
      indices.push_back(j);
    }
  }
  const Extent k = indices.size();
  Matrix block(k, k, 0.0);
  std::vector<long double> residual(k, 0.0L);
  for (Extent i = 0U; i < k; ++i) {
    residual[i] = static_cast<long double>(sample.rows.at(row, indices[i])) -
                  mean.at(indices[i]);
    for (Extent j = 0U; j < k; ++j) {
      block.at(i, j) = covariance.at(indices[i], indices[j]);
    }
  }
  auto system = factorPsd(block.view(), policy);
  if (!system) {
    return fail<ObservedBlock>(system.status().code(),
                               std::string(system.status().message()));
  }
  if (system.value().rank < k) {
    long double residualSquared = 0.0L;
    long double projectedSquared = 0.0L;
    for (const long double value : residual) {
      residualSquared += value * value;
    }
    const long double tolerance =
        policy.iteration.abs +
        policy.iteration.rel * std::max(1.0L, std::sqrt(residualSquared));
    // A singular covariance block is accepted only when the residual lies in
    // its retained eigenspace; compute the orthogonal projector directly.
    auto eigen = linear::decomposeSymmetricEigen(
        block.view(), linear::SymmetricEigenOptions{
                          .rankTolerance = {.absolute = policy.rank_abs,
                                            .relative = policy.rank_rel}});
    if (!eigen) {
      return fail<ObservedBlock>(eigen.status().code(),
                                 std::string(eigen.status().message()));
    }
    for (Extent component = 0U; component < k; ++component) {
      if (eigen.value().eigenvalues().at(component) <=
          eigen.value().rankThreshold()) {
        continue;
      }
      long double projection = 0.0L;
      for (Extent i = 0U; i < k; ++i) {
        projection +=
            eigen.value().eigenvectors().at(i, component) * residual[i];
      }
      projectedSquared += projection * projection;
    }
    const long double outsideSquared =
        std::max(0.0L, residualSquared - projectedSquared);
    if (std::sqrt(outsideSquared) > tolerance) {
      return fail<ObservedBlock>(
          ErrorCode::InvalidDomain,
          "observed residual violates singular Gaussian support");
    }
  }
  return Result<ObservedBlock>::success(
      ObservedBlock{.indices = std::move(indices),
                    .residual = std::move(residual),
                    .system = std::move(system).value()});
}

Result<Scalar>
observedLogLikelihood(const SampleView& sample, ConstVectorView mean,
                      ConstMatrixView covariance, const SolvePolicy& policy,
                      std::uint64_t& evaluations, Limits limits) {
  long double total = 0.0L;
  for (Extent row = 0U; row < sample.rows.rows(); ++row) {
    bool any = false;
    for (Extent j = 0U; j < sample.rows.columns(); ++j) {
      any = any || observed(sample, row, j);
    }
    if (!any) {
      continue;
    }
    auto block = observedBlock(sample, row, mean, covariance, policy,
                               evaluations, limits);
    if (!block) {
      return fail<Scalar>(block.status().code(),
                          std::string(block.status().message()));
    }
    long double quadratic = 0.0L;
    for (Extent i = 0U; i < block.value().indices.size(); ++i) {
      for (Extent j = 0U; j < block.value().indices.size(); ++j) {
        quadratic += block.value().residual[i] *
                     block.value().system.inverse.at(i, j) *
                     block.value().residual[j];
      }
    }
    total -= 0.5L * (block.value().system.rank *
                         std::log(2.0L * std::numbers::pi_v<long double>) +
                     block.value().system.log_pseudodeterminant + quadratic);
  }
  if (!std::isfinite(total) ||
      std::abs(total) > std::numeric_limits<Scalar>::max()) {
    return fail<Scalar>(
        ErrorCode::NonFiniteResult,
        "observed Gaussian log likelihood is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(total));
}

struct Score final {
  long double mean_norm{};
  long double covariance_norm{};
};

Result<Score> observedScore(const SampleView& sample, ConstVectorView mean,
                            ConstMatrixView covariance,
                            const SolvePolicy& policy,
                            std::uint64_t& evaluations, Limits limits) {
  const Extent d = sample.rows.columns();
  std::vector<long double> meanScore(d, 0.0L);
  std::vector<long double> covarianceScore(d * d, 0.0L);
  Extent count = 0U;
  for (Extent row = 0U; row < sample.rows.rows(); ++row) {
    bool any = false;
    for (Extent j = 0U; j < d; ++j) {
      any = any || observed(sample, row, j);
    }
    if (!any) {
      continue;
    }
    auto block = observedBlock(sample, row, mean, covariance, policy,
                               evaluations, limits);
    if (!block) {
      return fail<Score>(block.status().code(),
                         std::string(block.status().message()));
    }
    ++count;
    const Extent k = block.value().indices.size();
    std::vector<long double> whitened(k, 0.0L);
    for (Extent i = 0U; i < k; ++i) {
      for (Extent j = 0U; j < k; ++j) {
        whitened[i] +=
            block.value().system.inverse.at(i, j) * block.value().residual[j];
      }
      meanScore[block.value().indices[i]] += whitened[i];
    }
    for (Extent i = 0U; i < k; ++i) {
      for (Extent j = 0U; j < k; ++j) {
        covarianceScore[block.value().indices[i] * d +
                        block.value().indices[j]] +=
            0.5L *
            (whitened[i] * whitened[j] - block.value().system.inverse.at(i, j));
      }
    }
  }
  if (count == 0U) {
    return fail<Score>(ErrorCode::InsufficientSamples,
                       "no observed rows contribute to the Gaussian score");
  }
  long double meanNorm = 0.0L;
  long double covarianceNorm = 0.0L;
  long double meanScale = 1.0L;
  long double covarianceScale = 1.0L;
  for (Extent i = 0U; i < d; ++i) {
    meanNorm = std::max(meanNorm, std::abs(meanScore[i]) / count);
    meanScale =
        std::max(meanScale, std::abs(static_cast<long double>(mean.at(i))));
    for (Extent j = 0U; j < d; ++j) {
      covarianceNorm = std::max(covarianceNorm,
                                std::abs(covarianceScore[i * d + j]) / count);
      covarianceScale =
          std::max(covarianceScale,
                   std::abs(static_cast<long double>(covariance.at(i, j))));
    }
  }
  return Result<Score>::success(
      Score{.mean_norm = meanNorm * meanScale,
            .covariance_norm = covarianceNorm * covarianceScale});
}

bool positiveFinitePolicy(const SolvePolicy& policy) {
  return std::isfinite(policy.iteration.abs) &&
         std::isfinite(policy.iteration.rel) && policy.iteration.abs >= 0.0 &&
         policy.iteration.rel >= 0.0 && std::isfinite(policy.rank_abs) &&
         policy.rank_abs >= 0.0 && std::isfinite(policy.rank_rel) &&
         policy.rank_rel >= 0.0;
}

Status validateEmInitialization(const MissingPolicy& policy, Extent d,
                                Limits limits) {
  if (!policy.initial_mean || !policy.initial_covariance) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "Gaussian EM requires both an initial mean and covariance");
  }
  if (policy.initial_mean->size() != d ||
      policy.initial_covariance->rows() != d ||
      policy.initial_covariance->columns() != d) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "Gaussian EM initialization dimensions differ from the sample");
  }
  if (!positiveFinitePolicy(policy.policy) ||
      policy.policy.iteration.max_iterations == 0U ||
      policy.policy.iteration.max_evaluations == 0U) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "Gaussian EM requires finite tolerances and positive work caps");
  }
  for (Extent i = 0U; i < d; ++i) {
    if (!std::isfinite(policy.initial_mean->at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "Gaussian EM initial mean is non-finite");
    }
    for (Extent j = 0U; j < d; ++j) {
      const Scalar value = policy.initial_covariance->at(i, j);
      if (!std::isfinite(value)) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "Gaussian EM initial covariance is non-finite");
      }
      const Scalar reverse = policy.initial_covariance->at(j, i);
      const Scalar tolerance =
          policy.policy.iteration.abs +
          policy.policy.iteration.rel *
              std::max({1.0, std::abs(value), std::abs(reverse)});
      if (std::abs(value - reverse) > tolerance) {
        return Status::failure(
            ErrorCode::NotSymmetric,
            "Gaussian EM initial covariance is not symmetric");
      }
    }
  }
  if (d > std::numeric_limits<Extent>::max() / d || d * d > limits.elements ||
      d * d > limits.bytes / (sizeof(long double) * 16U)) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "Gaussian EM workspace exceeds configured limits");
  }
  const auto system =
      factorPsd(policy.initial_covariance->view(), policy.policy);
  if (!system) {
    return Status::failure(system.status().code(),
                           std::string(system.status().message()));
  }
  if (system.value().rank == 0U) {
    return Status::failure(
        ErrorCode::RankDeficient,
        "Gaussian EM initial covariance has zero support rank");
  }
  return Status::success();
}

Result<Extent> validateIdentifiability(const SampleView& sample,
                                       Index minimumSupport) {
  const Extent n = sample.rows.rows();
  const Extent d = sample.rows.columns();
  std::vector<Extent> support(d * d, 0U);
  for (Extent row = 0U; row < n; ++row) {
    bool any = false;
    for (Extent i = 0U; i < d; ++i) {
      any = any || observed(sample, row, i);
      for (Extent j = i; j < d; ++j) {
        if (observed(sample, row, i) && observed(sample, row, j)) {
          ++support[i * d + j];
          if (i != j) {
            ++support[j * d + i];
          }
        }
      }
    }
    static_cast<void>(any);
  }
  for (Extent i = 0U; i < d; ++i) {
    if (support[i * d + i] == 0U) {
      return fail<Extent>(ErrorCode::InsufficientSamples,
                          "an entire channel is unobserved");
    }
  }
  for (Extent i = 0U; i < d; ++i) {
    for (Extent j = i; j < d; ++j) {
      const Extent count = support[i * d + j];
      if (count == 0U && i != j) {
        return fail<Extent>(ErrorCode::RankDeficient,
                            "a covariance pair is never jointly observed");
      }
    }
  }
  for (Extent i = 0U; i < d; ++i) {
    for (Extent j = i; j < d; ++j) {
      const Extent count = support[i * d + j];
      if (count < minimumSupport) {
        return fail<Extent>(
            ErrorCode::InsufficientSamples,
            "an observed covariance pair is below minimum_support");
      }
    }
  }
  return Result<Extent>::success(n);
}

struct EmParameters final {
  Vector mean;
  Matrix covariance;
  Extent effective_rows{};
};

Result<EmParameters> emUpdate(const SampleView& sample, ConstVectorView mean,
                              ConstMatrixView covariance,
                              const SolvePolicy& policy,
                              std::uint64_t& evaluations, Limits limits) {
  const Extent d = sample.rows.columns();
  std::vector<long double> sumMean(d, 0.0L);
  std::vector<long double> sumProducts(d * d, 0.0L);
  Extent effectiveRows = 0U;
  for (Extent row = 0U; row < sample.rows.rows(); ++row) {
    bool any = false;
    for (Extent j = 0U; j < d; ++j) {
      any = any || observed(sample, row, j);
    }
    if (!any) {
      continue;
    }
    auto block = observedBlock(sample, row, mean, covariance, policy,
                               evaluations, limits);
    if (!block) {
      return fail<EmParameters>(block.status().code(),
                                std::string(block.status().message()));
    }
    ++effectiveRows;
    std::vector<bool> present(d, false);
    std::vector<long double> conditionalMean(d, 0.0L);
    for (Extent j = 0U; j < d; ++j) {
      conditionalMean[j] = mean.at(j);
    }
    for (const Extent j : block.value().indices) {
      present[j] = true;
      conditionalMean[j] = sample.rows.at(row, j);
    }
    for (Extent j = 0U; j < d; ++j) {
      if (present[j]) {
        continue;
      }
      long double shift = 0.0L;
      for (Extent oi = 0U; oi < block.value().indices.size(); ++oi) {
        long double coefficient = 0.0L;
        for (Extent oj = 0U; oj < block.value().indices.size(); ++oj) {
          coefficient += covariance.at(j, block.value().indices[oj]) *
                         block.value().system.inverse.at(oj, oi);
        }
        shift += coefficient * block.value().residual[oi];
      }
      conditionalMean[j] += shift;
    }
    Matrix conditionalCovariance(d, d, 0.0);
    for (Extent i = 0U; i < d; ++i) {
      if (present[i]) {
        continue;
      }
      for (Extent j = 0U; j < d; ++j) {
        if (present[j]) {
          continue;
        }
        long double reduction = 0.0L;
        for (Extent oi = 0U; oi < block.value().indices.size(); ++oi) {
          for (Extent oj = 0U; oj < block.value().indices.size(); ++oj) {
            reduction += covariance.at(i, block.value().indices[oi]) *
                         block.value().system.inverse.at(oi, oj) *
                         covariance.at(block.value().indices[oj], j);
          }
        }
        conditionalCovariance.at(i, j) = static_cast<Scalar>(
            static_cast<long double>(covariance.at(i, j)) - reduction);
      }
    }
    for (Extent i = 0U; i < d; ++i) {
      sumMean[i] += conditionalMean[i];
      for (Extent j = 0U; j < d; ++j) {
        sumProducts[i * d + j] += conditionalMean[i] * conditionalMean[j] +
                                  conditionalCovariance.at(i, j);
      }
    }
  }
  if (effectiveRows == 0U) {
    return fail<EmParameters>(
        ErrorCode::InsufficientSamples,
        "no row contributes observed Gaussian likelihood");
  }
  Vector nextMean(d, 0.0);
  Matrix nextCovariance(d, d, 0.0);
  const auto count = static_cast<long double>(effectiveRows);
  for (Extent i = 0U; i < d; ++i) {
    nextMean.at(i) = static_cast<Scalar>(sumMean[i] / count);
  }
  for (Extent i = 0U; i < d; ++i) {
    for (Extent j = 0U; j < d; ++j) {
      const long double value =
          sumProducts[i * d + j] / count -
          static_cast<long double>(nextMean.at(i)) * nextMean.at(j);
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<EmParameters>(
            ErrorCode::NonFiniteResult,
            "Gaussian EM covariance update is not representable");
      }
      nextCovariance.at(i, j) = static_cast<Scalar>(value);
    }
  }
  return Result<EmParameters>::success(
      EmParameters{.mean = std::move(nextMean),
                   .covariance = std::move(nextCovariance),
                   .effective_rows = effectiveRows});
}

} // namespace

Result<SelectedSample> sampleWithMissingPolicy(const SampleView& x,
                                               const MissingPolicy& policy,
                                               Limits limits) {
  auto dimension = validateSample(x, limits);
  if (!dimension) {
    return fail<SelectedSample>(dimension.status().code(),
                                std::string(dimension.status().message()));
  }
  if (!policy.initial_mean && !policy.initial_covariance) {
    // No initialization is the normal selection path.
  } else {
    return fail<SelectedSample>(
        ErrorCode::InvalidDomain,
        "sample selection does not accept EM initialization");
  }
  if (policy.mode == MissingMode::gaussian_em) {
    return fail<SelectedSample>(ErrorCode::InvalidDomain,
                                "Gaussian EM returns covariance sufficient "
                                "statistics, not imputed rows");
  }
  if (policy.mode != MissingMode::reject &&
      policy.mode != MissingMode::complete_row &&
      policy.mode != MissingMode::pairwise) {
    return fail<SelectedSample>(ErrorCode::InvalidDomain,
                                "unsupported missing-data selection mode");
  }
  const Extent n = x.rows.rows();
  const Extent d = dimension.value();
  Extent outputRows = 0U;
  if (policy.mode == MissingMode::reject) {
    for (Extent i = 0U; i < n; ++i) {
      if (!completeRow(x, i)) {
        return fail<SelectedSample>(
            ErrorCode::InvalidDomain,
            "reject mode requires complete observations");
      }
      ++outputRows;
    }
  } else if (policy.mode == MissingMode::complete_row) {
    for (Extent i = 0U; i < n; ++i) {
      if (completeRow(x, i)) {
        ++outputRows;
      }
    }
  } else {
    outputRows = n;
  }
  if (outputRows > std::numeric_limits<Extent>::max() / d) {
    return fail<SelectedSample>(ErrorCode::SizeOverflow,
                                "selected sample extent overflows");
  }
  const Extent cells = outputRows * d;
  const auto bytesFor = [](Extent count, Extent size) {
    return count > std::numeric_limits<Extent>::max() / size
               ? std::optional<Extent>{}
               : std::optional<Extent>{count * size};
  };
  const auto dataBytes = bytesFor(cells, sizeof(Scalar));
  const auto maskBytes = bytesFor(cells, sizeof(std::uint8_t));
  const auto rowsBytes = bytesFor(outputRows, sizeof(Index));
  const auto shapeBytes = bytesFor(x.observation_shape.size(), sizeof(Index));
  if (!dataBytes || !maskBytes || !rowsBytes || !shapeBytes ||
      *dataBytes > limits.bytes || *maskBytes > limits.bytes - *dataBytes ||
      *rowsBytes > limits.bytes - *dataBytes - *maskBytes ||
      *shapeBytes > limits.bytes - *dataBytes - *maskBytes - *rowsBytes ||
      cells > limits.elements) {
    return fail<SelectedSample>(ErrorCode::ResourceLimit,
                                "selected sample exceeds configured limits");
  }
  std::vector<Extent> selected;
  selected.reserve(outputRows);
  for (Extent i = 0U; i < n; ++i) {
    if (policy.mode == MissingMode::pairwise || completeRow(x, i)) {
      selected.push_back(i);
    }
  }
  Matrix output(outputRows, d, 0.0);
  std::vector<std::uint8_t> outputMask(cells, 0U);
  for (Extent outRow = 0U; outRow < outputRows; ++outRow) {
    const Extent inRow = selected[outRow];
    for (Extent j = 0U; j < d; ++j) {
      if (observed(x, inRow, j)) {
        output.at(outRow, j) = x.rows.at(inRow, j);
        outputMask[outRow * d + j] = 1U;
      }
    }
  }
  std::vector<Index> shape(x.observation_shape.begin(),
                           x.observation_shape.end());
  auto shapeArray = Array<Index>::copy(shape, limits);
  auto rowsArray = Array<Index>::copy(selected, limits);
  auto maskArray = Array<std::uint8_t>::copy(outputMask, limits);
  if (!shapeArray || !rowsArray || !maskArray) {
    return fail<SelectedSample>(
        ErrorCode::ResourceLimit,
        "selected sample provenance exceeds configured limits");
  }
  return Result<SelectedSample>::success(
      SelectedSample{.data = std::move(output),
                     .shape = std::move(shapeArray).value(),
                     .source_rows = std::move(rowsArray).value(),
                     .mask = std::move(maskArray).value()});
}

Result<Estimate<Matrix>> covarianceWithMissing(const SampleView& x,
                                               const MissingPolicy& policy,
                                               Limits limits) {
  auto dimension = validateSample(x, limits);
  if (!dimension) {
    return fail<Estimate<Matrix>>(dimension.status().code(),
                                  std::string(dimension.status().message()));
  }
  const Extent n = x.rows.rows();
  const Extent d = dimension.value();
  if (policy.minimum_support == 0U) {
    return fail<Estimate<Matrix>>(ErrorCode::InvalidDomain,
                                  "minimum support must be positive");
  }
  if (policy.mode == MissingMode::gaussian_em) {
    if (policy.initial_mean.has_value() !=
        policy.initial_covariance.has_value()) {
      return fail<Estimate<Matrix>>(
          ErrorCode::InvalidDomain,
          "Gaussian EM requires both initialization values");
    }
    if (!policy.initial_mean.has_value() ||
        !policy.initial_covariance.has_value()) {
      return fail<Estimate<Matrix>>(
          ErrorCode::InvalidDomain,
          "Gaussian EM requires both initialization values");
    }
    if (d > std::numeric_limits<Extent>::max() / d || d * d > limits.elements ||
        d * d > limits.bytes / (sizeof(long double) * 16U)) {
      return fail<Estimate<Matrix>>(
          ErrorCode::ResourceLimit,
          "Gaussian EM workspace exceeds configured limits");
    }
    auto initialization = validateEmInitialization(policy, d, limits);
    if (!initialization.ok()) {
      return fail<Estimate<Matrix>>(initialization.code(),
                                    std::string(initialization.message()));
    }
    auto identified = validateIdentifiability(x, policy.minimum_support);
    if (!identified) {
      return fail<Estimate<Matrix>>(identified.status().code(),
                                    std::string(identified.status().message()));
    }
    const std::uint64_t evaluationCap = std::min<std::uint64_t>(
        policy.policy.iteration.max_evaluations, limits.evaluations);
    if (evaluationCap == 0U) {
      return fail<Estimate<Matrix>>(ErrorCode::NoConvergence,
                                    "Gaussian EM evaluation budget is zero");
    }
    Limits workLimits = limits;
    workLimits.evaluations = evaluationCap;
    Vector mean = policy.initial_mean.value();
    Matrix covariance = policy.initial_covariance.value();
    std::uint64_t evaluations = 0U;
    auto oldLikelihood =
        observedLogLikelihood(x, mean.view(), covariance.view(), policy.policy,
                              evaluations, workLimits);
    if (!oldLikelihood) {
      const ErrorCode code =
          oldLikelihood.status().code() == ErrorCode::ResourceLimit
              ? ErrorCode::NoConvergence
              : oldLikelihood.status().code();
      return fail<Estimate<Matrix>>(
          code, std::string(oldLikelihood.status().message()));
    }
    Scalar likelihood = oldLikelihood.value();
    auto currentSystem = factorPsd(covariance.view(), policy.policy);
    if (!currentSystem) {
      return fail<Estimate<Matrix>>(
          currentSystem.status().code(),
          std::string(currentSystem.status().message()));
    }
    bool converged = false;
    Extent effectiveRows = 0U;
    for (Index iteration = 0U;
         iteration < policy.policy.iteration.max_iterations; ++iteration) {
      auto candidate = emUpdate(x, mean.view(), covariance.view(),
                                policy.policy, evaluations, workLimits);
      if (!candidate) {
        const ErrorCode code =
            candidate.status().code() == ErrorCode::ResourceLimit
                ? ErrorCode::NoConvergence
                : candidate.status().code();
        return fail<Estimate<Matrix>>(
            code, std::string(candidate.status().message()));
      }
      auto candidateSystem =
          factorPsd(candidate.value().covariance.view(), policy.policy);
      if (!candidateSystem) {
        return fail<Estimate<Matrix>>(
            candidateSystem.status().code(),
            std::string(candidateSystem.status().message()));
      }
      if (candidateSystem.value().rank != currentSystem.value().rank) {
        return fail<Estimate<Matrix>>(
            ErrorCode::RankDeficient,
            "Gaussian EM update changed the retained support rank");
      }
      auto newLikelihood = observedLogLikelihood(
          x, candidate.value().mean.view(), candidate.value().covariance.view(),
          policy.policy, evaluations, workLimits);
      if (!newLikelihood) {
        const ErrorCode code =
            newLikelihood.status().code() == ErrorCode::ResourceLimit
                ? ErrorCode::NoConvergence
                : newLikelihood.status().code();
        return fail<Estimate<Matrix>>(
            code, std::string(newLikelihood.status().message()));
      }
      const long double objectiveTolerance =
          static_cast<long double>(policy.policy.iteration.abs) +
          static_cast<long double>(policy.policy.iteration.rel) *
              std::max(1.0L, std::abs(static_cast<long double>(likelihood)));
      const long double objectiveChange =
          static_cast<long double>(newLikelihood.value()) - likelihood;
      if (objectiveChange < -objectiveTolerance) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NoConvergence,
            "Gaussian EM observed likelihood decreased beyond tolerance");
      }
      bool parametersConverged = true;
      for (Extent i = 0U; i < d; ++i) {
        const long double newMean = candidate.value().mean.at(i);
        const long double oldMean = mean.at(i);
        const long double meanTolerance =
            policy.policy.iteration.abs +
            policy.policy.iteration.rel *
                std::max({1.0L, std::abs(newMean), std::abs(oldMean)});
        parametersConverged =
            parametersConverged && std::abs(newMean - oldMean) <= meanTolerance;
        for (Extent j = 0U; j < d; ++j) {
          const long double newValue = candidate.value().covariance.at(i, j);
          const long double oldValue = covariance.at(i, j);
          const long double tolerance =
              policy.policy.iteration.abs +
              policy.policy.iteration.rel *
                  std::max({1.0L, std::abs(newValue), std::abs(oldValue)});
          parametersConverged =
              parametersConverged && std::abs(newValue - oldValue) <= tolerance;
        }
      }
      auto score = observedScore(x, candidate.value().mean.view(),
                                 candidate.value().covariance.view(),
                                 policy.policy, evaluations, workLimits);
      if (!score) {
        const ErrorCode code = score.status().code() == ErrorCode::ResourceLimit
                                   ? ErrorCode::NoConvergence
                                   : score.status().code();
        return fail<Estimate<Matrix>>(code,
                                      std::string(score.status().message()));
      }
      const long double scoreTolerance =
          static_cast<long double>(policy.policy.iteration.abs) +
          static_cast<long double>(policy.policy.iteration.rel);
      const bool objectiveConverged =
          std::abs(objectiveChange) <= objectiveTolerance;
      const bool stationary = score.value().mean_norm <= scoreTolerance &&
                              score.value().covariance_norm <= scoreTolerance;
      mean = std::move(candidate.value().mean);
      covariance = std::move(candidate.value().covariance);
      likelihood = newLikelihood.value();
      currentSystem = std::move(candidateSystem);
      effectiveRows = candidate.value().effective_rows;
      if (objectiveConverged && parametersConverged && stationary) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      return fail<Estimate<Matrix>>(ErrorCode::NoConvergence,
                                    "Gaussian EM did not satisfy likelihood, "
                                    "parameter, and score convergence");
    }
    auto meta = diagnostics(
        n, effectiveRows, WeightRule::unweighted, CenterRule::estimated,
        "observed-data Gaussian EM",
        "stationary candidate for the unrestricted Gaussian likelihood; local "
        "convergence is not a global-optimum guarantee",
        {}, {}, limits);
    if (!meta) {
      return fail<Estimate<Matrix>>(meta.status().code(),
                                    std::string(meta.status().message()));
    }
    return Result<Estimate<Matrix>>::success(
        Estimate<Matrix>{.value = std::move(covariance),
                         .diagnostics = std::move(meta).value(),
                         .error_covariance = std::nullopt});
  }
  if (policy.initial_mean || policy.initial_covariance) {
    return fail<Estimate<Matrix>>(
        ErrorCode::InvalidDomain,
        "initialization is valid only for Gaussian EM");
  }
  if (policy.mode == MissingMode::reject ||
      policy.mode == MissingMode::complete_row) {
    if (n > limits.bytes / sizeof(Extent)) {
      return fail<Estimate<Matrix>>(
          ErrorCode::ResourceLimit,
          "complete-row support exceeds configured limits");
    }
    std::vector<Extent> rows;
    for (Extent i = 0U; i < n; ++i) {
      if (completeRow(x, i)) {
        rows.push_back(i);
      } else if (policy.mode == MissingMode::reject) {
        return fail<Estimate<Matrix>>(
            ErrorCode::InvalidDomain,
            "reject mode requires complete observations");
      }
    }
    if (rows.size() < policy.minimum_support) {
      return fail<Estimate<Matrix>>(
          ErrorCode::InsufficientSamples,
          "complete-row support is below minimum_support");
    }
    auto values = covarianceOnRows(x, rows, limits);
    if (!values) {
      return fail<Estimate<Matrix>>(values.status().code(),
                                    std::string(values.status().message()));
    }
    auto meta =
        diagnostics(n, rows.size(), WeightRule::unweighted,
                    CenterRule::estimated, "complete-row empirical covariance",
                    "empirical covariance with common complete-row support", {},
                    {}, limits);
    if (!meta) {
      return fail<Estimate<Matrix>>(meta.status().code(),
                                    std::string(meta.status().message()));
    }
    return Result<Estimate<Matrix>>::success(
        Estimate<Matrix>{.value = std::move(values).value(),
                         .diagnostics = std::move(meta).value(),
                         .error_covariance = std::nullopt});
  }
  if (policy.mode != MissingMode::pairwise) {
    return fail<Estimate<Matrix>>(ErrorCode::InvalidDomain,
                                  "unsupported missing-data covariance mode");
  }
  if (d > std::numeric_limits<Extent>::max() / d || d * d > limits.elements ||
      d * d > limits.bytes / (sizeof(Scalar) + sizeof(TupleSupport)) ||
      n > limits.bytes / sizeof(Extent)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::ResourceLimit,
        "pairwise covariance exceeds configured limits");
  }
  std::vector<Extent> channelSupport(d, 0U);
  Extent rowsWithObservations = 0U;
  for (Extent i = 0U; i < n; ++i) {
    bool any = false;
    for (Extent j = 0U; j < d; ++j) {
      const bool valid = observed(x, i, j);
      channelSupport[j] += valid ? 1U : 0U;
      any = any || valid;
    }
    if (any) {
      ++rowsWithObservations;
    }
  }
  if (std::any_of(channelSupport.begin(), channelSupport.end(),
                  [&](Extent count) { return count == 0U; })) {
    return fail<Estimate<Matrix>>(ErrorCode::InsufficientSamples,
                                  "an entire channel is unobserved");
  }
  Matrix output(d, d, 0.0);
  std::vector<TupleSupport> tupleSupport;
  tupleSupport.reserve(d * d);
  for (Extent j = 0U; j < d; ++j) {
    for (Extent k = j; k < d; ++k) {
      std::vector<Extent> rows;
      long double meanJ = 0.0L;
      long double meanK = 0.0L;
      for (Extent i = 0U; i < n; ++i) {
        if (observed(x, i, j) && observed(x, i, k)) {
          rows.push_back(i);
          meanJ += static_cast<long double>(x.rows.at(i, j));
          meanK += static_cast<long double>(x.rows.at(i, k));
        }
      }
      if (rows.size() < policy.minimum_support) {
        return fail<Estimate<Matrix>>(
            ErrorCode::InsufficientSamples,
            "a covariance pair is below minimum_support");
      }
      meanJ /= static_cast<long double>(rows.size());
      meanK /= static_cast<long double>(rows.size());
      long double sum = 0.0L;
      for (const Extent row : rows) {
        sum += (static_cast<long double>(x.rows.at(row, j)) - meanJ) *
               (static_cast<long double>(x.rows.at(row, k)) - meanK) /
               static_cast<long double>(rows.size());
      }
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max()) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "pairwise covariance is not representable");
      }
      output.at(j, k) = static_cast<Scalar>(sum);
      output.at(k, j) = static_cast<Scalar>(sum);
      const std::array<Index, 2U> coordinates{j, k};
      auto coordinateArray = Array<Index>::copy(coordinates, limits);
      if (!coordinateArray) {
        return fail<Estimate<Matrix>>(
            coordinateArray.status().code(),
            std::string(coordinateArray.status().message()));
      }
      const auto count = static_cast<Scalar>(rows.size());
      tupleSupport.push_back(TupleSupport{
          .coordinates = std::move(coordinateArray).value(),
          .support =
              Support{.input = n,
                      .valid = rows.size(),
                      .positive = rows.size(),
                      .frequency_mass = std::nullopt,
                      .weight_mass = count,
                      .normalized_weight_powers = {1.0, 1.0 / count,
                                                   1.0 / (count * count)},
                      .effective_count = count,
                      .degrees_of_freedom = std::nullopt,
                      .squared_weight_mass = std::nullopt}});
    }
  }
  std::vector<ErrorCode> warnings;
  auto eigen = linear::decomposeSymmetricEigen(output.view());
  if (!eigen) {
    return fail<Estimate<Matrix>>(eigen.status().code(),
                                  std::string(eigen.status().message()));
  }
  if (eigen.value().definiteness() ==
      linear::PositiveDefiniteness::NotPositiveSemidefinite) {
    warnings.push_back(ErrorCode::NotPositiveSemidefinite);
  }
  auto meta =
      diagnostics(n, rowsWithObservations, WeightRule::unweighted,
                  CenterRule::estimated, "pairwise empirical covariance",
                  "pair-specific support and centering; may be indefinite",
                  std::move(tupleSupport), std::move(warnings), limits);
  if (!meta) {
    return fail<Estimate<Matrix>>(meta.status().code(),
                                  std::string(meta.status().message()));
  }
  return Result<Estimate<Matrix>>::success(
      Estimate<Matrix>{.value = std::move(output),
                       .diagnostics = std::move(meta).value(),
                       .error_covariance = std::nullopt});
}

} // namespace vss::math::statistics
