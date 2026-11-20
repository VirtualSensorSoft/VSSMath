#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

Result<Extent> validateSample(const SampleView& sample) {
  const auto rows = sample.rows.rows();
  const auto columns = sample.rows.columns();
  if (columns == 0U || sample.rows.empty()) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "population coordinates and rows must be nonempty");
  }
  if (sample.observation_shape.empty()) {
    if (columns != 1U) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "scalar sample must have exactly one column");
    }
  } else {
    Extent product = 1U;
    for (const Extent extent : sample.observation_shape) {
      if (extent == 0U) {
        return fail<Extent>(ErrorCode::InvalidExtent,
                            "observation shape extents must be positive");
      }
      if (product > std::numeric_limits<Extent>::max() / extent) {
        return fail<Extent>(ErrorCode::SizeOverflow,
                            "observation shape product overflows");
      }
      product *= extent;
    }
    if (product != columns) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "observation shape does not match sample columns");
    }
  }
  if (rows > std::numeric_limits<Extent>::max() / columns) {
    return fail<Extent>(ErrorCode::SizeOverflow, "sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != rows * columns) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "validity mask must contain one byte per sample cell");
  }
  for (const std::uint8_t byte : sample.valid) {
    if (byte > 1U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "validity mask bytes must be zero or one");
    }
  }
  return Result<Extent>::success(columns);
}

bool rowValid(const SampleView& sample, Index row) {
  if (sample.valid.empty()) {
    return true;
  }
  const auto columns = sample.rows.columns();
  for (Index column = 0U; column < columns; ++column) {
    if (sample.valid[row * columns + column] == 0U) {
      return false;
    }
  }
  return true;
}

bool commonRowValid(const SampleView& left, const SampleView* right,
                    Index row) {
  return rowValid(left, row) && (right == nullptr || rowValid(*right, row));
}

Status validatePolicy(PopulationPolicy policy, const CenterSpec& center,
                      Extent dimension) {
  if (policy.estimand != Estimand::observed &&
      policy.estimand != Estimand::measurement_error &&
      policy.estimand != Estimand::estimate &&
      policy.estimand != Estimand::calibration &&
      policy.estimand != Estimand::process &&
      policy.estimand != Estimand::fitted) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "population estimand is not supported");
  }
  if (policy.missingness != MissingMode::reject &&
      policy.missingness != MissingMode::complete_row) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "population covariance supports reject or complete-row missingness");
  }
  if (center.rule == CenterRule::estimated) {
    if (center.center.has_value()) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "estimated center must not provide a center vector");
    }
  } else if (center.rule == CenterRule::known) {
    if (!center.center.has_value() || center.center->size() != dimension) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "known center must match the sample dimension");
    }
    for (Index i = 0U; i < dimension; ++i) {
      if (!std::isfinite(center.center->at(i))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "known center contains a non-finite value");
      }
    }
  } else {
    return Status::failure(ErrorCode::InvalidDomain,
                           "center rule is not supported");
  }

  if (policy.normalization == VarianceMode::iid_sample &&
      center.rule != CenterRule::estimated) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "IID sample normalization requires an estimated center");
  }
  if (policy.normalization == VarianceMode::known_center &&
      center.rule != CenterRule::known) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "known-center normalization requires a supplied center");
  }
  if (policy.normalization != VarianceMode::empirical &&
      policy.normalization != VarianceMode::iid_sample &&
      policy.normalization != VarianceMode::known_center) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "variance normalization is not supported");
  }
  return Status::success();
}

Result<Index> retainedRows(const SampleView& left, const SampleView* right,
                           PopulationPolicy policy) {
  if (left.rows.rows() == 0U ||
      (right != nullptr && right->rows.rows() != left.rows.rows())) {
    return fail<Index>(right != nullptr ? ErrorCode::DimensionMismatch
                                        : ErrorCode::InsufficientSamples,
                       "population rows are empty or not paired");
  }
  Index count = 0U;
  for (Index row = 0U; row < left.rows.rows(); ++row) {
    if (!commonRowValid(left, right, row)) {
      if (policy.missingness == MissingMode::reject) {
        return fail<Index>(
            ErrorCode::InvalidDomain,
            "invalid cells are rejected by the missing-data policy");
      }
      continue;
    }
    for (Index column = 0U; column < left.rows.columns(); ++column) {
      if (!std::isfinite(left.rows.at(row, column))) {
        return fail<Index>(ErrorCode::NonFiniteInput,
                           "left sample contains a non-finite retained value");
      }
    }
    if (right != nullptr) {
      for (Index column = 0U; column < right->rows.columns(); ++column) {
        if (!std::isfinite(right->rows.at(row, column))) {
          return fail<Index>(
              ErrorCode::NonFiniteInput,
              "right sample contains a non-finite retained value");
        }
      }
    }
    ++count;
  }
  if (count == 0U) {
    return fail<Index>(ErrorCode::InsufficientSamples,
                       "no complete rows remain for population estimation");
  }
  if (policy.normalization == VarianceMode::iid_sample && count < 2U) {
    return fail<Index>(ErrorCode::InsufficientSamples,
                       "IID sample covariance requires at least two rows");
  }
  return Result<Index>::success(count);
}

Result<Vector> estimatedCenter(const SampleView& sample,
                               const SampleView* paired, Index count) {
  Vector center(sample.rows.columns());
  for (Index column = 0U; column < sample.rows.columns(); ++column) {
    Scalar scale = 0.0;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (!commonRowValid(sample, paired, row)) {
        continue;
      }
      const Scalar value = sample.rows.at(row, column);
      if (!std::isfinite(value)) {
        return fail<Vector>(ErrorCode::NonFiniteInput,
                            "sample contains a non-finite retained value");
      }
      scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) {
      center.at(column) = 0.0;
      continue;
    }
    detail::CompensatedSum sum;
    for (Index row = 0U; row < sample.rows.rows(); ++row) {
      if (commonRowValid(sample, paired, row)) {
        sum.add(sample.rows.at(row, column) / scale);
      }
    }
    const Scalar result = scale * (sum.total() / static_cast<Scalar>(count));
    if (!std::isfinite(result)) {
      return fail<Vector>(ErrorCode::NonFiniteResult,
                          "estimated population center is non-finite");
    }
    center.at(column) = result;
  }
  return Result<Vector>::success(std::move(center));
}

Scalar centeredValue(const SampleView& sample, Index row, Index column,
                     const CenterSpec& center, const Vector* estimated) {
  return sample.rows.at(row, column) - (center.rule == CenterRule::known
                                            ? center.center->at(column)
                                            : estimated->at(column));
}

Scalar scaledProduct(Scalar leftScale, Scalar rightScale, Scalar sum,
                     Scalar divisor) {
  if (leftScale == 0.0 || rightScale == 0.0 || sum == 0.0) {
    return 0.0;
  }
  int leftExponent = 0;
  int rightExponent = 0;
  const Scalar leftFraction = std::frexp(leftScale, &leftExponent);
  const Scalar rightFraction = std::frexp(rightScale, &rightExponent);
  const Scalar fraction = leftFraction * rightFraction * (sum / divisor);
  return std::scalbn(fraction, leftExponent + rightExponent);
}

Result<Matrix> covarianceValues(const SampleView& left, const SampleView* right,
                                const CenterSpec& leftCenter,
                                const CenterSpec* rightCenter,
                                const Vector* leftEstimated,
                                const Vector* rightEstimated,
                                PopulationPolicy policy, Index count) {
  const Extent rows = left.rows.columns();
  const Extent columns = right == nullptr ? rows : right->rows.columns();
  const Scalar divisor = static_cast<Scalar>(
      policy.normalization == VarianceMode::iid_sample ? count - 1U : count);
  Matrix output(rows, columns);
  for (Index i = 0U; i < rows; ++i) {
    const Index begin = right == nullptr ? i : 0U;
    for (Index j = begin; j < columns; ++j) {
      Scalar leftScale = 0.0;
      Scalar rightScale = 0.0;
      for (Index row = 0U; row < left.rows.rows(); ++row) {
        if (!commonRowValid(left, right, row)) {
          continue;
        }
        const Scalar x = centeredValue(left, row, i, leftCenter, leftEstimated);
        const Scalar y =
            right == nullptr
                ? centeredValue(left, row, j, leftCenter, leftEstimated)
                : centeredValue(*right, row, j, *rightCenter, rightEstimated);
        if (!std::isfinite(x) || !std::isfinite(y)) {
          return fail<Matrix>(ErrorCode::NonFiniteResult,
                              "centered population value is non-finite");
        }
        leftScale = std::max(leftScale, std::abs(x));
        rightScale = std::max(rightScale, std::abs(y));
      }
      detail::CompensatedSum sum;
      if (leftScale != 0.0 && rightScale != 0.0) {
        for (Index row = 0U; row < left.rows.rows(); ++row) {
          if (!commonRowValid(left, right, row)) {
            continue;
          }
          const Scalar x =
              centeredValue(left, row, i, leftCenter, leftEstimated);
          const Scalar y =
              right == nullptr
                  ? centeredValue(left, row, j, leftCenter, leftEstimated)
                  : centeredValue(*right, row, j, *rightCenter, rightEstimated);
          sum.add((x / leftScale) * (y / rightScale));
        }
      }
      const Scalar value =
          scaledProduct(leftScale, rightScale, sum.total(), divisor);
      if (!std::isfinite(value)) {
        return fail<Matrix>(ErrorCode::NonFiniteResult,
                            "population covariance is non-finite");
      }
      output.at(i, j) = value;
      if (right == nullptr && i != j) {
        output.at(j, i) = value;
      }
    }
  }
  return Result<Matrix>::success(std::move(output));
}

Result<Estimate<Matrix>> makeEstimate(const SampleView& left,
                                      const SampleView* right,
                                      const CenterSpec& leftCenter,
                                      const CenterSpec* rightCenter,
                                      PopulationPolicy policy, Limits limits) {
  const auto leftDimension = validateSample(left);
  if (!leftDimension) {
    return fail<Estimate<Matrix>>(
        leftDimension.status().code(),
        std::string(leftDimension.status().message()));
  }
  Result<Extent> rightDimension = Result<Extent>::success(0U);
  if (right != nullptr) {
    rightDimension = validateSample(*right);
    if (!rightDimension) {
      return fail<Estimate<Matrix>>(
          rightDimension.status().code(),
          std::string(rightDimension.status().message()));
    }
  }
  const auto leftPolicy =
      validatePolicy(policy, leftCenter, leftDimension.value());
  if (!leftPolicy.ok()) {
    return fail<Estimate<Matrix>>(leftPolicy.code(),
                                  std::string(leftPolicy.message()));
  }
  if (rightCenter != nullptr) {
    const auto rightPolicy =
        validatePolicy(policy, *rightCenter, rightDimension.value());
    if (!rightPolicy.ok()) {
      return fail<Estimate<Matrix>>(rightPolicy.code(),
                                    std::string(rightPolicy.message()));
    }
  }
  if (right != nullptr && left.rows.rows() != right->rows.rows()) {
    return fail<Estimate<Matrix>>(
        ErrorCode::DimensionMismatch,
        "cross-covariance rows must be explicitly paired");
  }
  const Extent outputRows = leftDimension.value();
  const Extent outputColumns =
      right == nullptr ? outputRows : rightDimension.value();
  if (outputRows > std::numeric_limits<Extent>::max() / outputColumns) {
    return fail<Estimate<Matrix>>(ErrorCode::SizeOverflow,
                                  "covariance output extent overflows");
  }
  const Extent outputElements = outputRows * outputColumns;
  if (outputElements > limits.elements ||
      outputElements > limits.bytes / sizeof(Scalar)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::ResourceLimit,
        "covariance output exceeds configured limits");
  }
  const Extent leftCenterElements =
      leftCenter.rule == CenterRule::estimated ? outputRows : 0U;
  const Extent rightCenterElements =
      right != nullptr && rightCenter->rule == CenterRule::estimated
          ? outputColumns
          : 0U;
  if (leftCenterElements >
      std::numeric_limits<Extent>::max() - rightCenterElements) {
    return fail<Estimate<Matrix>>(ErrorCode::SizeOverflow,
                                  "covariance workspace extent overflows");
  }
  const Extent centerElements = leftCenterElements + rightCenterElements;
  if (centerElements > std::numeric_limits<Extent>::max() - outputElements ||
      outputElements + centerElements > limits.bytes / sizeof(Scalar)) {
    return fail<Estimate<Matrix>>(
        ErrorCode::ResourceLimit,
        "covariance workspace exceeds configured limits");
  }
  const auto countResult = retainedRows(left, right, policy);
  if (!countResult) {
    return fail<Estimate<Matrix>>(countResult.status().code(),
                                  std::string(countResult.status().message()));
  }
  const Index count = countResult.value();

  std::optional<Vector> leftEstimated;
  if (leftCenter.rule == CenterRule::estimated) {
    auto value = estimatedCenter(left, right, count);
    if (!value) {
      return fail<Estimate<Matrix>>(value.status().code(),
                                    std::string(value.status().message()));
    }
    leftEstimated.emplace(std::move(value).value());
  }
  std::optional<Vector> rightEstimated;
  if (right != nullptr && rightCenter->rule == CenterRule::estimated) {
    auto value = estimatedCenter(*right, &left, count);
    if (!value) {
      return fail<Estimate<Matrix>>(value.status().code(),
                                    std::string(value.status().message()));
    }
    rightEstimated.emplace(std::move(value).value());
  }
  const auto covariance = covarianceValues(
      left, right, leftCenter, rightCenter,
      leftEstimated ? &*leftEstimated : nullptr,
      rightEstimated ? &*rightEstimated : nullptr, policy, count);
  if (!covariance) {
    return fail<Estimate<Matrix>>(covariance.status().code(),
                                  std::string(covariance.status().message()));
  }

  const CenterRule diagnosticCenter = leftCenter.rule;
  const Support support{
      left.rows.rows(),
      count,
      count,
      std::nullopt,
      static_cast<Scalar>(count),
      {1.0, 1.0 / static_cast<Scalar>(count),
       1.0 / (static_cast<Scalar>(count) * static_cast<Scalar>(count))},
      static_cast<Scalar>(count),
      policy.normalization == VarianceMode::iid_sample
          ? std::optional<Scalar>(static_cast<Scalar>(count - 1U))
          : std::nullopt,
      std::nullopt};
  EstimatorDiagnostics diagnostics{
      policy.estimand,
      WeightRule::unweighted,
      diagnosticCenter,
      Approximation::exact,
      support,
      {},
      {},
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "scale-normalized compensated two-pass population covariance",
      policy.normalization == VarianceMode::iid_sample
          ? "independent identically distributed paired rows"
          : "descriptive finite-population covariance",
      {}};
  return Result<Estimate<Matrix>>::success(
      {std::move(covariance).value(), std::move(diagnostics), std::nullopt});
}

bool validSolvePolicy(const SolvePolicy& policy) {
  return std::isfinite(policy.rank_abs) && policy.rank_abs >= 0.0 &&
         std::isfinite(policy.rank_rel) && policy.rank_rel >= 0.0 &&
         policy.rank_rel < 1.0 &&
         std::isfinite(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor >= 0.0 &&
         policy.reciprocal_condition_floor <= 1.0 &&
         std::isfinite(policy.iteration.abs) && policy.iteration.abs >= 0.0 &&
         std::isfinite(policy.iteration.rel) && policy.iteration.rel >= 0.0 &&
         (policy.rank == RankPolicy::reject ||
          policy.rank == RankPolicy::supported_subspace ||
          policy.rank == RankPolicy::minimum_norm);
}

Scalar correlationAllowance(Extent dimension) {
  return 64.0 * static_cast<Scalar>(std::max<Extent>(1U, dimension)) *
         std::numeric_limits<Scalar>::epsilon();
}

Scalar normalizedRatio(Scalar numerator, Scalar firstVariance,
                       Scalar secondVariance) {
  if (numerator == 0.0) {
    return 0.0;
  }
  int numeratorExponent = 0;
  int firstExponent = 0;
  int secondExponent = 0;
  const Scalar numeratorFraction = std::frexp(numerator, &numeratorExponent);
  Scalar firstFraction = std::frexp(firstVariance, &firstExponent);
  Scalar secondFraction = std::frexp(secondVariance, &secondExponent);
  const int firstRootExponent =
      firstExponent % 2 == 0 ? firstExponent / 2 : (firstExponent - 1) / 2;
  const int secondRootExponent =
      secondExponent % 2 == 0 ? secondExponent / 2 : (secondExponent - 1) / 2;
  if (firstExponent % 2 != 0) {
    firstFraction *= 2.0;
  }
  if (secondExponent % 2 != 0) {
    secondFraction *= 2.0;
  }
  const Scalar fraction =
      numeratorFraction / std::sqrt(firstFraction) / std::sqrt(secondFraction);
  return std::scalbn(fraction, numeratorExponent - firstRootExponent -
                                   secondRootExponent);
}

struct LongCompensatedSum final {
  long double sum{};
  long double correction{};

  void add(long double value) {
    const long double adjusted = value - correction;
    const long double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
};

Result<linear::SymmetricEigenDecomposition>
decomposeCorrelation(ConstMatrixView matrix, const SolvePolicy& policy) {
  linear::SymmetricEigenOptions options{};
  options.rankTolerance = {.absolute = 0.0, .relative = 0.0};
  options.symmetryTolerance = {.absolute = 0.0,
                               .relative = correlationAllowance(matrix.rows())};
  options.maximumSweeps = policy.iteration.max_iterations;
  if (options.maximumSweeps == 0U) {
    bool diagonal = true;
    for (Index row = 0U; row < matrix.rows(); ++row) {
      for (Index column = row + 1U; column < matrix.columns(); ++column) {
        diagonal = diagonal && matrix.at(row, column) == 0.0;
      }
    }
    if (!diagonal) {
      return Result<linear::SymmetricEigenDecomposition>::failure(
          ErrorCode::NoConvergence,
          "zero sweep budget cannot certify a non-diagonal spectrum");
    }
    options.maximumSweeps = 1U;
  }
  return linear::decomposeSymmetricEigen(matrix, options);
}

Result<covariance::CovarianceResult>
correlationResult(Matrix value, const SolvePolicy& policy) {
  const auto decomposition = decomposeCorrelation(value.view(), policy);
  if (!decomposition.hasValue()) {
    return Result<covariance::CovarianceResult>::failure(
        decomposition.status().code(),
        std::string(decomposition.status().message()));
  }
  const Scalar allowance = correlationAllowance(value.rows());
  const auto eigenvalues = decomposition.value().eigenvalues();
  Scalar largest = 0.0;
  for (Index i = 0U; i < eigenvalues.size(); ++i) {
    if (eigenvalues.at(i) < -allowance) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::NotPositiveSemidefinite,
          "correlation matrix is not positive semidefinite");
    }
    largest = std::max(largest, std::max(0.0, eigenvalues.at(i)));
  }
  const Scalar cutoff = std::max(policy.rank_abs, policy.rank_rel * largest);
  Index rank = 0U;
  Scalar minimum = std::numeric_limits<Scalar>::infinity();
  Scalar maximum = 0.0;
  for (Index i = 0U; i < eigenvalues.size(); ++i) {
    const Scalar eigenvalue = eigenvalues.at(i);
    if (eigenvalue > cutoff) {
      ++rank;
      minimum = std::min(minimum, eigenvalue);
      maximum = std::max(maximum, eigenvalue);
    }
  }
  std::optional<Scalar> reciprocal;
  covariance::CovarianceKind classification =
      covariance::CovarianceKind::semidefinite_singular;
  if (rank == value.rows()) {
    reciprocal = minimum / maximum;
    classification = *reciprocal < policy.reciprocal_condition_floor
                         ? covariance::CovarianceKind::ill_conditioned
                         : covariance::CovarianceKind::positive_definite;
  }
  return Result<covariance::CovarianceResult>::success(
      {.covariance = std::move(value),
       .classification = classification,
       .rank = rank,
       .cutoff = cutoff,
       .reciprocal_condition = reciprocal});
}

Result<Matrix> standardizedCovariance(ConstMatrixView input,
                                      const SolvePolicy& policy, Limits limits,
                                      bool permitZeroVarianceCoordinates) {
  const Extent dimension = input.rows();
  if (dimension == 0U) {
    return fail<Matrix>(ErrorCode::InvalidExtent,
                        "covariance matrix must be nonempty");
  }
  if (dimension != input.columns()) {
    return fail<Matrix>(ErrorCode::DimensionMismatch,
                        "covariance matrix must be square");
  }
  if (dimension > std::numeric_limits<Extent>::max() / dimension) {
    return fail<Matrix>(ErrorCode::SizeOverflow,
                        "covariance matrix extent overflows");
  }
  const Extent elements = dimension * dimension;
  // The input copy, normalized result, and Jacobi work/eigenvector matrices
  // remain live together. Account conservatively before inspecting values.
  if (elements > limits.elements / 5U ||
      elements > limits.bytes / (5U * sizeof(Scalar))) {
    return fail<Matrix>(ErrorCode::ResourceLimit,
                        "correlation workspace exceeds configured limits");
  }
  bool hasZeroVariance = false;
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = 0U; column < dimension; ++column) {
      if (!std::isfinite(input.at(row, column))) {
        return fail<Matrix>(ErrorCode::NonFiniteInput,
                            "covariance contains a non-finite value");
      }
    }
  }
  Scalar scale = 0.0;
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = 0U; column < dimension; ++column) {
      scale = std::max(scale, std::abs(input.at(row, column)));
    }
  }
  const Scalar allowance = correlationAllowance(dimension);
  Matrix symmetric(dimension, dimension, 0.0);
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = row; column < dimension; ++column) {
      if (scale != 0.0 && std::abs(input.at(row, column) / scale -
                                   input.at(column, row) / scale) > allowance) {
        return fail<Matrix>(ErrorCode::NotSymmetric,
                            "covariance mirrored entries are inconsistent");
      }
      const Scalar entry =
          std::midpoint(input.at(row, column), input.at(column, row));
      symmetric.at(row, column) = entry;
      symmetric.at(column, row) = entry;
    }
  }
  for (Index row = 0U; row < dimension; ++row) {
    const Scalar variance = symmetric.at(row, row);
    if (variance < 0.0) {
      return fail<Matrix>(ErrorCode::NotPositiveSemidefinite,
                          "covariance has a negative variance");
    }
    if (variance == 0.0) {
      for (Index column = 0U; column < dimension; ++column) {
        if (input.at(row, column) != 0.0 || input.at(column, row) != 0.0) {
          return fail<Matrix>(
              ErrorCode::NotPositiveSemidefinite,
              "zero-variance coordinate has nonzero covariance");
        }
      }
      hasZeroVariance = true;
    }
  }
  Matrix correlation(dimension, dimension, 0.0);
  for (Index row = 0U; row < dimension; ++row) {
    if (symmetric.at(row, row) == 0.0) {
      continue;
    }
    for (Index column = row; column < dimension; ++column) {
      if (symmetric.at(column, column) == 0.0) {
        continue;
      }
      const Scalar value =
          normalizedRatio(symmetric.at(row, column), symmetric.at(row, row),
                          symmetric.at(column, column));
      if (!std::isfinite(value)) {
        return fail<Matrix>(ErrorCode::NonFiniteResult,
                            "correlation normalization is non-finite");
      }
      correlation.at(row, column) = value;
      correlation.at(column, row) = value;
    }
    correlation.at(row, row) = 1.0;
  }
  const auto decomposition = decomposeCorrelation(correlation.view(), policy);
  if (!decomposition.hasValue()) {
    return Result<Matrix>::failure(
        decomposition.status().code(),
        std::string(decomposition.status().message()));
  }
  for (Index i = 0U; i < dimension; ++i) {
    if (decomposition.value().eigenvalues().at(i) < -allowance) {
      return fail<Matrix>(ErrorCode::NotPositiveSemidefinite,
                          "covariance is not positive semidefinite");
    }
  }
  if (hasZeroVariance && !permitZeroVarianceCoordinates) {
    return fail<Matrix>(ErrorCode::InvalidDomain,
                        "zero-variance coordinates have undefined correlation");
  }
  return Result<Matrix>::success(std::move(correlation));
}

} // namespace

Result<Estimate<Matrix>> covarianceMatrix(const SampleView& sample,
                                          const CenterSpec& center,
                                          PopulationPolicy policy,
                                          Limits limits) {
  return makeEstimate(sample, nullptr, center, nullptr, policy, limits);
}

Result<Estimate<Matrix>>
crossCovariance(const SampleView& left, const SampleView& right,
                const CenterSpec& leftCenter, const CenterSpec& rightCenter,
                PopulationPolicy policy, Limits limits) {
  return makeEstimate(left, &right, leftCenter, &rightCenter, policy, limits);
}

Result<covariance::CovarianceResult>
correlationMatrix(ConstMatrixView input, SolvePolicy policy, Limits limits) {
  if (!validSolvePolicy(policy)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidTolerance, "correlation solve policy is invalid");
  }
  if (input.rows() == 0U && input.columns() == 0U) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InsufficientSamples, "correlation needs a nonempty matrix");
  }
  auto correlation = standardizedCovariance(input, policy, limits, false);
  if (!correlation.hasValue()) {
    return Result<covariance::CovarianceResult>::failure(
        correlation.status().code(),
        std::string(correlation.status().message()));
  }
  return correlationResult(std::move(correlation).value(), policy);
}

Result<covariance::CovarianceResult>
partialCorrelation(ConstMatrixView input, std::span<const Index> selected,
                   std::span<const Index> conditioned, SolvePolicy policy,
                   Limits limits) {
  if (!validSolvePolicy(policy)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidTolerance,
        "partial-correlation solve policy is invalid");
  }
  if (selected.empty()) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InsufficientSamples,
        "partial correlation requires at least one selected coordinate");
  }
  const Extent inputDimension = input.rows();
  if (inputDimension == 0U) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidExtent, "covariance matrix must be nonempty");
  }
  if (inputDimension != input.columns()) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::DimensionMismatch, "covariance matrix must be square");
  }
  const auto square = [](Extent extent, Extent& result) {
    if (extent != 0U && extent > std::numeric_limits<Extent>::max() / extent) {
      return false;
    }
    result = extent * extent;
    return true;
  };
  Extent inputElements = 0U;
  Extent conditionedElements = 0U;
  Extent selectedElements = 0U;
  if (!square(inputDimension, inputElements) ||
      !square(conditioned.size(), conditionedElements) ||
      !square(selected.size(), selectedElements)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::SizeOverflow, "partial-correlation workspace overflows");
  }
  // Bound peak simultaneous square matrices: standardized joint plus its
  // eigensolver, conditioning block plus decomposition, and conditional and
  // returned matrices. Check before scanning the caller's matrix.
  if (inputElements > std::numeric_limits<Extent>::max() / 5U ||
      conditionedElements > std::numeric_limits<Extent>::max() / 5U ||
      selectedElements > std::numeric_limits<Extent>::max() / 5U) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::SizeOverflow, "partial-correlation workspace overflows");
  }
  const Extent inputWorkspace = 5U * inputElements;
  const Extent conditionedWorkspace = 5U * conditionedElements;
  const Extent selectedWorkspace = 5U * selectedElements;
  if (inputWorkspace >
          std::numeric_limits<Extent>::max() - conditionedWorkspace ||
      inputWorkspace + conditionedWorkspace >
          std::numeric_limits<Extent>::max() - selectedWorkspace) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::SizeOverflow, "partial-correlation workspace overflows");
  }
  const Extent workspace =
      inputWorkspace + conditionedWorkspace + selectedWorkspace;
  if (workspace > limits.elements ||
      workspace > limits.bytes / sizeof(Scalar)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::ResourceLimit,
        "partial-correlation workspace exceeds configured limits");
  }
  std::vector<bool> seen(inputDimension, false);
  for (const Index index : selected) {
    if (index >= inputDimension) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::IndexOutOfRange, "selected coordinate is out of range");
    }
    if (seen[index]) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::InvalidDomain, "selected coordinates must be unique");
    }
    seen[index] = true;
  }
  for (const Index index : conditioned) {
    if (index >= inputDimension) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::IndexOutOfRange, "conditioned coordinate is out of range");
    }
    if (seen[index]) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::InvalidDomain,
          "selected and conditioned coordinates must be unique and disjoint");
    }
    seen[index] = true;
  }
  auto joint = standardizedCovariance(input, policy, limits, true);
  if (!joint.hasValue()) {
    return Result<covariance::CovarianceResult>::failure(
        joint.status().code(), std::string(joint.status().message()));
  }
  const Extent dimension = joint.value().rows();
  const Extent kept = selected.size();
  const Extent given = conditioned.size();
  for (const Index index : selected) {
    if (joint.value().at(index, index) == 0.0) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::InvalidDomain,
          "selected coordinate has zero marginal variance");
    }
  }
  if (kept > std::numeric_limits<Extent>::max() / kept) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::SizeOverflow, "partial-correlation extent overflows");
  }
  const Extent outputElements = kept * kept;
  if (outputElements > limits.elements ||
      outputElements > limits.bytes / sizeof(Scalar)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::ResourceLimit, "partial-correlation output exceeds limits");
  }

  Matrix conditional(kept, kept, 0.0);
  if (given == 0U) {
    for (Index row = 0U; row < kept; ++row) {
      for (Index column = 0U; column < kept; ++column) {
        conditional.at(row, column) =
            joint.value().at(selected[row], selected[column]);
      }
    }
  } else {
    Matrix conditionedBlock(given, given, 0.0);
    for (Index row = 0U; row < given; ++row) {
      for (Index column = 0U; column < given; ++column) {
        conditionedBlock.at(row, column) =
            joint.value().at(conditioned[row], conditioned[column]);
      }
    }
    const auto decomposition =
        decomposeCorrelation(conditionedBlock.view(), policy);
    if (!decomposition.hasValue()) {
      return Result<covariance::CovarianceResult>::failure(
          decomposition.status().code(),
          std::string(decomposition.status().message()));
    }
    const auto eigenvalues = decomposition.value().eigenvalues();
    Scalar largest = 0.0;
    for (Index i = 0U; i < given; ++i) {
      if (eigenvalues.at(i) < -correlationAllowance(dimension)) {
        return fail<covariance::CovarianceResult>(
            ErrorCode::NotPositiveSemidefinite,
            "conditioned correlation block is not positive semidefinite");
      }
      largest = std::max(largest, std::max(0.0, eigenvalues.at(i)));
    }
    const Scalar cutoff = std::max(policy.rank_abs, policy.rank_rel * largest);
    std::vector<Index> retained;
    for (Index i = 0U; i < given; ++i) {
      if (eigenvalues.at(i) > cutoff) {
        retained.push_back(i);
      }
    }
    if (policy.rank == RankPolicy::reject && retained.size() != given) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::RankDeficient,
          "conditioning block has directions excluded by the rank policy");
    }
    if (!retained.empty() && policy.reciprocal_condition_floor > 0.0) {
      const Scalar reciprocal =
          eigenvalues.at(retained.front()) / eigenvalues.at(retained.back());
      if (reciprocal < policy.reciprocal_condition_floor) {
        return fail<covariance::CovarianceResult>(
            ErrorCode::IllConditioned,
            "retained conditioning subspace is below the condition floor");
      }
    }
    const Scalar compatibilityAllowance = correlationAllowance(dimension);
    for (Index k = 0U; k < given; ++k) {
      if (eigenvalues.at(k) > cutoff) {
        continue;
      }
      for (Index row = 0U; row < kept; ++row) {
        LongCompensatedSum projection;
        for (Index column = 0U; column < given; ++column) {
          projection.add(static_cast<long double>(joint.value().at(
                             selected[row], conditioned[column])) *
                         decomposition.value().eigenvectors().at(column, k));
        }
        const long double permitted =
            std::sqrt(std::max(0.0, eigenvalues.at(k))) +
            compatibilityAllowance;
        if (std::abs(projection.sum) > permitted) {
          return fail<covariance::CovarianceResult>(
              ErrorCode::NotPositiveSemidefinite,
              "selected coordinates are incompatible with discarded "
              "conditioning directions");
        }
      }
    }
    for (Index row = 0U; row < kept; ++row) {
      for (Index column = row; column < kept; ++column) {
        LongCompensatedSum reduction;
        for (const Index k : retained) {
          LongCompensatedSum leftProjection;
          LongCompensatedSum rightProjection;
          for (Index q = 0U; q < given; ++q) {
            const long double eigenvector =
                decomposition.value().eigenvectors().at(q, k);
            leftProjection.add(static_cast<long double>(joint.value().at(
                                   selected[row], conditioned[q])) *
                               eigenvector);
            rightProjection.add(static_cast<long double>(joint.value().at(
                                    selected[column], conditioned[q])) *
                                eigenvector);
          }
          reduction.add(leftProjection.sum * rightProjection.sum /
                        eigenvalues.at(k));
        }
        const long double residual = static_cast<long double>(joint.value().at(
                                         selected[row], selected[column])) -
                                     reduction.sum;
        const Scalar value = static_cast<Scalar>(residual);
        if (!std::isfinite(value)) {
          return fail<covariance::CovarianceResult>(
              ErrorCode::NonFiniteResult,
              "conditional correlation covariance is non-finite");
        }
        conditional.at(row, column) = value;
        conditional.at(column, row) = value;
      }
    }
  }
  const Scalar allowance = correlationAllowance(dimension);
  for (Index i = 0U; i < kept; ++i) {
    if (conditional.at(i, i) < -allowance) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::NotPositiveSemidefinite,
          "conditional variance is materially negative");
    }
    if (conditional.at(i, i) <= allowance) {
      return fail<covariance::CovarianceResult>(
          ErrorCode::InvalidDomain,
          "conditional variance is zero or unresolved at working precision");
    }
  }
  Matrix output(kept, kept, 0.0);
  for (Index row = 0U; row < kept; ++row) {
    for (Index column = row; column < kept; ++column) {
      const Scalar value =
          normalizedRatio(conditional.at(row, column), conditional.at(row, row),
                          conditional.at(column, column));
      if (!std::isfinite(value)) {
        return fail<covariance::CovarianceResult>(
            ErrorCode::NonFiniteResult,
            "partial-correlation normalization is non-finite");
      }
      output.at(row, column) = value;
      output.at(column, row) = value;
    }
    output.at(row, row) = 1.0;
  }
  return correlationResult(std::move(output), policy);
}

} // namespace vss::math::statistics
