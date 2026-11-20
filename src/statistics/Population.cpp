#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
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

} // namespace vss::math::statistics
