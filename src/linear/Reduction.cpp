#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/Reduction.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedAdd(Index lhs, Index rhs, Index& result) noexcept {
  if (rhs > std::numeric_limits<Index>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(Index lhs, Index rhs, Index& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Index>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool reserveLimits(Index elements, Limits limits, Limits& factorLimits) {
  Index bytes = 0U;
  if (!checkedMultiply(elements, sizeof(Scalar), bytes) ||
      elements > limits.elements || bytes > limits.bytes) {
    return false;
  }
  factorLimits = {.elements = limits.elements - elements,
                  .bytes = limits.bytes - bytes,
                  .evaluations = limits.evaluations};
  return true;
}

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

Status validateSampleView(const SampleView& samples) {
  Index flattened = 1U;
  for (const Index extent : samples.observation_shape) {
    if (extent == 0U) {
      return Status::failure(ErrorCode::InvalidExtent,
                             "sample observation extents must be positive");
    }
    if (!checkedMultiply(flattened, extent, flattened)) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "sample observation shape overflows");
    }
  }
  if (flattened != samples.rows.columns()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "sample shape does not match row width");
  }
  Index cells = 0U;
  if (!checkedMultiply(samples.rows.rows(), samples.rows.columns(), cells)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "sample population size overflows");
  }
  if (!samples.valid.empty() && samples.valid.size() != cells) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "sample validity mask has the wrong size");
  }
  for (const std::uint8_t observed : samples.valid) {
    if (observed > 1U) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "sample validity mask entries must be zero or one");
    }
    if (observed == 0U) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "PCA requires complete observations");
    }
  }
  return Status::success();
}

bool representable(long double value) noexcept {
  return std::isfinite(value) &&
         std::abs(value) <= std::numeric_limits<Scalar>::max() &&
         (value == 0.0L || static_cast<Scalar>(value) != 0.0);
}

class ScaledSumSquares final {
public:
  void add(long double value) noexcept {
    const long double magnitude = std::abs(value);
    if (magnitude == 0.0L) {
      return;
    }
    if (scale_ < magnitude) {
      const long double ratio = scale_ / magnitude;
      sum_ = 1.0L + (sum_ * ratio * ratio);
      scale_ = magnitude;
    } else {
      const long double ratio = magnitude / scale_;
      sum_ += ratio * ratio;
    }
  }

  [[nodiscard]] long double norm() const noexcept {
    return scale_ == 0.0L ? 0.0L : scale_ * std::sqrt(sum_);
  }

private:
  long double scale_{};
  long double sum_{1.0L};
};

} // namespace

Result<LowRank> lowRankApproximation(ConstMatrixView input, Index rank,
                                     SolvePolicy policy, Limits limits) {
  const Index rows = input.rows();
  const Index columns = input.columns();
  const Index thin = std::min(rows, columns);
  if (rank > thin) {
    return failure<LowRank>(
        ErrorCode::InvalidExtent,
        "requested approximation rank exceeds matrix shape");
  }
  Index leftElements = 0U;
  Index rightElements = 0U;
  Index outputElements = 0U;
  if (!checkedMultiply(rows, rank, leftElements) ||
      !checkedMultiply(rank, columns, rightElements) ||
      !checkedAdd(leftElements, rightElements, outputElements) ||
      !checkedAdd(outputElements, rank, outputElements)) {
    return failure<LowRank>(ErrorCode::ResourceLimit,
                            "low-rank output dimensions overflow");
  }
  Limits factorLimits{};
  if (!reserveLimits(outputElements, limits, factorLimits)) {
    return failure<LowRank>(ErrorCode::ResourceLimit,
                            "low-rank factors and approximation exceed limits");
  }
  if (!finite(input)) {
    return failure<LowRank>(ErrorCode::NonFiniteInput,
                            "low-rank input must be finite");
  }
  const auto factor = factorSVD(input, policy, factorLimits);
  if (!factor.hasValue()) {
    return copyFailure<LowRank>(factor.status());
  }

  Matrix left(rows, rank, 0.0);
  Vector singular(rank, 0.0);
  Matrix rightTranspose(rank, columns, 0.0);
  for (Index mode = 0U; mode < rank; ++mode) {
    singular.at(mode) = factor.value().singular.at(mode);
    for (Index row = 0U; row < rows; ++row) {
      left.at(row, mode) = factor.value().left.at(row, mode);
    }
    for (Index column = 0U; column < columns; ++column) {
      rightTranspose.at(mode, column) =
          factor.value().right_transpose.at(mode, column);
    }
  }

  ScaledSumSquares residual;
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      long double reconstructed = 0.0L;
      for (Index mode = 0U; mode < rank; ++mode) {
        reconstructed +=
            static_cast<long double>(left.at(row, mode)) *
            static_cast<long double>(singular.at(mode)) *
            static_cast<long double>(rightTranspose.at(mode, column));
      }
      if (!representable(reconstructed)) {
        return failure<LowRank>(ErrorCode::NonFiniteResult,
                                "low-rank reconstruction is not representable");
      }
      residual.add(static_cast<long double>(input.at(row, column)) -
                   reconstructed);
    }
  }
  const long double residualNorm = residual.norm();
  if (!representable(residualNorm)) {
    return failure<LowRank>(ErrorCode::NonFiniteResult,
                            "low-rank residual norm is not representable");
  }
  return Result<LowRank>::success(
      {.left = std::move(left),
       .singular = std::move(singular),
       .right_transpose = std::move(rightTranspose),
       .residual_frobenius = static_cast<Scalar>(residualNorm)});
}

Result<PCA> pca(const SampleView& samples, Index rank, bool whiten,
                SolvePolicy policy, Limits limits) {
  const Index count = samples.rows.rows();
  const Index dimensions = samples.rows.columns();
  if (count == 0U || dimensions == 0U) {
    return failure<PCA>(ErrorCode::InsufficientSamples,
                        "PCA requires observations and variables");
  }
  if (count < 2U) {
    return failure<PCA>(ErrorCode::InsufficientSamples,
                        "PCA sample variance requires at least two rows");
  }
  const Status sampleStatus = validateSampleView(samples);
  if (!sampleStatus.ok()) {
    return Result<PCA>::failure(sampleStatus.code(),
                                std::string(sampleStatus.message()));
  }
  const Index thin = std::min(count - 1U, dimensions);
  if (rank > thin) {
    return failure<PCA>(ErrorCode::InvalidExtent,
                        "requested PCA rank exceeds centered sample rank");
  }

  Index centeredElements = 0U;
  Index componentElements = 0U;
  Index scoreElements = 0U;
  Index outputElements = 0U;
  if (!checkedMultiply(count, dimensions, centeredElements) ||
      !checkedMultiply(dimensions, rank, componentElements) ||
      !checkedMultiply(count, rank, scoreElements) ||
      !checkedAdd(centeredElements, componentElements, outputElements) ||
      !checkedAdd(outputElements, scoreElements, outputElements) ||
      !checkedAdd(outputElements, dimensions, outputElements) ||
      !checkedAdd(outputElements, rank, outputElements)) {
    return failure<PCA>(ErrorCode::ResourceLimit,
                        "PCA output dimensions overflow");
  }
  Limits factorLimits{};
  if (!reserveLimits(outputElements, limits, factorLimits)) {
    return failure<PCA>(ErrorCode::ResourceLimit,
                        "PCA centered data and outputs exceed limits");
  }
  if (!finite(samples.rows)) {
    return failure<PCA>(ErrorCode::NonFiniteInput,
                        "PCA samples must be finite");
  }

  Vector center(dimensions, 0.0);
  for (Index column = 0U; column < dimensions; ++column) {
    long double sum = 0.0L;
    for (Index row = 0U; row < count; ++row) {
      sum += static_cast<long double>(samples.rows.at(row, column));
    }
    const long double mean = sum / static_cast<long double>(count);
    if (!representable(mean)) {
      return failure<PCA>(ErrorCode::NonFiniteResult,
                          "PCA sample center is not representable");
    }
    center.at(column) = static_cast<Scalar>(mean);
  }

  Matrix centered(count, dimensions, 0.0);
  for (Index row = 0U; row < count; ++row) {
    for (Index column = 0U; column < dimensions; ++column) {
      const long double value =
          static_cast<long double>(samples.rows.at(row, column)) -
          static_cast<long double>(center.at(column));
      if (!representable(value)) {
        return failure<PCA>(ErrorCode::NonFiniteResult,
                            "PCA centered data is not representable");
      }
      centered.at(row, column) = static_cast<Scalar>(value);
    }
  }

  const auto factor = factorSVD(centered.view(), policy, factorLimits);
  if (!factor.hasValue()) {
    return copyFailure<PCA>(factor.status());
  }
  Matrix axes(dimensions, rank, 0.0);
  Matrix scores(count, rank, 0.0);
  Vector explained(rank, 0.0);
  const auto sampleDenominator = static_cast<long double>(count - 1U);
  for (Index mode = 0U; mode < rank; ++mode) {
    const Scalar singular = factor.value().singular.at(mode);
    if (whiten && singular <= factor.value().cutoff) {
      return failure<PCA>(ErrorCode::RankDeficient,
                          "PCA cannot whiten a zero or discarded component");
    }
    const long double sigma = singular;
    const long double variance = sigma * sigma / sampleDenominator;
    if (!representable(variance)) {
      return failure<PCA>(ErrorCode::NonFiniteResult,
                          "PCA explained variance is not representable");
    }
    explained.at(mode) = static_cast<Scalar>(variance);
    for (Index dimension = 0U; dimension < dimensions; ++dimension) {
      axes.at(dimension, mode) =
          factor.value().right_transpose.at(mode, dimension);
    }
    for (Index row = 0U; row < count; ++row) {
      long double score =
          static_cast<long double>(factor.value().left.at(row, mode)) * sigma;
      if (whiten) {
        score = static_cast<long double>(factor.value().left.at(row, mode)) *
                std::sqrt(sampleDenominator);
      }
      if (!representable(score)) {
        return failure<PCA>(ErrorCode::NonFiniteResult,
                            "PCA score is not representable");
      }
      scores.at(row, mode) = static_cast<Scalar>(score);
    }
  }

  return Result<PCA>::success({.center = std::move(center),
                               .axes = std::move(axes),
                               .scores = std::move(scores),
                               .explained_variance = std::move(explained),
                               .whitened = whiten});
}

} // namespace vss::math::linear
