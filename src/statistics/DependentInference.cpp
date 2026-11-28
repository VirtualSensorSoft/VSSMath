#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/DependentInference.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedAdd(std::size_t left, std::size_t right,
                std::size_t& result) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(std::size_t left, std::size_t right,
                     std::size_t& result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

struct Scaled final {
  Scalar fraction{0.0};
  int exponent{0};
};

Scaled scaled(Scalar value) noexcept {
  if (value == 0.0) {
    return {};
  }
  int exponent = 0;
  return {.fraction = std::frexp(value, &exponent), .exponent = exponent};
}

Scaled multiply(Scaled left, Scaled right) noexcept {
  if (left.fraction == 0.0 || right.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(left.fraction * right.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = left.exponent + right.exponent + adjustment};
}

Scaled divide(Scaled numerator, Scaled denominator) noexcept {
  if (numerator.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(numerator.fraction / denominator.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = numerator.exponent - denominator.exponent + adjustment};
}

class ScaledAccumulator final {
public:
  void add(Scaled value) noexcept {
    if (value.fraction == 0.0) {
      return;
    }
    if (!initialized_) {
      exponent_ = value.exponent;
      initialized_ = true;
    } else if (value.exponent > exponent_) {
      const Scalar factor = std::scalbn(1.0, exponent_ - value.exponent);
      sum_ *= factor;
      correction_ *= factor;
      exponent_ = value.exponent;
    }
    const Scalar addend =
        std::scalbn(value.fraction, value.exponent - exponent_);
    const Scalar total = sum_ + addend;
    const Scalar directCorrection = (sum_ - total) + addend;
    const Scalar alternateCorrection = (addend - total) + sum_;
    correction_ += std::abs(sum_) >= std::abs(addend) ? directCorrection
                                                      : alternateCorrection;
    sum_ = total;
  }

  [[nodiscard]] Scaled value() const noexcept {
    const Scalar total = sum_ + correction_;
    if (!initialized_ || total == 0.0) {
      return {};
    }
    int adjustment = 0;
    const Scalar fraction = std::frexp(total, &adjustment);
    return {.fraction = fraction, .exponent = exponent_ + adjustment};
  }

private:
  Scalar sum_{0.0};
  Scalar correction_{0.0};
  int exponent_{0};
  bool initialized_{false};
};

Status validateSample(const SampleView& sample) {
  const std::size_t rows = sample.rows.rows();
  const std::size_t columns = sample.rows.columns();
  if (rows == 0U || columns == 0U) {
    return Status::failure(
        ErrorCode::InsufficientSamples,
        "Newey-West requires nonempty score rows and coordinates");
  }
  if (!sample.observation_shape.empty()) {
    // NOLINTNEXTLINE(misc-const-correctness): checkedMultiply updates this
    // output.
    std::size_t shaped = 1U;
    for (const Index extent : sample.observation_shape) {
      if (extent == 0U || !checkedMultiply(shaped, extent, shaped)) {
        return Status::failure(ErrorCode::InvalidExtent,
                               "score observation shape is invalid");
      }
    }
    if (shaped != columns) {
      return Status::failure(
          ErrorCode::DimensionMismatch,
          "score observation shape does not match row width");
    }
  } else if (columns != 1U) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "empty score observation shape requires scalar rows");
  }
  std::size_t elements = 0U;
  if (!checkedMultiply(rows, columns, elements)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "score sample extent overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != elements) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "score mask has the wrong extent");
  }
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const std::size_t offset = (row * columns) + column;
      if (!sample.valid.empty()) {
        const std::uint8_t valid = sample.valid[offset];
        if (valid > 1U) {
          return Status::failure(ErrorCode::InvalidDomain,
                                 "score mask is not boolean");
        }
        if (valid == 0U) {
          return Status::failure(
              ErrorCode::InvalidDomain,
              "Newey-West requires complete ordered score rows");
        }
      }
      if (!std::isfinite(sample.rows.at(row, column))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "score sample contains a non-finite cell");
      }
    }
  }
  return Status::success();
}

Status preflight(std::size_t dimension, Limits limits) {
  std::size_t square = 0U;
  std::size_t pairBytes = 0U;
  // NOLINTNEXTLINE(misc-const-correctness): checkedMultiply output parameter.
  std::size_t matrixBytes = 0U;
  std::size_t workspaceElements = 0U;
  // NOLINTNEXTLINE(misc-const-correctness): checkedMultiply output parameter.
  std::size_t workspaceBytes = 0U;
  // NOLINTNEXTLINE(misc-const-correctness): checkedMultiply output parameter.
  std::size_t vectorBytes = 0U;
  std::size_t totalBytes = 0U;
  if (!checkedMultiply(dimension, dimension, square) ||
      !checkedMultiply(square, 5U, workspaceElements) ||
      !checkedMultiply(workspaceElements, sizeof(Scalar), workspaceBytes) ||
      !checkedMultiply(square, sizeof(ScaledAccumulator), pairBytes) ||
      !checkedMultiply(square, sizeof(Scalar), matrixBytes) ||
      !checkedMultiply(dimension, 8U * sizeof(Scalar), vectorBytes) ||
      !checkedAdd(workspaceBytes, pairBytes, totalBytes) ||
      !checkedAdd(totalBytes, matrixBytes, totalBytes) ||
      !checkedAdd(totalBytes, vectorBytes, totalBytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "Newey-West workspace size overflows");
  }
  if (square > limits.elements || totalBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "Newey-West exceeds configured limits");
  }
  return Status::success();
}

} // namespace

Result<covariance::CovarianceResult> neweyWest(const SampleView& scores,
                                               Index lag, CenterRule center,
                                               Limits limits) {
  if (center != CenterRule::estimated && center != CenterRule::known) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidDomain, "Newey-West center rule is invalid");
  }
  const Status validated = validateSample(scores);
  if (!validated.ok()) {
    return copyFailure<covariance::CovarianceResult>(validated);
  }
  const std::size_t count = scores.rows.rows();
  const std::size_t dimension = scores.rows.columns();
  if (lag >= count) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidDomain, "Newey-West lag must be less than N");
  }
  const Status budget = preflight(dimension, limits);
  if (!budget.ok()) {
    return copyFailure<covariance::CovarianceResult>(budget);
  }
  std::size_t windowCount = 0U;
  std::size_t windowLength = 0U;
  std::size_t denominatorCount = 0U;
  if (!checkedAdd(count, lag, windowCount) ||
      !checkedAdd(lag, 1U, windowLength) ||
      !checkedMultiply(count, windowLength, denominatorCount)) {
    return fail<covariance::CovarianceResult>(ErrorCode::SizeOverflow,
                                              "Newey-West extent overflows");
  }

  std::vector<Scalar> scales(dimension, 0.0);
  std::vector<Scalar> centers(dimension, 0.0);
  std::vector<detail::CompensatedSum> centerSums(dimension);
  for (Index column = 0U; column < dimension; ++column) {
    for (Index row = 0U; row < count; ++row) {
      scales[column] =
          std::max(scales[column], std::abs(scores.rows.at(row, column)));
    }
    if (center == CenterRule::estimated && scales[column] != 0.0) {
      for (Index row = 0U; row < count; ++row) {
        centerSums[column].add(scores.rows.at(row, column) / scales[column]);
      }
      centers[column] = centerSums[column].total() / static_cast<Scalar>(count);
    }
  }

  const std::size_t square = dimension * dimension;
  std::vector<ScaledAccumulator> products(square);
  std::vector<detail::CompensatedSum> window(dimension);
  std::vector<Scalar> windowValues(dimension, 0.0);
  for (std::size_t index = 0U; index < windowCount; ++index) {
    for (Index column = 0U; column < dimension; ++column) {
      if (index < count && scales[column] != 0.0) {
        const Scalar value =
            (scores.rows.at(index, column) / scales[column]) - centers[column];
        window[column].add(value);
      }
      if (index > lag) {
        const std::size_t outgoing = index - lag - 1U;
        if (scales[column] != 0.0) {
          const Scalar value =
              (scores.rows.at(outgoing, column) / scales[column]) -
              centers[column];
          window[column].add(-value);
        }
      }
      windowValues[column] = window[column].total();
    }
    for (Index row = 0U; row < dimension; ++row) {
      for (Index column = row; column < dimension; ++column) {
        const Scaled term =
            multiply(scaled(windowValues[row]), scaled(windowValues[column]));
        products[(row * dimension) + column].add(term);
      }
    }
  }

  Matrix estimate(dimension, dimension);
  const Scaled denominator = scaled(static_cast<Scalar>(denominatorCount));
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = row; column < dimension; ++column) {
      Scaled value = products[(row * dimension) + column].value();
      value = multiply(value, scaled(scales[row]));
      value = multiply(value, scaled(scales[column]));
      const Scaled normalized = divide(value, denominator);
      const Scalar result =
          std::scalbn(normalized.fraction, normalized.exponent);
      if (!std::isfinite(result)) {
        return fail<covariance::CovarianceResult>(
            ErrorCode::NonFiniteResult,
            "Newey-West covariance is not representable as a finite matrix");
      }
      estimate.at(row, column) = result;
      // Exact symmetric mirror of the upper-triangle value.
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      estimate.at(column, row) = result;
    }
  }

  const auto diagnostic = covariance::analyzeCovariance(estimate.view());
  if (!diagnostic.hasValue()) {
    return copyFailure<covariance::CovarianceResult>(diagnostic.status());
  }
  const auto& spectrum = diagnostic.value();
  if (spectrum.classification() ==
      covariance::CovarianceClassification::NotPositiveSemidefinite) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::NoConvergence,
        "Newey-West result is materially indefinite under C014 analysis");
  }
  covariance::CovarianceKind classification =
      covariance::CovarianceKind::semidefinite_singular;
  if (spectrum.classification() ==
      covariance::CovarianceClassification::PositiveDefinite) {
    classification = covariance::CovarianceKind::positive_definite;
  }
  std::optional<Scalar> reciprocalCondition;
  if (spectrum.spectralConditionNumber().has_value()) {
    const Scalar condition = spectrum.spectralConditionNumber().value();
    if (condition > 0.0 && !std::isnan(condition)) {
      reciprocalCondition = std::isinf(condition) ? 0.0 : 1.0 / condition;
    }
  }
  return Result<covariance::CovarianceResult>::success(
      {.covariance = std::move(estimate),
       .classification = classification,
       .rank = spectrum.rank(),
       .cutoff = spectrum.rankThreshold(),
       .reciprocal_condition = reciprocalCondition});
}

Result<BootstrapResult> dependentBootstrap(const SampleView& x,
                                           const Statistic& f,
                                           const BootstrapPolicy& p,
                                           probability::Rng& rng,
                                           Limits limits) {
  if (p.kind != BootstrapKind::moving_block &&
      p.kind != BootstrapKind::circular_block) {
    return fail<BootstrapResult>(
        ErrorCode::InvalidDomain,
        "dependent bootstrap requires a moving- or circular-block policy");
  }
  return bootstrapEstimate(x, f, p, rng, limits);
}

} // namespace vss::math::statistics
