#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Order.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedMultiply(std::size_t left, std::size_t right,
                     std::size_t& product) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  product = left * right;
  return true;
}

bool checkedAdd(std::size_t left, std::size_t right,
                std::size_t& sum) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  sum = left + right;
  return true;
}

bool validPolicy(const BootstrapPolicy& policy) noexcept {
  const bool kind = policy.kind == BootstrapKind::iid ||
                    policy.kind == BootstrapKind::moving_block ||
                    policy.kind == BootstrapKind::circular_block;
  const bool interval = policy.interval == QuantileInterval::percentile ||
                        policy.interval == QuantileInterval::basic;
  const bool invalid = policy.invalid == InvalidDraw::reject ||
                       policy.invalid == InvalidDraw::report_conditional;
  return kind && interval && invalid;
}

Status validateSample(const SampleView& sample, std::size_t& rows,
                      std::size_t& columns, std::size_t& elements) {
  rows = sample.rows.rows();
  columns = sample.rows.columns();
  if (rows == 0U || columns == 0U) {
    return Status::failure(ErrorCode::InsufficientSamples,
                           "bootstrap requires a nonempty sample");
  }
  if (sample.observation_shape.empty()) {
    if (columns != 1U) {
      return Status::failure(
          ErrorCode::DimensionMismatch,
          "empty bootstrap observation shape requires scalar rows");
    }
  } else {
    std::size_t shaped = 1U;
    for (const Index extent : sample.observation_shape) {
      if (extent == 0U || !checkedMultiply(shaped, extent, shaped)) {
        return Status::failure(ErrorCode::InvalidExtent,
                               "bootstrap observation shape is invalid");
      }
    }
    if (shaped != columns) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "bootstrap shape does not match sample columns");
    }
  }
  if (!checkedMultiply(rows, columns, elements)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "bootstrap sample extent overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != elements) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "bootstrap validity mask has the wrong extent");
  }
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const std::size_t offset = row * columns + column;
      if (!sample.valid.empty()) {
        const std::uint8_t valid = sample.valid[offset];
        if (valid > 1U) {
          return Status::failure(ErrorCode::InvalidDomain,
                                 "bootstrap validity mask is not boolean");
        }
        if (valid == 0U) {
          return Status::failure(
              ErrorCode::InvalidDomain,
              "bootstrap rejects masked cells without compacting the sample");
        }
      }
      if (!std::isfinite(sample.rows.at(row, column))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "bootstrap sample contains a non-finite cell");
      }
    }
  }
  return Status::success();
}

Status preflightInput(std::size_t sampleElements, Index draws,
                      bool originalCall, Limits limits) {
  if constexpr (sizeof(Index) > sizeof(std::uint64_t)) {
    if (draws > std::numeric_limits<std::uint64_t>::max()) {
      return Status::failure(ErrorCode::ResourceLimit,
                             "bootstrap callback count is not representable");
    }
  }
  std::uint64_t calls = static_cast<std::uint64_t>(draws);
  if (originalCall) {
    if (calls == std::numeric_limits<std::uint64_t>::max()) {
      return Status::failure(ErrorCode::ResourceLimit,
                             "bootstrap callback count overflows");
    }
    ++calls;
  }
  if (calls > limits.evaluations) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "bootstrap callback evaluation limit is too small");
  }
  std::size_t bytes = 0U;
  std::size_t vectorBytes = 0U;
  std::size_t totalBytes = 0U;
  if (!checkedMultiply(sampleElements, sizeof(Scalar), bytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "bootstrap replicate byte count overflows");
  }
  if (!checkedMultiply(static_cast<std::size_t>(draws), sizeof(Vector),
                       vectorBytes) ||
      !checkedAdd(bytes, vectorBytes, totalBytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "bootstrap attempt storage overflows");
  }
  if (sampleElements > limits.elements || totalBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "bootstrap replicate exceeds configured limits");
  }
  return Status::success();
}

Status preflightOutput(std::size_t sampleElements, Index draws,
                       std::size_t width, Limits limits) {
  std::size_t replicateValues = 0U;
  std::size_t twoReplicateValues = 0U;
  std::size_t extra = 0U;
  std::size_t totalElements = 0U;
  if (!checkedMultiply(static_cast<std::size_t>(draws), width,
                       replicateValues) ||
      !checkedMultiply(replicateValues, 2U, twoReplicateValues) ||
      !checkedMultiply(width, 5U, extra) ||
      !checkedAdd(sampleElements, twoReplicateValues, totalElements) ||
      !checkedAdd(totalElements, extra, totalElements)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "bootstrap result extent overflows");
  }
  std::size_t scalarBytes = 0U;
  std::size_t vectorBytes = 0U;
  std::size_t totalBytes = 0U;
  if (!checkedMultiply(totalElements, sizeof(Scalar), scalarBytes) ||
      !checkedMultiply(static_cast<std::size_t>(draws), sizeof(Vector),
                       vectorBytes) ||
      !checkedAdd(scalarBytes, vectorBytes, totalBytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "bootstrap result byte count overflows");
  }
  if (totalElements > limits.elements || totalBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "bootstrap result exceeds configured limits");
  }
  return Status::success();
}

Result<Index> uniformIndex(probability::Rng& rng, Index bound) {
  if (bound == 0U) {
    return failure<Index>(ErrorCode::InvalidExtent,
                          "bootstrap random-index bound is invalid");
  }
  if constexpr (sizeof(Index) > sizeof(std::uint64_t)) {
    if (bound > std::numeric_limits<std::uint64_t>::max()) {
      return failure<Index>(
          ErrorCode::InvalidExtent,
          "bootstrap random-index bound is not representable");
    }
  }
  const auto wideBound = static_cast<std::uint64_t>(bound);
  if (wideBound == 1U) {
    static_cast<void>(rng.next());
    return Result<Index>::success(0U);
  }
  const std::uint64_t threshold = (std::uint64_t{0} - wideBound) % wideBound;
  for (Index attempt = 0U; attempt < 128U; ++attempt) {
    const std::uint64_t word = rng.next();
    if (word >= threshold) {
      return Result<Index>::success(static_cast<Index>(word % wideBound));
    }
  }
  return failure<Index>(ErrorCode::NoConvergence,
                        "bootstrap random-index rejection cap exceeded");
}

Status fillReplicate(const SampleView& sample, BootstrapKind kind,
                     Index blockLength, probability::Rng& rng,
                     Matrix& replicate) {
  const Index rowCount = sample.rows.rows();
  const Index columnCount = sample.rows.columns();
  Index outputRow = 0U;
  while (outputRow < rowCount) {
    const Index startCount = kind == BootstrapKind::moving_block
                                 ? rowCount - blockLength + 1U
                                 : rowCount;
    const auto start = uniformIndex(rng, startCount);
    if (!start.hasValue()) {
      return start.status();
    }
    Index sourceRow = start.value();
    for (Index blockOffset = 0U;
         blockOffset < blockLength && outputRow < rowCount; ++blockOffset) {
      for (Index column = 0U; column < columnCount; ++column) {
        replicate.at(outputRow, column) = sample.rows.at(sourceRow, column);
      }
      ++outputRow;
      if (kind == BootstrapKind::circular_block) {
        sourceRow = sourceRow + 1U == rowCount ? 0U : sourceRow + 1U;
      } else if (blockOffset + 1U < blockLength && outputRow < rowCount) {
        ++sourceRow;
      }
    }
  }
  return Status::success();
}

bool conditionallyOmittable(ErrorCode code) noexcept {
  return code == ErrorCode::InvalidDomain ||
         code == ErrorCode::NonFiniteInput ||
         code == ErrorCode::NonFiniteResult;
}

Result<Vector> evaluateStatistic(const Statistic& statistic,
                                 const SampleView& sample) {
  try {
    return statistic(sample);
  } catch (const std::bad_alloc&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "bootstrap statistic allocation failed");
  } catch (const std::exception& error) {
    return Result<Vector>::failure(ErrorCode::CallbackFailure,
                                   std::string("bootstrap statistic threw: ") +
                                       error.what());
  } catch (...) {
    return failure<Vector>(ErrorCode::CallbackFailure,
                           "bootstrap statistic threw an unknown exception");
  }
}

Result<Scalar> standardError(const std::vector<Vector>& values,
                             std::size_t coordinate) {
  if (values.size() < 2U) {
    return failure<Scalar>(ErrorCode::InsufficientSamples,
                           "standard error requires two retained replicates");
  }
  Scalar scale = 0.0;
  for (const Vector& value : values) {
    scale = std::max(scale, std::abs(value.at(coordinate)));
  }
  if (scale == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  long double mean = 0.0L;
  for (const Vector& value : values) {
    mean += static_cast<long double>(value.at(coordinate) / scale);
  }
  mean /= static_cast<long double>(values.size());
  long double squared = 0.0L;
  for (const Vector& value : values) {
    const long double difference =
        static_cast<long double>(value.at(coordinate) / scale) - mean;
    squared += difference * difference;
  }
  const long double divisor = static_cast<long double>(values.size() - 1U) *
                              static_cast<long double>(values.size());
  const long double error =
      static_cast<long double>(scale) * std::sqrt(squared / divisor);
  const Scalar result = static_cast<Scalar>(error);
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "bootstrap Monte Carlo error is not representable");
  }
  return Result<Scalar>::success(result);
}

} // namespace

Result<BootstrapResult> bootstrapEstimate(const SampleView& sample,
                                          const Statistic& statistic,
                                          const BootstrapPolicy& policy,
                                          probability::Rng& rng,
                                          Limits limits) {
  if (!validPolicy(policy) || !statistic || !std::isfinite(policy.confidence)) {
    return failure<BootstrapResult>(ErrorCode::InvalidDomain,
                                    "bootstrap policy or callback is invalid");
  }
  if (policy.draws == 0U || policy.confidence <= 0.0 ||
      policy.confidence >= 1.0) {
    return failure<BootstrapResult>(ErrorCode::InvalidDomain,
                                    "bootstrap draws or confidence is invalid");
  }
  std::size_t rows = 0U;
  std::size_t columns = 0U;
  std::size_t sampleElements = 0U;
  const Status sampleStatus =
      validateSample(sample, rows, columns, sampleElements);
  if (!sampleStatus.ok()) {
    return copyFailure<BootstrapResult>(sampleStatus);
  }
  const Index blockLength = policy.block_length;
  if ((policy.kind == BootstrapKind::iid && blockLength != 1U) ||
      (policy.kind != BootstrapKind::iid &&
       (blockLength == 0U || blockLength > rows))) {
    return failure<BootstrapResult>(ErrorCode::InvalidDomain,
                                    "bootstrap block length is invalid");
  }
  const Status inputBudget =
      preflightInput(sampleElements, policy.draws,
                     policy.interval == QuantileInterval::basic, limits);
  if (!inputBudget.ok()) {
    return copyFailure<BootstrapResult>(inputBudget);
  }

  try {
    Matrix replicate(rows, columns, 0.0);
    std::vector<Vector> estimates;
    estimates.reserve(policy.draws);
    std::optional<Vector> original;
    std::size_t width = 0U;
    if (policy.interval == QuantileInterval::basic) {
      auto evaluated = evaluateStatistic(statistic, sample);
      if (!evaluated.hasValue()) {
        return copyFailure<BootstrapResult>(evaluated.status());
      }
      original = std::move(evaluated).value();
      if (original->empty()) {
        return failure<BootstrapResult>(ErrorCode::InvalidExtent,
                                        "bootstrap statistic is empty");
      }
      width = original->size();
      for (Index i = 0U; i < width; ++i) {
        if (!std::isfinite(original->at(i))) {
          return failure<BootstrapResult>(ErrorCode::NonFiniteResult,
                                          "original statistic is non-finite");
        }
      }
      const Status outputBudget =
          preflightOutput(sampleElements, policy.draws, width, limits);
      if (!outputBudget.ok()) {
        return copyFailure<BootstrapResult>(outputBudget);
      }
    }

    Index rejected = 0U;
    const Scalar alpha = 1.0 - policy.confidence;
    const Scalar lowerProbability = alpha * 0.5;
    const Scalar upperProbability = 1.0 - lowerProbability;
    for (Index draw = 0U; draw < policy.draws; ++draw) {
      const Status fill =
          fillReplicate(sample, policy.kind, blockLength, rng, replicate);
      if (!fill.ok()) {
        return copyFailure<BootstrapResult>(fill);
      }
      const SampleView replicateView{.rows = std::as_const(replicate).view(),
                                     .observation_shape =
                                         sample.observation_shape,
                                     .valid = {}};
      auto evaluated = evaluateStatistic(statistic, replicateView);
      if (!evaluated.hasValue()) {
        const ErrorCode code = evaluated.status().code();
        if (policy.invalid == InvalidDraw::report_conditional &&
            conditionallyOmittable(code)) {
          ++rejected;
          continue;
        }
        return copyFailure<BootstrapResult>(evaluated.status());
      }
      Vector estimate = std::move(evaluated).value();
      if (estimate.empty()) {
        return failure<BootstrapResult>(ErrorCode::InvalidExtent,
                                        "bootstrap statistic is empty");
      }
      if (width == 0U) {
        width = estimate.size();
        const Status outputBudget =
            preflightOutput(sampleElements, policy.draws, width, limits);
        if (!outputBudget.ok()) {
          return copyFailure<BootstrapResult>(outputBudget);
        }
      } else if (estimate.size() != width) {
        return failure<BootstrapResult>(ErrorCode::DimensionMismatch,
                                        "bootstrap statistic width changed");
      }
      for (Index coordinate = 0U; coordinate < width; ++coordinate) {
        if (!std::isfinite(estimate.at(coordinate))) {
          if (policy.invalid == InvalidDraw::report_conditional) {
            ++rejected;
            estimate = Vector();
            break;
          }
          return failure<BootstrapResult>(ErrorCode::NonFiniteResult,
                                          "bootstrap statistic is non-finite");
        }
      }
      if (!estimate.empty()) {
        estimates.push_back(std::move(estimate));
      }
    }
    if (estimates.empty()) {
      return failure<BootstrapResult>(ErrorCode::InsufficientSamples,
                                      "bootstrap retained no valid replicates");
    }

    Matrix resultEstimates(estimates.size(), width, 0.0);
    for (Index row = 0U; row < estimates.size(); ++row) {
      for (Index coordinate = 0U; coordinate < width; ++coordinate) {
        resultEstimates.at(row, coordinate) = estimates[row].at(coordinate);
      }
    }
    Matrix intervals(width, 2U, 0.0);
    std::optional<Vector> monteCarloError;
    if (estimates.size() >= 2U) {
      monteCarloError.emplace(width, 0.0);
    }
    Vector coordinateValues(estimates.size(), 0.0);
    for (Index coordinate = 0U; coordinate < width; ++coordinate) {
      for (Index row = 0U; row < estimates.size(); ++row) {
        coordinateValues.at(row) = estimates[row].at(coordinate);
      }
      const auto lower = quantile(coordinateValues.view(), lowerProbability,
                                  QuantileRule::linear, limits);
      if (!lower.hasValue()) {
        return copyFailure<BootstrapResult>(lower.status());
      }
      const auto upper = quantile(coordinateValues.view(), upperProbability,
                                  QuantileRule::linear, limits);
      if (!upper.hasValue()) {
        return copyFailure<BootstrapResult>(upper.status());
      }
      Scalar lowerEndpoint = lower.value();
      Scalar upperEndpoint = upper.value();
      if (policy.interval == QuantileInterval::basic) {
        lowerEndpoint = std::fma(2.0, original->at(coordinate), -upper.value());
        upperEndpoint = std::fma(2.0, original->at(coordinate), -lower.value());
        if (!std::isfinite(lowerEndpoint) || !std::isfinite(upperEndpoint)) {
          return failure<BootstrapResult>(
              ErrorCode::NonFiniteResult,
              "reflected bootstrap interval is not representable");
        }
      }
      intervals.at(coordinate, 0U) = lowerEndpoint;
      intervals.at(coordinate, 1U) = upperEndpoint;
      if (monteCarloError.has_value()) {
        const auto error = standardError(estimates, coordinate);
        if (!error.hasValue()) {
          return copyFailure<BootstrapResult>(error.status());
        }
        monteCarloError->at(coordinate) = error.value();
      }
    }
    return Result<BootstrapResult>::success(
        BootstrapResult{.estimates = std::move(resultEstimates),
                        .intervals = std::move(intervals),
                        .rejected = rejected,
                        .monte_carlo_error = std::move(monteCarloError)});
  } catch (const std::bad_alloc&) {
    return failure<BootstrapResult>(ErrorCode::ResourceLimit,
                                    "bootstrap allocation failed");
  } catch (const std::length_error&) {
    return failure<BootstrapResult>(ErrorCode::SizeOverflow,
                                    "bootstrap allocation extent is invalid");
  }
}

} // namespace vss::math::statistics
