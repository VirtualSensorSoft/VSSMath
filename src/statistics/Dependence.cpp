#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Dependence.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

class CompensatedSum final {
public:
  void add(long double value) {
    const long double next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] long double value() const { return sum_ + correction_; }

private:
  long double sum_{};
  long double correction_{};
};

struct RankedValue final {
  Scalar value;
  Extent index;
};

Result<std::uint64_t> pairCount(Extent n) {
  if (n > std::numeric_limits<std::uint64_t>::max()) {
    return fail<std::uint64_t>(
        ErrorCode::SizeOverflow,
        "sample size cannot be represented for pair enumeration");
  }
  const auto count = static_cast<std::uint64_t>(n);
  if (count > 1U &&
      count > std::numeric_limits<std::uint64_t>::max() / (count - 1U)) {
    return fail<std::uint64_t>(ErrorCode::SizeOverflow,
                               "unordered pair count overflows");
  }
  return Result<std::uint64_t>::success(count * (count - 1U) / 2U);
}

Status validateVectorPair(ConstVectorView x, ConstVectorView y) {
  if (x.size() != y.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "dependence inputs must have equal lengths");
  }
  if (x.size() < 2U) {
    return Status::failure(
        ErrorCode::InsufficientSamples,
        "dependence requires at least two paired observations");
  }
  for (Extent i = 0U; i < x.size(); ++i) {
    if (!std::isfinite(x.at(i)) || !std::isfinite(y.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "dependence inputs must be finite");
    }
  }
  return Status::success();
}

Result<std::vector<long double>> averageRanks(ConstVectorView values,
                                              Limits limits) {
  const Extent n = values.size();
  if (n > limits.bytes / (sizeof(RankedValue) + sizeof(long double))) {
    return fail<std::vector<long double>>(
        ErrorCode::ResourceLimit, "rank workspace exceeds configured limits");
  }
  std::vector<RankedValue> sorted;
  sorted.reserve(n);
  for (Extent i = 0U; i < n; ++i) {
    sorted.push_back(RankedValue{.value = values.at(i), .index = i});
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const RankedValue& left, const RankedValue& right) {
              return left.value < right.value;
            });
  std::vector<long double> ranks(n, 0.0L);
  for (Extent begin = 0U; begin < n;) {
    Extent end = begin + 1U;
    while (end < n && sorted[end].value == sorted[begin].value) {
      ++end;
    }
    const long double average =
        (static_cast<long double>(begin + 1U) + static_cast<long double>(end)) /
        2.0L;
    for (Extent i = begin; i < end; ++i) {
      ranks[sorted[i].index] = average;
    }
    begin = end;
  }
  return Result<std::vector<long double>>::success(std::move(ranks));
}

Result<long double> pearsonRanks(const std::vector<long double>& x,
                                 const std::vector<long double>& y) {
  CompensatedSum xSum;
  CompensatedSum ySum;
  for (Extent i = 0U; i < x.size(); ++i) {
    xSum.add(x[i]);
    ySum.add(y[i]);
  }
  const auto n = static_cast<long double>(x.size());
  const long double xMean = xSum.value() / n;
  const long double yMean = ySum.value() / n;
  CompensatedSum cross;
  CompensatedSum xSquares;
  CompensatedSum ySquares;
  for (Extent i = 0U; i < x.size(); ++i) {
    const long double dx = x[i] - xMean;
    const long double dy = y[i] - yMean;
    cross.add(dx * dy);
    xSquares.add(dx * dx);
    ySquares.add(dy * dy);
  }
  if (!(xSquares.value() > 0.0L) || !(ySquares.value() > 0.0L)) {
    return fail<long double>(
        ErrorCode::InvalidDomain,
        "Spearman correlation requires positive rank variance");
  }
  return Result<long double>::success(
      cross.value() / std::sqrt(xSquares.value() * ySquares.value()));
}

bool shapeMatches(const SampleView& sample) {
  if (sample.observation_shape.empty()) {
    return sample.rows.columns() == 1U;
  }
  Extent product = 1U;
  for (const Index extent : sample.observation_shape) {
    if (extent == 0U || product > std::numeric_limits<Extent>::max() / extent) {
      return false;
    }
    product *= extent;
  }
  return product == sample.rows.columns();
}

Status validateSample(const SampleView& sample) {
  const Extent n = sample.rows.rows();
  const Extent d = sample.rows.columns();
  if (n < 2U || d == 0U) {
    return Status::failure(
        ErrorCode::InsufficientSamples,
        "distance correlation requires at least two nonempty observations");
  }
  if (!shapeMatches(sample)) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "sample shape metadata does not match its row width");
  }
  if (!sample.valid.empty() && (n > std::numeric_limits<Extent>::max() / d ||
                                sample.valid.size() != n * d)) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "sample validity mask must match every sample cell");
  }
  for (const std::uint8_t bit : sample.valid) {
    if (bit > 1U) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "sample validity mask entries must be zero or one");
    }
    if (bit == 0U) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "distance correlation requires complete observations");
    }
  }
  for (Extent i = 0U; i < n; ++i) {
    for (Extent j = 0U; j < d; ++j) {
      if (!std::isfinite(sample.rows.at(i, j))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "distance correlation inputs must be finite");
      }
    }
  }
  return Status::success();
}

Result<long double> euclideanDistance(const SampleView& sample, Extent left,
                                      Extent right) {
  long double norm = 0.0L;
  for (Extent j = 0U; j < sample.rows.columns(); ++j) {
    const long double difference =
        static_cast<long double>(sample.rows.at(left, j)) -
        static_cast<long double>(sample.rows.at(right, j));
    if (!std::isfinite(difference)) {
      return fail<long double>(ErrorCode::NonFiniteResult,
                               "pairwise coordinate difference overflows");
    }
    norm = std::hypot(norm, difference);
    if (!std::isfinite(norm)) {
      return fail<long double>(ErrorCode::NonFiniteResult,
                               "Euclidean distance is not representable");
    }
  }
  return Result<long double>::success(norm);
}

bool checkedWork(Extent n, Extent dimensions, Extent otherDimensions,
                 std::uint64_t& work) {
  const auto max = std::numeric_limits<std::uint64_t>::max();
  if (n > max || dimensions > max || otherDimensions > max ||
      dimensions > max - otherDimensions) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(n);
  const auto width = static_cast<std::uint64_t>(dimensions + otherDimensions);
  if (count != 0U && count > max / count) {
    return false;
  }
  const std::uint64_t cells = count * count;
  if (cells != 0U && width > max / cells) {
    return false;
  }
  const std::uint64_t allPairs = cells * width;
  const std::uint64_t halfPairs = (count * (count - 1U) / 2U) * width;
  if (allPairs > max - halfPairs) {
    return false;
  }
  work = allPairs + halfPairs;
  return true;
}

} // namespace

Result<Scalar> spearmanCorrelation(ConstVectorView x, ConstVectorView y,
                                   Limits limits) {
  const Status valid = validateVectorPair(x, y);
  if (!valid.ok()) {
    return fail<Scalar>(valid.code(), std::string(valid.message()));
  }
  if (limits.elements < 1U ||
      x.size() >
          limits.bytes / (2U * (sizeof(RankedValue) + sizeof(long double)))) {
    return fail<Scalar>(ErrorCode::ResourceLimit,
                        "Spearman rank workspace exceeds configured limits");
  }
  auto xRanks = averageRanks(x, limits);
  auto yRanks = averageRanks(y, limits);
  if (!xRanks || !yRanks) {
    const Status& status = !xRanks ? xRanks.status() : yRanks.status();
    return fail<Scalar>(status.code(), std::string(status.message()));
  }
  const auto correlation = pearsonRanks(xRanks.value(), yRanks.value());
  if (!correlation) {
    return fail<Scalar>(correlation.status().code(),
                        std::string(correlation.status().message()));
  }
  const long double value = correlation.value();
  if (!std::isfinite(value) || value < -1.0L || value > 1.0L) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "Spearman correlation is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

Result<Scalar> kendallTau(ConstVectorView x, ConstVectorView y, Limits limits) {
  const Status valid = validateVectorPair(x, y);
  if (!valid.ok()) {
    return fail<Scalar>(valid.code(), std::string(valid.message()));
  }
  if (limits.elements < 1U) {
    return fail<Scalar>(ErrorCode::ResourceLimit,
                        "Kendall result exceeds configured element limit");
  }
  auto pairs = pairCount(x.size());
  if (!pairs) {
    return fail<Scalar>(pairs.status().code(),
                        std::string(pairs.status().message()));
  }
  if (pairs.value() > limits.evaluations) {
    return fail<Scalar>(
        ErrorCode::ResourceLimit,
        "Kendall pair comparisons exceed configured evaluation limit");
  }
  std::uint64_t concordant = 0U;
  std::uint64_t discordant = 0U;
  std::uint64_t tiedOnlyX = 0U;
  std::uint64_t tiedOnlyY = 0U;
  for (Extent i = 0U; i < x.size(); ++i) {
    for (Extent j = i + 1U; j < x.size(); ++j) {
      const bool tieX = x.at(i) == x.at(j);
      const bool tieY = y.at(i) == y.at(j);
      if (tieX && !tieY) {
        ++tiedOnlyX;
      } else if (!tieX && tieY) {
        ++tiedOnlyY;
      } else if (!tieX) {
        const bool sameSign = (x.at(i) < x.at(j)) == (y.at(i) < y.at(j));
        if (sameSign) {
          ++concordant;
        } else {
          ++discordant;
        }
      }
    }
  }
  const long double numerator = static_cast<long double>(concordant) -
                                static_cast<long double>(discordant);
  const long double comparable = static_cast<long double>(concordant) +
                                 static_cast<long double>(discordant);
  const long double factorX = comparable + static_cast<long double>(tiedOnlyX);
  const long double factorY = comparable + static_cast<long double>(tiedOnlyY);
  if (!(factorX > 0.0L) || !(factorY > 0.0L)) {
    return fail<Scalar>(ErrorCode::InvalidDomain,
                        "Kendall tau-b is undefined for a zero denominator");
  }
  const long double value = numerator / std::sqrt(factorX * factorY);
  if (!std::isfinite(value) || value < -1.0L || value > 1.0L) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "Kendall tau-b is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

Result<Scalar> distanceCorrelation(const SampleView& x, const SampleView& y,
                                   Limits limits) {
  const Status xStatus = validateSample(x);
  const Status yStatus = validateSample(y);
  if (!xStatus.ok()) {
    return fail<Scalar>(xStatus.code(), std::string(xStatus.message()));
  }
  if (!yStatus.ok()) {
    return fail<Scalar>(yStatus.code(), std::string(yStatus.message()));
  }
  const Extent n = x.rows.rows();
  if (n != y.rows.rows()) {
    return fail<Scalar>(
        ErrorCode::DimensionMismatch,
        "distance correlation populations must have equal row counts");
  }
  if (limits.elements < 1U) {
    return fail<Scalar>(
        ErrorCode::ResourceLimit,
        "distance-correlation result exceeds configured element limit");
  }
  constexpr std::size_t workspaceBytesPerRow =
      2U * sizeof(CompensatedSum) + 2U * sizeof(long double);
  if (n > limits.bytes / workspaceBytesPerRow) {
    return fail<Scalar>(
        ErrorCode::ResourceLimit,
        "distance correlation row workspace exceeds configured limits");
  }
  std::uint64_t requiredWork = 0U;
  if (!checkedWork(n, x.rows.columns(), y.rows.columns(), requiredWork)) {
    return fail<Scalar>(ErrorCode::SizeOverflow,
                        "distance correlation evaluation count overflows");
  }
  if (requiredWork > limits.evaluations) {
    return fail<Scalar>(
        ErrorCode::ResourceLimit,
        "distance correlation exceeds configured evaluation limit");
  }
  std::vector<CompensatedSum> xRows(n);
  std::vector<CompensatedSum> yRows(n);
  CompensatedSum xGrandSum;
  CompensatedSum yGrandSum;
  for (Extent i = 0U; i < n; ++i) {
    for (Extent j = i + 1U; j < n; ++j) {
      auto xDistance = euclideanDistance(x, i, j);
      auto yDistance = euclideanDistance(y, i, j);
      if (!xDistance || !yDistance) {
        const Status& status =
            !xDistance ? xDistance.status() : yDistance.status();
        return fail<Scalar>(status.code(), std::string(status.message()));
      }
      xRows[i].add(xDistance.value());
      xRows[j].add(xDistance.value());
      yRows[i].add(yDistance.value());
      yRows[j].add(yDistance.value());
      xGrandSum.add(2.0L * xDistance.value());
      yGrandSum.add(2.0L * yDistance.value());
    }
  }
  const auto denominator = static_cast<long double>(n);
  std::vector<long double> xMeans(n);
  std::vector<long double> yMeans(n);
  for (Extent i = 0U; i < n; ++i) {
    xMeans[i] = xRows[i].value() / denominator;
    yMeans[i] = yRows[i].value() / denominator;
  }
  const long double xGrand = xGrandSum.value() / (denominator * denominator);
  const long double yGrand = yGrandSum.value() / (denominator * denominator);
  CompensatedSum cross;
  CompensatedSum xSquares;
  CompensatedSum ySquares;
  CompensatedSum absoluteCross;
  for (Extent i = 0U; i < n; ++i) {
    for (Extent j = 0U; j < n; ++j) {
      auto xDistance = euclideanDistance(x, i, j);
      auto yDistance = euclideanDistance(y, i, j);
      if (!xDistance || !yDistance) {
        const Status& status =
            !xDistance ? xDistance.status() : yDistance.status();
        return fail<Scalar>(status.code(), std::string(status.message()));
      }
      const long double a = xDistance.value() - xMeans[i] - xMeans[j] + xGrand;
      const long double b = yDistance.value() - yMeans[i] - yMeans[j] + yGrand;
      cross.add(a * b);
      xSquares.add(a * a);
      ySquares.add(b * b);
      absoluteCross.add(std::abs(a * b));
    }
  }
  const long double scale = denominator * denominator;
  long double covarianceSquared = cross.value() / scale;
  const long double varianceXSquared = xSquares.value() / scale;
  const long double varianceYSquared = ySquares.value() / scale;
  const long double roundoff = 128.0L *
                               std::numeric_limits<long double>::epsilon() *
                               absoluteCross.value() / scale;
  if (covarianceSquared < 0.0L) {
    if (covarianceSquared < -roundoff) {
      return fail<Scalar>(ErrorCode::NonFiniteResult,
                          "distance covariance squared is materially negative");
    }
    covarianceSquared = 0.0L;
  }
  if (!(varianceXSquared > 0.0L) || !(varianceYSquared > 0.0L) ||
      !std::isfinite(varianceXSquared) || !std::isfinite(varianceYSquared)) {
    return fail<Scalar>(
        ErrorCode::InvalidDomain,
        "distance correlation is undefined for a constant population");
  }
  long double squaredCorrelation =
      (covarianceSquared / std::sqrt(varianceXSquared)) /
      std::sqrt(varianceYSquared);
  if (!std::isfinite(squaredCorrelation) || squaredCorrelation < 0.0L) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "distance correlation is not representable");
  }
  if (squaredCorrelation > 1.0L) {
    const long double correlationRoundoff =
        roundoff / std::sqrt(varianceXSquared) / std::sqrt(varianceYSquared) +
        128.0L * std::numeric_limits<long double>::epsilon() *
            (std::abs(squaredCorrelation) + 1.0L);
    if (squaredCorrelation > 1.0L + correlationRoundoff) {
      return fail<Scalar>(
          ErrorCode::NonFiniteResult,
          "distance correlation exceeds one beyond roundoff tolerance");
    }
    squaredCorrelation = 1.0L;
  }
  const long double value = std::sqrt(squaredCorrelation);
  if (!std::isfinite(value) || value > std::numeric_limits<Scalar>::max()) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "distance correlation is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

} // namespace vss::math::statistics
