#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/signal/TemporalUtilities.hpp>

namespace vss::math::signal {
namespace {

using Wide = long double;

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedAdd(Index left, Index right, Index& result) noexcept {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(Index left, Index right, Index& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

// A binary64 value is an integer multiple of 2^-1074. The largest possible
// sum of size_t-many values needs at most 2162 bits on a 64-bit platform.
constexpr std::size_t dyadicWords =
    (2098U + static_cast<std::size_t>(std::numeric_limits<Index>::digits) +
     31U) /
    32U;
using Magnitude = std::array<std::uint32_t, dyadicWords>;

static_assert(std::numeric_limits<Index>::digits <= 64,
              "dyadic prefix sums support at most 64-bit Index");
static_assert(sizeof(Scalar) == sizeof(std::uint64_t) &&
                  std::numeric_limits<Scalar>::is_iec559 &&
                  std::numeric_limits<Scalar>::digits == 53,
              "dyadic prefix sums require binary64 Scalar");

bool addWord(Magnitude& magnitude, std::size_t position,
             std::uint32_t word) noexcept {
  while (word != 0U) {
    if (position >= magnitude.size()) {
      return false;
    }
    const std::uint64_t total =
        static_cast<std::uint64_t>(magnitude[position]) + word;
    magnitude[position] = static_cast<std::uint32_t>(total);
    word = static_cast<std::uint32_t>(total >> 32U);
    ++position;
  }
  return true;
}

bool addShifted(Magnitude& magnitude, std::uint64_t significand,
                std::size_t shift) noexcept {
  const std::size_t base = shift / 32U;
  const auto offset = static_cast<unsigned>(shift % 32U);
  const std::array<std::uint32_t, 2U> words{
      static_cast<std::uint32_t>(significand),
      static_cast<std::uint32_t>(significand >> 32U)};
  for (std::size_t i = 0U; i < words.size(); ++i) {
    if (words[i] == 0U) {
      continue;
    }
    const std::uint32_t low =
        offset == 0U ? words[i]
                     : static_cast<std::uint32_t>(words[i] << offset);
    if (!addWord(magnitude, base + i, low)) {
      return false;
    }
    if (offset != 0U &&
        !addWord(magnitude, base + i + 1U,
                 static_cast<std::uint32_t>(words[i] >> (32U - offset)))) {
      return false;
    }
  }
  return true;
}

int compare(const Magnitude& left, const Magnitude& right) noexcept {
  for (std::size_t i = left.size(); i > 0U; --i) {
    if (left[i - 1U] != right[i - 1U]) {
      return left[i - 1U] > right[i - 1U] ? 1 : -1;
    }
  }
  return 0;
}

bool subtract(const Magnitude& larger, const Magnitude& smaller,
              Magnitude& result) noexcept {
  std::uint64_t borrow = 0U;
  for (std::size_t i = 0U; i < larger.size(); ++i) {
    const std::uint64_t a = larger[i];
    const std::uint64_t b = static_cast<std::uint64_t>(smaller[i]) + borrow;
    result[i] = static_cast<std::uint32_t>(a - b);
    borrow = a < b ? 1U : 0U;
  }
  return borrow == 0U;
}

bool bit(const Magnitude& magnitude, std::size_t position) noexcept {
  return ((magnitude[position / 32U] >> (position % 32U)) & 1U) != 0U;
}

bool anyBelow(const Magnitude& magnitude, std::size_t position) noexcept {
  const std::size_t complete = position / 32U;
  for (std::size_t i = 0U; i < complete; ++i) {
    if (magnitude[i] != 0U) {
      return true;
    }
  }
  const auto remainder = static_cast<unsigned>(position % 32U);
  if (remainder == 0U) {
    return false;
  }
  const std::uint32_t mask = (std::uint32_t{1U} << remainder) - 1U;
  return (magnitude[complete] & mask) != 0U;
}

std::size_t highestBit(const Magnitude& magnitude) noexcept {
  for (std::size_t i = magnitude.size(); i > 0U; --i) {
    if (magnitude[i - 1U] != 0U) {
      return (i - 1U) * 32U +
             static_cast<std::size_t>(31 - std::countl_zero(magnitude[i - 1U]));
    }
  }
  return 0U;
}

class ExactSum final {
public:
  bool add(Scalar value) noexcept {
    if (value == 0.0) {
      return true;
    }
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent = static_cast<unsigned>((bits >> 52U) & 0x7ffU);
    std::uint64_t significand = bits & ((std::uint64_t{1U} << 52U) - 1U);
    std::size_t shift = 0U;
    if (exponent != 0U) {
      significand |= std::uint64_t{1U} << 52U;
      shift = static_cast<std::size_t>(exponent - 1U);
    }
    return addShifted((bits >> 63U) == 0U ? positive_ : negative_, significand,
                      shift);
  }

  [[nodiscard]] int sign() const noexcept {
    return compare(positive_, negative_);
  }

  [[nodiscard]] bool toScalar(Scalar& result) const noexcept {
    Magnitude magnitude{};
    const int ordering = sign();
    if (ordering == 0) {
      result = 0.0;
      return true;
    }
    if (!subtract(ordering > 0 ? positive_ : negative_,
                  ordering > 0 ? negative_ : positive_, magnitude)) {
      return false;
    }
    const std::size_t top = highestBit(magnitude);
    std::uint64_t significand = 0U;
    std::uint64_t encoded = 0U;
    if (top < 52U) {
      significand = magnitude[0];
    } else {
      const std::size_t shift = top - 52U;
      for (std::size_t position = top + 1U; position > shift; --position) {
        significand = (significand << 1U) |
                      static_cast<std::uint64_t>(bit(magnitude, position - 1U));
      }
      if (shift != 0U) {
        const std::size_t guardPosition = shift - 1U;
        const bool guard = bit(magnitude, guardPosition);
        const bool sticky = anyBelow(magnitude, guardPosition);
        if (guard && (sticky || (significand & 1U) != 0U)) {
          ++significand;
        }
      }
      std::size_t roundedTop = top;
      if (significand == (std::uint64_t{1U} << 53U)) {
        significand >>= 1U;
        ++roundedTop;
      }
      const std::size_t exponent = roundedTop - 51U;
      if (exponent >= 0x7ffU) {
        return false;
      }
      encoded = static_cast<std::uint64_t>(exponent) << 52U;
      encoded |= significand & ((std::uint64_t{1U} << 52U) - 1U);
    }
    if (ordering < 0) {
      encoded |= std::uint64_t{1U} << 63U;
    }
    if (top < 52U) {
      encoded |= significand;
    }
    result = std::bit_cast<Scalar>(encoded);
    return std::isfinite(result);
  }

  [[nodiscard]] bool scaled(Wide& fraction, int& exponent) const noexcept {
    Magnitude magnitude{};
    const int ordering = sign();
    if (ordering == 0 ||
        !subtract(ordering > 0 ? positive_ : negative_,
                  ordering > 0 ? negative_ : positive_, magnitude)) {
      fraction = 0.0L;
      exponent = 0;
      return ordering == 0;
    }
    const std::size_t top = highestBit(magnitude);
    const auto retained = static_cast<unsigned>(std::min<std::size_t>(
        top + 1U, static_cast<std::size_t>(
                      std::min(std::numeric_limits<Wide>::digits, 64))));
    std::uint64_t leading = 0U;
    for (std::size_t position = top + 1U; position > top + 1U - retained;
         --position) {
      leading = (leading << 1U) |
                static_cast<std::uint64_t>(bit(magnitude, position - 1U));
    }
    fraction =
        std::scalbn(static_cast<Wide>(leading), -static_cast<int>(retained));
    if (ordering < 0) {
      fraction = -fraction;
    }
    exponent = static_cast<int>(top) - 1073;
    return std::isfinite(fraction);
  }

private:
  Magnitude positive_{};
  Magnitude negative_{};
};

class CompensatedSum final {
public:
  void add(Wide value) noexcept {
    const Wide next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }
  [[nodiscard]] Wide value() const noexcept { return sum_ + correction_; }

private:
  Wide sum_{};
  Wide correction_{};
};

bool validExtrapolation(Extrapolation value) noexcept {
  return value == Extrapolation::reject || value == Extrapolation::hold ||
         value == Extrapolation::linear;
}

bool representable(Wide value, Scalar& result) noexcept {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
    return false;
  }
  result = static_cast<Scalar>(value);
  return std::isfinite(result);
}

bool interpolationFraction(Scalar value, Scalar left, Scalar right,
                           Wide& fraction) noexcept {
  Wide numerator = static_cast<Wide>(value) - static_cast<Wide>(left);
  Wide denominator = static_cast<Wide>(right) - static_cast<Wide>(left);
  if (!std::isfinite(numerator) || !std::isfinite(denominator)) {
    numerator =
        static_cast<Wide>(value) * 0.5L - static_cast<Wide>(left) * 0.5L;
    denominator =
        static_cast<Wide>(right) * 0.5L - static_cast<Wide>(left) * 0.5L;
  }
  if (!std::isfinite(numerator) || !std::isfinite(denominator) ||
      denominator <= 0.0L) {
    return false;
  }
  fraction = numerator / denominator;
  return std::isfinite(fraction);
}

ErrorCode validateSource(const SeriesView& source) noexcept {
  const Index count = source.times().size();
  if (source.rows().rows() != count || source.rows().columns() == 0U ||
      (!source.valid().empty() && source.valid().size() != count)) {
    return ErrorCode::DimensionMismatch;
  }
  for (Index row = 0U; row < count; ++row) {
    const Scalar time = source.times().at(row);
    if (!std::isfinite(time)) {
      return ErrorCode::NonFiniteInput;
    }
    if (row > 0U && !(time > source.times().at(row - 1U))) {
      return ErrorCode::InvalidDomain;
    }
    const bool active = source.valid().empty() || source.valid()[row] == 1U;
    if (!source.valid().empty() && source.valid()[row] > 1U) {
      return ErrorCode::InvalidDomain;
    }
    if (active) {
      for (Index column = 0U; column < source.rows().columns(); ++column) {
        if (!std::isfinite(source.rows().at(row, column))) {
          return ErrorCode::NonFiniteInput;
        }
      }
    }
  }
  return ErrorCode::None;
}

bool copyKnot(const SeriesView& source, Index row, Matrix& output,
              Index targetRow) {
  if (!source.valid().empty() && source.valid()[row] == 0U) {
    return false;
  }
  for (Index column = 0U; column < source.rows().columns(); ++column) {
    output.at(targetRow, column) = source.rows().at(row, column);
  }
  return true;
}

} // namespace

Result<Vector> cumulativeSum(ConstVectorView values, Limits limits) {
  const Index count = values.size();
  if (count > limits.elements || count > limits.bytes / sizeof(Scalar)) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative sum output exceeds configured limits");
  }
  for (Index i = 0U; i < count; ++i) {
    if (!std::isfinite(values.at(i))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "cumulative sum input must be finite");
    }
  }
  try {
    Vector result(count);
    ExactSum exact;
    CompensatedSum compensated;
    for (Index i = 0U; i < count; ++i) {
      const Scalar value = values.at(i);
      if (!exact.add(value)) {
        return failure<Vector>(ErrorCode::NonFiniteResult,
                               "exact cumulative sum capacity was exceeded");
      }
      compensated.add(static_cast<Wide>(value));
      Scalar rounded = 0.0;
      if (!exact.toScalar(rounded)) {
        return failure<Vector>(ErrorCode::NonFiniteResult,
                               "a cumulative prefix is not representable");
      }
      if (rounded == 0.0) {
        result.at(i) = 0.0;
        continue;
      }
      const Wide estimate = compensated.value();
      if (std::isfinite(estimate) &&
          std::abs(estimate) <=
              static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
        const Scalar fast = static_cast<Scalar>(estimate);
        result.at(i) = fast == rounded ? fast : rounded;
      } else {
        result.at(i) = rounded;
      }
    }
    return Result<Vector>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative sum allocation failed");
  } catch (const std::length_error&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "cumulative sum extent is not allocatable");
  }
}

Result<LagSummary> summarizeLags(const LagSeries& values, Scalar sampleStep,
                                 Index cutoff, Limits limits) {
  const auto lags = values.lags();
  const auto correlations = values.values();
  const auto counts = values.counts();
  if (lags.size() != correlations.size() || lags.size() != counts.size()) {
    return failure<LagSummary>(ErrorCode::DimensionMismatch,
                               "lag, value and count extents must match");
  }
  if (!std::isfinite(sampleStep)) {
    return failure<LagSummary>(ErrorCode::NonFiniteInput,
                               "sample step must be finite");
  }
  if (sampleStep <= 0.0) {
    return failure<LagSummary>(ErrorCode::InvalidDomain,
                               "sample step must be positive");
  }
  if (lags.empty()) {
    return failure<LagSummary>(ErrorCode::InsufficientSamples,
                               "lag summary requires at least one entry");
  }
  Index slots = 0U;
  Index chargedElements = 0U;
  Index scalarBytes = 0U;
  Index flagBytes = 0U;
  Index totalBytes = 0U;
  if (!checkedAdd(cutoff, 1U, slots) ||
      !checkedMultiply(slots, 2U, chargedElements) ||
      !checkedMultiply(slots, sizeof(Scalar), scalarBytes) ||
      !checkedAdd(scalarBytes, slots, flagBytes) ||
      !checkedAdd(flagBytes, sizeof(LagSummary), totalBytes) ||
      chargedElements > limits.elements || totalBytes > limits.bytes) {
    return failure<LagSummary>(ErrorCode::ResourceLimit,
                               "lag summary workspace exceeds limits");
  }
  if (cutoff > static_cast<Index>(std::numeric_limits<std::int64_t>::max())) {
    return failure<LagSummary>(
        ErrorCode::InsufficientSamples,
        "required positive lag range is not representable");
  }
  for (Index i = 0U; i < counts.size(); ++i) {
    if (!std::isfinite(correlations.at(i))) {
      return failure<LagSummary>(ErrorCode::NonFiniteInput,
                                 "lag association contains a non-finite value");
    }
    if (counts[i] == 0U) {
      return failure<LagSummary>(ErrorCode::InvalidDomain,
                                 "lag counts must be positive");
    }
  }
  try {
    std::vector<Scalar> required(slots, 0.0);
    std::vector<std::uint8_t> seen(slots, 0U);
    bool haveZero = false;
    Index zeroCount = 0U;
    std::int64_t peakLag = lags.front();
    Scalar peakValue = correlations.at(0U);
    Scalar peakMagnitude = std::abs(peakValue);
    for (Index i = 0U; i < lags.size(); ++i) {
      const std::int64_t lag = lags[i];
      const Scalar association = correlations.at(i);
      const Scalar magnitude = std::abs(association);
      if (magnitude > peakMagnitude ||
          (magnitude == peakMagnitude && lag < peakLag)) {
        peakLag = lag;
        peakValue = association;
        peakMagnitude = magnitude;
      }
      if (lag == 0) {
        haveZero = true;
        ++zeroCount;
        if (association != 1.0) {
          return failure<LagSummary>(ErrorCode::InvalidDomain,
                                     "zero-lag autocorrelation must equal one");
        }
      } else if (lag > 0 && static_cast<std::uint64_t>(lag) <= cutoff) {
        const Index slot = static_cast<Index>(lag);
        if (seen[slot] != 0U) {
          return failure<LagSummary>(ErrorCode::InvalidDomain,
                                     "required positive lags must be unique");
        }
        seen[slot] = 1U;
        required[slot] = association;
      }
    }
    if (!haveZero || zeroCount != 1U) {
      return failure<LagSummary>(ErrorCode::InvalidDomain,
                                 "exactly one zero lag is required");
    }
    for (Index lag = 1U; lag <= cutoff; ++lag) {
      if (seen[lag] == 0U) {
        return failure<LagSummary>(ErrorCode::InsufficientSamples,
                                   "a required positive lag is missing");
      }
    }
    ExactSum exact;
    static_cast<void>(exact.add(1.0));
    CompensatedSum compensated;
    compensated.add(1.0L);
    for (Index lag = 1U; lag <= cutoff; ++lag) {
      if (!exact.add(required[lag]) || !exact.add(required[lag])) {
        return failure<LagSummary>(
            ErrorCode::NonFiniteResult,
            "exact correlation-time sum capacity exceeded");
      }
      compensated.add(static_cast<Wide>(required[lag]));
      compensated.add(static_cast<Wide>(required[lag]));
    }
    if (exact.sign() < 0) {
      return failure<LagSummary>(ErrorCode::InvalidDomain,
                                 "truncated correlation time is negative");
    }
    Scalar time = 0.0;
    if (exact.sign() == 0) {
      time = 0.0;
    } else {
      Wide fraction = 0.0L;
      int exponent = 0;
      if (!exact.scaled(fraction, exponent)) {
        return failure<LagSummary>(ErrorCode::NonFiniteResult,
                                   "correlation-time sum cannot be scaled");
      }
      int stepExponent = 0;
      const Wide stepFraction =
          std::frexp(static_cast<Wide>(sampleStep), &stepExponent);
      int adjustment = 0;
      const Wide productFraction =
          std::frexp(fraction * stepFraction, &adjustment);
      const long long productExponent =
          static_cast<long long>(exponent) + stepExponent + adjustment;
      if (productExponent > std::numeric_limits<int>::max() ||
          productExponent < std::numeric_limits<int>::min()) {
        return failure<LagSummary>(ErrorCode::NonFiniteResult,
                                   "correlation time is not representable");
      }
      const Wide product =
          std::scalbn(productFraction, static_cast<int>(productExponent));
      if (!representable(product, time)) {
        return failure<LagSummary>(ErrorCode::NonFiniteResult,
                                   "correlation time is not representable");
      }
    }
    const Wide estimate = compensated.value();
    if (std::isfinite(estimate) && estimate >= 0.0L) {
      const Wide fastTime = estimate * static_cast<Wide>(sampleStep);
      Scalar fastRounded = 0.0;
      if (representable(fastTime, fastRounded) && fastRounded == time) {
        time = fastRounded;
      }
    }
    return Result<LagSummary>::success(
        LagSummary{peakLag, peakValue, time, cutoff});
  } catch (const std::bad_alloc&) {
    return failure<LagSummary>(ErrorCode::ResourceLimit,
                               "lag summary allocation failed");
  } catch (const std::length_error&) {
    return failure<LagSummary>(ErrorCode::ResourceLimit,
                               "lag summary extent is not allocatable");
  }
}

Result<AlignedSeries> alignSeries(std::span<const SeriesView> inputs,
                                  ConstVectorView target,
                                  const InterpolateSpec& policy,
                                  Limits limits) {
  if (!validExtrapolation(policy.extrapolation)) {
    return failure<AlignedSeries>(ErrorCode::InvalidDomain,
                                  "unknown interpolation extrapolation policy");
  }
  if (policy.maximum_gap.has_value() &&
      (!std::isfinite(*policy.maximum_gap) || *policy.maximum_gap <= 0.0)) {
    return failure<AlignedSeries>(ErrorCode::InvalidTolerance,
                                  "maximum interpolation gap must be positive");
  }
  for (Index row = 0U; row < target.size(); ++row) {
    if (!std::isfinite(target.at(row))) {
      return failure<AlignedSeries>(ErrorCode::NonFiniteInput,
                                    "target time contains a non-finite value");
    }
    if (row > 0U && !(target.at(row) > target.at(row - 1U))) {
      return failure<AlignedSeries>(ErrorCode::InvalidDomain,
                                    "target times must be strictly increasing");
    }
  }
  for (const SeriesView& source : inputs) {
    const ErrorCode status = validateSource(source);
    if (status != ErrorCode::None) {
      return failure<AlignedSeries>(status,
                                    "source series has invalid shape or data");
    }
  }

  const Index targetCount = target.size();
  Index elements = targetCount;
  Index validityBytes = 0U;
  Index sourceBytes = 0U;
  Index cursorBytes = inputs.empty() ? 0U : sizeof(Index);
  for (const SeriesView& source : inputs) {
    Index cells = 0U;
    Index timeAndData = 0U;
    if (!checkedMultiply(targetCount, source.rows().columns(), cells) ||
        !checkedAdd(targetCount, cells, timeAndData) ||
        !checkedAdd(elements, timeAndData, elements) ||
        !checkedAdd(validityBytes, targetCount, validityBytes) ||
        !checkedAdd(sourceBytes, sizeof(AlignedSourceSeries), sourceBytes)) {
      return failure<AlignedSeries>(
          ErrorCode::ResourceLimit,
          "aligned series size arithmetic overflowed");
    }
  }
  Index outputBytes = 0U;
  Index allBytes = 0U;
  if (!checkedMultiply(elements, sizeof(Scalar), outputBytes) ||
      !checkedAdd(outputBytes, validityBytes, allBytes) ||
      !checkedAdd(allBytes, sourceBytes, allBytes) ||
      !checkedAdd(allBytes, cursorBytes, allBytes) ||
      !checkedAdd(allBytes, sizeof(AlignedSeries) + sizeof(Vector), allBytes) ||
      elements > limits.elements || allBytes > limits.bytes) {
    return failure<AlignedSeries>(
        ErrorCode::ResourceLimit,
        "aligned series output or workspace exceeds limits");
  }
  try {
    Vector targetCopy(targetCount);
    for (Index row = 0U; row < targetCount; ++row) {
      targetCopy.at(row) = target.at(row);
    }
    std::vector<AlignedSourceSeries> outputs;
    outputs.reserve(inputs.size());
    for (const SeriesView& source : inputs) {
      const Index sourceCount = source.times().size();
      const Index channels = source.rows().columns();
      Vector times(targetCount);
      Matrix rows(targetCount, channels, 0.0);
      std::vector<std::uint8_t> valid(targetCount, 0U);
      Index cursor = 0U;
      for (Index targetRow = 0U; targetRow < targetCount; ++targetRow) {
        const Scalar query = target.at(targetRow);
        times.at(targetRow) = query;
        if (sourceCount == 0U) {
          continue;
        }
        bool exterior = false;
        Index left = 0U;
        Index right = 0U;
        if (query < source.times().at(0U)) {
          exterior = true;
          if (policy.extrapolation == Extrapolation::reject) {
            continue;
          }
          if (policy.extrapolation == Extrapolation::hold) {
            if (copyKnot(source, 0U, rows, targetRow)) {
              valid[targetRow] = 1U;
            }
            continue;
          }
          if (sourceCount < 2U) {
            continue;
          }
          left = 0U;
          right = 1U;
        } else if (query > source.times().at(sourceCount - 1U)) {
          exterior = true;
          if (policy.extrapolation == Extrapolation::reject) {
            continue;
          }
          if (policy.extrapolation == Extrapolation::hold) {
            if (copyKnot(source, sourceCount - 1U, rows, targetRow)) {
              valid[targetRow] = 1U;
            }
            continue;
          }
          if (sourceCount < 2U) {
            continue;
          }
          right = sourceCount - 1U;
          left = right - 1U;
        } else {
          while (cursor + 1U < sourceCount &&
                 source.times().at(cursor + 1U) < query) {
            ++cursor;
          }
          if (source.times().at(cursor) == query) {
            if (copyKnot(source, cursor, rows, targetRow)) {
              valid[targetRow] = 1U;
            }
            continue;
          }
          if (cursor + 1U < sourceCount &&
              source.times().at(cursor + 1U) == query) {
            ++cursor;
            if (copyKnot(source, cursor, rows, targetRow)) {
              valid[targetRow] = 1U;
            }
            continue;
          }
          if (cursor + 1U >= sourceCount) {
            continue;
          }
          left = cursor;
          right = cursor + 1U;
        }
        if ((!source.valid().empty() &&
             (source.valid()[left] == 0U || source.valid()[right] == 0U)) ||
            (policy.maximum_gap.has_value() &&
             static_cast<Wide>(source.times().at(right)) -
                     static_cast<Wide>(source.times().at(left)) >
                 static_cast<Wide>(*policy.maximum_gap))) {
          continue;
        }
        Wide fraction = 0.0L;
        if (!interpolationFraction(query, source.times().at(left),
                                   source.times().at(right), fraction)) {
          return failure<AlignedSeries>(
              ErrorCode::NonFiniteResult,
              "interpolation coordinate is not representable");
        }
        for (Index channel = 0U; channel < channels; ++channel) {
          const Scalar low = source.rows().at(left, channel);
          const Scalar high = source.rows().at(right, channel);
          Wide interpolated = 0.0L;
          if (!exterior && fraction >= 0.0L && fraction <= 1.0L) {
            interpolated = static_cast<Wide>(
                std::lerp(low, high, static_cast<Scalar>(fraction)));
          } else {
            interpolated =
                static_cast<Wide>(low) +
                fraction * (static_cast<Wide>(high) - static_cast<Wide>(low));
          }
          Scalar value = 0.0;
          if (!representable(interpolated, value)) {
            return failure<AlignedSeries>(
                ErrorCode::NonFiniteResult,
                "interpolated value is not representable");
          }
          rows.at(targetRow, channel) = value;
        }
        valid[targetRow] = 1U;
      }
      outputs.emplace_back(AlignedSourceSeries(
          std::move(times), std::move(rows), std::move(valid)));
    }
    return Result<AlignedSeries>::success(
        AlignedSeries(std::move(targetCopy), std::move(outputs)));
  } catch (const std::bad_alloc&) {
    return failure<AlignedSeries>(ErrorCode::ResourceLimit,
                                  "aligned series allocation failed");
  } catch (const std::length_error&) {
    return failure<AlignedSeries>(ErrorCode::ResourceLimit,
                                  "aligned series extent is not allocatable");
  }
}

} // namespace vss::math::signal
