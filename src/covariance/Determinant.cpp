#include "DeterminantInteger.hpp"
#include "linear/DenseWorkspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {
namespace {

using detail::DeterminantInteger;
using linear::detail::StorageBudget;
using linear::detail::Wide;
using Exponent = std::int64_t;

constexpr Scalar scalarEpsilon = std::numeric_limits<Scalar>::epsilon();
constexpr int precision = std::numeric_limits<Scalar>::digits;
constexpr int minimumQuantum =
    std::numeric_limits<Scalar>::min_exponent - precision;
constexpr int maximumExponent = std::numeric_limits<Scalar>::max_exponent;
constexpr Wide infinity = std::numeric_limits<Wide>::infinity();

enum class Output : std::uint8_t { determinant, logMagnitude };

struct Bounds final {
  Wide lower{};
  Wide upper{};
};

bool finite(Bounds bounds) {
  return std::isfinite(bounds.lower) && std::isfinite(bounds.upper);
}

Wide center(Bounds bounds) { return std::midpoint(bounds.lower, bounds.upper); }

bool zero(Bounds bounds) { return bounds.lower == 0 && bounds.upper == 0; }

Bounds outward(Wide lower, Wide upper) {
  return {.lower = std::nextafter(lower, -infinity),
          .upper = std::nextafter(upper, infinity)};
}

Bounds product(Bounds left, Bounds right) {
  if (zero(left) || zero(right)) {
    return {};
  }
  if (left.lower == 1 && left.upper == 1) {
    return right;
  }
  if (right.lower == 1 && right.upper == 1) {
    return left;
  }
  const std::array products{left.lower * right.lower, left.lower * right.upper,
                            left.upper * right.lower, left.upper * right.upper};
  const auto [lower, upper] = std::ranges::minmax_element(products);
  return outward(*lower, *upper);
}

Bounds quotient(Bounds left, Bounds right) {
  if (zero(left)) {
    return {};
  }
  const std::array ratios{left.lower / right.lower, left.lower / right.upper,
                          left.upper / right.lower, left.upper / right.upper};
  const auto [lower, upper] = std::ranges::minmax_element(ratios);
  return outward(*lower, *upper);
}

Bounds difference(Bounds left, Bounds right) {
  if (zero(right)) {
    return left;
  }
  return outward(left.lower - right.upper, left.upper - right.lower);
}

struct ScaledDeterminant final {
  Bounds magnitude{.lower = 1, .upper = 1};
  Exponent exponent{};
  Exponent logExponent{};
  Wide logSum{};
  Wide logCorrection{};
  bool negative{};
};

bool consumePivot(ScaledDeterminant& result, Bounds pivot) {
  if (pivot.lower <= 0 && pivot.upper >= 0) {
    return false;
  }
  if (pivot.upper < 0) {
    result.negative = !result.negative;
    pivot = {.lower = -pivot.upper, .upper = -pivot.lower};
  }
  const int pivotExponent = std::ilogb(pivot.upper);
  const auto normalized =
      Bounds{.lower = std::scalbn(pivot.lower, -pivotExponent),
             .upper = std::scalbn(pivot.upper, -pivotExponent)};
  if (std::scalbn(normalized.lower, pivotExponent) != pivot.lower) {
    return false;
  }
  result.logExponent += pivotExponent;
  const Wide adjusted = std::log(center(normalized)) - result.logCorrection;
  const Wide next = result.logSum + adjusted;
  result.logCorrection = (next - result.logSum) - adjusted;
  result.logSum = next;
  const auto combined = product(result.magnitude, normalized);
  if (!finite(combined) || combined.lower <= 0) {
    return false;
  }
  const int shift = std::ilogb(combined.upper);
  const Wide lower = std::scalbn(combined.lower, -shift);
  if (std::scalbn(lower, shift) != combined.lower) {
    return false;
  }
  result.magnitude = {.lower = lower,
                      .upper = std::scalbn(combined.upper, -shift)};
  result.exponent += static_cast<Exponent>(pivotExponent) + shift;
  return true;
}

// Power-of-two row scaling keeps the input exact. Outward-rounded interval
// operations enclose each pivot, including when elimination loses digits.
std::optional<ScaledDeterminant> boundedLU(ConstMatrixView input) {
  const Index size = input.rows();
  std::vector<Bounds> work(size * size);
  const auto entry = [&work, size](Index rowIndex,
                                   Index columnIndex) -> Bounds& {
    return work.at((rowIndex * size) + columnIndex);
  };
  ScaledDeterminant result;
  for (Index row = 0; row < size; ++row) {
    Scalar maximum = 0;
    for (Index column = 0; column < size; ++column) {
      maximum = std::max(maximum, std::abs(input.at(row, column)));
    }
    if (maximum == 0) {
      return ScaledDeterminant{.magnitude = {}};
    }
    const int shift = std::ilogb(maximum);
    result.exponent += shift;
    result.logExponent += shift;
    for (Index column = 0; column < size; ++column) {
      const auto original = static_cast<Wide>(input.at(row, column));
      const Wide scaled = std::scalbn(original, -shift);
      if (std::scalbn(scaled, shift) != original) {
        return std::nullopt;
      }
      entry(row, column) = {.lower = scaled, .upper = scaled};
    }
  }
  for (Index pivotIndex = 0; pivotIndex < size; ++pivotIndex) {
    Index pivotRow = pivotIndex;
    for (Index row = pivotIndex + 1; row < size; ++row) {
      if (std::abs(center(entry(row, pivotIndex))) >
          std::abs(center(entry(pivotRow, pivotIndex)))) {
        pivotRow = row;
      }
    }
    if (pivotRow != pivotIndex) {
      for (Index trailingColumn = pivotIndex; trailingColumn < size;
           ++trailingColumn) {
        std::swap(entry(pivotIndex, trailingColumn),
                  entry(pivotRow, trailingColumn));
      }
      result.negative = !result.negative;
    }
    const auto pivot = entry(pivotIndex, pivotIndex);
    if (!consumePivot(result, pivot)) {
      return std::nullopt;
    }
    for (Index row = pivotIndex + 1; row < size; ++row) {
      if (zero(entry(row, pivotIndex))) {
        continue;
      }
      const auto factor = quotient(entry(row, pivotIndex), pivot);
      if (!finite(factor)) {
        return std::nullopt;
      }
      for (Index trailingColumn = pivotIndex + 1; trailingColumn < size;
           ++trailingColumn) {
        auto& cell = entry(row, trailingColumn);
        cell = difference(cell,
                          product(factor, entry(pivotIndex, trailingColumn)));
        if (!finite(cell)) {
          return std::nullopt;
        }
      }
    }
  }
  const auto bounds = result.magnitude;
  if (bounds.upper - bounds.lower >
      16 * static_cast<Wide>(size) * scalarEpsilon * bounds.lower) {
    return std::nullopt;
  }
  return result;
}

struct Dyadic final {
  std::uint64_t significand{};
  int exponent{};
};

Dyadic decompose(Scalar value) {
  if (value == 0) {
    return {};
  }
  int exponent = 0;
  const auto fraction = std::frexp(std::abs(value), &exponent);
  const auto bits = static_cast<std::uint64_t>(std::ldexp(fraction, precision));
  const int zeros = std::countr_zero(bits);
  return {.significand = bits >> zeros,
          .exponent = exponent - precision + zeros};
}

struct ExactDeterminant final {
  DeterminantInteger magnitude;
  Exponent exponent{};
};

std::optional<Index> minorBitBound(Index dimension, Index entryBits) {
  // Hadamard: each row norm of an order-k minor is bounded by
  // sqrt(k) times the largest integer entry. Use ceil(log2(d)) as a
  // conservative allowance, also covering rectangular trailing work.
  const Index bitsPerRow =
      entryBits + static_cast<Index>(std::bit_width(dimension));
  if (dimension > (std::numeric_limits<Index>::max() - 2) / 2 / bitsPerRow) {
    return std::nullopt;
  }
  return dimension * bitsPerRow;
}

Result<ExactDeterminant> exactDyadic(ConstMatrixView input, Limits limits) {
  const Index size = input.rows();
  int leastExponent = maximumExponent;
  int greatestBit = minimumQuantum;
  for (Index row = 0; row < size; ++row) {
    for (Index column = 0; column < size; ++column) {
      const auto value = decompose(input.at(row, column));
      if (value.significand != 0) {
        leastExponent = std::min(leastExponent, value.exponent);
        greatestBit = std::max(
            greatestBit, value.exponent + std::bit_width(value.significand));
      }
    }
  }
  if (leastExponent == maximumExponent) {
    return Result<ExactDeterminant>::success({});
  }
  const auto minorBits =
      minorBitBound(size, static_cast<Index>(greatestBit - leastExponent));
  StorageBudget budget(limits);
  constexpr Index bits = DeterminantInteger::limbBits;
  if (!minorBits) {
    return Result<ExactDeterminant>::failure(
        ErrorCode::ResourceLimit, "exact determinant bit bound overflows");
  }
  const Index temporaryBits = (2 * *minorBits) + 2;
  if (!budget.claim<DeterminantInteger>(size, size) ||
      !budget.claim<DeterminantInteger::Limb>(size * size,
                                              (*minorBits / bits) + 3) ||
      !budget.claim<DeterminantInteger::Limb>(8, (temporaryBits / bits) + 2)) {
    return Result<ExactDeterminant>::failure(
        ErrorCode::ResourceLimit, "exact determinant workspace exceeds limits");
  }

  std::vector<DeterminantInteger> work(size * size);
  const auto entry = [&work, size](Index rowIndex,
                                   Index columnIndex) -> DeterminantInteger& {
    return work.at((rowIndex * size) + columnIndex);
  };
  for (Index row = 0; row < size; ++row) {
    for (Index column = 0; column < size; ++column) {
      const Scalar scalar = input.at(row, column);
      const auto value = decompose(scalar);
      if (value.significand != 0) {
        entry(row, column) = DeterminantInteger(
            value.significand,
            static_cast<Index>(value.exponent - leastExponent),
            std::signbit(scalar));
      }
    }
  }

  bool negative = false;
  // Bareiss update: the numerator is a difference of products of minors
  // and divides exactly by the preceding pivot. Keep that pivot in work.
  for (Index pivotIndex = 0; pivotIndex + 1 < size; ++pivotIndex) {
    Index pivotRow = pivotIndex;
    while (pivotRow < size && entry(pivotRow, pivotIndex).zero()) {
      ++pivotRow;
    }
    if (pivotRow == size) {
      return Result<ExactDeterminant>::success({});
    }
    if (pivotRow != pivotIndex) {
      for (Index trailingColumn = pivotIndex; trailingColumn < size;
           ++trailingColumn) {
        std::swap(entry(pivotIndex, trailingColumn),
                  entry(pivotRow, trailingColumn));
      }
      negative = !negative;
    }
    const auto& pivot = entry(pivotIndex, pivotIndex);
    for (Index row = pivotIndex + 1; row < size; ++row) {
      for (Index trailingColumn = pivotIndex + 1; trailingColumn < size;
           ++trailingColumn) {
        auto numerator = DeterminantInteger::subtract(
            DeterminantInteger::multiply(pivot, entry(row, trailingColumn)),
            DeterminantInteger::multiply(entry(row, pivotIndex),
                                         entry(pivotIndex, trailingColumn)));
        if (pivotIndex != 0 &&
            !numerator.divideExactly(entry(pivotIndex - 1, pivotIndex - 1))) {
          return Result<ExactDeterminant>::failure(
              ErrorCode::NoConvergence, "exact determinant division failed");
        }
        entry(row, trailingColumn) = std::move(numerator);
      }
      entry(row, pivotIndex) = {};
    }
  }
  auto result = std::move(entry(size - 1, size - 1));
  if (negative) {
    result.negate();
  }
  return Result<ExactDeterminant>::success(
      {.magnitude = std::move(result),
       .exponent = static_cast<Exponent>(size) * leastExponent});
}

Result<Scalar> outOfRange() {
  return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                 "determinant exceeds Scalar range");
}

Wide logScaled(Wide significand, Exponent exponent) {
  return std::log(significand) +
         (static_cast<Wide>(exponent) * std::log(Wide{2}));
}

Scalar singularOutput(Output output) {
  return output == Output::determinant
             ? Scalar{0}
             : -std::numeric_limits<Scalar>::infinity();
}

Result<Scalar> convertExact(const ExactDeterminant& exact, Output output) {
  const auto& integer = exact.magnitude;
  const auto width = static_cast<Exponent>(integer.bits());
  const Exponent leadingPower = exact.exponent + width - 1;
  if (output == Output::logMagnitude) {
    const auto kept = static_cast<int>(std::min(width, Exponent{precision}));
    const auto top = integer.leading(static_cast<Index>(width - kept));
    const Wide mantissa = std::scalbn(static_cast<Wide>(top), 1 - kept);
    return Result<Scalar>::success(
        static_cast<Scalar>(logScaled(mantissa, leadingPower)));
  }
  if (leadingPower >= maximumExponent) {
    return outOfRange();
  }
  const Exponent quantum =
      std::max<Exponent>(minimumQuantum, leadingPower - precision + 1);
  const Exponent discarded = quantum - exact.exponent;
  const auto sign = integer.negative() ? Scalar{-1} : Scalar{1};
  std::uint64_t rounded = 0;
  if (discarded <= 0) {
    rounded = integer.leading(0) << static_cast<unsigned>(-discarded);
  } else if (discarded <= width) {
    const auto low = static_cast<Index>(discarded);
    rounded = integer.leading(low);
    if (integer.bit(low - 1) &&
        ((rounded & 1U) != 0 || integer.anyBelow(low - 1))) {
      ++rounded;
    }
  }
  if (rounded != 0 && quantum + static_cast<Exponent>(std::bit_width(rounded)) >
                          maximumExponent) {
    return outOfRange();
  }
  return Result<Scalar>::success(std::copysign(
      std::scalbn(static_cast<Scalar>(rounded), static_cast<int>(quantum)),
      sign));
}

std::optional<Result<Scalar>> convertBounded(const ScaledDeterminant& value,
                                             Output output) {
  if (output == Output::logMagnitude) {
    return Result<Scalar>::success(
        static_cast<Scalar>(std::fma(static_cast<Wide>(value.logExponent),
                                     std::log(Wide{2}), value.logSum)));
  }
  if (value.exponent > maximumExponent) {
    return outOfRange();
  }
  if (value.exponent < minimumQuantum - 2) {
    return Result<Scalar>::success(value.negative ? -0.0 : 0.0);
  }
  // Let exact rounding decide subnormal and overflow boundaries. Well inside
  // the normal range the interval bounds relative conversion error directly.
  if (value.exponent < std::numeric_limits<Scalar>::min_exponent ||
      value.exponent >= maximumExponent - 1) {
    return std::nullopt;
  }
  const auto rounded = static_cast<Scalar>(
      std::scalbn(center(value.magnitude), static_cast<int>(value.exponent)));
  return Result<Scalar>::success(value.negative ? -rounded : rounded);
}

Result<Scalar> compute(ConstMatrixView input, SolvePolicy policy, Limits limits,
                       Output output) {
  using ResultType = Result<Scalar>;
  const Index size = input.rows();
  if (size != input.columns()) {
    return ResultType::failure(ErrorCode::DimensionMismatch,
                               "determinant requires a square matrix");
  }
  if (!linear::detail::validNumericalPolicy(policy)) {
    return ResultType::failure(ErrorCode::InvalidTolerance,
                               "determinant tolerances are invalid");
  }
  if (policy.rank != RankPolicy::reject &&
      policy.rank != RankPolicy::minimum_norm &&
      policy.rank != RankPolicy::supported_subspace) {
    return ResultType::failure(ErrorCode::InvalidDomain,
                               "determinant rank policy is invalid");
  }
  if (size == 0) {
    return ResultType::success(output == Output::determinant ? 1.0 : 0.0);
  }
  StorageBudget budget(limits);
  if (!budget.claim<Bounds>(size, size)) {
    return ResultType::failure(ErrorCode::ResourceLimit,
                               "determinant workspace exceeds limits");
  }
  for (Index row = 0; row < size; ++row) {
    for (Index column = 0; column < size; ++column) {
      if (!std::isfinite(input.at(row, column))) {
        return ResultType::failure(ErrorCode::NonFiniteInput,
                                   "determinant input must be finite");
      }
    }
  }
  const auto bounded = boundedLU(input);
  if (bounded && zero(bounded->magnitude)) {
    return ResultType::success(singularOutput(output));
  }
  const auto rounded =
      bounded ? convertBounded(*bounded, output) : std::nullopt;
  bool singular = false;
  const auto evaluateExact = [&]() -> ResultType {
    const auto exact = exactDyadic(input, limits);
    if (!exact.hasValue()) {
      const auto& status = exact.status();
      return ResultType::failure(status.code(), std::string(status.message()));
    }
    if (exact.value().magnitude.zero()) {
      singular = true;
      return ResultType::success(singularOutput(output));
    }
    return convertExact(exact.value(), output);
  };
  // Exact integer storage is released before the independent spectral check.
  auto value = rounded ? *rounded : evaluateExact();
  if (!value.hasValue() &&
      value.status().code() != ErrorCode::NonFiniteResult) {
    return value;
  }
  if (singular) {
    return value;
  }
  const auto status =
      linear::detail::certifyNonsingularSpectrum(input, policy, limits);
  if (!status.ok()) {
    return ResultType::failure(status.code(), std::string(status.message()));
  }
  return value;
}

} // namespace

Result<Scalar> determinant(ConstMatrixView matrix, SolvePolicy policy,
                           Limits limits) {
  return compute(matrix, policy, limits, Output::determinant);
}

Result<Scalar> logAbsoluteDeterminant(ConstMatrixView matrix,
                                      SolvePolicy policy, Limits limits) {
  return compute(matrix, policy, limits, Output::logMagnitude);
}

} // namespace vss::math::covariance
