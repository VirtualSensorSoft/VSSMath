#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/PackedSymmetric.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs > std::numeric_limits<Extent>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

Result<Extent> packedCount(Index dimension, Index order) {
  if (dimension == 0U || order < 2U || order > 4U) {
    return failure<Extent>(
        ErrorCode::InvalidExtent,
        "packed symmetry requires positive dimension and order 2..4");
  }
  Extent n = 0U;
  if (!checkedAdd(dimension, order - 1U, n)) {
    return failure<Extent>(ErrorCode::SizeOverflow,
                           "packed symmetric binomial extent overflows");
  }
  Extent value = 1U;
  for (Index i = 1U; i <= order; ++i) {
    Extent numerator = n - order + i;
    Extent denominator = i;
    const Extent first = std::gcd(numerator, denominator);
    numerator /= first;
    denominator /= first;
    const Extent second = std::gcd(value, denominator);
    value /= second;
    denominator /= second;
    if (denominator != 1U || !checkedMultiply(value, numerator, value)) {
      return failure<Extent>(ErrorCode::SizeOverflow,
                             "packed symmetric binomial extent overflows");
    }
  }
  return Result<Extent>::success(value);
}

bool withinTolerance(Scalar lhs, Scalar rhs, Tolerance tolerance) noexcept {
  const Scalar scale = std::max(std::abs(lhs), std::abs(rhs));
  if (scale == 0.0) {
    return true;
  }
  const Scalar normalizedDifference = std::abs(lhs / scale - rhs / scale);
  return normalizedDifference <=
         tolerance.absolute / scale + tolerance.relative;
}

Extent tupleOffset(const std::array<Index, 4>& tuple, Index order,
                   Index dimension) noexcept {
  Extent offset = 0U;
  for (Index axis = 0U; axis < order; ++axis) {
    offset = offset * dimension + tuple[axis];
  }
  return offset;
}

void decode(Extent linear, Index order, Index dimension,
            std::array<Index, 4>& tuple) noexcept {
  for (Index axis = order; axis > 0U; --axis) {
    const Index current = axis - 1U;
    tuple[current] = linear % dimension;
    linear /= dimension;
  }
}

bool nextCombination(std::array<Index, 4>& tuple, Index order,
                     Index dimension) noexcept {
  for (Index axis = order; axis > 0U; --axis) {
    const Index current = axis - 1U;
    if (tuple[current] + 1U < dimension) {
      const Index next = tuple[current] + 1U;
      for (Index suffix = current; suffix < order; ++suffix) {
        tuple[suffix] = next;
      }
      return true;
    }
  }
  return false;
}

Result<Extent> validatePacked(const PackedSymmetric& input) {
  const auto count = packedCount(input.dimension, input.order);
  if (!count.hasValue()) {
    return count;
  }
  if (input.lexicographic_values.size() != count.value()) {
    return failure<Extent>(
        ErrorCode::DimensionMismatch,
        "packed symmetric storage length does not match metadata");
  }
  for (Extent i = 0U; i < count.value(); ++i) {
    if (!isFinite(input.lexicographic_values.at(i))) {
      return failure<Extent>(ErrorCode::NonFiniteInput,
                             "packed symmetric coefficient is non-finite");
    }
  }
  return count;
}

Extent orbitSize(const std::array<Index, 4>& tuple, Index order) noexcept {
  constexpr std::array<Extent, 5> factorial{1U, 1U, 2U, 6U, 24U};
  Extent multiplicity = factorial[order];
  Index run = 1U;
  for (Index i = 1U; i <= order; ++i) {
    if (i < order && tuple[i] == tuple[i - 1U]) {
      ++run;
    } else {
      multiplicity /= factorial[run];
      run = 1U;
    }
  }
  return multiplicity;
}

} // namespace

Result<PackedSymmetric> packSymmetric(const Tensor& input, Tolerance tolerance,
                                      Limits limits) {
  const Index order = input.shape().rank();
  if (order < 2U || order > 4U) {
    return failure<PackedSymmetric>(
        ErrorCode::InvalidExtent, "packed symmetry supports tensor ranks 2..4");
  }
  const Index dimension = input.shape().extent(0U);
  if (dimension == 0U) {
    return failure<PackedSymmetric>(
        ErrorCode::InvalidExtent, "packed symmetry requires positive extents");
  }
  for (Index axis = 1U; axis < order; ++axis) {
    if (input.shape().extent(axis) != dimension) {
      return failure<PackedSymmetric>(
          ErrorCode::DimensionMismatch,
          "packed symmetry requires equal axis extents");
    }
  }
  const Status toleranceStatus = validate(tolerance);
  if (!toleranceStatus.ok()) {
    return Result<PackedSymmetric>::failure(
        toleranceStatus.code(), std::string(toleranceStatus.message()));
  }
  for (Extent i = 0U; i < input.size(); ++i) {
    if (!isFinite(input.data()[i])) {
      return failure<PackedSymmetric>(ErrorCode::NonFiniteInput,
                                      "dense symmetric input is non-finite");
    }
  }
  const auto count = packedCount(dimension, order);
  if (!count.hasValue()) {
    return Result<PackedSymmetric>::failure(
        count.status().code(), std::string(count.status().message()));
  }
  Extent valueBytes = 0U;
  if (!checkedMultiply(count.value(), sizeof(Scalar), valueBytes)) {
    return failure<PackedSymmetric>(
        ErrorCode::SizeOverflow, "packed symmetric output byte size overflows");
  }
  if (count.value() > limits.elements ||
      valueBytes >
          limits.bytes - std::min(limits.bytes, sizeof(PackedSymmetric))) {
    return failure<PackedSymmetric>(
        ErrorCode::ResourceLimit,
        "packed symmetric output exceeds configured limits");
  }

  std::array<Index, 4> coordinate{};
  for (Extent linear = 0U; linear < input.size(); ++linear) {
    decode(linear, order, dimension, coordinate);
    auto canonical = coordinate;
    std::sort(canonical.begin(),
              canonical.begin() + static_cast<std::ptrdiff_t>(order));
    if (!withinTolerance(input.data()[linear],
                         input.data()[tupleOffset(canonical, order, dimension)],
                         tolerance)) {
      return failure<PackedSymmetric>(
          ErrorCode::NotSymmetric,
          "dense tensor differs within a symmetry orbit");
    }
  }

  Vector packed(count.value());
  coordinate.fill(0U);
  for (Extent i = 0U; i < count.value(); ++i) {
    packed.at(i) = input.data()[tupleOffset(coordinate, order, dimension)];
    if (i + 1U < count.value()) {
      static_cast<void>(nextCombination(coordinate, order, dimension));
    }
  }
  return Result<PackedSymmetric>::success(
      PackedSymmetric{order, dimension, std::move(packed)});
}

Result<Tensor> unpackSymmetric(const PackedSymmetric& input, Limits limits) {
  const auto count = validatePacked(input);
  if (!count.hasValue()) {
    return Result<Tensor>::failure(count.status().code(),
                                   std::string(count.status().message()));
  }
  std::array<Extent, 4> extents{};
  std::fill_n(extents.begin(), input.order, input.dimension);
  const auto shape =
      TensorShape::create(std::span<const Extent>{extents.data(), input.order});
  if (!shape.hasValue()) {
    return Result<Tensor>::failure(shape.status().code(),
                                   std::string(shape.status().message()));
  }
  Extent shapeBytes = 0U;
  Extent valueBytes = 0U;
  Extent totalBytes = 0U;
  if (!checkedMultiply(input.order, sizeof(Extent), shapeBytes) ||
      !checkedMultiply(shape.value().elementCount(), sizeof(Scalar),
                       valueBytes) ||
      !checkedAdd(shapeBytes, valueBytes, totalBytes)) {
    return failure<Tensor>(ErrorCode::SizeOverflow,
                           "dense symmetric output byte size overflows");
  }
  if (shape.value().elementCount() > limits.elements ||
      totalBytes > limits.bytes) {
    return failure<Tensor>(ErrorCode::ResourceLimit,
                           "dense symmetric output exceeds configured limits");
  }
  auto output =
      Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
  if (!output.hasValue()) {
    return output;
  }
  std::array<Index, 4> tuple{};
  for (Extent packedIndex = 0U; packedIndex < count.value(); ++packedIndex) {
    const Scalar value = input.lexicographic_values.at(packedIndex);
    do {
      output.value().data()[tupleOffset(tuple, input.order, input.dimension)] =
          value;
    } while (std::next_permutation(
        tuple.begin(),
        tuple.begin() + static_cast<std::ptrdiff_t>(input.order)));
    if (packedIndex + 1U < count.value()) {
      static_cast<void>(nextCombination(tuple, input.order, input.dimension));
    }
  }
  return output;
}

Result<Scalar> symmetricContraction(const PackedSymmetric& lhs,
                                    const PackedSymmetric& rhs) {
  const auto lhsCount = validatePacked(lhs);
  if (!lhsCount.hasValue()) {
    return Result<Scalar>::failure(lhsCount.status().code(),
                                   std::string(lhsCount.status().message()));
  }
  const auto rhsCount = validatePacked(rhs);
  if (!rhsCount.hasValue()) {
    return Result<Scalar>::failure(rhsCount.status().code(),
                                   std::string(rhsCount.status().message()));
  }
  if (lhs.order != rhs.order || lhs.dimension != rhs.dimension) {
    return failure<Scalar>(
        ErrorCode::DimensionMismatch,
        "packed symmetric operands have incompatible metadata");
  }
  Scalar lhsScale = 0.0;
  Scalar rhsScale = 0.0;
  for (Extent i = 0U; i < lhsCount.value(); ++i) {
    lhsScale = std::max(lhsScale, std::abs(lhs.lexicographic_values.at(i)));
    rhsScale = std::max(rhsScale, std::abs(rhs.lexicographic_values.at(i)));
  }
  if (lhsScale == 0.0 || rhsScale == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  long double sum = 0.0L;
  long double correction = 0.0L;
  std::array<Index, 4> tuple{};
  for (Extent i = 0U; i < lhsCount.value(); ++i) {
    const long double term =
        static_cast<long double>(orbitSize(tuple, lhs.order)) *
        (lhs.lexicographic_values.at(i) / lhsScale) *
        (rhs.lexicographic_values.at(i) / rhsScale);
    const long double adjusted = term - correction;
    const long double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
    if (i + 1U < lhsCount.value()) {
      static_cast<void>(nextCombination(tuple, lhs.order, lhs.dimension));
    }
  }
  int lhsExponent = 0;
  int rhsExponent = 0;
  const Scalar lhsMantissa = std::frexp(lhsScale, &lhsExponent);
  const Scalar rhsMantissa = std::frexp(rhsScale, &rhsExponent);
  const int exponent = lhsExponent + rhsExponent;
  const long double normalized =
      static_cast<long double>(lhsMantissa) * rhsMantissa * sum;
  const long double result = std::scalbn(normalized, exponent);
  const Scalar rounded = static_cast<Scalar>(result);
  if (!std::isfinite(rounded)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "packed symmetric contraction is non-finite");
  }
  return Result<Scalar>::success(rounded);
}

} // namespace vss::math
