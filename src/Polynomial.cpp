#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Polynomial.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {
namespace {

// A normalized extended value keeps a binary exponent separately, so an
// intermediate polynomial term may exceed Scalar's range and later cancel.
struct Extended final {
  long double fraction{0.0L};
  long long exponent{0};
};

Extended extended(long double value) noexcept {
  int exponent = 0;
  const long double fraction = std::frexp(value, &exponent);
  return {fraction, exponent};
}

Extended add(Extended left, Extended right) noexcept {
  if (left.fraction == 0.0L) {
    return right;
  }
  if (right.fraction == 0.0L) {
    return left;
  }
  if (left.exponent < right.exponent) {
    const Extended temporary = left;
    left = right;
    right = temporary;
  }
  const long long difference = left.exponent - right.exponent;
  long double aligned = 0.0L;
  if (difference <= std::numeric_limits<long double>::digits + 2) {
    aligned = std::scalbn(right.fraction, -static_cast<int>(difference));
  }
  int shift = 0;
  const long double fraction = std::frexp(left.fraction + aligned, &shift);
  return {fraction, left.exponent + shift};
}

Extended subtract(Extended left, Extended right) noexcept {
  right.fraction = -right.fraction;
  return add(left, right);
}

Extended multiply(Extended left, Extended right) noexcept {
  int shift = 0;
  const long double fraction = std::frexp(left.fraction * right.fraction,
                                          &shift);
  return {fraction, left.exponent + right.exponent + shift};
}

Extended divide(Extended numerator, Extended denominator) noexcept {
  int shift = 0;
  const long double fraction =
      std::frexp(numerator.fraction / denominator.fraction, &shift);
  return {fraction, numerator.exponent - denominator.exponent + shift};
}

bool toScalar(Extended value, Scalar& result) noexcept {
  if (value.fraction == 0.0L) {
    result = 0.0;
    return true;
  }
  constexpr long long maxExponent =
      std::numeric_limits<Scalar>::max_exponent;
  if (value.exponent > maxExponent) {
    return false;
  }
  if (value.exponent <
      std::numeric_limits<Scalar>::min_exponent -
          std::numeric_limits<Scalar>::digits - 2) {
    result = std::copysign(0.0, static_cast<Scalar>(value.fraction));
    return true;
  }
  const long double expanded =
      std::scalbn(value.fraction, static_cast<int>(value.exponent));
  result = static_cast<Scalar>(expanded);
  return std::isfinite(result);
}

Status validateBasis(const BasisSpec& basis) {
  if (!std::isfinite(basis.origin) || !std::isfinite(basis.scale)) {
    return Status::failure(ErrorCode::NonFiniteInput,
                           "polynomial affine parameters must be finite");
  }
  if (basis.scale == 0.0 ||
      (basis.kind != BasisKind::power &&
       basis.kind != BasisKind::chebyshev_first)) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "polynomial basis or affine scale is invalid");
  }
  return Status::success();
}

Extended normalized(Scalar x, const BasisSpec& basis) noexcept {
  const Extended difference =
      subtract(extended(static_cast<long double>(x)),
               extended(static_cast<long double>(basis.origin)));
  return divide(difference,
                extended(static_cast<long double>(basis.scale)));
}

bool checkedMultiply(Extent left, Extent right, Extent& product) noexcept {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  product = left * right;
  return true;
}

bool withinLimits(Extent rows, Extent columns, Limits limits) noexcept {
  Extent elements = 0U;
  if (!checkedMultiply(rows, columns, elements) || elements > limits.elements ||
      elements > std::numeric_limits<std::size_t>::max() / sizeof(Scalar)) {
    return false;
  }
  const std::size_t payload =
      static_cast<std::size_t>(elements) * sizeof(Scalar);
  constexpr std::size_t shapeBytes = 2U * sizeof(Extent);
  return payload <= std::numeric_limits<std::size_t>::max() - shapeBytes &&
         payload + shapeBytes <= limits.bytes;
}

Result<Scalar> nonFiniteResult() {
  return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                 "polynomial value is not representable");
}

} // namespace

Result<Matrix> basisMatrix(ConstVectorView x, Index degree,
                          const BasisSpec& basis, Limits limits) {
  const Status basisStatus = validateBasis(basis);
  if (!basisStatus.ok()) {
    return Result<Matrix>::failure(basisStatus.code(),
                                   std::string(basisStatus.message()));
  }
  for (Index row = 0U; row < x.size(); ++row) {
    if (!std::isfinite(x.at(row))) {
      return Result<Matrix>::failure(
          ErrorCode::NonFiniteInput,
          "polynomial coordinate is non-finite at row " +
              std::to_string(row));
    }
  }
  if (degree == std::numeric_limits<Index>::max()) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "polynomial degree increment overflows");
  }
  const Extent columns = degree + 1U;
  if (!withinLimits(x.size(), columns, limits)) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "polynomial basis matrix exceeds limits");
  }

  try {
    Matrix output(x.size(), columns, 0.0);
    for (Index row = 0U; row < x.size(); ++row) {
      output.at(row, 0U) = 1.0;
      if (degree == 0U) {
        continue;
      }
      const Extended t = normalized(x.at(row), basis);
      Extended previous = extended(1.0L);
      Extended current = t;
      Scalar converted = 0.0;
      if (!toScalar(current, converted)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "polynomial basis element is not representable");
      }
      output.at(row, 1U) = converted;
      for (Index column = 2U; column <= degree; ++column) {
        Extended next;
        if (basis.kind == BasisKind::power) {
          next = multiply(current, t);
        } else {
          next = subtract(multiply(multiply(extended(2.0L), t), current),
                          previous);
        }
        if (!toScalar(next, converted)) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "polynomial basis element is not representable");
        }
        output.at(row, column) = converted;
        previous = current;
        current = next;
      }
    }
    return Result<Matrix>::success(std::move(output));
  } catch (const std::bad_alloc&) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "polynomial basis allocation failed");
  } catch (const std::length_error&) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "polynomial basis extent is unsupported");
  }
}

Result<Scalar> evaluatePolynomial(ConstVectorView coefficients, Scalar x,
                                  const BasisSpec& basis) {
  if (coefficients.empty()) {
    return Result<Scalar>::failure(ErrorCode::InsufficientSamples,
                                   "polynomial requires a coefficient");
  }
  const Status basisStatus = validateBasis(basis);
  if (!basisStatus.ok()) {
    return Result<Scalar>::failure(basisStatus.code(),
                                   std::string(basisStatus.message()));
  }
  if (!std::isfinite(x)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "polynomial coordinate must be finite");
  }
  for (Index index = 0U; index < coefficients.size(); ++index) {
    if (!std::isfinite(coefficients.at(index))) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteInput,
          "polynomial coefficient is non-finite at index " +
              std::to_string(index));
    }
  }
  if (coefficients.size() == 1U) {
    return Result<Scalar>::success(coefficients.at(0U));
  }

  Index highest = coefficients.size() - 1U;
  while (highest > 0U && coefficients.at(highest) == 0.0) {
    --highest;
  }
  if (highest == 0U) {
    return Result<Scalar>::success(coefficients.at(0U));
  }
  const Extended t = normalized(x, basis);

  Extended value;
  if (basis.kind == BasisKind::power) {
    value = extended(static_cast<long double>(coefficients.at(highest)));
    for (Index index = highest; index > 0U; --index) {
      value = add(multiply(value, t),
                  extended(static_cast<long double>(coefficients.at(index - 1U))));
    }
  } else if (t.exponent == 1 &&
             (t.fraction == 0.5L || t.fraction == -0.5L)) {
    Scalar scale = 0.0;
    for (Index index = 0U; index <= highest; ++index) {
      scale = std::max(scale, std::abs(coefficients.at(index)));
    }
    if (scale == 0.0) {
      return Result<Scalar>::success(0.0);
    }
    long double sum = 0.0L;
    long double correction = 0.0L;
    const bool positive = t.fraction > 0.0L;
    for (Index index = 0U; index <= highest; ++index) {
      const bool negate = !positive && (index % 2U != 0U);
      const long double term =
          (negate ? -1.0L : 1.0L) *
          (static_cast<long double>(coefficients.at(index)) /
           static_cast<long double>(scale));
      const long double adjusted = term - correction;
      const long double next = sum + adjusted;
      correction = (next - sum) - adjusted;
      sum = next;
    }
    value = multiply(extended(static_cast<long double>(scale)), extended(sum));
  } else {
    Extended next = extended(0.0L);
    Extended afterNext = extended(0.0L);
    for (Index index = highest; index > 0U; --index) {
      const Extended current = subtract(
          add(extended(static_cast<long double>(coefficients.at(index))),
              multiply(multiply(extended(2.0L), t), next)),
          afterNext);
      afterNext = next;
      next = current;
    }
    value = subtract(
        add(extended(static_cast<long double>(coefficients.at(0U))),
            multiply(t, next)),
        afterNext);
  }

  Scalar result = 0.0;
  if (!toScalar(value, result)) {
    return nonFiniteResult();
  }
  return Result<Scalar>::success(result);
}

} // namespace vss::math
