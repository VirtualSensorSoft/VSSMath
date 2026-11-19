#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Scaling.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

template <class View> bool allFinite(const View& view) noexcept {
  for (Extent i = 0U; i < view.size(); ++i) {
    if (!isFinite(view.at(i))) {
      return false;
    }
  }
  return true;
}

bool allFinite(ConstMatrixView view) noexcept {
  for (Index row = 0U; row < view.rows(); ++row) {
    for (Index column = 0U; column < view.columns(); ++column) {
      if (!isFinite(view.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

Scalar product3(Scalar first, Scalar second, Scalar third) noexcept {
  if (first == 0.0 || second == 0.0 || third == 0.0) {
    return std::copysign(0.0, first * second * third);
  }
  int firstExponent = 0;
  int secondExponent = 0;
  int thirdExponent = 0;
  const Scalar firstMantissa = std::frexp(first, &firstExponent);
  const Scalar secondMantissa = std::frexp(second, &secondExponent);
  const Scalar thirdMantissa = std::frexp(third, &thirdExponent);
  const int exponent = firstExponent + secondExponent + thirdExponent;
  const long double mantissa =
      static_cast<long double>(firstMantissa) * secondMantissa * thirdMantissa;
  return static_cast<Scalar>(std::scalbn(mantissa, exponent));
}

} // namespace

Result<Vector> convertLinearUnits(ConstVectorView value, ConstVectorView scale,
                                  ConstVectorView offset, Limits limits) {
  if (value.size() != scale.size() || value.size() != offset.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "value, scale, and offset extents must match");
  }
  if (!allFinite(value) || !allFinite(scale) || !allFinite(offset)) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "affine conversion input is non-finite");
  }
  Extent outputBytes = 0U;
  if (!checkedMultiply(value.size(), sizeof(Scalar), outputBytes)) {
    return failure<Vector>(ErrorCode::SizeOverflow,
                           "affine conversion output byte size overflows");
  }
  if (value.size() > limits.elements || outputBytes > limits.bytes) {
    return failure<Vector>(
        ErrorCode::ResourceLimit,
        "affine conversion exceeds configured output limits");
  }
  Vector result(value.size());
  for (Index i = 0U; i < value.size(); ++i) {
    result.at(i) = std::fma(scale.at(i), value.at(i), offset.at(i));
    if (!isFinite(result.at(i))) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "affine conversion result is non-finite");
    }
  }
  return Result<Vector>::success(std::move(result));
}

Result<Matrix> scaleCovariance(ConstMatrixView value, ConstVectorView left,
                               ConstVectorView right, Limits limits) {
  if (value.rows() != left.size() || value.columns() != right.size()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "covariance and left/right scaling extents differ");
  }
  if (!allFinite(value) || !allFinite(left) || !allFinite(right)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "covariance scaling input is non-finite");
  }
  Extent elements = 0U;
  Extent bytes = 0U;
  if (!checkedMultiply(value.rows(), value.columns(), elements) ||
      !checkedMultiply(elements, sizeof(Scalar), bytes)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "scaled covariance output size overflows");
  }
  if (elements > limits.elements || bytes > limits.bytes) {
    return failure<Matrix>(
        ErrorCode::ResourceLimit,
        "scaled covariance exceeds configured output limits");
  }
  Matrix result(value.rows(), value.columns());
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      result.at(row, column) =
          product3(left.at(row), value.at(row, column), right.at(column));
      if (!isFinite(result.at(row, column))) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "scaled covariance result is non-finite");
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

} // namespace vss::math
