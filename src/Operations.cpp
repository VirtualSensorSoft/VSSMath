#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {
namespace {

template <class T> Result<T> fromValidation(const Status& validation) {
  return Result<T>::failure(validation.code(),
                            std::string(validation.message()));
}

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool finite(ConstVectorView value) {
  for (Index index = 0U; index < value.size(); ++index) {
    if (!isFinite(value.at(index))) {
      return false;
    }
  }
  return true;
}

bool finite(ConstMatrixView value) {
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      if (!isFinite(value.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool checkedElementCount(Extent rows, Extent columns, Extent& count) noexcept {
  if (columns != 0U && rows > std::numeric_limits<Extent>::max() / columns) {
    return false;
  }
  count = rows * columns;
  return true;
}

template <class T> Result<T> validateOptions(OperationOptions options) {
  const Status validation = validate(options.tolerance);
  if (!validation.ok()) {
    return fromValidation<T>(validation);
  }
  return Result<T>::success(T{});
}

} // namespace

Result<Vector> add(ConstVectorView lhs, ConstVectorView rhs,
                   OperationOptions options) {
  auto option_check = validateOptions<Vector>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.size() != rhs.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "vector dimensions do not match");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "vector input is non-finite");
  }
  Vector result(lhs.size());
  for (Index index = 0U; index < lhs.size(); ++index) {
    result.at(index) = lhs.at(index) + rhs.at(index);
    if (options.requireFinite && !isFinite(result.at(index))) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "vector addition produced a non-finite result");
    }
  }
  return Result<Vector>::success(std::move(result));
}

Result<Vector> subtract(ConstVectorView lhs, ConstVectorView rhs,
                        OperationOptions options) {
  auto option_check = validateOptions<Vector>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.size() != rhs.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "vector dimensions do not match");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "vector input is non-finite");
  }
  Vector result(lhs.size());
  for (Index index = 0U; index < lhs.size(); ++index) {
    result.at(index) = lhs.at(index) - rhs.at(index);
    if (options.requireFinite && !isFinite(result.at(index))) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "vector subtraction produced a non-finite result");
    }
  }
  return Result<Vector>::success(std::move(result));
}

Result<Vector> scale(ConstVectorView value, Scalar factor,
                     OperationOptions options) {
  auto option_check = validateOptions<Vector>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (options.requireFinite && (!finite(value) || !isFinite(factor))) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "scale input is non-finite");
  }
  Vector result(value.size());
  for (Index index = 0U; index < value.size(); ++index) {
    result.at(index) = value.at(index) * factor;
    if (options.requireFinite && !isFinite(result.at(index))) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "scale produced a non-finite result");
    }
  }
  return Result<Vector>::success(std::move(result));
}

Result<Scalar> dot(ConstVectorView lhs, ConstVectorView rhs,
                   OperationOptions options) {
  auto option_check = validateOptions<Scalar>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.size() != rhs.size()) {
    return failure<Scalar>(ErrorCode::DimensionMismatch,
                           "vector dimensions do not match");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "dot-product input is non-finite");
  }
  Scalar result = 0.0;
  for (Index index = 0U; index < lhs.size(); ++index) {
    result += lhs.at(index) * rhs.at(index);
    if (options.requireFinite && !isFinite(result)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "dot product produced a non-finite result");
    }
  }
  return Result<Scalar>::success(result);
}

Result<Scalar> norm2(ConstVectorView value, OperationOptions options) {
  auto option_check = validateOptions<Scalar>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (options.requireFinite && !finite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "norm input is non-finite");
  }
  Scalar scale_value = 0.0;
  for (Index index = 0U; index < value.size(); ++index) {
    scale_value = std::max(scale_value, std::abs(value.at(index)));
  }
  if (scale_value == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  Scalar sum = 0.0;
  for (Index index = 0U; index < value.size(); ++index) {
    const Scalar scaled = value.at(index) / scale_value;
    sum += scaled * scaled;
    if (options.requireFinite && !isFinite(sum)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "norm accumulation produced a non-finite result");
    }
  }
  const Scalar result = scale_value * std::sqrt(sum);
  if (options.requireFinite && !isFinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "norm produced a non-finite result");
  }
  return Result<Scalar>::success(result);
}

Result<Vector> multiply(ConstMatrixView matrix, ConstVectorView vector,
                        OperationOptions options) {
  auto option_check = validateOptions<Vector>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (matrix.columns() != vector.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "matrix and vector dimensions do not match");
  }
  if (options.requireFinite && (!finite(matrix) || !finite(vector))) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "matrix-vector input is non-finite");
  }
  Vector result(matrix.rows());
  for (Index row = 0U; row < matrix.rows(); ++row) {
    Scalar sum = 0.0;
    for (Index column = 0U; column < matrix.columns(); ++column) {
      sum += matrix.at(row, column) * vector.at(column);
      if (options.requireFinite && !isFinite(sum)) {
        return failure<Vector>(
            ErrorCode::NonFiniteResult,
            "matrix-vector product produced a non-finite result");
      }
    }
    result.at(row) = sum;
  }
  return Result<Vector>::success(std::move(result));
}

Result<Matrix> multiply(ConstMatrixView lhs, ConstMatrixView rhs,
                        OperationOptions options) {
  auto option_check = validateOptions<Matrix>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.columns() != rhs.rows()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "matrix dimensions do not match");
  }
  Extent result_size = 0U;
  if (!checkedElementCount(lhs.rows(), rhs.columns(), result_size)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: matrix result element count");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "matrix input is non-finite");
  }
  Matrix result(lhs.rows(), rhs.columns());
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < rhs.columns(); ++column) {
      Scalar sum = 0.0;
      for (Index shared = 0U; shared < lhs.columns(); ++shared) {
        sum += lhs.at(row, shared) * rhs.at(shared, column);
        if (options.requireFinite && !isFinite(sum)) {
          return failure<Matrix>(ErrorCode::NonFiniteResult,
                                 "matrix product produced a non-finite result");
        }
      }
      result.at(row, column) = sum;
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Matrix> transpose(ConstMatrixView value, OperationOptions options) {
  auto option_check = validateOptions<Matrix>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  Extent result_size = 0U;
  if (!checkedElementCount(value.rows(), value.columns(), result_size)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: transpose result element count");
  }
  if (options.requireFinite && !finite(value)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "transpose input is non-finite");
  }
  Matrix result(value.columns(), value.rows());
  for (Index source_row = 0U; source_row < value.rows(); ++source_row) {
    for (Index source_column = 0U; source_column < value.columns();
         ++source_column) {
      result.at( // NOLINT(readability-suspicious-call-argument): transpose
                 // coordinate swap
          source_column, source_row) = value.at(source_row, source_column);
      if (options.requireFinite &&
          !isFinite(result.at( // NOLINT(readability-suspicious-call-argument):
                               // transpose coordinate swap
              source_column, source_row))) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "transpose produced a non-finite result");
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

} // namespace vss::math
