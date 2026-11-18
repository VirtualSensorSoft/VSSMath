#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/NumericalError.hpp>
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

bool finite(const Tensor& value) {
  for (Index i = 0U; i < value.size(); ++i) {
    if (!isFinite(value.data()[i])) {
      return false;
    }
  }
  return true;
}

bool sameShape(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape().rank() != rhs.shape().rank()) {
    return false;
  }
  for (Index axis = 0U; axis < lhs.shape().rank(); ++axis) {
    if (lhs.shape().extent(axis) != rhs.shape().extent(axis)) {
      return false;
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

Status preflightStorage(Extent elements, const Limits& limits) {
  if (elements > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "elementwise output byte count overflows");
  }
  if (elements > limits.elements) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "elementwise output exceeds configured limits");
  }
  const Extent bytes = elements * sizeof(Scalar);
  if (bytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "elementwise output exceeds configured byte limit");
  }
  return Status::success();
}

template <class PairAt>
Result<Scalar> scaledProductSum(Extent count, PairAt pairAt) {
  int commonExponent = std::numeric_limits<int>::min();
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pairAt(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhsExponent = 0;
    int rhsExponent = 0;
    static_cast<void>(std::frexp(lhs, &lhsExponent));
    static_cast<void>(std::frexp(rhs, &rhsExponent));
    commonExponent = std::max(commonExponent, lhsExponent + rhsExponent);
  }
  if (commonExponent == std::numeric_limits<int>::min()) {
    return Result<Scalar>::success(0.0);
  }

  Scalar sum = 0.0;
  Scalar correction = 0.0;
  const auto accumulate = [&sum, &correction](Scalar value) {
    const Scalar next = sum + value;
    if (std::abs(sum) >= std::abs(value)) {
      correction += (sum - next) + value;
    } else {
      correction += (value - next) + sum;
    }
    sum = next;
  };
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pairAt(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhsExponent = 0;
    int rhsExponent = 0;
    const Scalar lhsFraction = std::frexp(lhs, &lhsExponent);
    const Scalar rhsFraction = std::frexp(rhs, &rhsExponent);
    const int productExponent = lhsExponent + rhsExponent;
    const Scalar productFraction = lhsFraction * rhsFraction;
    const Scalar productError =
        std::fma(lhsFraction, rhsFraction, -productFraction);
    const int shift = productExponent - commonExponent;
    accumulate(std::scalbn(productFraction, shift));
    accumulate(std::scalbn(productError, shift));
  }
  const Scalar scaled = sum + correction;
  const Scalar value = std::scalbn(scaled, commonExponent);
  if (!isFinite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "Frobenius inner product is not representable");
  }
  return Result<Scalar>::success(value == 0.0 ? 0.0 : value);
}

template <class ValueAt>
Result<Scalar> scaledNorm(Extent count, ValueAt valueAt) {
  Scalar norm = 0.0;
  for (Index i = 0U; i < count; ++i) {
    norm = std::hypot(norm, valueAt(i));
  }
  if (!isFinite(norm)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "Frobenius norm is not representable");
  }
  return Result<Scalar>::success(norm);
}

template <class T> Result<T> validateOptions(OperationOptions options) {
  const Status validation = validate(options.tolerance);
  if (!validation.ok()) {
    return fromValidation<T>(validation);
  }
  return Result<T>::success(T{});
}

template <class T> T unwrap(Result<T> result) {
  if (result.hasValue()) {
    return std::move(result).value();
  }
  throw NumericalError(result.status().code(), result.status().message());
}

} // namespace

Result<Vector> hadamard(ConstVectorView lhs, ConstVectorView rhs,
                        Limits limits) {
  if (lhs.size() != rhs.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "vector dimensions do not match");
  }
  const Status capacity = preflightStorage(lhs.size(), limits);
  if (!capacity.ok()) {
    return fromValidation<Vector>(capacity);
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "Hadamard vector input is non-finite");
  }
  Vector result(lhs.size());
  for (Index i = 0U; i < lhs.size(); ++i) {
    const Scalar product = lhs.at(i) * rhs.at(i);
    if (!isFinite(product)) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "Hadamard vector result is non-finite");
    }
    result.at(i) = product;
  }
  return Result<Vector>::success(std::move(result));
}

Result<Matrix> hadamard(ConstMatrixView lhs, ConstMatrixView rhs,
                        Limits limits) {
  if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "matrix dimensions do not match");
  }
  Extent count = 0U;
  if (!checkedElementCount(lhs.rows(), lhs.columns(), count)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: Hadamard matrix element count");
  }
  const Status capacity = preflightStorage(count, limits);
  if (!capacity.ok()) {
    return fromValidation<Matrix>(capacity);
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "Hadamard matrix input is non-finite");
  }
  Matrix result(lhs.rows(), lhs.columns());
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < lhs.columns(); ++column) {
      const Scalar product = lhs.at(row, column) * rhs.at(row, column);
      if (!isFinite(product)) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "Hadamard matrix result is non-finite");
      }
      result.at(row, column) = product;
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Scalar> frobenius(ConstMatrixView value) {
  Extent count = 0U;
  if (!checkedElementCount(value.rows(), value.columns(), count)) {
    return failure<Scalar>(ErrorCode::SizeOverflow,
                           "SizeOverflow: matrix element count");
  }
  if (!finite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "Frobenius matrix input is non-finite");
  }
  return scaledNorm(count, [&value](Index i) {
    const Index columns = value.columns();
    return value.at(i / columns, i % columns);
  });
}

Result<Scalar> frobeniusInner(ConstMatrixView lhs, ConstMatrixView rhs) {
  if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
    return failure<Scalar>(ErrorCode::DimensionMismatch,
                           "Frobenius matrix shapes do not match");
  }
  Extent count = 0U;
  if (!checkedElementCount(lhs.rows(), lhs.columns(), count)) {
    return failure<Scalar>(ErrorCode::SizeOverflow,
                           "SizeOverflow: Frobenius matrix element count");
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "Frobenius matrix input is non-finite");
  }
  return scaledProductSum(count, [&lhs, &rhs](Index i) {
    const Index columns = lhs.columns();
    const Index row = i / columns;
    const Index column = i % columns;
    return std::pair{lhs.at(row, column), rhs.at(row, column)};
  });
}

Result<Scalar> frobenius(const Tensor& value) {
  if (!finite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "Frobenius tensor input is non-finite");
  }
  return scaledNorm(value.size(),
                    [&value](Index i) { return value.data()[i]; });
}

Result<Scalar> frobeniusInner(const Tensor& lhs, const Tensor& rhs) {
  if (!sameShape(lhs, rhs)) {
    return failure<Scalar>(ErrorCode::DimensionMismatch,
                           "Frobenius tensor shapes do not match");
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "Frobenius tensor input is non-finite");
  }
  return scaledProductSum(lhs.size(), [&lhs, &rhs](Index i) {
    return std::pair{lhs.data()[i], rhs.data()[i]};
  });
}

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

Result<Matrix> add(ConstMatrixView lhs, ConstMatrixView rhs,
                   OperationOptions options) {
  auto option_check = validateOptions<Matrix>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "matrix dimensions do not match");
  }
  Extent result_size = 0U;
  if (!checkedElementCount(lhs.rows(), lhs.columns(), result_size)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: matrix sum element count");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "matrix input is non-finite");
  }
  Matrix result(lhs.rows(), lhs.columns());
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < lhs.columns(); ++column) {
      result.at(row, column) = lhs.at(row, column) + rhs.at(row, column);
      if (options.requireFinite && !isFinite(result.at(row, column))) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "matrix addition produced a non-finite result");
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Matrix> subtract(ConstMatrixView lhs, ConstMatrixView rhs,
                        OperationOptions options) {
  auto option_check = validateOptions<Matrix>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "matrix dimensions do not match");
  }
  Extent result_size = 0U;
  if (!checkedElementCount(lhs.rows(), lhs.columns(), result_size)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: matrix difference element count");
  }
  if (options.requireFinite && (!finite(lhs) || !finite(rhs))) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "matrix input is non-finite");
  }
  Matrix result(lhs.rows(), lhs.columns());
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < lhs.columns(); ++column) {
      result.at(row, column) = lhs.at(row, column) - rhs.at(row, column);
      if (options.requireFinite && !isFinite(result.at(row, column))) {
        return failure<Matrix>(
            ErrorCode::NonFiniteResult,
            "matrix subtraction produced a non-finite result");
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Matrix> scale(ConstMatrixView value, Scalar factor,
                     OperationOptions options) {
  auto option_check = validateOptions<Matrix>(options);
  if (!option_check.status().ok()) {
    return option_check;
  }
  Extent result_size = 0U;
  if (!checkedElementCount(value.rows(), value.columns(), result_size)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "SizeOverflow: matrix scale element count");
  }
  if (options.requireFinite && (!finite(value) || !isFinite(factor))) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "matrix scale input is non-finite");
  }
  Matrix result(value.rows(), value.columns());
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      result.at(row, column) = value.at(row, column) * factor;
      if (options.requireFinite && !isFinite(result.at(row, column))) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "matrix scale produced a non-finite result");
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Vector operator+(const Vector& lhs, const Vector& rhs) {
  return unwrap(add(lhs.view(), rhs.view()));
}
Vector operator-(const Vector& lhs, const Vector& rhs) {
  return unwrap(subtract(lhs.view(), rhs.view()));
}
Vector operator*(const Vector& value, Scalar factor) {
  return unwrap(scale(value.view(), factor));
}
Vector operator*(Scalar factor, const Vector& value) { return value * factor; }

Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
  return unwrap(add(lhs.view(), rhs.view()));
}
Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
  return unwrap(subtract(lhs.view(), rhs.view()));
}
Matrix operator*(const Matrix& value, Scalar factor) {
  return unwrap(scale(value.view(), factor));
}
Matrix operator*(Scalar factor, const Matrix& value) { return value * factor; }
Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
  return unwrap(multiply(lhs.view(), rhs.view()));
}
Vector operator*(const Matrix& matrix, const Vector& vector) {
  return unwrap(multiply(matrix.view(), vector.view()));
}

} // namespace vss::math
