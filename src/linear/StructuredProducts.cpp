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
#include <vss/math/Tolerance.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

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

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!isFinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

Status preflight(Extent rows, Extent columns, Limits limits) {
  Extent elements = 0U;
  Extent bytes = 0U;
  if (!checkedMultiply(rows, columns, elements) ||
      !checkedMultiply(elements, sizeof(Scalar), bytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "structured matrix output size overflows");
  }
  if (elements > limits.elements || bytes > limits.bytes) {
    return Status::failure(
        ErrorCode::ResourceLimit,
        "structured matrix exceeds configured output limits");
  }
  return Status::success();
}

} // namespace

Result<Matrix> kronecker(ConstMatrixView lhs, ConstMatrixView rhs,
                         Limits limits) {
  Extent rows = 0U;
  Extent columns = 0U;
  if (!checkedMultiply(lhs.rows(), rhs.rows(), rows) ||
      !checkedMultiply(lhs.columns(), rhs.columns(), columns)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "Kronecker output dimensions overflow");
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "Kronecker operand is non-finite");
  }
  const Status capacity = preflight(rows, columns, limits);
  if (!capacity.ok()) {
    return Result<Matrix>::failure(capacity.code(),
                                   std::string(capacity.message()));
  }
  Matrix result(rows, columns);
  for (Index i = 0U; i < lhs.rows(); ++i) {
    for (Index j = 0U; j < lhs.columns(); ++j) {
      for (Index k = 0U; k < rhs.rows(); ++k) {
        for (Index l = 0U; l < rhs.columns(); ++l) {
          const Scalar product = lhs.at(i, j) * rhs.at(k, l);
          if (!isFinite(product)) {
            return failure<Matrix>(ErrorCode::NonFiniteResult,
                                   "Kronecker product is non-finite");
          }
          result.at(i * rhs.rows() + k, j * rhs.columns() + l) = product;
        }
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Matrix> khatriRao(ConstMatrixView lhs, ConstMatrixView rhs,
                         Limits limits) {
  if (lhs.columns() != rhs.columns()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "Khatri-Rao operands need equal column counts");
  }
  Extent rows = 0U;
  if (!checkedMultiply(lhs.rows(), rhs.rows(), rows)) {
    return failure<Matrix>(ErrorCode::SizeOverflow,
                           "Khatri-Rao output row count overflows");
  }
  if (!finite(lhs) || !finite(rhs)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "Khatri-Rao operand is non-finite");
  }
  const Status capacity = preflight(rows, lhs.columns(), limits);
  if (!capacity.ok()) {
    return Result<Matrix>::failure(capacity.code(),
                                   std::string(capacity.message()));
  }
  Matrix result(rows, lhs.columns());
  for (Index column = 0U; column < lhs.columns(); ++column) {
    for (Index i = 0U; i < lhs.rows(); ++i) {
      for (Index k = 0U; k < rhs.rows(); ++k) {
        const Scalar product = lhs.at(i, column) * rhs.at(k, column);
        if (!isFinite(product)) {
          return failure<Matrix>(ErrorCode::NonFiniteResult,
                                 "Khatri-Rao product is non-finite");
        }
        result.at(i * rhs.rows() + k, column) = product;
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Scalar> trace(ConstMatrixView value) {
  if (value.rows() != value.columns()) {
    return failure<Scalar>(ErrorCode::DimensionMismatch,
                           "trace requires a square matrix");
  }
  if (!finite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "trace input is non-finite");
  }
  long double sum = 0.0L;
  long double correction = 0.0L;
  for (Index i = 0U; i < value.rows(); ++i) {
    const long double adjusted =
        static_cast<long double>(value.at(i, i)) - correction;
    const long double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  const auto result = static_cast<Scalar>(sum);
  if (!isFinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "trace result is non-finite");
  }
  return Result<Scalar>::success(result);
}

} // namespace vss::math
