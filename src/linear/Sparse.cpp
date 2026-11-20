#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/linear/Sparse.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool addFits(std::size_t lhs, std::size_t rhs) noexcept {
  return rhs <= std::numeric_limits<std::size_t>::max() - lhs;
}

bool multiplyFits(std::size_t lhs, std::size_t rhs) noexcept {
  return lhs == 0U || rhs <= std::numeric_limits<std::size_t>::max() / lhs;
}

bool accumulateCompensated(Scalar value, Scalar& sum,
                           Scalar& correction) noexcept {
  const Scalar next = sum + value;
  if (!std::isfinite(next)) {
    return false;
  }
  if (std::abs(sum) >= std::abs(value)) {
    correction += (sum - next) + value;
  } else {
    correction += (value - next) + sum;
  }
  sum = next;
  return std::isfinite(correction);
}

Result<Scalar> scaledSum(const std::vector<Triplet>& entries, Index begin,
                         Index end) {
  int common_exponent = std::numeric_limits<int>::min();
  for (Index i = begin; i < end; ++i) {
    int exponent = 0;
    static_cast<void>(std::frexp(entries[i].value, &exponent));
    common_exponent = std::max(common_exponent, exponent);
  }
  if (common_exponent == std::numeric_limits<int>::min()) {
    return Result<Scalar>::success(0.0);
  }

  Scalar sum = 0.0;
  Scalar correction = 0.0;
  for (Index i = begin; i < end; ++i) {
    int exponent = 0;
    const Scalar fraction = std::frexp(entries[i].value, &exponent);
    const Scalar scaled = std::scalbn(fraction, exponent - common_exponent);
    if (!accumulateCompensated(scaled, sum, correction)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "sparse duplicate sum is not representable");
    }
  }
  const Scalar value = std::scalbn(sum + correction, common_exponent);
  if (!std::isfinite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "sparse duplicate sum is not representable");
  }
  return Result<Scalar>::success(value);
}

Status validateCSR(const CSR& matrix) {
  if (matrix.rows == std::numeric_limits<Extent>::max() ||
      matrix.row_offsets.view().size() != matrix.rows + 1U) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "CSR row offsets have an invalid extent");
  }
  const auto offsets = matrix.row_offsets.view();
  const auto columns = matrix.column_indices.view();
  const auto values = matrix.values.view();
  if (columns.size() != values.size() || offsets.empty() ||
      offsets.front() != 0U || offsets.back() != values.size()) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "CSR arrays have inconsistent extents");
  }
  for (Index row = 0U; row < matrix.rows; ++row) {
    const Index begin = offsets[row];
    const Index end = offsets[row + 1U];
    if (begin > end || end > values.size()) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "CSR row offsets are not monotone");
    }
    Index previous = 0U;
    bool has_previous = false;
    for (Index entry = begin; entry < end; ++entry) {
      if (columns[entry] >= matrix.columns ||
          (has_previous && columns[entry] <= previous)) {
        return Status::failure(ErrorCode::InvalidDomain,
                               "CSR columns are not canonical");
      }
      if (!std::isfinite(values[entry])) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "CSR contains a non-finite value");
      }
      if (values[entry] == 0.0) {
        return Status::failure(ErrorCode::InvalidDomain,
                               "canonical CSR must omit stored zeros");
      }
      previous = columns[entry];
      has_previous = true;
    }
  }
  return Status::success();
}

Result<Scalar> scaledRowDot(const CSR& matrix, ConstVectorView vector,
                            Index begin, Index end) {
  const auto columns = matrix.column_indices.view();
  const auto values = matrix.values.view();
  int common_exponent = std::numeric_limits<int>::min();
  for (Index entry = begin; entry < end; ++entry) {
    const Scalar lhs = values[entry];
    const Scalar rhs = vector.at(columns[entry]);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    static_cast<void>(std::frexp(lhs, &lhs_exponent));
    static_cast<void>(std::frexp(rhs, &rhs_exponent));
    common_exponent = std::max(common_exponent, lhs_exponent + rhs_exponent);
  }
  if (common_exponent == std::numeric_limits<int>::min()) {
    return Result<Scalar>::success(0.0);
  }

  Scalar sum = 0.0;
  Scalar correction = 0.0;
  for (Index entry = begin; entry < end; ++entry) {
    const Scalar lhs = values[entry];
    const Scalar rhs = vector.at(columns[entry]);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const Scalar lhs_fraction = std::frexp(lhs, &lhs_exponent);
    const Scalar rhs_fraction = std::frexp(rhs, &rhs_exponent);
    const int exponent = lhs_exponent + rhs_exponent;
    const Scalar product = lhs_fraction * rhs_fraction;
    const Scalar product_error = std::fma(lhs_fraction, rhs_fraction, -product);
    const int shift = exponent - common_exponent;
    if (!accumulateCompensated(std::scalbn(product, shift), sum, correction) ||
        !accumulateCompensated(std::scalbn(product_error, shift), sum,
                               correction)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "sparse row accumulation is not representable");
    }
  }
  const Scalar value = std::scalbn(sum + correction, common_exponent);
  if (!std::isfinite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "sparse row result is not representable");
  }
  return Result<Scalar>::success(value);
}

} // namespace

Result<CSR> sparseCSRFromTriplets(Extent rows, Extent columns,
                                  TripletView entries, Limits limits) {
  if (rows == std::numeric_limits<Extent>::max()) {
    return failure<CSR>(ErrorCode::SizeOverflow,
                        "CSR row offset extent overflows");
  }
  const Extent offset_count = rows + 1U;
  if (entries.size() > limits.elements || offset_count > limits.elements) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "CSR construction exceeds the element limit");
  }
  if (!multiplyFits(entries.size(), sizeof(Triplet)) ||
      !multiplyFits(offset_count, sizeof(Index)) ||
      !multiplyFits(entries.size(), sizeof(Index) + sizeof(Scalar))) {
    return failure<CSR>(ErrorCode::SizeOverflow,
                        "CSR construction storage size overflows");
  }
  const auto triplet_bytes = entries.size() * sizeof(Triplet);
  const auto offset_bytes = offset_count * sizeof(Index);
  const auto value_bytes = entries.size() * (sizeof(Index) + sizeof(Scalar));
  if (!addFits(triplet_bytes, offset_bytes) ||
      !addFits(triplet_bytes + offset_bytes, value_bytes) ||
      triplet_bytes + offset_bytes + value_bytes > limits.bytes) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "CSR construction exceeds the byte limit");
  }
  if (entries.size() > (limits.elements - offset_count) / 3U) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "CSR construction workspace exceeds the element limit");
  }

  std::vector<Triplet> sorted(entries.begin(), entries.end());
  for (const auto& entry : sorted) {
    if (entry.row >= rows || entry.column >= columns) {
      return failure<CSR>(ErrorCode::IndexOutOfRange,
                          "sparse coordinate lies outside the matrix shape");
    }
    if (!std::isfinite(entry.value)) {
      return failure<CSR>(ErrorCode::NonFiniteInput,
                          "sparse triplet contains a non-finite value");
    }
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const Triplet& lhs, const Triplet& rhs) {
              if (lhs.row != rhs.row) {
                return lhs.row < rhs.row;
              }
              if (lhs.column != rhs.column) {
                return lhs.column < rhs.column;
              }
              return lhs.value < rhs.value;
            });

  std::vector<Index> offsets(offset_count, 0U);
  std::vector<Index> output_columns;
  std::vector<Scalar> output_values;
  output_columns.reserve(sorted.size());
  output_values.reserve(sorted.size());
  Index cursor = 0U;
  for (Index row = 0U; row < rows; ++row) {
    while (cursor < sorted.size() && sorted[cursor].row == row) {
      const Index begin = cursor;
      const Index column = sorted[cursor].column;
      while (cursor < sorted.size() && sorted[cursor].row == row &&
             sorted[cursor].column == column) {
        ++cursor;
      }
      const auto sum = scaledSum(sorted, begin, cursor);
      if (!sum.hasValue()) {
        return failure<CSR>(sum.status().code(),
                            "sparse duplicate coalescing failed");
      }
      if (sum.value() != 0.0) {
        output_columns.push_back(column);
        output_values.push_back(sum.value());
      }
    }
    offsets[row + 1U] = output_values.size();
  }

  if (offsets.size() + output_columns.size() + output_values.size() >
          limits.elements ||
      offsets.size() * sizeof(Index) + output_columns.size() * sizeof(Index) +
              output_values.size() * sizeof(Scalar) >
          limits.bytes) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "canonical CSR output exceeds configured limits");
  }
  auto row_offsets =
      Array<Index>::copy(std::span<const Index>{offsets}, limits);
  auto column_indices =
      Array<Index>::copy(std::span<const Index>{output_columns}, limits);
  auto values =
      Array<Scalar>::copy(std::span<const Scalar>{output_values}, limits);
  if (!row_offsets.hasValue() || !column_indices.hasValue() ||
      !values.hasValue()) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "canonical CSR arrays exceed configured limits");
  }
  return Result<CSR>::success(CSR{rows, columns, std::move(row_offsets).value(),
                                  std::move(column_indices).value(),
                                  std::move(values).value()});
}

Result<Vector> apply(const CSR& matrix, ConstVectorView vector, Limits limits) {
  const auto valid = validateCSR(matrix);
  if (!valid.ok()) {
    return failure<Vector>(valid.code(), "CSR matrix is not canonical");
  }
  if (vector.size() != matrix.columns) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "CSR column count must equal vector size");
  }
  for (Index column = 0U; column < vector.size(); ++column) {
    if (!std::isfinite(vector.at(column))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "sparse input vector contains a non-finite value");
    }
  }
  if (matrix.rows > limits.elements ||
      !multiplyFits(matrix.rows, sizeof(Scalar)) ||
      matrix.rows * sizeof(Scalar) > limits.bytes) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "sparse result exceeds configured limits");
  }

  Vector result(matrix.rows, 0.0);
  const auto offsets = matrix.row_offsets.view();
  for (Index row = 0U; row < matrix.rows; ++row) {
    const auto value =
        scaledRowDot(matrix, vector, offsets[row], offsets[row + 1U]);
    if (!value.hasValue()) {
      return failure<Vector>(value.status().code(),
                             "sparse matrix-vector product failed");
    }
    result.data()[row] = value.value();
  }
  return Result<Vector>::success(std::move(result));
}

} // namespace vss::math::linear
