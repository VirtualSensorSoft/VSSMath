#include "ScaledContraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
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

  detail::ScaledAccumulator sum;
  for (Index i = begin; i < end; ++i) {
    const auto value = static_cast<detail::Wide>(entries[i].value);
    sum.add(std::scalbn(value, -common_exponent));
  }
  const auto value = static_cast<Scalar>(sum.value(common_exponent));
  if (!std::isfinite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "sparse duplicate sum is not representable");
  }
  return Result<Scalar>::success(value);
}

Status validateCSR(const CSR& matrix) {
  if (matrix.rows == std::numeric_limits<Extent>::max() ||
      matrix.row_offsets.view().size() != matrix.rows + 1U) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "CSR row offsets have an invalid extent");
  }
  const auto offsets = matrix.row_offsets.view();
  const auto columns = matrix.columns_by_entry.view();
  const auto values = matrix.values.view();
  if (columns.size() != values.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "CSR index and value arrays differ in length");
  }
  if (offsets.empty() || offsets.front() != 0U ||
      offsets.back() != values.size()) {
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
      if (columns[entry] >= matrix.columns) {
        return Status::failure(ErrorCode::IndexOutOfRange,
                               "CSR column lies outside the matrix shape");
      }
      if (has_previous && columns[entry] <= previous) {
        return Status::failure(ErrorCode::InvalidDomain,
                               "CSR columns are not canonical");
      }
      if (!std::isfinite(values.at(entry))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "CSR contains a non-finite value");
      }
      if (values.at(entry) == 0.0) {
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
  const auto columns = matrix.columns_by_entry.view();
  const auto values = matrix.values.view();
  const auto product = detail::scaledDot(end - begin, [&](Index offset) {
    const Index entry = begin + offset;
    return std::pair{values.at(entry), vector.at(columns[entry])};
  });
  const auto value = static_cast<Scalar>(product);
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
  const Extent entry_count = entries.size();
  for (const auto& entry : entries) {
    if (entry.row >= rows || entry.column >= columns) {
      return failure<CSR>(ErrorCode::IndexOutOfRange,
                          "sparse coordinate lies outside the matrix shape");
    }
    if (!std::isfinite(entry.value)) {
      return failure<CSR>(ErrorCode::NonFiniteInput,
                          "sparse triplet contains a non-finite value");
    }
  }
  // The stable merge sort has two simultaneously live Triplet arrays. The
  // coalescing/finalization stage is preflighted separately below, because its
  // peak is different and the sort workspace is released before it begins.
  if (!multiplyFits(entry_count, 2U) ||
      !multiplyFits(entry_count, 2U * sizeof(Triplet))) {
    return failure<CSR>(ErrorCode::SizeOverflow,
                        "CSR sorting workspace size overflows");
  }
  if (2U * entry_count > limits.elements ||
      2U * entry_count * sizeof(Triplet) > limits.bytes) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "CSR sorting workspace exceeds configured limits");
  }
  std::vector<Triplet> first(entries.begin(), entries.end());
  std::vector<Triplet> second(entry_count);
  const auto before = [](const Triplet& lhs, const Triplet& rhs) {
    return lhs.row < rhs.row ||
           (lhs.row == rhs.row && lhs.column <= rhs.column);
  };
  std::vector<Triplet>* source = &first;
  std::vector<Triplet>* destination = &second;
  for (Extent width = 1U; width < entry_count;) {
    for (Extent begin = 0U; begin < entry_count;) {
      const Extent middle = begin + std::min(width, entry_count - begin);
      const Extent end = middle + std::min(width, entry_count - middle);
      Extent left = begin;
      Extent right = middle;
      for (Extent out = begin; out < end; ++out) {
        if (left < middle &&
            (right == end || before((*source)[left], (*source)[right]))) {
          (*destination)[out] = (*source)[left++];
        } else {
          (*destination)[out] = (*source)[right++];
        }
      }
      begin = end;
    }
    std::swap(source, destination);
    if (width > entry_count / 2U) {
      break;
    }
    width *= 2U;
  }
  std::vector<Triplet> sorted = std::move(*source);
  std::vector<Triplet>().swap(first);
  std::vector<Triplet>().swap(second);

  // During coalescing, sorted triplets, row offsets, and worst-case entry
  // staging are live. During Array/Vector ownership transfer the final arrays
  // overlap their source staging, but the sorted scratch has been released.
  if (!multiplyFits(offset_count, 2U) || !multiplyFits(entry_count, 4U) ||
      !addFits(offset_count, 3U * entry_count) ||
      !addFits(2U * offset_count, 4U * entry_count) ||
      !multiplyFits(entry_count, sizeof(Triplet)) ||
      !multiplyFits(offset_count, 2U * sizeof(Index)) ||
      !multiplyFits(entry_count, 2U * (sizeof(Index) + sizeof(Scalar)))) {
    return failure<CSR>(ErrorCode::SizeOverflow,
                        "CSR coalescing workspace size overflows");
  }
  const Extent coalesce_elements = offset_count + 3U * entry_count;
  const Extent finalize_elements = 2U * offset_count + 4U * entry_count;
  const Extent sorted_bytes = entry_count * sizeof(Triplet);
  const Extent offset_bytes = offset_count * sizeof(Index);
  const Extent entry_output_bytes =
      entry_count * (sizeof(Index) + sizeof(Scalar));
  if (!addFits(sorted_bytes, offset_bytes) ||
      !addFits(sorted_bytes + offset_bytes, entry_output_bytes) ||
      !addFits(offset_bytes * 2U, entry_output_bytes * 2U)) {
    return failure<CSR>(ErrorCode::SizeOverflow,
                        "CSR peak byte accounting overflows");
  }
  const Extent coalesce_bytes =
      sorted_bytes + offset_bytes + entry_output_bytes;
  const Extent finalize_bytes = offset_bytes * 2U + entry_output_bytes * 2U;
  if (std::max(coalesce_elements, finalize_elements) > limits.elements ||
      std::max(coalesce_bytes, finalize_bytes) > limits.bytes) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "CSR coalescing/finalization peak exceeds limits");
  }

  std::vector<Index> offsets(offset_count, 0U);
  std::vector<Index> output_columns;
  std::vector<Scalar> output_values;
  output_columns.reserve(entry_count);
  output_values.reserve(entry_count);
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

  auto row_offsets =
      Array<Index>::copy(std::span<const Index>{offsets}, limits);
  auto column_indices =
      Array<Index>::copy(std::span<const Index>{output_columns}, limits);
  if (!row_offsets.hasValue() || !column_indices.hasValue()) {
    return failure<CSR>(ErrorCode::ResourceLimit,
                        "canonical CSR arrays exceed configured limits");
  }
  Vector values(std::span<const Scalar>{output_values});
  return Result<CSR>::success(CSR{rows, columns, std::move(row_offsets).value(),
                                  std::move(column_indices).value(),
                                  std::move(values)});
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
    result.at(row) = value.value();
  }
  return Result<Vector>::success(std::move(result));
}

} // namespace vss::math::linear
