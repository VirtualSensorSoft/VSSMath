#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>

namespace vss::math {
namespace {

void checkRow(Index row, Extent rows) {
  if (row >= rows) {
    throw std::out_of_range("IndexOutOfRange: matrix row " +
                            std::to_string(row));
  }
}

void checkColumn(Index column, Extent columns) {
  if (column >= columns) {
    throw std::out_of_range("IndexOutOfRange: matrix column " +
                            std::to_string(column));
  }
}

} // namespace

Matrix::Matrix(Extent rows, Extent columns, Scalar value)
    : rows_(rows), columns_(columns) {
  if (columns != 0U && rows > std::numeric_limits<Extent>::max() / columns) {
    throw std::length_error("SizeOverflow: matrix element count");
  }
  values_.assign(rows * columns, value);
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), columns_(other.columns_),
      values_(std::move(other.values_)) {
  other.rows_ = 0U;
  other.columns_ = 0U;
  other.values_.clear();
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    values_ = std::move(other.values_);
    other.rows_ = 0U;
    other.columns_ = 0U;
    other.values_.clear();
  }
  return *this;
}

Extent Matrix::rows() const noexcept { return rows_; }

Extent Matrix::columns() const noexcept { return columns_; }

Extent Matrix::size() const noexcept {
  return static_cast<Extent>(values_.size());
}

bool Matrix::empty() const noexcept { return values_.empty(); }

Scalar* Matrix::data() noexcept { return values_.data(); }

const Scalar* Matrix::data() const noexcept { return values_.data(); }

Scalar& Matrix::at(Index row, Index column) {
  checkRow(row, rows_);
  checkColumn(column, columns_);
  return values_[row * columns_ + column];
}

const Scalar& Matrix::at(Index row, Index column) const {
  checkRow(row, rows_);
  checkColumn(column, columns_);
  return values_[row * columns_ + column];
}

void Matrix::fill(Scalar value) {
  std::fill(values_.begin(), values_.end(), value);
}

MatrixView Matrix::view() noexcept {
  return MatrixView(data(), rows(), columns(), columns(), 1U);
}

ConstMatrixView Matrix::view() const noexcept {
  return ConstMatrixView(data(), rows(), columns(), columns(), 1U);
}

} // namespace vss::math
