#ifndef VSS_MATH_MATRIX_VIEW_HPP
#define VSS_MATH_MATRIX_VIEW_HPP

#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

class Matrix;
class ConstMatrixView;

/// @brief A checked, non-owning mutable view of strided matrix storage.
///
/// The view borrows row-major strided storage, never owns it, and never
/// extends its lifetime. Concurrent reads of stable shared storage are safe;
/// any access that can modify the storage must be externally synchronized with
/// every aliased view.
class MatrixView final {
public:
  /// @brief Validates storage, extents, and strides before creating a view.
  /// @param storage Storage that must outlive the returned view.
  /// @param rows Logical row extent.
  /// @param columns Logical column extent.
  /// @param rowStride Storage distance between adjacent logical rows.
  /// @param columnStride Storage distance between adjacent logical columns.
  /// @return A view or an InvalidExtent/SizeOverflow failure.
  /// @post A non-empty view reaches its logical last element without exceeding
  ///       `storage`; a zero-area view may borrow empty storage.
  /// @note Complexity: Constant time and constant auxiliary space.
  [[nodiscard]] static Result<MatrixView> create(std::span<Scalar> storage,
                                                 Extent rows, Extent columns,
                                                 Extent rowStride,
                                                 Extent columnStride = 1U) {
    if (rows == 0U || columns == 0U) {
      return Result<MatrixView>::success(
          MatrixView(storage.data(), rows, columns, rowStride, columnStride));
    }
    if (rowStride == 0U || columnStride == 0U) {
      return Result<MatrixView>::failure(ErrorCode::InvalidExtent,
                                         "matrix view strides must be nonzero");
    }
    Extent rowOffset = 0U;
    Extent columnOffset = 0U;
    if (!checkedMultiply(rows - 1U, rowStride, rowOffset) ||
        !checkedMultiply(columns - 1U, columnStride, columnOffset)) {
      return Result<MatrixView>::failure(ErrorCode::SizeOverflow,
                                         "SizeOverflow: matrix view offset");
    }
    Extent lastOffset = 0U;
    if (!checkedAdd(rowOffset, columnOffset, lastOffset)) {
      return Result<MatrixView>::failure(ErrorCode::SizeOverflow,
                                         "SizeOverflow: matrix view offset");
    }
    if (lastOffset >= storage.size()) {
      return Result<MatrixView>::failure(
          ErrorCode::InvalidExtent, "matrix view exceeds supplied storage");
    }
    return Result<MatrixView>::success(
        MatrixView(storage.data(), rows, columns, rowStride, columnStride));
  }

  /// @brief Returns the logical row count.
  [[nodiscard]] Extent rows() const noexcept { return rows_; }
  /// @brief Returns the logical column count.
  [[nodiscard]] Extent columns() const noexcept { return columns_; }
  /// @brief Returns the storage distance between adjacent rows.
  [[nodiscard]] Extent rowStride() const noexcept { return rowStride_; }
  /// @brief Returns the storage distance between adjacent columns.
  [[nodiscard]] Extent columnStride() const noexcept { return columnStride_; }
  /// @brief Reports whether either logical extent is zero.
  [[nodiscard]] bool empty() const noexcept {
    return rows_ == 0U || columns_ == 0U;
  }
  /// @brief Returns the borrowed first storage address.
  /// @note The pointer is non-owning and becomes invalid when the source
  ///       storage is destroyed, moved, or reallocated.
  [[nodiscard]] Scalar* data() const noexcept { return data_; }

  /// @brief Accesses an element after row-then-column bounds checking.
  /// @throws std::out_of_range when either index is invalid.
  /// @note Complexity: Constant time.
  [[nodiscard]] Scalar& at(Index row, Index column) const {
    checkRow(row);
    checkColumn(column);
    return data_ // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                 // checked offsets
        [row * rowStride_ + column * columnStride_];
  }

  /// @brief Creates a checked borrowing subview retaining the parent strides.
  /// @param originRow Logical row of the first element.
  /// @param originColumn Logical column of the first element.
  /// @param rows Requested row extent.
  /// @param columns Requested column extent.
  /// @return A subview or InvalidExtent when the requested rectangle is
  ///         outside the parent view.
  /// @post The returned view retains both parent strides and storage lifetime.
  /// @note Complexity: Constant time.
  [[nodiscard]] Result<MatrixView> subview(Index originRow, Index originColumn,
                                           Extent rows, Extent columns) const {
    if (originRow > rows_ || rows > rows_ - originRow ||
        originColumn > columns_ || columns > columns_ - originColumn) {
      return Result<MatrixView>::failure(ErrorCode::InvalidExtent,
                                         "matrix subview exceeds parent view");
    }
    if (rows == 0U || columns == 0U) {
      return Result<MatrixView>::success(
          MatrixView(data_, rows, columns, rowStride_, columnStride_));
    }
    return Result<MatrixView>::success(MatrixView(
        data_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                // checked offset
            originRow *
                rowStride_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                             // checked offset
            originColumn * columnStride_,
        rows, columns, rowStride_, columnStride_));
  }

  /// @brief Converts a mutable view to an immutable borrowing view.
  /// @return A view of the same storage, extents, and strides without mutation
  ///         capability.
  operator ConstMatrixView() const noexcept;

private:
  friend class Matrix;
  MatrixView(Scalar* data, Extent rows, Extent columns, Extent rowStride,
             Extent columnStride)
      : data_(data), rows_(rows), columns_(columns), rowStride_(rowStride),
        columnStride_(columnStride) {}

  static bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
    if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
      return false;
    }
    result = lhs * rhs;
    return true;
  }

  static bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
    if (lhs > std::numeric_limits<Extent>::max() - rhs) {
      return false;
    }
    result = lhs + rhs;
    return true;
  }

  void checkRow(Index row) const {
    if (row >= rows_) {
      throw std::out_of_range("IndexOutOfRange: matrix view row " +
                              std::to_string(row));
    }
  }
  void checkColumn(Index column) const {
    if (column >= columns_) {
      throw std::out_of_range("IndexOutOfRange: matrix view column " +
                              std::to_string(column));
    }
  }

  Scalar* data_;
  Extent rows_;
  Extent columns_;
  Extent rowStride_;
  Extent columnStride_;
};

/// @brief A checked, non-owning immutable view of strided matrix storage.
///
/// The view borrows storage and never extends its lifetime. Concurrent reads
/// are safe when the underlying storage remains alive and is not concurrently
/// modified; synchronization is the caller's responsibility otherwise.
class ConstMatrixView final {
public:
  /// @brief Validates storage, extents, and strides before creating a view.
  /// @param storage Storage that must outlive the returned view.
  /// @param rows Logical row extent.
  /// @param columns Logical column extent.
  /// @param rowStride Storage distance between adjacent logical rows.
  /// @param columnStride Storage distance between adjacent logical columns.
  /// @return A view or an InvalidExtent/SizeOverflow failure.
  /// @post A non-empty view reaches its logical last element without exceeding
  ///       `storage`; a zero-area view may borrow empty storage.
  /// @note Complexity: Constant time and constant auxiliary space.
  [[nodiscard]] static Result<ConstMatrixView>
  create(std::span<const Scalar> storage, Extent rows, Extent columns,
         Extent rowStride, Extent columnStride = 1U) {
    if (rows == 0U || columns == 0U) {
      return Result<ConstMatrixView>::success(ConstMatrixView(
          storage.data(), rows, columns, rowStride, columnStride));
    }
    if (rowStride == 0U || columnStride == 0U) {
      return Result<ConstMatrixView>::failure(
          ErrorCode::InvalidExtent, "matrix view strides must be nonzero");
    }
    Extent rowOffset = 0U;
    Extent columnOffset = 0U;
    if (!checkedMultiply(rows - 1U, rowStride, rowOffset) ||
        !checkedMultiply(columns - 1U, columnStride, columnOffset)) {
      return Result<ConstMatrixView>::failure(
          ErrorCode::SizeOverflow, "SizeOverflow: matrix view offset");
    }
    Extent lastOffset = 0U;
    if (!checkedAdd(rowOffset, columnOffset, lastOffset)) {
      return Result<ConstMatrixView>::failure(
          ErrorCode::SizeOverflow, "SizeOverflow: matrix view offset");
    }
    if (lastOffset >= storage.size()) {
      return Result<ConstMatrixView>::failure(
          ErrorCode::InvalidExtent, "matrix view exceeds supplied storage");
    }
    return Result<ConstMatrixView>::success(ConstMatrixView(
        storage.data(), rows, columns, rowStride, columnStride));
  }

  /// @brief Returns the logical row count.
  [[nodiscard]] Extent rows() const noexcept { return rows_; }
  /// @brief Returns the logical column count.
  [[nodiscard]] Extent columns() const noexcept { return columns_; }
  /// @brief Returns the storage distance between adjacent rows.
  [[nodiscard]] Extent rowStride() const noexcept { return rowStride_; }
  /// @brief Returns the storage distance between adjacent columns.
  [[nodiscard]] Extent columnStride() const noexcept { return columnStride_; }
  /// @brief Reports whether either logical extent is zero.
  [[nodiscard]] bool empty() const noexcept {
    return rows_ == 0U || columns_ == 0U;
  }
  /// @brief Returns the borrowed first storage address.
  /// @note The pointer is non-owning and becomes invalid when the source
  ///       storage is destroyed, moved, or reallocated.
  [[nodiscard]] const Scalar* data() const noexcept { return data_; }

  /// @brief Accesses an element after row-then-column bounds checking.
  /// @throws std::out_of_range when either index is invalid.
  /// @note Complexity: Constant time.
  [[nodiscard]] const Scalar& at(Index row, Index column) const {
    checkRow(row);
    checkColumn(column);
    return data_ // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                 // checked offsets
        [row * rowStride_ + column * columnStride_];
  }

  /// @brief Creates a checked immutable borrowing subview retaining strides.
  /// @param originRow Logical row of the first element.
  /// @param originColumn Logical column of the first element.
  /// @param rows Requested row extent.
  /// @param columns Requested column extent.
  /// @return A subview or InvalidExtent when the requested rectangle is
  ///         outside the parent view.
  /// @post The returned view retains both parent strides and storage lifetime.
  /// @note Complexity: Constant time.
  [[nodiscard]] Result<ConstMatrixView> subview(Index originRow,
                                                Index originColumn, Extent rows,
                                                Extent columns) const {
    if (originRow > rows_ || rows > rows_ - originRow ||
        originColumn > columns_ || columns > columns_ - originColumn) {
      return Result<ConstMatrixView>::failure(
          ErrorCode::InvalidExtent, "matrix subview exceeds parent view");
    }
    if (rows == 0U || columns == 0U) {
      return Result<ConstMatrixView>::success(
          ConstMatrixView(data_, rows, columns, rowStride_, columnStride_));
    }
    return Result<ConstMatrixView>::success(ConstMatrixView(
        data_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                // checked offset
            originRow *
                rowStride_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                             // checked offset
            originColumn * columnStride_,
        rows, columns, rowStride_, columnStride_));
  }

private:
  friend class Matrix;
  friend class MatrixView;
  ConstMatrixView(const Scalar* data, Extent rows, Extent columns,
                  Extent rowStride, Extent columnStride)
      : data_(data), rows_(rows), columns_(columns), rowStride_(rowStride),
        columnStride_(columnStride) {}

  static bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
    if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
      return false;
    }
    result = lhs * rhs;
    return true;
  }

  static bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
    if (lhs > std::numeric_limits<Extent>::max() - rhs) {
      return false;
    }
    result = lhs + rhs;
    return true;
  }

  void checkRow(Index row) const {
    if (row >= rows_) {
      throw std::out_of_range("IndexOutOfRange: matrix view row " +
                              std::to_string(row));
    }
  }
  void checkColumn(Index column) const {
    if (column >= columns_) {
      throw std::out_of_range("IndexOutOfRange: matrix view column " +
                              std::to_string(column));
    }
  }

  const Scalar* data_;
  Extent rows_;
  Extent columns_;
  Extent rowStride_;
  Extent columnStride_;
};

inline MatrixView::operator ConstMatrixView() const noexcept {
  return ConstMatrixView(data_, rows_, columns_, rowStride_, columnStride_);
}

} // namespace vss::math

#endif // VSS_MATH_MATRIX_VIEW_HPP
