#ifndef VSS_MATH_MATRIX_HPP
#define VSS_MATH_MATRIX_HPP

#include <vector>
#include <vss/math/Core.hpp>

namespace vss::math {

class MatrixView;
class ConstMatrixView;

/// @brief Owns a contiguous row-major two-dimensional scalar array.
///
/// Copying performs a deep copy. Moving transfers storage and leaves the
/// source with zero rows and columns. Zero-area shapes preserve their requested
/// extents while owning no elements. Stored values are not finite-validated.
/// Values have no physical unit imposed by this library. The object owns its
/// storage and does not borrow from constructor arguments after construction.
/// Concurrent operations on distinct matrices are safe. Concurrent const
/// access to one matrix is safe; any access that can modify the matrix or its
/// storage must be externally synchronized with every aliased access.
class Matrix final {
public:
  /// @brief Constructs an empty matrix with zero rows and columns.
  /// @post `rows() == 0`, `columns() == 0`, and `empty()` is `true`.
  /// @note Complexity: Constant time and constant auxiliary space.
  Matrix() = default;

  /// @brief Constructs a row-major matrix initialized to `value`.
  /// @param rows Number of rows.
  /// @param columns Number of columns.
  /// @param value Initial value for every element.
  /// @throws std::length_error if `rows * columns` overflows.
  /// @throws std::bad_alloc if the representable storage cannot be allocated.
  /// @post Zero-area shapes retain both requested extents and own no elements.
  /// @note Complexity: Linear initialization time and O(rows * columns)
  ///       storage.
  Matrix(Extent rows, Extent columns, Scalar value = 0.0);

  /// @brief Performs a deep copy.
  /// @post Both matrices own independent storage with equal shape and values.
  /// @throws std::bad_alloc if copying cannot allocate storage.
  /// @note Complexity: Linear in the number of elements.
  Matrix(const Matrix&) = default;

  /// @brief Replaces this matrix with a deep copy.
  /// @post The target owns an independent copy of the source shape and values.
  /// @throws std::bad_alloc if copying cannot allocate storage; the standard
  ///         container assignment guarantee applies.
  /// @note Complexity: Linear in the source element count in the general case.
  Matrix& operator=(const Matrix&) = default;

  /// @brief Transfers storage and resets the source shape to zero.
  /// @post The destination contains the former shape and values; the source
  ///       has zero rows, zero columns, and zero elements.
  Matrix(Matrix&& other) noexcept;

  /// @brief Transfers storage and resets the source shape to zero.
  /// @post The destination contains the former shape and values; the source
  ///       has zero rows, zero columns, and zero elements.
  Matrix& operator=(Matrix&& other) noexcept;

  /// @brief Destroys the matrix and its owned storage.
  /// @note Existing views and data pointers become invalid.
  ~Matrix() = default;

  /// @brief Returns the row extent.
  [[nodiscard]] Extent rows() const noexcept;

  /// @brief Returns the column extent.
  [[nodiscard]] Extent columns() const noexcept;

  /// @brief Returns the number of owned elements.
  [[nodiscard]] Extent size() const noexcept;

  /// @brief Reports whether the matrix owns no elements.
  [[nodiscard]] bool empty() const noexcept;

  /// @brief Returns a pointer to contiguous row-major storage.
  /// @note The pointer is invalidated by operations that reallocate storage.
  /// @return A pointer to the first element, or the standard empty-container
  ///         pointer when the matrix owns no elements.
  [[nodiscard]] Scalar* data() noexcept;

  /// @brief Returns a const pointer to contiguous row-major storage.
  /// @note The pointer is invalidated by operations that reallocate storage.
  /// @return A pointer to the first element, or the standard empty-container
  ///         pointer when the matrix owns no elements.
  [[nodiscard]] const Scalar* data() const noexcept;

  /// @brief Accesses an element after checking row then column bounds.
  /// @param row Row index in the range `[0, rows())`.
  /// @param column Column index in the range `[0, columns())`.
  /// @return A mutable reference to the element.
  /// @throws std::out_of_range when either index is invalid.
  /// @note Complexity: Constant time.
  [[nodiscard]] Scalar& at(Index row, Index column);

  /// @brief Accesses an element after checking row then column bounds.
  /// @param row Row index in the range `[0, rows())`.
  /// @param column Column index in the range `[0, columns())`.
  /// @return A const reference to the element.
  /// @throws std::out_of_range when either index is invalid.
  /// @note Complexity: Constant time.
  [[nodiscard]] const Scalar& at(Index row, Index column) const;

  /// @brief Assigns `value` to every element in linear time.
  /// @param value Value to store; finite-value validation is not performed.
  /// @post Every stored element equals `value`; existing views remain valid.
  /// @note Complexity: Linear in the number of elements and constant auxiliary
  ///       space.
  void fill(Scalar value);

  /// @brief Returns a mutable borrowing view of all elements.
  /// @note The view is invalid after this matrix is moved, destroyed, or
  /// resized.
  /// @return A mutable row-major view whose lifetime is bounded by this
  ///         matrix's stable storage lifetime.
  [[nodiscard]] MatrixView view() noexcept;

  /// @brief Returns an immutable borrowing view of all elements.
  /// @note The view is invalid after this matrix is moved, destroyed, or
  /// resized.
  /// @return An immutable row-major view whose lifetime is bounded by this
  ///         matrix's stable storage lifetime.
  [[nodiscard]] ConstMatrixView view() const noexcept;

private:
  Extent rows_{0};
  Extent columns_{0};
  std::vector<Scalar> values_;
};

} // namespace vss::math

#endif // VSS_MATH_MATRIX_HPP
