#ifndef VSS_MATH_VECTOR_HPP
#define VSS_MATH_VECTOR_HPP

#include <vector>
#include <vss/math/Core.hpp>

namespace vss::math {

class VectorView;
class ConstVectorView;

/// @brief Owns a contiguous one-dimensional sequence of scalar values.
///
/// Copying performs a deep copy. Moving transfers ownership and leaves the
/// source with size zero. This type does not validate whether stored values are
/// finite; operation-specific policies perform that validation later.
/// Values have no physical unit imposed by this library. The object owns its
/// storage and does not borrow from constructor arguments after construction.
/// Concurrent operations on distinct vectors are safe. Concurrent const access
/// to one vector is safe; any access that can modify the vector or its storage
/// must be externally synchronized with every aliased access.
class Vector final {
public:
  /// @brief Constructs an empty vector.
  /// @post `size() == 0` and `empty()` is `true`.
  /// @note Complexity: Constant time and constant auxiliary space.
  Vector() = default;

  /// @brief Constructs a vector with every element initialized to `value`.
  /// @param size Number of elements to own.
  /// @param value Initial value for every element.
  /// @throws std::length_error or std::bad_alloc if the requested storage is
  ///         not representable or cannot be allocated.
  /// @note Complexity: Linear initialization time and O(size) storage.
  explicit Vector(Extent size, Scalar value = 0.0);

  /// @brief Takes ownership of the supplied values.
  /// @param values Values to move into contiguous storage.
  /// @post The vector owns the supplied sequence; the argument remains valid
  ///       but its contents are unspecified by the standard move operation.
  /// @note Complexity: Constant time under the standard vector move contract.
  explicit Vector(std::vector<Scalar> values);

  /// @brief Performs a deep copy.
  /// @post Both vectors own independent storage with equal values.
  /// @throws std::bad_alloc if copying cannot allocate storage.
  /// @note Complexity: Linear in the vector extent.
  Vector(const Vector&) = default;

  /// @brief Replaces this vector with a deep copy.
  /// @post The target owns an independent copy and its prior storage is
  ///       released according to the standard vector assignment contract.
  /// @throws std::bad_alloc if copying cannot allocate storage; the standard
  ///         container assignment guarantee applies.
  /// @note Complexity: Linear in the source extent in the general case.
  Vector& operator=(const Vector&) = default;

  /// @brief Transfers storage and clears the source.
  /// @post The destination contains the former values and the source is empty.
  Vector(Vector&& other) noexcept;

  /// @brief Transfers storage and clears the source.
  /// @post The destination contains the former values and the source is empty.
  Vector& operator=(Vector&& other) noexcept;

  /// @brief Destroys the vector and its owned storage.
  /// @note Existing views and data pointers become invalid.
  ~Vector() = default;

  /// @brief Returns the number of elements.
  [[nodiscard]] Extent size() const noexcept;

  /// @brief Reports whether no elements are owned.
  [[nodiscard]] bool empty() const noexcept;

  /// @brief Returns a pointer to contiguous storage.
  /// @note The pointer is invalidated by operations that reallocate storage.
  /// @return A pointer to the first element, or the standard empty-container
  ///         pointer when the vector is empty.
  [[nodiscard]] Scalar* data() noexcept;

  /// @brief Returns a const pointer to contiguous storage.
  /// @note The pointer is invalidated by operations that reallocate storage.
  /// @return A pointer to the first element, or the standard empty-container
  ///         pointer when the vector is empty.
  [[nodiscard]] const Scalar* data() const noexcept;

  /// @brief Accesses an element after checking its index.
  /// @param index Element index in the range `[0, size())`.
  /// @return A mutable reference to the element.
  /// @throws std::out_of_range when `index >= size()`.
  /// @note Complexity: Constant time.
  [[nodiscard]] Scalar& at(Index index);

  /// @brief Accesses an element after checking its index.
  /// @param index Element index in the range `[0, size())`.
  /// @return A const reference to the element.
  /// @throws std::out_of_range when `index >= size()`.
  /// @note Complexity: Constant time.
  [[nodiscard]] const Scalar& at(Index index) const;

  /// @brief Assigns `value` to every element in linear time.
  /// @param value Value to store; finite-value validation is not performed.
  /// @post The vector's values are all equal to `value`; existing views remain
  ///       valid because no reallocation occurs.
  /// @note Complexity: Linear in the vector extent and constant auxiliary
  ///       space.
  void fill(Scalar value);

  /// @brief Returns a mutable borrowing view of all elements.
  /// @note The view is invalid after this vector is moved, destroyed, or
  /// resized.
  /// @return A mutable view whose lifetime is bounded by this vector's stable
  ///       storage lifetime.
  [[nodiscard]] VectorView view() noexcept;

  /// @brief Returns an immutable borrowing view of all elements.
  /// @note The view is invalid after this vector is moved, destroyed, or
  /// resized.
  /// @return An immutable view whose lifetime is bounded by this vector's
  ///         stable storage lifetime.
  [[nodiscard]] ConstVectorView view() const noexcept;

private:
  std::vector<Scalar> values_;
};

} // namespace vss::math

#endif // VSS_MATH_VECTOR_HPP
