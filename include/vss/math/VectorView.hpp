#ifndef VSS_MATH_VECTOR_VIEW_HPP
#define VSS_MATH_VECTOR_VIEW_HPP

#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

class Vector;
class ConstVectorView;

/// @brief A checked, non-owning mutable view of strided scalar storage.
///
/// The view borrows storage, never owns it, and never extends its lifetime.
/// Concurrent reads of stable shared storage are safe; any access that can
/// modify the storage must be externally synchronized with every aliased view.
class VectorView final {
public:
  /// @brief Validates storage and creates a borrowing view.
  /// @param storage Storage that must outlive the returned view.
  /// @param size Logical number of elements.
  /// @param stride Element distance in the supplied storage.
  /// @return A view or an InvalidExtent/SizeOverflow failure.
  /// @post A non-empty view reaches exactly its logical last element without
  ///       exceeding `storage`; a zero-size view may borrow empty storage.
  /// @note Complexity: Constant time and constant auxiliary space.
  [[nodiscard]] static Result<VectorView>
  create(std::span<Scalar> storage, Extent size, Extent stride = 1U) {
    if (size == 0U) {
      return Result<VectorView>::success(
          VectorView(storage.data(), 0U, stride));
    }
    if (stride == 0U) {
      return Result<VectorView>::failure(ErrorCode::InvalidExtent,
                                         "vector view stride must be nonzero");
    }
    const Extent lastIndex = size - 1U;
    if (lastIndex > std::numeric_limits<Extent>::max() / stride) {
      return Result<VectorView>::failure(ErrorCode::SizeOverflow,
                                         "SizeOverflow: vector view offset");
    }
    const Extent lastOffset = lastIndex * stride;
    if (lastOffset >= storage.size()) {
      return Result<VectorView>::failure(
          ErrorCode::InvalidExtent, "vector view exceeds supplied storage");
    }
    return Result<VectorView>::success(
        VectorView(storage.data(), size, stride));
  }

  /// @brief Returns the logical element count.
  [[nodiscard]] Extent size() const noexcept { return size_; }

  /// @brief Returns the storage distance between adjacent logical elements.
  [[nodiscard]] Extent stride() const noexcept { return stride_; }

  /// @brief Reports whether the logical view has no elements.
  [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }

  /// @brief Returns the borrowed first storage address.
  /// @note The pointer is non-owning and becomes invalid when the source
  ///       storage is destroyed, moved, or reallocated.
  [[nodiscard]] Scalar* data() const noexcept { return data_; }

  /// @brief Accesses one logical element after bounds checking.
  /// @throws std::out_of_range when `index >= size()`.
  /// @note Complexity: Constant time.
  [[nodiscard]] Scalar& at(Index index) const {
    checkIndex(index);
    return data_ // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                 // checked offset
        [index * stride_];
  }

  /// @brief Creates a checked borrowing subview.
  /// @param origin Logical index of the first element in this view.
  /// @param size Number of logical elements requested.
  /// @return A subview or InvalidExtent when the requested range is outside
  /// the parent view.
  /// @post The returned view retains this view's stride and storage lifetime.
  /// @note Complexity: Constant time.
  [[nodiscard]] Result<VectorView> subview(Index origin, Extent size) const {
    if (origin > size_ || size > size_ - origin) {
      return Result<VectorView>::failure(ErrorCode::InvalidExtent,
                                         "vector subview exceeds parent view");
    }
    if (size == 0U) {
      return Result<VectorView>::success(VectorView(data_, 0U, stride_));
    }
    return Result<VectorView>::success(VectorView(
        data_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                // checked offset
            origin * stride_,
        size, stride_));
  }

  /// @brief Converts a mutable view to an immutable borrowing view.
  /// @return A view of the same storage, extents, and stride without mutation
  ///         capability.
  operator ConstVectorView() const noexcept;

private:
  friend class Vector;
  VectorView(Scalar* data, Extent size, Extent stride)
      : data_(data), size_(size), stride_(stride) {}

  void checkIndex(Index index) const {
    if (index >= size_) {
      throw std::out_of_range("IndexOutOfRange: vector view index " +
                              std::to_string(index));
    }
  }

  Scalar* data_;
  Extent size_;
  Extent stride_;
};

/// @brief A checked, non-owning immutable view of strided scalar storage.
///
/// The view borrows storage and never extends its lifetime. Concurrent reads
/// are safe when the underlying storage remains alive and is not concurrently
/// modified; synchronization is the caller's responsibility otherwise.
class ConstVectorView final {
public:
  /// @brief Validates storage and creates an immutable borrowing view.
  /// @param storage Storage that must outlive the returned view.
  /// @param size Logical number of elements.
  /// @param stride Element distance in the supplied storage.
  /// @return A view or an InvalidExtent/SizeOverflow failure.
  /// @post A non-empty view reaches exactly its logical last element without
  ///       exceeding `storage`; a zero-size view may borrow empty storage.
  /// @note Complexity: Constant time and constant auxiliary space.
  [[nodiscard]] static Result<ConstVectorView>
  create(std::span<const Scalar> storage, Extent size, Extent stride = 1U) {
    if (size == 0U) {
      return Result<ConstVectorView>::success(
          ConstVectorView(storage.data(), 0U, stride));
    }
    if (stride == 0U) {
      return Result<ConstVectorView>::failure(
          ErrorCode::InvalidExtent, "vector view stride must be nonzero");
    }
    const Extent lastIndex = size - 1U;
    if (lastIndex > std::numeric_limits<Extent>::max() / stride) {
      return Result<ConstVectorView>::failure(
          ErrorCode::SizeOverflow, "SizeOverflow: vector view offset");
    }
    const Extent lastOffset = lastIndex * stride;
    if (lastOffset >= storage.size()) {
      return Result<ConstVectorView>::failure(
          ErrorCode::InvalidExtent, "vector view exceeds supplied storage");
    }
    return Result<ConstVectorView>::success(
        ConstVectorView(storage.data(), size, stride));
  }

  /// @brief Returns the logical element count.
  [[nodiscard]] Extent size() const noexcept { return size_; }

  /// @brief Returns the storage distance between adjacent logical elements.
  [[nodiscard]] Extent stride() const noexcept { return stride_; }

  /// @brief Reports whether the logical view has no elements.
  [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }

  /// @brief Returns the borrowed first storage address.
  /// @note The pointer is non-owning and becomes invalid when the source
  ///       storage is destroyed, moved, or reallocated.
  [[nodiscard]] const Scalar* data() const noexcept { return data_; }

  /// @brief Accesses one logical element after bounds checking.
  /// @throws std::out_of_range when `index >= size()`.
  /// @note Complexity: Constant time.
  [[nodiscard]] const Scalar& at(Index index) const {
    checkIndex(index);
    return data_ // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                 // checked offset
        [index * stride_];
  }

  /// @brief Creates a checked immutable borrowing subview.
  /// @param origin Logical index of the first element in this view.
  /// @param size Number of logical elements requested.
  /// @return A subview or InvalidExtent when the requested range is outside
  ///         the parent view.
  /// @post The returned view retains this view's stride and storage lifetime.
  /// @note Complexity: Constant time.
  [[nodiscard]] Result<ConstVectorView> subview(Index origin,
                                                Extent size) const {
    if (origin > size_ || size > size_ - origin) {
      return Result<ConstVectorView>::failure(
          ErrorCode::InvalidExtent, "vector subview exceeds parent view");
    }
    if (size == 0U) {
      return Result<ConstVectorView>::success(
          ConstVectorView(data_, 0U, stride_));
    }
    return Result<ConstVectorView>::success(ConstVectorView(
        data_ + // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                // checked offset
            origin * stride_,
        size, stride_));
  }

private:
  friend class Vector;
  friend class VectorView;
  ConstVectorView(const Scalar* data, Extent size, Extent stride)
      : data_(data), size_(size), stride_(stride) {}

  void checkIndex(Index index) const {
    if (index >= size_) {
      throw std::out_of_range("IndexOutOfRange: vector view index " +
                              std::to_string(index));
    }
  }

  const Scalar* data_;
  Extent size_;
  Extent stride_;
};

inline VectorView::operator ConstVectorView() const noexcept {
  return ConstVectorView(data_, size_, stride_);
}

} // namespace vss::math

#endif // VSS_MATH_VECTOR_VIEW_HPP
