#ifndef VSS_MATH_ARRAY_HPP
#define VSS_MATH_ARRAY_HPP

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

/// @brief Owns a bounded sequence while exposing only immutable views.
/// @details `Array` provides value semantics for index lists and other
///          auxiliary records used by numerical APIs. Its storage container is
///          private; callers can copy input values through `copy()` and borrow
///          a `std::span<const T>` while this object remains alive and
///          unmodified. The element and byte ceilings are checked before the
///          owned sequence is allocated. Copying an existing Array performs a
///          deep copy and follows the normal allocation exception contract.
/// @tparam T Copy-constructible element type.
template <class T> class Array final {
  static_assert(std::is_copy_constructible_v<T>);

public:
  /// @brief Constructs an empty array.
  Array() noexcept = default;

  /// @brief Creates an owning copy of a borrowed sequence.
  /// @param values Borrowed input; no reference to it is retained.
  /// @param limits Maximum owned scalar count and bytes.
  /// @return An owning copy, or `ResourceLimit` when either ceiling is
  ///         exceeded.
  /// @throws std::bad_alloc if permitted storage cannot be allocated.
  [[nodiscard]] static Result<Array> copy(std::span<const T> values,
                                          Limits limits) {
    if (values.size() > limits.elements ||
        values.size() > limits.bytes / sizeof(T)) {
      return Result<Array>::failure(ErrorCode::ResourceLimit,
                                    "Array exceeds configured storage limits");
    }
    Array result;
    result.values_.assign(values.begin(), values.end());
    return Result<Array>::success(std::move(result));
  }

  /// @brief Returns an immutable borrowed view of the owned values.
  /// @return A view valid until this array is destroyed or replaced.
  [[nodiscard]] std::span<const T> view() const noexcept {
    return std::span<const T>{values_};
  }

private:
  std::vector<T> values_;
};

} // namespace vss::math

#endif // VSS_MATH_ARRAY_HPP
