#ifndef VSS_MATH_RESULT_HPP
#define VSS_MATH_RESULT_HPP

#include <optional>
#include <stdexcept>
#include <utility>
#include <vss/math/Error.hpp>

namespace vss::math {

/// @brief Holds either one value or a diagnostic status.
///
/// A successful result owns its value. References returned by @ref value remain
/// valid only while the Result object is alive and its value is not moved from.
/// A failed result never contains a value.
/// @details Result has no shared mutable state. Distinct results may be used
///          concurrently; concurrent access to one result requires the caller
///          to prevent conflicting mutation or moves.
/// @tparam T Value type; it need not be copyable.
template <class T> class Result final {
public:
  /// @brief Creates a successful result by moving `value` into owned storage.
  /// @param value Value to store.
  /// @return A result containing `value`.
  /// @throws std::bad_alloc if construction of owned storage or its status
  ///         diagnostic cannot allocate.
  [[nodiscard]] static Result success(T value) {
    return Result(std::optional<T>{std::move(value)}, Status::success());
  }

  /// @brief Creates a failed result with no value.
  /// @param code Failure code; `ErrorCode::None` is invalid.
  /// @param message Non-empty diagnostic message.
  /// @return A result containing the validated failure status.
  /// @throws std::invalid_argument if the status arguments are invalid.
  /// @throws std::bad_alloc if the diagnostic cannot be allocated.
  [[nodiscard]] static Result failure(ErrorCode code, std::string message) {
    return Result(std::nullopt, Status::failure(code, std::move(message)));
  }

  /// @brief Reports whether this result owns a value.
  /// @return `true` for a successful result and `false` for a failed result.
  /// @note Complexity: Constant time.
  [[nodiscard]] bool hasValue() const noexcept { return value_.has_value(); }

  /// @brief Explicitly converts this result to its value-presence state.
  /// @return The same result as @ref hasValue.
  /// @note Complexity: Constant time.
  explicit operator bool() const noexcept { return hasValue(); }

  /// @brief Accesses the owned value from a mutable lvalue.
  /// @return A mutable reference whose lifetime is tied to this Result.
  /// @throws std::logic_error if this result is failed.
  /// @note Complexity: Constant time.
  [[nodiscard]] T& value() & {
    ensureValue();
    return *value_; // NOLINT(bugprone-unchecked-optional-access): ensureValue
                    // establishes the value invariant and preserves the
                    // documented logic_error.
  }

  /// @brief Accesses the owned value from a const lvalue.
  /// @return A const reference whose lifetime is tied to this Result.
  /// @throws std::logic_error if this result is failed.
  /// @note Complexity: Constant time.
  [[nodiscard]] const T& value() const& {
    ensureValue();
    return *value_; // NOLINT(bugprone-unchecked-optional-access): ensureValue
                    // establishes the value invariant and preserves the
                    // documented logic_error.
  }

  /// @brief Moves the owned value from an rvalue result.
  /// @return An rvalue reference to the owned value.
  /// @throws std::logic_error if this result is failed.
  /// @post The returned value may be moved from; the Result remains valid but
  ///       its value state is caller-managed after the move.
  /// @note Complexity: Constant time.
  [[nodiscard]] T&& value() && {
    ensureValue();
    return std::move(*value_); // NOLINT(bugprone-unchecked-optional-access):
                               // ensureValue establishes the value invariant
                               // and preserves the documented logic_error.
  }

  /// @brief Returns the status associated with this result.
  /// @return A reference valid while this Result object remains alive.
  /// @note Complexity: Constant time.
  [[nodiscard]] const Status& status() const noexcept { return status_; }

private:
  Result(std::optional<T> value, Status status)
      : value_(std::move(value)), status_(std::move(status)) {}

  void ensureValue() const {
    if (!value_) {
      throw std::logic_error("cannot access the value of a failed result");
    }
  }

  std::optional<T> value_;
  Status status_;
};

} // namespace vss::math

#endif // VSS_MATH_RESULT_HPP
