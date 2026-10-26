#ifndef VSS_MATH_ERROR_HPP
#define VSS_MATH_ERROR_HPP

#include <string>
#include <string_view>
#include <vss/math/Core.hpp>

namespace vss::math {

/// @brief Categorizes a failed mathematical operation.
/// @details `None` represents success. The remaining values are stable
///          categories for expected numerical and shape failures; callers
///          should branch on this enum rather than parse diagnostic text.
enum class ErrorCode : std::uint8_t {
  /// @brief Indicates that no error occurred.
  None,
  /// @brief An extent was invalid for the requested object.
  InvalidExtent,
  /// @brief A size calculation overflowed its representation.
  SizeOverflow,
  /// @brief An index was outside the valid range.
  IndexOutOfRange,
  /// @brief Operand dimensions were incompatible.
  DimensionMismatch,
  /// @brief An input contained a non-finite value.
  NonFiniteInput,
  /// @brief An operation produced a non-finite value.
  NonFiniteResult,
  /// @brief A tolerance contained an invalid value.
  InvalidTolerance,
  /// @brief A required factorization or solve was singular.
  Singular,
  /// @brief A required factorization or solve was rank deficient.
  RankDeficient,
  /// @brief A numerical operation was judged ill-conditioned.
  IllConditioned,
  /// @brief A matrix failed the requested symmetry check.
  NotSymmetric,
  /// @brief A matrix failed the positive-definiteness check.
  NotPositiveDefinite,
  /// @brief A matrix failed the positive-semidefiniteness check.
  NotPositiveSemidefinite,
  /// @brief An input lies outside the mathematical domain of an operation.
  InvalidDomain,
  /// @brief An operation received fewer samples than its contract requires.
  InsufficientSamples,
  /// @brief A user callback reported an operation failure.
  CallbackFailure,
  /// @brief An iterative operation did not meet its convergence contract.
  NoConvergence
};

/// @brief Describes success or failure without throwing for normal outcomes.
/// @details A successful status has code `None` and an empty message. A failed
///          status has a non-`None` code and owns a non-empty diagnostic.
///          Status has no global state and is safe to read concurrently when
///          the same object is not being modified.
class Status final {
public:
  /// @brief Creates a successful status with code `ErrorCode::None`.
  /// @post `ok()` is true, `code()` is `None`, and `message()` is empty.
  [[nodiscard]] static Status success() noexcept;

  /// @brief Creates a failed status after validating its error code and
  /// message.
  /// @param code A failure code; `ErrorCode::None` is invalid here.
  /// @param message A non-empty diagnostic message owned by the returned
  /// status.
  /// @throws std::invalid_argument if `code` is `None` or `message` is empty.
  [[nodiscard]] static Status failure(ErrorCode code, std::string message);

  /// @brief Reports whether this status represents success.
  /// @return `true` only when the code is `ErrorCode::None`.
  /// @note Complexity: Constant time.
  [[nodiscard]] bool ok() const noexcept;

  /// @brief Returns the status error code.
  /// @return The stored error code.
  /// @note Complexity: Constant time.
  [[nodiscard]] ErrorCode code() const noexcept;

  /// @brief Returns the diagnostic message without copying it.
  /// @return A view valid while this status object remains alive and unchanged.
  /// @note Complexity: Constant time.
  [[nodiscard]] std::string_view message() const noexcept;

private:
  Status(ErrorCode code, std::string message);

  ErrorCode code_;
  std::string message_;
};

} // namespace vss::math

#endif // VSS_MATH_ERROR_HPP
