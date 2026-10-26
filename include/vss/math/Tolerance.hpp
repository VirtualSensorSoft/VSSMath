#ifndef VSS_MATH_TOLERANCE_HPP
#define VSS_MATH_TOLERANCE_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>

namespace vss::math {

/// @brief Absolute and relative tolerances for approximate comparisons.
struct Tolerance final {
  /// @brief Maximum absolute difference accepted by a comparison.
  /// @note This is a dimensionless scalar unless the caller assigns a domain
  ///       unit. It must be finite and non-negative when passed to
  ///       `validate()` or `nearlyEqual()`.
  Scalar absolute{1.0e-12};
  /// @brief Maximum difference proportional to the larger operand magnitude.
  /// @note This is a dimensionless scalar unless the caller assigns a domain
  ///       unit. It must be finite and non-negative when passed to
  ///       `validate()` or `nearlyEqual()`.
  Scalar relative{1.0e-10};
};

/// @brief Common options controlling a numerical operation.
struct OperationOptions final {
  /// @brief Tolerance used by operations that compare numerical values.
  /// @note Basic Core operations validate this value uniformly even when their
  ///       arithmetic does not compare values.
  Tolerance tolerance{};
  /// @brief Whether the operation requires finite inputs and results.
  /// @note `true` returns `NonFiniteInput` or `NonFiniteResult`; `false`
  ///       permits the implementation's ordinary IEEE arithmetic for this
  ///       call only.
  bool requireFinite{true};
};

/// @brief Validates both fields of a tolerance.
/// @param tolerance Tolerance to validate.
/// @return Success, or `ErrorCode::InvalidTolerance` with a diagnostic.
/// @post Inputs are not modified.
/// @note Complexity: Constant time and constant auxiliary space.
[[nodiscard]] Status validate(Tolerance tolerance);

/// @brief Tests whether a scalar is finite.
/// @param value Scalar to inspect.
/// @return `true` for neither NaN nor positive or negative infinity.
/// @note Complexity: Constant time and constant auxiliary space.
[[nodiscard]] bool isFinite(Scalar value) noexcept;

/// @brief Compares two finite scalars using absolute and relative tolerances.
///
/// NaN operands compare unequal. Equal same-signed infinities compare equal;
/// all other non-finite operands compare unequal. The tolerance is validated
/// before any equality decision.
/// @param lhs Left-hand scalar.
/// @param rhs Right-hand scalar.
/// @param tolerance Comparison tolerance.
/// @return Whether the operands are approximately equal.
/// @throws std::invalid_argument if `tolerance` is invalid.
/// @note Complexity: Constant time and constant auxiliary space.
[[nodiscard]] bool nearlyEqual(Scalar lhs, Scalar rhs,
                               Tolerance tolerance = {});

} // namespace vss::math

#endif // VSS_MATH_TOLERANCE_HPP
