#ifndef VSS_MATH_NUMERICAL_ERROR_HPP
#define VSS_MATH_NUMERICAL_ERROR_HPP

#include <stdexcept>
#include <string_view>
#include <vss/math/Error.hpp>

namespace vss::math {

/// @brief Exception adapter for failed checked numerical operations.
/// @details The originating ErrorCode is retained for callers that prefer
///          exceptions while preserving machine-readable failure categories.
class NumericalError final : public std::runtime_error {
public:
  /// @brief Constructs an exception with its numerical error category.
  /// @param code Original failure code; it must not be ErrorCode::None.
  /// @param message Diagnostic text from the failed checked operation.
  /// @throws std::invalid_argument when @p code is ErrorCode::None.
  NumericalError(ErrorCode code, std::string_view message);

  /// @brief Returns the error code carried from the checked operation.
  [[nodiscard]] ErrorCode code() const noexcept;

private:
  ErrorCode code_;
};

} // namespace vss::math

#endif // VSS_MATH_NUMERICAL_ERROR_HPP
