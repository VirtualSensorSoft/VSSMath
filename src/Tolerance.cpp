#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

Status validate(Tolerance tolerance) {
  if (!isFinite(tolerance.absolute) || tolerance.absolute < 0.0) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "absolute tolerance must be finite and non-negative");
  }
  if (!isFinite(tolerance.relative) || tolerance.relative < 0.0) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "relative tolerance must be finite and non-negative");
  }
  return Status::success();
}

bool isFinite(Scalar value) noexcept { return std::isfinite(value); }

bool nearlyEqual(Scalar lhs, Scalar rhs, Tolerance tolerance) {
  const Status validation = validate(tolerance);
  if (!validation.ok()) {
    throw std::invalid_argument(std::string(validation.message()));
  }

  if (std::isnan(lhs) || std::isnan(rhs)) {
    return false;
  }
  if (lhs == rhs) {
    return true;
  }
  if (!isFinite(lhs) || !isFinite(rhs)) {
    return false;
  }

  const Scalar difference = std::abs(lhs - rhs);
  const Scalar scale = std::max(std::abs(lhs), std::abs(rhs));
  const Scalar limit = std::max(tolerance.absolute, tolerance.relative * scale);
  // Account for one representational rounding step at the operand scale.
  const Scalar rounding =
      std::numeric_limits<Scalar>::epsilon() * std::max(1.0, scale);
  return difference <= limit + rounding;
}

} // namespace vss::math
