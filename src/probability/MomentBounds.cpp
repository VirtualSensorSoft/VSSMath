#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vss/math/Error.hpp>
#include <vss/math/probability/MomentBounds.hpp>

namespace vss::math::probability {
namespace {
constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();

Result<Scalar> validate(Scalar variance, Scalar radius) {
  if (!std::isfinite(variance) || !std::isfinite(radius)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "variance and radius must be finite");
  }
  if (variance < 0.0 || radius <= 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "variance must be nonnegative and radius positive");
  }
  return Result<Scalar>::success(variance);
}

Scalar upward(Scalar value) noexcept { return std::nextafter(value, infinity); }

Scalar downward(Scalar value) noexcept { return std::nextafter(value, 0.0); }

Scalar positiveUpper(Scalar value) noexcept {
  if (value == 0.0) {
    return std::numeric_limits<Scalar>::denorm_min();
  }
  return std::min(value, 1.0);
}
} // namespace

Result<Scalar> chebyshevBound(Scalar variance, Scalar radius) {
  const auto checked = validate(variance, radius);
  if (!checked) {
    return Result<Scalar>::failure(checked.status().code(),
                                   std::string(checked.status().message()));
  }
  if (variance == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  const Scalar rootUpper = upward(std::sqrt(variance));
  if (radius <= downward(std::sqrt(variance))) {
    return Result<Scalar>::success(1.0);
  }
  const Scalar ratio = std::min(1.0, upward(rootUpper / radius));
  const Scalar square = std::min(1.0, upward(ratio * ratio));
  return Result<Scalar>::success(positiveUpper(square));
}

Result<Scalar> cantelliBound(Scalar variance, Scalar radius) {
  const auto checked = validate(variance, radius);
  if (!checked) {
    return Result<Scalar>::failure(checked.status().code(),
                                   std::string(checked.status().message()));
  }
  if (variance == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  const Scalar rootUpper = upward(std::sqrt(variance));
  Scalar result = 0.0;
  if (radius >= std::sqrt(variance)) {
    const Scalar ratio = upward(rootUpper / radius);
    const Scalar square = upward(ratio * ratio);
    const Scalar denominatorLower = downward(1.0 + square);
    result = upward(square / denominatorLower);
  } else {
    const Scalar ratioLower = downward(radius / rootUpper);
    const Scalar squareLower = downward(ratioLower * ratioLower);
    const Scalar denominatorLower = downward(1.0 + squareLower);
    result = upward(1.0 / denominatorLower);
  }
  return Result<Scalar>::success(std::clamp(result, 0.0, 1.0));
}

} // namespace vss::math::probability
