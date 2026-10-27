#ifndef VSS_MATH_DETAIL_NUMERICAL_CLASSIFICATION_HPP
#define VSS_MATH_DETAIL_NUMERICAL_CLASSIFICATION_HPP

#include <cmath>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>

namespace vss::math::detail {

/// @brief Classifies a normalized negative-variance roundoff magnitude.
/// @param candidateNorm Non-negative normalized candidate magnitude.
/// @param reference Non-negative normalized reference magnitude.
/// @return `None` when the candidate is within the prescribed roundoff band;
///         otherwise `NonFiniteResult`.
/// @pre Both arguments are finite and non-negative.
/// @note Complexity: Constant time and constant auxiliary storage.
[[nodiscard]] inline ErrorCode
classifyVarianceRoundoff(Scalar candidateNorm, Scalar reference) noexcept {
  if (!std::isfinite(candidateNorm) || candidateNorm < 0.0 ||
      !std::isfinite(reference) || reference < 0.0) {
    return ErrorCode::NonFiniteResult;
  }
  const Scalar threshold =
      32.0 * std::numeric_limits<Scalar>::epsilon() * reference;
  if (!std::isfinite(threshold)) {
    return ErrorCode::NonFiniteResult;
  }
  return candidateNorm <= threshold ? ErrorCode::None
                                    : ErrorCode::NonFiniteResult;
}

/// @brief Validates and gently clamps a probability to the unit interval.
/// @param candidate Probability candidate to classify.
/// @return The unchanged candidate when it is in `[0,1]`, the nearest endpoint
///         for a roundoff excursion no larger than `32*epsilon`, or a failed
///         result with `NonFiniteResult`.
/// @note Complexity: Constant time and constant auxiliary storage.
[[nodiscard]] inline Result<Scalar> classifyUnitProbability(Scalar candidate) {
  if (!std::isfinite(candidate)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "probability candidate is non-finite");
  }
  if (candidate >= 0.0 && candidate <= 1.0) {
    return Result<Scalar>::success(candidate);
  }
  const Scalar distance = candidate < 0.0 ? -candidate : candidate - 1.0;
  const Scalar limit = 32.0 * std::numeric_limits<Scalar>::epsilon();
  if (std::isfinite(distance) && distance <= limit) {
    return Result<Scalar>::success(candidate < 0.0 ? 0.0 : 1.0);
  }
  return Result<Scalar>::failure(
      ErrorCode::NonFiniteResult,
      "probability candidate lies outside the unit interval");
}

} // namespace vss::math::detail

#endif // VSS_MATH_DETAIL_NUMERICAL_CLASSIFICATION_HPP
