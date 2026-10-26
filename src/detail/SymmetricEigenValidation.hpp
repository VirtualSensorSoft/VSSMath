#ifndef VSS_MATH_DETAIL_SYMMETRIC_EIGEN_VALIDATION_HPP
#define VSS_MATH_DETAIL_SYMMETRIC_EIGEN_VALIDATION_HPP

#include <cmath>
#include <vss/math/Error.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::linear::detail {

inline bool validRankTolerance(RankTolerance tolerance) noexcept {
  return std::isfinite(tolerance.absolute) &&
         std::isfinite(tolerance.relative) && tolerance.absolute >= 0.0 &&
         tolerance.relative >= 0.0;
}

inline bool validSymmetryTolerance(Tolerance tolerance) noexcept {
  return std::isfinite(tolerance.absolute) &&
         std::isfinite(tolerance.relative) && tolerance.absolute >= 0.0 &&
         tolerance.relative >= 0.0;
}

inline Status
validateSymmetricEigenOptions(const SymmetricEigenOptions& options) {
  if (options.maximumSweeps == 0U || options.maximumSweeps > 1024U) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "symmetric eigendecomposition sweep limit must be in [1, 1024]");
  }
  if (!validRankTolerance(options.rankTolerance) ||
      !validSymmetryTolerance(options.symmetryTolerance)) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "symmetric eigendecomposition tolerance is invalid");
  }
  return Status::success();
}

} // namespace vss::math::linear::detail

#endif // VSS_MATH_DETAIL_SYMMETRIC_EIGEN_VALIDATION_HPP
