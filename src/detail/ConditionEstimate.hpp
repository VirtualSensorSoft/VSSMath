#ifndef VSS_MATH_DETAIL_CONDITION_ESTIMATE_HPP
#define VSS_MATH_DETAIL_CONDITION_ESTIMATE_HPP

#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>

namespace vss::math::detail {

/// @brief Estimates a represented matrix's one-norm condition number.
/// @details The callable solves one basis-vector inverse action at a time;
///          this helper never constructs or exposes an inverse matrix. Basis
///          vectors and reductions are processed in increasing logical order.
///          A finite-input norm-product overflow is preserved as positive
///          infinity because it is an allowed condition diagnostic.
template <class InverseAction>
[[nodiscard]] Result<Scalar>
estimateOneNormCondition(Extent size, Scalar matrixOneNorm,
                         InverseAction& inverseAction) {
  if (size == 0U || !std::isfinite(matrixOneNorm) || matrixOneNorm < 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidExtent,
        "condition estimate requires a non-empty finite matrix norm");
  }

  Scalar inverseOneNorm = 0.0;
  for (Index column = 0U; column < size; ++column) {
    auto action = inverseAction(column);
    if (!action.hasValue()) {
      return Result<Scalar>::failure(action.status().code(),
                                     std::string(action.status().message()));
    }
    CompensatedSum columnSum;
    const Vector& response = action.value();
    if (response.size() != size) {
      return Result<Scalar>::failure(
          ErrorCode::DimensionMismatch,
          "condition inverse action returned the wrong extent");
    }
    for (Index row = 0U; row < size; ++row) {
      const Scalar magnitude = std::abs(response.at(row));
      if (!std::isfinite(magnitude)) {
        return Result<Scalar>::failure(
            ErrorCode::NonFiniteResult,
            "condition inverse action produced a non-finite value");
      }
      columnSum.add(magnitude);
    }
    const Scalar columnNorm = columnSum.total();
    if (!std::isfinite(columnNorm)) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "condition inverse action norm is not finite");
    }
    inverseOneNorm = std::max(inverseOneNorm, columnNorm);
  }

  const Scalar estimate = matrixOneNorm * inverseOneNorm;
  if (std::isnan(estimate)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "condition norm product is not a number");
  }
  if (!std::isfinite(estimate) &&
      estimate != std::numeric_limits<Scalar>::infinity()) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "condition norm product is not a valid diagnostic");
  }
  return Result<Scalar>::success(estimate);
}

} // namespace vss::math::detail

#endif // VSS_MATH_DETAIL_CONDITION_ESTIMATE_HPP
