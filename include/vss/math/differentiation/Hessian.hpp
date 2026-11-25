#ifndef VSS_MATH_DIFFERENTIATION_HESSIAN_HPP
#define VSS_MATH_DIFFERENTIATION_HESSIAN_HPP

#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>

namespace vss::math::differentiation {

/// @brief Controls finite-difference Hessian steps and refinement.
/// @details The step span is borrowed only for the duration of the call.
struct HessianPolicy final {
  /// @brief Coordinate scales, explicit steps, and optional closed bounds.
  std::span<const ParameterStep> steps;
  /// @brief Relative consistency threshold between successive estimates.
  Scalar check_tolerance;
  /// @brief Number of step halvings allowed after the initial estimate.
  Index halvings;
  /// @brief Allows bound-aware one-sided stencils when central ones do not fit.
  bool one_sided;
};

/// @brief Estimates a scalar function's symmetric Hessian.
[[nodiscard]] Result<Matrix> numericalHessian(const ScalarFunction& function,
                                              ConstVectorView parameters,
                                              const HessianPolicy& policy,
                                              Limits limits);

/// @brief Estimates one symmetric Hessian per vector-function output.
/// @details The returned tensor axes are [output, parameter, parameter].
[[nodiscard]] Result<Tensor>
numericalOutputHessians(const VectorFunction& function,
                        ConstVectorView parameters, const HessianPolicy& policy,
                        Limits limits);

/// @brief Estimates the Hessian multiplied by a direction without returning it.
[[nodiscard]] Result<Vector>
hessianVectorProduct(const ScalarFunction& function, ConstVectorView parameters,
                     ConstVectorView direction, const HessianPolicy& policy,
                     Limits limits);

} // namespace vss::math::differentiation

#endif // VSS_MATH_DIFFERENTIATION_HESSIAN_HPP
