#ifndef VSS_MATH_DIFFERENTIATION_FORWARD_AD_HPP
#define VSS_MATH_DIFFERENTIATION_FORWARD_AD_HPP

#include <cstdint>
#include <functional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::differentiation {

/// @brief Supported checked scalar unary operations for forward-mode duals.
enum class DualUnaryOp : std::uint8_t {
  negate, ///< Negation; derivative multiplier is -1.
  exp,    ///< Exponential; derivative multiplier is exp(x).
  log,    ///< Natural logarithm; requires x > 0.
  log1p,  ///< log(1+x); requires x > -1.
  expm1,  ///< exp(x)-1, evaluated with the standard stable primitive.
  sqrt,   ///< Square root; differentiable domain requires x > 0.
  sin,    ///< Sine.
  cos,    ///< Cosine.
  tanh,   ///< Hyperbolic tangent.
  abs     ///< Absolute value; x == 0 is rejected as nondifferentiable.
};

/// @brief Supported checked scalar binary operations for forward-mode duals.
enum class DualBinaryOp : std::uint8_t {
  add,      ///< Addition.
  subtract, ///< Subtraction.
  multiply, ///< Multiplication.
  divide,   ///< Division; the right primal must be nonzero.
  power     ///< Real power; the left primal must be positive.
};

/// @brief A scalar primal and its directional derivatives.
struct Dual final {
  Scalar primal;  ///< Finite scalar value.
  Vector tangent; ///< Owned tangent components, one per active seed direction.
};

/// @brief Owned value and output-by-input Jacobian from forward AD.
struct ADResult final {
  Vector value;    ///< Function outputs in callback order.
  Matrix jacobian; ///< Shape [output count, input count].
};

/// @brief Synchronous callable receiving a borrowed dual input sequence.
/// @details The input span and its tangents are valid only during the call.
///          The callable must return one owned dual per function output, each
///          with tangent width equal to the current seed block width.
using DualFunction = std::function<Result<Array<Dual>>(std::span<const Dual>)>;

/// @brief Applies one checked unary dual operation.
[[nodiscard]] Result<Dual> dualUnary(DualUnaryOp operation, const Dual& value,
                                     Limits limits);
/// @brief Applies one checked binary dual operation.
[[nodiscard]] Result<Dual> dualBinary(DualBinaryOp operation, const Dual& left,
                                      const Dual& right, Limits limits);
/// @brief Evaluates a vector-valued callable in bounded forward-seed blocks.
[[nodiscard]] Result<ADResult>
forwardAutomaticDerivative(const DualFunction& function,
                           ConstVectorView parameters, Index blockWidth,
                           Limits limits);

} // namespace vss::math::differentiation

#endif // VSS_MATH_DIFFERENTIATION_FORWARD_AD_HPP
