#ifndef VSS_MATH_ROOTS_HPP
#define VSS_MATH_ROOTS_HPP

#include <functional>
#include <optional>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Owns a scalar root candidate and its convergence evidence.
struct RootResult final {
  Scalar value{};    ///< Final scalar root candidate.
  Scalar residual{}; ///< Function value at the candidate.
  std::optional<Interval>
      enclosure;                ///< Final sign-changing bracket, when retained.
  Index function_evaluations{}; ///< Number of function callback evaluations.
  Index derivative_evaluations{}; ///< Number of derivative evaluations.
  Index iterations{};             ///< Number of solver iterations completed.
  Termination termination{
      Termination::converged}; ///< Completion or stop reason.
};

/// @brief Vector-valued residual callback; its input view expires on return.
using VectorFunction = std::function<Result<Vector>(ConstVectorView)>;

/// @brief Jacobian callback returning an owned square matrix.
using JacobianFunction = std::function<Result<Matrix>(ConstVectorView)>;

/// @brief Owns the last verified solution of a square nonlinear system.
struct VectorRoot final {
  Vector value;           ///< Last accepted vector solution.
  Scalar residual_norm{}; ///< Norm of the function residual at the solution.
  Index function_evaluations{}; ///< Number of function callback evaluations.
  Index jacobian_evaluations{}; ///< Number of Jacobian callback evaluations.
  Index iterations{};           ///< Number of solver iterations completed.
  Termination termination{
      Termination::converged}; ///< Completion or stop reason.
};

/// @brief Finds a scalar root by sign-preserving bisection.
[[nodiscard]] Result<RootResult> rootBisection(const UnaryFunction& function,
                                               Interval bracket,
                                               const IterPolicy& policy);

/// @brief Finds a scalar root with safeguarded Brent-Dekker interpolation.
[[nodiscard]] Result<RootResult> rootBrent(const UnaryFunction& function,
                                           Interval bracket,
                                           const IterPolicy& policy);

/// @brief Finds a scalar root by Newton iteration, optionally safeguarded by a
///        sign-changing bracket.
[[nodiscard]] Result<RootResult> rootNewton(const UnaryFunction& function,
                                            const UnaryFunction& derivative,
                                            Scalar initial,
                                            const IterPolicy& policy);

/// @brief Solves a square nonlinear system by damped Newton iteration.
[[nodiscard]] Result<VectorRoot>
findVectorRoot(const VectorFunction& function, const JacobianFunction& jacobian,
               Vector initial, const IterPolicy& policy, Limits limits);

} // namespace vss::math

#endif // VSS_MATH_ROOTS_HPP
