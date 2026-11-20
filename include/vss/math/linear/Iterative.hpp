#ifndef VSS_MATH_LINEAR_ITERATIVE_HPP
#define VSS_MATH_LINEAR_ITERATIVE_HPP

#include <cstdint>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Structured.hpp>

namespace vss::math {

/// @brief Describes a normal algorithm termination state.
enum class Termination : std::uint8_t {
  converged, ///< The operation met its stated convergence criterion.
  cap,       ///< The operation reached its configured work or iteration cap.
  ill_conditioned,  ///< The requested numerical condition was not satisfied.
  infeasible,       ///< The constrained problem has no feasible result.
  not_identifiable, ///< The requested result is not identified by the input.
  callback_failure  ///< A callback could not provide a required evaluation.
};

/// @brief Owned verified result from a matrix-free iterative linear solve.
struct IterativeSolve final {
  /// @brief Last true-residual-verified solution.
  Vector value;
  /// @brief Euclidean norm of the independently recomputed residual.
  Scalar residual_norm{};
  /// @brief Number of completed Krylov steps.
  Index iterations{};
  /// @brief Whether tolerance was met or the iteration cap was reached.
  Termination termination{Termination::converged};
};

} // namespace vss::math

namespace vss::math::linear {

/// @brief Solves an SPD matrix-free system using preconditioned conjugate
/// gradients.
/// @details The supplied map and optional SPD inverse-preconditioner must be
///          fixed and deterministic for the call. Only forward actions are
///          used. Every returned candidate is checked by a fresh action and
///          the true Euclidean residual. A cap returns the last verified
///          iterate; structural promises are caller supplied and are not
///          globally certified by sampling. Rank, condition, and bracket
///          fields in `SolvePolicy` are inapplicable. Callback errors are
///          propagated, callback exceptions become `CallbackFailure`, and
///          resource exhaustion returns `ResourceLimit`.
[[nodiscard]] Result<IterativeSolve> cg(const LinearMap& a, ConstVectorView b,
                                        ConstVectorView x0,
                                        const LinearMap* preconditioner,
                                        SolvePolicy policy, Limits limits);

/// @brief Solves a symmetric matrix-free system using preconditioned MINRES.
/// @details The supplied symmetric map and optional SPD inverse-preconditioner
///          must be fixed and deterministic for the call. Only forward actions
///          are used; no square root of the preconditioner is formed. Every
///          returned candidate is checked by a fresh action and the true
///          Euclidean residual.
[[nodiscard]] Result<IterativeSolve>
minres(const LinearMap& a, ConstVectorView b, ConstVectorView x0,
       const LinearMap* preconditioner, SolvePolicy policy, Limits limits);

/// @brief Solves a general matrix-free system with restarted
/// right-preconditioned GMRES.
/// @details The supplied map and optional nonsingular inverse-preconditioner
///          must be fixed and deterministic for the call. Orthogonalization
///          uses two-pass modified Gram--Schmidt. Every returned candidate is
///          checked by a fresh action and the true Euclidean residual.
[[nodiscard]] Result<IterativeSolve>
gmres(const LinearMap& a, ConstVectorView b, ConstVectorView x0,
      const LinearMap* preconditioner, SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_ITERATIVE_HPP
