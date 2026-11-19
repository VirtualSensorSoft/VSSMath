#ifndef VSS_MATH_LINEAR_RANK_AWARE_HPP
#define VSS_MATH_LINEAR_RANK_AWARE_HPP

#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {

/// @brief Owned estimate and numerical-identifiability diagnostics for a
///        linear system.
/// @details `parameters` has one column per right-hand side. When supplied,
///          each column of `identified_basis` is a right singular vector in
///          the retained numerical support. `residual_norms` are Euclidean
///          norms evaluated against the original design and observations.
///          Covariance is optional because a solve alone does not specify a
///          stochastic model.
struct LinearEstimate final {
  /// @brief Minimum-norm parameter columns in the original coordinate order.
  Matrix parameters;
  /// @brief Parameter covariance when a separate stochastic model provides it.
  std::optional<Matrix> covariance;
  /// @brief Orthonormal basis for the retained, numerically identified space.
  std::optional<Matrix> identified_basis;
  /// @brief Original-system Euclidean residual norm for each right-hand side.
  Vector residual_norms;
  /// @brief Number of singular directions strictly above `rank_cutoff`.
  Index rank{};
  /// @brief Inclusive singular-value cutoff in the original input units.
  Scalar rank_cutoff{};
};

/// @brief Computes rank-aware minimum-norm least-squares estimates.
/// @details For `A` of shape m×n and `B` of shape m×p, returns
///          `V_r diag(1/sigma_r) U_r^T B`, where `A = U Sigma V^T` is its
///          thin SVD. The inclusive cutoff is
///          `tau = max(rank_abs, rank_rel * sigma_max)` in original input
///          units; only singular values strictly greater than `tau` are
///          retained. A singular value equal to the cutoff is discarded.
///          Residual norms are computed from the original `A` and `B`, not a
///          projected right-hand side. `reject` fails when rank is below
///          `min(m,n)`; `supported_subspace` and `minimum_norm` both use the
///          retained subspace for this least-squares objective. A wide
///          full-row-rank system is valid and returns its minimum-norm
///          solution. A positive reciprocal-condition floor compares
///          `sigma_rank / sigma_max` and rejects lower retained support.
///          Rank-zero systems return zero parameters and the norms of the
///          original observation columns. Empty axes and zero right-hand-side
///          columns preserve their conforming result shapes. Covariance is
///          disengaged because no stochastic model is supplied. No failure
///          publishes a partial estimate. Factorization refinement is bounded
///          by `policy.iteration.max_iterations`; a zero cap permits directly
///          certifiable diagonal or permuted-diagonal spectra only. Dense work
///          is O(m n k + p m n), with `k=min(m,n)`; storage is O(m n + n p)
///          plus the bounded factor and identified basis.
/// @param design Finite m×n design matrix; borrowed for the duration of call.
/// @param observations Finite m×p right-hand sides in caller column order.
/// @param policy Rank cutoff, condition floor, and SVD iteration policy.
/// @param limits Peak output/workspace limits, including retained factors.
/// @return Owned parameters, rank, cutoff, identified basis, and residuals, or
///         a failure without partial estimate data. Failures include
///         `DimensionMismatch`, `InvalidTolerance`, `InvalidDomain`,
///         `NonFiniteInput`, `RankDeficient`, `IllConditioned`,
///         `NoConvergence`, `ResourceLimit`, and `NonFiniteResult` as
///         applicable.
/// @throws std::bad_alloc if permitted storage cannot be allocated.
[[nodiscard]] Result<LinearEstimate>
leastSquaresMinNorm(ConstMatrixView design, ConstMatrixView observations,
                    SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_RANK_AWARE_HPP
