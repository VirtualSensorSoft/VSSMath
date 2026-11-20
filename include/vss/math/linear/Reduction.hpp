#ifndef VSS_MATH_LINEAR_REDUCTION_HPP
#define VSS_MATH_LINEAR_REDUCTION_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Truncated singular-value factors and their Frobenius residual.
/// @details The represented matrix is left * diag(singular) * right_transpose.
///          Repeated singular values can produce different, equally optimal
///          retained subspaces; compare the represented result rather than a
///          particular basis when no spectral gap identifies that basis.
struct LowRank final {
  /// @brief Retained left singular vectors, one mode per column.
  Matrix left;
  /// @brief Retained nonnegative singular values in descending order.
  Vector singular;
  /// @brief Retained right singular vectors, one mode per row.
  Matrix right_transpose;
  /// @brief Frobenius norm of the input minus the represented factors.
  Scalar residual_frobenius;
};

/// @brief Centered principal-component factors and projected observations.
/// @details Each input column is centered by its empirical mean. Explained
///          variances use the sample denominator N-1. Whitening scales each
///          retained score to unit sample variance.
struct PCA final {
  /// @brief Empirical column means removed from the observation rows.
  Vector center;
  /// @brief Retained principal directions in input coordinates, by column.
  Matrix axes;
  /// @brief Projection of each centered observation onto the retained axes.
  Matrix scores;
  /// @brief Retained sample variances, in descending singular-value order.
  Vector explained_variance;
  /// @brief Whether scores are scaled to unit sample variance.
  bool whitened;
};

} // namespace vss::math

namespace vss::math::linear {

/// @brief Computes an uncentered rank-r truncated singular-value approximation.
/// @details The factors represent the leading terms of the input matrix's
///          thin SVD; no m×n reconstruction matrix is allocated. Rank zero
///          returns empty factors and the Frobenius norm of the full input.
/// @param input Finite m×n matrix, borrowed and unchanged.
/// @param rank Number of leading singular modes, in [0,min(m,n)].
/// @param policy SVD rank, conditioning and convergence policy.
/// @param limits Peak element and byte budgets, including full thin-factor
///        workspace and returned factors.
/// @return Retained factors and Frobenius residual, or a failure without
///         partial output.
[[nodiscard]] Result<LowRank> lowRankApproximation(ConstMatrixView input,
                                                   Index rank,
                                                   SolvePolicy policy,
                                                   Limits limits);

/// @brief Computes centered PCA for observations stored one per row.
/// @details Rank zero is a valid mean-only result when whitening is disabled.
///          Whitening rejects zero or policy-discarded singular directions;
///          no jitter or variance floor is added.
/// @param samples Complete finite population with at least two rows and one
///        variable, with a conforming flattened shape and no missing cells.
/// @param rank Requested number of leading principal components; must not
///        exceed min(N-1,d), the centered sample rank bound.
/// @param whiten If true, divide each retained score by its positive sample
///        standard deviation. A requested zero-variance component cannot be
///        whitened and returns RankDeficient.
/// @param policy SVD rank, conditioning and convergence policy.
/// @param limits Peak element and byte budgets for centering, factor and
///        output storage.
/// @return Column centers, component axes, scores and sample explained
///        variances, or a failure without partial output.
[[nodiscard]] Result<PCA> pca(const SampleView& samples, Index rank,
                              bool whiten, SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_REDUCTION_HPP
