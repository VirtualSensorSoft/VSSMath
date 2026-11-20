#ifndef VSS_MATH_LINEAR_REDUCTION_HPP
#define VSS_MATH_LINEAR_REDUCTION_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {

/// @brief Borrowed row-major matrix of observations, one sample per row.
using SampleView = ConstMatrixView;

/// @brief Truncated singular-value representation and its approximation.
struct LowRank final {
  /// @brief Left singular vectors for the retained modes, one per column.
  Matrix left;
  /// @brief Retained nonnegative singular values in descending order.
  Vector singular_values;
  /// @brief Retained right singular vectors, one per row.
  Matrix right_transpose;
  /// @brief Reconstructed matrix from the retained singular modes.
  Matrix approximation;
  /// @brief Frobenius norm of input minus approximation.
  Scalar residual_norm{};
  /// @brief SVD numerical-rank cutoff in the original input units.
  Scalar rank_cutoff{};
};

/// @brief Centered principal-component decomposition of row observations.
struct PCA final {
  /// @brief Column means subtracted from the input observations.
  Vector center;
  /// @brief Retained principal directions in input coordinates, by column.
  Matrix components;
  /// @brief Projected rows; whitened to unit sample variance when requested.
  Matrix scores;
  /// @brief Sample variance explained by each retained component.
  Vector explained_variance;
  /// @brief Whether the returned scores were whitened.
  bool whitened{};
  /// @brief SVD numerical-rank cutoff in centered input units.
  Scalar rank_cutoff{};
};

/// @brief Computes the rank-r truncated singular-value approximation.
/// @param input Finite m×n matrix, borrowed and unchanged.
/// @param rank Requested number of leading singular modes; must not exceed
///        min(m,n). Rank zero yields the zero approximation.
/// @param policy SVD rank, conditioning and convergence policy.
/// @param limits Peak element and byte budgets for factor and output storage.
/// @return Retained factors, approximation and Frobenius residual, or a
///         failure without partial output.
[[nodiscard]] Result<LowRank> lowRankApproximation(ConstMatrixView input,
                                                   Index rank,
                                                   SolvePolicy policy,
                                                   Limits limits);

/// @brief Computes centered PCA for observations stored one per row.
/// @param samples Finite N×d sample matrix with at least two rows and one
///        variable. Rows are observations; columns are variables.
/// @param rank Requested number of leading principal components; must not
///        exceed min(N,d).
/// @param whiten If true, divide each retained score by its positive sample
///        standard deviation. A requested zero-variance component cannot be
///        whitened and returns RankDeficient.
/// @param policy SVD rank, conditioning and convergence policy.
/// @param limits Peak element and byte budgets for centering, factor and
///        output storage.
/// @return Column centers, component directions, scores and sample explained
///        variances, or a failure without partial output.
[[nodiscard]] Result<PCA> pca(const SampleView& samples, Index rank,
                              bool whiten, SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_REDUCTION_HPP
