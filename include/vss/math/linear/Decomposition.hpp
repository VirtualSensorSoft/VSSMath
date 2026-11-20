#ifndef VSS_MATH_LINEAR_DECOMPOSITION_HPP
#define VSS_MATH_LINEAR_DECOMPOSITION_HPP

#include <cstdint>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Owned thin factors of a real singular value decomposition.
struct SVD final {
  /// @brief Orthonormal columns of U.
  Matrix left;
  /// @brief Nonnegative singular values in descending order.
  Vector singular;
  /// @brief Orthonormal rows of V transpose.
  Matrix right_transpose;
  /// @brief Number of singular values strictly above the cutoff.
  Index rank;
  /// @brief Numerical rank cutoff in input units.
  Scalar cutoff;
};

namespace linear {

/// @brief Selects whether QR keeps input columns or pivots by residual norm.
enum class QRColumnOrder : std::uint8_t {
  original, ///< Keep the identity column permutation.
  pivoted   ///< Pivot by descending active trailing-column norm.
};

/// @brief Owning full orthogonal QR factors and their rank diagnostics.
/// @details For input `A`, the represented relation is
///          `A[:, permutation] = orthogonal * upper`. `orthogonal` is square
///          with order equal to the input row count. `upper` has the input
///          shape and is exactly zero below its diagonal. The immutable
///          `permutation[j]` identifies the original input column represented
///          by factor column `j`. `rank` counts singular values strictly
///          greater than `cutoff`; both are certified from the input singular
///          spectrum, never inferred from the diagonal of `upper`.
struct FullQR final {
  /// @brief Owning m by m factor whose columns are orthonormal.
  Matrix orthogonal;
  /// @brief Owning m by n upper-trapezoidal factor.
  Matrix upper;
  /// @brief Original-column index for each factor column.
  Array<Index> permutation;
  /// @brief Singular-spectrum rank at the requested strict cutoff.
  Index rank;
  /// @brief Rank cutoff in the original input units.
  Scalar cutoff;
};

/// @brief Computes full Householder QR factors for any finite matrix shape.
/// @param input Borrowed input matrix; it is not modified.
/// @param order Original input order or deterministic residual-norm pivoting.
/// @param policy Singular-spectrum rank cutoff and certification policy.
///        Solve rejection and reciprocal-condition rules do not reject this
///        decomposition. Bracket, restart and callback fields do not apply.
/// @param limits Peak owned and workspace element/byte ceilings.
/// @details The input is not modified. Original order returns the identity
///          permutation. Pivoted order chooses the largest remaining scaled
///          trailing column norm at every step and breaks exact ties by the
///          smallest original column index. Rank loss does not shorten either
///          output factor. Empty dimensions retain their full stated extents:
///          Q is the row-sized identity, and the permutation is identity.
///          Rank rejection and reciprocal-condition solve policies do not
///          reject this decomposition; the iteration limit applies to SVD rank
///          certification. There are no callbacks or condition estimate.
/// @return Full factors, permutation and SVD-certified numerical rank, or
///         `InvalidDomain`, `InvalidTolerance`, `NonFiniteInput`,
///         `NonFiniteResult`, `NoConvergence` or `ResourceLimit` without
///         partial output.
/// @throws std::bad_alloc if storage within the configured limits cannot be
///         allocated.
/// @note For `k=min(m,n)`, complexity is O(m^2 k + m n k), plus writing full
///       outputs when k is zero. Storage is O(m^2 + m n + n).
[[nodiscard]] Result<FullQR> factorFullQR(ConstMatrixView input,
                                          QRColumnOrder order,
                                          SolvePolicy policy, Limits limits);

/// @brief Complete generalized eigensystem for a symmetric definite pencil.
/// @details Eigenvalues are ascending. Each column of `vectors` is normalized
///          in the metric of the original second matrix, so
///          `vectors^T * B * vectors = I`. Residual norms are Euclidean norms
///          of `A*v - lambda*B*v` in the original input units. All members own
///          their storage.
struct GeneralizedEigen final {
  /// @brief Ascending real generalized eigenvalues.
  Vector values;
  /// @brief Corresponding B-orthonormal eigenvectors in columns.
  Matrix vectors;
  /// @brief Euclidean residual norm for each eigenpair.
  Vector residual_norms;
};

/// @brief Solves the real symmetric-definite generalized eigenproblem.
/// @param a Finite square symmetric matrix; indefinite values are supported.
/// @param b Equally sized finite square symmetric positive-definite metric.
/// @param policy Rank, metric conditioning, and spectral iteration policy.
/// @param limits Maximum peak owned/workspace elements and bytes.
/// @return All eigenpairs in ascending order, or a failure without partial
///         eigenpairs. Empty equally-sized matrices return an empty result.
/// @throws std::bad_alloc if storage within the configured limits cannot be
///         allocated.
/// @note Complexity is O(n^3) time and O(n^2) storage for an n by n pencil.
[[nodiscard]] Result<GeneralizedEigen>
factorGeneralizedSymmetric(ConstMatrixView a, ConstMatrixView b,
                           SolvePolicy policy, Limits limits);

/// @brief Owned symmetric-indefinite factorization with 1x1 and 2x2 pivots.
/// @details The represented input satisfies
///          `A[permutation, permutation] = unit_lower * block_d *
///          transpose(unit_lower)`. `unit_lower` has an exact unit diagonal;
///          `block_d` is symmetric and block diagonal according to
///          `pivot_sizes`. The entries of `pivot_sizes` are one or two and
///          sum to the factor order. Numerical rank is classified from the
///          singular spectrum of the original symmetrized input, without
///          zeroing pivots in the reconstruction. All members own their
///          storage; index sequences are immutable through their public view.
struct LDLT final {
  /// @brief Unit lower-triangular factor in pivoted coordinates.
  Matrix unit_lower;
  /// @brief Symmetric block-diagonal factor in pivoted coordinates.
  Matrix block_d;
  /// @brief Mapping from pivoted coordinates to original row/column indices.
  Array<Index> permutation;
  /// @brief Ordered block widths, each exactly one or two.
  Array<std::uint8_t> pivot_sizes;
  /// @brief Count of singular values strictly above the configured cutoff.
  Index rank;
};

/// @brief Computes a scaled Bunch--Kaufman factorization of a symmetric input.
/// @param input Borrowed finite square matrix; it is not modified.
/// @param policy Rank cutoff and spectral-certification iteration ceiling.
///        Solve rank policy, condition floor, callback cap, and restart do not
///        alter factor construction.
/// @param limits Maximum simultaneous output and workspace elements and bytes.
/// @return Complete factors and numerical rank, or a status with no partial
///         factors. Indefinite and exactly singular inputs are supported.
/// @throws std::bad_alloc if storage within the configured ceilings cannot be
///         allocated.
/// @note Complexity is O(n^3) time and O(n^2) storage, including rank
///       certification, for an n by n input.
[[nodiscard]] Result<LDLT> factorLDLT(ConstMatrixView input, SolvePolicy policy,
                                      Limits limits);

/// @brief Factors a finite real matrix into U, singular values and V transpose.
/// @details The singular values are nonnegative and descending. Both factors
///          have min(rows, columns) singular directions, including zero modes.
///          Rank counts singular values strictly above max(rank_abs,
///          rank_rel * largest singular value) in input units. Solve rejection
///          rules do not reject a decomposition. An exhausted iteration cap or
///          unresolved cutoff returns NoConvergence. No partial factors escape.
/// @param input Borrowed matrix, needed only during the call.
/// @param policy Rank and iteration policy. Bracket, restart and evaluation
///               fields do not apply.
/// @param limits Ceilings for peak elements and bytes, including workspace.
/// @return An owning factorization or a failure status.
/// @throws std::bad_alloc If permitted storage cannot be allocated.
[[nodiscard]] Result<SVD> factorSVD(ConstMatrixView input, SolvePolicy policy,
                                    Limits limits);

} // namespace linear
} // namespace vss::math
#endif
