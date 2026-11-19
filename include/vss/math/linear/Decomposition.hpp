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
