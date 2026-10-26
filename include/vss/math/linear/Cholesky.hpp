#ifndef VSS_MATH_LINEAR_CHOLESKY_HPP
#define VSS_MATH_LINEAR_CHOLESKY_HPP

#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear {

/// @brief Owns a deterministic Cholesky factorization of a positive-definite
///        symmetric matrix.
/// @details The represented canonical matrix satisfies `S = L L^T`, where
///          `L` is lower triangular with strictly positive diagonal entries.
///          Factorization copies its input and never modifies borrowed source
///          storage. The factor and all diagnostics are independent owning
///          state; no jitter or positive-definiteness repair is performed.
///          Factorization takes O(n^3) time and O(n^2) storage for an n by n
///          matrix. Distinct factorization objects may be used concurrently;
///          concurrent access to one object is read-only safe.
class CholeskyFactorization final {
public:
  CholeskyFactorization() = delete;

  /// @brief Performs a deep copy of the factorization.
  CholeskyFactorization(const CholeskyFactorization&) = default;

  /// @brief Replaces this factorization with a deep copy.
  CholeskyFactorization& operator=(const CholeskyFactorization&) = default;

  /// @brief Transfers factor storage and resets the source to an empty state.
  CholeskyFactorization(CholeskyFactorization&& other) noexcept;

  /// @brief Replaces this factorization by transferring factor storage.
  CholeskyFactorization& operator=(CholeskyFactorization&& other) noexcept;

  ~CholeskyFactorization() = default;

  /// @brief Returns the represented matrix extent.
  [[nodiscard]] Extent size() const noexcept;

  /// @brief Returns the full rank of the successful factorization.
  [[nodiscard]] Extent rank() const noexcept;

  /// @brief Returns the strict pre-square-root pivot threshold.
  [[nodiscard]] Scalar rankThreshold() const noexcept;

  /// @brief Returns the tolerance used for mirrored-entry validation.
  [[nodiscard]] Tolerance symmetryTolerance() const noexcept;

  /// @brief Returns the owning lower triangular factor `L`.
  [[nodiscard]] const Matrix& lowerFactor() const noexcept;

  /// @brief Returns the deterministic one-norm condition diagnostic for `S`.
  [[nodiscard]] const ConditionReport& conditionReport() const noexcept;

  /// @brief Returns `log(det(S))` for the positive-definite matrix.
  [[nodiscard]] Scalar logDeterminant() const noexcept;

  /// @brief Solves `S x = b` using the stored factor.
  /// @param rightHandSide Vector with exactly `size()` entries.
  /// @param options Policy for an ill-conditioned but positive-definite factor.
  /// @return Solution and its condition diagnostic, or a failure.
  /// @note Complexity: O(n^2) time and O(n) temporary storage.
  [[nodiscard]] Result<SolveOutcome<Vector>>
  solve(ConstVectorView rightHandSide, SolveOptions options = {}) const;

  /// @brief Solves `S X = B` for all columns of `B`.
  /// @param rightHandSides Matrix with `size()` rows and at least one column.
  /// @param options Policy for an ill-conditioned but positive-definite factor.
  /// @return Solution and its condition diagnostic, or a failure.
  /// @note Complexity: O(k n^2) time for k right-hand sides.
  [[nodiscard]] Result<SolveOutcome<Matrix>>
  solve(ConstMatrixView rightHandSides, SolveOptions options = {}) const;

  /// @brief Whitens a residual by solving `L z = residual`.
  /// @param residual Vector with exactly `size()` entries.
  /// @param options Policy for an ill-conditioned factor.
  /// @return The whitened vector and its condition diagnostic, or a failure.
  /// @note This intentionally does not solve `S z = residual`.
  [[nodiscard]] Result<SolveOutcome<Vector>>
  whiten(ConstVectorView residual, SolveOptions options = {}) const;

private:
  friend Result<CholeskyFactorization>
  factorizeCholesky(ConstMatrixView matrix,
                    SymmetricFactorizationOptions options);

  CholeskyFactorization(Extent size, Extent rank, Scalar rankThreshold,
                        Tolerance symmetryTolerance, Matrix lowerFactor,
                        ConditionReport conditionReport,
                        Scalar logDeterminant) noexcept;

  void resetMovedFrom() noexcept;

  Extent size_;
  Extent rank_;
  Scalar rankThreshold_;
  Tolerance symmetryTolerance_;
  Matrix lowerFactor_;
  ConditionReport conditionReport_;
  Scalar logDeterminant_;
};

/// @brief Factorizes a non-empty symmetric positive-definite matrix.
/// @param matrix Borrowed square input; mirrored entries are checked first.
/// @param options Rank, condition, and symmetry thresholds.
/// @return A complete factorization or a validation/numerical failure.
/// @note The lower triangle is authoritative after symmetry validation.
///       Complexity: O(n^3) time and O(n^2) owned storage.
[[nodiscard]] Result<CholeskyFactorization>
factorizeCholesky(ConstMatrixView matrix,
                  SymmetricFactorizationOptions options = {});

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_CHOLESKY_HPP
