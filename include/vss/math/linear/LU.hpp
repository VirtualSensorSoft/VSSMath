#ifndef VSS_MATH_LINEAR_LU_HPP
#define VSS_MATH_LINEAR_LU_HPP

#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear {

class LUFactorization;

/// @brief Computes a partial-row-pivoted LU factorization.
/// @param matrix Non-empty square matrix to factorize.
/// @param options Rank and condition thresholds used by the factorization.
/// @return A completed factorization, including a diagnostic singular result,
///         or a validation/numerical failure.
/// @note The input is copied and never modified. Complexity is O(n^3) time and
///       O(n^2) owned factor storage for an n by n matrix.
[[nodiscard]] Result<LUFactorization>
factorizeLU(ConstMatrixView matrix, FactorizationOptions options = {});

/// @brief Owns a deterministic partial-row-pivoted LU factorization.
/// @details The public factors satisfy `P * A = L * U`, where
///          `(P * x)[i] == x[permutation()[i]]`. Singular factorizations are
///          successful diagnostic values; their solve operations return
///          `ErrorCode::Singular`. Factor storage is independent of the input
///          matrix and all solve operations are sequential and deterministic.
class LUFactorization final {
public:
  LUFactorization() = delete;
  /// @brief Performs a deep copy of the factorization and diagnostics.
  LUFactorization(const LUFactorization&) = default;
  /// @brief Replaces this factorization with a deep copy.
  LUFactorization& operator=(const LUFactorization&) = default;
  /// @brief Transfers factor storage and resets the source to an empty state.
  LUFactorization(LUFactorization&& other) noexcept;
  /// @brief Replaces this factorization by transferring factor storage.
  LUFactorization& operator=(LUFactorization&& other) noexcept;
  ~LUFactorization() = default;

  /// @brief Returns the factorized matrix extent.
  [[nodiscard]] Extent size() const noexcept;

  /// @brief Returns the effective rank-classification threshold.
  [[nodiscard]] Scalar rankThreshold() const noexcept;

  /// @brief Reports whether any pivot was not strictly above the threshold.
  [[nodiscard]] bool isSingular() const noexcept;

  /// @brief Returns the first zero-based unusable pivot, if any.
  [[nodiscard]] std::optional<Index> firstUnusablePivot() const noexcept;

  /// @brief Returns the row permutation defining `P * x`.
  [[nodiscard]] std::span<const Index> permutation() const noexcept;

  /// @brief Returns the owning unit-lower factor.
  [[nodiscard]] const Matrix& lowerFactor() const noexcept;

  /// @brief Returns the owning upper factor.
  [[nodiscard]] const Matrix& upperFactor() const noexcept;

  /// @brief Returns the one-norm condition diagnostic.
  [[nodiscard]] const ConditionReport& conditionReport() const noexcept;

  /// @brief Returns the determinant sign, or zero for a singular factorization.
  [[nodiscard]] int determinantSign() const noexcept;

  /// @brief Returns the natural logarithm of the absolute determinant.
  /// @details A singular factorization intentionally reports negative
  ///          infinity.
  [[nodiscard]] Scalar logAbsDeterminant() const noexcept;

  /// @brief Solves one right-hand side using the stored factors.
  /// @param rightHandSide Vector with `size()` entries.
  /// @param options Policy for an ill-conditioned but nonsingular factor.
  /// @return The solution and its condition report.
  [[nodiscard]] Result<SolveOutcome<Vector>>
  solve(ConstVectorView rightHandSide, SolveOptions options = {}) const;

  /// @brief Solves multiple right-hand sides using the stored factors.
  /// @param rightHandSides Matrix whose row count equals `size()`.
  /// @param options Policy for an ill-conditioned but nonsingular factor.
  /// @return The solution matrix and its condition report.
  [[nodiscard]] Result<SolveOutcome<Matrix>>
  solve(ConstMatrixView rightHandSides, SolveOptions options = {}) const;

private:
  friend Result<LUFactorization> factorizeLU(ConstMatrixView matrix,
                                             FactorizationOptions options);

  LUFactorization(Extent size, Scalar rankThreshold, bool singular,
                  std::optional<Index> firstUnusablePivot,
                  std::vector<Index> permutation, Matrix lowerFactor,
                  Matrix upperFactor, ConditionReport conditionReport,
                  int determinantSign, Scalar logAbsDeterminant) noexcept;

  void resetMovedFrom() noexcept;

  Extent size_;
  Scalar rankThreshold_;
  bool singular_;
  std::optional<Index> firstUnusablePivot_;
  std::vector<Index> permutation_;
  Matrix lowerFactor_;
  Matrix upperFactor_;
  ConditionReport conditionReport_;
  int determinantSign_;
  Scalar logAbsDeterminant_;
};

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_LU_HPP
