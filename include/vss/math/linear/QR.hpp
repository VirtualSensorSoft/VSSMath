#ifndef VSS_MATH_LINEAR_QR_HPP
#define VSS_MATH_LINEAR_QR_HPP

#include <optional>
#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear {

/// @brief Computes a deterministic column-pivoted Householder QR factorization.
/// @param matrix Non-empty matrix with at least as many rows as columns.
/// @param options Rank and condition thresholds used by the factorization.
/// @return A completed factorization, including rank and condition diagnostics,
///         or a validation/numerical failure.
/// @details The input is copied and never modified. The returned factors obey
///          `A P = Q R`, where `P` is described by @ref
///          QRFactorization::permutation, `Q` is the economy orthogonal factor,
///          and `R` is upper triangular with non-negative diagonal entries.
///          Rectangular factorizations do not estimate a condition number.
/// @note Complexity is O(m n^2) time and O(m n) owned factor storage for an
///       m by n matrix with m >= n. No external numerical dependency is used.
[[nodiscard]] Result<class QRFactorization>
factorizeQR(ConstMatrixView matrix, FactorizationOptions options = {});

/// @brief Owns a column-pivoted Householder QR factorization.
/// @details The object owns all factor storage and never aliases the input.
///          `permutation()[j]` is the original column stored at factor column
///          `j`; therefore column `j` of `A P` is original column
///          `permutation()[j]`. The public factors satisfy `A P = Q R`.
///          Householder vectors remain private and a dense Q is materialized
///          only by @ref economyQ. The public diagonal of R is normalized to
///          non-negative values, including positive zero.
///
///          A rectangular full-column-rank factorization has no condition
///          report. A square factorization owns a report and its solve policy
///          rejects estimates meeting or exceeding the configured limit unless
///          `IllConditionedSolvePolicy::AllowWithDiagnostic` is selected.
///          All operations are sequential and deterministic; callers must
///          externally synchronize concurrent mutation or moves.
///
///          Move construction and move assignment reset the source to an empty
///          moved-from state: all extents, rank threshold, norm, permutation,
///          factors, reflector storage, and condition report are empty or zero.
///          In that state both predicates are false, @ref economyQ returns an
///          empty 0 by 0 matrix, and both solve overloads return
///          `ErrorCode::InvalidExtent`.
class QRFactorization final {
public:
  /// @brief No default factorization exists.
  QRFactorization() = delete;
  /// @brief Performs a deep copy of the factorization and diagnostics.
  QRFactorization(const QRFactorization&) = default;
  /// @brief Replaces this factorization with a deep copy.
  QRFactorization& operator=(const QRFactorization&) = default;
  /// @brief Transfers factor storage and resets the source to empty.
  QRFactorization(QRFactorization&& other) noexcept;
  /// @brief Replaces this factorization by transferring storage.
  QRFactorization& operator=(QRFactorization&& other) noexcept;
  ~QRFactorization() = default;

  /// @brief Returns the input row extent m.
  [[nodiscard]] Extent rows() const noexcept;
  /// @brief Returns the input column extent n.
  [[nodiscard]] Extent columns() const noexcept;
  /// @brief Returns the numerical column rank.
  [[nodiscard]] Extent rank() const noexcept;
  /// @brief Returns the strict rank-classification threshold.
  [[nodiscard]] Scalar rankThreshold() const noexcept;
  /// @brief Reports whether all columns are numerically usable.
  [[nodiscard]] bool hasFullColumnRank() const noexcept;
  /// @brief Reports whether the factorized matrix is a non-empty square.
  [[nodiscard]] bool isSquare() const noexcept;
  /// @brief Returns the original-column index stored at each factor column.
  /// @return A non-owning span valid while this factorization is unchanged.
  [[nodiscard]] std::span<const Index> permutation() const noexcept;
  /// @brief Materializes the m by n economy orthogonal factor Q.
  /// @return A new owning matrix; no dense Q is stored by the factorization.
  /// @note Complexity is O(m n^2) time and O(m n) returned storage.
  [[nodiscard]] Matrix economyQ() const;
  /// @brief Returns the owning n by n normalized upper factor R.
  [[nodiscard]] const Matrix& upperFactor() const noexcept;
  /// @brief Returns the original matrix one-norm used for diagnostics.
  [[nodiscard]] Scalar matrixOneNorm() const noexcept;
  /// @brief Returns the optional square-system condition diagnostic.
  [[nodiscard]] const std::optional<ConditionReport>&
  conditionReport() const noexcept;

  /// @brief Solves one full-column-rank least-squares system.
  /// @param rightHandSide Vector with rows() entries.
  /// @param options Policy for an ill-conditioned square factorization.
  /// @return The solution in original-column order and an optional diagnostic.
  /// @note Complexity is O(m n + n^2) time and O(m+n) temporary/output space.
  [[nodiscard]] Result<SolveOutcome<Vector>>
  leastSquares(ConstVectorView rightHandSide, SolveOptions options = {}) const;

  /// @brief Solves multiple full-column-rank least-squares systems.
  /// @param rightHandSides Matrix with rows() rows and at least one column.
  /// @param options Policy for an ill-conditioned square factorization.
  /// @return Solutions in original-column order and an optional diagnostic.
  /// @note For k right-hand sides, complexity is O(k(m n+n^2)) time and
  ///       O(nk+m+n) output/temporary space.
  [[nodiscard]] Result<SolveOutcome<Matrix>>
  leastSquares(ConstMatrixView rightHandSides, SolveOptions options = {}) const;

private:
  friend Result<QRFactorization> factorizeQR(ConstMatrixView matrix,
                                             FactorizationOptions options);

  QRFactorization(Extent rows, Extent columns, Extent rank,
                  Scalar rankThreshold, std::vector<Index> permutation,
                  Matrix reflectors, std::vector<Scalar> reflectorCoefficients,
                  std::vector<Scalar> normalizationSigns, Matrix upperFactor,
                  Scalar matrixOneNorm,
                  std::optional<ConditionReport> conditionReport) noexcept;

  void resetMovedFrom() noexcept;

  Extent rows_;
  Extent columns_;
  Extent rank_;
  Scalar rankThreshold_;
  std::vector<Index> permutation_;
  Matrix reflectors_;
  std::vector<Scalar> reflectorCoefficients_;
  std::vector<Scalar> normalizationSigns_;
  Matrix upperFactor_;
  Scalar matrixOneNorm_;
  std::optional<ConditionReport> conditionReport_;
};

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_QR_HPP
