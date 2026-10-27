#ifndef VSS_MATH_COVARIANCE_COVARIANCE_HPP
#define VSS_MATH_COVARIANCE_COVARIANCE_HPP

#include <cstdint>
#include <optional>
#include <utility>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {

/// @brief Classification produced by covariance spectral analysis.
enum class CovarianceClassification : std::uint8_t {
  /// @brief Every eigenvalue is strictly above the rank threshold.
  PositiveDefinite,
  /// @brief No eigenvalue is below the negative rank threshold.
  PositiveSemidefinite,
  /// @brief At least one eigenvalue is below the negative rank threshold.
  NotPositiveSemidefinite
};

/// @brief Selects whether weak eigendirections are retained in diagnostics.
enum class WeakDirectionOutput : std::uint8_t { Omit, Include };

/// @brief Eigenvalue validation options shared by covariance operations.
/// @details The nested policy is validated before any input extent or value is
///          inspected. Invalid sweep limits return `InvalidDomain`; invalid
///          tolerance components return `InvalidTolerance`. The aggregate is
///          passive and allocation-free; it is safe to reuse concurrently when
///          callers do not mutate it.
struct CovarianceValidationOptions final {
  /// @brief Numerical eigensolver and spectral-validation policy.
  linear::SymmetricEigenOptions eigen{};
};

/// @brief Controls covariance diagnostics and weak-direction output.
/// @details `conditionLimit` must be finite and strictly greater than one;
///          otherwise the operation returns `InvalidTolerance`. Invalid enum
///          values return `InvalidDomain`. No input or output is allocated by
///          this passive aggregate.
struct CovarianceAnalysisOptions final {
  /// @brief Validation policy applied before spectral analysis.
  CovarianceValidationOptions validation{};
  /// @brief Maximum finite spectral condition estimate considered acceptable.
  Scalar conditionLimit{1.0e10};
  /// @brief Selects whether weak eigenvectors are copied into diagnostics.
  WeakDirectionOutput weakDirections{WeakDirectionOutput::Omit};
};

/// @brief Controls correlation diagonal validation and eigenvalue validation.
/// @details The unit-diagonal tolerance must contain finite, non-negative
///          components. Invalid components return `InvalidTolerance`; nested
///          eigen-option failures are reported first.
struct CorrelationOptions final {
  /// @brief Validation policy applied to the correlation matrix.
  CovarianceValidationOptions validation{};
  /// @brief Tolerance for checking that every correlation diagonal is one.
  Tolerance unitDiagonalTolerance{.absolute = 0.0, .relative = 1.0e-12};
};

/// @brief Owns deterministic spectral diagnostics for one covariance input.
/// @details The object is created only by @ref analyzeCovariance and owns a
///          deep copy of any requested weak-direction matrix. A successful
///          object has passed symmetry validation. The classification is
///          positive-definite, positive-semidefinite, or indefinite; an
///          indefinite input is a successful diagnostic result, not a failure.
///          Eigenvalues and weak directions follow ascending deterministic
///          order. A finite positive-definite condition estimate is reported,
///          except that one explicitly positive infinity represents a checked
///          quotient overflow; no other field is infinite. The view returned by
///          weakDirectionEigenvectors() is invalidated by destruction, move, or
///          assignment of its owner. Copying is deep. A moved-from object
///          remains valid but its values are unspecified and must not be
///          inspected before reassignment or destruction. The value owns no
///          shared state, cache, thread, or external resource.
/// @par Requirements
/// VSS-MATH-COV-001, VSS-MATH-COV-002, VSS-MATH-COV-005,
/// VSS-MATH-COV-006, and VSS-MATH-COV-008.
class CovarianceDiagnostics final {
public:
  /// @brief Default construction is unavailable; use analyzeCovariance().
  CovarianceDiagnostics() = delete;
  /// @brief Copy-constructs diagnostics with a deep copy of owned directions.
  CovarianceDiagnostics(const CovarianceDiagnostics&) = default;
  /// @brief Copy-assigns diagnostics with a deep copy of owned directions.
  CovarianceDiagnostics& operator=(const CovarianceDiagnostics&) = default;
  /// @brief Move-constructs diagnostics and resets the source to a valid state.
  CovarianceDiagnostics(CovarianceDiagnostics&& other) noexcept;
  /// @brief Move-assigns diagnostics and resets the source to a valid state.
  CovarianceDiagnostics& operator=(CovarianceDiagnostics&& other) noexcept;
  ~CovarianceDiagnostics() = default;

  /// @brief Returns the spectral definiteness classification.
  [[nodiscard]] CovarianceClassification classification() const noexcept {
    return classification_;
  }
  /// @brief Returns whether the input symmetry check succeeded.
  [[nodiscard]] bool passedSymmetryValidation() const noexcept {
    return passedSymmetryValidation_;
  }
  /// @brief Returns the numerical rank under the eigen rank threshold.
  [[nodiscard]] Extent rank() const noexcept { return rank_; }
  /// @brief Returns the smallest ordered eigenvalue.
  [[nodiscard]] Scalar minimumEigenvalue() const noexcept {
    return minimumEigenvalue_;
  }
  /// @brief Returns the largest ordered eigenvalue.
  [[nodiscard]] Scalar maximumEigenvalue() const noexcept {
    return maximumEigenvalue_;
  }
  /// @brief Returns the rank threshold used by spectral classification.
  [[nodiscard]] Scalar rankThreshold() const noexcept { return rankThreshold_; }
  /// @brief Returns the finite positive-definite spectral condition estimate.
  /// @details The optional is empty for positive-semidefinite and indefinite
  ///          classifications. A positive infinity is allowed only when the
  ///          finite quotient would overflow the scalar representation.
  [[nodiscard]] const std::optional<Scalar>&
  spectralConditionNumber() const noexcept {
    return spectralConditionNumber_;
  }
  /// @brief Returns requested weak eigenvectors in original coordinates.
  /// @details Omitted output is `0 x 0`; requested output with no directions is
  ///          `n x 0`; otherwise columns are ascending weak eigendirections.
  [[nodiscard]] ConstMatrixView weakDirectionEigenvectors() const noexcept {
    return weakDirectionEigenvectors_.view();
  }

private:
  CovarianceDiagnostics(CovarianceClassification classification,
                        bool passedSymmetryValidation, Extent rank,
                        Scalar minimumEigenvalue, Scalar maximumEigenvalue,
                        Scalar rankThreshold,
                        std::optional<Scalar> spectralConditionNumber,
                        Matrix weakDirectionEigenvectors) noexcept;
  void resetMovedFrom() noexcept;

  friend Result<CovarianceDiagnostics>
  analyzeCovariance(ConstMatrixView covariance,
                    CovarianceAnalysisOptions options);

  CovarianceClassification classification_;
  bool passedSymmetryValidation_;
  Extent rank_;
  Scalar minimumEigenvalue_;
  Scalar maximumEigenvalue_;
  Scalar rankThreshold_;
  std::optional<Scalar> spectralConditionNumber_;
  Matrix weakDirectionEigenvectors_;
};

/// @brief Owns a correlation matrix and its standard deviations.
/// @details The correlation matrix is dimensionless and the standard deviations
///          retain the input covariance units. Copying is deep. Moving leaves
///          the source in a documented empty state (`0 x 0` matrix and empty
///          vector), so source accessors remain valid after a move. Inputs are
///          never modified. The value owns no shared state, cache, thread, or
///          external resource and is safe to read concurrently after
///          publication.
/// @par Requirements
/// VSS-MATH-COV-001, VSS-MATH-COV-002, and VSS-MATH-COV-007.
class CorrelationConversion final {
public:
  /// @brief Default construction is unavailable; use toCorrelation().
  CorrelationConversion() = delete;
  /// @brief Copy-constructs the conversion with deep-owned matrix/vector data.
  CorrelationConversion(const CorrelationConversion&) = default;
  /// @brief Copy-assigns the conversion with deep-owned matrix/vector data.
  CorrelationConversion& operator=(const CorrelationConversion&) = default;
  /// @brief Move-constructs the conversion and empties the source.
  CorrelationConversion(CorrelationConversion&& other) noexcept;
  /// @brief Move-assigns the conversion and empties the source.
  CorrelationConversion& operator=(CorrelationConversion&& other) noexcept;
  ~CorrelationConversion() = default;

  /// @brief Returns the owned dimensionless correlation matrix.
  [[nodiscard]] const Matrix& correlation() const noexcept {
    return correlation_;
  }
  /// @brief Returns the owned positive standard-deviation vector.
  [[nodiscard]] const Vector& standardDeviations() const noexcept {
    return standardDeviations_;
  }

private:
  CorrelationConversion(Matrix correlation, Vector standardDeviations) noexcept;
  void resetMovedFrom() noexcept;

  friend Result<CorrelationConversion>
  toCorrelation(ConstMatrixView covariance,
                CovarianceValidationOptions options);

  Matrix correlation_;
  Vector standardDeviations_;
};

/// @brief Analyzes symmetry, rank, definiteness, and conditioning.
/// @details Validation is option-first, followed by non-empty square shape,
///          checked storage arithmetic, finite values, symmetry, and the
///          existing dense symmetric eigensolver. Weak directions are copied
///          only when requested and are ordered by ascending eigenvalue.
///          Returns `InvalidExtent`, `DimensionMismatch`, `SizeOverflow`,
///          `NonFiniteInput`, `NotSymmetric`, or an eigensolver failure before
///          allocating a result. Indefinite input succeeds with a
///          `NotPositiveSemidefinite` classification. The dense eigensolver
///          costs O(n^3) time and O(n^2) storage; requested directions add
///          O(n*k) storage. The function is stateless and thread-safe when
///          input views and options are not concurrently mutated.
/// @par Requirements
/// VSS-MATH-COV-001, VSS-MATH-COV-002, VSS-MATH-COV-005, and VSS-MATH-COV-006.
[[nodiscard]] Result<CovarianceDiagnostics>
analyzeCovariance(ConstMatrixView covariance,
                  CovarianceAnalysisOptions options = {});

/// @brief Computes a squared Mahalanobis distance through Cholesky whitening.
/// @details Validates the solve-policy enumerator, factorization state,
///          residual extent, and residual finiteness before delegating one
///          whitening solve. It then uses scale-safe sum-of-squares and checks
///          the final square before constructing the scalar solve outcome.
///          Possible failures include `InvalidDomain`, `InvalidExtent`,
///          `DimensionMismatch`, `NonFiniteInput`, `IllConditioned`, and
///          `NonFiniteResult`; the condition report is preserved unchanged.
///          Cost is O(n^2) time and O(n) temporary storage. The function is
///          stateless and thread-safe when its inputs are not concurrently
///          mutated.
/// @par Requirements
/// VSS-MATH-COV-003 and VSS-MATH-COV-008.
[[nodiscard]] Result<linear::SolveOutcome<Scalar>>
mahalanobisSquared(const linear::CholeskyFactorization& factorization,
                   ConstVectorView residual, linear::SolveOptions options = {});

/// @brief Propagates covariance as `J S_C J^T` with deterministic reductions.
/// @details Requires a non-empty compatible Jacobian and symmetric positive
///          semidefinite covariance. It validates Jacobian values before
///          covariance values, uses the canonical lower triangle, computes
///          `T = J*S_C` followed by the lower triangle of `T*J^T` with
///          compensated reductions, and mirrors each completed value exactly.
///          Possible failures include `InvalidDomain`, `InvalidExtent`,
///          `DimensionMismatch`, `SizeOverflow`, `NonFiniteInput`,
///          `NotSymmetric`, `NotPositiveSemidefinite`, and
///          `NonFiniteResult`. Cost is O(p*n*n + p*p*n) time, O(p*n + p*p)
///          output/temporary storage. No partial result is published. The
///          function is stateless and thread-safe for non-mutated inputs.
/// @par Requirements
/// VSS-MATH-COV-004 and VSS-MATH-COV-008.
[[nodiscard]] Result<Matrix>
propagateCovariance(ConstMatrixView jacobian, ConstMatrixView covariance,
                    CovarianceValidationOptions options = {});

/// @brief Converts a positive-semidefinite covariance to correlation form.
/// @details Requires finite, symmetric, positive-diagonal, positive-
///          semidefinite input. It computes `sigma_i=sqrt(S_C(i,i))`, then
///          `(S_C(i,j)/sigma_i)/sigma_j` on the lower triangle and mirrors
///          exact values; diagonal entries are set to exact one. Possible
///          failures include `InvalidExtent`, `DimensionMismatch`,
///          `SizeOverflow`, `NonFiniteInput`, `NotSymmetric`,
///          `InvalidDomain`, `NotPositiveSemidefinite`, and
///          `NonFiniteResult`. Cost is O(n^3) time and O(n^2) output storage.
///          Inputs are never modified; the function is stateless and
///          thread-safe for non-mutated inputs.
/// @par Requirements
/// VSS-MATH-COV-001, VSS-MATH-COV-002, and VSS-MATH-COV-007.
[[nodiscard]] Result<CorrelationConversion>
toCorrelation(ConstMatrixView covariance,
              CovarianceValidationOptions options = {});

/// @brief Converts a positive-semidefinite correlation to covariance form.
/// @details Requires finite symmetric input with unit diagonal under the
///          supplied tolerance and strictly positive finite deviations. It
///          computes `(S_R(i,j)*sigma_i)*sigma_j` on the lower triangle and
///          mirrors each completed value exactly. Possible failures include
///          `InvalidExtent`, `DimensionMismatch`, `SizeOverflow`,
///          `NonFiniteInput`, `NotSymmetric`, `InvalidDomain`,
///          `NotPositiveSemidefinite`, and `NonFiniteResult`. Cost is O(n^3)
///          time and O(n^2) output storage. Inputs are never modified; the
///          function is stateless and thread-safe for non-mutated inputs.
/// @par Requirements
/// VSS-MATH-COV-002 and VSS-MATH-COV-007.
[[nodiscard]] Result<Matrix> fromCorrelation(ConstMatrixView correlation,
                                             ConstVectorView standardDeviations,
                                             CorrelationOptions options = {});

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_COVARIANCE_HPP
