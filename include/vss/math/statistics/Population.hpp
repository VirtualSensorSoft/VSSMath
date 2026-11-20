#ifndef VSS_MATH_STATISTICS_POPULATION_HPP
#define VSS_MATH_STATISTICS_POPULATION_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::statistics {

/// @brief Selects the denominator and interpretation of population spread.
enum class VarianceMode : std::uint8_t { empirical, iid_sample, known_center };

/// @brief Selects the common support rule for population estimates.
enum class MissingMode : std::uint8_t {
  reject,
  complete_row,
  pairwise,
  gaussian_em
};

/// @brief Describes whether population covariance is centered on a sample
///        estimate or on a supplied point.
struct CenterSpec final {
  /// @brief Identifies an estimated or explicitly supplied center.
  CenterRule rule{CenterRule::estimated};
  /// @brief Borrowed center coordinates, read only during the estimator call;
  ///        required for `known` and absent for `estimated`.
  std::optional<ConstVectorView> center;
};

/// @brief Declares the population estimator and common missing-row policy.
struct PopulationPolicy final {
  /// @brief Selects the empirical, IID sample, or known-center denominator.
  VarianceMode normalization;
  /// @brief Selects rejection or complete-row retention for invalid cells.
  MissingMode missingness;
  /// @brief Names the statistical quantity the estimate describes.
  Estimand estimand;
};

/// @brief Owns an estimate with the diagnostics that qualify its
/// interpretation.
/// @tparam T Owning value type returned by the estimator.
template <class T> struct Estimate final {
  /// @brief Owning estimated value.
  T value;
  /// @brief Support, estimator, and interpretation metadata.
  EstimatorDiagnostics diagnostics;
  /// @brief Optional covariance of the estimate itself, when computed.
  std::optional<Matrix> error_covariance;
};

/// @brief Estimates covariance among flattened coordinates of one population.
/// @param sample Rows of observations, optionally with a cell validity mask.
/// @param center Estimated center or supplied coordinate center.
/// @param policy Denominator, common-support, and estimand declaration.
/// @param limits Maximum output and workspace elements and bytes.
/// @return A d-by-d covariance matrix and support diagnostics.
/// @details Uses a two-pass, scale-normalized compensated calculation. The
///          empirical and known-center modes divide by the retained row count;
///          the IID sample mode divides by count minus one and requires an
///          estimated center. `complete_row` drops any row with an invalid
///          cell. Pairwise and Gaussian-EM support are not implemented here.
/// @note Complexity: O(Nd + d^2) time and O(d^2 + d) owned output/workspace.
[[nodiscard]] Result<Estimate<Matrix>>
covarianceMatrix(const SampleView& sample, const CenterSpec& center,
                 PopulationPolicy policy, Limits limits);

/// @brief Estimates cross-covariance of paired rows in operand order.
/// @param left First population; each row must pair with the same row in right.
/// @param right Second population with an equal row count.
/// @param leftCenter Center declaration for left coordinates.
/// @param rightCenter Center declaration for right coordinates.
/// @param policy Denominator, common-support, and estimand declaration.
/// @param limits Maximum output and workspace elements and bytes.
/// @return A d-left by d-right cross-covariance matrix and support diagnostics.
/// @details Row order declares the pairing; the two operands may have different
///          coordinate dimensions and shapes. Requires common row support
///          across both operands. Swapping the operands transposes the returned
///          matrix. Missing rows follow the policy described by
///          @ref covarianceMatrix.
/// @note Complexity: O(N(d-left + d-right) + d-left*d-right) time and
///       O(d-left*d-right + d-left + d-right) owned output/workspace.
[[nodiscard]] Result<Estimate<Matrix>>
crossCovariance(const SampleView& left, const SampleView& right,
                const CenterSpec& leftCenter, const CenterSpec& rightCenter,
                PopulationPolicy policy, Limits limits);

/// @brief Converts a covariance matrix to its dimensionless correlation form.
/// @details Validates the complete input as finite, symmetric and positive
///          semidefinite before normalization. A zero-variance coordinate is
///          accepted only when its entire covariance row and column are zero;
///          it has no defined correlation and therefore causes
///          `InvalidDomain`. Positive-variance coordinates are normalized in
///          scaled arithmetic. Singular positive-semidefinite correlation
///          matrices are valid. Rank and conditioning diagnostics describe the
///          returned correlation matrix using `policy`.
/// @param covariance Finite square covariance matrix, borrowed and unchanged.
/// @param policy Rank cutoff, condition floor, and eigensolver work cap.
/// @param limits Peak scratch and output-storage limits.
/// @return Owned correlation matrix and spectral diagnostics.
[[nodiscard]] Result<covariance::CovarianceResult>
correlationMatrix(ConstMatrixView covariance, SolvePolicy policy,
                  Limits limits);

/// @brief Computes correlations among selected coordinates conditional on
/// others.
/// @details Standardizes the complete joint covariance first, then forms the
///          Schur complement of the conditioning block using its retained
///          eigenspace. `reject` fails when conditioning directions are
///          discarded; supported-subspace and minimum-norm policies use the
///          Moore-Penrose inverse on retained directions. The result preserves
///          selected-coordinate order and is normalized to correlation form.
///          Correlation rank decisions are consequently invariant to positive
///          changes of input coordinate units.
/// @param covariance Finite square joint covariance, borrowed and unchanged.
/// @param selected Unique indices whose conditional correlation is returned.
/// @param conditioned Unique disjoint indices conditioned upon.
/// @param policy Rank cutoff, condition floor, rank handling, and sweep cap.
/// @param limits Peak scratch and output-storage limits.
/// @return Owned conditional correlation and spectral diagnostics.
[[nodiscard]] Result<covariance::CovarianceResult>
partialCorrelation(ConstMatrixView covariance, std::span<const Index> selected,
                   std::span<const Index> conditioned, SolvePolicy policy,
                   Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_POPULATION_HPP
