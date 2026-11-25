#ifndef VSS_MATH_STATISTICS_MISSING_HPP
#define VSS_MATH_STATISTICS_MISSING_HPP

#include <optional>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {

/// @brief Configuration for missing-data selection and covariance estimates.
/// @details Gaussian EM requires an explicit mean/covariance initialization;
///          the owning values are copied by the estimator and never modified.
struct MissingPolicy final {
  /// @brief Reject, complete-row, pairwise or explicit Gaussian-EM policy.
  MissingMode mode;
  /// @brief Bounded numerical solve and convergence policy used by Gaussian EM.
  SolvePolicy policy;
  /// @brief Minimum observed support required for each estimated quantity.
  Index minimum_support;
  /// @brief Owned Gaussian-EM initial mean; required only in Gaussian-EM mode.
  std::optional<Vector> initial_mean = std::nullopt;
  /// @brief Owned Gaussian-EM initial covariance; required only in Gaussian-EM
  /// mode.
  std::optional<Matrix> initial_covariance = std::nullopt;
};

/// @brief Owned selected rows and their source provenance.
struct SelectedSample final {
  /// @brief Owned selected or complete observation rows.
  Matrix data;
  /// @brief Logical shape of one observation in row-major order.
  Array<Index> shape;
  /// @brief Original zero-based source row for each retained output row.
  Array<Index> source_rows;
  /// @brief Copied validity bits corresponding to selected output cells.
  Array<std::uint8_t> mask;
};

/// @brief Selects, copies or preserves rows according to an explicit mask rule.
[[nodiscard]] Result<SelectedSample>
sampleWithMissingPolicy(const SampleView& x, const MissingPolicy& policy,
                        Limits limits);

/// @brief Estimates empirical covariance under an explicit missing-data rule.
/// @details Pairwise mode returns the descriptive matrix even when it is
///          indefinite and records a PSD warning. Gaussian EM estimates a
///          stationary candidate under the declared observed-data Gaussian
///          model; it does not imply a global optimum.
[[nodiscard]] Result<Estimate<Matrix>>
covarianceWithMissing(const SampleView& x, const MissingPolicy& policy,
                      Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_MISSING_HPP
