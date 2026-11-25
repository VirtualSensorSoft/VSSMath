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
  MissingMode mode;
  SolvePolicy policy;
  Index minimum_support;
  std::optional<Vector> initial_mean = std::nullopt;
  std::optional<Matrix> initial_covariance = std::nullopt;
};

/// @brief Owned selected rows and their source provenance.
struct SelectedSample final {
  Matrix data;
  Array<Index> shape;
  Array<Index> source_rows;
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
