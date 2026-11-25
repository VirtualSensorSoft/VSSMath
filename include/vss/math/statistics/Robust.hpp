#ifndef VSS_MATH_STATISTICS_ROBUST_HPP
#define VSS_MATH_STATISTICS_ROBUST_HPP

#include <optional>
#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {

/// @brief Shared tuning and optional fixed inputs for robust estimators.
/// @details Views are borrowed for the duration of an estimator call. The
///          cutoff must be positive and finite. `scale_factor` must be
///          positive and finite. `fixed_scale`, `metric`, and `center` are
///          operation-specific: robustScale validates but does not use them;
///          robustLocation uses fixed_scale when supplied; robustCovariance
///          uses center and metric when supplied. The iteration policy bounds
///          the safeguarded robust-location root search.
struct RobustPolicy final {
  Scalar huber_cutoff;
  Scalar scale_factor;
  std::optional<ConstVectorView> fixed_scale;
  std::optional<ConstMatrixView> metric;
  IterPolicy iteration;
  std::optional<ConstVectorView> center;
};

/// @brief Estimates coordinatewise median absolute deviation scales.
/// @details Returns `scale_factor * median(abs(x - median(x)))` for each
///          coordinate using the library's linear median convention. Constant
///          coordinates validly have zero scale.
[[nodiscard]] Result<Estimate<Vector>>
robustScale(const SampleView& x, const RobustPolicy& policy, Limits limits);

/// @brief Solves the coordinatewise Huber location score equations.
/// @details Uses a safeguarded bracket on each coordinate. Flat minimizer
///          intervals are represented by their midpoint. A zero estimated
///          scale is accepted only for a constant coordinate.
[[nodiscard]] Result<Estimate<Vector>>
robustLocation(const SampleView& x, const RobustPolicy& policy, Limits limits);

/// @brief Computes radial Huber-winsorized scatter about an explicit or
///        estimated center.
/// @details The returned matrix is `N^-1 sum(z_i z_i^T)` with residuals
///          clipped in the supplied SPD metric or the estimated diagonal MAD
///          metric. It is not recentered or Gaussian-consistency corrected.
[[nodiscard]] Result<Estimate<Matrix>>
robustCovariance(const SampleView& x, const RobustPolicy& policy,
                 Limits limits);

/// @brief Combines two positive-semidefinite covariance matrices with an
///        explicit shrinkage fraction.
/// @details Returns `(1-lambda) * covariance + lambda * target`; endpoint
///          results preserve the selected input exactly after validating both.
[[nodiscard]] Result<covariance::CovarianceResult>
shrinkCovariance(ConstMatrixView covariance, ConstMatrixView target,
                 Scalar lambda, SolvePolicy policy, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_ROBUST_HPP
