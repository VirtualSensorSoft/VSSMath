#ifndef VSS_MATH_STATISTICS_INFERENCE_HPP
#define VSS_MATH_STATISTICS_INFERENCE_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Explicitly acknowledges the normal, independent, identically
///        distributed sampling premise used by a statistical procedure.
struct NormalIID final {
  /// @brief True only when the caller accepts the stated model premise.
  bool acknowledged{};
};

} // namespace vss::math

namespace vss::math::statistics {

/// @brief Declares which asymptotic Ljung–Box reference law is applicable.
enum class LjungBoxReference : std::uint8_t { white_noise, fitted_arma };

/// @brief Statistic and only those inferential fields justified by its model.
struct TestResult final {
  /// @brief Non-negative test statistic.
  Scalar statistic{};
  /// @brief Upper-tail probability when an applicable reference was declared.
  std::optional<Scalar> p_value;
  /// @brief Reference degrees of freedom when defined by the test.
  std::optional<Scalar> effective_df;
  /// @brief Monte Carlo error; absent for deterministic asymptotic tests.
  std::optional<Scalar> sampling_error;
  /// @brief Description of the calibration actually used.
  std::string calibration_method;
};

/// @brief Computes the equal-tail confidence interval for a normal-IID
///        population variance.
/// @details For sample variance s² and ν=n−1, returns
///          [νs²/χ²(1−α/2;ν), νs²/χ²(α/2;ν)], where α=1−confidence.
///          The caller must explicitly acknowledge `NormalIID`. A zero sample
///          variance returns [0,0]. This interval is for population variance;
///          it is not the standard uncertainty sqrt(s²).
/// @param sampleVariance Finite non-negative unbiased sample variance.
/// @param n Number of independent observations; at least two are required.
/// @param confidence Requested central coverage in (0,1).
/// @param premise Explicit acknowledgement of the normal-IID model.
/// @return Equal-tail variance interval or a numerical/domain failure.
[[nodiscard]] Result<Interval> varianceConfidenceInterval(Scalar sampleVariance,
                                                          Index n,
                                                          Scalar confidence,
                                                          NormalIID premise);

/// @brief Tests a multivariate Gaussian-IID mean against a fixed target.
/// @details Uses the unbiased sample covariance, requires n>d and a full-rank,
///          conditioned covariance, and never forms its inverse. The reported
///          statistic is n (x̄−μ₀)ᵀ S⁻¹ (x̄−μ₀); its p-value uses the exact
///          F(d,n−d) transformation under the acknowledged Gaussian-IID null.
///          Degrees of freedom and sampling error are not reported because
///          this test has two distinct degrees of freedom and no resampling.
[[nodiscard]] Result<TestResult> hotellingT2(const SampleView& sample,
                                             ConstVectorView center,
                                             NormalIID premise,
                                             SolvePolicy policy, Limits limits);

/// @brief Computes the Ljung–Box statistic without assuming a reference law.
/// @details Requires complete finite regularly indexed residuals with positive
///          variance and 1≤lag<n, fittedDegrees<lag. It returns Q and
///          effective degrees of freedom but no p-value.
[[nodiscard]] Result<TestResult> ljungBox(ConstVectorView residuals, Index lag,
                                          Index fittedDegrees, Limits limits);

/// @brief Computes Ljung–Box Q and its explicitly selected asymptotic tail.
/// @details `white_noise` requires fittedDegrees=0. `fitted_arma` requires a
///          positive adjustment and caller-supplied applicability of the
///          fitted-residual chi-square approximation. The tag records an
///          assumption; it does not verify stationarity or model fit.
[[nodiscard]] Result<TestResult> ljungBox(ConstVectorView residuals, Index lag,
                                          Index fittedDegrees,
                                          LjungBoxReference reference,
                                          Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_INFERENCE_HPP
