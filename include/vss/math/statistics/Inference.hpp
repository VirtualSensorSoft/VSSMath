#ifndef VSS_MATH_STATISTICS_INFERENCE_HPP
#define VSS_MATH_STATISTICS_INFERENCE_HPP

#include <vss/math/Result.hpp>
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

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_INFERENCE_HPP
