#ifndef VSS_MATH_SIGNAL_SMOOTHING_HPP
#define VSS_MATH_SIGNAL_SMOOTHING_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::signal {

/// @brief Smooths a finite sample series or estimates one of its derivatives.
/// @param samples Borrowed input samples; the input is never modified.
/// @param window Positive odd number of samples in each local fit.
/// @param degree Polynomial degree, strictly less than `window`.
/// @param derivative Derivative order, no greater than `degree`.
/// @param step Positive finite spacing between adjacent samples.
/// @param limits Per-call output and workspace limits.
/// @return An owning vector with one smoothed value or derivative per input
///         sample, or a diagnostic failure.
/// @details Each target uses a full window. At the left and right boundaries,
///          the window is shifted inward rather than padded or shortened. A
///          pivoted QR least-squares fit is evaluated at the target's sample
///          coordinate. Derivatives are converted from the scaled local
///          coordinate to the physical sample coordinate using
///          `derivative! / (scale * step)^derivative`.
///
///          Empty input or a valid window larger than the sample count returns
///          `InsufficientSamples`. Non-finite samples or `step` return
///          `NonFiniteInput`; finite invalid parameters return `InvalidDomain`.
///          A numerically rank-deficient local design returns `RankDeficient`,
///          an unrepresentable output returns `NonFiniteResult`, and a request
///          exceeding `limits` returns `ResourceLimit`. The output and
///          simultaneously required numerical workspaces are preflighted
///          before allocation.
/// @note Work is O(N W (p+1) + W^2 (p+1)^2); storage is O(N + W(p+1) +
///       (p+1)^2), where N is the sample count, W is `window`, and p is
///       `degree`.
[[nodiscard]] Result<Vector> savitzkyGolay(ConstVectorView samples,
                                           Index window, Index degree,
                                           Index derivative, Scalar step,
                                           Limits limits);

} // namespace vss::math::signal

#endif // VSS_MATH_SIGNAL_SMOOTHING_HPP
