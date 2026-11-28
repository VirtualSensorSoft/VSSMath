#ifndef VSS_MATH_STATISTICS_DEPENDENT_INFERENCE_HPP
#define VSS_MATH_STATISTICS_DEPENDENT_INFERENCE_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Bootstrap.hpp>

namespace vss::math::statistics {

/// @brief Estimates the long-run covariance of an ordered score population.
/// @details For complete finite rows s_t in R^d and a fixed lag L with
///          0 <= L < N, forms the Bartlett estimate
///          Gamma_0 + sum_{k=1}^L (1-k/(L+1))(Gamma_k+Gamma_k^T), where
///          Gamma_k = N^-1 sum_{t=k}^{N-1} u_t u_{t-k}^T. `known` uses
///          u_t=s_t; `estimated` subtracts the coordinate-wise sample mean.
///          The result is a long-run score covariance; covariance of a sample
///          mean is this matrix divided by N only when the caller requests it.
///          This operation uses the equivalent Bartlett-window sum-of-squares
///          form, preserving a positive-semidefinite result up to roundoff.
///          It reports finite arithmetic only and does not test stationarity,
///          select a bandwidth, or certify inferential coverage. Asymptotic
///          interpretation requires the caller's stationary-process, moment,
///          dependence-decay, and lag-growth premises.
///          Work is O((N+L)d^2+d^3) and bounded workspace is O(d^2+d).
///          Scaled compensated accumulators avoid overflowing intermediate
///          centered sums and products; the output matrix is mirrored exactly
///          symmetric before its C014 classification.
/// @param scores N by d complete ordered rows; flattened coordinates retain
///               their input order.
/// @param lag Fixed nonnegative Bartlett lag, strictly less than N.
/// @param center Whether scores are known zero-centered or estimated-centered.
/// @param limits Per-call result and workspace limits.
/// @return The owned long-run covariance and default C014 spectral diagnostics.
/// @retval ErrorCode::InsufficientSamples Empty row or coordinate extent.
/// @retval ErrorCode::InvalidDomain Invalid center, mask, or lag.
/// @retval ErrorCode::DimensionMismatch Invalid shape or mask extent.
/// @retval ErrorCode::NonFiniteInput A score is not finite.
/// @retval ErrorCode::NonFiniteResult The computed covariance is not finite.
/// @retval ErrorCode::NoConvergence The computed matrix is materially
///         indefinite under default C014 analysis.
/// @retval ErrorCode::SizeOverflow A required size is not representable.
/// @retval ErrorCode::ResourceLimit A configured output or workspace cap is
///         insufficient.
[[nodiscard]] Result<covariance::CovarianceResult>
neweyWest(const SampleView& scores, Index lag, CenterRule center,
          Limits limits);

/// @brief Computes a statistic's conditional distribution using dependent
///        whole-row block resampling.
/// @details Admits only moving-block and circular-block policies. It delegates
///          the sampling, statistic evaluation, interval, invalid-draw,
///          resource and random-stream semantics to @ref bootstrapEstimate.
///          Moving and circular blocks retain the source row order within
///          each selected block, and the final block is truncated only at the
///          requested replicate size. These methods do not assert exact
///          population coverage. Their inferential interpretation requires
///          the stationary-dependence premise and a caller-selected block
///          length appropriate to the target process.
/// @param x Ordered complete source population.
/// @param f Statistic recomputed for every admitted replicate.
/// @param p Moving- or circular-block policy; IID policy is not admitted.
/// @param rng Mutable deterministic random stream.
/// @param limits Per-call output, workspace and callback limits.
/// @return The conditional bootstrap result or the delegated failure status.
/// @retval ErrorCode::InvalidDomain An IID policy was supplied.
[[nodiscard]] Result<BootstrapResult>
dependentBootstrap(const SampleView& x, const Statistic& f,
                   const BootstrapPolicy& p, probability::Rng& rng,
                   Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_DEPENDENT_INFERENCE_HPP
