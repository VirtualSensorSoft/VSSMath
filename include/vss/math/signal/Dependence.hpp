#ifndef VSS_MATH_SIGNAL_DEPENDENCE_HPP
#define VSS_MATH_SIGNAL_DEPENDENCE_HPP

#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::signal {

/// @brief Selects the divisor used for a finite-lag covariance estimate.
/// @details `full_count` divides every lag sum by the complete input length N;
///          `overlap_count` divides by the number N-k of matched pairs. These
///          are normalization conventions and do not assert estimator bias.
enum class LagNormalization : std::uint8_t {
  /// @brief Divide every lag sum by the complete input length N.
  full_count,
  /// @brief Divide each lag sum by its matched-pair count N-k.
  overlap_count
};

/// @brief Owns covariance or correlation values aligned to requested lags.
/// @details The three arrays have identical extents. Lag and count storage is
///          private; public access is immutable and valid while this object
///          remains alive. Values are exposed through a borrowing view tied to
///          this object's lifetime.
class LagSeries final {
public:
  /// @brief Returns requested lags in their original order, including repeats.
  [[nodiscard]] std::span<const std::int64_t> lags() const noexcept {
    return lags_;
  }
  /// @brief Returns result values aligned with @ref lags.
  [[nodiscard]] ConstVectorView values() const noexcept {
    return values_.view();
  }
  /// @brief Returns the matched observation count for each requested lag.
  [[nodiscard]] std::span<const Index> counts() const noexcept {
    return counts_;
  }

private:
  [[nodiscard]] static Result<LagSeries>
  compute(ConstVectorView samples, std::span<const Index> lags,
          LagNormalization normalization, Limits limits, bool correlation);
  [[nodiscard]] static Result<LagSeries>
  computeCross(ConstVectorView x, ConstVectorView y,
               std::span<const std::int64_t> lags,
               LagNormalization normalization, Limits limits, bool correlation);

  LagSeries(std::vector<std::int64_t> lags, Vector values,
            std::vector<Index> counts)
      : lags_(std::move(lags)), values_(std::move(values)),
        counts_(std::move(counts)) {}

  std::vector<std::int64_t> lags_;
  Vector values_;
  std::vector<Index> counts_;

  friend Result<LagSeries> autocovariance(ConstVectorView samples,
                                          std::span<const Index> lags,
                                          LagNormalization normalization,
                                          Limits limits);
  friend Result<LagSeries> autocorrelation(ConstVectorView samples,
                                           std::span<const Index> lags,
                                           LagNormalization normalization,
                                           Limits limits);
  friend Result<LagSeries> crossCovariance(ConstVectorView x, ConstVectorView y,
                                           std::span<const std::int64_t> lags,
                                           LagNormalization normalization,
                                           Limits limits);
  friend Result<LagSeries> crossCorrelation(ConstVectorView x,
                                            ConstVectorView y,
                                            std::span<const std::int64_t> lags,
                                            LagNormalization normalization,
                                            Limits limits);
};

/// @brief Computes finite-lag autocovariance around the complete-sample mean.
/// @details For lag k, sums `(x[t]-mean)*(x[t-k]-mean)` over t=k…N−1 and
///          divides by N or N−k according to `normalization`. Requested lag
///          order and repetitions are preserved. Work is O(N+N·L), storage
///          O(L), and the result is preflighted against both output limits.
///          Empty input gives InsufficientSamples; nonfinite samples give
///          NonFiniteInput; an unknown normalization gives InvalidDomain; an
///          out-of-range lag gives IndexOutOfRange; and an unrepresentable
///          result gives NonFiniteResult.
/// @param samples Complete finite sequence; the mean is estimated from all N
///        samples and a borrowed view must remain valid for the call.
/// @param lags Unsigned sample-index lags, each in `[0,N)`; order and repeats
///        are retained.
/// @param normalization Full-count or overlap-count divisor policy.
/// @param limits Simultaneous output element and byte ceilings.
/// @return Owned lag/value/count data, or a diagnostic failure without a
///         partially published result.
[[nodiscard]] Result<LagSeries> autocovariance(ConstVectorView samples,
                                               std::span<const Index> lags,
                                               LagNormalization normalization,
                                               Limits limits);

/// @brief Computes autocovariance normalized by the full-count lag-zero value.
/// @details The numerator follows `normalization`, while the denominator is
///          always the full-sample variance. Overlap-normalized values may
///          therefore exceed unit magnitude. Zero variance is InvalidDomain.
///          Work is O(N+N·L), storage O(L). Input, lag, policy, limit and
///          representability failures use the same codes documented for
///          @ref autocovariance.
/// @param samples Complete finite sequence; the mean and denominator variance
///        use all N samples.
/// @param lags Unsigned sample-index lags, each in `[0,N)`; order and repeats
///        are retained.
/// @param normalization Selects the numerator divisor only; the denominator
///        variance always uses N.
/// @param limits Simultaneous output element and byte ceilings.
[[nodiscard]] Result<LagSeries> autocorrelation(ConstVectorView samples,
                                                std::span<const Index> lags,
                                                LagNormalization normalization,
                                                Limits limits);

/// @brief Computes finite-lag cross-covariance for two aligned sequences.
/// @details Both complete sequences are centered by their own full-sample
///          means. Positive lag `k` pairs `x[t+k]` with `y[t]`; negative lag
///          `-k` pairs `x[t]` with `y[t+k]`. The matched products are divided
///          by N or by N-|k| according to `normalization`. Sequences must have
///          equal length and equal cadence. Requested signed-lag order and
///          repetitions are preserved. Work is O(N+N·L), storage O(L), and
///          result storage is checked against both configured limits.
/// @param x First complete finite sequence.
/// @param y Second complete finite sequence with the same extent and cadence.
/// @param lags Signed sample-index lags, each satisfying `|k| < N`.
/// @param normalization Full-count or overlap-count divisor policy.
/// @param limits Simultaneous output element and byte ceilings.
/// @return Owned lag/value/count data, or a diagnostic failure without a
///         partially published result.
/// @retval ErrorCode::DimensionMismatch The sequence extents differ.
/// @retval ErrorCode::InsufficientSamples Either sequence is empty.
/// @retval ErrorCode::IndexOutOfRange A signed lag has no matched pair.
/// @retval ErrorCode::InvalidDomain The normalization policy is unsupported.
/// @retval ErrorCode::NonFiniteInput Either sequence contains a non-finite
///         sample.
/// @retval ErrorCode::ResourceLimit Result storage exceeds configured limits.
/// @retval ErrorCode::NonFiniteResult A covariance result is not representable.
[[nodiscard]] Result<LagSeries>
crossCovariance(ConstVectorView x, ConstVectorView y,
                std::span<const std::int64_t> lags,
                LagNormalization normalization, Limits limits);

/// @brief Computes cross-covariance normalized by both full-sample variances.
/// @details Uses the same signed-lag orientation and numerator normalization
///          as @ref crossCovariance, divided by
///          `sqrt(gamma_x(0)*gamma_y(0))`, where both lag-zero variances use
///          divisor N. With overlap-count normalization, the result can exceed
///          unit magnitude. Zero variance in either complete sequence is
///          InvalidDomain. Work is O(N+N·L), storage O(L).
/// @param x First complete finite sequence.
/// @param y Second complete finite sequence with the same extent and cadence.
/// @param lags Signed sample-index lags, each satisfying `|k| < N`.
/// @param normalization Selects the numerator divisor only; the denominator
///        always uses the full-sample variances.
/// @param limits Simultaneous output element and byte ceilings.
/// @return Owned lag/value/count data, or a diagnostic failure without a
///         partially published result.
/// @retval ErrorCode::DimensionMismatch The sequence extents differ.
/// @retval ErrorCode::InsufficientSamples Either sequence is empty.
/// @retval ErrorCode::IndexOutOfRange A signed lag has no matched sample pair.
/// @retval ErrorCode::InvalidDomain The normalization policy is unsupported,
///         or either full-sample variance is zero.
/// @retval ErrorCode::NonFiniteInput Either sequence contains a non-finite
///         sample.
/// @retval ErrorCode::ResourceLimit Result storage exceeds configured limits.
/// @retval ErrorCode::NonFiniteResult A correlation result is not
/// representable.
[[nodiscard]] Result<LagSeries>
crossCorrelation(ConstVectorView x, ConstVectorView y,
                 std::span<const std::int64_t> lags,
                 LagNormalization normalization, Limits limits);

/// @brief Computes partial autocorrelations from a supplied autocovariance.
/// @details Applies the Levinson–Durbin recursion to `autocovariance[0…order]`
///          and returns reflection coefficients in orders one through
///          `order`. Extra entries are ignored. Order zero requires a finite,
///          positive lag-zero covariance and returns an empty vector. Every
///          positive-order step requires a certified positive innovation
///          variance; an exact or policy-rank-deficient innovation returns
///          `RankDeficient`, while a materially negative innovation returns
///          `NotPositiveSemidefinite`. The absolute and relative rank
///          tolerances are applied to innovation variances with lag-zero
///          covariance as the relative scale. The reciprocal-condition floor
///          is applied to each innovation ratio `E/gamma[0]`. Positive-order
///          recursions are capped by `policy.iteration.max_iterations`.
///          Work is O(order²), with O(order) scratch and output storage, all
///          preflighted against `limits`. No partial result is returned.
/// @param autocovariance Finite autocovariance sequence beginning at lag zero;
///        the first `order + 1` entries are used.
/// @param order Requested highest partial-autocorrelation order.
/// @param policy Numerical rank, conditioning and iteration policy.
/// @param limits Simultaneous element and byte caps for scratch and output.
/// @return Owned reflection coefficients, or a diagnostic failure.
/// @retval ErrorCode::InsufficientSamples The input does not contain lag zero
///         and all requested lags.
/// @retval ErrorCode::NonFiniteInput A used autocovariance entry is nonfinite.
/// @retval ErrorCode::InvalidDomain Lag-zero covariance is not positive.
/// @retval ErrorCode::InvalidTolerance A numerical tolerance is invalid.
/// @retval ErrorCode::RankDeficient The Toeplitz extension has a zero or
///         policy-rank-deficient innovation.
/// @retval ErrorCode::NotPositiveSemidefinite A used lag violates the
///         covariance bound `|gamma[k]| <= gamma[0]`, or a materially negative
///         innovation identifies an indefinite Toeplitz extension.
/// @retval ErrorCode::IllConditioned An innovation ratio is below the
///         configured reciprocal-condition floor.
/// @retval ErrorCode::NoConvergence The iteration cap is below `order`.
/// @retval ErrorCode::ResourceLimit Scratch or result storage exceeds limits.
[[nodiscard]] Result<Vector>
partialAutocorrelation(ConstVectorView autocovariance, Index order,
                       SolvePolicy policy, Limits limits);

/// @brief Computes the variance of an equal-weight mean from autocovariance.
/// @details Uses the first `count` lags of a declared stationary
///          autocovariance model:
///          `(count*gamma[0] + 2*sum((count-k)*gamma[k])) / count²`.
///          Requires nonnegative `gamma[0]`, zero lags when it is zero, and
///          `abs(gamma[k]) <= gamma[0]`. Full Toeplitz positive
///          semidefiniteness is a caller premise; a negative contracted
///          variance is rejected. A scaled compensated sum provides the
///          ordinary evaluation, with fixed-storage exact dyadic accumulation
///          preserving sign and zero under cancellation. Work is O(count), with
///          O(1) auxiliary storage.
/// @param autocovariance Autocovariance values beginning at lag zero.
/// @param count Number of observations in the equal-weight mean; must be
///        positive and no larger than the supplied number of lags.
/// @return Finite variance, or an error when the covariance input or result is
///         unusable.
/// @retval ErrorCode::InsufficientSamples Count is zero or the input has fewer
///         than `count` lags.
/// @retval ErrorCode::NonFiniteInput A used autocovariance value is nonfinite.
/// @retval ErrorCode::InvalidDomain Lag-zero covariance is negative.
/// @retval ErrorCode::NotPositiveSemidefinite A lag violates the covariance
///         bound or the exact contraction is negative.
/// @retval ErrorCode::NonFiniteResult The positive exact variance rounds to no
///         finite nonzero `Scalar` value.
[[nodiscard]] Result<Scalar>
meanVarianceWithAutocorrelation(ConstVectorView autocovariance, Index count);

} // namespace vss::math::signal

#endif // VSS_MATH_SIGNAL_DEPENDENCE_HPP
