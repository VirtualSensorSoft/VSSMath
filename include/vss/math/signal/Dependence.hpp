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

} // namespace vss::math::signal

#endif // VSS_MATH_SIGNAL_DEPENDENCE_HPP
