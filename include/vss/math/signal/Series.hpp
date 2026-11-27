#ifndef VSS_MATH_SIGNAL_SERIES_HPP
#define VSS_MATH_SIGNAL_SERIES_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math {

using LinearEstimate = linear::LinearEstimate;
using WeightSpec = statistics::WeightSpec;

/// @brief A borrowed, timestamped, row-major multichannel series.
/// @details `times` has one finite, strictly increasing timestamp per row of
///          `rows`. An empty `valid` span means every row is valid; otherwise
///          it contains one binary validity byte per row. `uniformStep`, when
///          present, declares a cadence that operations may validate against
///          the timestamps. All views borrow storage and must not outlive it.
class SeriesView final {
public:
  /// @brief Creates a borrowed series view.
  /// @param timeValues One timestamp per observation row.
  /// @param rowValues Observation rows, with one or more channels.
  /// @param validity Optional binary complete-row mask; empty means all valid.
  /// @param cadence Optional declared uniform sampling interval.
  SeriesView(ConstVectorView timeValues, ConstMatrixView rowValues,
             std::span<const std::uint8_t> validity = {},
             std::optional<Scalar> cadence = std::nullopt) noexcept
      : times_(timeValues), rows_(rowValues), valid_(validity),
        uniformStep_(cadence) {}

  /// @brief Returns the borrowed timestamp axis.
  [[nodiscard]] ConstVectorView times() const noexcept { return times_; }
  /// @brief Returns observations with rows aligned to `times()`.
  [[nodiscard]] ConstMatrixView rows() const noexcept { return rows_; }
  /// @brief Returns the optional borrowed binary complete-row validity mask.
  [[nodiscard]] std::span<const std::uint8_t> valid() const noexcept {
    return valid_;
  }
  /// @brief Returns the optional declared uniform sampling step.
  [[nodiscard]] std::optional<Scalar> uniformStep() const noexcept {
    return uniformStep_;
  }

private:
  ConstVectorView times_;
  ConstMatrixView rows_;
  std::span<const std::uint8_t> valid_;
  std::optional<Scalar> uniformStep_;
};

/// @brief Integrates one complete scalar series using the composite
///        trapezoidal rule on its actual timestamps.
/// @details The returned value at row zero is positive zero. Later entries are
///          compensated prefixes of `(t[i]-t[i-1])*(y[i]+y[i-1])/2`; unequal
///          intervals are preserved. No interpolation or missing-row fill is
///          performed. If `uniformStep` is supplied, cadence is checked using
///          the C054 eight-epsilon scaled-time rule. Time is O(n), with O(1)
///          scratch beyond the n-element output.
/// @param series Single-channel series with complete finite observations.
/// @param limits Output element and byte ceilings.
/// @return Owned integral prefixes, or a validation, resource, or
///         representability failure without partial output.
[[nodiscard]] Result<Vector> cumulativeIntegral(const SeriesView& series,
                                                Limits limits);

/// @brief Fits a weighted polynomial trend to a complete multichannel series.
/// @details The basis is `1, (t-t[0]), …, (t-t[0])^degree` in original time
///          units. One nonnegative row weight is shared by all channels.
///          Positive weights are max-scaled and normalized; zero-weight rows
///          do not enter the C008 rank-aware least-squares problem. Frequency
///          counts are retained as integer weights. No covariance model is
///          inferred, so the returned covariance is disengaged. Workspace is
///          preflighted with the supplied limits and the C008 solve policy.
/// @param series Complete finite series with strictly increasing timestamps.
/// @param degree Nonnegative polynomial degree.
/// @param weights Supported row-weight interpretation (see C115).
/// @param policy C008 rank, conditioning, and convergence policy.
/// @param limits Simultaneous output/workspace ceilings.
/// @return Coefficients, rank diagnostics, and weighted residual norm per
///         channel, or a failure without partial estimates.
[[nodiscard]] Result<LinearEstimate> trend(const SeriesView& series,
                                           Index degree,
                                           const WeightSpec& weights,
                                           SolvePolicy policy, Limits limits);

} // namespace vss::math

#endif // VSS_MATH_SIGNAL_SERIES_HPP
