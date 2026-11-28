#ifndef VSS_MATH_SIGNAL_TEMPORAL_UTILITIES_HPP
#define VSS_MATH_SIGNAL_TEMPORAL_UTILITIES_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Interpolation.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/signal/Dependence.hpp>
#include <vss/math/signal/Series.hpp>

namespace vss::math::signal {

class AlignedSeries;

/// @cond INTERNAL
namespace detail {
struct AlignedSeriesBuilder;
} // namespace detail
/// @endcond

/// @brief Owns one source's data interpolated onto a common time grid.
/// @details Invalid rows have a zero-filled value row and a zero validity
///          byte. Times and values are owned; no input storage is borrowed.
class AlignedSourceSeries final {
public:
  /// @brief Returns the owned target timestamps for this source.
  [[nodiscard]] const Vector& times() const noexcept { return times_; }
  /// @brief Returns owned row-major interpolated channel values.
  [[nodiscard]] const Matrix& rows() const noexcept { return rows_; }
  /// @brief Returns one validity byte per target row.
  [[nodiscard]] std::span<const std::uint8_t> valid() const noexcept {
    return valid_;
  }
  /// @brief Returns no cadence because target times need not be uniform.
  [[nodiscard]] std::optional<Scalar> uniformStep() const noexcept {
    return uniformStep_;
  }

private:
  AlignedSourceSeries(Vector times, Matrix rows,
                      std::vector<std::uint8_t> valid) noexcept
      : times_(std::move(times)), rows_(std::move(rows)),
        valid_(std::move(valid)) {}

  Vector times_;
  Matrix rows_;
  std::vector<std::uint8_t> valid_;
  std::optional<Scalar> uniformStep_;

  friend class AlignedSeries;
  friend struct detail::AlignedSeriesBuilder;
};

/// @brief Owns one target grid and each input series aligned to that grid.
/// @details Source order and channel widths match the input span. Each output
///          source owns its own timestamp copy, values and validity mask.
class AlignedSeries final {
public:
  /// @brief Returns the owned common target grid.
  [[nodiscard]] const Vector& targetTimes() const noexcept {
    return targetTimes_;
  }
  /// @brief Returns aligned sources in the same order as the input span.
  [[nodiscard]] std::span<const AlignedSourceSeries> sources() const noexcept {
    return sources_;
  }

private:
  AlignedSeries(Vector targetTimes,
                std::vector<AlignedSourceSeries> sources) noexcept
      : targetTimes_(std::move(targetTimes)), sources_(std::move(sources)) {}

  Vector targetTimes_;
  std::vector<AlignedSourceSeries> sources_;

  friend struct detail::AlignedSeriesBuilder;
};

/// @brief Peak association and truncated integrated correlation time.
/// @details `peak_lag` is a signed sample lag; ties select the lowest signed
///          lag. `correlation_time` is `sample_step * (1 + 2 sum(rho[k]))`
///          over positive lags through `cutoff`.
struct LagSummary final {
  std::int64_t peak_lag{};   ///< Signed lag with greatest absolute association.
  Scalar peak_value{};       ///< Original signed association at `peak_lag`.
  Scalar correlation_time{}; ///< Truncated integrated correlation time.
  Index cutoff{};            ///< Inclusive positive-lag cutoff used.
};

/// @brief Computes inclusive prefix sums without a leading zero.
/// @details Finite binary64 values are accumulated in an exact fixed-width
///          dyadic accumulator. A compensated long-double estimate is used
///          when it rounds identically; otherwise exact round-to-nearest,
///          ties-to-even conversion supplies the prefix. Every prefix must be
///          representable, even if later values would cancel it. Empty input
///          returns an empty vector. Work is O(n) with fixed O(1) scratch.
/// @param values Finite borrowed values.
/// @param limits Output and workspace ceilings.
/// @return Owned inclusive prefixes or a diagnostic failure.
[[nodiscard]] Result<Vector> cumulativeSum(ConstVectorView values,
                                           Limits limits);

/// @brief Summarizes the peak and truncated correlation time of lag data.
/// @details Requires a normalized autocorrelation with exactly one zero lag
///          equal to one and one entry for every positive lag through cutoff.
///          All entries are considered for peak selection; ties choose the
///          lowest signed lag. No missing lag is imputed and counts do not
///          weight the result. Work is O(L+cutoff), storage O(cutoff).
/// @param values Borrowed lag/value/count arrays.
/// @param sample_step Positive time represented by one lag step.
/// @param cutoff Inclusive positive-lag summation limit.
/// @param limits Scratch and result ceilings.
/// @return Peak lag/value and truncated correlation time.
[[nodiscard]] Result<LagSummary> summarizeLags(const LagSeries& values,
                                               Scalar sample_step, Index cutoff,
                                               Limits limits);

/// @brief Aligns independent multichannel sources onto a caller-owned grid.
/// @details Times must already share an origin and units. Exact valid knots
///          are copied; interior brackets use linear interpolation. Missing
///          support, rejected extrapolation and over-wide gaps invalidate only
///          that source row. Invalid rows contain positive-zero placeholders.
///          Each result owns all its storage and preserves source order.
/// @param inputs Borrowed source series with finite strictly increasing times.
/// @param target Finite strictly increasing common target timestamps.
/// @param policy Extrapolation and inclusive maximum-gap policy.
/// @param limits Simultaneous output and bounded workspace ceilings.
/// @return Owned aligned grid and one independently masked result per input.
[[nodiscard]] Result<AlignedSeries>
alignSeries(std::span<const SeriesView> inputs, ConstVectorView target,
            const InterpolateSpec& policy, Limits limits);

} // namespace vss::math::signal

#endif // VSS_MATH_SIGNAL_TEMPORAL_UTILITIES_HPP
