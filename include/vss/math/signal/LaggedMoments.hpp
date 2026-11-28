#ifndef VSS_MATH_SIGNAL_LAGGED_MOMENTS_HPP
#define VSS_MATH_SIGNAL_LAGGED_MOMENTS_HPP

#include <cstdint>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorViews.hpp>
#include <vss/math/signal/Series.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::signal {

using statistics::MomentEstimate;
using statistics::MomentRequest;

/// @brief Declares signed index lags and selected channels for joint moments.
/// @details Each entry of @ref lags has one corresponding nonempty component
///          list. Component indices address the concatenation of the x and y
///          channel axes. The selected components from all lags are stacked in
///          caller order before @ref request is applied. `known` selects a
///          zero center; this structure does not carry a separate center.
struct LagMomentSpec final {
  /// @brief Signed sample-index shifts, one for each component list.
  std::span<const std::int64_t> lags;
  /// @brief Per-lag component lists in the concatenated x/y coordinate space.
  std::span<const IndexList> components;
  /// @brief Estimated stacked-population center or known zero center.
  CenterRule center;
  /// @brief C031 order and output-layout request on the stacked coordinates.
  MomentRequest request;
};

/// @brief Estimates requested moments of aligned observations at signed lags.
/// @param x First validated, timestamped multichannel series.
/// @param y Second validated series on the identical uniform time grid.
/// @param spec Per-lag component selection, center rule, and moment request.
/// @param limits Output and workspace element, byte, and evaluation ceilings.
/// @return Owned moments with diagnostics for the exact retained paired rows.
/// @details A positive lag pairs x at anchor+lag with y at anchor; a negative
///          lag pairs x at anchor+lag with y at anchor. One common anchor range
///          is used for every requested lag. If any selected row is masked,
///          the whole stacked observation is omitted. The routine streams
///          anchors in order and retains no N-by-D resampled matrix. It is a
///          finite empirical statistic and infers no stationarity.
/// @retval ErrorCode::DimensionMismatch Series extents or parallel spec lists
///         disagree.
/// @retval ErrorCode::InvalidDomain Time grids, cadence, masks, center, or
///         moment request are inconsistent.
/// @retval ErrorCode::IndexOutOfRange A selected component is outside x/y.
/// @retval ErrorCode::InsufficientSamples No common valid lagged rows remain.
/// @retval ErrorCode::ResourceLimit Workspace, output, or requested work
///         exceeds @p limits.
[[nodiscard]] Result<MomentEstimate>
laggedHigherMoments(const SeriesView& x, const SeriesView& y,
                    const LagMomentSpec& spec, Limits limits);

} // namespace vss::math::signal

#endif // VSS_MATH_SIGNAL_LAGGED_MOMENTS_HPP
