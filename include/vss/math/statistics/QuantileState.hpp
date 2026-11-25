#ifndef VSS_MATH_STATISTICS_QUANTILE_STATE_HPP
#define VSS_MATH_STATISTICS_QUANTILE_STATE_HPP

#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Order.hpp>

namespace vss::math::statistics {

/// @brief One deterministic Greenwald-Khanna quantile-summary tuple.
/// @details `count` is the tuple's g increment and `slack` is its delta rank
///          uncertainty. Values are ordered by value then stable arrival order.
struct QuantileMarker final {
  Scalar value;
  Index count;
  Index slack;
};

/// @brief A bounded deterministic summary of a finite scalar stream.
/// @details `count` counts all accepted observations; marker storage is
///          separately bounded by `capacity`. No source observations are
///          retained after they are summarized. The owner must not mutate the
///          public fields; every update/query validates the representation.
struct QuantileState final {
  Array<QuantileMarker> markers;
  Scalar epsilon;
  Index capacity;
  Index count;
  Limits limits;
};

/// @brief A sketch estimate and its possible one-based retained rank interval.
/// @details This is a sample-rank certificate, not a population confidence
///          interval. Duplicate values may represent multiple stable
///          occurrences; the interval identifies a possible retained tuple
///          rank and does not imply arrival-identity equality.
struct QuantileBound final {
  Scalar value;
  Index lower_rank;
  Index upper_rank;
};

/// @brief Creates an empty deterministic quantile sketch.
/// @param epsilon Rank error fraction in the open interval (0, 0.5).
/// @param capacity Maximum number of retained tuples, not stream length.
/// @param limits Bound the owned tuple storage and scratch used by updates.
[[nodiscard]] Result<QuantileState>
makeStreamingQuantile(Scalar epsilon, Index capacity, Limits limits);

/// @brief Adds one finite observation transactionally to a quantile sketch.
/// @details The Greenwald-Khanna summary uses deterministic stable insertion
///          for ties and right-to-left compression. When epsilon*N<1 it retains
///          the exact ordered observations. Capacity exhaustion leaves the
///          state unchanged and returns ResourceLimit.
[[nodiscard]] Status add(QuantileState& state, Scalar observation);

/// @brief Queries a sketch for a target empirical quantile and rank bound.
/// @param probability Target probability in [0, 1]; endpoints return exact
///        stream extrema.
[[nodiscard]] Result<QuantileBound>
streamingQuantile(const QuantileState& state, Scalar probability);

/// @brief Computes an exact quantile of the caller's current finite window.
/// @details Sorts a borrowed view for this call and applies the same
///          QuantileRule as @ref quantile. It does not retain or evict data.
[[nodiscard]] Result<Scalar> rollingQuantile(ConstVectorView finiteWindow,
                                             Scalar probability,
                                             QuantileRule rule, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_QUANTILE_STATE_HPP
