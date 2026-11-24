#ifndef VSS_MATH_STATISTICS_MOMENT_STATE_HPP
#define VSS_MATH_STATISTICS_MOMENT_STATE_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {

/// @brief Owning form of the moment request retained by streaming states.
/// @details Selector tuples and an optional contraction map are copied during
///          factory construction; no caller-owned view is retained.
struct OwnedMomentRequest final {
  /// @brief Bit mask selecting central moment orders one through four.
  std::uint8_t orders{};
  /// @brief Coordinate representation used for published terms.
  MomentRepresentation representation{MomentRepresentation::component};
  /// @brief Owned ordered selector tuples for selected representation.
  Array<Array<Index>> selected;
  /// @brief Owned linear contraction map, when the request contracts data.
  std::optional<Matrix> contraction;
};

/// @brief Fixed-storage, mergeable online central-moment accumulator.
/// @details Successful additions retain only bounded coordinate and closed
///          moment storage, never observation history. The anchor and offset
///          fields preserve small deviations when the absolute mean is large.
///          Mutation is not thread-safe; callers must externally synchronize
///          aliases. `scratch_*` members are implementation workspace and
///          must not be modified by callers.
struct OnlineMoments final {
  /// @brief Original flattened observation shape; empty denotes scalar input.
  Array<Index> shape;
  /// @brief Published output shape (contracted shape where applicable).
  Array<Index> output_shape;
  /// @brief Number of flattened values required in each input observation.
  Index input_dimension{};
  /// @brief Closure tuple orders backing all requested terms.
  Array<Index> tuple_orders;
  /// @brief Flattened closure coordinates, grouped by tuple order.
  Array<Index> tuple_indices;
  /// @brief Normalized centered sums aligned with the closure tuples.
  Vector centered_sums;
  /// @brief Preallocated transactional update workspace.
  Vector scratch_sums;
  /// @brief Rounded absolute weighted mean in output coordinates.
  Vector mean;
  /// @brief Per-coordinate stable reference point for mean offsets.
  Vector anchor;
  /// @brief Mean displacement from @ref anchor in output coordinates.
  Vector mean_offset;
  /// @brief Preallocated coordinate-shift workspace.
  Vector scratch_delta;
  /// @brief Preallocated mean-offset workspace.
  Vector scratch_mean;
  /// @brief Preallocated projected-observation workspace.
  Vector scratch_observation;
  /// @brief Sum of positive admitted weights.
  Scalar mass{};
  /// @brief Normalized raw weight-power sums of orders one through three.
  std::array<Scalar, 3U> raw_weight_powers{};
  /// @brief Number of admitted observations, including zero weights.
  Index count{};
  /// @brief Number of admitted observations with positive weight.
  Index positive_count{};
  /// @brief Whether all admitted weights were exactly one.
  bool all_unit_weights{true};
  /// @brief Copied request that defines this state's interpretation.
  OwnedMomentRequest stored_request;
  /// @brief Limits used to construct the retained state.
  Limits limits{};
};

/// @brief Fixed-capacity FIFO moment accumulator with two ordered aggregate
///        stacks.
/// @details Back-stack prefixes are combined in insertion order; transfer to
///          the front stack reverses item order while preserving queue order.
///          The fixed buffers are allocated by the factory. Mutation and
///          finalization scratch are not safe for concurrent access.
struct RollingMoments final {
  /// @brief Original flattened observation shape.
  Array<Index> shape;
  /// @brief Output shape after any requested contraction.
  Array<Index> output_shape;
  /// @brief Flattened input observation length.
  Index input_dimension{};
  /// @brief Immutable closure tuple metadata shared with both stack plans.
  Array<Index> tuple_orders;
  /// @brief Flattened closure coordinates in tuple-order groups.
  Array<Index> tuple_indices;
  /// @brief FIFO observations stored in arrival order, including evicted slots.
  Vector observations;
  /// @brief Weights corresponding to the FIFO observation slots.
  Vector weights;
  /// @brief Front-stack observations and weights in transfer order.
  Vector front_observations;
  /// @brief Weights corresponding to the front-stack observation slots.
  Vector front_weights;
  /// @brief Front-stack prefix means, anchors, offsets and centered sums.
  Vector front_means;
  /// @brief Stable per-prefix anchor coordinates for front-stack means.
  Vector front_anchors;
  /// @brief Mean offsets from the front-stack anchors.
  Vector front_offsets;
  /// @brief Normalized centered moment sums for front-stack prefixes.
  Vector front_sums;
  /// @brief Positive weight masses for front-stack prefixes.
  Vector front_masses;
  /// @brief Positive observation counts for front-stack prefixes.
  Vector front_positive_counts;
  /// @brief Encoded all-unit-weight flags for front-stack prefixes.
  Vector front_unit_flags;
  /// @brief Normalized raw weight-power sums for front-stack prefixes.
  Vector front_weight_powers;
  /// @brief Back-stack prefix means, anchors, offsets and centered sums.
  Vector back_means;
  /// @brief Stable per-prefix anchor coordinates for back-stack means.
  Vector back_anchors;
  /// @brief Mean offsets from the back-stack anchors.
  Vector back_offsets;
  /// @brief Normalized centered moment sums for back-stack prefixes.
  Vector back_sums;
  /// @brief Positive weight masses for back-stack prefixes.
  Vector back_masses;
  /// @brief Positive observation counts for back-stack prefixes.
  Vector back_positive_counts;
  /// @brief Encoded all-unit-weight flags for back-stack prefixes.
  Vector back_unit_flags;
  /// @brief Normalized raw weight-power sums for back-stack prefixes.
  Vector back_weight_powers;
  /// @brief Preallocated state workspaces used for transactional stack ops.
  mutable OnlineMoments work;
  /// @brief Second preallocated state workspace for transfer and finalization.
  mutable OnlineMoments work2;
  /// @brief Maximum number of retained observations.
  Index capacity{};
  /// @brief Number of queue items represented by front/back stacks.
  Index front_count{};
  /// @brief Number of queue items in the back stack.
  Index back_count{};
  /// @brief Current number of retained readings.
  Index count{};
  /// @brief Copied request shared by both aggregate stacks.
  OwnedMomentRequest stored_request;
  /// @brief Resource ceilings used when this state was constructed.
  Limits limits{};
};

/// @brief Creates a fixed-storage online moment accumulator.
/// @param shape Original flattened observation shape; empty means scalar.
/// @param request Orders and output coordinate layout.
/// @param limits Maximum retained elements and bytes.
/// @return An empty accumulator or a typed validation/resource error.
[[nodiscard]] Result<OnlineMoments>
makeOnlineMoments(std::span<const Index> shape, const MomentRequest& request,
                  Limits limits);
/// @brief Admits one weighted observation without retaining its history.
/// @return Success or a typed error; rejected observations leave active
///         aggregates unchanged.
[[nodiscard]] Status add(OnlineMoments& state, ConstVectorView observation,
                         Scalar weight);
/// @brief Merges a disjoint compatible state into the destination.
/// @details Callers must ensure the two states represent disjoint observation
///          sets. A rejected merge leaves the destination's active aggregate
///          unchanged. Arbitrary aliasing cannot be detected.
[[nodiscard]] Status merge(OnlineMoments& destination,
                           const OnlineMoments& source);
/// @brief Publishes requested central moments and support diagnostics.
[[nodiscard]] Result<MomentEstimate> finalize(const OnlineMoments& state,
                                              WeightRule rule, Limits limits);
/// @brief Clears active observations while retaining all owned capacity.
[[nodiscard]] Status reset(OnlineMoments& state);

/// @brief Creates a positive-capacity bounded FIFO moment accumulator.
[[nodiscard]] Result<RollingMoments>
makeRollingMoments(std::span<const Index> shape, Index capacity,
                   const MomentRequest& request, Limits limits);
/// @brief Appends an observation, evicting exactly the oldest when full.
[[nodiscard]] Status add(RollingMoments& state, ConstVectorView observation,
                         Scalar weight);
/// @brief Removes the oldest retained observation.
/// @return InsufficientSamples without mutation when the queue is empty.
[[nodiscard]] Status evict(RollingMoments& state);
/// @brief Publishes moments for the current FIFO contents.
[[nodiscard]] Result<MomentEstimate> finalize(const RollingMoments& state,
                                              WeightRule rule, Limits limits);
/// @brief Clears the queue and stack metadata without reallocating capacity.
[[nodiscard]] Status reset(RollingMoments& state);

} // namespace vss::math::statistics

#endif
