#ifndef VSS_MATH_STATISTICS_WEIGHTED_HPP
#define VSS_MATH_STATISTICS_WEIGHTED_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Population.hpp>

namespace vss::math::statistics {

/// @brief Supplies exactly one supported weight representation to an
///        estimator.
/// @details C025 supports empirical row weights and componentwise weights for
///          the weighted mean. Other rules are reserved for separately
///          specified estimators and are rejected here. All views are borrowed
///          only for the duration of the call. Frequency counts are reserved
///          for the frequency-weighted contracts.
struct WeightSpec final {
  /// @brief Declares the statistical interpretation of the supplied weights.
  WeightRule rule;
  /// @brief Optional nonnegative weight for each observation row.
  std::optional<ConstVectorView> rowWeights;
  /// @brief Optional nonnegative weight for each observation cell.
  std::optional<ConstMatrixView> componentWeights;
  /// @brief Reserved exact replication counts; unsupported by C025.
  std::span<const std::uint64_t> frequencyCounts;
};

/// @brief Estimates a weighted coordinate mean from a borrowed population.
/// @param sample Rows of observations and optional cell-validity mask.
/// @param weights Empirical row weights or componentwise mean weights.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return Owned coordinate means and per-coordinate support/weight
///         diagnostics.
/// @details Positive weights are normalized by their per-support maximum;
///          each coordinate uses only valid cells with positive weight.
///          Componentwise weights are accepted only by this operation.
///          Complexity is O(Nd); owned output/workspace is O(d).
[[nodiscard]] Result<Estimate<Vector>> weightedMean(const SampleView& sample,
                                                    const WeightSpec& weights,
                                                    Limits limits);

/// @brief Estimates weighted empirical covariance on common complete rows.
/// @param sample Rows of observations and optional cell-validity mask.
/// @param weights Nonnegative empirical row weights, aligned with rows.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return Owned d-by-d covariance and support diagnostics.
/// @details Uses the normalized weights `a_i = w_i / sum(w)` and a centered
///          outer-product sum. Every retained row must have all coordinates
///          valid; a row with any invalid cell is omitted. Componentwise and
///          other weight rules are rejected. Complexity is O(Nd^2); owned
///          output/workspace is O(d^2).
[[nodiscard]] Result<Estimate<Matrix>>
weightedEmpiricalCovariance(const SampleView& sample, const WeightSpec& weights,
                            Limits limits);

/// @brief Estimates covariance corrected for fixed, observation-independent
///        weights under an IID observation model.
/// @param sample Rows of observations and optional cell-validity mask.
/// @param fixedWeights A `fixed_iid` WeightSpec with one nonnegative row weight
///        per observation; component and frequency weights are not accepted.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return Owned covariance and support diagnostics identifying the fixed-IID
///         weighting premise.
/// @details Uses C025's weighted empirical covariance divided by
///          `1 - sum(a_i^2)`, where `a_i` are normalized weights over retained
///          complete rows. The correction is accumulated as the stable
///          unordered-pair sum `2 sum_{i<j}(s_i s_j) / (sum_i s_i)^2`, with
///          maximum-scaled weights `s_i`. Requires at least two positive
///          retained weights and rejects corrections no greater than
///          `64 * epsilon`. This is unbiased only when weights are fixed
///          independently of IID observations. Complexity is O(Nd^2); owned
///          output and workspace are O(d^2).
[[nodiscard]] Result<Estimate<Matrix>>
unbiasedFixedWeightedCovariance(const SampleView& sample,
                                const WeightSpec& fixedWeights, Limits limits);

/// @brief Estimates the sampling covariance of a fixed-IID weighted mean.
/// @details Multiplies C026's unbiased population covariance by the sum of
///          squared normalized fixed weights. This estimates uncertainty of
///          the weighted mean under independent, identically distributed
///          observations and fixed weights.
[[nodiscard]] Result<Estimate<Matrix>>
weightedMeanCovariance(const SampleView& sample, const WeightSpec& fixedWeights,
                       Limits limits);

/// @brief Estimates covariance from integer observation replication counts.
/// @param sample Rows of observations and optional cell-validity mask.
/// @param counts Exact nonnegative replication count for each sample row.
/// @param mode Selects descriptive empirical or unbiased IID-sample scaling.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return Owned covariance with exact retained replication mass and stored-row
///         support diagnostics.
/// @details Counts are summed with checked `uint64_t` arithmetic and are never
///          expanded into replicated rows. Incomplete rows are omitted using
///          the covariance complete-row rule; zero-count rows do not contribute
///          or require finite values. `iid_sample` has an unbiased
///          interpretation only when the counts describe replicated IID
///          observations. Complexity is O(Nd^2); owned output and workspace
///          are O(d^2). Ratios are formed after exact integer-total checks and
///          accumulated with compensation; scalar rounding is limited to the
///          normalized numerical calculation.
[[nodiscard]] Result<Estimate<Matrix>>
frequencyWeightedCovariance(const SampleView& sample,
                            std::span<const std::uint64_t> counts,
                            VarianceMode mode, Limits limits);

/// @brief Estimates a componentwise mean from known independent measurement
///        variances.
/// @param sample Rows of observations and optional cell-validity mask.
/// @param variances Positive known variances aligned with sample rows and
///        coordinates; the view is borrowed only for this call.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return The precision-weighted component means and diagonal sampling
///         covariance under the supplied independent-error model.
/// @details For each coordinate, uses normalized inverse-variance weights
///          without forming reciprocals, so representable subnormal variances
///          are supported. Invalid sample cells are omitted independently by
///          coordinate; variance entries must nevertheless all be finite and
///          positive. The model assumes a common component mean and mutually
///          independent zero-mean measurement errors. Per-coordinate support
///          is authoritative; global support counts rows with at least one
///          admitted coordinate. Generic weight-mass, effective-count and
///          residual-degree-of-freedom diagnostics are not defined here.
///          Complexity is O(Nd); owned output and workspace are O(d).
[[nodiscard]] Result<Estimate<Vector>>
precisionWeightedMean(const SampleView& sample, ConstMatrixView variances,
                      Limits limits);

/// @brief Propagates a supplied joint covariance through a linear estimate map.
/// @param map Finite k-by-j linear map from source coordinates to estimates.
/// @param jointCovariance Finite symmetric positive-semidefinite j-by-j joint
///        covariance, including any cross-coordinate or cross-time terms.
/// @param limits Maximum result and conservative workspace storage.
/// @return The owned k-by-k covariance `map * jointCovariance * map^T`.
/// @details Preserves correlations and shared-error terms. Complexity is
///          O(kj^2 + k^2j); storage is O(kj + j^2 + k^2).
[[nodiscard]] Result<Matrix>
linearEstimateCovariance(ConstMatrixView map, ConstMatrixView jointCovariance,
                         Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_WEIGHTED_HPP
