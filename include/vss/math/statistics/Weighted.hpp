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

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_WEIGHTED_HPP
