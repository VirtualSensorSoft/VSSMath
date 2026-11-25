#ifndef VSS_MATH_STATISTICS_ESTIMATOR_DIAGNOSTICS_HPP
#define VSS_MATH_STATISTICS_ESTIMATOR_DIAGNOSTICS_HPP

#include <span>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {

/// @brief Validates estimator diagnostics without inferring missing values.
/// @param diagnostics Diagnostics to validate; the input is not modified.
/// @return An owned copy if every populated field is finite and coherent.
/// @retval ErrorCode::InvalidDomain A field is contradictory, out of range,
///         or records a fatal warning as successful output.
/// @retval ErrorCode::NonFiniteInput A populated numeric diagnostic is not
///         finite.
/// @details Missing optional fields and statistical premises remain missing;
///          this function never derives them from the estimate.
[[nodiscard]] Result<EstimatorDiagnostics>
validateEstimatorDiagnostics(const EstimatorDiagnostics& diagnostics);

/// @brief Estimates requested raw moments independently for each group.
/// @param sample Rows are observations and columns are flattened coordinates.
/// @param groupIds One identifier per input row.
/// @param weights Unweighted, empirical row, or compatible component weights.
/// @param request The same moment orders/layout are evaluated per group.
/// @param limits Bounds the combined output, diagnostics, and workspace.
/// @return Flattened values and one support record per output scalar.
/// @retval ErrorCode::DimensionMismatch Group identifiers or weights do not
///         align with the sample.
/// @retval ErrorCode::InsufficientSamples A group has no complete,
///         positive-weight support for the requested estimator.
/// @details Groups are emitted in ascending identifier order. Values within
///          each group follow increasing order and the layout order of
///          @ref estimateRawMoments. A row with any invalid cell is omitted
///          (complete-row policy). Each group's center and weight normalization
///          are independent. Global mass, effective count, degrees of freedom,
///          and weight powers are absent because groups need not share a law.
///          Coordinate support aligns one-for-one with the flattened values.
[[nodiscard]] Result<Estimate<Vector>>
groupedStatistics(const SampleView& sample, std::span<const Index> groupIds,
                  const WeightSpec& weights, const MomentRequest& request,
                  Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_ESTIMATOR_DIAGNOSTICS_HPP
