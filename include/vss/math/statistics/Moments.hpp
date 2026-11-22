#ifndef VSS_MATH_STATISTICS_MOMENTS_HPP
#define VSS_MATH_STATISTICS_MOMENTS_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {

/// @brief Selects how coordinate products of one mathematical order are stored.
enum class MomentRepresentation : std::uint8_t {
  component,
  selected,
  packed,
  dense,
  contraction
};

/// @brief Describes the orders, tuple layout and optional linear map requested.
/// @details Bits 0 through 3 of @ref orders select mathematical orders one
///          through four. `selected` borrows an ordered list of equal-length
///          coordinate tuples for the duration of the estimator call.
struct MomentRequest final {
  /// @brief Bit mask selecting orders one, two, three and four respectively.
  std::uint8_t orders;
  /// @brief Storage representation requested for each selected order.
  MomentRepresentation representation;
  /// @brief Borrowed tuple list used only by the selected representation.
  std::span<const std::span<const Index>> selected;
  /// @brief Optional checked map used only by the contraction representation.
  std::optional<ConstMatrixView> contraction;
};

/// @brief Owns values for one order and records their tuple storage layout.
struct MomentTerm final {
  /// @brief Mathematical order in the range one through four.
  Index order;
  /// @brief Layout used for @ref values; contractions are reported as dense.
  MomentRepresentation layout;
  /// @brief Selected tuples, flattened consecutively; empty for implicit
  /// layouts.
  Array<Index> flattened_tuples;
  /// @brief Owned terms in the documented layout order.
  Vector values;
};

/// @brief Owns requested moment estimates with support and center metadata.
struct MomentEstimate final {
  /// @brief Shape of each original observation; empty denotes a scalar.
  Array<Index> observation_shape;
  /// @brief Requested moment terms in increasing mathematical order.
  Array<MomentTerm> terms;
  /// @brief Weight, center, support and estimator-premise diagnostics.
  EstimatorDiagnostics diagnostics;
  /// @brief Weighted population mean in output coordinates when available.
  std::optional<Vector> population_mean;
};

/// @brief Estimates empirical raw coordinate-product moments of orders 1–4.
/// @param sample Borrowed finite observations stored one row per observation.
/// @param weights Unweighted, empirical row weights, or compatible component
///        weights for unmixed terms.
/// @param request Requested orders, representation and optional map.
/// @param limits Maximum simultaneously owned output and workspace resources.
/// @return Owned moments and diagnostics, or a typed validation/numerical
/// error.
/// @details Uses the weighted population ratio without an IID correction.
///          Supports are shared for unweighted/row-weighted terms; component
///          weights are permitted only for unmixed coordinate powers. Dense
///          terms are lexicographic, packed terms are nondecreasing tuples,
///          and selected terms preserve caller order. The input and all views
///          are borrowed only for this call. No callbacks are invoked.
[[nodiscard]] Result<MomentEstimate>
estimateRawMoments(const SampleView& sample, const WeightSpec& weights,
                   const MomentRequest& request, Limits limits);

/// @brief Estimates empirical central moments around an estimated or known
///        center.
/// @param sample Borrowed finite observations stored one row per observation.
/// @param weights Unweighted, empirical row weights, or compatible component
///        weights for unmixed terms.
/// @param center Estimated center on the same support, or a borrowed known
///        center in original observation coordinates.
/// @param request Requested orders, representation and optional map.
/// @param limits Maximum simultaneously owned output and workspace resources.
/// @return Owned central moments and diagnostics, or a typed error.
/// @details With an estimated center, first central terms are exactly zero.
///          A known center may produce nonzero first central terms. A supplied
///          contraction is applied to every coordinate axis before moments
///          are formed; its output shape is the one-dimensional map range.
[[nodiscard]] Result<MomentEstimate>
estimateCentralMoments(const SampleView& sample, const WeightSpec& weights,
                       const CenterSpec& center, const MomentRequest& request,
                       Limits limits);

/// @brief Estimates an unbiased order-three central moment tensor using fixed
///        row weights independent of IID observations.
/// @param sample Complete finite observations, one observation per row.
/// @param fixedWeights A `fixed_iid` specification with one nonnegative weight
///        per row and at least three positive weights.
/// @param request Exactly the order-three bit and one supported moment layout.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return The corrected moment tensor, or a typed validation, resource, or
///         numerical-certification error.
/// @details Applies the correction `H W^2/(6 e_3)` using the positive
///          elementary symmetric sum of raw weights. All requested entries
///          share one weighted center and row support. Optional linear
///          contractions are applied before forming the central tensor. The
///          fixed-IID premise is supplied by the caller and cannot be inferred
///          from numeric weights. No sampling covariance is estimated.
[[nodiscard]] Result<MomentEstimate>
unbiasedFixedWeightedThirdMoment(const SampleView& sample,
                                 const WeightSpec& fixedWeights,
                                 const MomentRequest& request, Limits limits);

/// @brief Estimates an unbiased order-four central moment tensor using fixed
///        row weights independent of IID observations.
/// @param sample Complete finite observations, one observation per row.
/// @param fixedWeights A `fixed_iid` specification with one nonnegative weight
///        per row and at least four positive weights.
/// @param request Exactly the order-four bit and one supported moment layout.
/// @param limits Maximum output and simultaneously required workspace storage.
/// @return The corrected fourth central moment tensor, or a typed validation,
///         resource, or numerical-certification error.
/// @details Solves the scaled two-by-two expectation system for the weighted
///          empirical fourth central moment and the mean of the three
///          covariance pairings of the empirical second central moment. The
///          result is only the first solution coordinate. All requested
///          entries share one weighted center and row support. Fixed-IID
///          assumptions are supplied by the caller and cannot be inferred
///          from numeric weights. No estimator covariance is estimated.
[[nodiscard]] Result<MomentEstimate>
unbiasedFixedWeightedFourthMoment(const SampleView& sample,
                                  const WeightSpec& fixedWeights,
                                  const MomentRequest& request, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_MOMENTS_HPP
