#ifndef VSS_MATH_AFFINE_MAPS_HPP
#define VSS_MATH_AFFINE_MAPS_HPP

#include <span>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math {

using statistics::MomentEstimate;
using statistics::MomentRequest;

/// @brief Applies an affine map to a flattened tensor and reshapes the result.
/// @param value Owning arbitrary-rank input tensor; flattening is row-major.
/// @param linear Finite `output_count`-by-`input_count` row-major map.
/// @param offset Finite vector with one offset per output coordinate.
/// @param outputShape Positive output extents whose checked product equals the
///        number of rows in `linear`; an empty shape denotes a scalar.
/// @param limits Bounds output elements, bytes, and checked shape construction.
/// @return Owned tensor containing `linear*flatten(value)+offset`.
/// @details Inputs are unchanged. Dimension, shape, finite-value, overflow and
///          resource failures return a typed status without a partial result.
[[nodiscard]] Result<Tensor>
applyAffine(const Tensor& value, ConstMatrixView linear, ConstVectorView offset,
            std::span<const Extent> outputShape, Limits limits);

/// @brief Affinely transforms a mean and requested central moments or
/// cumulants.
/// @param source Moment record with a finite population mean and complete
///        requested source terms.
/// @param linear Finite linear part, with one row per intermediate coordinate.
/// @param offset Finite affine offset.
/// @param request Orders 1 through 4 and output layout; contraction layout
///        supplies an additional linear projection.
/// @param limits Bounds the transformed map, output, and arithmetic work.
/// @return New moment record; input and its views are unchanged.
/// @retval ErrorCode::InvalidDomain Missing requested source orders/tuples,
///         unsupported source moment semantics, or inconsistent request.
/// @retval ErrorCode::DimensionMismatch Incompatible map or contraction shape.
/// @retval ErrorCode::ResourceLimit Output, lookup or arithmetic work exceeds
///         configured limits.
/// @details Mean is transformed by the composed affine map. For central moments
///          and cumulants, orders two through four transform multilinearly on
///          each axis and are independent of the offset. Selected sparse input
///          is never completed with assumed zeros; an omitted contributing
///          tuple fails unless its map coefficient is structurally zero.
[[nodiscard]] Result<MomentEstimate>
transformMoments(const MomentEstimate& source, ConstMatrixView linear,
                 ConstVectorView offset, const MomentRequest& request,
                 Limits limits);

} // namespace vss::math

#endif
