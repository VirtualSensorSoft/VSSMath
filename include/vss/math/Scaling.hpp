#ifndef VSS_MATH_SCALING_HPP
#define VSS_MATH_SCALING_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Applies a componentwise forward affine unit conversion.
/// @details Computes @c y[i]=scale[i]*x[i]+offset[i]. Negative and zero
///          scales are valid; a zero scale is a constant affine map. This
///          function does not infer or invert unit definitions.
/// @param value Borrowed input values.
/// @param scale Borrowed finite scale factors.
/// @param offset Borrowed finite additive offsets.
/// @param limits Output element and byte ceilings.
/// @return An owning converted vector or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Vector> convertLinearUnits(ConstVectorView value,
                                                ConstVectorView scale,
                                                ConstVectorView offset,
                                                Limits limits);

/// @brief Applies distinct left and right scaling to a covariance-like matrix.
/// @details Computes @c result(i,j)=left[i]*value(i,j)*right[j]. Rectangular
///          cross-covariance matrices are supported and no symmetry is assumed.
/// @param value Borrowed finite square covariance or rectangular
/// cross-covariance.
/// @param left Borrowed finite row scaling, with one value per row.
/// @param right Borrowed finite column scaling, with one value per column.
/// @param limits Output element and byte ceilings.
/// @return An owning transformed matrix or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Matrix> scaleCovariance(ConstMatrixView value,
                                             ConstVectorView left,
                                             ConstVectorView right,
                                             Limits limits);

} // namespace vss::math

#endif // VSS_MATH_SCALING_HPP
