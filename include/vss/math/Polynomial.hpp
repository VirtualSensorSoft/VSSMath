#ifndef VSS_MATH_POLYNOMIAL_HPP
#define VSS_MATH_POLYNOMIAL_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Selects the unnormalized basis used by polynomial operations.
enum class BasisKind : unsigned char { power, chebyshev_first };

/// @brief Defines the affine coordinate used to evaluate a polynomial basis.
/// @details Every basis operation uses t=(x-origin)/scale. A negative scale is
///          valid and reverses the coordinate orientation.
struct BasisSpec final {
  BasisKind kind{BasisKind::power};
  Scalar origin{0.0};
  Scalar scale{1.0};
};

/// @brief Builds rows of an unnormalized power or first-kind Chebyshev basis.
/// @param x Finite coordinates in caller order; repeated coordinates are kept.
/// @param degree Highest basis index; the output has degree+1 columns.
/// @param basis Basis kind and affine coordinate.
/// @param limits Maximum output elements and bytes.
/// @return An owned matrix with one row per coordinate.
/// @retval ErrorCode::NonFiniteInput An input coordinate or affine parameter
///         is non-finite.
/// @retval ErrorCode::InvalidDomain The basis kind is unknown or scale is zero.
/// @retval ErrorCode::ResourceLimit The output shape or requested storage is
///         not representable or exceeds limits.
/// @retval ErrorCode::NonFiniteResult A required basis element is not finite in
///         Scalar representation.
[[nodiscard]] Result<Matrix> basisMatrix(ConstVectorView x, Index degree,
                                         const BasisSpec& basis,
                                         Limits limits);

/// @brief Evaluates coefficients in ascending basis order using a stable
///        recurrence.
/// @param coefficients Coefficients from constant to highest basis index.
/// @param x Finite coordinate at which to evaluate.
/// @param basis Basis kind and affine coordinate.
/// @return A finite scalar polynomial value.
/// @retval ErrorCode::InsufficientSamples No coefficient was supplied.
/// @retval ErrorCode::NonFiniteInput A coefficient, coordinate, or affine
///         parameter is non-finite.
/// @retval ErrorCode::InvalidDomain The basis kind is unknown or scale is zero.
/// @retval ErrorCode::NonFiniteResult The exact recurrence result is outside
///         the finite Scalar range.
[[nodiscard]] Result<Scalar> evaluatePolynomial(ConstVectorView coefficients,
                                                Scalar x,
                                                const BasisSpec& basis);

} // namespace vss::math

#endif // VSS_MATH_POLYNOMIAL_HPP
