#ifndef VSS_MATH_GEOMETRY_HPP
#define VSS_MATH_GEOMETRY_HPP

#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/RatioEstimate.hpp>

namespace vss::math {

using statistics::VectorDifferential;

/// @brief Applies a proper orthogonal matrix to a vector.
/// @param value Borrowed finite vector, unchanged by the operation.
/// @param rotation Borrowed square matrix validated as orthogonal with
///        determinant +1 using `tolerance`.
/// @param tolerance Finite nonnegative absolute and relative tolerances.
/// @param limits Bounds the output and simultaneous validation workspace.
/// @return The owned vector `rotation * value`.
/// @retval ErrorCode::DimensionMismatch Incompatible or nonsquare dimensions.
/// @retval ErrorCode::InvalidDomain The matrix is not in SO(n).
/// @retval ErrorCode::NonFiniteInput An operand contains NaN or infinity.
/// @retval ErrorCode::ResourceLimit The output or validation workspace exceeds
///         a configured ceiling.
/// @retval ErrorCode::NonFiniteResult The transformed value is not
/// representable.
/// @details Uses deterministic row-major products with fused multiply-adds and
///          no eigenvector or projective normalization convention.
[[nodiscard]] Result<Vector> rotationTransform(ConstVectorView value,
                                               ConstMatrixView rotation,
                                               Tolerance tolerance,
                                               Limits limits);

/// @brief Normalizes a nonzero vector and returns its analytic derivatives.
/// @param value Borrowed finite vector, unchanged by the operation.
/// @param limits Bounds the value, Jacobian, Hessian and shape metadata.
/// @return Unit direction, Jacobian and Hessian with axes
///         `[output,input,input]`.
/// @retval ErrorCode::InvalidDomain The input is empty or has zero norm.
/// @retval ErrorCode::NonFiniteInput An input coordinate is not finite.
/// @retval ErrorCode::NonFiniteResult A derivative cannot be represented.
/// @retval ErrorCode::ResourceLimit The cubic Hessian or total result exceeds
///         a configured ceiling.
/// @details For `r=||x||` and `n=x/r`, returns
///          `J_ij=(delta_ij-n_i*n_j)/r` and
///          `H_ijk=(3*n_i*n_j*n_k-delta_ij*n_k-delta_ik*n_j-delta_jk*n_i)/r^2`.
///          The norm is computed with scaled sum-of-squares arithmetic.
[[nodiscard]] Result<VectorDifferential>
normalizeVectorDerivatives(ConstVectorView value, Limits limits);

/// @brief Wraps an angular difference into the principal half-open interval.
/// @param value Measured or predicted angle.
/// @param reference Reference angle subtracted from `value`.
/// @param period Strictly positive finite period in the same units.
/// @return `value-reference` modulo `period`, in `[-period/2,period/2)`.
/// @retval ErrorCode::InvalidDomain A period is nonpositive or too small to
///         represent a half-period.
/// @retval ErrorCode::NonFiniteInput An input is not finite.
/// @details Exact positive half-period ties map to the negative endpoint.
///          Both operands are reduced before subtraction to avoid overflow.
[[nodiscard]] Result<Scalar>
wrappedAngularResidual(Scalar value, Scalar reference, Scalar period);

/// @brief Applies a rigid 3D transform to points stored one per row.
/// @param points Borrowed N-by-3 coordinates.
/// @param rotation Proper 3-by-3 rotation in SO(3).
/// @param translation Finite three-coordinate translation.
/// @param tolerance SO(3) validation tolerance.
/// @param limits Bounds the complete output and validation workspace.
/// @return Owned N-by-3 points with rows `R*x_i+t`.
/// @retval ErrorCode::InvalidExtent Points must have exactly three columns.
/// @retval ErrorCode::InvalidDomain Rotation must be proper orthogonal.
/// @retval ErrorCode::NonFiniteInput An input coordinate is non-finite.
/// @retval ErrorCode::NonFiniteResult A transformed coordinate overflows.
/// @details Rejects reflections and singular matrices; no output is returned
///          unless every row is transformed successfully.
[[nodiscard]] Result<Matrix> rigidTransform(ConstMatrixView points,
                                            ConstMatrixView rotation,
                                            ConstVectorView translation,
                                            Tolerance tolerance, Limits limits);

/// @brief Forms a rigid point-residual Jacobian for a right perturbation.
/// @param position Finite 3D point in the local frame.
/// @param rotation Proper 3-by-3 rotation.
/// @param limits Bounds the 3-by-6 output and SO(3) validation workspace.
/// @return Jacobian columns ordered as rotation then translation.
/// @details For `R exp([delta]x)` the rotational block is `-R[x]x`; the
///          translational block is the 3-by-3 identity.
[[nodiscard]] Result<Matrix> rigidResidualJacobian(ConstVectorView position,
                                                   ConstMatrixView rotation,
                                                   Limits limits);

/// @brief Propagates a covariance through a local linear coordinate map.
/// @param jacobian Finite output-by-input derivative matrix.
/// @param covariance Finite symmetric positive-semidefinite input covariance.
/// @param limits Bounds the returned covariance and peak propagation workspace.
/// @return `J*C*J^T` in output coordinate order.
/// @retval ErrorCode::DimensionMismatch Incompatible matrix dimensions.
/// @retval ErrorCode::NotSymmetric The covariance is not symmetric.
/// @retval ErrorCode::NotPositiveSemidefinite The covariance is indefinite.
/// @retval ErrorCode::NonFiniteInput or `NonFiniteResult` for invalid values.
[[nodiscard]] Result<Matrix>
localCoordinateCovariance(ConstMatrixView jacobian, ConstMatrixView covariance,
                          Limits limits);

} // namespace vss::math

#endif
