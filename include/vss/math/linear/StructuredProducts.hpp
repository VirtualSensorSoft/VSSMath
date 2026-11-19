#ifndef VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP
#define VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

/// @brief Forms the block Kronecker product in row-major matrix order.
/// @param lhs Borrowed finite left matrix.
/// @param rhs Borrowed finite right matrix.
/// @param limits Output element and byte ceilings.
/// @return An owning Kronecker product or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Matrix> kronecker(ConstMatrixView lhs, ConstMatrixView rhs,
                                       Limits limits);

/// @brief Forms the columnwise Kronecker (Khatri--Rao) matrix product.
/// @param lhs Borrowed finite left matrix.
/// @param rhs Borrowed finite right matrix with the same column count.
/// @param limits Output element and byte ceilings.
/// @return An owning columnwise product or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Matrix> khatriRao(ConstMatrixView lhs, ConstMatrixView rhs,
                                       Limits limits);

/// @brief Computes the compensated sum of a square matrix's diagonal.
/// @param value Borrowed finite square matrix; the empty 0x0 trace is zero.
/// @return The trace or a shape, finite-input, or range diagnostic.
[[nodiscard]] Result<Scalar> trace(ConstMatrixView value);

} // namespace vss::math

#endif // VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP
