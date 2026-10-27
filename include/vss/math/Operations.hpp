#ifndef VSS_MATH_OPERATIONS_HPP
#define VSS_MATH_OPERATIONS_HPP

#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Adds corresponding vector elements in deterministic index order.
/// @param lhs Left-hand borrowed vector.
/// @param rhs Right-hand borrowed vector; it must have the same size as @p lhs.
/// @param options Validation and finite-value policy for this call.
/// @return An owning vector on success, or a diagnostic failure with no value.
/// @pre Both views remain valid and non-aliased for the duration of the call;
///      the operation does not mutate either view.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: Linear in the vector extent and constant auxiliary space.
[[nodiscard]] Result<Vector> add(ConstVectorView lhs, ConstVectorView rhs,
                                 OperationOptions options = {});

/// @brief Subtracts corresponding vector elements in deterministic index order.
/// @param lhs Left-hand borrowed vector.
/// @param rhs Right-hand borrowed vector; it must have the same size as @p lhs.
/// @param options Validation and finite-value policy for this call.
/// @return An owning vector on success, or a diagnostic failure with no value.
/// @pre Both views remain valid and non-aliased for the duration of the call;
///      the operation does not mutate either view.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: Linear in the vector extent and constant auxiliary space.
[[nodiscard]] Result<Vector> subtract(ConstVectorView lhs, ConstVectorView rhs,
                                      OperationOptions options = {});

/// @brief Multiplies every vector element by a scalar factor.
/// @param value Borrowed input vector.
/// @param factor Scalar multiplier; it has no physical unit imposed by this
///        library.
/// @param options Validation and finite-value policy for this call.
/// @return An owning vector on success, or a diagnostic failure with no value.
/// @pre The view remains valid and unmodified by other threads for the call.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: Linear in the vector extent and constant auxiliary space.
[[nodiscard]] Result<Vector> scale(ConstVectorView value, Scalar factor,
                                   OperationOptions options = {});

/// @brief Computes a deterministic scalar dot product.
/// @param lhs Left-hand borrowed vector.
/// @param rhs Right-hand borrowed vector; it must have the same size as @p lhs.
/// @param options Validation and finite-value policy for this call.
/// @return The scalar dot product, or a diagnostic failure with no value.
/// @pre Both views remain valid and non-aliased for the duration of the call.
/// @note Complexity: Linear in the vector extent and constant auxiliary space.
[[nodiscard]] Result<Scalar> dot(ConstVectorView lhs, ConstVectorView rhs,
                                 OperationOptions options = {});

/// @brief Computes a Euclidean norm using scaled sum-of-squares.
/// @param value Borrowed input vector.
/// @param options Validation and finite-value policy for this call.
/// @return A non-negative scalar norm, or a diagnostic failure with no value.
/// @pre The view remains valid and unmodified by other threads for the call.
/// @note Complexity: Linear in the vector extent and constant auxiliary space.
[[nodiscard]] Result<Scalar> norm2(ConstVectorView value,
                                   OperationOptions options = {});

/// @brief Multiplies a matrix by a vector in row-major deterministic order.
/// @param matrix Borrowed row-major matrix with @c matrix.columns() equal to
///        @p vector's size.
/// @param vector Borrowed input vector.
/// @param options Validation and finite-value policy for this call.
/// @return An owning vector with @c matrix.rows() elements, or a diagnostic
///         failure with no value.
/// @pre Both views remain valid and non-aliased for the duration of the call.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: O(rows * columns) time and O(1) auxiliary space.
[[nodiscard]] Result<Vector> multiply(ConstMatrixView matrix,
                                      ConstVectorView vector,
                                      OperationOptions options = {});

/// @brief Multiplies two matrices in row-major deterministic order.
/// @param lhs Left-hand borrowed matrix.
/// @param rhs Right-hand borrowed matrix; @c lhs.columns() must equal
///        @c rhs.rows().
/// @param options Validation and finite-value policy for this call.
/// @return An owning matrix with shape `(lhs.rows(), rhs.columns())`, or a
///         diagnostic failure with no value.
/// @pre Both views remain valid and non-aliased for the duration of the call.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: O(lhs.rows() * lhs.columns() * rhs.columns()) time and
///       O(1) auxiliary space apart from the returned matrix.
[[nodiscard]] Result<Matrix> multiply(ConstMatrixView lhs, ConstMatrixView rhs,
                                      OperationOptions options = {});

/// @brief Returns the transpose of a matrix without mutating its input.
/// @param value Borrowed input matrix.
/// @param options Validation and finite-value policy for this call.
/// @return An owning matrix with swapped extents, or a diagnostic failure with
///         no value.
/// @pre The view remains valid and unmodified by other threads for the call.
/// @throws std::bad_alloc if the result cannot be allocated.
/// @note Complexity: O(rows * columns) time and O(1) auxiliary space apart
///       from the returned matrix.
[[nodiscard]] Result<Matrix> transpose(ConstMatrixView value,
                                       OperationOptions options = {});

} // namespace vss::math

#endif // VSS_MATH_OPERATIONS_HPP
