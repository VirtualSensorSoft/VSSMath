#ifndef VSS_MATH_TENSOR_OPERATIONS_HPP
#define VSS_MATH_TENSOR_OPERATIONS_HPP

#include <span>
#include <vss/math/NumericalError.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

/// @brief Adds tensors with identical shapes and no broadcasting.
/// @param lhs Left input tensor.
/// @param rhs Right input tensor with exactly the same rank and extents.
/// @param limits Maximum permitted elements for the new tensor.
/// @param options Finite-value validation policy.
/// @return An owning sum, or a failure with no value.
/// @throws std::bad_alloc if result storage cannot be allocated.
[[nodiscard]] Result<Tensor> add(const Tensor& lhs, const Tensor& rhs,
                                 TensorLimits limits = TensorLimits{1'048'576U},
                                 OperationOptions options = {});

/// @brief Subtracts tensors with identical shapes and no broadcasting.
/// @param lhs Left input tensor.
/// @param rhs Right input tensor with exactly the same rank and extents.
/// @param limits Maximum permitted elements for the new tensor.
/// @param options Finite-value validation policy.
/// @return An owning difference, or a failure with no value.
/// @throws std::bad_alloc if result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
subtract(const Tensor& lhs, const Tensor& rhs,
         TensorLimits limits = TensorLimits{1'048'576U},
         OperationOptions options = {});

/// @brief Multiplies every element by a scalar without changing the shape.
/// @param value Input tensor.
/// @param factor Dimensionless scalar multiplier.
/// @param limits Maximum permitted elements for the new tensor.
/// @param options Finite-value validation policy.
/// @return An owning scaled tensor, or a failure with no value.
/// @throws std::bad_alloc if result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
scale(const Tensor& value, Scalar factor,
      TensorLimits limits = TensorLimits{1'048'576U},
      OperationOptions options = {});

/// @brief Forms the outer product, concatenating the operand shapes.
/// @param lhs Left input tensor.
/// @param rhs Right input tensor.
/// @param limits Maximum permitted elements for the result.
/// @param options Finite-value validation policy.
/// @return An owning tensor with shape `lhs.shape() + rhs.shape()`, or a
///         failure with no value.
/// @throws std::bad_alloc if result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
outer(const Tensor& lhs, const Tensor& rhs,
      TensorLimits limits = TensorLimits{1'048'576U},
      OperationOptions options = {});

/// @brief Contracts paired axes of two tensors by summing their products.
/// @details Axis pairs are processed in the caller-provided order. The result
///          axes are the uncontracted left axes followed by the uncontracted
///          right axes, each retaining its original relative order. Empty axis
///          lists therefore compute an outer product. Rank-zero operands
///          follow scalar rules.
/// @param lhs Left input tensor.
/// @param rhs Right input tensor.
/// @param lhsAxes Axes to contract in the left operand.
/// @param rhsAxes Corresponding axes to contract in the right operand.
/// @param limits Maximum permitted elements for the result.
/// @param options Finite-value validation policy.
/// @return An owning contracted tensor, or a failure with no value.
/// @throws std::bad_alloc if result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
contract(const Tensor& lhs, const Tensor& rhs, std::span<const Index> lhsAxes,
         std::span<const Index> rhsAxes,
         TensorLimits limits = TensorLimits{1'048'576U},
         OperationOptions options = {});

/// @brief Adds equal-shaped tensors; a checked failure throws NumericalError.
[[nodiscard]] Tensor operator+(const Tensor& lhs, const Tensor& rhs);
/// @brief Subtracts equal-shaped tensors; a checked failure throws
///        NumericalError.
[[nodiscard]] Tensor operator-(const Tensor& lhs, const Tensor& rhs);
/// @brief Scales a tensor; a checked failure throws NumericalError.
[[nodiscard]] Tensor operator*(const Tensor& value, Scalar factor);
/// @brief Scales a tensor; a checked failure throws NumericalError.
[[nodiscard]] Tensor operator*(Scalar factor, const Tensor& value);

} // namespace vss::math

#endif // VSS_MATH_TENSOR_OPERATIONS_HPP
