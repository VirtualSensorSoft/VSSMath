#ifndef VSS_MATH_TENSOR_ELEMENTWISE_HPP
#define VSS_MATH_TENSOR_ELEMENTWISE_HPP

#include <functional>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>

namespace vss::math {

/// @brief Unary scalar operation used by ordered tensor transforms.
using UnaryFunction = std::function<Result<Scalar>(Scalar)>;

/// @brief Applies a scalar callback in stored row-major order.
/// @param value Borrowed input tensor.
/// @param function Callback invoked exactly once per input element on success.
/// @param limits Output storage and callback-attempt ceilings.
/// @return A fully built tensor, or no value if validation/callback fails.
/// @throws std::bad_alloc if permitted output storage cannot be allocated.
/// @note Callback side effects cannot be rolled back after a later failure.
[[nodiscard]] Result<Tensor> transformElements(const Tensor& value,
                                               const UnaryFunction& function,
                                               Limits limits);

/// @brief Copies a tensor into an explicitly mapped destination shape.
/// @param value Borrowed input tensor.
/// @param destinationShape Destination extents.
/// @param destinationAxisForSource Destination axis for each source axis.
/// @param limits Output element, byte, and bounded workspace ceilings.
/// @return An owning broadcasted tensor or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
broadcastTo(const Tensor& value, std::span<const Extent> destinationShape,
            std::span<const Index> destinationAxisForSource, Limits limits);

} // namespace vss::math

#endif // VSS_MATH_TENSOR_ELEMENTWISE_HPP
