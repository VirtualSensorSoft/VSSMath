#ifndef VSS_MATH_TENSOR_VIEWS_HPP
#define VSS_MATH_TENSOR_VIEWS_HPP

#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>

namespace vss::math {

/// @brief Copies a checked rectangular, positive-stride region of a tensor.
/// @param input Borrowed row-major tensor.
/// @param begin First input index on each axis.
/// @param extent Number of output elements on each axis.
/// @param step Positive input-index increment on each axis.
/// @param limits Output element and byte ceilings.
/// @return An owning tensor with the selected values, or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Tensor> slice(const Tensor& input,
                                   std::span<const Extent> begin,
                                   std::span<const Extent> extent,
                                   std::span<const Extent> step, Limits limits);

/// @brief Copies a tensor while preserving its row-major linear element order.
/// @param input Borrowed tensor to reshape.
/// @param extent Requested output extents; an empty span denotes a scalar.
/// @param limits Output element and byte ceilings.
/// @return An owning reshaped tensor or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
reshape(const Tensor& input, std::span<const Extent> extent, Limits limits);

/// @brief Copies a tensor with output axes ordered by the supplied input axes.
/// @param input Borrowed tensor to permute.
/// @param axes For output axis `j`, the corresponding input axis `axes[j]`.
/// @param limits Output element and byte ceilings.
/// @return An owning tensor or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Tensor>
permuteAxes(const Tensor& input, std::span<const Index> axes, Limits limits);

} // namespace vss::math

#endif // VSS_MATH_TENSOR_VIEWS_HPP
