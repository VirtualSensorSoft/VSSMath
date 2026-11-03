#ifndef VSS_MATH_MOMENT_VALIDATION_HPP
#define VSS_MATH_MOMENT_VALIDATION_HPP

#include <vss/math/Error.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math::detail {

/// @brief Validates moment rank, shape, finite values, and permutation
/// symmetry.
[[nodiscard]] Status validateMomentTensor(const Tensor& tensor, Extent order,
                                          Tolerance tolerance);

} // namespace vss::math::detail

#endif // VSS_MATH_MOMENT_VALIDATION_HPP
