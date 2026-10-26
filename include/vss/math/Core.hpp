#ifndef VSS_MATH_CORE_HPP
#define VSS_MATH_CORE_HPP

#include <cstddef>
#include <cstdint>

namespace vss::math {

/// @brief Scalar type used by VirtualSensorMath numerical operations.
/// @details The first supported profile requires an 8-byte IEC 60559 binary
///          floating-point representation with radix 2. Values are
///          dimensionless unless a caller-defined domain assigns units.
using Scalar = double;

/// @brief Unsigned type used for indexing numerical objects.
/// @details Indices are zero-based. Values are bounded by the platform's
///          `std::size_t` representation.
using Index = std::size_t;

/// @brief Unsigned type used for vector and matrix extents.
/// @details Extent arithmetic is checked before allocation or pointer-offset
///          computation where the public API promises a `Result` failure.
using Extent = std::size_t;

} // namespace vss::math

#endif // VSS_MATH_CORE_HPP
