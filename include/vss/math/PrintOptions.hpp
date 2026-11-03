#ifndef VSS_MATH_PRINT_OPTIONS_HPP
#define VSS_MATH_PRINT_OPTIONS_HPP

#include <limits>
#include <vss/math/Core.hpp>

namespace vss::math {

/// @brief Bounds the amount of numerical data written by a print operation.
struct PrintOptions final {
  /// @brief Maximum number of scalar values to emit; the default is unlimited.
  /// @details If values are omitted, the output includes their omitted count.
  Extent maxElements{std::numeric_limits<Extent>::max()};
};

} // namespace vss::math

#endif // VSS_MATH_PRINT_OPTIONS_HPP
