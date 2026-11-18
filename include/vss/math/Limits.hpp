#ifndef VSS_MATH_LIMITS_HPP
#define VSS_MATH_LIMITS_HPP

#include <cstddef>
#include <cstdint>

namespace vss::math {

/// @brief Per-call ceilings for output storage and user callback work.
/// @details `elements` limits owned scalar output elements, `bytes` limits
///          simultaneously required output, shape, and bounded workspace, and
///          `evaluations` limits attempted user callback invocations. Inputs
///          are borrowed and are not charged as output storage. A denied
///          operation returns `ErrorCode::ResourceLimit` before allocation or
///          callback invocation where its contract requires preflight.
struct Limits final {
  /// @brief Maximum number of scalar elements in an owned output.
  std::size_t elements;
  /// @brief Maximum bytes for simultaneously required output and shape data.
  std::size_t bytes;
  /// @brief Maximum attempted user callback invocations.
  std::uint64_t evaluations;
};

} // namespace vss::math

#endif // VSS_MATH_LIMITS_HPP
