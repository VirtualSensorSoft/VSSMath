#ifndef VSS_MATH_THIRD_MOMENT_HPP
#define VSS_MATH_THIRD_MOMENT_HPP

#include <iosfwd>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

/// @brief Owns a validated third-order central-moment tensor.
/// @details The tensor has shape `[d,d,d]` for a positive component count `d`,
///          contains finite values, and is invariant under every axis
///          permutation. Distribution realizability is not implied.
class ThirdMoment final {
public:
  /// @brief Creates a validated deep copy of a third-order moment tensor.
  /// @param tensor Cubic rank-three tensor with finite permutation-symmetric
  /// values.
  /// @param tolerance Tolerance used for symmetry comparisons.
  /// @return The owned moment, or a diagnostic for invalid input.
  /// @throws std::bad_alloc if copying the tensor or diagnostic cannot
  /// allocate.
  [[nodiscard]] static Result<ThirdMoment> create(const Tensor& tensor,
                                                  Tolerance tolerance = {});

  /// @brief Copies this wrapper and its owned tensor.
  ThirdMoment(const ThirdMoment&) = default;
  /// @brief Replaces this wrapper with a deep copy.
  ThirdMoment& operator=(const ThirdMoment&) = default;
  /// @brief Transfers this wrapper's owned tensor.
  ThirdMoment(ThirdMoment&&) noexcept = default;
  /// @brief Replaces this wrapper by transferring its owned tensor.
  ThirdMoment& operator=(ThirdMoment&&) noexcept = default;
  /// @brief Destroys this wrapper and its owned tensor.
  ~ThirdMoment() = default;

  /// @brief Returns immutable access to the owned third central moment tensor.
  /// @return A reference valid while this wrapper remains alive and unchanged.
  [[nodiscard]] const Tensor& tensor() const noexcept;

private:
  explicit ThirdMoment(Tensor tensor) noexcept;
  Tensor tensor_;
};

} // namespace vss::math

namespace vss::math {
/// @brief Writes the named moment tensor with the requested value bound.
std::ostream& print(std::ostream& stream, const ThirdMoment& moment,
                    const PrintOptions& options);
/// @brief Writes a ThirdMoment using default print options.
std::ostream& operator<<(std::ostream& stream, const ThirdMoment& moment);
} // namespace vss::math

#endif // VSS_MATH_THIRD_MOMENT_HPP
