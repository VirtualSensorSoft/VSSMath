#ifndef VSS_MATH_FOURTH_MOMENT_HPP
#define VSS_MATH_FOURTH_MOMENT_HPP

#include <iosfwd>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

/// @brief Owns a validated fourth-order central-moment tensor.
/// @details The tensor has shape `[d,d,d,d]` for a positive component count
///          `d`, contains finite values, and is invariant under every axis
///          permutation. Distribution realizability is not implied.
class FourthMoment final {
public:
  /// @brief Creates a validated deep copy of a fourth-order moment tensor.
  /// @param tensor Rank-four tensor with equal positive extents and finite
  ///        permutation-symmetric values.
  /// @param tolerance Tolerance used for symmetry comparisons.
  /// @return The owned moment, or a diagnostic for invalid input.
  /// @throws std::bad_alloc if copying the tensor or diagnostic cannot
  /// allocate.
  [[nodiscard]] static Result<FourthMoment> create(const Tensor& tensor,
                                                   Tolerance tolerance = {});

  /// @brief Copies this wrapper and its owned tensor.
  FourthMoment(const FourthMoment&) = default;
  /// @brief Replaces this wrapper with a deep copy.
  FourthMoment& operator=(const FourthMoment&) = default;
  /// @brief Transfers this wrapper's owned tensor.
  FourthMoment(FourthMoment&&) noexcept = default;
  /// @brief Replaces this wrapper by transferring its owned tensor.
  FourthMoment& operator=(FourthMoment&&) noexcept = default;
  /// @brief Destroys this wrapper and its owned tensor.
  ~FourthMoment() = default;

  /// @brief Returns immutable access to the owned fourth central moment tensor.
  /// @return A reference valid while this wrapper remains alive and unchanged.
  [[nodiscard]] const Tensor& tensor() const noexcept;

private:
  explicit FourthMoment(Tensor tensor) noexcept;
  Tensor tensor_;
};

} // namespace vss::math

namespace vss::math {
/// @brief Writes the named moment tensor with the requested value bound.
std::ostream& print(std::ostream& stream, const FourthMoment& moment,
                    const PrintOptions& options);
/// @brief Writes a FourthMoment using default print options.
std::ostream& operator<<(std::ostream& stream, const FourthMoment& moment);
} // namespace vss::math

#endif // VSS_MATH_FOURTH_MOMENT_HPP
