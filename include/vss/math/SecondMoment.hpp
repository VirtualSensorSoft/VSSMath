#ifndef VSS_MATH_SECOND_MOMENT_HPP
#define VSS_MATH_SECOND_MOMENT_HPP

#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

/// @brief Owns a validated second-order central-moment tensor.
/// @details The tensor has shape `[d,d]` for a positive component count `d`,
///          contains finite values, and is symmetric under axis exchange.
///          Positive-semidefinite realizability is not implied.
class SecondMoment final {
public:
  /// @brief Creates a validated deep copy of a second-order moment tensor.
  /// @param tensor Square rank-two tensor with finite symmetric values.
  /// @param tolerance Tolerance used for symmetry comparisons.
  /// @return The owned moment, or a diagnostic for invalid input.
  /// @throws std::bad_alloc if copying the tensor or diagnostic cannot
  /// allocate.
  [[nodiscard]] static Result<SecondMoment> create(const Tensor& tensor,
                                                   Tolerance tolerance = {});

  /// @brief Copies this wrapper and its owned tensor.
  SecondMoment(const SecondMoment&) = default;
  /// @brief Replaces this wrapper with a deep copy.
  SecondMoment& operator=(const SecondMoment&) = default;
  /// @brief Transfers this wrapper's owned tensor.
  SecondMoment(SecondMoment&&) noexcept = default;
  /// @brief Replaces this wrapper by transferring its owned tensor.
  SecondMoment& operator=(SecondMoment&&) noexcept = default;
  /// @brief Destroys this wrapper and its owned tensor.
  ~SecondMoment() = default;

  /// @brief Returns immutable access to the owned covariance tensor.
  /// @return A reference valid while this wrapper remains alive and unchanged.
  [[nodiscard]] const Tensor& tensor() const noexcept;

private:
  explicit SecondMoment(Tensor tensor) noexcept;
  Tensor tensor_;
};

} // namespace vss::math

#endif // VSS_MATH_SECOND_MOMENT_HPP
