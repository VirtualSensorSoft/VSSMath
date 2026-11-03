#ifndef VSS_MATH_FIRST_MOMENT_HPP
#define VSS_MATH_FIRST_MOMENT_HPP

#include <iosfwd>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {

/// @brief Owns a validated first-order moment tensor (the mean).
/// @details The tensor has shape `[d]` for a positive component count `d` and
///          contains only finite values. The wrapper does not expose mutable
///          aliases to its owned tensor.
class FirstMoment final {
public:
  /// @brief Creates a validated deep copy of a first-order moment tensor.
  /// @param tensor Rank-one tensor with one or more finite components.
  /// @param tolerance Validated comparison tolerance for symmetry checks.
  /// @return The owned moment, or a diagnostic for invalid input.
  /// @throws std::bad_alloc if copying the tensor or diagnostic cannot
  /// allocate.
  [[nodiscard]] static Result<FirstMoment> create(const Tensor& tensor,
                                                  Tolerance tolerance = {});

  /// @brief Copies this wrapper and its owned tensor.
  FirstMoment(const FirstMoment&) = default;
  /// @brief Replaces this wrapper with a deep copy.
  FirstMoment& operator=(const FirstMoment&) = default;
  /// @brief Transfers this wrapper's owned tensor.
  FirstMoment(FirstMoment&&) noexcept = default;
  /// @brief Replaces this wrapper by transferring its owned tensor.
  FirstMoment& operator=(FirstMoment&&) noexcept = default;
  /// @brief Destroys this wrapper and its owned tensor.
  ~FirstMoment() = default;

  /// @brief Returns immutable access to the owned mean tensor.
  /// @return A reference valid while this wrapper remains alive and unchanged.
  [[nodiscard]] const Tensor& tensor() const noexcept;

private:
  explicit FirstMoment(Tensor tensor) noexcept;
  Tensor tensor_;
};

} // namespace vss::math

namespace vss::math {
/// @brief Writes the named moment tensor with the requested value bound.
std::ostream& print(std::ostream& stream, const FirstMoment& moment,
                    const PrintOptions& options);
/// @brief Writes a FirstMoment using default print options.
std::ostream& operator<<(std::ostream& stream, const FirstMoment& moment);
} // namespace vss::math

#endif // VSS_MATH_FIRST_MOMENT_HPP
