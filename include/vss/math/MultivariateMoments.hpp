#ifndef VSS_MATH_MULTIVARIATE_MOMENTS_HPP
#define VSS_MATH_MULTIVARIATE_MOMENTS_HPP

#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/ThirdMoment.hpp>

namespace vss::math {

/// @brief Owns optional typed moments for one fixed component dimension.
/// @details The first moment is a mean. Orders two through four are central
///          moments around a declared center. Missing orders remain distinct
///          from zero-valued tensors. The aggregate validates component counts
///          but does not establish that the moments describe a realizable
///          probability distribution.
class MultivariateMoments final {
public:
  /// @brief Creates an empty aggregate for a positive component dimension.
  /// @param dimension Number of scalar components represented by each moment.
  /// @return The aggregate, or InvalidExtent for a zero dimension.
  [[nodiscard]] static Result<MultivariateMoments> create(Extent dimension);

  MultivariateMoments(const MultivariateMoments&) = default;
  MultivariateMoments& operator=(const MultivariateMoments&) = default;
  MultivariateMoments(MultivariateMoments&&) noexcept = default;
  MultivariateMoments& operator=(MultivariateMoments&&) noexcept = default;
  ~MultivariateMoments() = default;

  /// @brief Returns the fixed number of components.
  [[nodiscard]] Extent dimension() const noexcept;

  /// @brief Replaces a moment only when its component dimension matches.
  /// @return Success or DimensionMismatch; on failure the old value is kept.
  /// @throws std::bad_alloc if copying the validated moment cannot allocate.
  [[nodiscard]] Status setFirstMoment(const FirstMoment& moment);
  /// @copydoc setFirstMoment
  [[nodiscard]] Status setSecondMoment(const SecondMoment& moment);
  /// @copydoc setFirstMoment
  [[nodiscard]] Status setThirdMoment(const ThirdMoment& moment);
  /// @copydoc setFirstMoment
  [[nodiscard]] Status setFourthMoment(const FourthMoment& moment);

  /// @brief Returns an immutable borrowed moment or nullptr when unavailable.
  [[nodiscard]] const FirstMoment* getFirstMoment() const noexcept;
  /// @copydoc getFirstMoment
  [[nodiscard]] const SecondMoment* getSecondMoment() const noexcept;
  /// @copydoc getFirstMoment
  [[nodiscard]] const ThirdMoment* getThirdMoment() const noexcept;
  /// @copydoc getFirstMoment
  [[nodiscard]] const FourthMoment* getFourthMoment() const noexcept;

  /// @brief Marks the first moment unavailable.
  void clearFirstMoment() noexcept;
  /// @copydoc clearFirstMoment
  void clearSecondMoment() noexcept;
  /// @copydoc clearFirstMoment
  void clearThirdMoment() noexcept;
  /// @copydoc clearFirstMoment
  void clearFourthMoment() noexcept;

  /// @brief Calculates the component standard deviation from its variance.
  /// @param component Zero-based component index.
  /// @return The square root of the diagonal second central moment.
  /// @retval ErrorCode::IndexOutOfRange The component is outside the dimension.
  /// @retval ErrorCode::InvalidDomain The second moment is absent or has a
  ///         negative diagonal value.
  /// @retval ErrorCode::NonFiniteResult The computed result is not finite.
  [[nodiscard]] Result<Scalar>
  componentStandardDeviation(Index component) const;

  /// @brief Calculates standardized component skewness.
  /// @details Requires an available mean, second central moment, and third
  ///          central moment. Zero variance is outside the standardized domain.
  [[nodiscard]] Result<Scalar> componentSkewness(Index component) const;
  /// @brief Calculates ordinary component kurtosis (not excess kurtosis).
  [[nodiscard]] Result<Scalar> componentKurtosis(Index component) const;
  /// @brief Calculates component excess kurtosis as ordinary kurtosis minus 3.
  [[nodiscard]] Result<Scalar> componentExcessKurtosis(Index component) const;

private:
  explicit MultivariateMoments(Extent dimension) noexcept;
  [[nodiscard]] Status validateDimension(const Tensor& tensor) const;
  [[nodiscard]] Result<Scalar> componentVariance(Index component) const;
  [[nodiscard]] Status validateStandardizedInputs(Index component) const;

  Extent dimension_;
  std::optional<FirstMoment> first_;
  std::optional<SecondMoment> second_;
  std::optional<ThirdMoment> third_;
  std::optional<FourthMoment> fourth_;
};

} // namespace vss::math

#endif // VSS_MATH_MULTIVARIATE_MOMENTS_HPP
