#ifndef VSS_MATH_INTERPOLATION_HPP
#define VSS_MATH_INTERPOLATION_HPP

#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Selects how an interpolation operation handles out-of-range queries.
enum class Extrapolation : unsigned char { reject, hold, linear };

/// @brief Policies for scalar piecewise-linear interpolation.
struct InterpolateSpec final {
  Extrapolation extrapolation{Extrapolation::reject};
  std::optional<Scalar> maximum_gap{};
};

/// @brief Selects natural or endpoint-slope cubic-spline boundary conditions.
enum class SplineBoundary : unsigned char { natural, clamped };

/// @brief Boundary conditions used to construct a cubic spline.
struct SplineSpec final {
  SplineBoundary boundary{SplineBoundary::natural};
  std::optional<Scalar> left_derivative{};
  std::optional<Scalar> right_derivative{};
};

/// @brief Owned, piecewise cubic with local normalized-coordinate coefficients.
/// @details `coefficients` has one row per adjacent node pair and four columns
///          in increasing powers of u=(x-nodes[i])/(nodes[i+1]-nodes[i]).
struct Spline final {
  Vector nodes;
  Matrix coefficients;
};

/// @brief Interpolates a scalar value on a finite, strictly increasing grid.
[[nodiscard]] Result<Scalar> interpolate(ConstVectorView nodes,
                                         ConstVectorView values, Scalar x,
                                         const InterpolateSpec& policy);

/// @brief Builds an owned natural or clamped cubic spline.
[[nodiscard]] Result<Spline> makeSpline(ConstVectorView nodes,
                                        ConstVectorView values,
                                        const SplineSpec& policy,
                                        Limits limits);

/// @brief Evaluates an owned or caller-constructed spline aggregate.
[[nodiscard]] Result<Scalar> evaluate(const Spline& spline, Scalar x,
                                      Extrapolation extrapolation);

} // namespace vss::math

#endif // VSS_MATH_INTERPOLATION_HPP
