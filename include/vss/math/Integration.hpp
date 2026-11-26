#ifndef VSS_MATH_INTEGRATION_HPP
#define VSS_MATH_INTEGRATION_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Owned integral estimate and its embedded-rule error estimate.
struct IntegralResult final {
  Scalar value{}; ///< Estimated integral over the requested oriented interval.
  Scalar error_estimate{}; ///< Embedded absolute error estimate.
  Index evaluations{};     ///< Number of callback evaluations used.
  Index panels{};          ///< Number of accepted quadrature panels.
  Termination termination{
      Termination::converged}; ///< Convergence or stop reason.
};

/// @brief Selects the composite rule for a sampled one-dimensional integral.
enum class SampleRule : unsigned char { trapezoid, simpson };

/// @brief Integrates a callback with the embedded Gauss-Kronrod 7/15 rule.
[[nodiscard]] Result<IntegralResult>
quadratureAdaptive(const UnaryFunction& function, Interval limits,
                   const IterPolicy& policy, Limits resources);

/// @brief Integrates strictly increasing samples using trapezoid or composite
///        Simpson quadrature.
[[nodiscard]] Result<Scalar> integrateSamples(ConstVectorView nodes,
                                              ConstVectorView values,
                                              SampleRule rule);

} // namespace vss::math

#endif // VSS_MATH_INTEGRATION_HPP
