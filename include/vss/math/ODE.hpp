#ifndef VSS_MATH_ODE_HPP
#define VSS_MATH_ODE_HPP

#include <functional>
#include <optional>
#include <vss/math/Integration.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Vector.hpp>

namespace vss::math {

/// @brief Derivative callback evaluated synchronously with borrowed state and
///        parameter views.
using OdeFunction = std::function<Result<Vector>(Scalar, ConstVectorView,
                                                 ConstVectorView)>;

/// @brief Jacobian callback returning an owned derivative matrix.
using OdeJacobianFunction =
    std::function<Result<Matrix>(Scalar, ConstVectorView, ConstVectorView)>;

/// @brief State and parameter Jacobian callbacks for sensitivity integration.
struct OdeJacobians final {
  OdeJacobianFunction state;
  OdeJacobianFunction parameters;
};

/// @brief Selects an adaptive ODE stepping method.
enum class OdeMethod : unsigned char { dormand_prince_54, backward_euler };

/// @brief Adaptive step controls and optional data for ODE integration.
struct OdePolicy final {
  OdeMethod method{OdeMethod::dormand_prince_54};
  Scalar min_step{};
  Scalar initial_step{};
  Scalar max_step{};
  IterPolicy iteration{};
  std::optional<Vector> parameters{};
  std::optional<Matrix> initial_sensitivity{};
};

/// @brief Requested-time state trajectory and step accounting.
struct Trajectory final {
  Vector times;
  Matrix states;
  Index accepted_steps{};
  Index rejected_steps{};
  Index attempts{};
  Termination termination{Termination::converged};
};

/// @brief State and parameter-sensitivity trajectories.
struct SensitivityTrajectory final {
  Vector times;
  Matrix states;
  Tensor sensitivities;
  Index accepted_steps{};
  Index rejected_steps{};
  Index attempts{};
  Termination termination{Termination::converged};
};

/// @brief Integrates an ODE to each requested time using adaptive steps.
[[nodiscard]] Result<Trajectory> odeIntegrate(const OdeFunction& function,
                                              Vector initial, Vector times,
                                              const OdePolicy& policy,
                                              Limits limits);

/// @brief Integrates an ODE together with sensitivities to its parameters.
[[nodiscard]] Result<SensitivityTrajectory>
odeSensitivity(const OdeFunction& function, const OdeJacobians& jacobians,
               Vector initial, Vector parameters, Vector times,
               const OdePolicy& policy, Limits limits);

} // namespace vss::math

#endif // VSS_MATH_ODE_HPP
