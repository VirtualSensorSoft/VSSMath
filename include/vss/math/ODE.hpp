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
using OdeFunction =
    std::function<Result<Vector>(Scalar, ConstVectorView, ConstVectorView)>;

/// @brief Jacobian callback returning an owned derivative matrix.
using OdeJacobianFunction =
    std::function<Result<Matrix>(Scalar, ConstVectorView, ConstVectorView)>;

/// @brief State and parameter Jacobian callbacks for sensitivity integration.
struct OdeJacobians final {
  OdeJacobianFunction state; ///< Derivative Jacobian with respect to state.
  OdeJacobianFunction
      parameters; ///< Derivative Jacobian with respect to parameters.
};

/// @brief Selects an adaptive ODE stepping method.
enum class OdeMethod : unsigned char { dormand_prince_54, backward_euler };

/// @brief Adaptive step controls and optional data for ODE integration.
struct OdePolicy final {
  OdeMethod method{OdeMethod::dormand_prince_54}; ///< Adaptive stepping method.
  Scalar min_step{};      ///< Smallest accepted step magnitude.
  Scalar initial_step{};  ///< Initial proposed step magnitude.
  Scalar max_step{};      ///< Largest proposed step magnitude.
  IterPolicy iteration{}; ///< Tolerances and work limits for integration.
  std::optional<Vector>
      parameters{}; ///< Parameters supplied to the RHS callback.
  std::optional<Matrix>
      initial_sensitivity{}; ///< Initial state-by-parameter Jacobian.
};

/// @brief Requested-time state trajectory and step accounting.
struct Trajectory final {
  Vector times;           ///< Requested output times in caller order.
  Matrix states;          ///< One state row for each requested time.
  Index accepted_steps{}; ///< Adaptive steps accepted by the solver.
  Index rejected_steps{}; ///< Adaptive steps rejected by the solver.
  Index attempts{};       ///< Total attempted adaptive steps.
  Termination termination{
      Termination::converged}; ///< Completion or stop reason.
};

/// @brief State and parameter-sensitivity trajectories.
struct SensitivityTrajectory final {
  Vector times;           ///< Requested output times in caller order.
  Matrix states;          ///< One state row for each requested time.
  Tensor sensitivities;   ///< Sensitivities with [time, state, parameter] axes.
  Index accepted_steps{}; ///< Adaptive steps accepted by the solver.
  Index rejected_steps{}; ///< Adaptive steps rejected by the solver.
  Index attempts{};       ///< Total attempted adaptive steps.
  Termination termination{
      Termination::converged}; ///< Completion or stop reason.
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
