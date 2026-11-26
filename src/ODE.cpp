#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/ODE.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/QR.hpp>

namespace vss::math {
namespace {

using Wide = long double;

struct Step final {
  Status status{Status::success()};
  bool rejected{false};
  std::vector<Scalar> state;
  std::vector<Scalar> sensitivity;
  Wide error{0.0L};
};

struct Engine final {
  const OdeFunction& function;
  const OdeJacobians* jacobians;
  const Vector& parameters;
  const OdePolicy& policy;
  Index stateSize;
  Index parameterSize;
  Index budget;
  Index evaluations{0U};

  Result<std::vector<Scalar>> derivative(Scalar time,
                                         const std::vector<Scalar>& state) {
    if (evaluations >= budget) {
      return Result<std::vector<Scalar>>::failure(
          ErrorCode::ResourceLimit, "ODE callback evaluation budget exhausted");
    }
    ++evaluations;
    Vector stateValue(stateSize);
    for (Index i = 0U; i < stateSize; ++i) {
      stateValue.at(i) = state[i];
    }
    try {
      auto result = function(time, stateValue.view(), parameters.view());
      if (!result) {
        return Result<std::vector<Scalar>>::failure(
            result.status().code(), std::string(result.status().message()));
      }
      if (result.value().size() != stateSize) {
        return Result<std::vector<Scalar>>::failure(
            ErrorCode::DimensionMismatch,
            "ODE derivative callback returned the wrong extent");
      }
      std::vector<Scalar> values(stateSize);
      for (Index i = 0U; i < stateSize; ++i) {
        values[i] = result.value().at(i);
        if (!std::isfinite(values[i])) {
          return Result<std::vector<Scalar>>::failure(
              ErrorCode::NonFiniteInput,
              "ODE derivative callback returned a non-finite component");
        }
      }
      return Result<std::vector<Scalar>>::success(std::move(values));
    } catch (...) {
      return Result<std::vector<Scalar>>::failure(
          ErrorCode::CallbackFailure, "ODE derivative callback threw");
    }
  }

  Result<Matrix> jacobian(const OdeJacobianFunction& callback, Scalar time,
                          const std::vector<Scalar>& state, Extent expectedRows,
                          Extent expectedColumns) {
    if (!callback) {
      return Result<Matrix>::failure(ErrorCode::CallbackFailure,
                                     "required ODE Jacobian callback is empty");
    }
    if (evaluations >= budget) {
      return Result<Matrix>::failure(
          ErrorCode::ResourceLimit, "ODE callback evaluation budget exhausted");
    }
    ++evaluations;
    Vector stateValue(stateSize);
    for (Index i = 0U; i < stateSize; ++i) {
      stateValue.at(i) = state[i];
    }
    try {
      auto result = callback(time, stateValue.view(), parameters.view());
      if (!result) {
        return Result<Matrix>::failure(result.status().code(),
                                       std::string(result.status().message()));
      }
      if (result.value().rows() != expectedRows ||
          result.value().columns() != expectedColumns) {
        return Result<Matrix>::failure(ErrorCode::DimensionMismatch,
                                       "ODE Jacobian has the wrong shape");
      }
      for (Index i = 0U; i < expectedRows; ++i) {
        for (Index j = 0U; j < expectedColumns; ++j) {
          if (!std::isfinite(result.value().at(i, j))) {
            return Result<Matrix>::failure(
                ErrorCode::NonFiniteInput,
                "ODE Jacobian callback returned non-finite data");
          }
        }
      }
      return result;
    } catch (...) {
      return Result<Matrix>::failure(ErrorCode::CallbackFailure,
                                     "ODE Jacobian callback threw");
    }
  }
};

bool toScalar(Wide value, Scalar& result) noexcept {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
    return false;
  }
  result = static_cast<Scalar>(value);
  return std::isfinite(result);
}

Wide componentError(Wide error, Scalar oldValue, Scalar newValue,
                    const IterPolicy& policy) noexcept {
  const Wide denominator = static_cast<Wide>(policy.abs) +
                           static_cast<Wide>(policy.rel) *
                               std::max(std::abs(static_cast<Wide>(oldValue)),
                                        std::abs(static_cast<Wide>(newValue)));
  if (denominator == 0.0L) {
    return error == 0.0L ? 0.0L : std::numeric_limits<Wide>::infinity();
  }
  return std::abs(error) / denominator;
}

bool checkedMul(std::size_t left, std::size_t right,
                std::size_t& output) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

bool checkedAdd(std::size_t left, std::size_t right,
                std::size_t& output) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

Status validateCommon(const OdeFunction& function, const Vector& initial,
                      const Vector& parameters, const Vector& times,
                      const OdePolicy& policy, Limits limits, bool sensitivity,
                      Extent parameterCount) {
  if (times.empty() || initial.empty()) {
    return Status::failure(
        ErrorCode::InsufficientSamples,
        "ODE requires a state and at least one requested time");
  }
  for (Index i = 0U; i < times.size(); ++i) {
    if (!std::isfinite(times.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "ODE requested time is non-finite");
    }
    if (i > 0U && !(times.at(i) > times.at(i - 1U))) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "ODE requested times must strictly increase");
    }
  }
  for (Index i = 0U; i < initial.size(); ++i) {
    if (!std::isfinite(initial.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "ODE initial state is non-finite");
    }
  }
  for (Index i = 0U; i < parameters.size(); ++i) {
    if (!std::isfinite(parameters.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "ODE parameter is non-finite");
    }
  }
  if (!std::isfinite(policy.min_step) || !std::isfinite(policy.initial_step) ||
      !std::isfinite(policy.max_step) || policy.min_step <= 0.0 ||
      policy.initial_step < policy.min_step ||
      policy.max_step < policy.initial_step ||
      !std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.abs < 0.0 ||
      policy.iteration.rel < 0.0 ||
      (policy.iteration.abs == 0.0 && policy.iteration.rel == 0.0)) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "ODE step controls or tolerances are invalid");
  }
  if (policy.iteration.bracket.has_value() ||
      (policy.method != OdeMethod::dormand_prince_54 &&
       policy.method != OdeMethod::backward_euler)) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "ODE solver kind or bracket policy is inapplicable");
  }
  if (sensitivity && policy.parameters.has_value()) {
    if (policy.parameters->size() != parameters.size()) {
      return Status::failure(
          ErrorCode::DimensionMismatch,
          "ODE policy and explicit parameter extents differ");
    }
    for (Index i = 0U; i < parameters.size(); ++i) {
      if (policy.parameters->at(i) != parameters.at(i)) {
        return Status::failure(
            ErrorCode::InvalidDomain,
            "ODE policy parameters differ from explicit values");
      }
    }
  }
  if (sensitivity && policy.initial_sensitivity.has_value() &&
      (policy.initial_sensitivity->rows() != initial.size() ||
       policy.initial_sensitivity->columns() != parameterCount)) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "initial ODE sensitivity has the wrong shape");
  }
  if (times.size() > 1U && !function) {
    return Status::failure(ErrorCode::CallbackFailure,
                           "ODE derivative callback is empty");
  }

  std::size_t states = 0U;
  std::size_t sensitivities = 0U;
  std::size_t total = 0U;
  std::size_t outputBytes = 0U;
  std::size_t scratch = 0U;
  std::size_t scratchBytes = 0U;
  if (parameterCount > (std::numeric_limits<std::size_t>::max() - 16U) / 12U ||
      !checkedMul(times.size(), initial.size(), states) ||
      (sensitivity && !checkedMul(states, parameterCount, sensitivities)) ||
      !checkedAdd(states, sensitivities, total) ||
      !checkedMul(total, sizeof(Scalar), outputBytes) ||
      !checkedMul(initial.size(), 16U + 12U * parameterCount, scratch) ||
      !checkedMul(scratch, sizeof(Scalar), scratchBytes) ||
      !checkedAdd(outputBytes, scratchBytes, outputBytes)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "ODE checked size arithmetic overflowed");
  }
  if (total > limits.elements || outputBytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "ODE output or stage workspace exceeds limits");
  }
  return Status::success();
}

Step dopriStep(Engine& engine, Scalar time, Scalar step,
               const std::vector<Scalar>& state,
               const std::vector<Scalar>& sensitivity,
               Index sensitivityColumns) {
  constexpr std::array<Scalar, 7> c{0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0};
  constexpr std::array<std::array<Scalar, 6>, 7> a{
      {{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
       {{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
       {{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0}},
       {{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0}},
       {{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
         0.0, 0.0}},
       {{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
         -5103.0 / 18656.0, 0.0}},
       {{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
         11.0 / 84.0}}}};
  constexpr std::array<Scalar, 7> fifth{
      35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0,  0.0};
  constexpr std::array<Scalar, 7> fourth{5179.0 / 57600.0,    0.0,
                                         7571.0 / 16695.0,    393.0 / 640.0,
                                         -92097.0 / 339200.0, 187.0 / 2100.0,
                                         1.0 / 40.0};
  const Index n = state.size();
  const Index q = sensitivityColumns;
  std::array<std::vector<Scalar>, 7> k;
  std::array<std::vector<Scalar>, 7> sk;
  for (Index stage = 0U; stage < 7U; ++stage) {
    std::vector<Scalar> yStage(n);
    std::vector<Scalar> sStage(sensitivity.size());
    for (Index i = 0U; i < n; ++i) {
      Wide value = state[i];
      for (Index previous = 0U; previous < stage; ++previous) {
        value += static_cast<Wide>(step) * a[stage][previous] * k[previous][i];
      }
      if (!toScalar(value, yStage[i])) {
        return {Status::success(), true, {}, {}, 0.0L};
      }
    }
    for (Index i = 0U; i < sensitivity.size(); ++i) {
      Wide value = sensitivity[i];
      for (Index previous = 0U; previous < stage; ++previous) {
        value += static_cast<Wide>(step) * a[stage][previous] * sk[previous][i];
      }
      if (!toScalar(value, sStage[i])) {
        return {Status::success(), true, {}, {}, 0.0L};
      }
    }
    const Wide stageTimeWide =
        static_cast<Wide>(time) + static_cast<Wide>(step) * c[stage];
    Scalar stageTime = 0.0;
    if (!toScalar(stageTimeWide, stageTime)) {
      return {Status::success(), true, {}, {}, 0.0L};
    }
    auto derivative = engine.derivative(stageTime, yStage);
    if (!derivative) {
      return {Status::failure(derivative.status().code(),
                              std::string(derivative.status().message())),
              false,
              {},
              {},
              0.0L};
    }
    k[stage] = std::move(derivative).value();
    if (q > 0U) {
      auto jy =
          engine.jacobian(engine.jacobians->state, stageTime, yStage, n, n);
      if (!jy) {
        return {Status::failure(jy.status().code(),
                                std::string(jy.status().message())),
                false,
                {},
                {},
                0.0L};
      }
      auto jp = engine.jacobian(engine.jacobians->parameters, stageTime, yStage,
                                n, q);
      if (!jp) {
        return {Status::failure(jp.status().code(),
                                std::string(jp.status().message())),
                false,
                {},
                {},
                0.0L};
      }
      sk[stage].assign(n * q, 0.0);
      for (Index i = 0U; i < n; ++i) {
        for (Index j = 0U; j < q; ++j) {
          Wide value = jp.value().at(i, j);
          for (Index kIndex = 0U; kIndex < n; ++kIndex) {
            value += static_cast<Wide>(jy.value().at(i, kIndex)) *
                     sStage[kIndex * q + j];
          }
          if (!toScalar(value, sk[stage][i * q + j])) {
            return {Status::success(), true, {}, {}, 0.0L};
          }
        }
      }
    }
  }
  std::vector<Scalar> candidate(n);
  std::vector<Scalar> candidateSensitivity(sensitivity.size());
  Wide error = 0.0L;
  for (Index i = 0U; i < n; ++i) {
    Wide fifthValue = state[i];
    Wide fourthValue = state[i];
    for (Index stage = 0U; stage < 7U; ++stage) {
      fifthValue += static_cast<Wide>(step) * fifth[stage] * k[stage][i];
      fourthValue += static_cast<Wide>(step) * fourth[stage] * k[stage][i];
    }
    if (!toScalar(fifthValue, candidate[i])) {
      return {Status::success(), true, {}, {}, 0.0L};
    }
    error =
        std::max(error, componentError(fifthValue - fourthValue, state[i],
                                       candidate[i], engine.policy.iteration));
  }
  for (Index i = 0U; i < sensitivity.size(); ++i) {
    Wide fifthValue = sensitivity[i];
    Wide fourthValue = sensitivity[i];
    for (Index stage = 0U; stage < 7U; ++stage) {
      fifthValue += static_cast<Wide>(step) * fifth[stage] * sk[stage][i];
      fourthValue += static_cast<Wide>(step) * fourth[stage] * sk[stage][i];
    }
    if (!toScalar(fifthValue, candidateSensitivity[i])) {
      return {Status::success(), true, {}, {}, 0.0L};
    }
    error =
        std::max(error, componentError(fifthValue - fourthValue, sensitivity[i],
                                       candidateSensitivity[i],
                                       engine.policy.iteration));
  }
  return {Status::success(), error > 1.0L, std::move(candidate),
          std::move(candidateSensitivity), error};
}

struct BackwardStep final {
  Status status{Status::success()};
  bool failed{false};
  std::vector<Scalar> state;
  std::vector<Scalar> sensitivity;
};

BackwardStep backwardEulerSubstep(Engine& engine, Scalar time, Scalar step,
                                  const std::vector<Scalar>& oldState,
                                  const std::vector<Scalar>& oldSensitivity,
                                  Index q) {
  const Index n = oldState.size();
  std::vector<Scalar> candidate = oldState;
  std::vector<Scalar> residual(n);
  Matrix jy(n, n, 0.0);
  bool converged = false;
  for (Index update = 0U; update <= 12U; ++update) {
    auto derivative =
        engine.derivative(static_cast<Scalar>(time + step), candidate);
    if (!derivative) {
      return {Status::failure(derivative.status().code(),
                              std::string(derivative.status().message())),
              false,
              {},
              {}};
    }
    Wide norm = 0.0L;
    for (Index i = 0U; i < n; ++i) {
      const Wide g = static_cast<Wide>(candidate[i]) - oldState[i] -
                     static_cast<Wide>(step) * derivative.value()[i];
      if (!toScalar(g, residual[i])) {
        return {Status::success(), true, {}, {}};
      }
      const Wide denominator =
          static_cast<Wide>(engine.policy.iteration.abs) +
          static_cast<Wide>(engine.policy.iteration.rel) *
              std::max(std::abs(static_cast<Wide>(candidate[i])),
                       std::abs(static_cast<Wide>(oldState[i])));
      const Wide component =
          denominator == 0.0L
              ? (g == 0.0L ? 0.0L : std::numeric_limits<Wide>::infinity())
              : std::abs(g) / denominator;
      norm = std::max(norm, component);
    }
    if (norm <= 0.01L) {
      converged = true;
      break;
    }
    if (update == 12U) {
      return {Status::success(), true, {}, {}};
    }

    if (engine.jacobians != nullptr && q > 0U) {
      auto stateJacobian =
          engine.jacobian(engine.jacobians->state,
                          static_cast<Scalar>(time + step), candidate, n, n);
      if (!stateJacobian) {
        return {Status::failure(stateJacobian.status().code(),
                                std::string(stateJacobian.status().message())),
                false,
                {},
                {}};
      }
      jy = std::move(stateJacobian).value();
    } else {
      for (Index column = 0U; column < n; ++column) {
        const Scalar original = candidate[column];
        const Wide proposed =
            static_cast<Wide>(original) +
            std::sqrt(std::numeric_limits<Scalar>::epsilon()) *
                std::max(1.0L, std::abs(static_cast<Wide>(original)));
        Scalar perturbed = 0.0;
        bool forward = toScalar(proposed, perturbed) && perturbed != original;
        if (!forward) {
          const Wide backward =
              static_cast<Wide>(original) -
              std::sqrt(std::numeric_limits<Scalar>::epsilon()) *
                  std::max(1.0L, std::abs(static_cast<Wide>(original)));
          if (!toScalar(backward, perturbed) || perturbed == original) {
            return {Status::success(), true, {}, {}};
          }
        }
        candidate[column] = perturbed;
        auto shifted =
            engine.derivative(static_cast<Scalar>(time + step), candidate);
        candidate[column] = original;
        if (!shifted) {
          return {Status::failure(shifted.status().code(),
                                  std::string(shifted.status().message())),
                  false,
                  {},
                  {}};
        }
        const Wide displacement = static_cast<Wide>(perturbed) - original;
        for (Index row = 0U; row < n; ++row) {
          Scalar entry = 0.0;
          if (!toScalar((static_cast<Wide>(shifted.value()[row]) -
                         derivative.value()[row]) /
                            displacement,
                        entry)) {
            return {Status::success(), true, {}, {}};
          }
          jy.at(row, column) = entry;
        }
      }
    }
    Matrix system(n, n, 0.0);
    Vector rhs(n);
    for (Index i = 0U; i < n; ++i) {
      rhs.at(i) = -residual[i];
      for (Index j = 0U; j < n; ++j) {
        system.at(i, j) = (i == j ? 1.0 : 0.0) - step * jy.at(i, j);
      }
    }
    linear::FactorizationOptions factorOptions{};
    factorOptions.rankTolerance = {
        0.0, 64.0 * std::numeric_limits<Scalar>::epsilon() *
                 static_cast<Scalar>(std::max<Index>(1U, n))};
    factorOptions.conditionLimit = 1.0e300;
    auto factor = linear::factorizeQR(system.view(), factorOptions);
    if (!factor || factor.value().rank() != n) {
      return {Status::success(), true, {}, {}};
    }
    auto direction = factor.value().leastSquares(
        rhs.view(),
        {.illConditionedPolicy =
             linear::IllConditionedSolvePolicy::AllowWithDiagnostic});
    if (!direction) {
      return {Status::success(), true, {}, {}};
    }
    bool accepted = false;
    for (Index backtrack = 0U; backtrack <= 8U; ++backtrack) {
      const Wide scale = std::ldexp(1.0L, -static_cast<int>(backtrack));
      std::vector<Scalar> trial(n);
      bool representable = true;
      for (Index i = 0U; i < n; ++i) {
        if (!toScalar(static_cast<Wide>(candidate[i]) +
                          scale * direction.value().solution().at(i),
                      trial[i])) {
          representable = false;
          break;
        }
      }
      if (!representable || trial == candidate) {
        continue;
      }
      auto trialDerivative =
          engine.derivative(static_cast<Scalar>(time + step), trial);
      if (!trialDerivative) {
        return {
            Status::failure(trialDerivative.status().code(),
                            std::string(trialDerivative.status().message())),
            false,
            {},
            {}};
      }
      Wide trialNorm = 0.0L;
      for (Index i = 0U; i < n; ++i) {
        const Wide g = static_cast<Wide>(trial[i]) - oldState[i] -
                       static_cast<Wide>(step) * trialDerivative.value()[i];
        const Wide denominator =
            static_cast<Wide>(engine.policy.iteration.abs) +
            static_cast<Wide>(engine.policy.iteration.rel) *
                std::max(std::abs(static_cast<Wide>(trial[i])),
                         std::abs(static_cast<Wide>(oldState[i])));
        trialNorm = std::max(
            trialNorm,
            denominator == 0.0L
                ? (g == 0.0L ? 0.0L : std::numeric_limits<Wide>::infinity())
                : std::abs(g) / denominator);
      }
      if (trialNorm < norm) {
        candidate = std::move(trial);
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      return {Status::success(), true, {}, {}};
    }
  }
  if (!converged) {
    return {Status::success(), true, {}, {}};
  }

  std::vector<Scalar> newSensitivity(oldSensitivity.size());
  if (q > 0U) {
    auto stateJacobian =
        engine.jacobian(engine.jacobians->state,
                        static_cast<Scalar>(time + step), candidate, n, n);
    if (!stateJacobian) {
      return {Status::failure(stateJacobian.status().code(),
                              std::string(stateJacobian.status().message())),
              false,
              {},
              {}};
    }
    auto parameterJacobian =
        engine.jacobian(engine.jacobians->parameters,
                        static_cast<Scalar>(time + step), candidate, n, q);
    if (!parameterJacobian) {
      return {
          Status::failure(parameterJacobian.status().code(),
                          std::string(parameterJacobian.status().message())),
          false,
          {},
          {}};
    }
    Matrix system(n, n, 0.0);
    for (Index i = 0U; i < n; ++i) {
      for (Index j = 0U; j < n; ++j) {
        system.at(i, j) =
            (i == j ? 1.0 : 0.0) - step * stateJacobian.value().at(i, j);
      }
    }
    linear::FactorizationOptions options{};
    options.rankTolerance = {0.0,
                             64.0 * std::numeric_limits<Scalar>::epsilon() *
                                 static_cast<Scalar>(std::max<Index>(1U, n))};
    options.conditionLimit = 1.0e300;
    auto factor = linear::factorizeQR(system.view(), options);
    if (!factor || factor.value().rank() != n) {
      return {Status::success(), true, {}, {}};
    }
    for (Index column = 0U; column < q; ++column) {
      Vector rhs(n);
      for (Index i = 0U; i < n; ++i) {
        Wide value =
            oldSensitivity[i * q + column] +
            static_cast<Wide>(step) * parameterJacobian.value().at(i, column);
        if (!toScalar(value, rhs.at(i))) {
          return {Status::success(), true, {}, {}};
        }
      }
      auto solved = factor.value().leastSquares(
          rhs.view(),
          {.illConditionedPolicy =
               linear::IllConditionedSolvePolicy::AllowWithDiagnostic});
      if (!solved) {
        return {Status::success(), true, {}, {}};
      }
      for (Index i = 0U; i < n; ++i) {
        newSensitivity[i * q + column] = solved.value().solution().at(i);
      }
    }
  }
  return {Status::success(), false, std::move(candidate),
          std::move(newSensitivity)};
}

Step backwardStep(Engine& engine, Scalar time, Scalar step,
                  const std::vector<Scalar>& state,
                  const std::vector<Scalar>& sensitivity, Index q) {
  auto full = backwardEulerSubstep(engine, time, step, state, sensitivity, q);
  if (!full.status.ok()) {
    return {full.status, false, {}, {}, 0.0L};
  }
  if (full.failed) {
    return {Status::success(), true, {}, {}, 0.0L};
  }
  auto halfOne =
      backwardEulerSubstep(engine, time, step / 2.0, state, sensitivity, q);
  if (!halfOne.status.ok()) {
    return {halfOne.status, false, {}, {}, 0.0L};
  }
  if (halfOne.failed) {
    return {Status::success(), true, {}, {}, 0.0L};
  }
  auto halfTwo = backwardEulerSubstep(engine, time + step / 2.0, step / 2.0,
                                      halfOne.state, halfOne.sensitivity, q);
  if (!halfTwo.status.ok()) {
    return {halfTwo.status, false, {}, {}, 0.0L};
  }
  if (halfTwo.failed) {
    return {Status::success(), true, {}, {}, 0.0L};
  }
  Wide error = 0.0L;
  for (Index i = 0U; i < state.size(); ++i) {
    error = std::max(
        error,
        componentError(static_cast<Wide>(halfTwo.state[i]) - full.state[i],
                       state[i], halfTwo.state[i], engine.policy.iteration));
  }
  for (Index i = 0U; i < sensitivity.size(); ++i) {
    error = std::max(error,
                     componentError(static_cast<Wide>(halfTwo.sensitivity[i]) -
                                        full.sensitivity[i],
                                    sensitivity[i], halfTwo.sensitivity[i],
                                    engine.policy.iteration));
  }
  return {Status::success(), error > 1.0L, std::move(halfTwo.state),
          std::move(halfTwo.sensitivity), error};
}

template <class TrajectoryType>
Result<TrajectoryType> failure(const Status& status) {
  return Result<TrajectoryType>::failure(status.code(),
                                         std::string(status.message()));
}

struct IntegrationOutput final {
  Trajectory trajectory;
  std::optional<Tensor> sensitivities;
};

Result<IntegrationOutput> integrate(const OdeFunction& function,
                                    const OdeJacobians* jacobians,
                                    Vector initial, Vector parameters,
                                    Vector times, const OdePolicy& policy,
                                    Limits limits, bool withSensitivity) {
  const Extent q = withSensitivity ? parameters.size() : 0U;
  const Status validation = validateCommon(function, initial, parameters, times,
                                           policy, limits, withSensitivity, q);
  if (!validation.ok()) {
    return failure<IntegrationOutput>(validation);
  }
  if (withSensitivity && q > 0U &&
      (!jacobians || !jacobians->state || !jacobians->parameters)) {
    return Result<IntegrationOutput>::failure(
        ErrorCode::CallbackFailure,
        "positive ODE sensitivity requires both Jacobian callbacks");
  }

  const Index n = initial.size();
  const Index maxEvaluations =
      std::min(policy.iteration.max_evaluations, limits.evaluations);
  Engine engine{function, jacobians, parameters, policy, n, q, maxEvaluations};
  std::vector<Scalar> state(n);
  for (Index i = 0U; i < n; ++i) {
    state[i] = initial.at(i);
  }
  std::vector<Scalar> sensitivity(n * q, 0.0);
  if (withSensitivity && policy.initial_sensitivity.has_value()) {
    for (Index i = 0U; i < n; ++i) {
      for (Index j = 0U; j < q; ++j) {
        const Scalar entry = policy.initial_sensitivity->at(i, j);
        if (!std::isfinite(entry)) {
          return Result<IntegrationOutput>::failure(
              ErrorCode::NonFiniteInput,
              "initial ODE sensitivity contains non-finite data");
        }
        sensitivity[i * q + j] = entry;
      }
    }
  }
  std::vector<Scalar> stateValues;
  std::vector<Scalar> sensitivityValues;
  stateValues.reserve(times.size() * n);
  sensitivityValues.reserve(times.size() * n * q);
  for (Index i = 0U; i < n; ++i) {
    stateValues.push_back(state[i]);
  }
  sensitivityValues.insert(sensitivityValues.end(), sensitivity.begin(),
                           sensitivity.end());
  Index completedTimes = 1U;
  Index accepted = 0U;
  Index rejected = 0U;
  Index attempts = 0U;
  Scalar time = times.at(0U);
  Scalar proposedStep = policy.initial_step;
  Termination termination = Termination::converged;

  for (Index requested = 1U; requested < times.size(); ++requested) {
    const Scalar target = times.at(requested);
    while (time < target) {
      if (attempts >= policy.iteration.max_iterations) {
        termination = Termination::cap;
        goto finish;
      }
      const Wide remaining = static_cast<Wide>(target) - time;
      const Scalar step = static_cast<Scalar>(
          std::min({static_cast<Wide>(proposedStep),
                    static_cast<Wide>(policy.max_step), remaining}));
      if (!(step > 0.0) || static_cast<Scalar>(time + step) == time) {
        return Result<IntegrationOutput>::failure(
            ErrorCode::NoConvergence,
            "ODE step cannot advance the representable clock");
      }
      const bool finalRemainder = static_cast<Wide>(step) == remaining;
      if (step < policy.min_step && !finalRemainder) {
        return Result<IntegrationOutput>::failure(
            ErrorCode::NoConvergence,
            "ODE subdivision would fall below minimum step");
      }
      ++attempts;
      Step trial =
          policy.method == OdeMethod::dormand_prince_54
              ? dopriStep(engine, time, step, state, sensitivity, q)
              : backwardStep(engine, time, step, state, sensitivity, q);
      if (!trial.status.ok()) {
        return failure<IntegrationOutput>(trial.status);
      }
      const bool rejectedTrial = trial.rejected || trial.error > 1.0L;
      const Index exponent =
          policy.method == OdeMethod::dormand_prince_54 ? 5U : 2U;
      Wide multiplier = trial.error == 0.0L
                            ? 5.0L
                            : 0.9L * std::pow(trial.error, -1.0L / exponent);
      multiplier = std::clamp(multiplier, 0.2L, 5.0L);
      if (rejectedTrial) {
        multiplier = std::min(multiplier, 0.9L);
        ++rejected;
        if (step <= policy.min_step ||
            (finalRemainder && step < policy.min_step)) {
          return Result<IntegrationOutput>::failure(
              ErrorCode::NoConvergence,
              "ODE rejected a trial at its minimum step");
        }
      } else {
        state = std::move(trial.state);
        sensitivity = std::move(trial.sensitivity);
        time = finalRemainder ? target : static_cast<Scalar>(time + step);
        ++accepted;
        if (time == target) {
          for (Scalar value : state) {
            stateValues.push_back(value);
          }
          sensitivityValues.insert(sensitivityValues.end(), sensitivity.begin(),
                                   sensitivity.end());
          ++completedTimes;
        }
      }
      proposedStep = std::clamp(static_cast<Scalar>(step * multiplier),
                                policy.min_step, policy.max_step);
    }
  }

finish:
  try {
    Vector completed(completedTimes);
    Matrix states(completedTimes, n, 0.0);
    for (Index row = 0U; row < completedTimes; ++row) {
      completed.at(row) = times.at(row);
      for (Index column = 0U; column < n; ++column) {
        states.at(row, column) = stateValues[row * n + column];
      }
    }
    if (!withSensitivity) {
      return Result<IntegrationOutput>::success(
          {{std::move(completed), std::move(states), accepted, rejected,
            attempts, termination},
           std::nullopt});
    }
    const std::array<Extent, 3> shapeExtents{completedTimes, n, q};
    auto shape = TensorShape::create(shapeExtents);
    if (!shape) {
      return Result<IntegrationOutput>::failure(
          shape.status().code(), std::string(shape.status().message()));
    }
    auto tensor =
        Tensor::create(shape.value(), 0.0, TensorLimits(limits.elements));
    if (!tensor) {
      return Result<IntegrationOutput>::failure(
          tensor.status().code(), std::string(tensor.status().message()));
    }
    for (Index row = 0U; row < completedTimes; ++row) {
      for (Index i = 0U; i < n; ++i) {
        for (Index j = 0U; j < q; ++j) {
          const std::array<Index, 3> index{row, i, j};
          tensor.value().at(index) = sensitivityValues[row * n * q + i * q + j];
        }
      }
    }
    return Result<IntegrationOutput>::success(
        {{std::move(completed), std::move(states), accepted, rejected, attempts,
          termination},
         std::move(tensor).value()});
  } catch (const std::bad_alloc&) {
    return Result<IntegrationOutput>::failure(ErrorCode::ResourceLimit,
                                              "ODE result allocation failed");
  } catch (const std::length_error&) {
    return Result<IntegrationOutput>::failure(
        ErrorCode::ResourceLimit, "ODE result extent is unsupported");
  }
}

} // namespace

Result<Trajectory> odeIntegrate(const OdeFunction& function, Vector initial,
                                Vector times, const OdePolicy& policy,
                                Limits limits) {
  try {
    const Vector empty;
    const Vector& parameters =
        policy.parameters.has_value() ? *policy.parameters : empty;
    auto output = integrate(function, nullptr, std::move(initial), parameters,
                            std::move(times), policy, limits, false);
    if (!output) {
      return failure<Trajectory>(output.status());
    }
    return Result<Trajectory>::success(std::move(output).value().trajectory);
  } catch (const std::bad_alloc&) {
    return Result<Trajectory>::failure(ErrorCode::ResourceLimit,
                                       "ODE integration allocation failed");
  } catch (const std::length_error&) {
    return Result<Trajectory>::failure(ErrorCode::ResourceLimit,
                                       "ODE integration extent is unsupported");
  }
}

Result<SensitivityTrajectory>
odeSensitivity(const OdeFunction& function, const OdeJacobians& jacobians,
               Vector initial, Vector parameters, Vector times,
               const OdePolicy& policy, Limits limits) {
  try {
    auto integrated = integrate(function, &jacobians, std::move(initial),
                                std::move(parameters), std::move(times), policy,
                                limits, true);
    if (!integrated) {
      return Result<SensitivityTrajectory>::failure(
          integrated.status().code(),
          std::string(integrated.status().message()));
    }
    auto output = std::move(integrated).value();
    if (!output.sensitivities.has_value()) {
      return Result<SensitivityTrajectory>::failure(
          ErrorCode::InvalidDomain, "ODE sensitivity output is missing");
    }
    auto trajectory = std::move(output.trajectory);
    return Result<SensitivityTrajectory>::success(
        {std::move(trajectory.times), std::move(trajectory.states),
         std::move(*output.sensitivities), trajectory.accepted_steps,
         trajectory.rejected_steps, trajectory.attempts,
         trajectory.termination});
  } catch (const std::bad_alloc&) {
    return Result<SensitivityTrajectory>::failure(
        ErrorCode::ResourceLimit, "ODE sensitivity allocation failed");
  } catch (const std::length_error&) {
    return Result<SensitivityTrajectory>::failure(
        ErrorCode::ResourceLimit, "ODE sensitivity extent is unsupported");
  }
}

} // namespace vss::math
