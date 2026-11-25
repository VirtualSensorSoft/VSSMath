#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/RatioEstimate.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool multiply(Index left, Index right, Index& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

bool add(Index left, Index right, Index& result) noexcept {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

struct Dimensions final {
  Index components{};
  Index observationValues{};
  Index joint{};
};

Result<Dimensions> dimensions(const RatioModel& model) {
  if (model.rows == 0U || !model.observations || !model.weights) {
    return failure<Dimensions>(ErrorCode::InvalidDomain,
                               "ratio model needs rows and both callbacks");
  }
  Index components = 1U;
  for (const Index extent : model.observation_shape) {
    if (extent == 0U || !multiply(components, extent, components)) {
      return failure<Dimensions>(ErrorCode::SizeOverflow,
                                 "ratio observation shape is invalid");
    }
  }
  Index observationValues = 0U;
  Index joint = 0U;
  if (!multiply(model.rows, components, observationValues) ||
      !add(observationValues, model.rows, joint) ||
      !add(joint, model.parameter_count, joint)) {
    return failure<Dimensions>(ErrorCode::SizeOverflow,
                               "ratio joint input extent overflows");
  }
  return Result<Dimensions>::success(
      Dimensions{components, observationValues, joint});
}

bool outputFits(Index elements, const Limits& limits) noexcept {
  return elements <= limits.elements &&
         elements <= limits.bytes / sizeof(Scalar);
}

struct RatioEvaluation final {
  Vector value;
  Vector weights;
};

Result<RatioEvaluation> evaluateRatio(const RatioModel& model,
                                      const Dimensions& dims,
                                      ConstVectorView joint,
                                      Scalar denominatorFloor) {
  if (joint.size() != dims.joint) {
    return failure<RatioEvaluation>(ErrorCode::DimensionMismatch,
                                    "ratio joint input has the wrong extent");
  }
  for (Index i = 0U; i < joint.size(); ++i) {
    if (!std::isfinite(joint.at(i))) {
      return failure<RatioEvaluation>(ErrorCode::NonFiniteInput,
                                      "ratio joint input is non-finite");
    }
  }

  Result<Vector> observations = Result<Vector>::failure(
      ErrorCode::CallbackFailure, "ratio observation callback threw");
  Result<Vector> weights = Result<Vector>::failure(
      ErrorCode::CallbackFailure, "ratio weight callback threw");
  try {
    observations = model.observations(joint);
  } catch (...) {
    return failure<RatioEvaluation>(ErrorCode::CallbackFailure,
                                    "ratio observation callback threw");
  }
  if (!observations.hasValue()) {
    return Result<RatioEvaluation>::failure(
        observations.status().code(),
        std::string(observations.status().message()));
  }
  if (observations.value().size() != dims.observationValues) {
    return failure<RatioEvaluation>(ErrorCode::DimensionMismatch,
                                    "ratio observation callback shape changed");
  }
  try {
    weights = model.weights(joint);
  } catch (...) {
    return failure<RatioEvaluation>(ErrorCode::CallbackFailure,
                                    "ratio weight callback threw");
  }
  if (!weights.hasValue()) {
    return Result<RatioEvaluation>::failure(
        weights.status().code(), std::string(weights.status().message()));
  }
  if (weights.value().size() != model.rows) {
    return failure<RatioEvaluation>(ErrorCode::DimensionMismatch,
                                    "ratio weight callback shape changed");
  }

  Scalar maximumWeight = 0.0;
  for (Index i = 0U; i < model.rows; ++i) {
    const Scalar weight = weights.value().at(i);
    if (!std::isfinite(weight)) {
      return failure<RatioEvaluation>(ErrorCode::NonFiniteResult,
                                      "ratio weight callback is non-finite");
    }
    if (weight < 0.0) {
      return failure<RatioEvaluation>(ErrorCode::InvalidDomain,
                                      "ratio weights must be nonnegative");
    }
    maximumWeight = std::max(maximumWeight, weight);
  }
  if (maximumWeight == 0.0) {
    return failure<RatioEvaluation>(ErrorCode::InvalidDomain,
                                    "ratio total weight must be positive");
  }

  long double scaledMass = 0.0L;
  for (Index i = 0U; i < model.rows; ++i) {
    scaledMass += static_cast<long double>(weights.value().at(i)) /
                  static_cast<long double>(maximumWeight);
  }
  const long double mass = static_cast<long double>(maximumWeight) * scaledMass;
  if (!(mass > static_cast<long double>(denominatorFloor))) {
    return failure<RatioEvaluation>(ErrorCode::InvalidDomain,
                                    "ratio denominator does not exceed floor");
  }

  Vector value(dims.components);
  for (Index component = 0U; component < dims.components; ++component) {
    long double numerator = 0.0L;
    for (Index row = 0U; row < model.rows; ++row) {
      const Scalar observation =
          observations.value().at(row * dims.components + component);
      if (!std::isfinite(observation)) {
        return failure<RatioEvaluation>(
            ErrorCode::NonFiniteResult,
            "ratio observation callback is non-finite");
      }
      const long double normalizedWeight =
          static_cast<long double>(weights.value().at(row)) /
          static_cast<long double>(maximumWeight);
      numerator += normalizedWeight * static_cast<long double>(observation);
    }
    const long double quotient = numerator / scaledMass;
    if (!std::isfinite(quotient) ||
        std::abs(quotient) > std::numeric_limits<Scalar>::max()) {
      return failure<RatioEvaluation>(ErrorCode::NonFiniteResult,
                                      "weighted ratio is unrepresentable");
    }
    value.at(component) = static_cast<Scalar>(quotient);
  }
  return Result<RatioEvaluation>::success(
      RatioEvaluation{std::move(value), std::move(weights).value()});
}

bool invalidDraw(ErrorCode code) noexcept {
  return code == ErrorCode::InvalidDomain ||
         code == ErrorCode::NonFiniteInput ||
         code == ErrorCode::NonFiniteResult;
}

Result<Support> ratioSupport(Index rows, Index positive,
                             std::span<const Scalar> weights) {
  Support support{.input = rows,
                  .valid = rows,
                  .positive = positive,
                  .frequency_mass = std::nullopt,
                  .weight_mass = std::nullopt,
                  .normalized_weight_powers = {},
                  .effective_count = std::nullopt,
                  .degrees_of_freedom = std::nullopt,
                  .squared_weight_mass = std::nullopt};
  Scalar maximum = 0.0;
  for (const Scalar weight : weights) {
    maximum = std::max(maximum, weight);
  }
  if (maximum == 0.0) {
    return Result<Support>::success(support);
  }
  long double sum = 0.0L;
  long double squares = 0.0L;
  long double cubes = 0.0L;
  long double fourths = 0.0L;
  for (const Scalar weight : weights) {
    const long double scaled = static_cast<long double>(weight) / maximum;
    sum += scaled;
    const long double square = scaled * scaled;
    squares += square;
    cubes += square * scaled;
    fourths += square * square;
  }
  const long double mass = static_cast<long double>(maximum) * sum;
  const long double squared =
      static_cast<long double>(maximum) * maximum * squares;
  if (mass <= std::numeric_limits<Scalar>::max()) {
    support.weight_mass = static_cast<Scalar>(mass);
  }
  if (squared <= std::numeric_limits<Scalar>::max()) {
    support.squared_weight_mass = static_cast<Scalar>(squared);
  }
  if (sum > 0.0L) {
    const long double sum2 = sum * sum;
    const long double sum3 = sum2 * sum;
    const long double sum4 = sum2 * sum2;
    support.normalized_weight_powers[0] = static_cast<Scalar>(squares / sum2);
    support.normalized_weight_powers[1] = static_cast<Scalar>(cubes / sum3);
    support.normalized_weight_powers[2] = static_cast<Scalar>(fourths / sum4);
  }
  return Result<Support>::success(support);
}

} // namespace

Result<VectorDifferential> weightedRatioDifferential(const SampleView& values,
                                                     ConstVectorView weights,
                                                     Limits limits) {
  const Index rows = values.rows.rows();
  const Index components = values.rows.columns();
  if (weights.size() != rows) {
    return failure<VectorDifferential>(
        ErrorCode::DimensionMismatch,
        "ratio weight count must equal observation row count");
  }
  if (rows == 0U || components == 0U) {
    return failure<VectorDifferential>(ErrorCode::InvalidDomain,
                                       "ratio requires complete nonempty rows");
  }
  if (!values.valid.empty()) {
    Index cells = 0U;
    if (!multiply(rows, components, cells) || values.valid.size() != cells) {
      return failure<VectorDifferential>(
          ErrorCode::DimensionMismatch,
          "ratio validity mask must match the observation shape");
    }
    for (const std::uint8_t valid : values.valid) {
      if (valid > 1U) {
        return failure<VectorDifferential>(
            ErrorCode::InvalidDomain,
            "ratio validity mask entries must be zero or one");
      }
      if (valid == 0U) {
        return failure<VectorDifferential>(
            ErrorCode::InvalidDomain, "ratio requires complete nonempty rows");
      }
    }
  }
  if (!values.observation_shape.empty()) {
    Index product = 1U;
    for (const Index extent : values.observation_shape) {
      if (extent == 0U || !multiply(product, extent, product)) {
        return failure<VectorDifferential>(
            ErrorCode::DimensionMismatch, "ratio observation shape is invalid");
      }
    }
    if (product != components) {
      return failure<VectorDifferential>(ErrorCode::DimensionMismatch,
                                         "ratio shape does not match columns");
    }
  }
  Index observationValues = 0U;
  Index input = 0U;
  Index jacobianElements = 0U;
  Index square = 0U;
  Index hessianElements = 0U;
  Index total = 0U;
  if (!multiply(rows, components, observationValues) ||
      !add(observationValues, rows, input) ||
      !multiply(components, input, jacobianElements) ||
      !multiply(input, input, square) ||
      !multiply(components, square, hessianElements) ||
      !add(components, jacobianElements, total) ||
      !add(total, hessianElements, total)) {
    return failure<VectorDifferential>(ErrorCode::SizeOverflow,
                                       "ratio derivative extent overflows");
  }
  if (!outputFits(total, limits)) {
    return failure<VectorDifferential>(ErrorCode::ResourceLimit,
                                       "ratio derivatives exceed limits");
  }
  Scalar maximumWeight = 0.0;
  for (Index row = 0U; row < rows; ++row) {
    const Scalar weight = weights.at(row);
    if (!std::isfinite(weight)) {
      return failure<VectorDifferential>(ErrorCode::NonFiniteInput,
                                         "ratio weight is non-finite");
    }
    if (weight < 0.0) {
      return failure<VectorDifferential>(ErrorCode::InvalidDomain,
                                         "ratio weight is negative");
    }
    maximumWeight = std::max(maximumWeight, weight);
  }
  if (maximumWeight == 0.0) {
    return failure<VectorDifferential>(ErrorCode::InvalidDomain,
                                       "ratio total weight must be positive");
  }
  long double scaledMass = 0.0L;
  for (Index row = 0U; row < rows; ++row) {
    scaledMass += static_cast<long double>(weights.at(row)) / maximumWeight;
  }

  Vector mean(components);
  for (Index component = 0U; component < components; ++component) {
    long double numerator = 0.0L;
    for (Index row = 0U; row < rows; ++row) {
      const Scalar x = values.rows.at(row, component);
      if (!std::isfinite(x)) {
        return failure<VectorDifferential>(ErrorCode::NonFiniteInput,
                                           "ratio observation is non-finite");
      }
      numerator +=
          (static_cast<long double>(weights.at(row)) / maximumWeight) * x;
    }
    const long double result = numerator / scaledMass;
    if (!std::isfinite(result) ||
        std::abs(result) > std::numeric_limits<Scalar>::max()) {
      return failure<VectorDifferential>(ErrorCode::NonFiniteResult,
                                         "weighted mean is unrepresentable");
    }
    mean.at(component) = static_cast<Scalar>(result);
  }

  Matrix jacobian(components, input);
  for (Index component = 0U; component < components; ++component) {
    for (Index row = 0U; row < rows; ++row) {
      const long double normalizedWeight =
          static_cast<long double>(weights.at(row)) / maximumWeight;
      jacobian.at(component, row * components + component) =
          static_cast<Scalar>(normalizedWeight / scaledMass);
      const long double derivative =
          (static_cast<long double>(values.rows.at(row, component)) -
           mean.at(component)) /
          static_cast<long double>(maximumWeight) / scaledMass;
      if (!std::isfinite(derivative) ||
          std::abs(derivative) > std::numeric_limits<Scalar>::max()) {
        return failure<VectorDifferential>(ErrorCode::NonFiniteResult,
                                           "ratio Jacobian is unrepresentable");
      }
      jacobian.at(component, observationValues + row) =
          static_cast<Scalar>(derivative);
    }
  }

  const std::array<Extent, 3U> extents{components, input, input};
  auto shape = TensorShape::create(extents);
  if (!shape.hasValue()) {
    return Result<VectorDifferential>::failure(
        shape.status().code(), std::string(shape.status().message()));
  }
  const Extent tensorElements = shape.value().elementCount();
  auto tensor =
      Tensor::create(shape.value(), 0.0,
                     TensorLimits(std::min<Extent>(
                         limits.elements, limits.bytes / sizeof(Scalar))));
  if (!tensor.hasValue()) {
    return Result<VectorDifferential>::failure(
        tensor.status().code(), std::string(tensor.status().message()));
  }
  Tensor hessian = std::move(tensor).value();
  if (hessian.size() != tensorElements) {
    return failure<VectorDifferential>(ErrorCode::InvalidExtent,
                                       "ratio Hessian storage has wrong size");
  }
  for (Index component = 0U; component < components; ++component) {
    for (Index row = 0U; row < rows; ++row) {
      const long double normalizedWeight =
          static_cast<long double>(weights.at(row)) / maximumWeight;
      for (Index other = 0U; other < rows; ++other) {
        const long double mixed =
            ((row == other ? 1.0L : 0.0L) - normalizedWeight / scaledMass) /
            static_cast<long double>(maximumWeight) / scaledMass;
        if (!std::isfinite(mixed) ||
            std::abs(mixed) > std::numeric_limits<Scalar>::max()) {
          return failure<VectorDifferential>(
              ErrorCode::NonFiniteResult, "ratio Hessian is unrepresentable");
        }
        const Index xIndex = row * components + component;
        const Index otherWeight = observationValues + other;
        const std::array<Index, 3U> xw{component, xIndex, otherWeight};
        const std::array<Index, 3U> wx{component, otherWeight, xIndex};
        hessian.at(xw) = static_cast<Scalar>(mixed);
        hessian.at(wx) = static_cast<Scalar>(mixed);
        const long double weightWeight =
            (2.0L * mean.at(component) - values.rows.at(row, component) -
             values.rows.at(other, component)) /
            static_cast<long double>(maximumWeight) /
            static_cast<long double>(maximumWeight) / scaledMass / scaledMass;
        const std::array<Index, 3U> ww{component, observationValues + row,
                                       observationValues + other};
        if (!std::isfinite(weightWeight) ||
            std::abs(weightWeight) > std::numeric_limits<Scalar>::max()) {
          return failure<VectorDifferential>(
              ErrorCode::NonFiniteResult, "ratio Hessian is unrepresentable");
        }
        hessian.at(ww) = static_cast<Scalar>(weightWeight);
      }
    }
  }
  return Result<VectorDifferential>::success(VectorDifferential{
      std::move(mean), std::move(jacobian), std::move(hessian)});
}

Result<PropagationResult>
propagateWeightedRatio(const RatioModel& model,
                       const JointUncertaintyInput& uncertainty,
                       const PropagationPolicy& policy, Limits limits) {
  const auto dimsResult = dimensions(model);
  if (!dimsResult.hasValue()) {
    return Result<PropagationResult>::failure(
        dimsResult.status().code(), std::string(dimsResult.status().message()));
  }
  const Dimensions dims = dimsResult.value();
  if (policy.method != Approximation::first_order ||
      !std::isfinite(policy.denominator_floor) ||
      policy.denominator_floor < 0.0) {
    return failure<PropagationResult>(ErrorCode::InvalidDomain,
                                      "unsupported ratio propagation policy");
  }
  if (uncertainty.center.size() != dims.joint ||
      uncertainty.covariance.rows() != dims.joint ||
      uncertainty.covariance.columns() != dims.joint) {
    return failure<PropagationResult>(
        ErrorCode::DimensionMismatch,
        "joint uncertainty extents do not match ratio model");
  }
  Index jacobianElements = 0U;
  Index covarianceElements = 0U;
  Index jointWorkspace = 0U;
  Index total = 0U;
  if (!multiply(dims.components, dims.joint, jacobianElements) ||
      !multiply(dims.components, dims.components, covarianceElements) ||
      !add(jacobianElements, covarianceElements, total) ||
      !add(total, dims.components, total) || !add(total, dims.joint, total) ||
      !multiply(dims.joint, dims.joint, jointWorkspace) ||
      !add(total, jointWorkspace, total)) {
    return failure<PropagationResult>(
        ErrorCode::SizeOverflow, "ratio propagation output extent overflows");
  }
  if (!outputFits(total, limits)) {
    return failure<PropagationResult>(ErrorCode::ResourceLimit,
                                      "ratio propagation exceeds limits");
  }
  Index evaluations = 0U;
  if (dims.joint > (std::numeric_limits<Index>::max() - 1U) / 2U ||
      !multiply(2U, 2U * dims.joint + 1U, evaluations) ||
      evaluations > limits.evaluations) {
    return failure<PropagationResult>(
        ErrorCode::ResourceLimit, "ratio derivative callbacks exceed limit");
  }
  if (!std::isfinite(policy.solve.rank_abs) ||
      !std::isfinite(policy.solve.rank_rel) || policy.solve.rank_abs < 0.0 ||
      policy.solve.rank_rel < 0.0) {
    return failure<PropagationResult>(
        ErrorCode::InvalidTolerance,
        "ratio covariance rank tolerance is invalid");
  }
  std::vector<differentiation::ParameterStep> steps;
  steps.reserve(dims.joint);
  for (Index i = 0U; i < dims.joint; ++i) {
    const Scalar scale = std::max(1.0, std::abs(uncertainty.center.at(i)));
    steps.emplace_back(scale);
  }
  std::optional<Vector> nominal;
  Vector weightsAtCenter(model.rows);
  std::optional<ErrorCode> callbackStatus;
  const differentiation::VectorFunction function =
      [&](ConstVectorView point) -> Result<Vector> {
    auto evaluated =
        evaluateRatio(model, dims, point, policy.denominator_floor);
    if (!evaluated.hasValue()) {
      callbackStatus = evaluated.status().code();
      return Result<Vector>::failure(evaluated.status().code(),
                                     std::string(evaluated.status().message()));
    }
    if (!nominal.has_value()) {
      nominal = evaluated.value().value;
      weightsAtCenter = evaluated.value().weights;
    }
    return Result<Vector>::success(std::move(evaluated).value().value);
  };
  const auto derivative =
      differentiation::numericalJacobian(function, uncertainty.center, steps);
  if (!derivative.hasValue()) {
    if (callbackStatus.has_value()) {
      return Result<PropagationResult>::failure(
          *callbackStatus, std::string(derivative.status().message()));
    }
    return Result<PropagationResult>::failure(
        derivative.status().code(), std::string(derivative.status().message()));
  }
  if (!nominal.has_value()) {
    return failure<PropagationResult>(ErrorCode::CallbackFailure,
                                      "ratio derivative omitted base value");
  }
  covariance::CovarianceValidationOptions covarianceOptions{};
  covarianceOptions.eigen.rankTolerance.absolute = policy.solve.rank_abs;
  covarianceOptions.eigen.rankTolerance.relative = policy.solve.rank_rel;
  if (policy.solve.iteration.max_iterations > 0U) {
    if (policy.solve.iteration.max_iterations > limits.evaluations) {
      return failure<PropagationResult>(
          ErrorCode::ResourceLimit,
          "ratio covariance eigensolver exceeds the iteration limit");
    }
    covarianceOptions.eigen.maximumSweeps =
        policy.solve.iteration.max_iterations;
  }
  auto outputCovariance = covariance::propagateCovariance(
      derivative.value().values().view(), uncertainty.covariance,
      covarianceOptions);
  if (!outputCovariance.hasValue()) {
    return Result<PropagationResult>::failure(
        outputCovariance.status().code(),
        std::string(outputCovariance.status().message()));
  }
  Index positive = 0U;
  std::vector<Scalar> weightValues(model.rows);
  for (Index i = 0U; i < model.rows; ++i) {
    weightValues[i] = weightsAtCenter.at(i);
    positive += weightValues[i] > 0.0 ? 1U : 0U;
  }
  auto support = ratioSupport(model.rows, positive, weightValues);
  if (!support.hasValue()) {
    return Result<PropagationResult>::failure(
        support.status().code(), std::string(support.status().message()));
  }
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::estimate,
      .weight = WeightRule::dependent,
      .center = CenterRule::estimated,
      .approximation = Approximation::first_order,
      .global = std::move(support).value(),
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "weighted_ratio_first_order",
      .premises = "joint latent covariance and first-order delta approximation",
      .warnings = {}};
  Vector nominalValue = nominal.value_or(Vector{});
  return Result<PropagationResult>::success(PropagationResult{
      std::move(nominalValue), std::move(outputCovariance).value(),
      Approximation::first_order, std::move(diagnostics),
      PropagationCovarianceCheck::positive_semidefinite});
}

Result<SimulationResult> simulateWeightedRatio(const RatioModel& model,
                                               const JointSampler& sampler,
                                               const SimPolicy& policy,
                                               probability::Rng& rng,
                                               Limits limits) {
  const auto dimsResult = dimensions(model);
  if (!dimsResult.hasValue()) {
    return Result<SimulationResult>::failure(
        dimsResult.status().code(), std::string(dimsResult.status().message()));
  }
  const Dimensions dims = dimsResult.value();
  if (!sampler || policy.draws == 0U ||
      (policy.failure != InvalidDraw::reject &&
       policy.failure != InvalidDraw::report_conditional) ||
      !std::isfinite(policy.denominator_floor) ||
      policy.denominator_floor < 0.0 || policy.batches > policy.draws ||
      (policy.batches > 1U && policy.draws % policy.batches != 0U)) {
    return failure<SimulationResult>(ErrorCode::InvalidDomain,
                                     "invalid ratio simulation policy");
  }
  for (const Scalar probability : policy.quantiles) {
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
      return failure<SimulationResult>(ErrorCode::InvalidDomain,
                                       "ratio quantile is outside [0,1]");
    }
  }
  Index callbackCount = 0U;
  Index attemptValues = 0U;
  Index retainedValues = 0U;
  Index quantileValues = 0U;
  Index covarianceValues = 0U;
  Index batchValues = 0U;
  Index total = 0U;
  if (!multiply(policy.draws, 3U, callbackCount) ||
      callbackCount > limits.evaluations ||
      !multiply(policy.draws, dims.components, attemptValues) ||
      !multiply(attemptValues, 2U, retainedValues) ||
      !multiply(dims.components, policy.quantiles.size(), quantileValues) ||
      !multiply(dims.components, dims.components, covarianceValues) ||
      !multiply(policy.batches, dims.components, batchValues) ||
      !add(retainedValues, quantileValues, total) ||
      !add(total, attemptValues, total) ||
      !add(total, covarianceValues, total) || !add(total, batchValues, total) ||
      !add(total, dims.components, total) || !outputFits(total, limits)) {
    return failure<SimulationResult>(
        ErrorCode::ResourceLimit, "ratio simulation exceeds configured limits");
  }
  if (policy.output_central_moments.has_value()) {
    const auto& request = *policy.output_central_moments;
    if (request.orders == 0U || (request.orders & 0xf0U) != 0U) {
      return failure<SimulationResult>(ErrorCode::InvalidDomain,
                                       "invalid output-moment request");
    }
    Index requested = 0U;
    for (Index order = 1U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      Index count = dims.components;
      if (request.representation == MomentRepresentation::dense) {
        for (Index axis = 1U; axis < order; ++axis) {
          if (!multiply(count, dims.components, count)) {
            return failure<SimulationResult>(
                ErrorCode::SizeOverflow,
                "requested output moments exceed representable dimensions");
          }
        }
      } else if (request.representation == MomentRepresentation::selected) {
        count = request.selected.size();
      } else if (request.representation == MomentRepresentation::packed) {
        count = 1U;
        for (Index i = 1U; i <= order; ++i) {
          const Index factor = dims.components + i - 1U;
          if (!multiply(count, factor, count)) {
            return failure<SimulationResult>(
                ErrorCode::SizeOverflow,
                "requested packed moments exceed representable dimensions");
          }
          count /= i;
        }
      } else if (request.representation == MomentRepresentation::contraction) {
        if (!request.contraction.has_value()) {
          return failure<SimulationResult>(ErrorCode::InvalidDomain,
                                           "moment contraction is absent");
        }
        count = request.contraction->rows();
        for (Index axis = 1U; axis < order; ++axis) {
          if (!multiply(count, request.contraction->rows(), count)) {
            return failure<SimulationResult>(
                ErrorCode::SizeOverflow,
                "contracted moments exceed representable dimensions");
          }
        }
      }
      if (!add(requested, count, requested)) {
        return failure<SimulationResult>(ErrorCode::SizeOverflow,
                                         "requested moment storage overflows");
      }
    }
    Index totalWithMoments = 0U;
    if (!add(requested, total, totalWithMoments) ||
        !outputFits(totalWithMoments, limits)) {
      return failure<SimulationResult>(
          ErrorCode::ResourceLimit, "output moments exceed configured limits");
    }
  }

  std::vector<Scalar> accepted;
  accepted.reserve(attemptValues);
  Matrix batchSums(policy.batches, dims.components);
  std::vector<Index> batchCounts(policy.batches, 0U);
  Index valid = 0U;
  Index rejected = 0U;
  for (Index attempt = 0U; attempt < policy.draws; ++attempt) {
    Result<Vector> joint = Result<Vector>::failure(ErrorCode::CallbackFailure,
                                                   "ratio sampler threw");
    try {
      joint = sampler(rng);
    } catch (...) {
      return failure<SimulationResult>(ErrorCode::CallbackFailure,
                                       "ratio sampler threw");
    }
    if (!joint.hasValue()) {
      return failure<SimulationResult>(ErrorCode::CallbackFailure,
                                       "ratio sampler failed");
    }
    if (joint.value().size() != dims.joint) {
      return failure<SimulationResult>(
          ErrorCode::DimensionMismatch,
          "ratio sampler returned wrong joint extent");
    }
    const auto ratio = evaluateRatio(model, dims, joint.value().view(),
                                     policy.denominator_floor);
    if (!ratio.hasValue()) {
      if (policy.failure == InvalidDraw::report_conditional &&
          invalidDraw(ratio.status().code())) {
        ++rejected;
        continue;
      }
      return Result<SimulationResult>::failure(
          ratio.status().code(), std::string(ratio.status().message()));
    }
    const Index batch =
        policy.batches == 0U ? 0U : attempt / (policy.draws / policy.batches);
    for (Index component = 0U; component < dims.components; ++component) {
      const Scalar value = ratio.value().value.at(component);
      accepted.push_back(value);
      if (policy.batches > 0U) {
        batchSums.at(batch, component) += value;
      }
    }
    if (policy.batches > 0U) {
      ++batchCounts[batch];
    }
    ++valid;
  }
  if (valid == 0U) {
    return failure<SimulationResult>(ErrorCode::InsufficientSamples,
                                     "ratio simulation retained no draws");
  }

  Matrix retained(valid, dims.components);
  for (Index row = 0U; row < valid; ++row) {
    for (Index component = 0U; component < dims.components; ++component) {
      retained.at(row, component) = accepted[row * dims.components + component];
    }
  }
  const auto shape = model.observation_shape;
  const SampleView outputSample{retained.view(), shape, {}};
  const WeightSpec unweighted{.rule = WeightRule::unweighted,
                              .rowWeights = std::nullopt,
                              .componentWeights = std::nullopt,
                              .frequencyCounts = {}};
  auto mean = weightedMean(outputSample, unweighted, limits);
  if (!mean.hasValue()) {
    return Result<SimulationResult>::failure(
        mean.status().code(), std::string(mean.status().message()));
  }
  const CenterSpec center{.rule = CenterRule::estimated,
                          .center = std::nullopt};
  const PopulationPolicy population{.normalization = VarianceMode::empirical,
                                    .missingness = MissingMode::reject,
                                    .estimand = Estimand::estimate};
  auto covariance = covarianceMatrix(outputSample, center, population, limits);
  if (!covariance.hasValue()) {
    return Result<SimulationResult>::failure(
        covariance.status().code(), std::string(covariance.status().message()));
  }
  Matrix quantiles(dims.components,
                   static_cast<Index>(policy.quantiles.size()));
  std::vector<Scalar> coordinate(valid);
  for (Index component = 0U; component < dims.components; ++component) {
    for (Index row = 0U; row < valid; ++row) {
      coordinate[row] = accepted[row * dims.components + component];
    }
    std::sort(coordinate.begin(), coordinate.end());
    for (Index column = 0U; column < policy.quantiles.size(); ++column) {
      const long double position =
          static_cast<long double>(valid - 1U) * policy.quantiles[column];
      const auto lower = static_cast<Index>(std::floor(position));
      const auto upper = static_cast<Index>(std::ceil(position));
      const long double fraction = position - lower;
      const long double value =
          (1.0L - fraction) * coordinate[lower] + fraction * coordinate[upper];
      quantiles.at(component, column) = static_cast<Scalar>(value);
    }
  }
  Vector monteCarloError;
  if (policy.batches >= 2U && rejected == 0U) {
    bool equalCounts = true;
    for (Index b = 0U; b < policy.batches; ++b) {
      equalCounts =
          equalCounts && batchCounts[b] == policy.draws / policy.batches;
    }
    if (equalCounts) {
      monteCarloError = Vector(dims.components);
      for (Index component = 0U; component < dims.components; ++component) {
        long double batchMean = 0.0L;
        for (Index b = 0U; b < policy.batches; ++b) {
          batchMean += batchSums.at(b, component) /
                       static_cast<long double>(batchCounts[b]);
        }
        batchMean /= policy.batches;
        long double sumSquares = 0.0L;
        for (Index b = 0U; b < policy.batches; ++b) {
          const long double difference =
              batchSums.at(b, component) /
                  static_cast<long double>(batchCounts[b]) -
              batchMean;
          sumSquares += difference * difference;
        }
        monteCarloError.at(component) = static_cast<Scalar>(
            std::sqrt(sumSquares / (static_cast<long double>(policy.batches) *
                                    (policy.batches - 1U))));
      }
    }
  }
  std::optional<MomentEstimate> centralMoments;
  if (policy.output_central_moments.has_value()) {
    auto estimate =
        estimateCentralMoments(outputSample, unweighted, center,
                               *policy.output_central_moments, limits);
    if (!estimate.hasValue()) {
      return Result<SimulationResult>::failure(
          estimate.status().code(), std::string(estimate.status().message()));
    }
    centralMoments = std::move(estimate).value();
  }
  const Support simulationSupport{.input = policy.draws,
                                  .valid = valid,
                                  .positive = valid,
                                  .frequency_mass = std::nullopt,
                                  .weight_mass = std::nullopt,
                                  .normalized_weight_powers = {},
                                  .effective_count = std::nullopt,
                                  .degrees_of_freedom = std::nullopt,
                                  .squared_weight_mass = std::nullopt};
  EstimatorDiagnostics diagnostics = mean.value().diagnostics;
  diagnostics.estimand = Estimand::estimate;
  diagnostics.weight = WeightRule::dependent;
  diagnostics.center = CenterRule::estimated;
  diagnostics.approximation = Approximation::monte_carlo;
  diagnostics.global = simulationSupport;
  diagnostics.sampling_error = std::nullopt;
  diagnostics.algorithm = "joint_weighted_ratio_monte_carlo";
  diagnostics.premises =
      rejected == 0U ? "joint sampler empirical output law"
                     : "conditional empirical law of accepted joint draws";
  return Result<SimulationResult>::success(SimulationResult{
      PropagationResult{std::move(mean).value().value,
                        std::move(covariance).value().value,
                        Approximation::monte_carlo, std::move(diagnostics),
                        PropagationCovarianceCheck::positive_semidefinite},
      std::move(quantiles), std::move(monteCarloError), valid, rejected,
      std::move(centralMoments)});
}

} // namespace vss::math::statistics
