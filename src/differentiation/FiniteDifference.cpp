#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>

namespace vss::math::differentiation {

namespace detail {

struct DerivativeCheckAccess final {
  static DerivativeCheckResult
  make(Matrix analyticValues, Matrix numericalValues, Matrix absoluteErrors,
       Matrix relativeErrors, Vector effectiveSteps,
       std::vector<std::uint8_t> passed) noexcept {
    return DerivativeCheckResult(
        std::move(analyticValues), std::move(numericalValues),
        std::move(absoluteErrors), std::move(relativeErrors),
        std::move(effectiveSteps), std::move(passed));
  }
};

} // namespace detail

namespace {

struct PreparedStep final {
  DifferenceFormula formula;
  Scalar effectiveStep;
  Scalar firstPoint;
  Scalar secondPoint;
};

Status invalidDomain(std::string message) {
  return Status::failure(ErrorCode::InvalidDomain, std::move(message));
}

const char* errorCodeName(ErrorCode code) noexcept {
  switch (code) {
  case ErrorCode::None:
    return "None";
  case ErrorCode::InvalidExtent:
    return "InvalidExtent";
  case ErrorCode::SizeOverflow:
    return "SizeOverflow";
  case ErrorCode::IndexOutOfRange:
    return "IndexOutOfRange";
  case ErrorCode::DimensionMismatch:
    return "DimensionMismatch";
  case ErrorCode::NonFiniteInput:
    return "NonFiniteInput";
  case ErrorCode::NonFiniteResult:
    return "NonFiniteResult";
  case ErrorCode::InvalidTolerance:
    return "InvalidTolerance";
  case ErrorCode::Singular:
    return "Singular";
  case ErrorCode::RankDeficient:
    return "RankDeficient";
  case ErrorCode::IllConditioned:
    return "IllConditioned";
  case ErrorCode::NotSymmetric:
    return "NotSymmetric";
  case ErrorCode::NotPositiveDefinite:
    return "NotPositiveDefinite";
  case ErrorCode::NotPositiveSemidefinite:
    return "NotPositiveSemidefinite";
  case ErrorCode::InvalidDomain:
    return "InvalidDomain";
  case ErrorCode::InsufficientSamples:
    return "InsufficientSamples";
  case ErrorCode::CallbackFailure:
    return "CallbackFailure";
  case ErrorCode::NoConvergence:
    return "NoConvergence";
  }
  return "Unknown";
}

template <class Value>
Result<Value> wrapCallbackFailure(const Status& status, std::string context) {
  context += " [";
  context += errorCodeName(status.code());
  context += "]: ";
  context += status.message();
  return Result<Value>::failure(ErrorCode::CallbackFailure, std::move(context));
}

bool inBounds(Scalar value, const ParameterStep& step) noexcept {
  return (!step.lowerBound.has_value() || value >= *step.lowerBound) &&
         (!step.upperBound.has_value() || value <= *step.upperBound);
}

bool checkedPoint(Scalar base, Scalar step, int direction, bool second,
                  const ParameterStep& configuration, Scalar& point) noexcept {
  Scalar distance = step;
  if (second) {
    distance = step + step;
    if (!std::isfinite(distance)) {
      return false;
    }
  }
  point = direction > 0 ? base + distance : base - distance;
  return std::isfinite(point) && point != base &&
         inBounds(point, configuration);
}

bool distinctOneSided(Scalar base, Scalar step, int direction,
                      const ParameterStep& configuration, Scalar& first,
                      Scalar& second) noexcept {
  if (!checkedPoint(base, step, direction, false, configuration, first) ||
      !checkedPoint(base, step, direction, true, configuration, second)) {
    return false;
  }
  return second != first;
}

bool centralCandidate(Scalar base, Scalar step, const ParameterStep& config,
                      Scalar& first, Scalar& second) noexcept {
  if (!checkedPoint(base, step, 1, false, config, first) ||
      !checkedPoint(base, step, -1, false, config, second)) {
    return false;
  }
  return first != second;
}

bool increaseStep(Scalar& step) noexcept {
  const Scalar next =
      std::nextafter(step, std::numeric_limits<Scalar>::infinity());
  if (!std::isfinite(next) || next <= step) {
    return false;
  }
  step = next;
  return true;
}

Scalar adjacentDistance(Scalar base, int direction) noexcept {
  if (direction > 0) {
    return std::nextafter(base, std::numeric_limits<Scalar>::infinity()) - base;
  }
  return base - std::nextafter(base, -std::numeric_limits<Scalar>::infinity());
}

Status prepareStep(Scalar base, const ParameterStep& config,
                   PreparedStep& prepared) {
  const Scalar magnitude = std::max(std::abs(base), config.scale);
  const Scalar proposed = config.explicitStep.value_or(
      std::cbrt(std::numeric_limits<Scalar>::epsilon()) * magnitude);
  if (!std::isfinite(proposed) || proposed < 0.0) {
    return invalidDomain("finite-difference step proposal is not finite");
  }

  const Scalar adjacentUp = adjacentDistance(base, 1);
  const Scalar adjacentDown = adjacentDistance(base, -1);
  const Scalar centralStep = std::max({proposed, adjacentUp, adjacentDown});
  if (std::isfinite(centralStep)) {
    Scalar first = 0.0;
    Scalar second = 0.0;
    Scalar candidate = centralStep;
    while (centralCandidate(base, candidate, config, first, second)) {
      prepared = {.formula = DifferenceFormula::Central,
                  .effectiveStep = candidate,
                  .firstPoint = first,
                  .secondPoint = second};
      return Status::success();
    }
    while (increaseStep(candidate)) {
      const Scalar plus = base + candidate;
      const Scalar minus = base - candidate;
      if (!std::isfinite(plus) || !std::isfinite(minus) ||
          !inBounds(plus, config) || !inBounds(minus, config)) {
        break;
      }
      if (centralCandidate(base, candidate, config, first, second)) {
        prepared = {.formula = DifferenceFormula::Central,
                    .effectiveStep = candidate,
                    .firstPoint = first,
                    .secondPoint = second};
        return Status::success();
      }
    }
  }

  for (const int direction : {1, -1}) {
    Scalar candidate = std::max(proposed, adjacentDistance(base, direction));
    if (!std::isfinite(candidate)) {
      continue;
    }
    Scalar first = 0.0;
    Scalar second = 0.0;
    while (
        distinctOneSided(base, candidate, direction, config, first, second)) {
      prepared = {.formula = direction > 0 ? DifferenceFormula::Forward
                                           : DifferenceFormula::Backward,
                  .effectiveStep = candidate,
                  .firstPoint = first,
                  .secondPoint = second};
      return Status::success();
    }
    while (increaseStep(candidate)) {
      const Scalar firstPoint =
          direction > 0 ? base + candidate : base - candidate;
      const Scalar secondPoint = direction > 0 ? base + candidate + candidate
                                               : base - candidate - candidate;
      if (!std::isfinite(firstPoint) || !std::isfinite(secondPoint) ||
          !inBounds(firstPoint, config) || !inBounds(secondPoint, config)) {
        break;
      }
      if (distinctOneSided(base, candidate, direction, config, first, second)) {
        prepared = {.formula = direction > 0 ? DifferenceFormula::Forward
                                             : DifferenceFormula::Backward,
                    .effectiveStep = candidate,
                    .firstPoint = first,
                    .secondPoint = second};
        return Status::success();
      }
    }
  }
  return invalidDomain(
      "no representable finite-difference stencil fits bounds");
}

Status validateConfiguration(ConstVectorView parameters,
                             std::span<const ParameterStep> steps) {
  if (parameters.empty()) {
    return Status::failure(
        ErrorCode::InvalidExtent,
        "finite-difference parameter extent must be positive");
  }
  if (steps.size() != parameters.size()) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "finite-difference step count does not match parameters");
  }
  for (Index index = 0U; index < parameters.size(); ++index) {
    if (!std::isfinite(parameters.at(index))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "finite-difference parameter is non-finite");
    }
    const ParameterStep& step = steps[index];
    if (!std::isfinite(step.scale) || step.scale <= 0.0) {
      return invalidDomain(
          "finite-difference parameter scale must be finite and positive");
    }
    if (step.explicitStep.has_value() &&
        (!std::isfinite(*step.explicitStep) || *step.explicitStep <= 0.0)) {
      return invalidDomain(
          "finite-difference explicit step must be finite and positive");
    }
    if (step.lowerBound.has_value() && !std::isfinite(*step.lowerBound)) {
      return invalidDomain("finite-difference lower bound must be finite");
    }
    if (step.upperBound.has_value() && !std::isfinite(*step.upperBound)) {
      return invalidDomain("finite-difference upper bound must be finite");
    }
    if (step.lowerBound.has_value() && step.upperBound.has_value() &&
        *step.lowerBound > *step.upperBound) {
      return invalidDomain("finite-difference bounds are reversed");
    }
    if (!inBounds(parameters.at(index), step)) {
      return invalidDomain(
          "finite-difference base parameter is outside bounds");
    }
  }
  return Status::success();
}

bool checkedAdd(Scalar lhs, Scalar rhs, Scalar& result) noexcept {
  result = lhs + rhs;
  return std::isfinite(result);
}

bool checkedSubtract(Scalar lhs, Scalar rhs, Scalar& result) noexcept {
  result = lhs - rhs;
  return std::isfinite(result);
}

bool checkedMultiply(Scalar lhs, Scalar rhs, Scalar& result) noexcept {
  result = lhs * rhs;
  return std::isfinite(result);
}

bool checkedDivide(Scalar lhs, Scalar rhs, Scalar& result) noexcept {
  result = lhs / rhs;
  return std::isfinite(result);
}

bool derivative(DifferenceFormula formula, Scalar base, Scalar first,
                Scalar second, Scalar step, Scalar& result) noexcept {
  Scalar numerator = 0.0;
  Scalar denominator = 0.0;
  if (!checkedMultiply(2.0, step, denominator) || denominator == 0.0) {
    return false;
  }
  switch (formula) {
  case DifferenceFormula::Central:
    if (!checkedSubtract(first, second, numerator)) {
      return false;
    }
    break;
  case DifferenceFormula::Forward: {
    Scalar term = 0.0;
    if (!checkedMultiply(-3.0, base, term) ||
        !checkedMultiply(4.0, first, numerator) ||
        !checkedAdd(term, numerator, term) ||
        !checkedSubtract(term, second, numerator)) {
      return false;
    }
    break;
  }
  case DifferenceFormula::Backward: {
    Scalar term = 0.0;
    if (!checkedMultiply(3.0, base, term) ||
        !checkedMultiply(-4.0, first, numerator) ||
        !checkedAdd(term, numerator, term) ||
        !checkedAdd(term, second, numerator)) {
      return false;
    }
    break;
  }
  }
  return checkedDivide(numerator, denominator, result);
}

Vector copyVector(ConstVectorView input) {
  Vector result(input.size());
  for (Index index = 0U; index < input.size(); ++index) {
    result.at(index) = input.at(index);
  }
  return result;
}

bool finiteVector(ConstVectorView values) noexcept {
  for (Index index = 0U; index < values.size(); ++index) {
    if (!std::isfinite(values.at(index))) {
      return false;
    }
  }
  return true;
}

bool finiteMatrix(ConstMatrixView values) noexcept {
  for (Index row = 0U; row < values.rows(); ++row) {
    for (Index column = 0U; column < values.columns(); ++column) {
      if (!std::isfinite(values.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool everyPassed(const std::vector<std::uint8_t>& values) noexcept {
  return std::ranges::all_of(values,
                             [](std::uint8_t value) { return value != 0U; });
}

Result<DerivativeCheckResult> makeDerivativeCheck(ConstMatrixView analytic,
                                                  ConstMatrixView numerical,
                                                  const Vector& effectiveSteps,
                                                  Tolerance tolerance) {
  const Status toleranceStatus = validate(tolerance);
  if (!toleranceStatus.ok()) {
    return Result<DerivativeCheckResult>::failure(
        toleranceStatus.code(), std::string(toleranceStatus.message()));
  }
  if (analytic.rows() == 0U || analytic.columns() == 0U ||
      analytic.rows() != numerical.rows() ||
      analytic.columns() != numerical.columns()) {
    const ErrorCode code = analytic.rows() == numerical.rows() &&
                                   analytic.columns() == numerical.columns()
                               ? ErrorCode::InvalidExtent
                               : ErrorCode::DimensionMismatch;
    return Result<DerivativeCheckResult>::failure(
        code, "derivative comparison shapes are invalid");
  }
  if (!finiteMatrix(analytic)) {
    return Result<DerivativeCheckResult>::failure(
        ErrorCode::NonFiniteInput,
        "analytic derivative contains non-finite values");
  }
  if (!finiteMatrix(numerical)) {
    return Result<DerivativeCheckResult>::failure(
        ErrorCode::NonFiniteResult,
        "numerical derivative contains non-finite values");
  }

  Matrix absoluteErrors(analytic.rows(), analytic.columns());
  Matrix relativeErrors(analytic.rows(), analytic.columns());
  std::vector<std::uint8_t> passed;
  passed.reserve(analytic.rows() * analytic.columns());
  const Scalar smallestNormal = std::numeric_limits<Scalar>::min();
  for (Index row = 0U; row < analytic.rows(); ++row) {
    for (Index column = 0U; column < analytic.columns(); ++column) {
      const Scalar analyticValue = analytic.at(row, column);
      const Scalar numericalValue = numerical.at(row, column);
      const Scalar absoluteError = std::abs(analyticValue - numericalValue);
      const Scalar denominator = std::max(
          {std::abs(analyticValue), std::abs(numericalValue), smallestNormal});
      const Scalar relativeError = absoluteError / denominator;
      const Scalar maximumMagnitude =
          std::max(std::abs(analyticValue), std::abs(numericalValue));
      const Scalar threshold =
          tolerance.absolute + (tolerance.relative * maximumMagnitude);
      if (!std::isfinite(absoluteError) || !std::isfinite(relativeError) ||
          !std::isfinite(threshold)) {
        return Result<DerivativeCheckResult>::failure(
            ErrorCode::NonFiniteResult,
            "derivative comparison produced a non-finite error");
      }
      absoluteErrors.at(row, column) = absoluteError;
      relativeErrors.at(row, column) = relativeError;
      const bool entryPassed = absoluteError <= threshold;
      passed.push_back(static_cast<std::uint8_t>(entryPassed));
    }
  }
  Matrix analyticCopy(analytic.rows(), analytic.columns());
  Matrix numericalCopy(numerical.rows(), numerical.columns());
  for (Index row = 0U; row < analytic.rows(); ++row) {
    for (Index column = 0U; column < analytic.columns(); ++column) {
      analyticCopy.at(row, column) = analytic.at(row, column);
      numericalCopy.at(row, column) = numerical.at(row, column);
    }
  }
  return Result<DerivativeCheckResult>::success(
      detail::DerivativeCheckAccess::make(
          std::move(analyticCopy), std::move(numericalCopy),
          std::move(absoluteErrors), std::move(relativeErrors), effectiveSteps,
          std::move(passed)));
}

} // namespace

ParameterStep::ParameterStep(Scalar parameterScale) noexcept
    : scale(parameterScale) {}

GradientResult::GradientResult(Vector values, Vector effectiveSteps,
                               std::vector<DifferenceFormula> formulas) noexcept
    : values_(std::move(values)), effectiveSteps_(std::move(effectiveSteps)),
      formulas_(std::move(formulas)) {}

GradientResult::GradientResult(GradientResult&&) noexcept = default;
GradientResult& GradientResult::operator=(GradientResult&&) noexcept = default;

JacobianResult::JacobianResult(Matrix values, Vector effectiveSteps,
                               std::vector<DifferenceFormula> formulas) noexcept
    : values_(std::move(values)), effectiveSteps_(std::move(effectiveSteps)),
      formulas_(std::move(formulas)) {}

JacobianResult::JacobianResult(JacobianResult&&) noexcept = default;
JacobianResult& JacobianResult::operator=(JacobianResult&&) noexcept = default;

DerivativeCheckResult::DerivativeCheckResult(
    Matrix analyticValues, Matrix numericalValues, Matrix absoluteErrors,
    Matrix relativeErrors, Vector effectiveSteps,
    std::vector<std::uint8_t> passed) noexcept
    : analyticValues_(std::move(analyticValues)),
      numericalValues_(std::move(numericalValues)),
      absoluteErrors_(std::move(absoluteErrors)),
      relativeErrors_(std::move(relativeErrors)),
      effectiveSteps_(std::move(effectiveSteps)), passed_(std::move(passed)),
      allPassed_(everyPassed(passed_)) {}

DerivativeCheckResult::DerivativeCheckResult(DerivativeCheckResult&&) noexcept =
    default;
DerivativeCheckResult&
DerivativeCheckResult::operator=(DerivativeCheckResult&&) noexcept = default;

bool DerivativeCheckResult::passed(Index row, Index column) const {
  if (row >= rows() || column >= columns()) {
    throw std::out_of_range("derivative check index out of range");
  }
  return passed_.at((row * columns()) + column) != 0U;
}

Result<GradientResult> numericalGradient(const ScalarFunction& function,
                                         ConstVectorView parameters,
                                         std::span<const ParameterStep> steps) {
  const Status validation = validateConfiguration(parameters, steps);
  if (!validation.ok()) {
    return Result<GradientResult>::failure(validation.code(),
                                           std::string(validation.message()));
  }
  if (!function) {
    return Result<GradientResult>::failure(
        ErrorCode::InvalidDomain, "finite-difference callback is empty");
  }
  std::vector<PreparedStep> prepared;
  prepared.reserve(parameters.size());
  for (Index index = 0U; index < parameters.size(); ++index) {
    PreparedStep item{};
    const Status status = prepareStep(parameters.at(index), steps[index], item);
    if (!status.ok()) {
      return Result<GradientResult>::failure(status.code(),
                                             std::string(status.message()));
    }
    prepared.push_back(item);
  }

  const auto baseResult = function(parameters);
  if (!baseResult.hasValue()) {
    return wrapCallbackFailure<GradientResult>(baseResult.status(),
                                               "base callback");
  }
  const Scalar base = baseResult.value();
  if (!std::isfinite(base)) {
    return Result<GradientResult>::failure(
        ErrorCode::NonFiniteResult,
        "base callback returned a non-finite scalar");
  }

  Vector work = copyVector(parameters);
  Vector values(parameters.size());
  Vector effectiveSteps(parameters.size());
  std::vector<DifferenceFormula> formulas;
  formulas.reserve(parameters.size());
  for (Index index = 0U; index < parameters.size(); ++index) {
    const PreparedStep& item = prepared[index];
    work.at(index) = item.firstPoint;
    const auto firstResult = function(work.view());
    if (!firstResult.hasValue()) {
      return wrapCallbackFailure<GradientResult>(
          firstResult.status(),
          "parameter " + std::to_string(index) + " " +
              (item.formula == DifferenceFormula::Backward ? "negative"
                                                           : "positive") +
              " callback");
    }
    const Scalar first = firstResult.value();
    if (!std::isfinite(first)) {
      return Result<GradientResult>::failure(
          ErrorCode::NonFiniteResult,
          "perturbed callback returned a non-finite scalar");
    }
    work.at(index) = item.secondPoint;
    const auto secondResult = function(work.view());
    if (!secondResult.hasValue()) {
      return wrapCallbackFailure<GradientResult>(
          secondResult.status(),
          "parameter " + std::to_string(index) + " " +
              (item.formula == DifferenceFormula::Forward ? "positive"
                                                          : "negative") +
              " callback");
    }
    const Scalar second = secondResult.value();
    if (!std::isfinite(second)) {
      return Result<GradientResult>::failure(
          ErrorCode::NonFiniteResult,
          "perturbed callback returned a non-finite scalar");
    }
    Scalar value = 0.0;
    if (!derivative(item.formula, base, first, second, item.effectiveStep,
                    value)) {
      return Result<GradientResult>::failure(
          ErrorCode::NonFiniteResult,
          "finite-difference scalar arithmetic produced a non-finite result");
    }
    values.at(index) = value;
    effectiveSteps.at(index) = item.effectiveStep;
    formulas.push_back(item.formula);
    work.at(index) = parameters.at(index);
  }
  return Result<GradientResult>::success(GradientResult(
      std::move(values), std::move(effectiveSteps), std::move(formulas)));
}

Result<JacobianResult> numericalJacobian(const VectorFunction& function,
                                         ConstVectorView parameters,
                                         std::span<const ParameterStep> steps) {
  const Status validation = validateConfiguration(parameters, steps);
  if (!validation.ok()) {
    return Result<JacobianResult>::failure(validation.code(),
                                           std::string(validation.message()));
  }
  if (!function) {
    return Result<JacobianResult>::failure(
        ErrorCode::InvalidDomain, "finite-difference callback is empty");
  }
  std::vector<PreparedStep> prepared;
  prepared.reserve(parameters.size());
  for (Index index = 0U; index < parameters.size(); ++index) {
    PreparedStep item{};
    const Status status = prepareStep(parameters.at(index), steps[index], item);
    if (!status.ok()) {
      return Result<JacobianResult>::failure(status.code(),
                                             std::string(status.message()));
    }
    prepared.push_back(item);
  }

  const auto baseResult = function(parameters);
  if (!baseResult.hasValue()) {
    return wrapCallbackFailure<JacobianResult>(baseResult.status(),
                                               "base callback");
  }
  Vector base = baseResult.value();
  if (base.empty()) {
    return Result<JacobianResult>::failure(
        ErrorCode::InvalidExtent, "base callback returned an empty vector");
  }
  if (!finiteVector(base.view())) {
    return Result<JacobianResult>::failure(
        ErrorCode::NonFiniteResult, "base callback returned non-finite values");
  }

  const Extent rows = base.size();
  const Extent columns = parameters.size();
  Matrix values(rows, columns);
  Vector work = copyVector(parameters);
  Vector effectiveSteps(columns);
  std::vector<DifferenceFormula> formulas;
  formulas.reserve(columns);
  for (Index column = 0U; column < columns; ++column) {
    const PreparedStep& item = prepared[column];
    work.at(column) = item.firstPoint;
    const auto firstResult = function(work.view());
    if (!firstResult.hasValue()) {
      return wrapCallbackFailure<JacobianResult>(
          firstResult.status(),
          "parameter " + std::to_string(column) + " " +
              (item.formula == DifferenceFormula::Backward ? "negative"
                                                           : "positive") +
              " callback");
    }
    Vector first = firstResult.value();
    if (first.size() != rows) {
      return Result<JacobianResult>::failure(
          ErrorCode::DimensionMismatch,
          "perturbed callback changed the output extent");
    }
    if (!finiteVector(first.view())) {
      return Result<JacobianResult>::failure(
          ErrorCode::NonFiniteResult,
          "perturbed callback returned non-finite values");
    }

    work.at(column) = item.secondPoint;
    const auto secondResult = function(work.view());
    if (!secondResult.hasValue()) {
      return wrapCallbackFailure<JacobianResult>(
          secondResult.status(),
          "parameter " + std::to_string(column) + " " +
              (item.formula == DifferenceFormula::Forward ? "positive"
                                                          : "negative") +
              " callback");
    }
    Vector second = secondResult.value();
    if (second.size() != rows) {
      return Result<JacobianResult>::failure(
          ErrorCode::DimensionMismatch,
          "perturbed callback changed the output extent");
    }
    if (!finiteVector(second.view())) {
      return Result<JacobianResult>::failure(
          ErrorCode::NonFiniteResult,
          "perturbed callback returned non-finite values");
    }
    for (Index row = 0U; row < rows; ++row) {
      Scalar value = 0.0;
      if (!derivative(item.formula, base.at(row), first.at(row), second.at(row),
                      item.effectiveStep, value)) {
        return Result<JacobianResult>::failure(
            ErrorCode::NonFiniteResult, "finite-difference Jacobian arithmetic "
                                        "produced a non-finite result");
      }
      values.at(row, column) = value;
    }
    effectiveSteps.at(column) = item.effectiveStep;
    formulas.push_back(item.formula);
    work.at(column) = parameters.at(column);
  }
  return Result<JacobianResult>::success(JacobianResult(
      std::move(values), std::move(effectiveSteps), std::move(formulas)));
}

Result<DerivativeCheckResult>
checkGradient(ConstVectorView analyticGradient,
              const GradientResult& numericalGradient,
              Tolerance comparisonTolerance) {
  const Status toleranceStatus = validate(comparisonTolerance);
  if (!toleranceStatus.ok()) {
    return Result<DerivativeCheckResult>::failure(
        toleranceStatus.code(), std::string(toleranceStatus.message()));
  }
  Matrix analytic(1U, analyticGradient.size());
  for (Index column = 0U; column < analyticGradient.size(); ++column) {
    analytic.at(0U, column) = analyticGradient.at(column);
  }
  Matrix numerical(1U, numericalGradient.values().size());
  for (Index column = 0U; column < numericalGradient.values().size();
       ++column) {
    numerical.at(0U, column) = numericalGradient.values().at(column);
  }
  return makeDerivativeCheck(analytic.view(), numerical.view(),
                             numericalGradient.effectiveSteps(),
                             comparisonTolerance);
}

Result<DerivativeCheckResult>
checkJacobian(ConstMatrixView analyticJacobian,
              const JacobianResult& numericalJacobian,
              Tolerance comparisonTolerance) {
  const Status toleranceStatus = validate(comparisonTolerance);
  if (!toleranceStatus.ok()) {
    return Result<DerivativeCheckResult>::failure(
        toleranceStatus.code(), std::string(toleranceStatus.message()));
  }
  return makeDerivativeCheck(
      analyticJacobian, numericalJacobian.values().view(),
      numericalJacobian.effectiveSteps(), comparisonTolerance);
}

} // namespace vss::math::differentiation
