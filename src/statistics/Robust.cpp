#include <algorithm>
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
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Robust.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

Result<Extent> validate(const SampleView& sample, Limits limits,
                        Extent requiredElements) {
  const Extent n = sample.rows.rows();
  const Extent d = sample.rows.columns();
  if (n == 0U || d == 0U) {
    return fail<Extent>(ErrorCode::InsufficientSamples,
                        "robust estimators require rows and coordinates");
  }
  if (!sample.observation_shape.empty()) {
    Extent product = 1U;
    for (const Extent extent : sample.observation_shape) {
      if (extent == 0U ||
          product > std::numeric_limits<Extent>::max() / extent) {
        return fail<Extent>(ErrorCode::InvalidExtent,
                            "invalid observation shape");
      }
      product *= extent;
    }
    if (product != d) {
      return fail<Extent>(ErrorCode::DimensionMismatch,
                          "observation shape does not match columns");
    }
  }
  if (n > std::numeric_limits<Extent>::max() / d) {
    return fail<Extent>(ErrorCode::SizeOverflow, "sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != n * d) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "validity mask must match all sample cells");
  }
  for (const std::uint8_t byte : sample.valid) {
    if (byte > 1U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "validity mask bytes must be zero or one");
    }
    if (byte == 0U) {
      return fail<Extent>(ErrorCode::InvalidDomain,
                          "robust estimators require complete rows");
    }
  }
  for (Extent i = 0U; i < n; ++i) {
    for (Extent j = 0U; j < d; ++j) {
      if (!std::isfinite(sample.rows.at(i, j))) {
        return fail<Extent>(ErrorCode::NonFiniteInput,
                            "sample contains a non-finite observation");
      }
    }
  }
  if (requiredElements > limits.elements ||
      requiredElements > limits.bytes / sizeof(Scalar)) {
    return fail<Extent>(ErrorCode::ResourceLimit,
                        "robust estimator storage exceeds configured limits");
  }
  return Result<Extent>::success(d);
}

Status validatePolicy(const RobustPolicy& policy, Extent dimension) {
  if (!std::isfinite(policy.huber_cutoff) || policy.huber_cutoff <= 0.0 ||
      !std::isfinite(policy.scale_factor) || policy.scale_factor <= 0.0) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "Huber cutoff and scale factor must be positive and finite");
  }
  if (policy.fixed_scale) {
    if (policy.fixed_scale->size() != dimension) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "fixed robust scale must match sample dimension");
    }
    for (Extent j = 0U; j < dimension; ++j) {
      if (!std::isfinite(policy.fixed_scale->at(j))) {
        return Status::failure(
            ErrorCode::NonFiniteInput,
            "fixed robust scale contains a non-finite value");
      }
    }
  }
  if (policy.metric) {
    if (policy.metric->rows() != dimension ||
        policy.metric->columns() != dimension) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "robust metric must match sample dimension");
    }
    for (Extent i = 0U; i < dimension; ++i) {
      for (Extent j = 0U; j < dimension; ++j) {
        if (!std::isfinite(policy.metric->at(i, j))) {
          return Status::failure(ErrorCode::NonFiniteInput,
                                 "robust metric contains a non-finite value");
        }
      }
    }
  }
  if (policy.center) {
    if (policy.center->size() != dimension) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "robust center must match sample dimension");
    }
    for (Extent j = 0U; j < dimension; ++j) {
      if (!std::isfinite(policy.center->at(j))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "robust center contains a non-finite value");
      }
    }
  }
  return Status::success();
}

Extent saturatingAdd(Extent left, Extent right) noexcept {
  return left > std::numeric_limits<Extent>::max() - right
             ? std::numeric_limits<Extent>::max()
             : left + right;
}

Extent saturatingMultiply(Extent left, Extent right) noexcept {
  return right != 0U && left > std::numeric_limits<Extent>::max() / right
             ? std::numeric_limits<Extent>::max()
             : left * right;
}

Result<Vector> scales(const SampleView& sample, const RobustPolicy& policy,
                      Limits limits) {
  const Extent n = sample.rows.rows();
  const Extent d = sample.rows.columns();
  Vector result(d, 0.0);
  Vector column(n, 0.0);
  Vector deviations(n, 0.0);
  for (Extent j = 0U; j < d; ++j) {
    for (Extent i = 0U; i < n; ++i) {
      column.at(i) = sample.rows.at(i, j);
    }
    auto center = median(column.view(), limits);
    if (!center) {
      return fail<Vector>(center.status().code(),
                          std::string(center.status().message()));
    }
    for (Extent i = 0U; i < n; ++i) {
      const long double difference =
          std::abs(static_cast<long double>(column.at(i)) - center.value());
      if (!std::isfinite(difference) ||
          difference > std::numeric_limits<Scalar>::max()) {
        return fail<Vector>(ErrorCode::NonFiniteResult,
                            "absolute deviation is not representable");
      }
      deviations.at(i) = static_cast<Scalar>(difference);
    }
    auto mad = median(deviations.view(), limits);
    if (!mad) {
      return fail<Vector>(mad.status().code(),
                          std::string(mad.status().message()));
    }
    const long double scaled = static_cast<long double>(policy.scale_factor) *
                               static_cast<long double>(mad.value());
    if (!std::isfinite(scaled) ||
        scaled > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
      return fail<Vector>(ErrorCode::NonFiniteResult,
                          "robust scale is not representable");
    }
    result.at(j) = static_cast<Scalar>(scaled);
  }
  return Result<Vector>::success(std::move(result));
}

Result<EstimatorDiagnostics> diagnostics(Extent n, WeightRule weight,
                                         CenterRule center,
                                         std::string algorithm,
                                         std::string premises) {
  const auto count = static_cast<Scalar>(n);
  const Support support{
      .input = n,
      .valid = n,
      .positive = n,
      .frequency_mass = std::nullopt,
      .weight_mass = count,
      .normalized_weight_powers = {1.0, 1.0 / count, 1.0 / (count * count)},
      .effective_count = count,
      .degrees_of_freedom = std::nullopt,
      .squared_weight_mass = std::nullopt};
  return Result<EstimatorDiagnostics>::success(
      EstimatorDiagnostics{.estimand = Estimand::observed,
                           .weight = weight,
                           .center = center,
                           .approximation = Approximation::exact,
                           .global = support,
                           .coordinate_support = {},
                           .tuple_support = {},
                           .reciprocal_condition = std::nullopt,
                           .numerical_error = std::nullopt,
                           .sampling_error = std::nullopt,
                           .rank = std::nullopt,
                           .algorithm = std::move(algorithm),
                           .premises = std::move(premises),
                           .warnings = {}});
}

long double score(const SampleView& sample, Extent coordinate, Scalar candidate,
                  Scalar scale, Scalar cutoff) {
  long double total = 0.0L;
  const auto s = static_cast<long double>(scale);
  const auto c = static_cast<long double>(cutoff);
  for (Extent i = 0U; i < sample.rows.rows(); ++i) {
    const long double residual =
        static_cast<long double>(sample.rows.at(i, coordinate)) -
        static_cast<long double>(candidate);
    const long double standardized = residual / s;
    total += std::clamp(standardized, -c, c);
  }
  return total / static_cast<long double>(sample.rows.rows());
}

Result<Scalar> huberLocation(const SampleView& sample, Extent coordinate,
                             Scalar scale, const RobustPolicy& policy) {
  Scalar minimum = sample.rows.at(0U, coordinate);
  Scalar maximum = minimum;
  for (Extent i = 1U; i < sample.rows.rows(); ++i) {
    minimum = std::min(minimum, sample.rows.at(i, coordinate));
    maximum = std::max(maximum, sample.rows.at(i, coordinate));
  }
  if (minimum == maximum) {
    return Result<Scalar>::success(minimum);
  }
  if (scale == 0.0) {
    return fail<Scalar>(
        ErrorCode::InvalidDomain,
        "nonconstant coordinate has zero MAD; supply fixed scale");
  }
  const auto& iteration = policy.iteration;
  if (!std::isfinite(iteration.abs) || !std::isfinite(iteration.rel) ||
      iteration.abs < 0.0 || iteration.rel < 0.0 ||
      (iteration.abs == 0.0 && iteration.rel == 0.0) ||
      iteration.max_iterations == 0U) {
    return fail<Scalar>(ErrorCode::InvalidTolerance,
                        "robust location requires positive finite root "
                        "tolerances and an iteration cap");
  }
  const long double absoluteTolerance = iteration.abs;
  const long double relativeTolerance = iteration.rel;
  auto rootBoundary = [&](bool lowerBoundary) -> Result<Scalar> {
    Scalar low = minimum;
    Scalar high = maximum;
    bool converged = false;
    for (Extent step = 0U; step < iteration.max_iterations; ++step) {
      const long double width = static_cast<long double>(high) - low;
      const long double tolerance =
          absoluteTolerance +
          (relativeTolerance *
           std::max(std::abs(static_cast<long double>(low)),
                    std::abs(static_cast<long double>(high))));
      if (width <= tolerance) {
        converged = true;
        break;
      }
      const auto middle =
          static_cast<Scalar>(static_cast<long double>(low) + (width * 0.5L));
      if (middle <= low || middle >= high) {
        converged = true;
        break;
      }
      const long double value =
          score(sample, coordinate, middle, scale, policy.huber_cutoff);
      if (lowerBoundary ? value > 0.0L : value >= 0.0L) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (!converged) {
      const long double width = static_cast<long double>(high) - low;
      const long double tolerance =
          absoluteTolerance +
          (relativeTolerance *
           std::max(std::abs(static_cast<long double>(low)),
                    std::abs(static_cast<long double>(high))));
      converged = width <= tolerance;
    }
    if (!converged) {
      return fail<Scalar>(ErrorCode::NoConvergence,
                          "Huber root search exhausted its iteration limit");
    }
    return Result<Scalar>::success(
        static_cast<Scalar>(static_cast<long double>(low) +
                            ((static_cast<long double>(high) - low) * 0.5L)));
  };
  auto lower = rootBoundary(true);
  if (!lower) {
    return lower;
  }
  auto upper = rootBoundary(false);
  if (!upper) {
    return upper;
  }
  return Result<Scalar>::success(static_cast<Scalar>(
      (static_cast<long double>(lower.value()) + upper.value()) * 0.5L));
}

} // namespace

Result<Estimate<Vector>>
robustScale(const SampleView& x, const RobustPolicy& policy, Limits limits) {
  const Extent n = x.rows.rows();
  const Extent d = x.rows.columns();
  const Extent required =
      saturatingAdd(saturatingMultiply(2U, n), saturatingMultiply(2U, d));
  auto valid = validate(x, limits, required);
  if (!valid) {
    return fail<Estimate<Vector>>(valid.status().code(),
                                  std::string(valid.status().message()));
  }
  auto policyStatus = validatePolicy(policy, valid.value());
  if (!policyStatus.ok()) {
    return fail<Estimate<Vector>>(policyStatus.code(),
                                  std::string(policyStatus.message()));
  }
  auto values = scales(x, policy, limits);
  if (!values) {
    return fail<Estimate<Vector>>(values.status().code(),
                                  std::string(values.status().message()));
  }
  auto meta =
      diagnostics(x.rows.rows(), WeightRule::unweighted, CenterRule::estimated,
                  "coordinatewise median absolute deviation",
                  "descriptive finite sample; no distributional premise");
  if (!meta) {
    return fail<Estimate<Vector>>(meta.status().code(),
                                  std::string(meta.status().message()));
  }
  return Result<Estimate<Vector>>::success(
      Estimate<Vector>{.value = std::move(values).value(),
                       .diagnostics = std::move(meta).value(),
                       .error_covariance = std::nullopt});
}

Result<Estimate<Vector>>
robustLocation(const SampleView& x, const RobustPolicy& policy, Limits limits) {
  const Extent n = x.rows.rows();
  const Extent d = x.rows.columns();
  const Extent required =
      saturatingAdd(saturatingMultiply(2U, n), saturatingMultiply(2U, d));
  auto valid = validate(x, limits, required);
  if (!valid) {
    return fail<Estimate<Vector>>(valid.status().code(),
                                  std::string(valid.status().message()));
  }
  auto policyStatus = validatePolicy(policy, valid.value());
  if (!policyStatus.ok()) {
    return fail<Estimate<Vector>>(policyStatus.code(),
                                  std::string(policyStatus.message()));
  }
  Vector scaleValues;
  if (policy.fixed_scale) {
    if (policy.fixed_scale->size() != valid.value()) {
      return fail<Estimate<Vector>>(
          ErrorCode::DimensionMismatch,
          "fixed robust scale must match sample dimension");
    }
    scaleValues = Vector(valid.value(), 0.0);
    for (Extent j = 0U; j < valid.value(); ++j) {
      const Scalar value = policy.fixed_scale->at(j);
      if (!std::isfinite(value) || value <= 0.0) {
        return fail<Estimate<Vector>>(
            ErrorCode::InvalidDomain,
            "fixed robust scales must be positive and finite");
      }
      scaleValues.at(j) = value;
    }
  } else {
    auto estimated = scales(x, policy, limits);
    if (!estimated) {
      return fail<Estimate<Vector>>(estimated.status().code(),
                                    std::string(estimated.status().message()));
    }
    scaleValues = std::move(estimated).value();
  }
  Vector output(valid.value(), 0.0);
  for (Extent j = 0U; j < valid.value(); ++j) {
    auto estimate = huberLocation(x, j, scaleValues.at(j), policy);
    if (!estimate) {
      return fail<Estimate<Vector>>(estimate.status().code(),
                                    std::string(estimate.status().message()));
    }
    output.at(j) = estimate.value();
  }
  auto meta =
      diagnostics(x.rows.rows(), WeightRule::unweighted, CenterRule::estimated,
                  "coordinatewise Huber M-location",
                  "bounded Huber score with declared scale and root tolerance");
  if (!meta) {
    return fail<Estimate<Vector>>(meta.status().code(),
                                  std::string(meta.status().message()));
  }
  return Result<Estimate<Vector>>::success(
      Estimate<Vector>{.value = std::move(output),
                       .diagnostics = std::move(meta).value(),
                       .error_covariance = std::nullopt});
}

Result<Estimate<Matrix>> robustCovariance(const SampleView& x,
                                          const RobustPolicy& policy,
                                          Limits limits) {
  const Extent n = x.rows.rows();
  const Extent d = x.rows.columns();
  const Extent squares = saturatingMultiply(d, d);
  const Extent duringScale =
      saturatingAdd(saturatingMultiply(2U, n),
                    saturatingAdd(squares, saturatingMultiply(2U, d)));
  const Extent duringScatter =
      saturatingAdd(saturatingMultiply(policy.metric ? 3U : 2U, squares),
                    saturatingMultiply(4U, d));
  const Extent required = std::max(duringScale, duringScatter);
  auto valid = validate(x, limits, required);
  if (!valid) {
    return fail<Estimate<Matrix>>(valid.status().code(),
                                  std::string(valid.status().message()));
  }
  auto policyStatus = validatePolicy(policy, valid.value());
  if (!policyStatus.ok()) {
    return fail<Estimate<Matrix>>(policyStatus.code(),
                                  std::string(policyStatus.message()));
  }
  Vector center(d, 0.0);
  const CenterRule centerRule =
      policy.center ? CenterRule::known : CenterRule::estimated;
  if (policy.center) {
    if (policy.center->size() != d) {
      return fail<Estimate<Matrix>>(
          ErrorCode::DimensionMismatch,
          "robust covariance center must match sample dimension");
    }
    for (Extent j = 0U; j < d; ++j) {
      center.at(j) = policy.center->at(j);
      if (!std::isfinite(center.at(j))) {
        return fail<Estimate<Matrix>>(ErrorCode::NonFiniteInput,
                                      "robust covariance center is non-finite");
      }
    }
  } else {
    auto location = robustLocation(x, policy, limits);
    if (!location) {
      return fail<Estimate<Matrix>>(location.status().code(),
                                    std::string(location.status().message()));
    }
    center = std::move(location).value().value;
  }
  Matrix metric(d, d, 0.0);
  Vector radialScales;
  if (policy.metric) {
    if (policy.metric->rows() != d || policy.metric->columns() != d) {
      return fail<Estimate<Matrix>>(ErrorCode::DimensionMismatch,
                                    "robust covariance metric must be d-by-d");
    }
    Matrix copy(d, d, 0.0);
    for (Extent i = 0U; i < d; ++i) {
      for (Extent j = 0U; j < d; ++j) {
        const Scalar v = policy.metric->at(i, j);
        if (!std::isfinite(v)) {
          return fail<Estimate<Matrix>>(
              ErrorCode::NonFiniteInput,
              "robust covariance metric is non-finite");
        }
        copy.at(i, j) = v;
      }
    }
    auto eig = linear::decomposeSymmetricEigen(copy.view());
    if (!eig) {
      return fail<Estimate<Matrix>>(eig.status().code(),
                                    std::string(eig.status().message()));
    }
    for (Extent i = 0U; i < d; ++i) {
      if (eig.value().eigenvalues().at(i) <= 0.0) {
        return fail<Estimate<Matrix>>(
            ErrorCode::InvalidDomain,
            "robust covariance metric must be positive definite");
      }
    }
    metric = std::move(copy);
  } else {
    Vector scaleValues;
    if (policy.fixed_scale) {
      if (policy.fixed_scale->size() != d) {
        return fail<Estimate<Matrix>>(
            ErrorCode::DimensionMismatch,
            "fixed robust scale must match sample dimension");
      }
      scaleValues = Vector(d, 0.0);
      for (Extent j = 0U; j < d; ++j) {
        scaleValues.at(j) = policy.fixed_scale->at(j);
      }
    } else {
      auto estimated = scales(x, policy, limits);
      if (!estimated) {
        return fail<Estimate<Matrix>>(
            estimated.status().code(),
            std::string(estimated.status().message()));
      }
      scaleValues = std::move(estimated).value();
    }
    for (Extent j = 0U; j < d; ++j) {
      if (!std::isfinite(scaleValues.at(j)) || scaleValues.at(j) < 0.0) {
        return fail<Estimate<Matrix>>(
            ErrorCode::InvalidDomain,
            "robust covariance scales must be finite and nonnegative");
      }
      bool constant = true;
      for (Extent i = 1U; i < x.rows.rows(); ++i) {
        constant = constant && x.rows.at(i, j) == x.rows.at(0U, j);
      }
      if (scaleValues.at(j) == 0.0 && !constant) {
        return fail<Estimate<Matrix>>(
            ErrorCode::InvalidDomain,
            "nonconstant coordinate has zero robust scale");
      }
    }
    radialScales = std::move(scaleValues);
  }
  Matrix output(d, d, 0.0);
  std::vector<long double> residual(d, 0.0L);
  Vector clipped(d, 0.0);
  const long double cutoff = policy.huber_cutoff;
  for (Extent row = 0U; row < x.rows.rows(); ++row) {
    for (Extent j = 0U; j < d; ++j) {
      residual.at(j) = static_cast<long double>(x.rows.at(row, j)) -
                       static_cast<long double>(center.at(j));
    }
    long double squared = 0.0L;
    if (policy.metric) {
      for (Extent i = 0U; i < d; ++i) {
        for (Extent j = 0U; j < d; ++j) {
          squared += residual.at(i) *
                     static_cast<long double>(metric.at(i, j)) * residual.at(j);
        }
      }
    } else {
      for (Extent j = 0U; j < d; ++j) {
        const Scalar scaleValue = radialScales.at(j);
        if (scaleValue > 0.0) {
          const long double standardized = residual.at(j) / scaleValue;
          squared += standardized * standardized;
        }
      }
    }
    if (!std::isfinite(squared) || squared < 0.0L) {
      return fail<Estimate<Matrix>>(
          ErrorCode::NonFiniteResult,
          "robust metric distance is not representable");
    }
    const long double rho = std::sqrt(squared);
    const long double factor =
        rho == 0.0L ? 1.0L : std::min(1.0L, cutoff / rho);
    for (Extent j = 0U; j < d; ++j) {
      const long double value = residual.at(j) * factor;
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<Estimate<Matrix>>(
            ErrorCode::NonFiniteResult,
            "radially clipped residual is not representable");
      }
      clipped.at(j) = static_cast<Scalar>(value);
    }
    for (Extent i = 0U; i < d; ++i) {
      for (Extent j = 0U; j < d; ++j) {
        const long double updated =
            static_cast<long double>(output.at(i, j)) +
            (static_cast<long double>(clipped.at(i)) * clipped.at(j) /
             static_cast<long double>(x.rows.rows()));
        if (!std::isfinite(updated) ||
            std::abs(updated) > std::numeric_limits<Scalar>::max()) {
          return fail<Estimate<Matrix>>(ErrorCode::NonFiniteResult,
                                        "robust scatter is not representable");
        }
        output.at(i, j) = static_cast<Scalar>(updated);
      }
    }
  }
  auto meta = diagnostics(x.rows.rows(), WeightRule::unweighted, centerRule,
                          "radial Huber-winsorized scatter",
                          "empirical scatter about the declared center and "
                          "metric; no consistency correction");
  if (!meta) {
    return fail<Estimate<Matrix>>(meta.status().code(),
                                  std::string(meta.status().message()));
  }
  return Result<Estimate<Matrix>>::success(
      Estimate<Matrix>{.value = std::move(output),
                       .diagnostics = std::move(meta).value(),
                       .error_covariance = std::nullopt});
}

Result<covariance::CovarianceResult>
shrinkCovariance(ConstMatrixView covarianceInput, ConstMatrixView targetInput,
                 Scalar lambda, SolvePolicy policy, Limits limits) {
  if (!std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::InvalidDomain, "shrinkage fraction must lie in [0,1]");
  }
  if (covarianceInput.rows() != covarianceInput.columns() ||
      targetInput.rows() != targetInput.columns() ||
      covarianceInput.rows() != targetInput.rows()) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::DimensionMismatch,
        "shrinkage inputs must be square matrices with matching dimensions");
  }
  if (!std::isfinite(policy.rank_abs) || !std::isfinite(policy.rank_rel) ||
      !std::isfinite(policy.reciprocal_condition_floor) ||
      policy.rank_abs < 0.0 || policy.rank_rel < 0.0 ||
      policy.reciprocal_condition_floor < 0.0) {
    return fail<covariance::CovarianceResult>(ErrorCode::InvalidTolerance,
                                              "invalid covariance rank policy");
  }
  const Extent d = covarianceInput.rows();
  if (d != 0U && d > std::numeric_limits<Extent>::max() / d) {
    return fail<covariance::CovarianceResult>(ErrorCode::SizeOverflow,
                                              "covariance extent overflows");
  }
  const Extent square = d * d;
  if (square > limits.elements || square > limits.bytes / sizeof(Scalar) ||
      square > std::numeric_limits<Extent>::max() / 6U ||
      square * 6U > limits.elements ||
      square * 6U > limits.bytes / sizeof(Scalar)) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::ResourceLimit,
        "shrinkage output and validation workspace exceed configured limits");
  }
  Matrix source(d, d, 0.0);
  Matrix target(d, d, 0.0);
  for (Extent i = 0U; i < d; ++i) {
    for (Extent j = 0U; j < d; ++j) {
      const Scalar a = covarianceInput.at(i, j);
      const Scalar b = targetInput.at(i, j);
      if (!std::isfinite(a) || !std::isfinite(b)) {
        return fail<covariance::CovarianceResult>(
            ErrorCode::NonFiniteInput,
            "shrinkage inputs must contain only finite entries");
      }
      source.at(i, j) = a;
      target.at(i, j) = b;
    }
  }
  if (d == 0U) {
    return Result<covariance::CovarianceResult>::success(
        covariance::CovarianceResult{
            .covariance = Matrix{},
            .classification = covariance::CovarianceKind::semidefinite_singular,
            .rank = 0U,
            .cutoff = 0.0,
            .reciprocal_condition = std::nullopt});
  }
  linear::SymmetricEigenOptions eigenOptions;
  eigenOptions.rankTolerance = {.absolute = policy.rank_abs,
                                .relative = policy.rank_rel};
  if (policy.iteration.max_iterations != 0U) {
    eigenOptions.maximumSweeps = policy.iteration.max_iterations;
  }
  auto sourceEigen =
      linear::decomposeSymmetricEigen(source.view(), eigenOptions);
  if (!sourceEigen) {
    return fail<covariance::CovarianceResult>(
        sourceEigen.status().code(),
        std::string(sourceEigen.status().message()));
  }
  auto targetEigen =
      linear::decomposeSymmetricEigen(target.view(), eigenOptions);
  if (!targetEigen) {
    return fail<covariance::CovarianceResult>(
        targetEigen.status().code(),
        std::string(targetEigen.status().message()));
  }
  if (sourceEigen.value().definiteness() ==
          linear::PositiveDefiniteness::NotPositiveSemidefinite ||
      targetEigen.value().definiteness() ==
          linear::PositiveDefiniteness::NotPositiveSemidefinite) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::NotPositiveSemidefinite,
        "both shrinkage inputs must be positive semidefinite");
  }
  Matrix output(d, d, 0.0);
  if (lambda == 0.0) {
    output = source;
  } else if (lambda == 1.0) {
    output = target;
  } else {
    for (Extent i = 0U; i < d; ++i) {
      for (Extent j = 0U; j < d; ++j) {
        output.at(i, j) = std::lerp(source.at(i, j), target.at(i, j), lambda);
        if (!std::isfinite(output.at(i, j))) {
          return fail<covariance::CovarianceResult>(
              ErrorCode::NonFiniteResult,
              "shrinkage result is not representable");
        }
      }
    }
  }
  auto resultEigen =
      linear::decomposeSymmetricEigen(output.view(), eigenOptions);
  if (!resultEigen) {
    return fail<covariance::CovarianceResult>(
        resultEigen.status().code(),
        std::string(resultEigen.status().message()));
  }
  if (resultEigen.value().definiteness() ==
      linear::PositiveDefiniteness::NotPositiveSemidefinite) {
    return fail<covariance::CovarianceResult>(
        ErrorCode::NotPositiveSemidefinite,
        "convex covariance result failed the declared PSD tolerance");
  }
  const Scalar cutoff = resultEigen.value().rankThreshold();
  const Extent rank = resultEigen.value().rank();
  covariance::CovarianceKind classification =
      rank == d ? covariance::CovarianceKind::positive_definite
                : covariance::CovarianceKind::semidefinite_singular;
  std::optional<Scalar> reciprocalCondition;
  if (rank == d) {
    const Scalar smallest = resultEigen.value().eigenvalues().at(0U);
    const Scalar largest = resultEigen.value().eigenvalues().at(d - 1U);
    const Scalar ratio = smallest / largest;
    reciprocalCondition = ratio;
    if (ratio < policy.reciprocal_condition_floor) {
      classification = covariance::CovarianceKind::ill_conditioned;
    }
  }
  return Result<covariance::CovarianceResult>::success(
      covariance::CovarianceResult{.covariance = std::move(output),
                                   .classification = classification,
                                   .rank = rank,
                                   .cutoff = cutoff,
                                   .reciprocal_condition =
                                       reciprocalCondition});
}

} // namespace vss::math::statistics
