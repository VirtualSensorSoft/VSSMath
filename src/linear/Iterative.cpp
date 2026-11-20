#include "DenseWorkspace.hpp"
#include "PCG.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Structured.hpp>

namespace vss::math::linear {
namespace {

using detail::EvaluationBudget;
using detail::Wide;
using WideVector = std::vector<Wide>;

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

template <class T> Result<T> forward(const Status& status) {
  return fail<T>(status.code(), std::string(status.message()));
}

bool checkedAdd(std::size_t lhs, std::size_t rhs,
                std::size_t& result) noexcept {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(std::size_t lhs, std::size_t rhs,
                     std::size_t& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool addRequirement(std::size_t count, std::size_t size, std::size_t& elements,
                    std::size_t& bytes) noexcept {
  std::size_t next_elements = 0U;
  std::size_t contribution = 0U;
  std::size_t next_bytes = 0U;
  if (!checkedAdd(elements, count, next_elements) ||
      !checkedMultiply(count, size, contribution) ||
      !checkedAdd(bytes, contribution, next_bytes)) {
    return false;
  }
  elements = next_elements;
  bytes = next_bytes;
  return true;
}

bool preflightVectors(Extent n, std::size_t wide_vectors,
                      std::size_t scalar_vectors, std::size_t wide_extra,
                      Limits limits) noexcept {
  std::size_t wide_count = 0U;
  std::size_t scalar_count = 0U;
  std::size_t elements = 0U;
  std::size_t bytes = 0U;
  return checkedMultiply(n, wide_vectors, wide_count) &&
         checkedMultiply(n, scalar_vectors, scalar_count) &&
         addRequirement(wide_count, sizeof(Wide), elements, bytes) &&
         addRequirement(wide_extra, sizeof(Wide), elements, bytes) &&
         addRequirement(scalar_count, sizeof(Scalar), elements, bytes) &&
         elements <= limits.elements && bytes <= limits.bytes;
}

bool preflightGmres(Extent n, Extent restart, Limits limits) noexcept {
  std::size_t restart_plus_one = 0U;
  std::size_t q_count = 0U;
  std::size_t z_count = 0U;
  std::size_t h_count = 0U;
  std::size_t scalar_vectors = 0U;
  std::size_t vector_count = 0U;
  std::size_t rotation_count = 0U;
  std::size_t elements = 0U;
  std::size_t bytes = 0U;
  if (!checkedAdd(restart, 1U, restart_plus_one) ||
      !checkedMultiply(n, restart_plus_one, q_count) ||
      !checkedMultiply(n, restart, z_count) ||
      !checkedMultiply(restart_plus_one, restart, h_count) ||
      !checkedMultiply(n, 2U, scalar_vectors) ||
      !checkedMultiply(n, 6U, vector_count) ||
      !checkedMultiply(restart, 3U, rotation_count)) {
    return false;
  }
  if (!checkedAdd(q_count, z_count, q_count) ||
      !checkedAdd(q_count, vector_count, q_count) ||
      !checkedAdd(q_count, h_count, q_count) ||
      !checkedAdd(q_count, rotation_count, q_count) ||
      !checkedAdd(q_count, restart_plus_one, q_count) ||
      !checkedAdd(scalar_vectors, n, scalar_vectors)) {
    return false;
  }
  return addRequirement(q_count, sizeof(Wide), elements, bytes) &&
         addRequirement(scalar_vectors, sizeof(Scalar), elements, bytes) &&
         elements <= limits.elements && bytes <= limits.bytes;
}

Status validate(const LinearMap& a, ConstVectorView b, ConstVectorView x0,
                const LinearMap* preconditioner, const SolvePolicy& policy,
                bool gmres_solver) {
  const Extent n = b.size();
  if (a.rows != n || a.columns != n || x0.size() != n ||
      (preconditioner != nullptr &&
       (preconditioner->rows != n || preconditioner->columns != n))) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "iterative solver shapes must be square and conforming");
  }
  if (!a.apply || (preconditioner != nullptr && !preconditioner->apply)) {
    return Status::failure(
        ErrorCode::CallbackFailure,
        "iterative solver requires each active apply action");
  }
  if (!std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "iterative tolerances must be finite and nonnegative");
  }
  if (gmres_solver && policy.restart == 0U) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "GMRES restart must be positive");
  }
  for (Index i = 0U; i < n; ++i) {
    if (!std::isfinite(b.at(i)) || !std::isfinite(x0.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "iterative solver input must be finite");
    }
  }
  return Status::success();
}

Status invoke(const LinearMap& map, std::span<const Wide> input,
              std::span<Wide> output) {
  Vector callback_input(input.size());
  for (Index i = 0U; i < input.size(); ++i) {
    if (!detail::representable(input[i])) {
      return Status::failure(ErrorCode::NonFiniteResult,
                             "iterative callback input is not representable");
    }
    callback_input.at(i) = static_cast<Scalar>(input[i]);
  }

  Result<Vector> returned = [&]() -> Result<Vector> {
    try {
      return map.apply(callback_input.view());
    } catch (...) {
      return Result<Vector>::failure(ErrorCode::CallbackFailure,
                                     "iterative callback threw an exception");
    }
  }();
  if (!returned.hasValue()) {
    return Status::failure(returned.status().code(),
                           std::string(returned.status().message()));
  }
  const Vector& values = returned.value();
  if (values.size() != output.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "iterative callback returned the wrong extent");
  }
  for (Index i = 0U; i < output.size(); ++i) {
    const Scalar value = values.at(i);
    if (!std::isfinite(value)) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "iterative callback returned a non-finite value");
    }
    output[i] = static_cast<Wide>(value);
  }
  return Status::success();
}

Status applyAction(const LinearMap& map, std::span<const Wide> input,
                   std::span<Wide> output, EvaluationBudget& evaluations) {
  if (!evaluations.charge()) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "iterative callback evaluation limit reached");
  }
  return invoke(map, input, output);
}

Status preconditionAction(const LinearMap* preconditioner,
                          std::span<const Wide> input, std::span<Wide> output,
                          EvaluationBudget& evaluations) {
  if (preconditioner == nullptr) {
    std::ranges::copy(input, output.begin());
    return Status::success();
  }
  return applyAction(*preconditioner, input, output, evaluations);
}

WideVector toWide(ConstVectorView values) {
  WideVector result(values.size());
  for (Index i = 0U; i < values.size(); ++i) {
    result[i] = static_cast<Wide>(values.at(i));
  }
  return result;
}

Wide rhsNorm(ConstVectorView values) {
  Wide result = 0;
  for (Index i = 0U; i < values.size(); ++i) {
    result = std::hypot(result, static_cast<Wide>(values.at(i)));
  }
  return result;
}

Wide toleranceFor(ConstVectorView b, const SolvePolicy& policy) {
  return static_cast<Wide>(policy.iteration.abs) +
         static_cast<Wide>(policy.iteration.rel) * rhsNorm(b);
}

Status trueResidual(const LinearMap& a, ConstVectorView b,
                    std::span<const Wide> candidate, std::span<Wide> residual,
                    std::span<Wide> image, EvaluationBudget& evaluations,
                    Wide& norm) {
  Status status = applyAction(a, candidate, image, evaluations);
  if (!status.ok()) {
    return status;
  }
  for (Index i = 0U; i < candidate.size(); ++i) {
    residual[i] = static_cast<Wide>(b.at(i)) - image[i];
  }
  norm = detail::scaledNorm(residual);
  if (!detail::representable(norm)) {
    return Status::failure(ErrorCode::NonFiniteResult,
                           "true residual is not representable");
  }
  return Status::success();
}

Result<IterativeSolve> finish(std::span<const Wide> x, Wide residual,
                              Index iterations, Termination termination) {
  if (!detail::representable(residual)) {
    return fail<IterativeSolve>(ErrorCode::NonFiniteResult,
                                "iterative residual is not representable");
  }
  Vector values(x.size());
  for (Index i = 0U; i < x.size(); ++i) {
    if (!detail::representable(x[i])) {
      return fail<IterativeSolve>(ErrorCode::NonFiniteResult,
                                  "iterative solution is not representable");
    }
    values.at(i) = static_cast<Scalar>(x[i]);
  }
  return Result<IterativeSolve>::success(
      {.value = std::move(values),
       .residual_norm = static_cast<Scalar>(residual),
       .iterations = iterations,
       .termination = termination});
}

Index evaluationLimit(const SolvePolicy& policy, Limits limits) {
  const auto maximum =
      static_cast<std::uint64_t>(std::numeric_limits<Index>::max());
  return static_cast<Index>(std::min(
      {policy.iteration.max_evaluations, limits.evaluations, maximum}));
}

Result<IterativeSolve> failure(const Status& status) {
  return forward<IterativeSolve>(status);
}

} // namespace

Result<IterativeSolve> cg(const LinearMap& a, ConstVectorView b,
                          ConstVectorView x0, const LinearMap* preconditioner,
                          SolvePolicy policy, Limits limits) {
  const Status valid = validate(a, b, x0, preconditioner, policy, false);
  if (!valid.ok()) {
    return failure(valid);
  }
  const Extent n = b.size();
  if (!preflightVectors(n, 5U, 2U, 0U, limits)) {
    return fail<IterativeSolve>(ErrorCode::ResourceLimit,
                                "CG workspace exceeds configured limits");
  }
  if (n == 0U) {
    return finish({}, 0.0L, 0U, Termination::converged);
  }

  WideVector x = toWide(x0);
  WideVector residual(n);
  WideVector z(n);
  WideVector direction(n);
  WideVector image(n);
  EvaluationBudget evaluations(evaluationLimit(policy, limits));
  const auto apply = [&a](std::span<const Wide> input, std::span<Wide> output) {
    return invoke(a, input, output);
  };
  const auto precondition = [&preconditioner](std::span<const Wide> input,
                                              std::span<Wide> output) {
    if (preconditioner == nullptr) {
      std::ranges::copy(input, output.begin());
      return Status::success();
    }
    return invoke(*preconditioner, input, output);
  };
  const auto report =
      detail::runPcg([&b](Index i) { return static_cast<Wide>(b.at(i)); },
                     apply, precondition, policy.iteration, evaluations,
                     {.solution = x,
                      .residual = residual,
                      .preconditioned = z,
                      .direction = direction,
                      .image = image},
                     preconditioner != nullptr);
  if (!report.hasValue()) {
    return forward<IterativeSolve>(report.status());
  }
  return finish(x, report.value().residual_norm, report.value().iterations,
                report.value().converged ? Termination::converged
                                         : Termination::cap);
}

Result<IterativeSolve> minres(const LinearMap& a, ConstVectorView b,
                              ConstVectorView x0,
                              const LinearMap* preconditioner,
                              SolvePolicy policy, Limits limits) {
  const Status valid = validate(a, b, x0, preconditioner, policy, false);
  if (!valid.ok()) {
    return failure(valid);
  }
  const Extent n = b.size();
  if (!preflightVectors(n, 12U, 2U, 0U, limits)) {
    return fail<IterativeSolve>(ErrorCode::ResourceLimit,
                                "MINRES workspace exceeds configured limits");
  }
  if (n == 0U) {
    return finish({}, 0.0L, 0U, Termination::converged);
  }
  WideVector x = toWide(x0);
  WideVector residual(n);
  WideVector v_previous(n, 0.0L);
  WideVector v(n);
  WideVector z(n);
  WideVector w(n);
  WideVector preconditioned_w(n);
  WideVector direction_two_back(n, 0.0L);
  WideVector direction_previous(n, 0.0L);
  WideVector direction(n);
  WideVector candidate(n);
  WideVector image(n);
  EvaluationBudget evaluations(evaluationLimit(policy, limits));
  const Wide tolerance = toleranceFor(b, policy);
  Wide residual_norm = 0;
  Status status =
      trueResidual(a, b, x, residual, image, evaluations, residual_norm);
  if (!status.ok()) {
    return failure(status);
  }
  if (residual_norm <= tolerance || n == 0U) {
    return finish(x, residual_norm, 0U, Termination::converged);
  }
  if (policy.iteration.max_iterations == 0U) {
    return finish(x, residual_norm, 0U, Termination::cap);
  }

  status = preconditionAction(preconditioner, residual, preconditioned_w,
                              evaluations);
  if (!status.ok()) {
    return failure(status);
  }
  Wide beta_squared = detail::compensatedDot(residual, preconditioned_w);
  if (!(beta_squared > 0.0L) || !std::isfinite(beta_squared)) {
    return fail<IterativeSolve>(
        beta_squared < 0.0L ? ErrorCode::NotPositiveDefinite
                            : ErrorCode::NoConvergence,
        "MINRES initial preconditioned norm broke down");
  }
  Wide beta = std::sqrt(beta_squared);
  for (Index i = 0U; i < n; ++i) {
    v[i] = residual[i] / beta;
    z[i] = preconditioned_w[i] / beta;
  }

  Wide cosine_previous = 1.0L;
  Wide sine_previous = 0.0L;
  Wide cosine_two_back = 1.0L;
  Wide sine_two_back = 0.0L;
  Wide phi_bar = beta;
  Index iteration = 0U;
  while (iteration < policy.iteration.max_iterations) {
    status = applyAction(a, z, w, evaluations);
    if (!status.ok()) {
      return failure(status);
    }
    if (iteration != 0U) {
      for (Index i = 0U; i < n; ++i) {
        w[i] -= beta * v_previous[i];
      }
    }
    const Wide alpha = detail::compensatedDot(z, w);
    for (Index i = 0U; i < n; ++i) {
      w[i] -= alpha * v[i];
    }

    const Wide w_norm = detail::scaledNorm(w);
    Wide beta_next = 0.0L;
    if (w_norm != 0.0L) {
      status =
          preconditionAction(preconditioner, w, preconditioned_w, evaluations);
      if (!status.ok()) {
        return failure(status);
      }
      beta_squared = detail::compensatedDot(w, preconditioned_w);
      if (beta_squared < 0.0L || !std::isfinite(beta_squared)) {
        return fail<IterativeSolve>(
            ErrorCode::NotPositiveDefinite,
            "MINRES preconditioner has negative energy");
      }
      if (beta_squared > 0.0L) {
        beta_next = std::sqrt(beta_squared);
      }
    }

    Wide epsilon = 0.0L;
    Wide delta = 0.0L;
    Wide rotated_diagonal = alpha;
    if (iteration != 0U) {
      const Wide previous_row = iteration >= 2U ? cosine_two_back * beta : beta;
      delta = cosine_previous * previous_row + sine_previous * alpha;
      rotated_diagonal =
          -sine_previous * previous_row + cosine_previous * alpha;
    }
    if (iteration >= 2U) {
      epsilon = sine_two_back * beta;
    }
    const Wide gamma = std::hypot(rotated_diagonal, beta_next);
    if (gamma == 0.0L || !std::isfinite(gamma)) {
      return fail<IterativeSolve>(
          ErrorCode::NoConvergence,
          "MINRES projected triangular system broke down");
    }
    const Wide cosine = rotated_diagonal / gamma;
    const Wide sine = beta_next / gamma;
    for (Index i = 0U; i < n; ++i) {
      direction[i] = (z[i] - delta * direction_previous[i] -
                      epsilon * direction_two_back[i]) /
                     gamma;
      candidate[i] = x[i] + cosine * phi_bar * direction[i];
    }
    const Wide next_phi_bar = -sine * phi_bar;
    Wide candidate_residual = 0.0L;
    status = trueResidual(a, b, candidate, residual, image, evaluations,
                          candidate_residual);
    if (!status.ok()) {
      return failure(status);
    }
    x.swap(candidate);
    ++iteration;
    residual_norm = candidate_residual;
    if (residual_norm <= tolerance) {
      return finish(x, residual_norm, iteration, Termination::converged);
    }
    if (beta_next == 0.0L) {
      return fail<IterativeSolve>(
          ErrorCode::NoConvergence,
          "MINRES Krylov space closed before convergence");
    }

    direction_two_back.swap(direction_previous);
    direction_previous.swap(direction);
    v_previous.swap(v);
    for (Index i = 0U; i < n; ++i) {
      v[i] = w[i] / beta_next;
      z[i] = preconditioned_w[i] / beta_next;
    }
    cosine_two_back = cosine_previous;
    sine_two_back = sine_previous;
    cosine_previous = cosine;
    sine_previous = sine;
    beta = beta_next;
    phi_bar = next_phi_bar;
  }
  return finish(x, residual_norm, iteration, Termination::cap);
}

Result<IterativeSolve> gmres(const LinearMap& a, ConstVectorView b,
                             ConstVectorView x0,
                             const LinearMap* preconditioner,
                             SolvePolicy policy, Limits limits) {
  const Status valid = validate(a, b, x0, preconditioner, policy, true);
  if (!valid.ok()) {
    return failure(valid);
  }
  const Extent n = b.size();
  if (n == 0U) {
    return finish({}, 0.0L, 0U, Termination::converged);
  }
  const Extent restart = std::min(policy.restart, n);
  if (!preflightGmres(n, restart, limits)) {
    return fail<IterativeSolve>(ErrorCode::ResourceLimit,
                                "GMRES workspace exceeds configured limits");
  }
  const Extent q_size = (restart + 1U) * n;
  const Extent z_size = restart * n;
  const Extent h_size = (restart + 1U) * restart;
  WideVector q(q_size, 0.0L);
  WideVector z(z_size, 0.0L);
  WideVector h(h_size, 0.0L);
  WideVector cosine(restart, 0.0L);
  WideVector sine(restart, 0.0L);
  WideVector g(restart + 1U, 0.0L);
  WideVector x = toWide(x0);
  WideVector cycle_base(n);
  WideVector candidate(n);
  WideVector residual(n);
  WideVector w(n);
  WideVector image(n);
  EvaluationBudget evaluations(evaluationLimit(policy, limits));
  const Wide tolerance = toleranceFor(b, policy);
  Wide residual_norm = 0.0L;
  Status status =
      trueResidual(a, b, x, residual, image, evaluations, residual_norm);
  if (!status.ok()) {
    return failure(status);
  }
  if (residual_norm <= tolerance) {
    return finish(x, residual_norm, 0U, Termination::converged);
  }
  Index iterations = 0U;
  if (policy.iteration.max_iterations == 0U) {
    return finish(x, residual_norm, 0U, Termination::cap);
  }

  while (iterations < policy.iteration.max_iterations) {
    std::ranges::copy(x, cycle_base.begin());
    const Wide cycle_residual = detail::scaledNorm(residual);
    if (cycle_residual == 0.0L) {
      return fail<IterativeSolve>(
          ErrorCode::NoConvergence,
          "GMRES residual vanished without convergence");
    }
    for (Index i = 0U; i < n; ++i) {
      q[i] = residual[i] / cycle_residual;
    }
    std::ranges::fill(h, 0.0L);
    std::ranges::fill(g, 0.0L);
    g[0U] = cycle_residual;
    const Extent cycle_limit =
        std::min<Extent>(restart, policy.iteration.max_iterations - iterations);
    bool closed = false;

    for (Index j = 0U; j < cycle_limit; ++j) {
      const std::span<const Wide> qj =
          std::span<const Wide>{q}.subspan(j * n, n);
      const std::span<Wide> zj = std::span<Wide>{z}.subspan(j * n, n);
      status = preconditionAction(preconditioner, qj, zj, evaluations);
      if (!status.ok()) {
        return failure(status);
      }
      status = applyAction(a, zj, w, evaluations);
      if (!status.ok()) {
        return failure(status);
      }
      for (Index pass = 0U; pass < 2U; ++pass) {
        for (Index i = 0U; i <= j; ++i) {
          const std::span<const Wide> qi =
              std::span<const Wide>{q}.subspan(i * n, n);
          const Wide correction = detail::compensatedDot(qi, w);
          h[i * restart + j] += correction;
          for (Index k = 0U; k < n; ++k) {
            w[k] = std::fma(-correction, qi[k], w[k]);
          }
        }
      }
      const Wide next_norm = detail::scaledNorm(w);
      h[(j + 1U) * restart + j] = next_norm;
      if (next_norm > 0.0L) {
        const std::span<Wide> qnext =
            std::span<Wide>{q}.subspan((j + 1U) * n, n);
        for (Index k = 0U; k < n; ++k) {
          qnext[k] = w[k] / next_norm;
        }
      } else {
        closed = true;
      }

      for (Index i = 0U; i < j; ++i) {
        const Wide upper = h[i * restart + j];
        const Wide lower = h[(i + 1U) * restart + j];
        h[i * restart + j] = cosine[i] * upper + sine[i] * lower;
        h[(i + 1U) * restart + j] = -sine[i] * upper + cosine[i] * lower;
      }
      const Wide diagonal = h[j * restart + j];
      const Wide subdiagonal = h[(j + 1U) * restart + j];
      const Wide rotation = std::hypot(diagonal, subdiagonal);
      if (rotation == 0.0L || !std::isfinite(rotation)) {
        return fail<IterativeSolve>(ErrorCode::NoConvergence,
                                    "GMRES projected system broke down");
      }
      cosine[j] = diagonal / rotation;
      sine[j] = subdiagonal / rotation;
      h[j * restart + j] = rotation;
      h[(j + 1U) * restart + j] = 0.0L;
      const Wide g_current = g[j];
      g[j] = cosine[j] * g_current;
      g[j + 1U] = -sine[j] * g_current;

      WideVector coefficients(j + 1U, 0.0L);
      for (Index reverse = 0U; reverse <= j; ++reverse) {
        const Index row = j - reverse;
        Wide value = g[row];
        for (Index column = row + 1U; column <= j; ++column) {
          value -= h[row * restart + column] * coefficients[column];
        }
        const Wide pivot = h[row * restart + row];
        if (pivot == 0.0L || !std::isfinite(pivot)) {
          return fail<IterativeSolve>(
              ErrorCode::NoConvergence,
              "GMRES projected triangular solve broke down");
        }
        coefficients[row] = value / pivot;
      }
      std::ranges::copy(cycle_base, candidate.begin());
      for (Index column = 0U; column <= j; ++column) {
        const std::span<const Wide> zj_values =
            std::span<const Wide>{z}.subspan(column * n, n);
        for (Index k = 0U; k < n; ++k) {
          candidate[k] =
              std::fma(coefficients[column], zj_values[k], candidate[k]);
        }
      }
      Wide candidate_norm = 0.0L;
      status = trueResidual(a, b, candidate, residual, image, evaluations,
                            candidate_norm);
      if (!status.ok()) {
        return failure(status);
      }
      x.swap(candidate);
      residual_norm = candidate_norm;
      ++iterations;
      if (residual_norm <= tolerance) {
        return finish(x, residual_norm, iterations, Termination::converged);
      }
      if (closed) {
        return fail<IterativeSolve>(
            ErrorCode::NoConvergence,
            "GMRES Krylov space closed before convergence");
      }
      if (iterations == policy.iteration.max_iterations) {
        return finish(x, residual_norm, iterations, Termination::cap);
      }
    }
  }
  return finish(x, residual_norm, iterations, Termination::cap);
}

} // namespace vss::math::linear
