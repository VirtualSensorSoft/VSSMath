#ifndef VSS_MATH_PCG_HPP
#define VSS_MATH_PCG_HPP

#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear::detail {

class EvaluationBudget final {
public:
  explicit EvaluationBudget(Index count) : remaining_(count) {}

  [[nodiscard]] bool charge(Index count = 1) {
    if (count > remaining_) {
      return false;
    }
    remaining_ -= count;
    return true;
  }

  [[nodiscard]] Index remaining() const { return remaining_; }

private:
  Index remaining_;
};

struct PcgWorkspace final {
  std::span<Wide> solution;
  std::span<Wide> residual;
  std::span<Wide> preconditioned;
  std::span<Wide> direction;
  std::span<Wide> image;
};

struct PcgProgress final {
  Wide residual_norm;
  Index iterations;
  bool converged;
};

inline Wide scaledNorm(std::span<const Wide> values) {
  Wide result = 0;
  for (const Wide value : values) {
    result = std::hypot(result, value);
  }
  return result;
}

inline Wide compensatedDot(std::span<const Wide> left,
                           std::span<const Wide> right) {
  Wide total = 0;
  Wide correction = 0;
  for (Index i = 0; i < left.size(); ++i) {
    const Wide item = left[i] * right[i];
    const Wide next = total + item;
    correction += std::fma(left[i], right[i], -item);
    correction += std::abs(total) >= std::abs(item) ? (total - next) + item
                                                    : (item - next) + total;
    total = next;
  }
  return total + correction;
}

// Callers supply checked, disjoint work buffers, an initial solution, and
// synchronous actions. Every candidate is checked at its published precision.
// The shared ledger includes the independent residual evaluations.
template <class Rhs, class Apply, class Precondition>
Result<PcgProgress>
runPcg(const Rhs& rhs, const Apply& apply, const Precondition& precondition,
       const IterPolicy& policy, EvaluationBudget& evaluations,
       PcgWorkspace workspace) {
  using Output = Result<PcgProgress>;
  const auto x = workspace.solution;
  const auto r = workspace.residual;
  const auto z = workspace.preconditioned;
  const auto direction = workspace.direction;
  const auto image = workspace.image;
  const auto checkedAction = [&evaluations](const auto& action,
                                            std::span<const Wide> input,
                                            std::span<Wide> output) {
    if (!evaluations.charge()) {
      return Status::failure(ErrorCode::ResourceLimit,
                             "PCG evaluation limit reached");
    }
    return action(input, output);
  };
  const auto copyFailure = [](const Status& status) {
    return Output::failure(status.code(), std::string(status.message()));
  };
  if (const auto status = checkedAction(apply, x, image); !status.ok()) {
    return copyFailure(status);
  }
  Wide rhs_norm = 0;
  for (Index i = 0; i < r.size(); ++i) {
    const Wide value = rhs(i);
    rhs_norm = std::hypot(rhs_norm, value);
    r[i] = value - image[i];
  }
  Wide true_residual = scaledNorm(r);
  const Wide tolerance = policy.abs + policy.rel * rhs_norm;
  Wide previous_energy = 0;
  Index step = 0;
  while (true_residual > tolerance && step < policy.max_iterations) {
    if (const auto status = checkedAction(precondition, r, z); !status.ok()) {
      return copyFailure(status);
    }
    const Wide energy = compensatedDot(r, z);
    if (!(energy > 0) || !std::isfinite(energy)) {
      return Output::failure(energy < 0 ? ErrorCode::NotPositiveDefinite
                                        : ErrorCode::NoConvergence,
                             "PCG preconditioned residual broke down");
    }
    if (step == 0) {
      std::ranges::copy(z, direction.begin());
    } else {
      const Wide beta = energy / previous_energy;
      for (Index i = 0; i < x.size(); ++i) {
        direction[i] = std::fma(beta, direction[i], z[i]);
      }
    }
    if (const auto status = checkedAction(apply, direction, image);
        !status.ok()) {
      return copyFailure(status);
    }
    const Wide curvature = compensatedDot(direction, image);
    if (!(curvature > 0) || !std::isfinite(curvature)) {
      return Output::failure(curvature < 0 ? ErrorCode::NotPositiveDefinite
                                           : ErrorCode::NoConvergence,
                             "PCG curvature broke down");
    }
    const Wide alpha = energy / curvature;
    for (Index i = 0; i < x.size(); ++i) {
      x[i] = std::fma(alpha, direction[i], x[i]);
      r[i] = std::fma(-alpha, image[i], r[i]);
      if (!representable(x[i])) {
        return Output::failure(ErrorCode::NonFiniteResult,
                               "PCG candidate is not representable");
      }
      z[i] = static_cast<Scalar>(x[i]);
    }
    if (const auto status = checkedAction(apply, z, image); !status.ok()) {
      return copyFailure(status);
    }
    for (Index i = 0; i < x.size(); ++i) {
      image[i] -= rhs(i);
    }
    true_residual = scaledNorm(image);
    previous_energy = energy;
    ++step;
  }
  if (!representable(true_residual)) {
    return Output::failure(ErrorCode::NonFiniteResult,
                           "PCG residual is not representable");
  }
  return Output::success({.residual_norm = true_residual,
                          .iterations = step,
                          .converged = true_residual <= tolerance});
}

} // namespace vss::math::linear::detail
#endif
