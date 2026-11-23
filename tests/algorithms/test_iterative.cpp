#include "Test.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Structured.hpp>

namespace {

using namespace vss::math;

constexpr Limits ample{4096U, 262144U, 100U};

SolvePolicy policy(Index maximum = 8U) {
  return {.iteration = {.abs = 1e-12,
                        .rel = 1e-12,
                        .max_iterations = maximum,
                        .max_evaluations = 100U,
                        .bracket = std::nullopt},
          .rank_abs = 0.0,
          .rank_rel = 0.0,
          .reciprocal_condition_floor = 0.0,
          .rank = RankPolicy::minimum_norm,
          .restart = 2U};
}

void solvesWithAllThreeMethods() {
  // VSS-MATH-TRACEABILITY: C013
  const Vector rhs{std::array<Scalar, 2U>{2.0, 8.0}};
  const Vector zero(2U);
  Index transpose_calls = 0U;
  const auto diagonal = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{2.0 * x.at(0U), 4.0 * x.at(1U)}});
  };
  const auto forbidden_transpose = [&transpose_calls](ConstVectorView) {
    ++transpose_calls;
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "C013 must not invoke transpose");
  };
  const auto inverse = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(0U) / 2.0, x.at(1U) / 4.0}});
  };
  const linear::LinearMap spd{2U, 2U, diagonal, forbidden_transpose};
  const linear::LinearMap preconditioner{2U, 2U, inverse, inverse};
  const auto cg_result = linear::cg(spd, rhs.view(), zero.view(),
                                    &preconditioner, policy(), ample);
  VSS_CHECK(cg_result.hasValue());
  VSS_CHECK(cg_result.value().termination == Termination::converged);
  VSS_CHECK(cg_result.value().iterations == 1U);
  VSS_CHECK(std::abs(cg_result.value().value.at(0U) - 1.0) < 1e-12);
  VSS_CHECK(std::abs(cg_result.value().value.at(1U) - 2.0) < 1e-12);
  const Scalar cg_residual =
      std::hypot(rhs.at(0U) - 2.0 * cg_result.value().value.at(0U),
                 rhs.at(1U) - 4.0 * cg_result.value().value.at(1U));
  VSS_CHECK(cg_residual < 1e-12);
  VSS_CHECK(std::abs(cg_result.value().residual_norm - cg_residual) < 1e-12);
  VSS_CHECK(transpose_calls == 0U);

  const auto swap = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(1U), x.at(0U)}});
  };
  const linear::LinearMap indefinite{2U, 2U, swap, forbidden_transpose};
  const auto minres_result = linear::minres(indefinite, rhs.view(), zero.view(),
                                            nullptr, policy(), ample);
  VSS_CHECK(minres_result.hasValue());
  VSS_CHECK(minres_result.value().termination == Termination::converged);
  VSS_CHECK(std::abs(minres_result.value().value.at(0U) - 8.0) < 1e-12);
  VSS_CHECK(std::abs(minres_result.value().value.at(1U) - 2.0) < 1e-12);
  const Scalar minres_residual =
      std::hypot(rhs.at(0U) - minres_result.value().value.at(1U),
                 rhs.at(1U) - minres_result.value().value.at(0U));
  VSS_CHECK(minres_residual < 1e-12);
  VSS_CHECK(std::abs(minres_result.value().residual_norm - minres_residual) <
            1e-12);
  VSS_CHECK(transpose_calls == 0U);
  const auto minres_preconditioned = linear::minres(
      indefinite, rhs.view(), zero.view(), &preconditioner, policy(), ample);
  VSS_CHECK(minres_preconditioned.hasValue());
  VSS_CHECK(std::abs(minres_preconditioned.value().value.at(0U) - 8.0) < 1e-12);
  VSS_CHECK(std::abs(minres_preconditioned.value().value.at(1U) - 2.0) < 1e-12);
  const Scalar preconditioned_residual =
      std::hypot(rhs.at(0U) - minres_preconditioned.value().value.at(1U),
                 rhs.at(1U) - minres_preconditioned.value().value.at(0U));
  VSS_CHECK(std::abs(minres_preconditioned.value().residual_norm -
                     preconditioned_residual) < 1e-12);
  VSS_CHECK(transpose_calls == 0U);

  const auto upper = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(0U) + x.at(1U), 2.0 * x.at(1U)}});
  };
  const linear::LinearMap nonsymmetric{2U, 2U, upper, {}};
  const auto gmres_result = linear::gmres(nonsymmetric, rhs.view(), zero.view(),
                                          nullptr, policy(), ample);
  VSS_CHECK(gmres_result.hasValue());
  VSS_CHECK(gmres_result.value().termination == Termination::converged);
  VSS_CHECK(std::abs(gmres_result.value().value.at(0U) + 2.0) < 1e-12);
  VSS_CHECK(std::abs(gmres_result.value().value.at(1U) - 4.0) < 1e-12);
  const Scalar gmres_residual =
      std::hypot(rhs.at(0U) - (gmres_result.value().value.at(0U) +
                               gmres_result.value().value.at(1U)),
                 rhs.at(1U) - 2.0 * gmres_result.value().value.at(1U));
  VSS_CHECK(gmres_residual < 1e-12);
  VSS_CHECK(std::abs(gmres_result.value().residual_norm - gmres_residual) <
            1e-12);
}

void returnsVerifiedInitialAndCappedIterates() {
  // VSS-MATH-TRACEABILITY: C013
  const Vector rhs{std::array<Scalar, 2U>{2.0, 8.0}};
  const Vector zero(2U);
  Index calls = 0U;
  Index preconditioner_calls = 0U;
  const auto diagonal = [&calls](ConstVectorView x) {
    ++calls;
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{2.0 * x.at(0U), 4.0 * x.at(1U)}});
  };
  const auto inverse = [&preconditioner_calls](ConstVectorView x) {
    ++preconditioner_calls;
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(0U) / 2.0, x.at(1U) / 4.0}});
  };
  const linear::LinearMap counted{2U, 2U, diagonal, {}};
  const linear::LinearMap pre{2U, 2U, inverse, {}};
  const Vector exact{std::array<Scalar, 2U>{1.0, 2.0}};
  const auto initial =
      linear::cg(counted, rhs.view(), exact.view(), &pre, policy(0U), ample);
  VSS_CHECK(initial.hasValue());
  VSS_CHECK(initial.value().iterations == 0U);
  VSS_CHECK(initial.value().termination == Termination::converged);
  VSS_CHECK(calls == 1U);
  VSS_CHECK(preconditioner_calls == 0U);

  calls = 0U;
  const auto capped =
      linear::cg(counted, rhs.view(), zero.view(), &pre, policy(0U), ample);
  VSS_CHECK(capped.hasValue());
  VSS_CHECK(capped.value().iterations == 0U);
  VSS_CHECK(capped.value().termination == Termination::cap);
  VSS_CHECK(std::abs(capped.value().residual_norm - std::sqrt(68.0)) < 1e-12);
  VSS_CHECK(calls == 1U);
  VSS_CHECK(preconditioner_calls == 0U);

  calls = 0U;
  preconditioner_calls = 0U;
  const auto denied = linear::cg(counted, rhs.view(), zero.view(), &pre,
                                 policy(), Limits{4096U, 262144U, 1U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(calls == 1U);
  VSS_CHECK(preconditioner_calls == 0U);
}

void reportsCurvatureBreakdownAndRestartedRightPreconditioning() {
  // VSS-MATH-TRACEABILITY: C013
  const Vector rhs{std::array<Scalar, 2U>{2.0, 8.0}};
  const Vector zero(2U);
  const auto negative = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{-x.at(0U), -x.at(1U)}});
  };
  const linear::LinearMap notSpd{2U, 2U, negative, {}};
  const auto cg_failure =
      linear::cg(notSpd, rhs.view(), zero.view(), nullptr, policy(), ample);
  VSS_CHECK(!cg_failure.hasValue());
  VSS_CHECK(cg_failure.status().code() == ErrorCode::NotPositiveDefinite);

  const auto swap = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(1U), x.at(0U)}});
  };
  const auto negative_pre = negative;
  const linear::LinearMap indefinite{2U, 2U, swap, {}};
  const linear::LinearMap invalid_pre{2U, 2U, negative_pre, {}};
  const auto minres_failure = linear::minres(
      indefinite, rhs.view(), zero.view(), &invalid_pre, policy(), ample);
  VSS_CHECK(!minres_failure.hasValue());
  VSS_CHECK(minres_failure.status().code() == ErrorCode::NotPositiveDefinite);

  const auto upper = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(0U) + x.at(1U), 2.0 * x.at(1U)}});
  };
  const auto inverse = [](ConstVectorView x) {
    return Result<Vector>::success(
        Vector{std::array<Scalar, 2U>{x.at(0U) / 2.0, x.at(1U) / 4.0}});
  };
  const linear::LinearMap nonsymmetric{2U, 2U, upper, {}};
  const linear::LinearMap pre{2U, 2U, inverse, {}};
  auto one_step = policy(1U);
  one_step.restart = 1U;
  const auto capped = linear::gmres(nonsymmetric, rhs.view(), zero.view(), &pre,
                                    one_step, ample);
  VSS_CHECK(capped.hasValue());
  VSS_CHECK(capped.value().termination == Termination::cap);
  VSS_CHECK(capped.value().iterations == 1U);
  VSS_CHECK(std::abs(capped.value().value.at(0U) - 38.0 / 25.0) < 1e-12);
  VSS_CHECK(std::abs(capped.value().value.at(1U) - 76.0 / 25.0) < 1e-12);
  const Scalar independent_residual =
      std::hypot(rhs.at(0U) - (38.0 + 76.0) / 25.0, rhs.at(1U) - 152.0 / 25.0);
  VSS_CHECK(std::abs(independent_residual - 80.0 / 25.0) < 1e-12);
  VSS_CHECK(std::abs(capped.value().residual_norm - independent_residual) <
            1e-12);
}

void handlesThreeDimensionalSystemsAgainstKnownSolutions() {
  // VSS-MATH-TRACEABILITY: C013
  const std::array<Scalar, 3U> expected{1.0, -2.0, 3.0};
  const auto multiply = [](std::array<Scalar, 9U> matrix) {
    return [matrix](ConstVectorView x) {
      std::array<Scalar, 3U> result{};
      for (Index row = 0U; row < 3U; ++row) {
        for (Index column = 0U; column < 3U; ++column) {
          result.at(row) += matrix.at(row * 3U + column) * x.at(column);
        }
      }
      return Result<Vector>::success(Vector(result));
    };
  };
  const auto vector_from = [](std::array<Scalar, 9U> matrix,
                              std::array<Scalar, 3U> x) {
    std::array<Scalar, 3U> result{};
    for (Index row = 0U; row < 3U; ++row) {
      for (Index column = 0U; column < 3U; ++column) {
        result.at(row) += matrix.at(row * 3U + column) * x.at(column);
      }
    }
    return Vector(result);
  };
  const auto independent_residual =
      [&vector_from](const std::array<Scalar, 9U>& matrix, const Vector& rhs,
                     const Vector& candidate) {
        std::array<Scalar, 3U> values{};
        const std::array<Scalar, 3U> candidate_values{
            candidate.at(0U), candidate.at(1U), candidate.at(2U)};
        const Vector applied = vector_from(matrix, candidate_values);
        for (Index i = 0U; i < values.size(); ++i) {
          values.at(i) = rhs.at(i) - applied.at(i);
        }
        return std::sqrt(values.at(0U) * values.at(0U) +
                         values.at(1U) * values.at(1U) +
                         values.at(2U) * values.at(2U));
      };
  const Vector solution(expected);
  const Vector zero(3U);
  const std::array<Scalar, 9U> spd_values{4.0, 1.0, 0.0, 1.0, 3.0,
                                          1.0, 0.0, 1.0, 2.0};
  const std::array<Scalar, 9U> inverse_values{0.25, 0.0, 0.0, 0.0, 1.0 / 3.0,
                                              0.0,  0.0, 0.0, 0.5};
  const linear::LinearMap spd{3U, 3U, multiply(spd_values), {}};
  const linear::LinearMap jacobi{3U, 3U, multiply(inverse_values), {}};
  const Vector spd_rhs = vector_from(spd_values, expected);
  const auto cg_result =
      linear::cg(spd, spd_rhs.view(), zero.view(), &jacobi, policy(), ample);
  VSS_CHECK(cg_result.hasValue());
  for (Index i = 0U; i < 3U; ++i) {
    VSS_CHECK(std::abs(cg_result.value().value.at(i) - expected.at(i)) < 1e-11);
  }
  const Scalar cg_true_residual =
      independent_residual(spd_values, spd_rhs, cg_result.value().value);
  VSS_CHECK(cg_true_residual < 1e-11);
  VSS_CHECK(std::abs(cg_result.value().residual_norm - cg_true_residual) <
            1e-11);

  const std::array<Scalar, 9U> indefinite_values{0.0, 1.0, 0.0, 1.0, 0.0,
                                                 1.0, 0.0, 1.0, 2.0};
  const linear::LinearMap indefinite{3U, 3U, multiply(indefinite_values), {}};
  const Vector indefinite_rhs = vector_from(indefinite_values, expected);
  const auto minres_result = linear::minres(
      indefinite, indefinite_rhs.view(), zero.view(), nullptr, policy(), ample);
  VSS_CHECK(minres_result.hasValue());
  for (Index i = 0U; i < 3U; ++i) {
    VSS_CHECK(std::abs(minres_result.value().value.at(i) - expected.at(i)) <
              1e-11);
  }
  const Scalar minres_true_residual = independent_residual(
      indefinite_values, indefinite_rhs, minres_result.value().value);
  VSS_CHECK(minres_true_residual < 1e-11);
  VSS_CHECK(std::abs(minres_result.value().residual_norm -
                     minres_true_residual) < 1e-11);

  const std::array<Scalar, 9U> general_values{1.0, 1.0, 0.0, 0.0, 2.0,
                                              1.0, 1.0, 0.0, 3.0};
  const linear::LinearMap general{3U, 3U, multiply(general_values), {}};
  const Vector general_rhs = vector_from(general_values, expected);
  auto full_restart = policy();
  full_restart.restart = 3U;
  const auto gmres_result = linear::gmres(
      general, general_rhs.view(), zero.view(), nullptr, full_restart, ample);
  VSS_CHECK(gmres_result.hasValue());
  for (Index i = 0U; i < 3U; ++i) {
    VSS_CHECK(std::abs(gmres_result.value().value.at(i) - expected.at(i)) <
              1e-11);
  }
  const Scalar gmres_true_residual = independent_residual(
      general_values, general_rhs, gmres_result.value().value);
  VSS_CHECK(gmres_true_residual < 1e-11);
  VSS_CHECK(std::abs(gmres_result.value().residual_norm - gmres_true_residual) <
            1e-11);
}

void validatesCallbacksPoliciesAndBreakdown() {
  // VSS-MATH-TRACEABILITY: C013
  const Vector rhs{std::array<Scalar, 2U>{2.0, 8.0}};
  const Vector zero(2U);
  const linear::LinearMap missing{2U, 2U, {}, {}};
  const auto absent =
      linear::cg(missing, rhs.view(), zero.view(), nullptr, policy(), ample);
  VSS_CHECK(!absent.hasValue());
  VSS_CHECK(absent.status().code() == ErrorCode::CallbackFailure);

  auto invalid_restart = policy();
  invalid_restart.restart = 0U;
  const linear::LinearMap valid_map{
      2U,
      2U,
      [](ConstVectorView x) {
        return Result<Vector>::success(
            Vector{std::array<Scalar, 2U>{x.at(0U), x.at(1U)}});
      },
      {}};
  const auto restart_error = linear::gmres(valid_map, rhs.view(), zero.view(),
                                           nullptr, invalid_restart, ample);
  VSS_CHECK(!restart_error.hasValue());
  VSS_CHECK(restart_error.status().code() == ErrorCode::InvalidDomain);

  const linear::LinearMap wrong_shape{
      2U,
      2U,
      [](ConstVectorView) { return Result<Vector>::success(Vector(1U)); },
      {}};
  const auto shape = linear::cg(wrong_shape, rhs.view(), zero.view(), nullptr,
                                policy(), ample);
  VSS_CHECK(!shape.hasValue());
  VSS_CHECK(shape.status().code() == ErrorCode::DimensionMismatch);

  Index limited_calls = 0U;
  const linear::LinearMap limited_map{
      2U,
      2U,
      [&limited_calls](ConstVectorView x) {
        ++limited_calls;
        return Result<Vector>::success(
            Vector{std::array<Scalar, 2U>{x.at(0U), x.at(1U)}});
      },
      {}};
  const auto limited =
      linear::gmres(limited_map, rhs.view(), zero.view(), nullptr, policy(),
                    Limits{1U, sizeof(Scalar), ample.evaluations});
  VSS_CHECK(!limited.hasValue());
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(limited_calls == 0U);

  const linear::LinearMap throws{2U,
                                 2U,
                                 [](ConstVectorView) -> Result<Vector> {
                                   throw std::runtime_error("callback");
                                 },
                                 {}};
  const auto callback_failure =
      linear::cg(throws, rhs.view(), zero.view(), nullptr, policy(), ample);
  VSS_CHECK(!callback_failure.hasValue());
  VSS_CHECK(callback_failure.status().code() == ErrorCode::CallbackFailure);

  const linear::LinearMap returns_error{2U,
                                        2U,
                                        [](ConstVectorView) {
                                          return Result<Vector>::failure(
                                              ErrorCode::InvalidDomain,
                                              "deliberate action failure");
                                        },
                                        {}};
  const auto propagated = linear::cg(returns_error, rhs.view(), zero.view(),
                                     nullptr, policy(), ample);
  VSS_CHECK(!propagated.hasValue());
  VSS_CHECK(propagated.status().code() == ErrorCode::InvalidDomain);

  const linear::LinearMap zero_action{
      2U,
      2U,
      [](ConstVectorView) { return Result<Vector>::success(Vector(2U)); },
      {}};
  const auto breakdown = linear::gmres(zero_action, rhs.view(), zero.view(),
                                       nullptr, policy(), ample);
  VSS_CHECK(!breakdown.hasValue());
  VSS_CHECK(breakdown.status().code() == ErrorCode::NoConvergence);

  auto nonfinite_policy = policy();
  nonfinite_policy.iteration.abs = std::numeric_limits<Scalar>::infinity();
  const auto tolerance = linear::cg(zero_action, rhs.view(), zero.view(),
                                    nullptr, nonfinite_policy, ample);
  VSS_CHECK(!tolerance.hasValue());
  VSS_CHECK(tolerance.status().code() == ErrorCode::InvalidTolerance);

  Index empty_calls = 0U;
  const linear::LinearMap empty{0U,
                                0U,
                                [&empty_calls](ConstVectorView) {
                                  ++empty_calls;
                                  return Result<Vector>::success(Vector{});
                                },
                                {}};
  const Vector empty_vector;
  const auto empty_result =
      linear::gmres(empty, empty_vector.view(), empty_vector.view(), nullptr,
                    policy(), ample);
  const auto empty_cg =
      linear::cg(empty, empty_vector.view(), empty_vector.view(), nullptr,
                 policy(), ample);
  const auto empty_minres =
      linear::minres(empty, empty_vector.view(), empty_vector.view(), nullptr,
                     policy(), ample);
  VSS_CHECK(empty_result.hasValue() && empty_result.value().value.empty());
  VSS_CHECK(empty_cg.hasValue() && empty_cg.value().value.empty());
  VSS_CHECK(empty_minres.hasValue() && empty_minres.value().value.empty());
  VSS_CHECK(empty_calls == 0U);
}

} // namespace

int main() try {
  solvesWithAllThreeMethods();
  returnsVerifiedInitialAndCappedIterates();
  reportsCurvatureBreakdownAndRestartedRightPreconditioning();
  handlesThreeDimensionalSystemsAgainstKnownSolutions();
  validatesCallbacksPoliciesAndBreakdown();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
