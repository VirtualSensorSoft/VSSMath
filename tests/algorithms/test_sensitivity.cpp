#include "Test.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/ProbabilityPolicy.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/Vector.hpp"
#include "vss/math/VectorView.hpp"
#include "vss/math/covariance/Budget.hpp"
#include "vss/math/differentiation/FiniteDifference.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/covariance/Sensitivity.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::covariance;
using probability::Distribution;
using probability::Rng;

ProbabilityPolicy samplingPolicy(std::uint64_t draws) {
  return {.iteration = {.abs = 1.0e-12,
                        .rel = 1.0e-12,
                        .max_iterations = 4096U,
                        .max_evaluations = 100000U,
                        .bracket = std::nullopt},
          .max_draws = draws,
          .abs_log_error = 1.0e-10};
}

SensitivitySpec fullSpec(Index draws) {
  return {.draws = draws,
          .replicates = 64U,
          .output_dimension = 1U,
          .first_order = true,
          .total_order = true,
          .sampling = samplingPolicy(Index{2U} * 64U * draws * 2U)};
}

Vector scalarResult(Scalar value) {
  Vector result(1U);
  result.at(0U) = value;
  return result;
}

void replicatedAdditiveAndInteractionIndicesMatchAnalyticTargets() {
  const auto x = probability::normal(0.0, 1.0);
  const auto y = probability::normal(0.0, 1.0);
  VSS_CHECK(x && y);
  if (!x || !y) {
    return;
  }
  const std::array<Distribution, 2> marginals{x.value(), y.value()};
  const IndependentMarginals law{marginals};
  const bool fullAcceptance =
      std::getenv("VSS_MATH_FULL_SENSITIVITY_ACCEPTANCE") != nullptr;
  const SensitivitySpec spec = fullSpec(fullAcceptance ? 131072U : 4096U);
  Rng additiveRng(0x51a17eULL);
  const differentiation::VectorFunction additive =
      [](ConstVectorView input) -> Result<Vector> {
    return Result<Vector>::success(
        scalarResult(input.at(0U) + (2.0 * input.at(1U))));
  };
  const auto additiveResult = sensitivityIndices(
      additive, law, spec, additiveRng,
      {.elements = 4096U, .bytes = 1048576U, .evaluations = 40000000U});
  VSS_CHECK(additiveResult.hasValue());
  if (additiveResult) {
    const auto& result = additiveResult.value();
    VSS_CHECK(result.first_order.rows() == 1U &&
              result.first_order.columns() == 2U);
    VSS_CHECK(result.total_order.rows() == 1U &&
              result.total_order.columns() == 2U);
    VSS_CHECK(result.monte_carlo_error.rows() == 2U &&
              result.monte_carlo_error.columns() == 2U);
    for (Index j = 0U; j < 2U; ++j) {
      const Scalar firstTarget = j == 0U ? 0.2 : 0.8;
      const Scalar totalTarget = firstTarget;
      VSS_CHECK(std::abs(result.first_order.at(0U, j) - firstTarget) <=
                6.0 * result.monte_carlo_error.at(0U, j));
      VSS_CHECK(std::abs(result.total_order.at(0U, j) - totalTarget) <=
                6.0 * result.monte_carlo_error.at(1U, j));
    }
  }

  Rng interactionRng(0x1a2b3cULL);
  const differentiation::VectorFunction interaction =
      [](ConstVectorView input) -> Result<Vector> {
    return Result<Vector>::success(scalarResult(input.at(0U) * input.at(1U)));
  };
  const auto interactionResult = sensitivityIndices(
      interaction, law, fullAcceptance ? fullSpec(16384U) : spec,
      interactionRng,
      {.elements = 4096U, .bytes = 1048576U, .evaluations = 40000000U});
  VSS_CHECK(interactionResult.hasValue());
  if (interactionResult) {
    const auto& result = interactionResult.value();
    for (Index j = 0U; j < 2U; ++j) {
      VSS_CHECK(std::abs(result.first_order.at(0U, j)) <=
                6.0 * result.monte_carlo_error.at(0U, j));
      VSS_CHECK(std::abs(result.total_order.at(0U, j) - 1.0) <=
                6.0 * result.monte_carlo_error.at(1U, j));
    }
  }
}

void requestedShapesCallbackFailuresAndPreflightAreExplicit() {
  const auto lawValue = probability::normal(0.0, 1.0);
  VSS_CHECK(lawValue.hasValue());
  if (!lawValue) {
    return;
  }
  const std::array<Distribution, 2> marginals{lawValue.value(),
                                              lawValue.value()};
  const IndependentMarginals law{marginals};
  SensitivitySpec const spec{.draws = 16U,
                             .replicates = 4U,
                             .output_dimension = 2U,
                             .first_order = true,
                             .total_order = false,
                             .sampling = samplingPolicy(128U)};
  Rng rng(7U);
  Index evaluations = 0U;
  const differentiation::VectorFunction twoOutput =
      [&evaluations](ConstVectorView input) -> Result<Vector> {
    ++evaluations;
    Vector result(2U);
    result.at(0U) = input.at(0U) + input.at(1U);
    result.at(1U) = input.at(0U) - input.at(1U);
    return Result<Vector>::success(std::move(result));
  };
  const auto shapeResult = sensitivityIndices(
      twoOutput, law, spec, rng,
      {.elements = 128U, .bytes = 8192U, .evaluations = 256U});
  VSS_CHECK(shapeResult.hasValue());
  VSS_CHECK(evaluations == Index{4U} * 16U * 4U);
  if (shapeResult) {
    VSS_CHECK(shapeResult.value().first_order.rows() == 2U);
    VSS_CHECK(shapeResult.value().first_order.columns() == 2U);
    VSS_CHECK(shapeResult.value().total_order.rows() == 0U);
    VSS_CHECK(shapeResult.value().total_order.columns() == 0U);
    VSS_CHECK(shapeResult.value().monte_carlo_error.rows() == 2U);
    VSS_CHECK(shapeResult.value().monte_carlo_error.columns() == 2U);
  }

  Index calls = 0U;
  const differentiation::VectorFunction failing =
      [&calls](ConstVectorView) -> Result<Vector> {
    ++calls;
    return Result<Vector>::failure(ErrorCode::InvalidDomain, "injected");
  };
  Rng failureRng(8U);
  const auto failed = sensitivityIndices(
      failing, law, spec, failureRng,
      {.elements = 128U, .bytes = 8192U, .evaluations = 256U});
  VSS_CHECK(!failed.hasValue());
  VSS_CHECK(failed.status().code() == ErrorCode::CallbackFailure);
  VSS_CHECK(calls == 1U);

  calls = 0U;
  const differentiation::VectorFunction changingDimension =
      [&calls](ConstVectorView) -> Result<Vector> {
    ++calls;
    return Result<Vector>::success(Vector(calls == 1U ? 2U : 1U));
  };
  Rng changingRng(81U);
  const auto changed = sensitivityIndices(
      changingDimension, law, spec, changingRng,
      {.elements = 128U, .bytes = 8192U, .evaluations = 256U});
  VSS_CHECK(!changed.hasValue());
  VSS_CHECK(changed.status().code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(calls == 2U);

  calls = 0U;
  const differentiation::VectorFunction constant =
      [&calls](ConstVectorView) -> Result<Vector> {
    ++calls;
    return Result<Vector>::success(scalarResult(4.0));
  };
  SensitivitySpec const small{.draws = 8U,
                              .replicates = 2U,
                              .output_dimension = 1U,
                              .first_order = true,
                              .total_order = true,
                              .sampling = samplingPolicy(64U)};
  Rng constantRng(9U);
  const auto zeroVariance = sensitivityIndices(
      constant, law, small, constantRng,
      {.elements = 128U, .bytes = 8192U, .evaluations = 64U});
  VSS_CHECK(!zeroVariance.hasValue());
  VSS_CHECK(zeroVariance.status().code() == ErrorCode::InvalidDomain);

  evaluations = 0U;
  Rng limitedRng(10U);
  const auto denied =
      sensitivityIndices(twoOutput, law, spec, limitedRng,
                         {.elements = 128U, .bytes = 8192U, .evaluations = 1U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(evaluations == 0U);

  evaluations = 0U;
  Rng storageLimitedRng(11U);
  const auto storageDenied =
      sensitivityIndices(twoOutput, law, spec, storageLimitedRng,
                         {.elements = 1U, .bytes = 8192U, .evaluations = 256U});
  VSS_CHECK(!storageDenied.hasValue());
  VSS_CHECK(storageDenied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(evaluations == 0U);
}

void uncertaintyBudgetAliasRetainsCrossTerms() {
  Matrix jacobian(1U, 2U, 1.0);
  Matrix left = Matrix(1U, 1U, 1.0);
  Matrix right = Matrix(1U, 1U, 1.0);
  Matrix crossCovariance = Matrix(1U, 1U, 0.5);
  const std::array<Index, 1U> leftCoordinate{0U};
  const std::array<Index, 1U> rightCoordinate{1U};
  const std::array<SourceBlock, 2U> sources{{
      {.id = 1U, .coordinates = leftCoordinate, .covariance = left.view()},
      {.id = 2U, .coordinates = rightCoordinate, .covariance = right.view()},
  }};
  const std::array<SourceCrossBlock, 1U> cross{{
      {.left = 1U, .right = 2U, .covariance = crossCovariance.view()},
  }};
  const auto result = uncertaintyBudget(
      jacobian.view(), sources, cross, DependencePolicy::exact_joint,
      {.elements = 128U, .bytes = 8192U, .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  if (result) {
    VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 3.0) < 1.0e-12);
  }
}
} // namespace

int main() try {
  replicatedAdditiveAndInteractionIndicesMatchAnalyticTargets();
  requestedShapesCallbackFailuresAndPreflightAreExplicit();
  uncertaintyBudgetAliasRetainsCrossTerms();
} catch (...) {
  return 1;
}

// VSS-MATH-TRACEABILITY: C093 CASE-C093.sobol-additive-normal
// VSS-MATH-TRACEABILITY: C093 CASE-C093.sobol-callback-failure
// VSS-MATH-TRACEABILITY: C093 CASE-C093.sobol-interaction-normal
// VSS-MATH-TRACEABILITY: C093 CASE-C093.sobol-output-shape
// VSS-MATH-TRACEABILITY: C093 CASE-C093.sobol-zero-variance
// VSS-MATH-TRACEABILITY: C093 CASE-C093.uncertainty-budget.cross-term
