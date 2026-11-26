#include "Test.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/Vector.hpp"
#include "vss/math/VectorView.hpp"
#include "vss/math/covariance/Budget.hpp"
#include "vss/math/covariance/Conditioning.hpp"
#include "vss/math/linear/SolvePolicy.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <span>
#include <vss/math/Error.hpp>
#include <vss/math/covariance/ErrorModels.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::covariance;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Index i = 0U;
  for (const auto& row : rows) {
    Index j = 0U;
    for (const Scalar value : row) {
      result.at(i, j++) = value;
    }
    ++i;
  }
  return result;
}

void independentAndGroupedContributionsAreAssembled() {
  Matrix identity = matrix({{1.0, 0.0}, {0.0, 1.0}});
  Matrix first = matrix({{1.0}});
  Matrix second = matrix({{1.0}});
  const std::array<Index, 1> firstCoordinate{0U};
  const std::array<Index, 1> secondCoordinate{1U};
  const std::array<SourceBlock, 2> sources{{
      {.id = 1U, .coordinates = firstCoordinate, .covariance = first.view()},
      {.id = 2U, .coordinates = secondCoordinate, .covariance = second.view()},
  }};
  Matrix crossCovariance = matrix({{0.25}});
  const std::array<SourceCrossBlock, 1> cross{{
      {.left = 1U, .right = 2U, .covariance = crossCovariance.view()},
  }};
  const Vector diagonal(std::array<Scalar, 2>{0.04, 0.04});
  const Vector state;
  const ErrorCovarianceSpec spec{.evaluate = {},
                                 .diagonal = diagonal.view(),
                                 .loading = identity.view(),
                                 .sources = sources,
                                 .cross = cross,
                                 .dependence = DependencePolicy::exact_joint};
  const auto result = assembleErrorCovariance(
      spec, state.view(), 0.0,
      {.elements = 512U, .bytes = 16384U, .evaluations = 10U});
  VSS_CHECK(result.hasValue());
  if (result) {
    VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 1.04) < 1.0e-12);
    VSS_CHECK(std::abs(result.value().covariance.at(1U, 1U) - 1.04) < 1.0e-12);
    VSS_CHECK(std::abs(result.value().covariance.at(0U, 1U) - 0.25) < 1.0e-12);
    VSS_CHECK(std::abs(result.value().covariance.at(1U, 0U) - 0.25) < 1.0e-12);
  }

  Matrix firstIndependentError = matrix({{0.04}});
  Matrix secondIndependentError = matrix({{0.04}});
  const std::array<SourceBlock, 2> independentSources{{
      {.id = 1U,
       .coordinates = firstCoordinate,
       .covariance = firstIndependentError.view()},
      {.id = 2U,
       .coordinates = secondCoordinate,
       .covariance = secondIndependentError.view()},
  }};
  const ErrorCovarianceSpec independent{
      .evaluate = {},
      .diagonal = std::nullopt,
      .loading = identity.view(),
      .sources = independentSources,
      .cross = {},
      .dependence = DependencePolicy::declared_independent_blocks};
  const auto independentResult = assembleErrorCovariance(
      independent, state.view(), 0.0,
      {.elements = 512U, .bytes = 16384U, .evaluations = 10U});
  VSS_CHECK(independentResult.hasValue());
  if (independentResult) {
    VSS_CHECK(independentResult.value().covariance.at(0U, 0U) == 0.04);
    VSS_CHECK(independentResult.value().covariance.at(1U, 1U) == 0.04);
    VSS_CHECK(independentResult.value().covariance.at(0U, 1U) == 0.0);
    VSS_CHECK(independentResult.value().covariance.at(1U, 0U) == 0.0);
  }
}

void conditionedEvaluationHonorsItsBudget() {
  Index calls = 0U;
  const ErrorCovarianceSpec spec{
      .evaluate =
          [&calls](ConstVectorView state, Scalar time) {
            ++calls;
            if (state.size() != 1U || state.at(0U) != 2.0 || time != 5.0) {
              return Result<Matrix>::failure(ErrorCode::InvalidDomain,
                                             "unexpected conditioning input");
            }
            return Result<Matrix>::success(matrix({{7.0}}));
          },
      .diagonal = std::nullopt,
      .loading = std::nullopt,
      .sources = {},
      .cross = {},
      .dependence = DependencePolicy::exact_joint};
  const Vector state(std::array<Scalar, 1>{2.0});
  const auto result = assembleErrorCovariance(
      spec, state.view(), 5.0,
      {.elements = 64U, .bytes = 4096U, .evaluations = 1U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(calls == 1U);
  if (result) {
    VSS_CHECK(result.value().covariance.at(0U, 0U) == 7.0);
  }
  const auto denied = assembleErrorCovariance(
      spec, state.view(), 5.0,
      {.elements = 64U, .bytes = 4096U, .evaluations = 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(calls == 1U);
}

void knownErrorSubtractionRetainsSignedResults() {
  Matrix observed = matrix({{5.0, 0.5}, {0.5, 3.0}});
  Matrix known = matrix({{1.0, 0.5}, {0.5, 0.5}});
  SolvePolicy const policy{.iteration = {.abs = 1.0e-12,
                                         .rel = 1.0e-12,
                                         .max_iterations = 4096U,
                                         .max_evaluations = 100000U,
                                         .bracket = std::nullopt},
                           .rank_abs = 0.0,
                           .rank_rel = 1.0e-12,
                           .reciprocal_condition_floor = 1.0e-10,
                           .rank = RankPolicy::minimum_norm,
                           .restart = 0U};
  const auto result = subtractKnownErrorCovariance(
      observed.view(), known.view(), policy,
      {.elements = 256U, .bytes = 8192U, .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  if (result) {
    VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 4.0) < 1.0e-12);
    VSS_CHECK(std::abs(result.value().covariance.at(1U, 1U) - 2.5) < 1.0e-12);
    VSS_CHECK(result.value().classification ==
              CovarianceKind::positive_definite);
  }

  Matrix smaller = matrix({{1.0, 0.0}, {0.0, 1.0}});
  Matrix larger = matrix({{3.0, 0.0}, {0.0, 3.0}});
  const auto signedResult = subtractKnownErrorCovariance(
      smaller.view(), larger.view(), policy,
      {.elements = 256U, .bytes = 8192U, .evaluations = 0U});
  VSS_CHECK(signedResult.hasValue());
  if (signedResult) {
    VSS_CHECK(signedResult.value().covariance.at(0U, 0U) == -2.0);
    VSS_CHECK(signedResult.value().classification ==
              CovarianceKind::indefinite);
  }

  Matrix invalidObserved = matrix({{1.0, 0.0}, {0.0, -0.1}});
  const auto invalidInput = subtractKnownErrorCovariance(
      invalidObserved.view(), known.view(), policy,
      {.elements = 256U, .bytes = 8192U, .evaluations = 0U});
  VSS_CHECK(!invalidInput.hasValue());
  VSS_CHECK(invalidInput.status().code() == ErrorCode::NotPositiveSemidefinite);
}
} // namespace

int main() {
  independentAndGroupedContributionsAreAssembled();
  conditionedEvaluationHonorsItsBudget();
  knownErrorSubtractionRetainsSignedResults();
}

// VSS-MATH-TRACEABILITY: C099 CASE-C099.block.independent
// VSS-MATH-TRACEABILITY: C099 CASE-C099.conditioned
// VSS-MATH-TRACEABILITY: C099 CASE-C099.grouped
// VSS-MATH-TRACEABILITY: C099 CASE-C099.intrinsic.independent
// VSS-MATH-TRACEABILITY: C099 CASE-C099.subtract.signed
