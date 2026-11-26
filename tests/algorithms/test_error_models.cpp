#include "Test.hpp"

#include <array>
#include <cmath>
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
      {1U, firstCoordinate, first.view()},
      {2U, secondCoordinate, second.view()},
  }};
  Matrix crossCovariance = matrix({{0.25}});
  const std::array<SourceCrossBlock, 1> cross{{
      {1U, 2U, crossCovariance.view()},
  }};
  const Vector diagonal(std::array<Scalar, 2>{0.04, 0.04});
  const Vector state;
  const ErrorCovarianceSpec spec{.evaluate = {},
                                 .diagonal = diagonal.view(),
                                 .loading = identity.view(),
                                 .sources = sources,
                                 .cross = cross,
                                 .dependence = DependencePolicy::exact_joint};
  const auto result =
      assembleErrorCovariance(spec, state.view(), 0.0, {512U, 16384U, 10U});
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
      {1U, firstCoordinate, firstIndependentError.view()},
      {2U, secondCoordinate, secondIndependentError.view()},
  }};
  const ErrorCovarianceSpec independent{
      .loading = identity.view(),
      .sources = independentSources,
      .dependence = DependencePolicy::declared_independent_blocks};
  const auto independentResult = assembleErrorCovariance(
      independent, state.view(), 0.0, {512U, 16384U, 10U});
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
      .evaluate = [&calls](ConstVectorView state, Scalar time) {
        ++calls;
        if (state.size() != 1U || state.at(0U) != 2.0 || time != 5.0) {
          return Result<Matrix>::failure(ErrorCode::InvalidDomain,
                                         "unexpected conditioning input");
        }
        return Result<Matrix>::success(matrix({{7.0}}));
      }};
  const Vector state(std::array<Scalar, 1>{2.0});
  const auto result =
      assembleErrorCovariance(spec, state.view(), 5.0, {64U, 4096U, 1U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(calls == 1U);
  if (result) {
    VSS_CHECK(result.value().covariance.at(0U, 0U) == 7.0);
  }
  const auto denied =
      assembleErrorCovariance(spec, state.view(), 5.0, {64U, 4096U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(calls == 1U);
}

void knownErrorSubtractionRetainsSignedResults() {
  Matrix observed = matrix({{5.0, 0.5}, {0.5, 3.0}});
  Matrix known = matrix({{1.0, 0.5}, {0.5, 0.5}});
  SolvePolicy policy{{1.0e-12, 1.0e-12, 4096U, 100000U, std::nullopt},
                     0.0,
                     1.0e-12,
                     1.0e-10,
                     RankPolicy::minimum_norm,
                     0U};
  const auto result = subtractKnownErrorCovariance(
      observed.view(), known.view(), policy, {256U, 8192U, 0U});
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
      smaller.view(), larger.view(), policy, {256U, 8192U, 0U});
  VSS_CHECK(signedResult.hasValue());
  if (signedResult) {
    VSS_CHECK(signedResult.value().covariance.at(0U, 0U) == -2.0);
    VSS_CHECK(signedResult.value().classification ==
              CovarianceKind::indefinite);
  }

  Matrix invalidObserved = matrix({{1.0, 0.0}, {0.0, -0.1}});
  const auto invalidInput = subtractKnownErrorCovariance(
      invalidObserved.view(), known.view(), policy, {256U, 8192U, 0U});
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
