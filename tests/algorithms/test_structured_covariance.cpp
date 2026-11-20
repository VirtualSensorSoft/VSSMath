#include "Test.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Structured.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::covariance;
using namespace vss::math::linear;

constexpr Limits budget{
    .elements = 4096U, .bytes = 32768U, .evaluations = 256U};

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index n = rows.size();
  const Index m = n != 0 ? rows.begin()->size() : 0U;
  Matrix out(n, m, 0.0);
  Index i = 0;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == m);
    Index j = 0;
    for (const Scalar x : row) {
      out.at(i, j++) = x;
    }
    ++i;
  }
  return out;
}

void diagonalLowRankAndRankAwareSolve() {
  const Vector diagonal(std::array<Scalar, 2>{1.0, 2.0});
  const Matrix u = matrix({{1.0}, {1.0}});
  const StructuredSpec spec{
      StructureKind::diagonal_low_rank, diagonal.view(), u.view(), {}, {}, {}};
  auto covariance = structuredCovariance(spec, SolvePolicy{}, budget);
  VSS_CHECK(covariance.hasValue());
  const Vector x(std::array<Scalar, 2>{2.0, 3.0});
  auto applied =
      applyStructuredCovariance(covariance.value(), x.view(), budget);
  VSS_CHECK(applied.hasValue());
  VSS_CHECK(applied.value().at(0) == 7.0 && applied.value().at(1) == 11.0);

  const Matrix rhs = matrix({{7.0}, {11.0}});
  SolvePolicy policy{};
  policy.rank = RankPolicy::reject;
  policy.iteration.max_iterations = 64;
  auto solved = solveStructuredCovariance(covariance.value(), rhs.view(),
                                          StructuredSolveKind::factorized,
                                          policy, budget);
  VSS_CHECK(solved.hasValue());
  VSS_CHECK(std::abs(solved.value().parameters.at(0, 0) - 2.0) < 1e-11);
  VSS_CHECK(std::abs(solved.value().parameters.at(1, 0) - 3.0) < 1e-11);

  SolvePolicy iterative{};
  iterative.rank = RankPolicy::reject;
  iterative.iteration = IterPolicy{.abs = 1e-12,
                                   .rel = 1e-12,
                                   .max_iterations = 8U,
                                   .max_evaluations = 16U,
                                   .bracket = std::nullopt};
  const auto pcg = solveStructuredCovariance(covariance.value(), rhs.view(),
                                             StructuredSolveKind::iterative,
                                             iterative, budget);
  VSS_CHECK(pcg.hasValue());
  VSS_CHECK(std::abs(pcg.value().parameters.at(0, 0) - 2.0) < 1e-11);
  VSS_CHECK(std::abs(pcg.value().parameters.at(1, 0) - 3.0) < 1e-11);
  const Matrix multiple = matrix({{7, 7}, {11, 11}});
  auto exhausted = solveStructuredCovariance(
      covariance.value(), multiple.view(), StructuredSolveKind::iterative,
      iterative,
      {.elements = budget.elements, .bytes = budget.bytes, .evaluations = 8});
  VSS_CHECK(!exhausted.hasValue() &&
            exhausted.status().code() == ErrorCode::ResourceLimit);

  iterative.rank_rel = 0.05;
  iterative.iteration.max_iterations = 1;
  auto unresolved = solveStructuredCovariance(covariance.value(), rhs.view(),
                                              StructuredSolveKind::iterative,
                                              iterative, budget);
  VSS_CHECK(!unresolved.hasValue() &&
            unresolved.status().code() == ErrorCode::NoConvergence);
}

void blockAndToeplitzValidation() {
  const std::array<Matrix, 2> blocks{matrix({{2.0}}), matrix({{0.0}})};
  auto block = structuredCovariance(
      StructuredSpec{StructureKind::block, Vector{}.view(), {}, blocks, {}, {}},
      SolvePolicy{}, budget);
  VSS_CHECK(block.hasValue() && block.value().dimension == 2U);
  const Vector x(std::array<Scalar, 2>{3.0, 4.0});
  auto y = applyStructuredCovariance(block.value(), x.view(), budget);
  VSS_CHECK(y.hasValue() && y.value().at(0) == 6.0 && y.value().at(1) == 0.0);

  SolvePolicy spectral{};
  spectral.iteration.max_iterations = 64;
  const Vector bad_lags(std::array<Scalar, 2>{1.0, 2.0});
  auto bad = structuredCovariance(
      StructuredSpec{StructureKind::toeplitz, bad_lags.view(), {}, {}, {}, {}},
      spectral, budget);
  VSS_CHECK(!bad.hasValue());
  VSS_CHECK(bad.status().code() == ErrorCode::NotPositiveSemidefinite);
  const Vector lags(std::array<Scalar, 2>{2.0, 0.5});
  auto toeplitz = structuredCovariance(
      StructuredSpec{StructureKind::toeplitz, lags.view(), {}, {}, {}, {}},
      spectral, budget);
  VSS_CHECK(toeplitz.hasValue());
  auto ty = applyStructuredCovariance(toeplitz.value(), x.view(), budget);
  VSS_CHECK(ty.hasValue());
  VSS_CHECK(ty.value().at(0) == 8.0 && ty.value().at(1) == 9.5);
}

void kroneckerAndGaussianActions() {
  const std::array<Matrix, 2> factors{matrix({{1.0, 0.0}, {0.0, 2.0}}),
                                      matrix({{3.0}})};
  auto kron = kroneckerCovariance(factors, SolvePolicy{}, budget);
  VSS_CHECK(kron.hasValue() && kron.value().dimension == 2U);
  const Vector x(std::array<Scalar, 2>{2.0, 3.0});
  auto y = applyStructuredCovariance(kron.value(), x.view(), budget);
  VSS_CHECK(y.hasValue() && y.value().at(0) == 6.0 && y.value().at(1) == 18.0);

  const Matrix points = matrix({{0.0}, {0.0}});
  auto gaussian =
      structuredCovariance(StructuredSpec{StructureKind::gaussian_kernel,
                                          Vector{}.view(),
                                          points.view(),
                                          {},
                                          2.0,
                                          1.0},
                           SolvePolicy{}, budget);
  VSS_CHECK(gaussian.hasValue());
  auto gy = applyStructuredCovariance(gaussian.value(), x.view(), budget);
  VSS_CHECK(gy.hasValue() && gy.value().at(0) == 10.0 &&
            gy.value().at(1) == 10.0);
}
void factorizedKroneckerUsesGlobalCutoffAndOriginalResidual() {
  const std::array factors{matrix({{1, 0}, {0, 3}}), matrix({{5, 0}, {0, 7}})};
  auto kron = kroneckerCovariance(factors, {}, budget);
  VSS_CHECK(kron.hasValue());
  const Vector ones(4, 1);
  auto action = applyStructuredCovariance(kron.value(), ones.view(), budget);
  VSS_CHECK(action.hasValue());
  const auto expected = std::array{5.0, 7.0, 15.0, 21.0};
  for (Index i = 0; i < expected.size(); ++i) {
    VSS_CHECK(action.value().at(i) == expected.at(i));
  }
  const auto rhs = matrix({{5}, {14}, {45}, {84}});
  const auto solve = solveStructuredCovariance(
      kron.value(), rhs.view(), StructuredSolveKind::factorized, {}, budget);
  VSS_CHECK(solve.hasValue() && solve.value().residual_norms.at(0) == 0);
  for (Index i = 0; i < 4; ++i) {
    VSS_CHECK(solve.value().parameters.at(i, 0) == static_cast<Scalar>(i + 1));
  }

  const std::array singular_factors{matrix({{2, 0}, {0, 0}}), factors.at(1)};
  auto singular = kroneckerCovariance(singular_factors, {}, budget);
  VSS_CHECK(singular.hasValue());
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.rank_abs = 10;
  const auto cutoff = solveStructuredCovariance(singular.value(), rhs.view(),
                                                StructuredSolveKind::factorized,
                                                policy, budget);
  VSS_CHECK(cutoff.hasValue() && cutoff.value().rank == 1);
  VSS_CHECK(cutoff.value().rank_cutoff == 10);
  VSS_CHECK(cutoff.value().parameters.at(1, 0) == 1);
  VSS_CHECK(cutoff.value().identified_basis.has_value() &&
            cutoff.value().identified_basis.value().at(1, 0) == 1);
  VSS_CHECK(std::abs(cutoff.value().residual_norms.at(0) - std::sqrt(9106.0)) <
            1e-12);
  const auto iterative =
      solveStructuredCovariance(singular.value(), rhs.view(),
                                StructuredSolveKind::iterative, policy, budget);
  VSS_CHECK(!iterative.hasValue() &&
            iterative.status().code() == ErrorCode::NoConvergence);
}

void projectedToeplitzAndKernelSolveWithoutDenseCovariance() {
  SolvePolicy policy{};
  policy.iteration.max_iterations = 64;
  const Vector lags(std::array{2.0, 1.0, 0.5});
  const auto toeplitz = structuredCovariance(
      {StructureKind::toeplitz, lags.view(), {}, {}, {}, {}}, policy, budget);
  VSS_CHECK(toeplitz.hasValue());
  const auto rhs = matrix({{3.5}, {4}, {3.5}});
  const auto solved = solveStructuredCovariance(toeplitz.value(), rhs.view(),
                                                StructuredSolveKind::factorized,
                                                policy, budget);
  VSS_CHECK(solved.hasValue());
  for (Index i = 0; i < 3; ++i) {
    VSS_CHECK(std::abs(solved.value().parameters.at(i, 0) - 1) < 1e-12);
  }

  const Vector empty;
  const Matrix points = matrix({{0}, {1}});
  const auto kernel = structuredCovariance(
      {StructureKind::gaussian_kernel, empty.view(), points.view(), {}, 2, 1},
      policy, budget);
  VSS_CHECK(kernel.hasValue());
  const Scalar observation = 2 + 2 * std::exp(-0.5);
  const Matrix expected(2, 1, observation);
  const auto fitted = solveStructuredCovariance(kernel.value(), expected.view(),
                                                StructuredSolveKind::factorized,
                                                policy, budget);
  VSS_CHECK(fitted.hasValue());
  VSS_CHECK(std::abs(fitted.value().parameters.at(0, 0) - 1) < 1e-12);
  VSS_CHECK(std::abs(fitted.value().parameters.at(1, 0) - 1) < 1e-12);

  const auto denied = structuredCovariance(
      {StructureKind::gaussian_kernel, empty.view(), points.view(), {}, 2, 1},
      policy,
      {.elements = 1,
       .bytes = budget.bytes,
       .evaluations = budget.evaluations});
  VSS_CHECK(!denied.hasValue() &&
            denied.status().code() == ErrorCode::ResourceLimit);
  auto mutated = kernel.value();
  mutated.diagonal_or_lags = Vector(2);
  const auto inactive =
      applyStructuredCovariance(mutated, Vector(2).view(), budget);
  VSS_CHECK(!inactive.hasValue() &&
            inactive.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() {
  diagonalLowRankAndRankAwareSolve();
  blockAndToeplitzValidation();
  kroneckerAndGaussianActions();
  factorizedKroneckerUsesGlobalCutoffAndOriginalResidual();
  projectedToeplitzAndKernelSolveWithoutDenseCovariance();
  return 0;
}
