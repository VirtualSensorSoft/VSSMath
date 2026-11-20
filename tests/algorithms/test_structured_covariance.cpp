#include "Test.hpp"

#include <array>
#include <cmath>
#include <span>
#include <vss/math/covariance/Structured.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::covariance;
using namespace vss::math::linear;

constexpr Limits budget{
    .elements = 4096U, .bytes = 32768U, .evaluations = 256U};

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index n = rows.size();
  const Index m = n ? rows.begin()->size() : 0U;
  Matrix out(n, m, 0.0);
  Index i = 0;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == m);
    Index j = 0;
    for (Scalar x : row)
      out.at(i, j++) = x;
    ++i;
  }
  return out;
}

void diagonalLowRankAndRankAwareSolve() {
  const Vector diagonal(std::array<Scalar, 2>{1.0, 2.0});
  const Matrix u = matrix({{1.0}, {1.0}});
  StructuredSpec spec{
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
  StructuredSpec diagonal_spec{
      StructureKind::diagonal_low_rank, diagonal.view(), {}, {}, {}, {}};
  auto diagonal_covariance =
      structuredCovariance(diagonal_spec, iterative, budget);
  VSS_CHECK(diagonal_covariance.hasValue());
  const Matrix diagonal_rhs = matrix({{2.0}, {6.0}});
  auto pcg = solveStructuredCovariance(
      diagonal_covariance.value(), diagonal_rhs.view(),
      StructuredSolveKind::iterative, iterative, budget);
  VSS_CHECK(pcg.hasValue());
  VSS_CHECK(std::abs(pcg.value().parameters.at(0, 0) - 2.0) < 1e-11);
  VSS_CHECK(std::abs(pcg.value().parameters.at(1, 0) - 3.0) < 1e-11);
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

  const Vector bad_lags(std::array<Scalar, 2>{1.0, 2.0});
  auto bad = structuredCovariance(
      StructuredSpec{StructureKind::toeplitz, bad_lags.view(), {}, {}, {}, {}},
      SolvePolicy{}, budget);
  VSS_CHECK(!bad.hasValue());
  VSS_CHECK(bad.status().code() == ErrorCode::NotPositiveSemidefinite);
  const Vector lags(std::array<Scalar, 2>{2.0, 0.5});
  auto toeplitz = structuredCovariance(
      StructuredSpec{StructureKind::toeplitz, lags.view(), {}, {}, {}, {}},
      SolvePolicy{}, budget);
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
} // namespace

int main() {
  diagonalLowRankAndRankAwareSolve();
  blockAndToeplitzValidation();
  kroneckerAndGaussianActions();
  return 0;
}
