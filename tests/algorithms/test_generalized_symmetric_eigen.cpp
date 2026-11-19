#include "Test.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {
using vss::math::ErrorCode;
using vss::math::Index;
using vss::math::Limits;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::SolvePolicy;
using vss::math::linear::GeneralizedEigen;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index n = rows.size();
  const Index m = n == 0U ? 0U : rows.begin()->size();
  Matrix result(n, m);
  Index i = 0U;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == m);
    Index j = 0U;
    for (const Scalar value : row) {
      result.at(i, j++) = value;
    }
    ++i;
  }
  return result;
}

SolvePolicy policy(Index cap = 64U) {
  SolvePolicy result{};
  result.iteration.max_iterations = cap;
  return result;
}

Limits generousLimits() {
  return {.elements = 1'000'000U,
          .bytes = std::size_t{64U} * 1024U * 1024U,
          .evaluations = 0U};
}

auto solve(const Matrix& a, const Matrix& b, SolvePolicy p = policy(),
           Limits limits = generousLimits()) {
  return vss::math::linear::factorGeneralizedSymmetric(a.view(), b.view(), p,
                                                       limits);
}

template <class T> void checkFailure(const T& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void checkNear(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-10) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
}

void diagonalPencilHasAscendingQuotientsAndBOrthonormalVectors() {
  const Matrix a = matrix({{2.0, 0.0}, {0.0, 12.0}});
  const Matrix b = matrix({{1.0, 0.0}, {0.0, 3.0}});
  const auto result = solve(a, b);
  VSS_CHECK(result.hasValue());
  const GeneralizedEigen& eigen = result.value();
  VSS_CHECK(eigen.values.size() == 2U);
  VSS_CHECK(eigen.vectors.rows() == 2U && eigen.vectors.columns() == 2U);
  VSS_CHECK(eigen.residual_norms.size() == 2U);
  checkNear(eigen.values.at(0U), 2.0);
  checkNear(eigen.values.at(1U), 4.0);
  for (Index i = 0U; i < 2U; ++i) {
    VSS_CHECK(eigen.residual_norms.at(i) <= 1.0e-12);
  }
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      Scalar product = 0.0;
      for (Index row = 0U; row < 2U; ++row) {
        for (Index col = 0U; col < 2U; ++col) {
          product += eigen.vectors.at(row, i) * b.at(row, col) *
                     eigen.vectors.at(col, j);
        }
      }
      checkNear(product, i == j ? 1.0 : 0.0);
    }
  }
}

void nondiagonalMetricProducesAnalyticSpectrum() {
  const Matrix a = matrix({{4.0, 2.0}, {2.0, 5.5}});
  const Matrix b = matrix({{2.0, 1.0}, {1.0, 2.0}});
  const auto result = solve(a, b);
  VSS_CHECK(result.hasValue());
  checkNear(result.value().values.at(0U), 2.0);
  checkNear(result.value().values.at(1U), 3.0);
  VSS_CHECK(result.value().residual_norms.at(0U) < 1.0e-11);
  VSS_CHECK(result.value().residual_norms.at(1U) < 1.0e-11);
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      Scalar product = 0.0;
      for (Index row = 0U; row < 2U; ++row) {
        for (Index col = 0U; col < 2U; ++col) {
          product += result.value().vectors.at(row, i) * b.at(row, col) *
                     result.value().vectors.at(col, j);
        }
      }
      checkNear(product, i == j ? 1.0 : 0.0);
    }
  }
}

void independentPowerOfTwoScalingPreservesPencilRatioAndMetricNorm() {
  const Scalar aScale = std::scalbn(1.0, 900);
  const Scalar bScale = std::scalbn(1.0, 899);
  const Matrix a = matrix({{aScale, 0.0}, {0.0, aScale}});
  const Matrix b = matrix({{bScale, 0.0}, {0.0, bScale}});
  const auto result = solve(a, b);
  VSS_CHECK(result.hasValue());
  checkNear(result.value().values.at(0U), 2.0);
  checkNear(result.value().values.at(1U), 2.0);
  for (Index col = 0U; col < 2U; ++col) {
    Scalar normSquared = 0.0;
    for (Index row = 0U; row < 2U; ++row) {
      normSquared += bScale * result.value().vectors.at(row, col) *
                     result.value().vectors.at(row, col);
    }
    checkNear(normSquared, 1.0, 1.0e-10);
  }
}

void indefiniteAAndRepeatedEigenvalueClusterAreSupported() {
  const Matrix b = matrix({{2.0, 0.0}, {0.0, 1.0}});
  const Matrix a = matrix({{-4.0, 0.0}, {0.0, -2.0}});
  const auto result = solve(a, b);
  VSS_CHECK(result.hasValue());
  checkNear(result.value().values.at(0U), -2.0);
  checkNear(result.value().values.at(1U), -2.0);
}

void repeatedEigenvalueClusterIsVerifiedByItsProjector() {
  const Matrix a = matrix({{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 4.0}});
  const Matrix b = matrix({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const auto result = solve(a, b);
  VSS_CHECK(result.hasValue());
  checkNear(result.value().values.at(0U), 2.0);
  checkNear(result.value().values.at(1U), 2.0);
  for (Index row = 0U; row < 3U; ++row) {
    for (Index column = 0U; column < 3U; ++column) {
      Scalar projector = 0.0;
      for (Index eigen = 0U; eigen < 2U; ++eigen) {
        projector +=
            result.value().vectors.at(row, eigen) *
            result.value()
                .vectors.at( // NOLINT(readability-suspicious-call-argument):
                             // Projector construction reads a
                             // basis row from the same column.
                    column, eigen);
      }
      checkNear(projector, row == column && row < 2U ? 1.0 : 0.0);
    }
  }
}

void rejectsNonPositiveAndNumericallyRankDeficientMetric() {
  checkFailure(solve(matrix({{1.0}}), matrix({{0.0}})),
               ErrorCode::NotPositiveDefinite);
  checkFailure(
      solve(matrix({{1.0, 0.0}, {0.0, 1.0}}), matrix({{1.0, 2.0}, {2.0, 1.0}})),
      ErrorCode::NotPositiveDefinite);
  checkFailure(solve(matrix({{1.0}}), matrix({{1.0, 0.0}, {0.0, -1.0}})),
               ErrorCode::DimensionMismatch);
  SolvePolicy permissiveRank = policy();
  permissiveRank.rank = vss::math::RankPolicy::minimum_norm;
  checkFailure(solve(matrix({{1.0, 0.0}, {0.0, 1.0}}),
                     matrix({{1.0, 0.0}, {0.0, 0.0}}), permissiveRank),
               ErrorCode::NotPositiveDefinite);
  SolvePolicy p = policy();
  p.rank_rel = 1.0e-6;
  checkFailure(solve(matrix({{1.0, 0.0}, {0.0, 1.0}}),
                     matrix({{1.0, 0.0}, {0.0, 1.0e-8}}), p),
               ErrorCode::RankDeficient);
  p = policy();
  p.rank_abs = 1.0e-6;
  checkFailure(solve(matrix({{1.0, 0.0}, {0.0, 1.0}}),
                     matrix({{1.0, 0.0}, {0.0, 1.0e-6}}), p),
               ErrorCode::RankDeficient);
}

void distinguishesMetricConditionFloorFromRankCutoff() {
  const Matrix a = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const Matrix b = matrix({{1.0, 0.0}, {0.0, 1.0e-8}});
  SolvePolicy p = policy();
  p.reciprocal_condition_floor = 1.0e-6;
  checkFailure(solve(a, b, p), ErrorCode::IllConditioned);
  p.reciprocal_condition_floor = 0.5;
  const auto boundary = solve(matrix({{1.0, 0.0}, {0.0, 1.0}}),
                              matrix({{1.0, 0.0}, {0.0, 0.5}}), p);
  VSS_CHECK(boundary.hasValue());
}

void validatesShapeSymmetryFinitenessPolicyAndResourceCeilings() {
  checkFailure(solve(matrix({{1.0, 2.0}}), matrix({{1.0}})),
               ErrorCode::DimensionMismatch);
  const auto toleratedSkew = solve(matrix({{1.0, 0.0}, {0.0, 1.0}}),
                                   matrix({{2.0, 1.0 + 1.0e-14}, {1.0, 2.0}}));
  VSS_CHECK(toleratedSkew.hasValue());
  checkNear(toleratedSkew.value().values.at(0U), 1.0 / 3.0);
  checkNear(toleratedSkew.value().values.at(1U), 1.0);
  checkFailure(
      solve(matrix({{1.0, 0.2}, {0.0, 1.0}}), matrix({{1.0, 0.0}, {0.0, 1.0}})),
      ErrorCode::NotSymmetric);
  checkFailure(
      solve(matrix({{1.0, 0.0}, {0.0, 1.0}}), matrix({{1.0, 0.2}, {0.0, 1.0}})),
      ErrorCode::NotSymmetric);
  checkFailure(solve(matrix({{std::numeric_limits<Scalar>::infinity()}}),
                     matrix({{1.0}})),
               ErrorCode::NonFiniteInput);
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Scalar smallest = std::numeric_limits<Scalar>::denorm_min();
  checkFailure(solve(matrix({{maximum}}), matrix({{smallest}})),
               ErrorCode::NonFiniteResult);
  SolvePolicy invalid = policy();
  invalid.rank_rel = 1.0;
  checkFailure(solve(matrix({{1.0}}), matrix({{1.0}}), invalid),
               ErrorCode::InvalidTolerance);
  checkFailure(solve(matrix({{1.0}}), matrix({{1.0}}), policy(),
                     Limits{.elements = 1U, .bytes = 1U, .evaluations = 0U}),
               ErrorCode::ResourceLimit);
}

void spectralIterationCapReturnsNoConvergence() {
  const Matrix a = matrix({{1.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 4.0}});
  const Matrix b = matrix({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  checkFailure(solve(a, b, policy(1U)), ErrorCode::NoConvergence);
}

void emptyPencilAndZeroCapDiagonalPencilWork() {
  const auto empty = solve(Matrix{}, Matrix{});
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().values.empty());
  VSS_CHECK(empty.value().vectors.rows() == 0U);
  VSS_CHECK(empty.value().vectors.columns() == 0U);

  const auto diagonal = solve(matrix({{2.0, 0.0}, {0.0, 6.0}}),
                              matrix({{1.0, 0.0}, {0.0, 3.0}}), policy(0U));
  VSS_CHECK(diagonal.hasValue());
  checkNear(diagonal.value().values.at(0U), 2.0);
  checkNear(diagonal.value().values.at(1U), 2.0);
  checkFailure(solve(matrix({{2.0, 1.0}, {1.0, 2.0}}),
                     matrix({{2.0, 1.0}, {1.0, 2.0}}), policy(0U)),
               ErrorCode::NoConvergence);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  diagonalPencilHasAscendingQuotientsAndBOrthonormalVectors();
  nondiagonalMetricProducesAnalyticSpectrum();
  independentPowerOfTwoScalingPreservesPencilRatioAndMetricNorm();
  indefiniteAAndRepeatedEigenvalueClusterAreSupported();
  repeatedEigenvalueClusterIsVerifiedByItsProjector();
  rejectsNonPositiveAndNumericallyRankDeficientMetric();
  distinguishesMetricConditionFloorFromRankCutoff();
  validatesShapeSymmetryFinitenessPolicyAndResourceCeilings();
  spectralIterationCapReturnsNoConvergence();
  emptyPencilAndZeroCapDiagonalPencilWork();
  return 0;
}

// VSS-MATH-TRACEABILITY: GEN-01 GEN-02 GEN-03 GEN-04
// VSS-MATH-TRACEABILITY: GEN-05 GEN-06 GEN-07 GEN-08
