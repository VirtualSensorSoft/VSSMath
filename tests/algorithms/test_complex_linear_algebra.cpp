#include "Test.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/linear/Complex.hpp>

namespace {

using vss::math::Complex;
using vss::math::ErrorCode;
using vss::math::Index;
using vss::math::Limits;
using vss::math::linear::ComplexMatrix;
using vss::math::linear::ComplexMatrixView;
using vss::math::linear::conjugateTranspose;
using vss::math::linear::eigenHermitian;
using vss::math::linear::factorComplexSVD;
using vss::math::linear::multiply;
using vss::math::linear::solveComplex;

vss::math::SolvePolicy
solvePolicy(vss::math::RankPolicy rank = vss::math::RankPolicy::reject) {
  return {.iteration = {.abs = 1.0e-12,
                        .rel = 1.0e-12,
                        .max_iterations = 128U,
                        .max_evaluations = 0U,
                        .bracket = std::nullopt},
          .rank_abs = 0.0,
          .rank_rel = 1.0e-12,
          .reciprocal_condition_floor = 0.0,
          .rank = rank,
          .restart = 0U};
}

Limits generousLimits() {
  return {
      .elements = 1000000U, .bytes = 128U * 1024U * 1024U, .evaluations = 0U};
}

void checkNear(Complex actual, Complex expected, double tolerance = 1.0e-13) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
}

ComplexMatrixView view(std::span<const Complex> values, Index rows,
                       Index columns, Index rowStride, Index columnStride) {
  return {values, rows, columns, rowStride, columnStride};
}

void complexProductsUseBothComponentsAndCorrectShapes() {
  const std::array<Complex, 1> scalarLeft{{{1.0, 1.0}}};
  const std::array<Complex, 1> scalarRight{{{2.0, -1.0}}};
  const auto scalar = multiply(
      view(scalarLeft, 1U, 1U, 1U, 1U), view(scalarRight, 1U, 1U, 1U, 1U),
      {.elements = 8U, .bytes = 256U, .evaluations = 0U});
  VSS_CHECK(scalar.hasValue());
  if (scalar) {
    VSS_CHECK(scalar.value().rows == 1U && scalar.value().columns == 1U);
    checkNear(scalar.value().values.view()[0U], {3.0, 1.0});
  }

  const std::array<Complex, 4> left{
      {{1.0, 2.0}, {-2.0, 1.0}, {0.5, -1.0}, {3.0, 0.25}}};
  const std::array<Complex, 4> right{
      {{-1.0, 0.5}, {2.0, 1.0}, {1.5, -2.0}, {-0.25, 3.0}}};
  const auto product =
      multiply(view(left, 2U, 2U, 2U, 1U), view(right, 2U, 2U, 2U, 1U),
               {.elements = 32U, .bytes = 1024U, .evaluations = 0U});
  VSS_CHECK(product.hasValue());
  if (product) {
    checkNear(product.value().values.view()[0U], Complex{-3.0, 4.0});
    checkNear(product.value().values.view()[1U], Complex{-2.5, -1.25});
    checkNear(product.value().values.view()[2U], Complex{5.0, -4.375});
    checkNear(product.value().values.view()[3U], Complex{0.5, 7.4375});
  }

  const Complex huge{1.0e308, 1.0e308};
  const std::array<Complex, 2> cancellingLeft{{huge, huge}};
  const std::array<Complex, 2> cancellingRight{{huge, -huge}};
  const auto cancellation =
      multiply(view(cancellingLeft, 1U, 2U, 2U, 1U),
               view(cancellingRight, 2U, 1U, 1U, 1U), generousLimits());
  VSS_CHECK(cancellation.hasValue());
  if (cancellation) {
    checkNear(cancellation.value().values.view()[0U], {}, 0.0);
  }
  const std::array<Complex, 1> overflowLeft{{{1.0e308, 0.0}}};
  const std::array<Complex, 1> overflowRight{{{1.0e308, 0.0}}};
  const auto overflow =
      multiply(view(overflowLeft, 1U, 1U, 1U, 1U),
               view(overflowRight, 1U, 1U, 1U, 1U), generousLimits());
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);

  const auto oneShort = multiply(
      view(scalarLeft, 1U, 1U, 1U, 1U), view(scalarRight, 1U, 1U, 1U, 1U),
      {.elements = 1U, .bytes = 2U * sizeof(Complex) - 1U, .evaluations = 0U});
  VSS_CHECK(!oneShort.hasValue());
  VSS_CHECK(oneShort.status().code() == ErrorCode::ResourceLimit);
}

void stridedViewsAndZeroContractedExtentAreSupported() {
  const std::array<Complex, 5> backing{
      {{1.0, 1.0}, {2.0, 0.0}, {99.0, 0.0}, {3.0, -1.0}, {4.0, 2.0}}};
  const auto strided =
      multiply(view(backing, 2U, 2U, 3U, 1U), view(backing, 2U, 2U, 3U, 1U),
               {.elements = 32U, .bytes = 1024U, .evaluations = 0U});
  VSS_CHECK(strided.hasValue());
  if (strided) {
    checkNear(strided.value().values.view()[0U], {6.0, 0.0});
  }

  const std::span<const Complex> empty;
  const auto zero =
      multiply(view(empty, 2U, 0U, 0U, 1U), view(empty, 0U, 3U, 1U, 1U),
               {.elements = 16U, .bytes = 512U, .evaluations = 0U});
  VSS_CHECK(zero.hasValue());
  if (zero) {
    VSS_CHECK(zero.value().rows == 2U && zero.value().columns == 3U);
    for (const Complex value : zero.value().values.view()) {
      VSS_CHECK(value == Complex{});
    }
  }
}

void conjugateTransposeConjugatesAndIsAnInvolution() {
  const std::array<Complex, 2> values{{{1.0, -1.0}, {2.0, 1.0}}};
  const auto adjoint =
      conjugateTranspose(view(values, 2U, 1U, 1U, 1U),
                         {.elements = 16U, .bytes = 512U, .evaluations = 0U});
  VSS_CHECK(adjoint.hasValue());
  if (!adjoint) {
    return;
  }
  VSS_CHECK(adjoint.value().rows == 1U && adjoint.value().columns == 2U);
  checkNear(adjoint.value().values.view()[0U], {1.0, 1.0});
  checkNear(adjoint.value().values.view()[1U], {2.0, -1.0});
  const auto restored =
      conjugateTranspose(view(adjoint.value().values.view(), 1U, 2U, 2U, 1U),
                         {.elements = 16U, .bytes = 512U, .evaluations = 0U});
  VSS_CHECK(restored.hasValue());
  if (restored) {
    checkNear(restored.value().values.view()[0U], values[0U]);
    checkNear(restored.value().values.view()[1U], values[1U]);
  }
  const std::span<const Complex> empty;
  const auto emptyAdjoint =
      conjugateTranspose(view(empty, 0U, 3U, 0U, 1U), generousLimits());
  VSS_CHECK(emptyAdjoint.hasValue());
  if (emptyAdjoint) {
    VSS_CHECK(emptyAdjoint.value().rows == 3U &&
              emptyAdjoint.value().columns == 0U);
  }
}

void invalidInputsAndInsufficientLimitsFailWithoutResults() {
  const std::array<Complex, 1> value{{{1.0, 0.0}}};
  const auto mismatch =
      multiply(view(value, 1U, 2U, 2U, 1U), view(value, 1U, 1U, 1U, 1U),
               {.elements = 8U, .bytes = 256U, .evaluations = 0U});
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);

  const auto overrun =
      conjugateTranspose(view(value, 2U, 1U, 1U, 1U),
                         {.elements = 8U, .bytes = 256U, .evaluations = 0U});
  VSS_CHECK(!overrun.hasValue());
  VSS_CHECK(overrun.status().code() == ErrorCode::DimensionMismatch);
  const auto offsetOverflow = conjugateTranspose(
      view(value, 3U, 1U, std::numeric_limits<Index>::max(), 1U),
      {.elements = 8U, .bytes = 256U, .evaluations = 0U});
  VSS_CHECK(!offsetOverflow.hasValue());
  VSS_CHECK(offsetOverflow.status().code() == ErrorCode::SizeOverflow);

  const auto limited =
      conjugateTranspose(view(value, 1U, 1U, 1U, 1U),
                         {.elements = 0U, .bytes = 0U, .evaluations = 0U});
  VSS_CHECK(!limited.hasValue());
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);

  const std::array<Complex, 1> nonfinite{
      {{std::numeric_limits<double>::quiet_NaN(), 0.0}}};
  const auto invalid =
      conjugateTranspose(view(nonfinite, 1U, 1U, 1U, 1U),
                         {.elements = 8U, .bytes = 256U, .evaluations = 0U});
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);
}

void complexSolveHandlesMultipleRightHandSidesAndRankPolicy() {
  const std::array<Complex, 4> coefficients{
      {{2.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}}};
  const std::array<Complex, 4> expected{
      {{1.0, 1.0}, {-1.0, 0.5}, {2.0, -1.0}, {0.25, 2.0}}};
  const std::array<Complex, 4> rhs{
      {{3.0, 4.0}, {-4.0, 1.25}, {5.0, -3.0}, {1.0, 5.0}}};
  const auto solution =
      solveComplex(view(coefficients, 2U, 2U, 2U, 1U),
                   view(rhs, 2U, 2U, 2U, 1U), solvePolicy(), generousLimits());
  VSS_CHECK(solution.hasValue());
  if (solution) {
    for (Index i = 0U; i < expected.size(); ++i) {
      checkNear(solution.value().values.view()[i], expected[i], 1.0e-12);
    }
  }

  const std::array<Complex, 4> singular{
      {{1.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}}};
  const std::array<Complex, 2> supported{{{3.0, 0.0}, {6.0, 0.0}}};
  const auto rejected = solveComplex(view(singular, 2U, 2U, 2U, 1U),
                                     view(supported, 2U, 1U, 1U, 1U),
                                     solvePolicy(), generousLimits());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::RankDeficient);
  const auto minimumNorm = solveComplex(
      view(singular, 2U, 2U, 2U, 1U), view(supported, 2U, 1U, 1U, 1U),
      solvePolicy(vss::math::RankPolicy::minimum_norm), generousLimits());
  VSS_CHECK(minimumNorm.hasValue());
  if (minimumNorm) {
    checkNear(minimumNorm.value().values.view()[0U], {1.5, 0.0}, 1.0e-10);
    checkNear(minimumNorm.value().values.view()[1U], {1.5, 0.0}, 1.0e-10);
  }
  const auto supportedSolve = solveComplex(
      view(singular, 2U, 2U, 2U, 1U), view(supported, 2U, 1U, 1U, 1U),
      solvePolicy(vss::math::RankPolicy::supported_subspace), generousLimits());
  VSS_CHECK(supportedSolve.hasValue());
  const std::array<Complex, 2> outside{{{3.0, 0.0}, {7.0, 0.0}}};
  const auto outsideRange = solveComplex(
      view(singular, 2U, 2U, 2U, 1U), view(outside, 2U, 1U, 1U, 1U),
      solvePolicy(vss::math::RankPolicy::supported_subspace), generousLimits());
  VSS_CHECK(!outsideRange.hasValue());
  VSS_CHECK(outsideRange.status().code() == ErrorCode::RankDeficient);

  const std::array<Complex, 4> illConditioned{
      {{1.0, 0.0}, {}, {}, {1.0e-8, 0.0}}};
  auto strictCondition = solvePolicy();
  strictCondition.reciprocal_condition_floor = 1.0e-6;
  const std::array<Complex, 2> illRhs{{{1.0, 0.0}, {1.0e-8, 0.0}}};
  const auto ill = solveComplex(view(illConditioned, 2U, 2U, 2U, 1U),
                                view(illRhs, 2U, 1U, 1U, 1U), strictCondition,
                                generousLimits());
  VSS_CHECK(!ill.hasValue());
  VSS_CHECK(ill.status().code() == ErrorCode::IllConditioned);
  const std::array<Complex, 1> empty{};
  const auto emptySolve =
      solveComplex(view(empty, 0U, 0U, 0U, 0U), view(empty, 0U, 2U, 0U, 1U),
                   solvePolicy(), generousLimits());
  VSS_CHECK(emptySolve.hasValue());
  if (emptySolve) {
    VSS_CHECK(emptySolve.value().rows == 0U &&
              emptySolve.value().columns == 2U);
  }

  const std::array<Complex, 1> hugeA{{{1.0e308, 0.0}}};
  const std::array<Complex, 1> hugeB{{{1.0e308, 0.0}}};
  const auto scaledSolve =
      solveComplex(view(hugeA, 1U, 1U, 1U, 1U), view(hugeB, 1U, 1U, 1U, 1U),
                   solvePolicy(), generousLimits());
  VSS_CHECK(scaledSolve.hasValue());
  if (scaledSolve) {
    checkNear(scaledSolve.value().values.view()[0U], {1.0, 0.0});
  }
}

void hermitianEigenUsesComplexPhaseAndChecksAdmissionSeparately() {
  const std::array<Complex, 4> matrix{
      {{3.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {-2.0, 0.0}}};
  const auto eigen = eigenHermitian(view(matrix, 2U, 2U, 2U, 1U), solvePolicy(),
                                    generousLimits());
  VSS_CHECK(eigen.hasValue());
  if (eigen) {
    const double root = std::sqrt(29.0);
    checkNear({eigen.value().values.at(0U), 0.0}, {(1.0 - root) / 2.0, 0.0},
              1.0e-10);
    checkNear({eigen.value().values.at(1U), 0.0}, {(1.0 + root) / 2.0, 0.0},
              1.0e-10);
    VSS_CHECK(eigen.value().vectors.rows == 2U &&
              eigen.value().vectors.columns == 2U);
    VSS_CHECK(eigen.value().residual_norms.at(0U) < 1.0e-10);
    VSS_CHECK(eigen.value().residual_norms.at(1U) < 1.0e-10);
  }
  auto nonHermitian = matrix;
  nonHermitian[2U] = {0.0, 0.8};
  const auto rejected = eigenHermitian(view(nonHermitian, 2U, 2U, 2U, 1U),
                                       solvePolicy(), generousLimits());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::NotSymmetric);
  const auto admitted = eigenHermitian(view(nonHermitian, 2U, 2U, 2U, 1U),
                                       {.absolute = 3.0, .relative = 0.0},
                                       solvePolicy(), generousLimits());
  VSS_CHECK(admitted.hasValue());
  auto tightSolve = solvePolicy();
  tightSolve.iteration.abs = 0.0;
  tightSolve.iteration.rel = 0.0;
  const auto stillRejected = eigenHermitian(view(nonHermitian, 2U, 2U, 2U, 1U),
                                            tightSolve, generousLimits());
  VSS_CHECK(!stillRejected.hasValue());
  VSS_CHECK(stillRejected.status().code() == ErrorCode::NotSymmetric);

  const std::array<Complex, 4> nominal{
      {{2.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {2.0, 0.0}}};
  const auto explicitTolerance = eigenHermitian(
      view(nominal, 2U, 2U, 2U, 1U), {.absolute = 0.0, .relative = 1.0e-14},
      solvePolicy(), generousLimits());
  VSS_CHECK(explicitTolerance.hasValue());
  if (explicitTolerance) {
    checkNear({explicitTolerance.value().values.at(0U), 0.0}, {1.0, 0.0});
    checkNear({explicitTolerance.value().values.at(1U), 0.0}, {3.0, 0.0});
  }

  const std::array<Complex, 9> identity{
      {{1.0, 0.0}, {}, {}, {}, {1.0, 0.0}, {}, {}, {}, {1.0, 0.0}}};
  const auto identityEigen = eigenHermitian(view(identity, 3U, 3U, 3U, 1U),
                                            solvePolicy(), generousLimits());
  VSS_CHECK(identityEigen.hasValue());
  if (identityEigen) {
    for (Index i = 0U; i < 3U; ++i) {
      checkNear({identityEigen.value().values.at(i), 0.0}, {1.0, 0.0});
      VSS_CHECK(identityEigen.value().residual_norms.at(i) < 1.0e-13);
    }
  }

  const auto eigenCap =
      eigenHermitian(view(nominal, 2U, 2U, 2U, 1U), solvePolicy(),
                     {.elements = 32U,
                      .bytes = 12U * 4U * sizeof(Complex) - 1U,
                      .evaluations = 0U});
  VSS_CHECK(!eigenCap.hasValue());
  VSS_CHECK(eigenCap.status().code() == ErrorCode::ResourceLimit);

  const std::array<Complex, 9> threeDimensional{{{4.0, 0.0},
                                                 {1.0, 2.0},
                                                 {0.0, -1.0},
                                                 {1.0, -2.0},
                                                 {3.0, 0.0},
                                                 {2.0, 0.5},
                                                 {0.0, 1.0},
                                                 {2.0, -0.5},
                                                 {5.0, 0.0}}};
  const auto eigen3 = eigenHermitian(view(threeDimensional, 3U, 3U, 3U, 1U),
                                     solvePolicy(), generousLimits());
  VSS_CHECK(eigen3.hasValue());
  if (eigen3) {
    for (Index i = 0U; i < 3U; ++i) {
      VSS_CHECK(eigen3.value().residual_norms.at(i) < 1.0e-10);
    }
  }
  const std::array<Complex, 4> repeated{{{2.0, 0.0}, {}, {}, {2.0, 0.0}}};
  const auto repeatedEigen = eigenHermitian(view(repeated, 2U, 2U, 2U, 1U),
                                            solvePolicy(), generousLimits());
  VSS_CHECK(repeatedEigen.hasValue());
  if (repeatedEigen) {
    checkNear({repeatedEigen.value().values.at(0U), 0.0}, {2.0, 0.0});
    checkNear({repeatedEigen.value().values.at(1U), 0.0}, {2.0, 0.0});
  }
  const std::span<const Complex> empty;
  const auto emptyEigen = eigenHermitian(view(empty, 0U, 0U, 0U, 0U),
                                         solvePolicy(), generousLimits());
  VSS_CHECK(emptyEigen.hasValue());
}

void complexSvdReconstructsRankOneTallAndWideMatrices() {
  const std::array<Complex, 6> tall{
      {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}}};
  auto svd = factorComplexSVD(view(tall, 2U, 3U, 3U, 1U), solvePolicy(),
                              generousLimits());
  VSS_CHECK(svd.hasValue());
  if (!svd) {
    return;
  }
  VSS_CHECK(svd.value().rank == 1U);
  VSS_CHECK(svd.value().left.rows == 2U && svd.value().left.columns == 2U);
  VSS_CHECK(svd.value().right_adjoint.rows == 2U &&
            svd.value().right_adjoint.columns == 3U);
  checkNear({svd.value().singular.at(0U), 0.0}, {std::sqrt(10.0), 0.0},
            1.0e-10);
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 3U; ++j) {
      Complex reconstructed{};
      for (Index q = 0U; q < 2U; ++q) {
        reconstructed += svd.value().left.values.view()[i * 2U + q] *
                         svd.value().singular.at(q) *
                         svd.value().right_adjoint.values.view()[q * 3U + j];
      }
      checkNear(reconstructed, tall[i * 3U + j], 1.0e-10);
    }
  }

  const std::array<Complex, 6> wide{
      {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}}};
  const std::array<Complex, 6> adjoint{{{1.0, 0.0},
                                        {0.0, -1.0},
                                        {2.0, 0.0},
                                        {0.0, -2.0},
                                        {3.0, 0.0},
                                        {0.0, -3.0}}};
  const auto tallAdjoint = factorComplexSVD(view(adjoint, 3U, 2U, 2U, 1U),
                                            solvePolicy(), generousLimits());
  VSS_CHECK(tallAdjoint.hasValue());
  const auto wideSvd = factorComplexSVD(view(wide, 2U, 3U, 3U, 1U),
                                        solvePolicy(), generousLimits());
  VSS_CHECK(wideSvd.hasValue());
  if (wideSvd) {
    VSS_CHECK(wideSvd.value().rank == 1U);
    VSS_CHECK(wideSvd.value().left.rows == 2U &&
              wideSvd.value().left.columns == 2U);
    VSS_CHECK(wideSvd.value().right_adjoint.rows == 2U &&
              wideSvd.value().right_adjoint.columns == 3U);
  }

  const std::array<Complex, 9> cutoffMatrix{
      {{4.0, 0.0}, {}, {}, {}, {1.0, 0.0}, {}, {}, {}, {0.0, 0.0}}};
  auto cutoffPolicy = solvePolicy();
  cutoffPolicy.rank_abs = 1.0;
  cutoffPolicy.rank_rel = 0.0;
  const auto cutoff = factorComplexSVD(view(cutoffMatrix, 3U, 3U, 3U, 1U),
                                       cutoffPolicy, generousLimits());
  VSS_CHECK(cutoff.hasValue());
  if (cutoff) {
    VSS_CHECK(cutoff.value().rank == 1U);
    VSS_CHECK(cutoff.value().singular.size() == 3U);
  }
  cutoffPolicy.rank_abs = 1.0 - 1.0e-10;
  const auto below = factorComplexSVD(view(cutoffMatrix, 3U, 3U, 3U, 1U),
                                      cutoffPolicy, generousLimits());
  VSS_CHECK(below.hasValue());
  if (below) {
    VSS_CHECK(below.value().rank == 2U);
  }
  const std::array<Complex, 9> clustered{
      {{3.0, 0.0}, {}, {}, {}, {3.0, 0.0}, {}, {}, {}, {1.0, 0.0}}};
  auto clusterPolicy = solvePolicy();
  clusterPolicy.rank_abs = 1.0;
  clusterPolicy.rank_rel = 0.0;
  const auto cluster = factorComplexSVD(view(clustered, 3U, 3U, 3U, 1U),
                                        clusterPolicy, generousLimits());
  VSS_CHECK(cluster.hasValue());
  if (cluster) {
    VSS_CHECK(cluster.value().rank == 2U);
    // The tied leading subspace is unique even though its basis is not.
    for (Index i = 0U; i < 3U; ++i) {
      for (Index j = 0U; j < 3U; ++j) {
        Complex projector{};
        for (Index q = 0U; q < 2U; ++q) {
          projector +=
              cluster.value().left.values.view()[i * 3U + q] *
              std::conj(cluster.value().left.values.view()[j * 3U + q]);
        }
        checkNear(projector, i == j && i < 2U ? Complex{1.0, 0.0} : Complex{},
                  1.0e-10);
      }
    }
  }

  const std::array<Complex, 9> rotatedCluster{{{2.4, 0.0},
                                               {},
                                               {0.6, 0.0},
                                               {},
                                               {3.0, 0.0},
                                               {},
                                               {-1.8, 0.0},
                                               {},
                                               {0.8, 0.0}}};
  const auto rotated = factorComplexSVD(view(rotatedCluster, 3U, 3U, 3U, 1U),
                                        clusterPolicy, generousLimits());
  VSS_CHECK(rotated.hasValue());
  if (rotated) {
    // The left leading subspace is the image of span(e1,e2) under the
    // unitary mixing e1 and e3 by cosine 0.8 and sine 0.6.
    const std::array<Complex, 9> expectedProjector{{{0.64, 0.0},
                                                    {},
                                                    {-0.48, 0.0},
                                                    {},
                                                    {1.0, 0.0},
                                                    {},
                                                    {-0.48, 0.0},
                                                    {},
                                                    {0.36, 0.0}}};
    for (Index i = 0U; i < 3U; ++i) {
      for (Index j = 0U; j < 3U; ++j) {
        Complex projector{};
        for (Index q = 0U; q < 2U; ++q) {
          projector +=
              rotated.value().left.values.view()[i * 3U + q] *
              std::conj(rotated.value().left.values.view()[j * 3U + q]);
        }
        checkNear(projector, expectedProjector[i * 3U + j], 1.0e-10);
      }
    }
  }

  const auto svdCap =
      factorComplexSVD(view(clustered, 3U, 3U, 3U, 1U), clusterPolicy,
                       {.elements = 64U,
                        .bytes = 24U * 9U * sizeof(Complex) - 1U,
                        .evaluations = 0U});
  VSS_CHECK(!svdCap.hasValue());
  VSS_CHECK(svdCap.status().code() == ErrorCode::ResourceLimit);
  const double tinyScale = 1.0e-200;
  const std::array<Complex, 4> tiny{
      {{3.0 * tinyScale, 0.0}, {}, {}, {tinyScale, 0.0}}};
  const auto tinySvd = factorComplexSVD(view(tiny, 2U, 2U, 2U, 1U),
                                        solvePolicy(), generousLimits());
  VSS_CHECK(tinySvd.hasValue());
  if (tinySvd) {
    VSS_CHECK(tinySvd.value().rank == 2U);
    checkNear({tinySvd.value().singular.at(0U) / tinyScale, 0.0}, {3.0, 0.0},
              1.0e-10);
  }
  const std::array<Complex, 1> nonfinite{
      {{0.0, std::numeric_limits<double>::infinity()}}};
  const auto rejectedSvd = factorComplexSVD(view(nonfinite, 1U, 1U, 1U, 1U),
                                            solvePolicy(), generousLimits());
  VSS_CHECK(!rejectedSvd.hasValue());
  VSS_CHECK(rejectedSvd.status().code() == ErrorCode::NonFiniteInput);

  std::array<Complex, 96U * 64U> conformance{};
  for (Index i = 0U; i < 64U; ++i) {
    conformance[i * 64U + i] = Complex{static_cast<double>(i + 1U), 0.0};
  }
  const auto boundedScale = factorComplexSVD(
      view(conformance, 96U, 64U, 64U, 1U), solvePolicy(), generousLimits());
  VSS_CHECK(boundedScale.hasValue());
  if (boundedScale) {
    VSS_CHECK(boundedScale.value().left.rows == 96U);
    VSS_CHECK(boundedScale.value().right_adjoint.columns == 64U);
  }
}

void complexAlgorithmsHandleDeterministicDenseRegimes() {
  std::array<Complex, 25U> hermitian{};
  for (Index row = 0U; row < 5U; ++row) {
    hermitian[row * 5U + row] = {static_cast<double>(row + 1U), 0.0};
    for (Index column = row + 1U; column < 5U; ++column) {
      const Complex value{0.11 *
                              static_cast<double>((row + 1U) * (column + 2U)),
                          0.07 * static_cast<double>(column - row)};
      hermitian[row * 5U + column] = value;
      hermitian[column * 5U + row] = std::conj(value);
    }
  }
  const auto eigen = eigenHermitian(view(hermitian, 5U, 5U, 5U, 1U),
                                    solvePolicy(), generousLimits());
  VSS_CHECK(eigen.hasValue());
  if (eigen) {
    for (Index i = 1U; i < 5U; ++i) {
      VSS_CHECK(eigen.value().values.at(i - 1U) <= eigen.value().values.at(i));
    }
    for (Index i = 0U; i < 5U; ++i) {
      VSS_CHECK(std::isfinite(eigen.value().residual_norms.at(i)));
    }
  }

  std::array<Complex, 15U> tall{};
  for (Index row = 0U; row < 5U; ++row) {
    for (Index column = 0U; column < 3U; ++column) {
      tall[row * 3U + column] = {
          0.3 * static_cast<double>((row + 1U) * (column + 2U)) +
              (row == column ? 1.0 : 0.0),
          0.17 * (static_cast<double>(row) - static_cast<double>(column))};
    }
  }
  const auto tallSvd = factorComplexSVD(view(tall, 5U, 3U, 3U, 1U),
                                        solvePolicy(), generousLimits());
  VSS_CHECK(tallSvd.hasValue());

  std::array<Complex, 15U> wide{};
  for (Index row = 0U; row < 3U; ++row) {
    for (Index column = 0U; column < 5U; ++column) {
      wide[row * 5U + column] = {
          0.2 * static_cast<double>((row + 2U) * (column + 1U)) +
              (row == column ? 0.5 : 0.0),
          0.13 *
              (static_cast<double>(row) - 2.0 * static_cast<double>(column))};
    }
  }
  const auto wideSvd = factorComplexSVD(view(wide, 3U, 5U, 5U, 1U),
                                        solvePolicy(), generousLimits());
  VSS_CHECK(wideSvd.hasValue());
}

} // namespace

int main() {
  complexProductsUseBothComponentsAndCorrectShapes();
  stridedViewsAndZeroContractedExtentAreSupported();
  conjugateTransposeConjugatesAndIsAnInvolution();
  invalidInputsAndInsufficientLimitsFailWithoutResults();
  complexSolveHandlesMultipleRightHandSidesAndRankPolicy();
  hermitianEigenUsesComplexPhaseAndChecksAdmissionSeparately();
  complexSvdReconstructsRankOneTallAndWideMatrices();
  complexAlgorithmsHandleDeterministicDenseRegimes();
}
