#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Structured.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

namespace {

using namespace vss::math;

Limits limits() {
  return {.elements = 16384U, .bytes = 1048576U, .evaluations = 64U};
}

template <class T, std::size_t Size>
Array<T> array(const std::array<T, Size>& values) {
  const auto result = Array<T>::copy(std::span<const T>{values}, limits());
  VSS_CHECK(result.hasValue());
  return result.value();
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index row_count = rows.size();
  const Index column_count = row_count == 0U ? 0U : rows.begin()->size();
  Matrix result(row_count, column_count, 0.0);
  Index row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == column_count);
    Index column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

void diagonalActionAndValidation() {
  // VSS-MATH-TRACEABILITY: C113
  const linear::DiagonalMatrix diagonal{
      Vector(std::array<Scalar, 3U>{2.0, -3.0, 0.0})};
  const Vector input(std::array<Scalar, 3U>{4.0, 5.0, 6.0});
  const auto output = linear::apply(diagonal, input.view(), limits());
  VSS_CHECK(output.hasValue());
  VSS_CHECK(output.value().at(0U) == 8.0);
  VSS_CHECK(output.value().at(1U) == -15.0);
  VSS_CHECK(output.value().at(2U) == 0.0);

  const Vector wrong(std::array<Scalar, 2U>{1.0, 2.0});
  const auto mismatch = linear::apply(diagonal, wrong.view(), limits());
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);

  const linear::DiagonalMatrix empty{Vector{}};
  const Vector empty_vector;
  const auto empty_result = linear::apply(empty, empty_vector.view(), limits());
  VSS_CHECK(empty_result.hasValue() && empty_result.value().empty());
}

void blockGridActionPreservesRectangularLayoutAndEmptyBlocks() {
  // VSS-MATH-TRACEABILITY: C113
  const std::array<Extent, 2U> row_extents{{1U, 2U}};
  const std::array<Extent, 2U> column_extents{{2U, 1U}};
  const std::array<linear::MatrixBlock, 2U> blocks{{
      {0U, 0U, matrix({{1.0, 2.0}})},
      {1U, 1U, matrix({{3.0}, {4.0}})},
  }};
  const linear::BlockMatrix block_matrix{array(row_extents),
                                         array(column_extents), array(blocks)};
  const Vector input(std::array<Scalar, 3U>{5.0, 6.0, 7.0});
  const auto output = linear::apply(block_matrix, input.view(), limits());
  VSS_CHECK(output.hasValue());
  VSS_CHECK(output.value().at(0U) == 17.0);
  VSS_CHECK(output.value().at(1U) == 21.0);
  VSS_CHECK(output.value().at(2U) == 28.0);

  const std::array<linear::MatrixBlock, 3U> with_off_diagonal{
      {blocks[0U], {1U, 0U, matrix({{1.0, 0.0}, {0.0, 1.0}})}, blocks[1U]}};
  const linear::BlockMatrix nonsymmetric{
      array(row_extents), array(column_extents), array(with_off_diagonal)};
  const auto off_diagonal = linear::apply(nonsymmetric, input.view(), limits());
  VSS_CHECK(off_diagonal.hasValue());
  VSS_CHECK(off_diagonal.value().at(0U) == 17.0);
  VSS_CHECK(off_diagonal.value().at(1U) == 26.0);
  VSS_CHECK(off_diagonal.value().at(2U) == 34.0);

  const linear::BlockMatrix zero_rows{Array<Extent>{},
                                      array(std::array<Extent, 1U>{2U}),
                                      Array<linear::MatrixBlock>{}};
  const Vector two(std::array<Scalar, 2U>{1.0, 2.0});
  const auto empty_output = linear::apply(zero_rows, two.view(), limits());
  VSS_CHECK(empty_output.hasValue() && empty_output.value().empty());
}

void projectionUsesRetainedColumnSpan() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix basis = matrix({{1.0}, {0.0}});
  const Vector input(std::array<Scalar, 2U>{3.0, 4.0});
  SolvePolicy policy{};
  const auto output = linear::orthogonalProjection(basis.view(), input.view(),
                                                   policy, limits());
  VSS_CHECK(output.hasValue());
  VSS_CHECK(output.value().at(0U) == 3.0);
  VSS_CHECK(output.value().at(1U) == 0.0);

  const Matrix empty_basis(2U, 0U);
  policy.rank = RankPolicy::reject;
  const auto zero = linear::orthogonalProjection(
      empty_basis.view(), input.view(), policy, limits());
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().at(0U) == 0.0 && zero.value().at(1U) == 0.0);
}

void regularizedLeastSquaresUsesSquareRootAugmentation() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix design = matrix({{2.0}});
  const Matrix observations = matrix({{6.0}});
  Index calls = 0U;
  linear::LinearMap penalty{
      .rows = 1U,
      .columns = 1U,
      .apply =
          [&calls](ConstVectorView input) {
            ++calls;
            return Result<Vector>::success(Vector(input.size(), input.at(0U)));
          },
      .transpose = {},
  };
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.iteration.max_iterations = 128U;
  policy.iteration.max_evaluations = 64U;
  const auto result = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 2.0, policy, limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().parameters.at(0U, 0U) - 2.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().residual_norms.at(0U) - 2.0) < 1.0e-12);
  VSS_CHECK(calls == 1U);

  calls = 0U;
  penalty.apply = {};
  const auto unregularized = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 0.0, policy, limits());
  VSS_CHECK(unregularized.hasValue());
  VSS_CHECK(unregularized.value().parameters.at(0U, 0U) == 3.0);
  VSS_CHECK(calls == 0U);

  const auto negative = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, -1.0, policy, limits());
  VSS_CHECK(!negative.hasValue());
  VSS_CHECK(negative.status().code() == ErrorCode::InvalidDomain);

  calls = 0U;
  penalty.apply = [&calls](ConstVectorView input) {
    ++calls;
    return Result<Vector>::success(Vector(input.size(), input.at(0U)));
  };
  Limits no_callback_budget = limits();
  no_callback_budget.evaluations = 0U;
  const auto denied =
      linear::regularizedLeastSquares(design.view(), observations.view(),
                                      penalty, 2.0, policy, no_callback_budget);
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(calls == 0U);

  const Matrix large_design = matrix({{1.0e308}});
  const Matrix large_observations = matrix({{1.0e308}});
  const linear::LinearMap large_penalty{
      .rows = 1U,
      .columns = 1U,
      .apply =
          [](ConstVectorView input) {
            return Result<Vector>::success(
                Vector(std::array<Scalar, 1U>{1.0e308 * input.at(0U)}));
          },
      .transpose = {},
  };
  const auto scaled = linear::regularizedLeastSquares(
      large_design.view(), large_observations.view(), large_penalty, 100.0,
      policy, limits());
  VSS_CHECK(scaled.hasValue());
  VSS_CHECK(std::abs(scaled.value().parameters.at(0U, 0U) - 1.0 / 101.0) <
            1.0e-12);
}

void diagonalAndBlockSolvesUseOneGlobalCutoff() {
  // VSS-MATH-TRACEABILITY: C113
  const std::array<Matrix, 2U> blocks{{matrix({{2.0}}), matrix({{4.0}})}};
  const BlockDiagonal block_diagonal{array(blocks)};
  const Matrix rhs = matrix({{2.0}, {8.0}});
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.rank_rel = 0.5;
  const auto solved =
      linear::solveBlockDiagonal(block_diagonal, rhs.view(), policy, limits());
  VSS_CHECK(solved.hasValue());
  VSS_CHECK(solved.value().rank == 1U);
  VSS_CHECK(solved.value().rank_cutoff == 2.0);
  VSS_CHECK(solved.value().parameters.at(0U, 0U) == 0.0);
  VSS_CHECK(solved.value().parameters.at(1U, 0U) == 2.0);
  VSS_CHECK(solved.value().residual_norms.at(0U) == 2.0);

  const Matrix bands =
      matrix({{0.0, 0.0, 2.0}, {1.0, 1.0, 3.0}, {1.0, 4.0, 0.0}});
  const BandMatrix band{3U, 1U, 1U, bands};
  const Matrix band_rhs = matrix({{4.0}, {12.0}, {14.0}});
  policy.rank_rel = 0.0;
  policy.rank_abs = 0.0;
  policy.iteration.max_iterations = 128U;
  const auto band_solution =
      linear::solveBanded(band, band_rhs.view(), policy, limits());
  VSS_CHECK(band_solution.hasValue());
  VSS_CHECK(band_solution.value().rank == 3U);
  VSS_CHECK(std::abs(band_solution.value().parameters.at(0U, 0U) - 1.0) <
            1.0e-12);
  VSS_CHECK(std::abs(band_solution.value().parameters.at(1U, 0U) - 2.0) <
            1.0e-12);
  VSS_CHECK(std::abs(band_solution.value().parameters.at(2U, 0U) - 3.0) <
            1.0e-12);
  VSS_CHECK(band_solution.value().residual_norms.at(0U) < 1.0e-12);
}

void compactBandUsesLinearStorageAndKeepsSingularResidual() {
  // VSS-MATH-TRACEABILITY: C113
  constexpr Index n = 64;
  Matrix band(n, 3);
  Matrix rhs(n, 1);
  for (Index i = 0; i < n; ++i) {
    band.at(i, 0) = i == 0 ? 0 : 1;
    band.at(i, 1) = 4;
    band.at(i, 2) = i + 1 == n ? 0 : 1;
    rhs.at(i, 0) = band.at(i, 0) + band.at(i, 1) + band.at(i, 2);
  }
  const auto compact =
      linear::solveBanded({n, 1, 1, std::move(band)}, rhs.view(), SolvePolicy{},
                          {.elements = 513, .bytes = 8192, .evaluations = 0});
  VSS_CHECK(compact.hasValue());
  VSS_CHECK(compact.value().rank == n && !compact.value().identified_basis);
  VSS_CHECK(compact.value().residual_norms.at(0) < 1e-12);
  for (Index i = 0; i < n; ++i) {
    VSS_CHECK(std::abs(compact.value().parameters.at(i, 0) - 1) < 1e-12);
  }

  const BandMatrix singular{2, 0, 0, matrix({{2}, {0}})};
  const Matrix observations = matrix({{2}, {8}});
  auto strict =
      linear::solveBanded(singular, observations.view(), {}, limits());
  VSS_CHECK(!strict.hasValue() &&
            strict.status().code() == ErrorCode::RankDeficient);
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  const auto permissive =
      linear::solveBanded(singular, observations.view(), policy, limits());
  VSS_CHECK(permissive.hasValue());
  VSS_CHECK(permissive.value().rank == 1);
  VSS_CHECK(permissive.value().parameters.at(0, 0) == 1);
  VSS_CHECK(permissive.value().parameters.at(1, 0) == 0);
  VSS_CHECK(permissive.value().residual_norms.at(0) == 8);
  const auto empty = linear::solveBanded({0, 0, 0, Matrix{}},
                                         Matrix(0, 2).view(), policy, limits());
  VSS_CHECK(empty.hasValue() && empty.value().parameters.columns() == 2);
}

void singularBlockSolveKeepsOriginalResidual() {
  // VSS-MATH-TRACEABILITY: C113
  const BlockDiagonal blocks{
      array(std::array{matrix({{2}}), matrix({{1, 1}, {1, 1}})})};
  const Matrix rhs = matrix({{2}, {3}, {1}});
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.iteration.max_iterations = 128;
  const auto result =
      linear::solveBlockDiagonal(blocks, rhs.view(), policy, limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2 && result.value().identified_basis);
  for (Index i = 0; i < 3; ++i) {
    VSS_CHECK(std::abs(result.value().parameters.at(i, 0) - 1) < 1e-12);
  }
  VSS_CHECK(std::abs(result.value().residual_norms.at(0) - std::sqrt(2.0)) <
            1e-12);
}

void redundantProjectionAndCallbackFailures() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix basis = matrix({{1, 1}, {1, 1}, {0, 0}});
  const Vector input(std::array{3.0, 1.0, 4.0});
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.iteration.max_iterations = 64;
  policy.iteration.max_evaluations = 4;
  auto projected = linear::orthogonalProjection(basis.view(), input.view(),
                                                policy, limits());
  VSS_CHECK(projected.hasValue());
  VSS_CHECK(std::abs(projected.value().at(0) - 2) < 1e-12);
  VSS_CHECK(std::abs(projected.value().at(1) - 2) < 1e-12);
  VSS_CHECK(projected.value().at(2) == 0);
  policy.rank = RankPolicy::reject;
  auto strict = linear::orthogonalProjection(basis.view(), input.view(), policy,
                                             limits());
  VSS_CHECK(!strict.hasValue() &&
            strict.status().code() == ErrorCode::RankDeficient);
  const auto huge = Vector(2, std::numeric_limits<Scalar>::max());
  auto overflow = linear::orthogonalProjection(matrix({{2}, {1}}).view(),
                                               huge.view(), policy, limits());
  VSS_CHECK(!overflow.hasValue() &&
            overflow.status().code() == ErrorCode::NonFiniteResult);

  const Matrix design = matrix({{1, 1}});
  const Matrix observations = matrix({{4}});
  Index calls = 0;
  linear::LinearMap penalty{.rows = 1,
                            .columns = 2,
                            .apply =
                                [&](ConstVectorView) {
                                  ++calls;
                                  return Result<Vector>::success(Vector(1, 1));
                                },
                            .transpose = {}};
  policy.rank = RankPolicy::minimum_norm;
  auto ridge = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 1, policy, limits());
  VSS_CHECK(ridge.hasValue() && calls == 2 && ridge.value().rank == 1);
  VSS_CHECK(std::abs(ridge.value().parameters.at(0, 0) - 1) < 1e-12);
  VSS_CHECK(std::abs(ridge.value().parameters.at(1, 0) - 1) < 1e-12);
  VSS_CHECK(std::abs(ridge.value().residual_norms.at(0) - 2) < 1e-12);
  calls = 0;
  policy.iteration.max_evaluations = 1;
  const auto denied = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 1, policy, limits());
  VSS_CHECK(!denied.hasValue() &&
            denied.status().code() == ErrorCode::ResourceLimit && calls == 0);
  policy.iteration.max_evaluations = 2;
  penalty.apply = [](ConstVectorView) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "user penalty rejected basis vector");
  };
  const auto callback = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 1, policy, limits());
  VSS_CHECK(!callback.hasValue() &&
            callback.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(callback.status().message() ==
            "user penalty rejected basis vector");
}

void structuredActionFailureAndAccuracyCases() {
  const auto one = array(std::array<Index, 1>{1});
  const auto two = array(std::array<Index, 1>{2});
  const Limits none{.elements = 0, .bytes = 0, .evaluations = 0};
  const auto expect = [](const Result<Vector>& result, ErrorCode code) {
    VSS_CHECK(!result.hasValue() && result.status().code() == code);
  };
  const auto value = std::numeric_limits<Scalar>::max();
  const auto unit = Vector(1, 1);
  const auto twice = Vector(1, 2);
  const Vector pair(std::array{2.0, -2.0});
  const auto cancellation =
      linear::apply(linear::BlockMatrix{one, two,
                                        array(std::array{linear::MatrixBlock{
                                            0, 0, matrix({{value, value}})}})},
                    pair.view(), limits());
  VSS_CHECK(cancellation.hasValue() && cancellation.value().at(0) == 0);
  const Scalar very_large = std::scalbn(1.0, 1023);
  const Vector mixed_input(std::array{very_large, -very_large, 1.0});
  const auto after_cancellation = linear::apply(
      linear::BlockMatrix{one, array(std::array<Index, 1>{3}),
                          array(std::array{linear::MatrixBlock{
                              0, 0, matrix({{very_large, very_large, 1}})}})},
      mixed_input.view(), limits());
  VSS_CHECK(after_cancellation.hasValue() &&
            after_cancellation.value().at(0) == 1);
  const auto operation = [&](const Matrix& tile, const Vector& input,
                             Limits bound) {
    return linear::apply(
        linear::BlockMatrix{one, two,
                            array(std::array{linear::MatrixBlock{0, 0, tile}})},
        input.view(), bound);
  };
  expect(operation(matrix({{value, value}}), Vector(2, 1), limits()),
         ErrorCode::NonFiniteResult);
  expect(operation(matrix({{1, 1}}), pair, none), ErrorCode::ResourceLimit);
  expect(operation(matrix({{1}}), pair, limits()),
         ErrorCode::DimensionMismatch);
  const Scalar large = std::scalbn(1.0, 400);
  const Scalar small = std::scalbn(1.0, -400);
  auto mixed = operation(matrix({{large, small}}),
                         Vector(std::array{small, large}), limits());
  VSS_CHECK(mixed.hasValue() && mixed.value().at(0) == 2);

  const linear::MatrixBlock valid{0, 0, Matrix(1, 2, 1)};
  expect(linear::apply(
             linear::BlockMatrix{one, two, array(std::array{valid, valid})},
             pair.view(), limits()),
         ErrorCode::InvalidDomain);
  expect(linear::apply(linear::BlockMatrix{one, two,
                                           array(std::array{linear::MatrixBlock{
                                               1, 0, valid.value}})},
                       pair.view(), limits()),
         ErrorCode::IndexOutOfRange);
  const auto widths = array(std::array<Index, 2>{1, 1});
  expect(linear::apply(
             linear::BlockMatrix{
                 one, widths,
                 array(std::array{linear::MatrixBlock{0, 1, Matrix(1, 1, 1)},
                                  linear::MatrixBlock{0, 0, Matrix(1, 1, 1)}})},
             pair.view(), limits()),
         ErrorCode::InvalidDomain);
  const auto absent =
      linear::apply(linear::BlockMatrix{one, two, {}}, pair.view(), limits());
  VSS_CHECK(absent.hasValue() && absent.value().at(0) == 0);
  const auto zero_columns = linear::apply(linear::BlockMatrix{two, {}, {}},
                                          Vector{}.view(), limits());
  VSS_CHECK(zero_columns.hasValue() && zero_columns.value().size() == 2 &&
            zero_columns.value().at(1) == 0);
  expect(linear::apply(
             linear::BlockMatrix{array(std::array<Index, 1>{0}), one, {}},
             unit.view(), limits()),
         ErrorCode::InvalidDomain);

  expect(linear::apply(linear::DiagonalMatrix{Vector(1, value)}, twice.view(),
                       limits()),
         ErrorCode::NonFiniteResult);
  expect(
      linear::apply(linear::DiagonalMatrix{Vector(1, 1)}, twice.view(), none),
      ErrorCode::ResourceLimit);
  expect(linear::apply(linear::DiagonalMatrix{Vector(
                           1, std::numeric_limits<Scalar>::infinity())},
                       twice.view(), limits()),
         ErrorCode::NonFiniteInput);
  const Scalar tiny = std::numeric_limits<Scalar>::denorm_min();
  const auto subnormal = linear::apply(linear::DiagonalMatrix{Vector(1, -tiny)},
                                       twice.view(), limits());
  VSS_CHECK(subnormal.hasValue() && subnormal.value().at(0) == -2 * tiny);
}

void exactProjectionCutoffAndSignedZeroPenalty() {
  // VSS-MATH-TRACEABILITY: C113
  const Vector point(std::array{3.0, 4.0});
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.rank_rel = 0.25;
  auto projected = linear::orthogonalProjection(matrix({{4, 0}, {0, 1}}).view(),
                                                point.view(), policy, limits());
  VSS_CHECK(projected.hasValue() && projected.value().at(0) == 3 &&
            projected.value().at(1) == 0);
  projected = linear::orthogonalProjection(matrix({{4, 0}, {0, 2}}).view(),
                                           point.view(), policy, limits());
  VSS_CHECK(projected.hasValue() && projected.value().at(0) == 3 &&
            projected.value().at(1) == 4);

  policy.rank_rel = 0;
  policy.iteration.max_iterations = 64;
  const Matrix design = matrix({{1, 1}});
  const Matrix observations = matrix({{4}});
  const linear::LinearMap penalty{
      .rows = 1,
      .columns = 2,
      .apply = [](ConstVectorView) -> Result<Vector> {
        throw std::runtime_error("intentional callback failure");
      },
      .transpose = {}};
  const auto none = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, -0.0, policy, limits());
  VSS_CHECK(none.hasValue());
  VSS_CHECK(std::abs(none.value().parameters.at(0, 0) - 2) < 1e-12);
  VSS_CHECK(std::abs(none.value().parameters.at(1, 0) - 2) < 1e-12);

  policy.iteration.max_evaluations = 2;
  const auto threw = linear::regularizedLeastSquares(
      design.view(), observations.view(), penalty, 1, policy, limits());
  VSS_CHECK(!threw.hasValue() &&
            threw.status().code() == ErrorCode::CallbackFailure);
}

void c113RegularizedLeastSquaresPolicyAndCallbackCases() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix design = matrix({{1.0, 1.0}});
  const Matrix observations = matrix({{4.0, 6.0}});
  SolvePolicy minimum_norm{};
  minimum_norm.rank = RankPolicy::minimum_norm;
  minimum_norm.iteration.max_iterations = 128U;
  minimum_norm.iteration.max_evaluations = 64U;

  Index calls = 0U;
  Index transpose_calls = 0U;
  const linear::LinearMap inactive{
      .rows = 17U,
      .columns = 99U,
      .apply =
          [&calls](ConstVectorView) {
            ++calls;
            throw std::runtime_error("inactive penalty must not be called");
            return Result<Vector>::success(Vector{});
          },
      .transpose =
          [&transpose_calls](ConstVectorView) {
            ++transpose_calls;
            throw std::runtime_error("transpose must not be called");
            return Result<Vector>::success(Vector{});
          }};
  const auto unregularized =
      linear::regularizedLeastSquares(design.view(), observations.view(),
                                      inactive, 0.0, minimum_norm, limits());
  VSS_CHECK(unregularized.hasValue());
  VSS_CHECK(unregularized.value().parameters.at(0U, 0U) == 2.0);
  VSS_CHECK(unregularized.value().parameters.at(1U, 0U) == 2.0);
  VSS_CHECK(unregularized.value().parameters.at(0U, 1U) == 3.0);
  VSS_CHECK(unregularized.value().parameters.at(1U, 1U) == 3.0);
  VSS_CHECK(unregularized.value().residual_norms.size() == 2U);
  VSS_CHECK(unregularized.value().residual_norms.at(0U) == 0.0);
  VSS_CHECK(unregularized.value().residual_norms.at(1U) == 0.0);
  VSS_CHECK(calls == 0U && transpose_calls == 0U);

  const linear::LinearMap no_rows{
      .rows = 0U, .columns = 2U, .apply = {}, .transpose = {}};
  const auto data_only = linear::regularizedLeastSquares(
      design.view(), observations.view(), no_rows, 3.0, minimum_norm, limits());
  VSS_CHECK(data_only.hasValue());
  VSS_CHECK(data_only.value().parameters.at(0U, 0U) == 2.0);
  VSS_CHECK(data_only.value().parameters.at(1U, 1U) == 3.0);

  const linear::LinearMap selective{
      .rows = 1U,
      .columns = 2U,
      .apply =
          [&calls](ConstVectorView input) {
            ++calls;
            return Result<Vector>::success(
                Vector(std::array<Scalar, 1U>{input.at(0U)}));
          },
      .transpose =
          [&transpose_calls](ConstVectorView) {
            ++transpose_calls;
            throw std::runtime_error("penalty transpose must not be called");
            return Result<Vector>::success(Vector{});
          }};
  calls = 0U;
  const auto selectively_regularized =
      linear::regularizedLeastSquares(design.view(), matrix({{4.0}}).view(),
                                      selective, 1.0, minimum_norm, limits());
  VSS_CHECK(selectively_regularized.hasValue());
  VSS_CHECK(selectively_regularized.value().rank == 2U);
  VSS_CHECK(std::abs(selectively_regularized.value().parameters.at(0U, 0U)) <
            1e-12);
  VSS_CHECK(std::abs(selectively_regularized.value().parameters.at(1U, 0U) -
                     4.0) < 1e-12);
  VSS_CHECK(calls == 2U && transpose_calls == 0U);

  SolvePolicy strict{};
  strict.rank = RankPolicy::reject;
  strict.iteration.max_iterations = 128U;
  strict.iteration.max_evaluations = 64U;
  const linear::LinearMap duplicate_penalty{
      .rows = 1U,
      .columns = 2U,
      .apply =
          [](ConstVectorView input) {
            return Result<Vector>::success(
                Vector(std::array<Scalar, 1U>{input.at(0U) + input.at(1U)}));
          },
      .transpose = {}};
  const auto rank_loss =
      linear::regularizedLeastSquares(design.view(), matrix({{4.0}}).view(),
                                      duplicate_penalty, 1.0, strict, limits());
  VSS_CHECK(!rank_loss.hasValue());
  VSS_CHECK(rank_loss.status().code() == ErrorCode::RankDeficient);

  const auto nonfinite_weight = linear::regularizedLeastSquares(
      design.view(), observations.view(), no_rows,
      std::numeric_limits<Scalar>::infinity(), minimum_norm, limits());
  VSS_CHECK(!nonfinite_weight.hasValue());
  VSS_CHECK(nonfinite_weight.status().code() == ErrorCode::NonFiniteInput);
  const auto nan_weight = linear::regularizedLeastSquares(
      design.view(), observations.view(), no_rows,
      std::numeric_limits<Scalar>::quiet_NaN(), minimum_norm, limits());
  VSS_CHECK(!nan_weight.hasValue());
  VSS_CHECK(nan_weight.status().code() == ErrorCode::NonFiniteInput);

  const linear::LinearMap wrong_extent{
      .rows = 1U,
      .columns = 2U,
      .apply =
          [](ConstVectorView) { return Result<Vector>::success(Vector{}); },
      .transpose = {}};
  const auto wrong_output = linear::regularizedLeastSquares(
      design.view(), matrix({{4.0}}).view(), wrong_extent, 1.0, minimum_norm,
      limits());
  VSS_CHECK(!wrong_output.hasValue());
  VSS_CHECK(wrong_output.status().code() == ErrorCode::DimensionMismatch);

  const linear::LinearMap nonfinite_output{
      .rows = 1U,
      .columns = 2U,
      .apply =
          [](ConstVectorView) {
            return Result<Vector>::success(Vector(std::array<Scalar, 1U>{
                std::numeric_limits<Scalar>::infinity()}));
          },
      .transpose = {}};
  const auto invalid_output = linear::regularizedLeastSquares(
      design.view(), matrix({{4.0}}).view(), nonfinite_output, 1.0,
      minimum_norm, limits());
  VSS_CHECK(!invalid_output.hasValue());
  VSS_CHECK(invalid_output.status().code() == ErrorCode::NonFiniteInput);
}

void c113StructuredSolvesPreserveAllRightHandSidesAndRejectPadding() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix diagonal_storage = matrix({{2.0}, {4.0}});
  const Matrix rhs = matrix({{2.0, 4.0}, {8.0, 12.0}});
  const auto band_result = linear::solveBanded(
      BandMatrix{2U, 0U, 0U, diagonal_storage}, rhs.view(), {}, limits());
  VSS_CHECK(band_result.hasValue());
  VSS_CHECK(band_result.value().parameters.rows() == 2U);
  VSS_CHECK(band_result.value().parameters.columns() == 2U);
  VSS_CHECK(band_result.value().parameters.at(0U, 0U) == 1.0);
  VSS_CHECK(band_result.value().parameters.at(0U, 1U) == 2.0);
  VSS_CHECK(band_result.value().parameters.at(1U, 0U) == 2.0);
  VSS_CHECK(band_result.value().parameters.at(1U, 1U) == 3.0);
  VSS_CHECK(band_result.value().residual_norms.size() == 2U);
  VSS_CHECK(band_result.value().residual_norms.at(0U) == 0.0);
  VSS_CHECK(band_result.value().residual_norms.at(1U) == 0.0);

  const BandMatrix padded{2U, 0U, 1U, matrix({{2.0, 0.0}, {4.0, 9.0}})};
  const auto padding_error =
      linear::solveBanded(padded, matrix({{2.0}, {4.0}}).view(), {}, limits());
  VSS_CHECK(!padding_error.hasValue());
  VSS_CHECK(padding_error.status().code() == ErrorCode::InvalidDomain);

  const std::array<Matrix, 2U> scalar_blocks{
      {matrix({{2.0}}), matrix({{4.0}})}};
  const BlockDiagonal block_diagonal{array(scalar_blocks)};
  const auto block_result =
      linear::solveBlockDiagonal(block_diagonal, rhs.view(), {}, limits());
  VSS_CHECK(block_result.hasValue());
  VSS_CHECK(block_result.value().parameters.at(0U, 0U) == 1.0);
  VSS_CHECK(block_result.value().parameters.at(0U, 1U) == 2.0);
  VSS_CHECK(block_result.value().parameters.at(1U, 0U) == 2.0);
  VSS_CHECK(block_result.value().parameters.at(1U, 1U) == 3.0);
  VSS_CHECK(block_result.value().residual_norms.size() == 2U);
  VSS_CHECK(block_result.value().residual_norms.at(0U) == 0.0);
  VSS_CHECK(block_result.value().residual_norms.at(1U) == 0.0);

  const BlockDiagonal empty_blocks{Array<Matrix>{}};
  const Matrix empty_rhs(0U, 2U);
  const auto empty_result =
      linear::solveBlockDiagonal(empty_blocks, empty_rhs.view(), {}, limits());
  VSS_CHECK(empty_result.hasValue());
  VSS_CHECK(empty_result.value().parameters.rows() == 0U);
  VSS_CHECK(empty_result.value().parameters.columns() == 2U);
  VSS_CHECK(empty_result.value().residual_norms.size() == 2U);
}

void c113ProjectionSatisfiesSubspaceInvariants() {
  // VSS-MATH-TRACEABILITY: C113
  const Matrix redundant_basis = matrix({{1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}});
  const Vector input(std::array<Scalar, 3U>{3.0, 1.0, 4.0});
  const Vector other(std::array<Scalar, 3U>{5.0, -3.0, 2.0});
  SolvePolicy policy{};
  policy.rank = RankPolicy::minimum_norm;
  policy.iteration.max_iterations = 64U;
  policy.iteration.max_evaluations = 64U;
  const auto projected = linear::orthogonalProjection(
      redundant_basis.view(), input.view(), policy, limits());
  const auto other_projected = linear::orthogonalProjection(
      redundant_basis.view(), other.view(), policy, limits());
  VSS_CHECK(projected.hasValue() && other_projected.hasValue());
  VSS_CHECK(std::abs(projected.value().at(0U) - 2.0) < 1e-12);
  VSS_CHECK(std::abs(projected.value().at(1U) - 2.0) < 1e-12);
  VSS_CHECK(std::abs(projected.value().at(2U)) < 1e-12);

  const auto projected_again = linear::orthogonalProjection(
      redundant_basis.view(), projected.value().view(), policy, limits());
  VSS_CHECK(projected_again.hasValue());
  const auto dot = [](ConstVectorView lhs, ConstVectorView rhs) {
    Scalar result = 0.0;
    for (Index i = 0U; i < lhs.size(); ++i) {
      result += lhs.at(i) * rhs.at(i);
    }
    return result;
  };
  for (Index i = 0U; i < 3U; ++i) {
    VSS_CHECK(std::abs(projected_again.value().at(i) -
                       projected.value().at(i)) < 1e-12);
  }
  VSS_CHECK(dot(projected.value().view(), projected.value().view()) <=
            dot(input.view(), input.view()));
  VSS_CHECK(std::abs(dot(input.view(), other_projected.value().view()) -
                     dot(projected.value().view(), other.view())) < 1e-12);
  const Vector residual(std::array<Scalar, 3U>{1.0, -1.0, 4.0});
  VSS_CHECK(std::abs(dot(
                residual.view(),
                Vector(std::array<Scalar, 3U>{1.0, 1.0, 0.0}).view())) < 1e-12);

  const Scalar inverse_root_two = 1.0 / std::numbers::sqrt2;
  const Matrix coordinate_basis = matrix({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}});
  const Matrix rotated_basis = matrix({{inverse_root_two, -inverse_root_two},
                                       {inverse_root_two, inverse_root_two},
                                       {0.0, 0.0}});
  const Vector rotated_input(std::array<Scalar, 3U>{3.0, 4.0, 5.0});
  const auto coordinate_projection = linear::orthogonalProjection(
      coordinate_basis.view(), rotated_input.view(), policy, limits());
  const auto rotated_projection = linear::orthogonalProjection(
      rotated_basis.view(), rotated_input.view(), policy, limits());
  VSS_CHECK(coordinate_projection.hasValue() && rotated_projection.hasValue());
  for (Index i = 0U; i < 3U; ++i) {
    VSS_CHECK(std::abs(coordinate_projection.value().at(i) -
                       rotated_projection.value().at(i)) < 1e-12);
  }

  const Matrix zero_rows(0U, 2U);
  const Vector empty_vector;
  policy.rank = RankPolicy::reject;
  const auto empty_projection = linear::orthogonalProjection(
      zero_rows.view(), empty_vector.view(), policy, limits());
  VSS_CHECK(empty_projection.hasValue() && empty_projection.value().empty());
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  diagonalActionAndValidation();
  blockGridActionPreservesRectangularLayoutAndEmptyBlocks();
  projectionUsesRetainedColumnSpan();
  regularizedLeastSquaresUsesSquareRootAugmentation();
  diagonalAndBlockSolvesUseOneGlobalCutoff();
  compactBandUsesLinearStorageAndKeepsSingularResidual();
  singularBlockSolveKeepsOriginalResidual();
  redundantProjectionAndCallbackFailures();
  structuredActionFailureAndAccuracyCases();
  exactProjectionCutoffAndSignedZeroPenalty();
  c113RegularizedLeastSquaresPolicyAndCallbackCases();
  c113StructuredSolvesPreserveAllRightHandSidesAndRejectPadding();
  c113ProjectionSatisfiesSubspaceInvariants();
  return 0;
}
