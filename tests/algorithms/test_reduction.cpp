#include "Test.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/Reduction.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount, 0.0);
  Index row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Index column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 128U;
  result.rank = RankPolicy::supported_subspace;
  return result;
}

Limits limits() { return {4096U, 262144U, 0U}; }

void lowRankApproximation() {
  const Matrix diagonal = matrix({{3.0, 0.0}, {0.0, 0.0}});
  const auto result =
      linear::lowRankApproximation(diagonal.view(), 1U, policy(), limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().singular.size() == 1U);
  VSS_CHECK(result.value().singular.at(0U) == 3.0);
  VSS_CHECK(result.value().left.rows() == 2U &&
            result.value().left.columns() == 1U);
  VSS_CHECK(result.value().right_transpose.rows() == 1U &&
            result.value().right_transpose.columns() == 2U);
  for (Index row = 0U; row < 2U; ++row) {
    for (Index column = 0U; column < 2U; ++column) {
      Scalar reconstructed = 0.0;
      for (Index mode = 0U; mode < result.value().singular.size(); ++mode) {
        reconstructed += result.value().left.at(row, mode) *
                         result.value().singular.at(mode) *
                         result.value().right_transpose.at(mode, column);
      }
      VSS_CHECK(reconstructed == diagonal.at(row, column));
    }
  }
  VSS_CHECK(result.value().residual_frobenius == 0.0);

  const auto zeroRank =
      linear::lowRankApproximation(diagonal.view(), 0U, policy(), limits());
  VSS_CHECK(zeroRank.hasValue());
  VSS_CHECK(zeroRank.value().left.rows() == 2U &&
            zeroRank.value().left.columns() == 0U);
  VSS_CHECK(zeroRank.value().right_transpose.rows() == 0U &&
            zeroRank.value().right_transpose.columns() == 2U);
  VSS_CHECK(zeroRank.value().residual_frobenius == 3.0);

  const Matrix nonzeroMeanInput = matrix({{5.0, 0.0}, {0.0, 3.0}});
  const auto uncentered = linear::lowRankApproximation(nonzeroMeanInput.view(),
                                                       1U, policy(), limits());
  VSS_CHECK(uncentered.hasValue());
  VSS_CHECK(uncentered.value().singular.at(0U) == 5.0);
  VSS_CHECK(uncentered.value().residual_frobenius == 3.0);

  const Matrix repeated = matrix({{2.0, 0.0}, {0.0, 2.0}});
  const auto repeatedRankOne =
      linear::lowRankApproximation(repeated.view(), 1U, policy(), limits());
  VSS_CHECK(repeatedRankOne.hasValue());
  VSS_CHECK(std::abs(repeatedRankOne.value().residual_frobenius - 2.0) <
            1.0e-12);

  const auto excessive =
      linear::lowRankApproximation(diagonal.view(), 3U, policy(), limits());
  VSS_CHECK(!excessive.hasValue());
  VSS_CHECK(excessive.status().code() == ErrorCode::InvalidExtent);

  const auto denied =
      linear::lowRankApproximation(diagonal.view(), 1U, policy(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);

  Matrix nonFinite = diagonal;
  nonFinite.at(1U, 1U) = std::numeric_limits<Scalar>::infinity();
  const auto invalid =
      linear::lowRankApproximation(nonFinite.view(), 1U, policy(), limits());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);
}

void centeredWhitenedPca() {
  const Matrix samples =
      matrix({{2.0, 0.0}, {-2.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}});
  const std::array<Index, 1> shape{2U};
  const SampleView sampleView{samples.view(), shape, {}};
  const auto result = linear::pca(sampleView, 1U, true, policy(), limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().center.size() == 2U);
  VSS_CHECK(std::abs(result.value().center.at(0U)) < 1.0e-14);
  VSS_CHECK(std::abs(result.value().center.at(1U)) < 1.0e-14);
  VSS_CHECK(result.value().axes.rows() == 2U);
  VSS_CHECK(result.value().axes.columns() == 1U);
  VSS_CHECK(std::abs(result.value().axes.at(0U, 0U)) > 1.0 - 1.0e-12);
  VSS_CHECK(std::abs(result.value().axes.at(1U, 0U)) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().explained_variance.at(0U) - 8.0 / 3.0) <
            1.0e-12);
  VSS_CHECK(result.value().scores.rows() == 4U);
  VSS_CHECK(result.value().scores.columns() == 1U);
  Scalar scoreMean = 0.0;
  for (Index row = 0U; row < result.value().scores.rows(); ++row) {
    scoreMean += result.value().scores.at(row, 0U) / 4.0;
  }
  Scalar sampleVariance = 0.0;
  for (Index row = 0U; row < result.value().scores.rows(); ++row) {
    const Scalar delta = result.value().scores.at(row, 0U) - scoreMean;
    sampleVariance += delta * delta / 3.0;
  }
  VSS_CHECK(std::abs(sampleVariance - 1.0) < 1.0e-12);

  const auto excessive = linear::pca(sampleView, 3U, false, policy(), limits());
  VSS_CHECK(!excessive.hasValue());
  VSS_CHECK(excessive.status().code() == ErrorCode::InvalidExtent);

  const auto meanOnly = linear::pca(sampleView, 0U, false, policy(), limits());
  VSS_CHECK(meanOnly.hasValue());
  VSS_CHECK(meanOnly.value().axes.columns() == 0U);
  VSS_CHECK(meanOnly.value().scores.columns() == 0U);
  VSS_CHECK(meanOnly.value().explained_variance.empty());

  const Matrix twoRows = matrix({{1.0, 2.0}, {3.0, 4.0}});
  const SampleView twoRowView{twoRows.view(), shape, {}};
  const auto centeredRankExceeded =
      linear::pca(twoRowView, 2U, false, policy(), limits());
  VSS_CHECK(!centeredRankExceeded.hasValue());
  VSS_CHECK(centeredRankExceeded.status().code() == ErrorCode::InvalidExtent);

  const Matrix oneRow = matrix({{4.0, 7.0}});
  const SampleView oneRowView{oneRow.view(), shape, {}};
  const auto insufficient =
      linear::pca(oneRowView, 0U, false, policy(), limits());
  VSS_CHECK(!insufficient.hasValue());
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);

  const Matrix translated =
      matrix({{12.0, -3.0}, {8.0, -3.0}, {10.0, -2.0}, {10.0, -4.0}});
  const SampleView translatedView{translated.view(), shape, {}};
  const auto translatedPca =
      linear::pca(translatedView, 1U, false, policy(), limits());
  VSS_CHECK(translatedPca.hasValue());
  VSS_CHECK(std::abs(translatedPca.value().center.at(0U) - 10.0) < 1.0e-14);
  VSS_CHECK(std::abs(translatedPca.value().center.at(1U) + 3.0) < 1.0e-14);
  VSS_CHECK(std::abs(translatedPca.value().explained_variance.at(0U) -
                     result.value().explained_variance.at(0U)) < 1.0e-12);

  const Scalar populationVariance = sampleVariance * 3.0 / 4.0;
  VSS_CHECK(std::abs(populationVariance - 0.75) < 1.0e-12);

  const Matrix constant = matrix({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}});
  const SampleView constantView{constant.view(), shape, {}};
  const auto zeroVariance =
      linear::pca(constantView, 1U, true, policy(), limits());
  VSS_CHECK(!zeroVariance.hasValue());
  VSS_CHECK(zeroVariance.status().code() == ErrorCode::RankDeficient);

  const auto denied =
      linear::pca(sampleView, 1U, false, policy(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);

  const std::array<Index, 1> wrongShape{3U};
  const SampleView malformedShape{samples.view(), wrongShape, {}};
  const auto shapeMismatch =
      linear::pca(malformedShape, 1U, false, policy(), limits());
  VSS_CHECK(!shapeMismatch.hasValue());
  VSS_CHECK(shapeMismatch.status().code() == ErrorCode::DimensionMismatch);

  const std::array<std::uint8_t, 8> missing{1U, 1U, 1U, 1U, 1U, 1U, 1U, 0U};
  const SampleView incomplete{samples.view(), shape, missing};
  const auto missingRejected =
      linear::pca(incomplete, 1U, false, policy(), limits());
  VSS_CHECK(!missingRejected.hasValue());
  VSS_CHECK(missingRejected.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() {
  lowRankApproximation();
  centeredWhitenedPca();
  return 0;
}
