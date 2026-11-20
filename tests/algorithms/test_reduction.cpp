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
  VSS_CHECK(result.value().singular_values.size() == 1U);
  VSS_CHECK(result.value().singular_values.at(0U) == 3.0);
  VSS_CHECK(result.value().approximation.at(0U, 0U) == 3.0);
  VSS_CHECK(result.value().approximation.at(0U, 1U) == 0.0);
  VSS_CHECK(result.value().approximation.at(1U, 0U) == 0.0);
  VSS_CHECK(result.value().approximation.at(1U, 1U) == 0.0);
  VSS_CHECK(result.value().residual_norm == 0.0);

  const auto excessive =
      linear::lowRankApproximation(diagonal.view(), 3U, policy(), limits());
  VSS_CHECK(!excessive.hasValue());
  VSS_CHECK(excessive.status().code() == ErrorCode::InvalidDomain);

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
  const auto result = linear::pca(samples.view(), 1U, true, policy(), limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().center.size() == 2U);
  VSS_CHECK(std::abs(result.value().center.at(0U)) < 1.0e-14);
  VSS_CHECK(std::abs(result.value().center.at(1U)) < 1.0e-14);
  VSS_CHECK(result.value().components.rows() == 2U);
  VSS_CHECK(result.value().components.columns() == 1U);
  VSS_CHECK(std::abs(result.value().components.at(0U, 0U)) > 1.0 - 1.0e-12);
  VSS_CHECK(std::abs(result.value().components.at(1U, 0U)) < 1.0e-12);
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

  const auto excessive =
      linear::pca(samples.view(), 3U, false, policy(), limits());
  VSS_CHECK(!excessive.hasValue());
  VSS_CHECK(excessive.status().code() == ErrorCode::InvalidDomain);

  const Matrix constant = matrix({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}});
  const auto zeroVariance =
      linear::pca(constant.view(), 1U, true, policy(), limits());
  VSS_CHECK(!zeroVariance.hasValue());
  VSS_CHECK(zeroVariance.status().code() == ErrorCode::RankDeficient);

  const auto denied =
      linear::pca(samples.view(), 1U, false, policy(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  lowRankApproximation();
  centeredWhitenedPca();
  return 0;
}
