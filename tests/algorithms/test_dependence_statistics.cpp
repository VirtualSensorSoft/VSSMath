#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Dependence.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>(values.begin(), values.size()));
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent n = rows.size();
  const Extent d = n == 0U ? 0U : rows.begin()->size();
  Matrix result(n, d, 0.0);
  Extent i = 0U;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == d);
    Extent j = 0U;
    for (const Scalar value : row) {
      result.at(i, j++) = value;
    }
    ++i;
  }
  return result;
}

Limits ample() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = std::numeric_limits<std::uint64_t>::max()};
}

SampleView sample(const Matrix& rows) {
  return {.rows = rows.view(), .observation_shape = {}, .valid = {}};
}

void close(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-12) {
  VSS_CHECK(std::abs(actual - expected) <= tolerance);
}

void rankCorrelationsUseAverageTiesAndKendallTauB() {
  const Vector x = vector({1.0, 1.0, 2.0});
  const Vector y = vector({1.0, 2.0, 2.0});
  const auto rho = spearmanCorrelation(x.view(), y.view(), ample());
  const auto tau = kendallTau(x.view(), y.view(), ample());
  VSS_CHECK(rho.hasValue());
  VSS_CHECK(tau.hasValue());
  close(rho.value(), 0.5);
  close(tau.value(), 0.5);

  const Vector ascending = vector({-0.0, 0.0, 1.0, 2.0});
  const Vector descending = vector({3.0, 3.0, 2.0, 1.0});
  const auto reversed =
      spearmanCorrelation(ascending.view(), descending.view(), ample());
  const auto reversedTau =
      kendallTau(ascending.view(), descending.view(), ample());
  VSS_CHECK(reversed.hasValue());
  VSS_CHECK(reversedTau.hasValue());
  close(reversed.value(), -1.0);
  close(reversedTau.value(), -1.0);

  const Vector constantValues = vector({2.0, 2.0, 2.0});
  const auto constantRanks =
      spearmanCorrelation(constantValues.view(), y.view(), ample());
  VSS_CHECK(!constantRanks.hasValue());
  VSS_CHECK(constantRanks.status().code() == ErrorCode::InvalidDomain);

  const auto constant =
      kendallTau(x.view(), vector({4.0, 4.0, 4.0}).view(), ample());
  VSS_CHECK(!constant.hasValue());
  VSS_CHECK(constant.status().code() == ErrorCode::InvalidDomain);
}

void dependenceRejectsMalformedInputsAndHonorsLimits() {
  const Vector x = vector({1.0, 2.0, 3.0});
  const Vector shortY = vector({1.0, 2.0});
  const auto mismatch = spearmanCorrelation(x.view(), shortY.view(), ample());
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);
  const Vector nonfinite =
      vector({1.0, 2.0, std::numeric_limits<Scalar>::infinity()});
  const auto invalid = kendallTau(x.view(), nonfinite.view(), ample());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);
  Limits capped = ample();
  capped.evaluations = 1U;
  const auto denied = kendallTau(x.view(), x.view(), capped);
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

void distanceCorrelationSupportsShapedSamplesWithoutQuadraticStorage() {
  const Matrix x = matrix({{-1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}});
  const Matrix same = matrix({{-1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}});
  const std::array<Index, 1U> vectorShape{2U};
  const SampleView xView{
      .rows = x.view(), .observation_shape = vectorShape, .valid = {}};
  const SampleView sameView{
      .rows = same.view(), .observation_shape = vectorShape, .valid = {}};
  const auto identical = distanceCorrelation(xView, sameView, ample());
  VSS_CHECK(identical.hasValue());
  close(identical.value(), 1.0);

  const Matrix scalarSide = matrix({{0.0}, {1.0}, {2.0}});
  const SampleView scalarView = sample(scalarSide);
  const Matrix tieSide = matrix({{0.0}, {0.0}, {1.0}});
  const auto independentReference =
      distanceCorrelation(scalarView, sample(tieSide), ample());
  VSS_CHECK(independentReference.hasValue());
  close(independentReference.value(), std::pow(10.0, 0.25) / 2.0, 1.0e-12);
  const auto differingDimensions =
      distanceCorrelation(scalarView, xView, ample());
  VSS_CHECK(differingDimensions.hasValue());
  VSS_CHECK(differingDimensions.value() >= 0.0);
  VSS_CHECK(differingDimensions.value() <= 1.0);

  const Matrix constant = matrix({{3.0}, {3.0}, {3.0}});
  const SampleView constantView = sample(constant);
  const auto undefined = distanceCorrelation(constantView, scalarView, ample());
  VSS_CHECK(!undefined.hasValue());
  VSS_CHECK(undefined.status().code() == ErrorCode::InvalidDomain);

  Limits capped = ample();
  capped.evaluations = 1U;
  const auto limited = distanceCorrelation(scalarView, xView, capped);
  VSS_CHECK(!limited.hasValue());
  VSS_CHECK(limited.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  rankCorrelationsUseAverageTiesAndKendallTauB();
  dependenceRejectsMalformedInputsAndHonorsLimits();
  distanceCorrelationSupportsShapedSamplesWithoutQuadraticStorage();
}
