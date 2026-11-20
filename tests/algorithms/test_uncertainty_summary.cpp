#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>

namespace {

using namespace vss::math;

Limits limits() { return {4096U, 262144U, 0U}; }

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows);

void scalarAndRectangularTransforms() {
  const Matrix left = Matrix(1U, 1U, 2.0);
  const Matrix cross = Matrix(1U, 1U, 0.5);
  const Matrix right = Matrix(1U, 1U, 3.0);
  const auto scalar = covariance::crossCovarianceFromMaps(
      left.view(), cross.view(), right.view(), limits());
  VSS_CHECK(scalar.hasValue());
  VSS_CHECK(scalar.value().at(0U, 0U) == 3.0);

  const Matrix left2 = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const Matrix block = matrix({{1.0, 1.0}, {1.0, -1.0}});
  const Matrix right1 = matrix({{-1.0, -1.0}});
  const auto rectangular = covariance::crossCovarianceFromMaps(
      left2.view(), block.view(), right1.view(), limits());
  VSS_CHECK(rectangular.hasValue());
  VSS_CHECK(rectangular.value().rows() == 2U);
  VSS_CHECK(rectangular.value().columns() == 1U);
  VSS_CHECK(rectangular.value().at(0U, 0U) == -2.0);
  VSS_CHECK(rectangular.value().at(1U, 0U) == 0.0);

  const Matrix reversedLeft = right1;
  const Matrix reversedBlock = matrix({{1.0, 1.0}, {1.0, -1.0}});
  const Matrix reversedRight = left2;
  const auto reversed = covariance::crossCovarianceFromMaps(
      reversedLeft.view(), reversedBlock.view(), reversedRight.view(),
      limits());
  VSS_CHECK(reversed.hasValue());
  VSS_CHECK(reversed.value().rows() == 1U && reversed.value().columns() == 2U);
  VSS_CHECK(reversed.value().at(0U, 0U) == -2.0);
  VSS_CHECK(reversed.value().at(0U, 1U) == 0.0);
}

void emptyScaledAndInvalidInputs() {
  const Matrix left(2U, 0U, 0.0);
  const Matrix cross(0U, 1U, 0.0);
  const Matrix right(1U, 1U, 2.0);
  const auto empty = covariance::crossCovarianceFromMaps(
      left.view(), cross.view(), right.view(), limits());
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().rows() == 2U && empty.value().columns() == 1U);
  VSS_CHECK(empty.value().at(0U, 0U) == 0.0);
  VSS_CHECK(empty.value().at(1U, 0U) == 0.0);

  const Matrix scaleLeft(1U, 1U, 2.0);
  const Matrix maximum(1U, 1U, std::numeric_limits<Scalar>::max());
  const Matrix quarter(1U, 1U, 0.25);
  const auto scaled = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), maximum.view(), quarter.view(), limits());
  VSS_CHECK(scaled.hasValue());
  VSS_CHECK(scaled.value().at(0U, 0U) ==
            std::numeric_limits<Scalar>::max() / 2.0);

  const Matrix badShape(2U, 2U, 0.0);
  const auto mismatch = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), badShape.view(), quarter.view(), limits());
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);

  const Matrix nonFinite(1U, 1U, std::numeric_limits<Scalar>::infinity());
  const auto invalid = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), nonFinite.view(), quarter.view(), limits());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NonFiniteInput);

  const auto denied = covariance::crossCovarianceFromMaps(
      scaleLeft.view(), maximum.view(), quarter.view(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

PropagationResult propagation(Vector mean, Matrix covariance) {
  return {std::move(mean), std::move(covariance), Approximation::exact, {},
          PropagationCovarianceCheck::positive_semidefinite};
}

void uncertaintySummaries() {
  auto value = propagation(Vector{std::array<Scalar, 2>{2.0, -3.0}},
                           matrix({{4.0, 0.0}, {0.0, 9.0}}));
  const auto summary = covariance::summarizeUncertainty(
      value, 2.0, std::nullopt, std::nullopt, limits());
  VSS_CHECK(summary.hasValue());
  VSS_CHECK(summary.value().nominal.at(0U) == 2.0);
  VSS_CHECK(summary.value().standard.at(0U) == 2.0);
  VSS_CHECK(summary.value().standard.at(1U) == 3.0);
  VSS_CHECK(summary.value().expanded.at(0U) == 4.0);
  VSS_CHECK(summary.value().expanded.at(1U) == 6.0);
  VSS_CHECK(summary.value().coverage_factor == 2.0);
  VSS_CHECK(!summary.value().lower && !summary.value().upper);

  const Vector lower{std::array<Scalar, 2>{-1.0, -8.0}};
  const Vector upper{std::array<Scalar, 2>{9.0, 1.0}};
  const auto bounded = covariance::summarizeUncertainty(
      value, 0.0, lower.view(), upper.view(), limits());
  VSS_CHECK(bounded.hasValue());
  VSS_CHECK(bounded.value().lower->at(0U) == -1.0);
  VSS_CHECK(bounded.value().upper->at(1U) == 1.0);
  VSS_CHECK(bounded.value().expanded.at(0U) == 0.0);

  auto rejected = [&](const PropagationResult& candidate, Scalar factor,
                      std::optional<ConstVectorView> low,
                      std::optional<ConstVectorView> high, Limits cap,
                      ErrorCode expected) {
    const auto result = covariance::summarizeUncertainty(
        candidate, factor, low, high, cap);
    VSS_CHECK(!result.hasValue());
    VSS_CHECK(result.status().code() == expected);
  };
  rejected(value, -1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::InvalidDomain);
  rejected(value, 1.0, lower.view(), std::nullopt, limits(),
           ErrorCode::InvalidDomain);
  const Vector wrongSize(1U);
  rejected(value, 1.0, wrongSize.view(), upper.view(), limits(),
           ErrorCode::DimensionMismatch);
  rejected(value, 1.0, lower.view(), wrongSize.view(), limits(),
           ErrorCode::DimensionMismatch);
  rejected(value, 1.0, Vector{std::array<Scalar, 2>{0.0,
             std::numeric_limits<Scalar>::quiet_NaN()}}.view(), upper.view(),
           limits(), ErrorCode::NonFiniteInput);
  rejected(value, std::numeric_limits<Scalar>::infinity(), std::nullopt,
           std::nullopt, limits(), ErrorCode::NonFiniteInput);
  rejected(value, 1.0, lower.view(), upper.view(), {0U, 0U, 0U},
           ErrorCode::ResourceLimit);

  auto mismatch = value;
  mismatch.covariance = Matrix(1U, 1U, 1.0);
  rejected(mismatch, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::DimensionMismatch);
  auto nonFinite = value;
  nonFinite.covariance.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  rejected(nonFinite, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::NonFiniteInput);

  auto indefinite = propagation(Vector(2U), matrix({{1.0, 1.25}, {1.25, 1.0}}));
  // The producer's positive tag is not trusted in place of the matrix check.
  rejected(indefinite, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::NotPositiveSemidefinite);
  auto asymmetric = propagation(Vector(2U), matrix({{1.0, 0.0}, {0.25, 1.0}}));
  rejected(asymmetric, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::NotSymmetric);
  auto asymmetricIndefinite = propagation(Vector(2U), matrix({{-1.0, 0.0},
                                                               {0.25, 1.0}}));
  rejected(asymmetricIndefinite, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::NotSymmetric);
  auto negativeDiagonal = propagation(Vector(1U), Matrix(1U, 1U, -1.0));
  rejected(negativeDiagonal, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::NotPositiveSemidefinite);

  auto empty = propagation(Vector{}, Matrix(0U, 0U));
  rejected(empty, 1.0, std::nullopt, std::nullopt, limits(),
           ErrorCode::InsufficientSamples);
  auto roundoffSpectrum =
      propagation(Vector(2U), matrix({{1.0, std::nextafter(1.0, 2.0)},
                                      {std::nextafter(1.0, 2.0), 1.0}}));
  const auto roundoff = covariance::summarizeUncertainty(
      roundoffSpectrum, 1.0, std::nullopt, std::nullopt, limits());
  VSS_CHECK(roundoff.hasValue());
  VSS_CHECK(roundoff.value().standard.at(0U) == 1.0);
  VSS_CHECK(roundoff.value().standard.at(1U) == 1.0);
  auto overflow = propagation(Vector(1U), Matrix(1U, 1U, 8.0));
  rejected(overflow, std::numeric_limits<Scalar>::max(), std::nullopt,
           std::nullopt, limits(), ErrorCode::NonFiniteResult);

  auto signedZero = propagation(Vector(1U), Matrix(1U, 1U, -0.0));
  const auto zero = covariance::summarizeUncertainty(
      signedZero, 1.0, std::nullopt, std::nullopt, limits());
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().standard.at(0U) == 0.0);
  VSS_CHECK(!std::signbit(zero.value().standard.at(0U)));
}

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

} // namespace

int main() {
  scalarAndRectangularTransforms();
  emptyScaledAndInvalidInputs();
  uncertaintySummaries();
  return 0;
}
