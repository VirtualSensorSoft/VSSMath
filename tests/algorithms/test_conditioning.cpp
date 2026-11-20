#include "Test.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::RankPolicy;
using vss::math::Scalar;
using vss::math::SolvePolicy;
using vss::math::covariance::CovarianceKind;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Extent row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Extent column = 0U;
    for (const Scalar value : values) {
      result.at(row, column) = value;
      ++column;
    }
    ++row;
  }
  return result;
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 64U;
  result.rank = RankPolicy::reject;
  return result;
}

template <class T> const T& require(const vss::math::Result<T>& result) {
  VSS_CHECK(result.hasValue());
  return result.value();
}

void analyticSchurAndConditionalCovariance() {
  const Matrix aa = matrix({{4.0}});
  const Matrix ab = matrix({{3.0}});
  const Matrix bb = matrix({{9.0}});
  const auto schur = vss::math::covariance::schurComplement(
      aa.view(), ab.view(), bb.view(), policy(), {64U, 4096U, 0U});
  VSS_CHECK(schur.hasValue());
  VSS_CHECK(std::abs(schur.value().at(0U, 0U) - 3.0) < 1.0e-12);

  const Matrix joint = matrix({{4.0, 3.0}, {3.0, 9.0}});
  const std::array<Index, 1U> keep{0U};
  const std::array<Index, 1U> given{1U};
  const auto conditional = vss::math::covariance::conditionalCovariance(
      joint.view(), keep, given, policy(), {128U, 8192U, 0U});
  VSS_CHECK(conditional.hasValue());
  VSS_CHECK(conditional.value().rank == 1U);
  VSS_CHECK(conditional.value().classification ==
            CovarianceKind::positive_definite);
  VSS_CHECK(std::abs(conditional.value().covariance.at(0U, 0U) - 3.0) <
            1.0e-12);
}

void exactDependenceAndIndexValidation() {
  const Matrix joint = matrix({{1.0, 1.0}, {1.0, 1.0}});
  const std::array<Index, 1U> keep{0U};
  const std::array<Index, 1U> given{1U};
  const auto exact = vss::math::covariance::conditionalCovariance(
      joint.view(), keep, given, policy(), {128U, 8192U, 0U});
  VSS_CHECK(exact.hasValue());
  VSS_CHECK(exact.value().rank == 0U);
  VSS_CHECK(exact.value().classification ==
            CovarianceKind::semidefinite_singular);
  VSS_CHECK(exact.value().covariance.at(0U, 0U) == 0.0);

  const std::array<Index, 2U> duplicate{0U, 0U};
  const auto invalid = vss::math::covariance::conditionalCovariance(
      joint.view(), duplicate, {}, policy(), {128U, 8192U, 0U});
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::InvalidDomain);
}

void singularPoliciesAndNegativeSpectrum() {
  const Matrix aa = matrix({{4.0}});
  const Matrix redundant = matrix({{4.0, 4.0}, {4.0, 4.0}});
  const Matrix compatible = matrix({{2.0, 2.0}});
  const auto rejected = vss::math::covariance::schurComplement(
      aa.view(), compatible.view(), redundant.view(), policy(),
      {128U, 8192U, 0U});
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::RankDeficient);

  SolvePolicy supported = policy();
  supported.rank = RankPolicy::supported_subspace;
  const auto supportedResult = vss::math::covariance::schurComplement(
      aa.view(), compatible.view(), redundant.view(), supported,
      {128U, 8192U, 0U});
  VSS_CHECK(supportedResult.hasValue());
  VSS_CHECK(std::abs(supportedResult.value().at(0U, 0U) - 3.0) < 1.0e-12);

  const Matrix incompatible = matrix({{2.0, -2.0}});
  const auto incompatibleResult = vss::math::covariance::schurComplement(
      aa.view(), incompatible.view(), redundant.view(), supported,
      {128U, 8192U, 0U});
  VSS_CHECK(!incompatibleResult.hasValue());
  VSS_CHECK(incompatibleResult.status().code() == ErrorCode::InvalidDomain);
  SolvePolicy minimum = policy();
  minimum.rank = RankPolicy::minimum_norm;
  const auto minimumResult = vss::math::covariance::schurComplement(
      aa.view(), incompatible.view(), redundant.view(), minimum,
      {128U, 8192U, 0U});
  VSS_CHECK(minimumResult.hasValue());
  VSS_CHECK(minimumResult.value().at(0U, 0U) == 4.0);

  const Matrix negative = matrix({{-4.0}});
  const Matrix loading = matrix({{3.0}});
  const auto indefinite = vss::math::covariance::schurComplement(
      aa.view(), loading.view(), negative.view(), policy(), {64U, 4096U, 0U});
  VSS_CHECK(indefinite.hasValue());
  VSS_CHECK(std::abs(indefinite.value().at(0U, 0U) - 6.25) < 1.0e-12);
}

void conditionalOrderAndValidation() {
  const Matrix joint = matrix({{4.0, 3.0}, {3.0, 9.0}});
  const std::array<Index, 2U> reversed{1U, 0U};
  const auto marginal = vss::math::covariance::conditionalCovariance(
      joint.view(), reversed, {}, policy(), {128U, 8192U, 0U});
  VSS_CHECK(marginal.hasValue());
  VSS_CHECK(marginal.value().covariance.at(0U, 0U) == 9.0);
  VSS_CHECK(marginal.value().covariance.at(1U, 1U) == 4.0);
  VSS_CHECK(marginal.value().covariance.at(0U, 1U) == 3.0);

  const std::array<Index, 1U> first{0U};
  const auto overlap = vss::math::covariance::conditionalCovariance(
      joint.view(), first, first, policy(), {128U, 8192U, 0U});
  VSS_CHECK(!overlap.hasValue());
  VSS_CHECK(overlap.status().code() == ErrorCode::InvalidDomain);
  const std::array<Index, 1U> outOfRange{2U};
  const auto invalidIndex = vss::math::covariance::conditionalCovariance(
      joint.view(), outOfRange, {}, policy(), {128U, 8192U, 0U});
  VSS_CHECK(!invalidIndex.hasValue());
  VSS_CHECK(invalidIndex.status().code() == ErrorCode::IndexOutOfRange);
  const auto emptyKeep = vss::math::covariance::conditionalCovariance(
      joint.view(), {}, first, policy(), {128U, 8192U, 0U});
  VSS_CHECK(!emptyKeep.hasValue());
  VSS_CHECK(emptyKeep.status().code() == ErrorCode::InsufficientSamples);

  const Matrix asymmetric = matrix({{1.0, 2.0}, {1.0, 1.0}});
  const auto asymSchur = vss::math::covariance::schurComplement(
      asymmetric.view(), Matrix(2U, 1U).view(), matrix({{1.0}}).view(),
      policy(), {128U, 8192U, 0U});
  VSS_CHECK(!asymSchur.hasValue());
  VSS_CHECK(asymSchur.status().code() == ErrorCode::NotSymmetric);
  const auto asymJoint = vss::math::covariance::conditionalCovariance(
      asymmetric.view(), first, std::array<Index, 1U>{1U}, policy(),
      {128U, 8192U, 0U});
  VSS_CHECK(!asymJoint.hasValue());
  VSS_CHECK(asymJoint.status().code() == ErrorCode::NotSymmetric);
}

void failureAndScaleBoundaries() {
  const Matrix aa = matrix({{4.0}});
  const Matrix ab = matrix({{3.0}});
  const Matrix bb = matrix({{9.0}});
  const auto denied = vss::math::covariance::schurComplement(
      aa.view(), ab.view(), bb.view(), policy(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);

  SolvePolicy invalidPolicy = policy();
  invalidPolicy.rank_rel = 1.0;
  const auto invalidTolerance = vss::math::covariance::schurComplement(
      aa.view(), ab.view(), bb.view(), invalidPolicy, {128U, 8192U, 0U});
  VSS_CHECK(!invalidTolerance.hasValue());
  VSS_CHECK(invalidTolerance.status().code() == ErrorCode::InvalidTolerance);

  SolvePolicy noSweeps = policy();
  noSweeps.iteration.max_iterations = 0U;
  const Matrix nonDiagonal = matrix({{2.0, 1.0}, {1.0, 2.0}});
  const Matrix twoByTwoCross = matrix({{1.0, 0.0}});
  const auto noConvergence = vss::math::covariance::schurComplement(
      aa.view(), twoByTwoCross.view(), nonDiagonal.view(), noSweeps,
      {256U, 16384U, 0U});
  VSS_CHECK(!noConvergence.hasValue());
  VSS_CHECK(noConvergence.status().code() == ErrorCode::NoConvergence);

  SolvePolicy lowCondition = policy();
  lowCondition.reciprocal_condition_floor = 0.5;
  const Matrix unequal = matrix({{4.0, 0.0}, {0.0, 1.0}});
  const auto illConditioned = vss::math::covariance::schurComplement(
      aa.view(), twoByTwoCross.view(), unequal.view(), lowCondition,
      {256U, 16384U, 0U});
  VSS_CHECK(!illConditioned.hasValue());
  VSS_CHECK(illConditioned.status().code() == ErrorCode::IllConditioned);

  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Matrix zero = matrix({{0.0}});
  const Matrix huge = matrix({{maximum}});
  const auto scaled = vss::math::covariance::schurComplement(
      zero.view(), huge.view(), huge.view(), policy(), {64U, 4096U, 0U});
  VSS_CHECK(scaled.hasValue());
  VSS_CHECK(scaled.value().at(0U, 0U) == -maximum);
  const Matrix half = matrix({{0.5}});
  const auto overflow = vss::math::covariance::schurComplement(
      zero.view(), huge.view(), half.view(), policy(), {64U, 4096U, 0U});
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
  const Matrix infinite = matrix({{std::numeric_limits<Scalar>::infinity()}});
  const auto nonFinite = vss::math::covariance::schurComplement(
      aa.view(), infinite.view(), bb.view(), policy(), {64U, 4096U, 0U});
  VSS_CHECK(!nonFinite.hasValue());
  VSS_CHECK(nonFinite.status().code() == ErrorCode::NonFiniteInput);
}

void whiteningAndMahalanobis() {
  const Matrix covariance = matrix({{4.0, 3.0}, {3.0, 9.0}});
  const vss::math::Vector x{std::array<Scalar, 2U>{2.0, 3.0}};
  const auto whitened = vss::math::covariance::whitenSupported(
      covariance.view(), x.view(), policy(), {128U, 8192U, 0U});
  VSS_CHECK(whitened.hasValue());
  VSS_CHECK(whitened.value().rank == 2U);
  VSS_CHECK(
      std::abs(whitened.value().value.at(0U) * whitened.value().value.at(0U) +
               whitened.value().value.at(1U) * whitened.value().value.at(1U) -
               4.0 / 3.0) < 1.0e-12);
  const auto distance =
      vss::math::covariance::mahalanobisSquared(whitened.value(), x.view());
  VSS_CHECK(distance.hasValue());
  VSS_CHECK(std::abs(distance.value() - 4.0 / 3.0) < 1.0e-12);

  const Matrix singular = matrix({{2.0, 2.0}, {2.0, 2.0}});
  const vss::math::Vector onSupport{std::array<Scalar, 2U>{2.0, 2.0}};
  const vss::math::Vector offSupport{std::array<Scalar, 2U>{2.0, -2.0}};
  SolvePolicy supported = policy();
  supported.rank = RankPolicy::supported_subspace;
  const auto singularWhite = vss::math::covariance::whitenSupported(
      singular.view(), onSupport.view(), supported, {128U, 8192U, 0U});
  VSS_CHECK(singularWhite.hasValue());
  VSS_CHECK(singularWhite.value().rank == 1U);
  VSS_CHECK(singularWhite.value().support_basis.rows() == 2U);
  VSS_CHECK(singularWhite.value().support_basis.columns() == 1U);
  VSS_CHECK(std::abs(singularWhite.value().value.at(0U) - 1.0) < 1.0e-12);
  VSS_CHECK(std::abs(singularWhite.value().value.at(1U) - 1.0) < 1.0e-12);
  VSS_CHECK(singularWhite.value().null_residual < 1.0e-12);
  const auto supportedDistance = vss::math::covariance::mahalanobisSquared(
      singularWhite.value(), onSupport.view());
  VSS_CHECK(supportedDistance.hasValue());
  VSS_CHECK(std::abs(supportedDistance.value() - 2.0) < 1.0e-12);
  const auto offSupportWhite = vss::math::covariance::whitenSupported(
      singular.view(), offSupport.view(), supported, {128U, 8192U, 0U});
  VSS_CHECK(!offSupportWhite.hasValue());
  VSS_CHECK(offSupportWhite.status().code() == ErrorCode::InvalidDomain);
  const auto offSupportDistance = vss::math::covariance::mahalanobisSquared(
      singularWhite.value(), offSupport.view());
  VSS_CHECK(!offSupportDistance.hasValue());
  VSS_CHECK(offSupportDistance.status().code() == ErrorCode::InvalidDomain);
  const vss::math::Vector mixedSupport{std::array<Scalar, 2U>{4.0, 0.0}};
  const auto mixedDistance = vss::math::covariance::mahalanobisSquared(
      singularWhite.value(), mixedSupport.view());
  VSS_CHECK(!mixedDistance.hasValue());
  VSS_CHECK(mixedDistance.status().code() == ErrorCode::InvalidDomain);
  const auto rejectedRank = vss::math::covariance::whitenSupported(
      singular.view(), onSupport.view(), policy(), {128U, 8192U, 0U});
  VSS_CHECK(!rejectedRank.hasValue());
  VSS_CHECK(rejectedRank.status().code() == ErrorCode::RankDeficient);

  const vss::math::Vector zero(2U, 0.0);
  const auto zeroWhitening = vss::math::covariance::whitenSupported(
      Matrix(2U, 2U).view(), zero.view(), supported, {128U, 8192U, 0U});
  VSS_CHECK(zeroWhitening.hasValue());
  VSS_CHECK(zeroWhitening.value().rank == 0U);
  const auto zeroDistance = vss::math::covariance::mahalanobisSquared(
      zeroWhitening.value(), zero.view());
  VSS_CHECK(zeroDistance.hasValue() && zeroDistance.value() == 0.0);
  const auto zeroOffSupport = vss::math::covariance::mahalanobisSquared(
      zeroWhitening.value(), onSupport.view());
  VSS_CHECK(!zeroOffSupport.hasValue());
  VSS_CHECK(zeroOffSupport.status().code() == ErrorCode::InvalidDomain);

  const auto wrongVector = vss::math::covariance::whitenSupported(
      covariance.view(), vss::math::Vector(1U).view(), policy(),
      {128U, 8192U, 0U});
  VSS_CHECK(!wrongVector.hasValue());
  VSS_CHECK(wrongVector.status().code() == ErrorCode::DimensionMismatch);
  const auto wrongDistance = vss::math::covariance::mahalanobisSquared(
      whitened.value(), vss::math::Vector(1U).view());
  VSS_CHECK(!wrongDistance.hasValue());
  VSS_CHECK(wrongDistance.status().code() == ErrorCode::DimensionMismatch);
  const auto resource = vss::math::covariance::whitenSupported(
      covariance.view(), x.view(), policy(), {0U, 0U, 0U});
  VSS_CHECK(!resource.hasValue());
  VSS_CHECK(resource.status().code() == ErrorCode::ResourceLimit);

  const vss::math::Vector nonFinite(2U,
                                    std::numeric_limits<Scalar>::infinity());
  const auto badInput = vss::math::covariance::whitenSupported(
      singular.view(), nonFinite.view(), supported, {128U, 8192U, 0U});
  VSS_CHECK(!badInput.hasValue());
  VSS_CHECK(badInput.status().code() == ErrorCode::NonFiniteInput);
  const auto badDistance = vss::math::covariance::mahalanobisSquared(
      singularWhite.value(), nonFinite.view());
  VSS_CHECK(!badDistance.hasValue());
  VSS_CHECK(badDistance.status().code() == ErrorCode::NonFiniteInput);
  auto malformed = singularWhite.value();
  malformed.support_basis = Matrix(2U, 2U);
  const auto malformedDistance =
      vss::math::covariance::mahalanobisSquared(malformed, onSupport.view());
  VSS_CHECK(!malformedDistance.hasValue());
  VSS_CHECK(malformedDistance.status().code() == ErrorCode::DimensionMismatch);
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const vss::math::Vector huge(2U, maximum);
  const auto overflowDistance = vss::math::covariance::mahalanobisSquared(
      singularWhite.value(), huge.view());
  VSS_CHECK(!overflowDistance.hasValue());
  VSS_CHECK(overflowDistance.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  analyticSchurAndConditionalCovariance();
  exactDependenceAndIndexValidation();
  singularPoliciesAndNegativeSpectrum();
  conditionalOrderAndValidation();
  failureAndScaleBoundaries();
  whiteningAndMahalanobis();
  return 0;
}
