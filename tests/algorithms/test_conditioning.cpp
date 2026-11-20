#include "Test.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/RankAware.hpp>
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

template <class T> const T& required(const std::optional<T>& value) {
  if (!value.has_value()) {
    throw std::logic_error("expected an engaged optional test result");
  }
  return value.value();
}

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

void generalizedLinearEstimation() {
  const auto limits = vss::math::Limits{4096U, 262144U, 0U};
  SolvePolicy supported = policy();
  supported.rank = RankPolicy::supported_subspace;

  const Matrix identity = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const Matrix observations = matrix({{2.0}, {-1.0}});
  const Matrix correlated = matrix({{4.0, 1.0}, {1.0, 2.0}});
  const auto identityEstimate = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), correlated.view(), supported,
      limits);
  VSS_CHECK(identityEstimate.hasValue());
  VSS_CHECK(identityEstimate.value().covariance.has_value());
  VSS_CHECK(std::abs(identityEstimate.value().parameters.at(0U, 0U) - 2.0) <
            1.0e-11);
  VSS_CHECK(std::abs(identityEstimate.value().parameters.at(1U, 0U) + 1.0) <
            1.0e-11);
  VSS_CHECK(std::abs(required(identityEstimate.value().covariance).at(0U, 0U) -
                     4.0) < 1.0e-11);
  VSS_CHECK(std::abs(required(identityEstimate.value().covariance).at(0U, 1U) -
                     1.0) < 1.0e-11);
  VSS_CHECK(std::abs(required(identityEstimate.value().covariance).at(1U, 1U) -
                     2.0) < 1.0e-11);

  const Matrix intercept = matrix({{1.0}, {1.0}});
  const Matrix pair = matrix({{1.0}, {3.0}});
  const Matrix shared = matrix({{4.0, 1.0}, {1.0, 1.0}});
  const auto sharedEstimate = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), pair.view(), shared.view(), supported, limits);
  VSS_CHECK(sharedEstimate.hasValue());
  VSS_CHECK(std::abs(sharedEstimate.value().parameters.at(0U, 0U) - 3.0) <
            1.0e-11);
  VSS_CHECK(std::abs(required(sharedEstimate.value().covariance).at(0U, 0U) -
                     1.0) < 1.0e-11);
  const SolvePolicy strictFull = policy();
  const auto strictEstimate = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), pair.view(), shared.view(), strictFull, limits);
  VSS_CHECK(strictEstimate.hasValue());

  const Matrix diagonal = matrix({{1.0, 0.0}, {0.0, 3.0}});
  const auto weighted = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), pair.view(), diagonal.view(), supported, limits);
  VSS_CHECK(weighted.hasValue());
  VSS_CHECK(std::abs(weighted.value().parameters.at(0U, 0U) - 1.5) < 1.0e-11);
  VSS_CHECK(std::abs(required(weighted.value().covariance).at(0U, 0U) - 0.75) <
            1.0e-11);

  const Matrix exactCovariance = matrix({{1.0, 0.0}, {0.0, 0.0}});
  const Matrix exactObservations = matrix({{1.0}, {3.0}});
  const auto exact = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), exactObservations.view(), exactCovariance.view(),
      supported, limits);
  VSS_CHECK(exact.hasValue());
  VSS_CHECK(exact.value().parameters.at(0U, 0U) == 3.0);
  VSS_CHECK(required(exact.value().covariance).at(0U, 0U) == 0.0);
  VSS_CHECK(std::abs(exact.value().residual_norms.at(0U) - 2.0) < 1.0e-11);

  SolvePolicy thresholded = supported;
  thresholded.rank_abs = 1.0e-12;
  const Matrix smallPositive = matrix({{1.0, 0.0}, {0.0, 1.0e-14}});
  const auto thresholdExact = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), smallPositive.view(), thresholded,
      limits);
  VSS_CHECK(thresholdExact.hasValue());
  VSS_CHECK(thresholdExact.value().parameters.at(1U, 0U) == -1.0);
  VSS_CHECK(required(thresholdExact.value().covariance).at(1U, 1U) == 0.0);

  const Matrix mixedDesign = matrix({{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}});
  const Matrix mixedObservations = matrix({{7.0, 8.0}, {1.0, 2.0}, {2.0, 5.0}});
  const Matrix mixedCovariance =
      matrix({{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 4.0}});
  const auto mixed = vss::math::linear::generalizedLinearEstimate(
      mixedDesign.view(), mixedObservations.view(), mixedCovariance.view(),
      supported, limits);
  VSS_CHECK(mixed.hasValue());
  VSS_CHECK(mixed.value().parameters.rows() == 2U);
  VSS_CHECK(mixed.value().parameters.columns() == 2U);
  VSS_CHECK(mixed.value().parameters.at(0U, 0U) == 7.0);
  VSS_CHECK(mixed.value().parameters.at(0U, 1U) == 8.0);
  VSS_CHECK(required(mixed.value().covariance).at(0U, 0U) == 0.0);
  VSS_CHECK(std::abs(required(mixed.value().covariance).at(0U, 1U)) < 1.0e-11);
  VSS_CHECK(std::abs(mixed.value().parameters.at(1U, 0U) - 1.2) < 1.0e-11);
  VSS_CHECK(std::abs(mixed.value().parameters.at(1U, 1U) - 2.6) < 1.0e-11);
  VSS_CHECK(std::abs(required(mixed.value().covariance).at(1U, 1U) - 0.8) <
            1.0e-11);
  VSS_CHECK(std::abs(mixed.value().residual_norms.at(0U) *
                         mixed.value().residual_norms.at(0U) -
                     0.2) < 1.0e-11);
  VSS_CHECK(std::abs(mixed.value().residual_norms.at(1U) *
                         mixed.value().residual_norms.at(1U) -
                     1.8) < 1.0e-11);

  const Matrix duplicate = matrix({{1.0, 1.0}});
  const Matrix one = matrix({{2.0}});
  const Matrix unit = matrix({{1.0}});
  const auto unidentified = vss::math::linear::generalizedLinearEstimate(
      duplicate.view(), one.view(), unit.view(), supported, limits);
  VSS_CHECK(unidentified.hasValue());
  VSS_CHECK(unidentified.value().rank == 1U);
  VSS_CHECK(!unidentified.value().covariance.has_value());
  VSS_CHECK(std::abs(unidentified.value().parameters.at(0U, 0U) - 1.0) <
            1.0e-11);
  VSS_CHECK(std::abs(unidentified.value().parameters.at(1U, 0U) - 1.0) <
            1.0e-11);
  VSS_CHECK(
      std::abs(required(unidentified.value().identified_basis).at(0U, 0U) -
               required(unidentified.value().identified_basis).at(1U, 0U)) <
      1.0e-11);
  VSS_CHECK(
      std::abs(required(unidentified.value().identified_basis).at(0U, 0U) *
                   required(unidentified.value().identified_basis).at(0U, 0U) -
               0.5) < 1.0e-11);

  const Matrix invalidCovariance = matrix({{1.0, 2.0}, {2.0, 1.0}});
  const auto invalid = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), invalidCovariance.view(), supported,
      limits);
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NotPositiveSemidefinite);
  const Matrix nonFiniteObservations =
      matrix({{std::numeric_limits<Scalar>::infinity()}, {1.0}});
  const auto nonFinite = vss::math::linear::generalizedLinearEstimate(
      identity.view(), nonFiniteObservations.view(), correlated.view(),
      supported, limits);
  VSS_CHECK(!nonFinite.hasValue());
  VSS_CHECK(nonFinite.status().code() == ErrorCode::NonFiniteInput);

  const Matrix asymmetric = matrix({{1.0, 0.25}, {0.2, 1.0}});
  const auto asymmetricResult = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), asymmetric.view(), supported,
      limits);
  VSS_CHECK(!asymmetricResult.hasValue());
  VSS_CHECK(asymmetricResult.status().code() == ErrorCode::NotSymmetric);

  const Matrix inconsistent = matrix({{1.0}, {2.0}});
  const Matrix noNoise(2U, 2U, 0.0);
  const auto inconsistentResult = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), inconsistent.view(), noNoise.view(), supported, limits);
  VSS_CHECK(!inconsistentResult.hasValue());
  VSS_CHECK(inconsistentResult.status().code() == ErrorCode::InvalidDomain);

  const SolvePolicy strict = policy();
  const auto strictRank = vss::math::linear::generalizedLinearEstimate(
      duplicate.view(), one.view(), unit.view(), strict, limits);
  VSS_CHECK(!strictRank.hasValue());
  VSS_CHECK(strictRank.status().code() == ErrorCode::RankDeficient);
  const auto strictSupport = vss::math::linear::generalizedLinearEstimate(
      intercept.view(), pair.view(), exactCovariance.view(), strict, limits);
  VSS_CHECK(!strictSupport.hasValue());
  VSS_CHECK(strictSupport.status().code() == ErrorCode::RankDeficient);

  const Matrix noColumns(2U, 0U, 0.0);
  const auto emptyRhs = vss::math::linear::generalizedLinearEstimate(
      identity.view(), noColumns.view(), correlated.view(), supported, limits);
  VSS_CHECK(emptyRhs.hasValue());
  VSS_CHECK(emptyRhs.value().parameters.rows() == 2U);
  VSS_CHECK(emptyRhs.value().parameters.columns() == 0U);
  VSS_CHECK(emptyRhs.value().residual_norms.empty());
  VSS_CHECK(emptyRhs.value().covariance.has_value());

  const Matrix noRows(0U, 2U, 0.0);
  const Matrix noObservationCovariance(0U, 0U, 0.0);
  const auto insufficient = vss::math::linear::generalizedLinearEstimate(
      noRows.view(), Matrix(0U, 1U, 0.0).view(), noObservationCovariance.view(),
      supported, limits);
  VSS_CHECK(!insufficient.hasValue());
  VSS_CHECK(insufficient.status().code() == ErrorCode::InsufficientSamples);

  SolvePolicy exhausted = supported;
  exhausted.iteration.max_iterations = 0U;
  const auto noConvergence = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), correlated.view(), exhausted,
      limits);
  VSS_CHECK(!noConvergence.hasValue());
  VSS_CHECK(noConvergence.status().code() == ErrorCode::NoConvergence);

  const auto denied = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), correlated.view(), supported,
      {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);

  SolvePolicy illConditioned = supported;
  illConditioned.reciprocal_condition_floor = 1.0e-3;
  const Matrix poorlyConditioned = matrix({{1.0, 0.0}, {0.0, 1.0e-6}});
  const auto conditionFailure = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), poorlyConditioned.view(),
      illConditioned, limits);
  VSS_CHECK(!conditionFailure.hasValue());
  VSS_CHECK(conditionFailure.status().code() == ErrorCode::IllConditioned);

  SolvePolicy invalidPolicy = supported;
  invalidPolicy.rank_rel = -1.0;
  const auto invalidTolerance = vss::math::linear::generalizedLinearEstimate(
      identity.view(), observations.view(), correlated.view(), invalidPolicy,
      limits);
  VSS_CHECK(!invalidTolerance.hasValue());
  VSS_CHECK(invalidTolerance.status().code() == ErrorCode::InvalidTolerance);

  SolvePolicy zeroCutoff = supported;
  zeroCutoff.rank_abs = 0.0;
  zeroCutoff.rank_rel = 0.0;
  const Matrix tinyDesign = matrix({{1.0e-200}});
  const Matrix tinyObservation = matrix({{0.0}});
  const Matrix scalarUnit = matrix({{1.0}});
  const auto nonrepresentable = vss::math::linear::generalizedLinearEstimate(
      tinyDesign.view(), tinyObservation.view(), scalarUnit.view(), zeroCutoff,
      limits);
  VSS_CHECK(!nonrepresentable.hasValue());
  VSS_CHECK(nonrepresentable.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  analyticSchurAndConditionalCovariance();
  exactDependenceAndIndexValidation();
  singularPoliciesAndNegativeSpectrum();
  conditionalOrderAndValidation();
  failureAndScaleBoundaries();
  whiteningAndMahalanobis();
  generalizedLinearEstimation();
  return 0;
}
