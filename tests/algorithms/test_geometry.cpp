#include "Test.hpp"
#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/Tensor.hpp"
#include "vss/math/TensorLimits.hpp"
#include "vss/math/TensorShape.hpp"
#include "vss/math/covariance/PropagationResult.hpp"
#include "vss/math/statistics/Moments.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/AffineMaps.hpp>
#include <vss/math/Geometry.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Circular.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::statistics;

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>{values.begin(), values.size()});
}

Result<Tensor> zeroNormalizationHessian() {
  const std::array<Extent, 3U> dimensions{2U, 2U, 2U};
  const auto shape = TensorShape::create(dimensions);
  if (!shape) {
    return Result<Tensor>::failure(shape.status().code(),
                                   "normalization Hessian fallback failed");
  }
  return Tensor::create(shape.value(), 0.0, TensorLimits(8U));
}

constexpr Limits limits{4096U, 4096U * sizeof(Scalar), 0U};
constexpr Tolerance tight{1.0e-13, 1.0e-13};

void wrappedAngles() {
  const auto acrossCut = wrappedAngularResidual(5.0, 355.0, 360.0);
  VSS_CHECK(acrossCut && acrossCut.value() == 10.0);
  const auto halfPeriod = wrappedAngularResidual(180.0, 0.0, 360.0);
  VSS_CHECK(halfPeriod && halfPeriod.value() == -180.0);
  VSS_CHECK(!wrappedAngularResidual(1.0, 0.0, 0.0));
  VSS_CHECK(!wrappedAngularResidual(1.0, 0.0,
                                    std::numeric_limits<Scalar>::quiet_NaN()));
  VSS_CHECK(!wrappedAngularResidual(std::numeric_limits<Scalar>::infinity(),
                                    0.0, 360.0));
  const auto huge =
      wrappedAngularResidual(std::numeric_limits<Scalar>::max(),
                             -std::numeric_limits<Scalar>::max(), 360.0);
  VSS_CHECK(huge && std::isfinite(huge.value()));
  const auto negativeTie = wrappedAngularResidual(-180.0, 0.0, 360.0);
  VSS_CHECK(negativeTie && negativeTie.value() == -180.0);
  const Scalar belowCut = std::nextafter(180.0, 0.0);
  const Scalar aboveCut = std::nextafter(180.0, 360.0);
  const auto belowResidual = wrappedAngularResidual(belowCut, 0.0, 360.0);
  const auto cutResidual = wrappedAngularResidual(180.0, 0.0, 360.0);
  const auto aboveResidual = wrappedAngularResidual(aboveCut, 0.0, 360.0);
  VSS_CHECK(belowResidual && belowResidual.value() == belowCut);
  VSS_CHECK(cutResidual && cutResidual.value() == -180.0);
  VSS_CHECK(aboveResidual && aboveResidual.value() == aboveCut - 360.0);
  const auto subnormalPeriod = wrappedAngularResidual(
      0.0, 0.0, std::numeric_limits<Scalar>::denorm_min());
  VSS_CHECK(!subnormalPeriod &&
            subnormalPeriod.status().code() == ErrorCode::InvalidDomain);
}

void rotationTransformCases() {
  Matrix quarterTurn(2U, 2U, 0.0);
  quarterTurn.at(0U, 1U) = -1.0;
  quarterTurn.at(1U, 0U) = 1.0;
  auto input = vector({2.0, 0.0});
  const auto rotated =
      rotationTransform(input.view(), quarterTurn.view(), tight, limits);
  VSS_CHECK(rotated && rotated.value().size() == 2U);
  VSS_CHECK(std::abs(rotated.value().at(0U)) < 1.0e-14);
  VSS_CHECK(std::abs(rotated.value().at(1U) - 2.0) < 1.0e-14);

  Matrix reflection(2U, 2U, 0.0);
  reflection.at(0U, 0U) = 1.0;
  reflection.at(1U, 1U) = -1.0;
  const auto rejected = rotationTransform(vector({1.0, 2.0}).view(),
                                          reflection.view(), tight, limits);
  VSS_CHECK(!rejected && rejected.status().code() == ErrorCode::InvalidDomain);
  Matrix shear(2U, 2U, 0.0);
  shear.at(0U, 0U) = 1.0;
  shear.at(1U, 1U) = 1.0;
  shear.at(0U, 1U) = 0.25;
  const auto nonOrthogonal =
      rotationTransform(vector({1.0, 2.0}).view(), shear.view(), tight, limits);
  VSS_CHECK(!nonOrthogonal &&
            nonOrthogonal.status().code() == ErrorCode::InvalidDomain);

  const auto nonfinite = rotationTransform(
      vector({std::numeric_limits<Scalar>::infinity(), 0.0}).view(),
      quarterTurn.view(), tight, limits);
  VSS_CHECK(!nonfinite &&
            nonfinite.status().code() == ErrorCode::NonFiniteInput);
  const auto capped = rotationTransform(
      vector({1.0, 0.0}).view(), quarterTurn.view(), tight, Limits{1U, 8U, 0U});
  VSS_CHECK(!capped && capped.status().code() == ErrorCode::ResourceLimit);
}

void normalizationDifferential() {
  const auto normalizedInput = vector({2.0, 0.0});
  const auto normalized =
      normalizeVectorDerivatives(normalizedInput.view(), limits);
  VSS_CHECK(normalized);
  VSS_CHECK(normalized.value().value.at(0U) == 1.0);
  VSS_CHECK(normalized.value().value.at(1U) == 0.0);
  VSS_CHECK(std::abs(normalized.value().jacobian.at(1U, 1U) - 0.5) < 1.0e-14);
  VSS_CHECK(normalized.value().hessian.has_value());
  const std::array<Index, 3U> hIndex{0U, 1U, 1U};
  const auto zeroHessian = zeroNormalizationHessian();
  VSS_CHECK(zeroHessian);
  const Tensor normalizedHessian =
      normalized.value().hessian.value_or(zeroHessian.value());
  VSS_CHECK(normalized.value().hessian.has_value() &&
            std::abs(normalizedHessian.at(hIndex) + 0.25) < 1.0e-14);

  const auto zeroInput = vector({0.0, 0.0});
  const auto zero = normalizeVectorDerivatives(zeroInput.view(), limits);
  VSS_CHECK(!zero && zero.status().code() == ErrorCode::InvalidDomain);
  const auto nonfinite = normalizeVectorDerivatives(
      vector({1.0, std::numeric_limits<Scalar>::infinity()}).view(), limits);
  VSS_CHECK(!nonfinite &&
            nonfinite.status().code() == ErrorCode::NonFiniteInput);
  const auto capped =
      normalizeVectorDerivatives(normalizedInput.view(), Limits{10U, 80U, 0U});
  VSS_CHECK(!capped && capped.status().code() == ErrorCode::ResourceLimit);

  const Scalar large = std::numeric_limits<Scalar>::max() * 0.9;
  const auto largeScale =
      normalizeVectorDerivatives(vector({large, large}).view(), limits);
  VSS_CHECK(largeScale);
  VSS_CHECK(std::abs(largeScale.value().value.at(0U) - std::sqrt(0.5)) <
            1.0e-14);
  const Scalar tiny = std::scalbn(1.0, -500);
  const auto smallScale =
      normalizeVectorDerivatives(vector({tiny, tiny}).view(), limits);
  VSS_CHECK(smallScale);
  VSS_CHECK(std::abs(smallScale.value().value.at(1U) - std::sqrt(0.5)) <
            1.0e-14);
  VSS_CHECK(std::isfinite(smallScale.value().jacobian.at(0U, 0U)));
  for (const int exponent : {-500, 500}) {
    const Scalar scale = std::scalbn(1.0, exponent);
    const auto scaled =
        normalizeVectorDerivatives(vector({2.0 * scale, 0.0}).view(), limits);
    VSS_CHECK(scaled && scaled.value().value.at(0U) == 1.0);
    const Scalar expected = 1.0 / (2.0 * scale);
    VSS_CHECK(std::abs(scaled.value().jacobian.at(1U, 1U) / expected - 1.0) <
              1.0e-14);
    VSS_CHECK(scaled.value().hessian.has_value());
    const Scalar scaleSquared = scale * scale;
    const Tensor scaledHessian =
        scaled.value().hessian.value_or(zeroHessian.value());
    VSS_CHECK(scaled.value().hessian.has_value() &&
              std::abs(scaledHessian.at(hIndex) * scaleSquared + 0.25) <
                  1.0e-14);
  }
}

void rigidGeometryAndLocalCovariance() {
  Matrix rotation(3U, 3U, 0.0);
  rotation.at(0U, 1U) = -1.0;
  rotation.at(1U, 0U) = 1.0;
  rotation.at(2U, 2U) = 1.0;
  Matrix points(1U, 3U);
  points.at(0U, 0U) = 1.0;
  Vector translation(3U);
  translation.at(0U) = 2.0;
  const auto moved = rigidTransform(points.view(), rotation.view(),
                                    translation.view(), tight, limits);
  VSS_CHECK(moved && moved.value().at(0U, 0U) == 2.0 &&
            moved.value().at(0U, 1U) == 1.0);
  const auto jac = rigidResidualJacobian(vector({1.0, 0.0, 0.0}).view(),
                                         rotation.view(), limits);
  VSS_CHECK(jac && jac.value().rows() == 3U && jac.value().columns() == 6U);
  VSS_CHECK(jac.value().at(0U, 2U) == -1.0 && jac.value().at(1U, 1U) == 0.0 &&
            jac.value().at(2U, 1U) == -1.0);
  VSS_CHECK(jac.value().at(0U, 3U) == 1.0 && jac.value().at(1U, 4U) == 1.0 &&
            jac.value().at(2U, 5U) == 1.0);
  // Right perturbation R exp([delta]x) and left perturbation
  // exp([delta]x) R differ for this noncommuting quarter-turn fixture.
  Matrix xQuarter(3U, 3U, 0.0);
  xQuarter.at(0U, 0U) = 1.0;
  xQuarter.at(1U, 2U) = -1.0;
  xQuarter.at(2U, 1U) = 1.0;
  const auto right = rigidResidualJacobian(vector({1.0, 0.0, 0.0}).view(),
                                           xQuarter.view(), limits);
  VSS_CHECK(right && right.value().at(0U, 2U) == 0.0 &&
            right.value().at(1U, 2U) == 0.0 && right.value().at(2U, 2U) == 1.0);
  Matrix halfTurnMatrix(3U, 3U, 0.0);
  halfTurnMatrix.at(0U, 0U) = -1.0;
  halfTurnMatrix.at(1U, 1U) = -1.0;
  halfTurnMatrix.at(2U, 2U) = 1.0;
  const auto halfTurn = rigidResidualJacobian(vector({1.0, 0.0, 0.0}).view(),
                                              halfTurnMatrix.view(), limits);
  VSS_CHECK(halfTurn && halfTurn.value().at(1U, 2U) == -1.0 &&
            halfTurn.value().at(2U, 1U) == -1.0);
  const Scalar belowPi = std::nextafter(std::numbers::pi, 0.0);
  const Scalar abovePi =
      std::nextafter(std::numbers::pi, std::numeric_limits<Scalar>::infinity());
  for (const Scalar angle : {belowPi, abovePi}) {
    Matrix nearCut(3U, 3U, 0.0);
    nearCut.at(0U, 0U) = std::cos(angle);
    nearCut.at(0U, 1U) = -std::sin(angle);
    nearCut.at(1U, 0U) = std::sin(angle);
    nearCut.at(1U, 1U) = std::cos(angle);
    nearCut.at(2U, 2U) = 1.0;
    VSS_CHECK(rigidResidualJacobian(vector({1.0, 0.0, 0.0}).view(),
                                    nearCut.view(), limits));
  }
  const Scalar belowHalfTurn = std::nextafter(180.0, 0.0);
  const Scalar aboveHalfTurn = std::nextafter(180.0, 360.0);
  for (const Scalar angle : {belowHalfTurn, aboveHalfTurn}) {
    const Scalar delta = (angle - 180.0) * std::numbers::pi / 180.0;
    const Scalar cosine = -std::cos(delta);
    const Scalar sine = -std::sin(delta);
    Matrix nearCut(3U, 3U, 0.0);
    nearCut.at(0U, 0U) = cosine;
    nearCut.at(0U, 1U) = -sine;
    nearCut.at(1U, 0U) = sine;
    nearCut.at(1U, 1U) = cosine;
    nearCut.at(2U, 2U) = 1.0;
    const auto nearCutJacobian = rigidResidualJacobian(
        vector({1.0, 0.0, 0.0}).view(), nearCut.view(), limits);
    VSS_CHECK(nearCutJacobian);
    VSS_CHECK(std::abs(nearCutJacobian.value().at(0U, 2U) + sine) < 1.0e-14);
    VSS_CHECK(std::abs(nearCutJacobian.value().at(1U, 2U) - cosine) < 1.0e-14);
    VSS_CHECK(nearCutJacobian.value().at(2U, 1U) == -1.0);
  }
  // Compare the exact action R exp(h[e_z]x)x with its analytic right
  // derivative e_y at the identity, at two smooth-chart step sizes.
  for (const int exponent : {-10, -15}) {
    const Scalar h = std::scalbn(1.0, exponent);
    Matrix plus(3U, 3U, 0.0);
    Matrix minus(3U, 3U, 0.0);
    plus.at(0U, 0U) = std::cos(h);
    plus.at(0U, 1U) = -std::sin(h);
    plus.at(1U, 0U) = std::sin(h);
    plus.at(1U, 1U) = std::cos(h);
    plus.at(2U, 2U) = 1.0;
    minus.at(0U, 0U) = plus.at(0U, 0U);
    minus.at(0U, 1U) = plus.at(1U, 0U);
    minus.at(1U, 0U) = plus.at(0U, 1U);
    minus.at(1U, 1U) = plus.at(1U, 1U);
    minus.at(2U, 2U) = 1.0;
    Matrix point(1U, 3U, 0.0);
    point.at(0U, 0U) = 1.0;
    Vector noTranslation(3U, 0.0);
    const auto p = rigidTransform(point.view(), plus.view(),
                                  noTranslation.view(), tight, limits);
    const auto m = rigidTransform(point.view(), minus.view(),
                                  noTranslation.view(), tight, limits);
    VSS_CHECK(p && m);
    const Scalar derivative =
        (p.value().at(0U, 1U) - m.value().at(0U, 1U)) / (2.0 * h);
    VSS_CHECK(std::abs(derivative - 1.0) <=
              h * h / 6.0 + 16.0 * std::numeric_limits<Scalar>::epsilon() / h);
    auto multiply3 = [](ConstMatrixView lhs, ConstMatrixView rhs) {
      Matrix product(3U, 3U, 0.0);
      for (Extent i = 0; i < 3U; ++i) {
        for (Extent j = 0; j < 3U; ++j) {
          for (Extent k = 0; k < 3U; ++k) {
            product.at(i, j) =
                std::fma(lhs.at(i, k), rhs.at(k, j), product.at(i, j));
          }
        }
      }
      return product;
    };
    const auto rightPlus = multiply3(xQuarter.view(), plus.view());
    const auto rightMinus = multiply3(xQuarter.view(), minus.view());
    const auto leftPlus = multiply3(plus.view(), xQuarter.view());
    const auto leftMinus = multiply3(minus.view(), xQuarter.view());
    const auto rp = rigidTransform(point.view(), rightPlus.view(),
                                   noTranslation.view(), tight, limits);
    const auto rm = rigidTransform(point.view(), rightMinus.view(),
                                   noTranslation.view(), tight, limits);
    const auto lp = rigidTransform(point.view(), leftPlus.view(),
                                   noTranslation.view(), tight, limits);
    const auto lm = rigidTransform(point.view(), leftMinus.view(),
                                   noTranslation.view(), tight, limits);
    VSS_CHECK(rp && rm && lp && lm);
    const Scalar rightY =
        (rp.value().at(0U, 1U) - rm.value().at(0U, 1U)) / (2.0 * h);
    const Scalar rightZ =
        (rp.value().at(0U, 2U) - rm.value().at(0U, 2U)) / (2.0 * h);
    const Scalar leftY =
        (lp.value().at(0U, 1U) - lm.value().at(0U, 1U)) / (2.0 * h);
    const Scalar leftZ =
        (lp.value().at(0U, 2U) - lm.value().at(0U, 2U)) / (2.0 * h);
    const Scalar tolerance =
        h * h / 6.0 + 32.0 * std::numeric_limits<Scalar>::epsilon() / h;
    VSS_CHECK(std::abs(rightY) <= tolerance &&
              std::abs(rightZ - 1.0) <= tolerance);
    VSS_CHECK(std::abs(leftY - 1.0) <= tolerance &&
              std::abs(leftZ) <= tolerance);
  }
  const auto wrongShape = rigidTransform(Matrix(1U, 2U).view(), rotation.view(),
                                         translation.view(), tight, limits);
  VSS_CHECK(!wrongShape &&
            wrongShape.status().code() == ErrorCode::InvalidExtent);
  Matrix j(2U, 2U);
  j.at(0U, 0U) = 1.0;
  j.at(0U, 1U) = 1.0;
  j.at(1U, 0U) = 1.0;
  j.at(1U, 1U) = -1.0;
  Matrix cov(2U, 2U);
  cov.at(0U, 0U) = 4.0;
  cov.at(1U, 1U) = 9.0;
  const auto transformed =
      localCoordinateCovariance(j.view(), cov.view(), limits);
  VSS_CHECK(transformed && transformed.value().at(0U, 0U) == 13.0 &&
            transformed.value().at(0U, 1U) == -5.0);
  Matrix invalid(2U, 2U);
  invalid.at(0U, 0U) = 1.0;
  invalid.at(1U, 1U) = -1.0;
  const auto rejected =
      localCoordinateCovariance(j.view(), invalid.view(), limits);
  VSS_CHECK(!rejected &&
            rejected.status().code() == ErrorCode::NotPositiveSemidefinite);
  const auto limited =
      localCoordinateCovariance(j.view(), cov.view(), Limits{1U, 8U, 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  Matrix rectangularJ(1U, 2U);
  rectangularJ.at(0U, 0U) = 1.0;
  rectangularJ.at(0U, 1U) = 1.0;
  const auto rectangularCov =
      localCoordinateCovariance(rectangularJ.view(), cov.view(), limits);
  VSS_CHECK(rectangularCov && rectangularCov.value().rows() == 1U &&
            rectangularCov.value().at(0U, 0U) == 13.0);

  Matrix zQuarterTurn(3U, 3U, 0.0);
  zQuarterTurn.at(0U, 1U) = -1.0;
  zQuarterTurn.at(1U, 0U) = 1.0;
  zQuarterTurn.at(2U, 2U) = 1.0;
  const auto rightChart = rigidResidualJacobian(vector({1.0, 0.0, 0.0}).view(),
                                                zQuarterTurn.view(), limits);
  VSS_CHECK(rightChart);
  Matrix rightJacobian(3U, 3U, 0.0);
  for (Extent row = 0U; row < 3U; ++row) {
    for (Extent column = 0U; column < 3U; ++column) {
      rightJacobian.at(row, column) = rightChart.value().at(row, column);
    }
  }
  Matrix leftJacobian(3U, 3U, 0.0);
  leftJacobian.at(0U, 2U) = -1.0;
  leftJacobian.at(2U, 0U) = 1.0;
  Matrix rightCovariance(3U, 3U, 0.0);
  rightCovariance.at(0U, 0U) = 4.0;
  rightCovariance.at(0U, 1U) = 1.0;
  rightCovariance.at(0U, 2U) = 2.0;
  rightCovariance.at(1U, 0U) = 1.0;
  rightCovariance.at(1U, 1U) = 9.0;
  rightCovariance.at(1U, 2U) = 3.0;
  rightCovariance.at(2U, 0U) = 2.0;
  rightCovariance.at(2U, 1U) = 3.0;
  rightCovariance.at(2U, 2U) = 16.0;
  Matrix leftCovariance(3U, 3U, 0.0);
  leftCovariance.at(0U, 0U) = 9.0;
  leftCovariance.at(0U, 1U) = -1.0;
  leftCovariance.at(0U, 2U) = -3.0;
  leftCovariance.at(1U, 0U) = -1.0;
  leftCovariance.at(1U, 1U) = 4.0;
  leftCovariance.at(1U, 2U) = 2.0;
  leftCovariance.at(2U, 0U) = -3.0;
  leftCovariance.at(2U, 1U) = 2.0;
  leftCovariance.at(2U, 2U) = 16.0;
  const auto rightPropagated = localCoordinateCovariance(
      rightJacobian.view(), rightCovariance.view(), limits);
  const auto leftPropagated = localCoordinateCovariance(
      leftJacobian.view(), leftCovariance.view(), limits);
  VSS_CHECK(rightPropagated && leftPropagated);
  Matrix expectedCovariance(3U, 3U, 0.0);
  expectedCovariance.at(0U, 0U) = 16.0;
  expectedCovariance.at(0U, 2U) = 3.0;
  expectedCovariance.at(2U, 0U) = 3.0;
  expectedCovariance.at(2U, 2U) = 9.0;
  for (Extent row = 0U; row < 3U; ++row) {
    for (Extent column = 0U; column < 3U; ++column) {
      VSS_CHECK(std::abs(rightPropagated.value().at(row, column) -
                         expectedCovariance.at(row, column)) < 1.0e-12);
      VSS_CHECK(std::abs(leftPropagated.value().at(row, column) -
                         expectedCovariance.at(row, column)) < 1.0e-12);
    }
  }
  Matrix overflowJ(2U, 2U);
  overflowJ.at(0U, 0U) = std::numeric_limits<Scalar>::max() / 2.0;
  overflowJ.at(1U, 1U) = 1.0;
  Matrix unitCov(2U, 2U);
  unitCov.at(0U, 0U) = 1.0;
  unitCov.at(1U, 1U) = 1.0;
  const auto overflow =
      localCoordinateCovariance(overflowJ.view(), unitCov.view(), limits);
  VSS_CHECK(!overflow &&
            overflow.status().code() == ErrorCode::NonFiniteResult);
}

void circularAndAffineValues() {
  const std::array<Scalar, 2U> angles{60.0, -60.0};
  const WeightSpec unit{.rule = WeightRule::unweighted,
                        .rowWeights = std::nullopt,
                        .componentWeights = std::nullopt,
                        .frequencyCounts = {}};
  const auto circ = circularStatistics(
      ConstVectorView::create(angles, 2U).value(), unit, 360.0, limits);
  const Scalar undefinedDirection = std::numeric_limits<Scalar>::quiet_NaN();
  const Scalar direction =
      circ ? circ.value().direction.value_or(undefinedDirection)
           : undefinedDirection;
  VSS_CHECK(circ && circ.value().direction.has_value() &&
            std::abs(direction) < 1.0e-12);
  VSS_CHECK(std::abs(circ.value().circular_variance - 0.5) < 1.0e-12);
  const std::array<Scalar, 2U> opposite{0.0, 180.0};
  const auto antipodal = circularStatistics(
      ConstVectorView::create(opposite, 2U).value(), unit, 360.0, limits);
  VSS_CHECK(antipodal && !antipodal.value().direction);

  std::array<Extent, 1U> shapeDims{2U};
  const auto shape = TensorShape::create(shapeDims);
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits(8U));
  const std::array<Index, 1U> tensorIndex0{0U};
  const std::array<Index, 1U> tensorIndex1{1U};
  tensor.value().at(tensorIndex0) = 1.0;
  tensor.value().at(tensorIndex1) = 2.0;
  Matrix affine(1U, 2U);
  affine.at(0U, 0U) = 1.0;
  affine.at(0U, 1U) = 2.0;
  Vector b(1U);
  b.at(0U) = 3.0;
  const std::array<Extent, 1U> outShape{1U};
  const auto mapped =
      applyAffine(tensor.value(), affine.view(), b.view(), outShape, limits);
  if (mapped) {
    VSS_CHECK(mapped.value().at(tensorIndex0) == 8.0);
  } else {
    VSS_CHECK(false);
  }
  const std::array<Extent, 1U> wrongDims{2U};
  const auto wrongShape =
      applyAffine(tensor.value(), affine.view(), b.view(), wrongDims, limits);
  VSS_CHECK(!wrongShape &&
            wrongShape.status().code() == ErrorCode::DimensionMismatch);
  const auto limited = applyAffine(tensor.value(), affine.view(), b.view(),
                                   outShape, Limits{0U, 0U, 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const std::array<Extent, 1U> hugeShape{2U};
  const auto overflowShape =
      applyAffine(tensor.value(), affine.view(), b.view(), hugeShape, limits);
  VSS_CHECK(!overflowShape &&
            overflowShape.status().code() == ErrorCode::DimensionMismatch);

  std::array<Scalar, 2U> rowWeights{1.0, 2.0};
  const WeightSpec weighted{.rule = WeightRule::empirical,
                            .rowWeights =
                                ConstVectorView::create(rowWeights, 2U).value(),
                            .componentWeights = std::nullopt,
                            .frequencyCounts = {}};
  const auto weightedCircular = circularStatistics(
      ConstVectorView::create(angles, 2U).value(), weighted, 360.0, limits);
  VSS_CHECK(weightedCircular && weightedCircular.value().direction);
  std::array<Scalar, 2U> scaledWeights{1.0e300, 2.0e300};
  const WeightSpec scaled{
      .rule = WeightRule::empirical,
      .rowWeights = ConstVectorView::create(scaledWeights, 2U).value(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto scaledCircular = circularStatistics(
      ConstVectorView::create(angles, 2U).value(), scaled, 360.0, limits);
  const Scalar scaledDirection =
      scaledCircular
          ? scaledCircular.value().direction.value_or(undefinedDirection)
          : undefinedDirection;
  const Scalar weightedDirection =
      weightedCircular
          ? weightedCircular.value().direction.value_or(undefinedDirection)
          : undefinedDirection;
  VSS_CHECK(scaledCircular && scaledCircular.value().direction.has_value() &&
            weightedCircular &&
            weightedCircular.value().direction.has_value() &&
            std::abs(scaledDirection - weightedDirection) < 1.0e-12);
  std::array<Scalar, 2U> nearAntipodal{1.0, 1.0};
  const WeightSpec unresolvedWeights{
      .rule = WeightRule::empirical,
      .rowWeights = ConstVectorView::create(nearAntipodal, 2U).value(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto unresolved =
      circularStatistics(ConstVectorView::create(opposite, 2U).value(),
                         unresolvedWeights, 360.0, limits);
  VSS_CHECK(unresolved && !unresolved.value().direction);
  VSS_CHECK(unresolved.value().diagnostics.warnings.view().size() == 1U &&
            unresolved.value().diagnostics.warnings.view()[0U] ==
                ErrorCode::IllConditioned);
  nearAntipodal[0U] = 1.0 + 128.0 * std::numeric_limits<Scalar>::epsilon();
  const WeightSpec resolvedWeights{
      .rule = WeightRule::empirical,
      .rowWeights = ConstVectorView::create(nearAntipodal, 2U).value(),
      .componentWeights = std::nullopt,
      .frequencyCounts = {}};
  const auto resolved =
      circularStatistics(ConstVectorView::create(opposite, 2U).value(),
                         resolvedWeights, 360.0, limits);
  VSS_CHECK(resolved && resolved.value().direction);
}

void centralMomentAffineTransform() {
  const Support support{.input = 3U,
                        .valid = 3U,
                        .positive = 3U,
                        .frequency_mass = std::nullopt,
                        .weight_mass = 3.0,
                        .normalized_weight_powers = {},
                        .effective_count = 3.0,
                        .degrees_of_freedom = 2.0,
                        .squared_weight_mass = std::nullopt};
  EstimatorDiagnostics diagnostics{.estimand = Estimand::observed,
                                   .weight = WeightRule::unweighted,
                                   .center = CenterRule::estimated,
                                   .approximation = Approximation::exact,
                                   .global = support,
                                   .coordinate_support = {},
                                   .tuple_support = {},
                                   .reciprocal_condition = std::nullopt,
                                   .numerical_error = std::nullopt,
                                   .sampling_error = std::nullopt,
                                   .rank = std::nullopt,
                                   .algorithm = "central-empirical",
                                   .premises = "test fixture",
                                   .warnings = {}};
  std::array<statistics::MomentTerm, 3U> terms{
      statistics::MomentTerm{.order = 1U,
                             .layout = MomentRepresentation::component,
                             .flattened_tuples = {},
                             .values = Vector(1U, 0.0)},
      statistics::MomentTerm{.order = 2U,
                             .layout = MomentRepresentation::component,
                             .flattened_tuples = {},
                             .values = Vector(1U, 5.0)},
      statistics::MomentTerm{.order = 4U,
                             .layout = MomentRepresentation::component,
                             .flattened_tuples = {},
                             .values = Vector(1U, 2.0)}};
  const auto termArray = Array<MomentTerm>::copy(terms, limits);
  Vector mean(1U, 4.0);
  MomentEstimate const source{.observation_shape = {},
                              .terms = termArray.value(),
                              .diagnostics = std::move(diagnostics),
                              .population_mean = mean};
  Matrix a(1U, 1U, 2.0);
  Vector b(1U, 3.0);
  const MomentRequest request{.orders = 0x0bU,
                              .representation = MomentRepresentation::component,
                              .selected = {},
                              .contraction = std::nullopt};
  const auto result =
      transformMoments(source, a.view(), b.view(), request, limits);
  const Vector missingMean(2U, std::numeric_limits<Scalar>::quiet_NaN());
  const Vector resultMean =
      result ? result.value().population_mean.value_or(missingMean)
             : missingMean;
  VSS_CHECK(result && result.value().population_mean.has_value() &&
            resultMean.at(0U) == 11.0);
  VSS_CHECK(result.value().terms.view().size() == 3U);
  VSS_CHECK(result.value().terms.view()[0U].values.at(0U) == 0.0);
  VSS_CHECK(result.value().terms.view()[1U].values.at(0U) == 20.0);
  VSS_CHECK(result.value().terms.view()[2U].values.at(0U) == 32.0);
  Matrix projection(1U, 1U, 0.5);
  const MomentRequest contracted{.orders = 0x0bU,
                                 .representation =
                                     MomentRepresentation::contraction,
                                 .selected = {},
                                 .contraction = projection.view()};
  const auto projected =
      transformMoments(source, a.view(), b.view(), contracted, limits);
  const Vector projectedMean =
      projected ? projected.value().population_mean.value_or(missingMean)
                : missingMean;
  VSS_CHECK(projected && projected.value().population_mean.has_value() &&
            projectedMean.at(0U) == 5.5);
  VSS_CHECK(projected.value().terms.view()[1U].values.at(0U) == 5.0);

  // Packed symmetric covariance input exercises both tuple orderings while
  // applying a non-diagonal affine map. The source stores [C00, C01, C11]
  // = [2, 3, 5]; A=[[1,1],[1,-1]] gives [13, -3, 1].
  std::array<statistics::MomentTerm, 1U> packedTerms{
      statistics::MomentTerm{.order = 2U,
                             .layout = MomentRepresentation::packed,
                             .flattened_tuples = {},
                             .values = vector({2.0, 3.0, 5.0})}};
  const auto packedTermArray = Array<MomentTerm>::copy(packedTerms, limits);
  const std::array<Extent, 1U> packedDimensions{2U};
  const auto packedShape = Array<Index>::copy(packedDimensions, limits);
  MomentEstimate const packedSource{.observation_shape = packedShape.value(),
                                    .terms = packedTermArray.value(),
                                    .diagnostics = source.diagnostics,
                                    .population_mean = vector({0.0, 0.0})};
  Matrix mixing(2U, 2U);
  mixing.at(0U, 0U) = 1.0;
  mixing.at(0U, 1U) = 1.0;
  mixing.at(1U, 0U) = 1.0;
  mixing.at(1U, 1U) = -1.0;
  Vector zeroShift(2U, 0.0);
  const MomentRequest packedRequest{.orders = 0x02U,
                                    .representation =
                                        MomentRepresentation::packed,
                                    .selected = {},
                                    .contraction = std::nullopt};
  const auto transformedPacked = transformMoments(
      packedSource, mixing.view(), zeroShift.view(), packedRequest, limits);
  VSS_CHECK(transformedPacked);
  const auto packedValues = transformedPacked.value().terms.view()[0U].values;
  VSS_CHECK(packedValues.at(0U) == 13.0 && packedValues.at(1U) == -3.0 &&
            packedValues.at(2U) == 1.0);

  // Rectangular, non-diagonal map with a direct four-point population
  // reference: X=(0,0), (1,1), (2,0), (3,1), A=[[1,2],[0,-1]], b=[3,-2].
  // Outputs are (3,-2), (6,-3), (5,-2), (8,-3). Their means are [5.5,-2.5],
  // and central covariance Cov(0,1)=-0.75.
  std::array<statistics::MomentTerm, 2U> vectorTerms{
      statistics::MomentTerm{.order = 1U,
                             .layout = MomentRepresentation::component,
                             .flattened_tuples = {},
                             .values = Vector(2U, 0.0)},
      statistics::MomentTerm{.order = 2U,
                             .layout = MomentRepresentation::dense,
                             .flattened_tuples = {},
                             .values = Vector(4U, 0.0)}};
  vectorTerms[0U].values.at(0U) = 1.5;
  vectorTerms[0U].values.at(1U) = 0.5;
  vectorTerms[1U].values.at(0U) = 1.25;
  vectorTerms[1U].values.at(1U) = 0.25;
  vectorTerms[1U].values.at(2U) = 0.25;
  vectorTerms[1U].values.at(3U) = 0.25;
  const auto vectorTermArray = Array<MomentTerm>::copy(vectorTerms, limits);
  const std::array<Extent, 1U> sourceDims{2U};
  const auto sourceShape = Array<Index>::copy(sourceDims, limits);
  MomentEstimate const vectorSource{.observation_shape = sourceShape.value(),
                                    .terms = vectorTermArray.value(),
                                    .diagnostics = source.diagnostics,
                                    .population_mean = vector({1.5, 0.5})};
  Matrix rectangular(2U, 2U);
  rectangular.at(0U, 0U) = 1.0;
  rectangular.at(0U, 1U) = 2.0;
  rectangular.at(1U, 1U) = -1.0;
  Vector shift = vector({3.0, -2.0});
  const MomentRequest denseRequest{.orders = 0x03U,
                                   .representation =
                                       MomentRepresentation::dense,
                                   .selected = {},
                                   .contraction = std::nullopt};
  const auto transformedVector = transformMoments(
      vectorSource, rectangular.view(), shift.view(), denseRequest, limits);
  VSS_CHECK(transformedVector);
  const Vector transformedVectorMean =
      transformedVector
          ? transformedVector.value().population_mean.value_or(missingMean)
          : missingMean;
  VSS_CHECK(transformedVector &&
            transformedVector.value().population_mean.has_value() &&
            std::abs(transformedVectorMean.at(0U) - 5.5) < 1.0e-14 &&
            std::abs(transformedVectorMean.at(1U) + 2.5) < 1.0e-14);
  VSS_CHECK(std::abs(transformedVector.value().terms.view()[1U].values.at(1U) +
                     0.75) < 1.0e-14);

  // Contract fixture from the independent specification oracle: transform
  // central moments and cumulants of a full-rank four-point 3D population.
  const std::array<std::array<Scalar, 3U>, 4U> population{{{{0.0, 0.0, 0.0}},
                                                           {{1.0, 0.0, 1.0}},
                                                           {{0.0, 2.0, -1.0}},
                                                           {{1.0, 1.0, 2.0}}}};
  Vector populationMean(3U, 0.0);
  for (const auto& point : population) {
    for (Extent j = 0; j < 3U; ++j) {
      populationMean.at(j) += point.at(j) / 4.0;
    }
  }
  auto populationMoment = [&](std::span<const Index> indices) {
    Scalar value = 0.0;
    for (const auto& point : population) {
      Scalar product = 1.0;
      for (Index const j : indices) {
        product *= point.at(j) - populationMean.at(j);
      }
      value += product / 4.0;
    }
    return value;
  };
  auto populationCovariance = [&](Index i, Index j) {
    const std::array<Index, 2U> indices{i, j};
    return populationMoment(indices);
  };
  auto makeTerms = [&](bool asCumulants) {
    std::vector<MomentTerm> allTerms;
    for (Index order = 1U; order <= 4U; ++order) {
      Extent count = 1U;
      for (Index i = 0; i < order; ++i) {
        count *= 3U;
      }
      Vector values(count, 0.0);
      for (Extent flat = 0; flat < count; ++flat) {
        Extent remainder = flat;
        std::array<Index, 4U> indices{};
        for (Index i = order; i > 0U; --i) {
          indices.at(i - 1U) = remainder % 3U;
          remainder /= 3U;
        }
        Scalar value =
            populationMoment(std::span<const Index>(indices).first(order));
        if (asCumulants && order == 4U) {
          value -= populationCovariance(indices[0U], indices[1U]) *
                       populationCovariance(indices[2U], indices[3U]) +
                   populationCovariance(indices[0U], indices[2U]) *
                       populationCovariance(indices[1U], indices[3U]) +
                   populationCovariance(indices[0U], indices[3U]) *
                       populationCovariance(indices[1U], indices[2U]);
        }
        values.at(flat) = value;
      }
      allTerms.push_back(MomentTerm{.order = order,
                                    .layout = MomentRepresentation::dense,
                                    .flattened_tuples = {},
                                    .values = std::move(values)});
    }
    return Array<MomentTerm>::copy(allTerms, limits);
  };
  const std::array<Extent, 1U> dims3d{3U};
  auto shape3d = Array<Index>::copy(dims3d, limits);
  VSS_CHECK(shape3d);
  const auto centralTerms = makeTerms(false);
  const auto cumulantTerms = makeTerms(true);
  VSS_CHECK(centralTerms && cumulantTerms);
  Matrix map(2U, 3U);
  map.at(0U, 0U) = 1.0;
  map.at(0U, 1U) = 2.0;
  map.at(1U, 1U) = 1.0;
  map.at(1U, 2U) = -1.0;
  Vector offset = vector({3.0, -2.0});
  const MomentRequest allDense{.orders = 0x0fU,
                               .representation = MomentRepresentation::dense,
                               .selected = {},
                               .contraction = std::nullopt};
  MomentEstimate const centralInput{.observation_shape = shape3d.value(),
                                    .terms = centralTerms.value(),
                                    .diagnostics = source.diagnostics,
                                    .population_mean = populationMean};
  const auto centralOutput = transformMoments(centralInput, map.view(),
                                              offset.view(), allDense, limits);
  VSS_CHECK(centralOutput);
  const Vector centralOutputMean =
      centralOutput
          ? centralOutput.value().population_mean.value_or(missingMean)
          : missingMean;
  VSS_CHECK(centralOutput &&
            centralOutput.value().population_mean.has_value() &&
            std::abs(centralOutputMean.at(0U) - 5.0) < 1.0e-14 &&
            std::abs(centralOutputMean.at(1U) + 1.75) < 1.0e-14);
  VSS_CHECK(std::abs(centralOutput.value().terms.view()[2U].values.at(1U) -
                     15.0 / 8.0) < 1.0e-13);
  VSS_CHECK(std::abs(centralOutput.value().terms.view()[3U].values.at(3U) -
                     269.0 / 32.0) < 1.0e-12);
  MomentEstimate cumulantInput{.observation_shape = shape3d.value(),
                               .terms = cumulantTerms.value(),
                               .diagnostics = source.diagnostics,
                               .population_mean = populationMean};
  cumulantInput.diagnostics.algorithm = "empirical-cumulant";
  const auto cumulantOutput = transformMoments(cumulantInput, map.view(),
                                               offset.view(), allDense, limits);
  VSS_CHECK(cumulantOutput);
  VSS_CHECK(std::abs(cumulantOutput.value().terms.view()[3U].values.at(3U) +
                     45.0 / 16.0) < 1.0e-12);
  Vector shiftedOffset = vector({13.0, -9.0});
  const auto shiftedOutput = transformMoments(
      centralInput, map.view(), shiftedOffset.view(), allDense, limits);
  VSS_CHECK(shiftedOutput);
  VSS_CHECK(shiftedOutput.value().terms.view()[2U].values.at(1U) ==
            centralOutput.value().terms.view()[2U].values.at(1U));

  const std::array<Extent, 1U> vectorShape{2U};
  const auto shapeArray = Array<Extent>::copy(vectorShape, limits);
  const std::array<Index, 2U> storedTuple{0U, 0U};
  const auto tupleArray = Array<Index>::copy(storedTuple, limits);
  std::array<statistics::MomentTerm, 1U> sparseTerms{
      statistics::MomentTerm{.order = 2U,
                             .layout = MomentRepresentation::selected,
                             .flattened_tuples = tupleArray.value(),
                             .values = Vector(1U, 1.0)}};
  const auto sparseTermArray =
      Array<statistics::MomentTerm>::copy(sparseTerms, limits);
  MomentEstimate const sparse{.observation_shape = shapeArray.value(),
                              .terms = sparseTermArray.value(),
                              .diagnostics = source.diagnostics,
                              .population_mean = Vector(2U, 0.0)};
  Matrix mix(1U, 2U);
  mix.at(0U, 0U) = 1.0;
  mix.at(0U, 1U) = 1.0;
  const std::array<Index, 2U> outputTuple{0U, 0U};
  const std::array<std::span<const Index>, 1U> outputTuples{outputTuple};
  const MomentRequest sparseRequest{.orders = 0x02U,
                                    .representation =
                                        MomentRepresentation::selected,
                                    .selected = outputTuples,
                                    .contraction = std::nullopt};
  Vector zeroOffset(1U, 0.0);
  const auto missing = transformMoments(sparse, mix.view(), zeroOffset.view(),
                                        sparseRequest, limits);
  VSS_CHECK(!missing && missing.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() {
  wrappedAngles();
  rotationTransformCases();
  normalizationDifferential();
  rigidGeometryAndLocalCovariance();
  circularAndAffineValues();
  centralMomentAffineTransform();
  return 0;
}
