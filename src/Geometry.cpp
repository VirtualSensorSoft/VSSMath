#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/MatrixView.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/Tensor.hpp"
#include "vss/math/TensorLimits.hpp"
#include "vss/math/TensorShape.hpp"
#include "vss/math/Tolerance.hpp"
#include "vss/math/Vector.hpp"
#include "vss/math/VectorView.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vss/math/Geometry.hpp>
#include <vss/math/covariance/Covariance.hpp>

namespace vss::math {
namespace {
bool within(Extent elements, Limits limits) {
  return elements <= limits.elements &&
         elements <= limits.bytes / sizeof(Scalar);
}

bool withinCombined(Extent first, Extent second, Limits limits) {
  return second <= std::numeric_limits<Extent>::max() - first &&
         within(first + second, limits);
}

Result<bool> properRotation(ConstMatrixView rotation, Tolerance tolerance,
                            Limits limits) {
  const Extent n = rotation.rows();
  if (n == 0U || rotation.columns() != n) {
    return Result<bool>::failure(ErrorCode::DimensionMismatch,
                                 "rotation must be nonempty and square");
  }
  if (n > std::numeric_limits<Extent>::max() / n || !within(n * n, limits)) {
    return Result<bool>::failure(ErrorCode::ResourceLimit,
                                 "rotation validation exceeds limits");
  }
  const auto tolStatus = validate(tolerance);
  if (!tolStatus.ok()) {
    return Result<bool>::failure(tolStatus.code(),
                                 std::string(tolStatus.message()));
  }
  for (Extent i = 0; i < n; ++i) {
    for (Extent j = 0; j < n; ++j) {
      Scalar dot = 0.0;
      for (Extent k = 0; k < n; ++k) {
        const Scalar a = rotation.at(k, i);
        const Scalar b = rotation.at(k, j);
        if (!isFinite(a) || !isFinite(b)) {
          return Result<bool>::failure(ErrorCode::NonFiniteInput,
                                       "rotation contains non-finite entries");
        }
        dot = std::fma(a, b, dot);
      }
      if (!nearlyEqual(dot, i == j ? 1.0 : 0.0, tolerance)) {
        return Result<bool>::failure(ErrorCode::InvalidDomain,
                                     "rotation is not orthogonal");
      }
    }
  }
  // Determinant by pivoted elimination; only its sign and nonsingularity
  // matter.
  Matrix work(n, n);
  for (Extent i = 0; i < n; ++i) {
    for (Extent j = 0; j < n; ++j) {
      work.at(i, j) = rotation.at(i, j);
    }
  }
  int determinantSign = 1;
  for (Extent k = 0; k < n; ++k) {
    Extent pivot = k;
    for (Extent i = k + 1U; i < n; ++i) {
      if (std::abs(work.at(i, k)) > std::abs(work.at(pivot, k))) {
        pivot = i;
      }
    }
    if (work.at(pivot, k) == 0.0) {
      return Result<bool>::failure(ErrorCode::InvalidDomain,
                                   "rotation is singular");
    }
    if (pivot != k) {
      for (Extent j = k; j < n; ++j) {
        std::swap(work.at(k, j), work.at(pivot, j));
      }
      determinantSign = -determinantSign;
    }
    const Scalar d = work.at(k, k);
    if (d < 0.0) {
      determinantSign = -determinantSign;
    }
    for (Extent i = k + 1U; i < n; ++i) {
      const Scalar f = work.at(i, k) / d;
      for (Extent j = k + 1U; j < n; ++j) {
        work.at(i, j) = std::fma(-f, work.at(k, j), work.at(i, j));
      }
    }
  }
  if (determinantSign < 0) {
    return Result<bool>::failure(ErrorCode::InvalidDomain,
                                 "rotation must have positive determinant");
  }
  return Result<bool>::success(true);
}
} // namespace

Result<Vector> rotationTransform(ConstVectorView value,
                                 ConstMatrixView rotation, Tolerance tolerance,
                                 Limits limits) {
  if (value.size() != rotation.columns() ||
      rotation.rows() != rotation.columns()) {
    return Result<Vector>::failure(ErrorCode::DimensionMismatch,
                                   "rotation and vector dimensions differ");
  }
  if ((rotation.rows() != 0U &&
       rotation.rows() >
           std::numeric_limits<Extent>::max() / rotation.rows()) ||
      !withinCombined(value.size(), rotation.rows() * rotation.rows(),
                      limits)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "rotation output exceeds limits");
  }
  auto proper = properRotation(rotation, tolerance, limits);
  if (!proper) {
    return Result<Vector>::failure(proper.status().code(),
                                   std::string(proper.status().message()));
  }
  Vector result(value.size());
  for (Extent i = 0; i < value.size(); ++i) {
    if (!isFinite(value.at(i))) {
      return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                     "vector contains non-finite values");
    }
    Scalar sum = 0.0;
    for (Extent j = 0; j < value.size(); ++j) {
      sum = std::fma(rotation.at(i, j), value.at(j), sum);
    }
    if (!isFinite(sum)) {
      return Result<Vector>::failure(ErrorCode::NonFiniteResult,
                                     "rotation result is non-finite");
    }
    result.at(i) = sum;
  }
  return Result<Vector>::success(std::move(result));
}

Result<VectorDifferential> normalizeVectorDerivatives(ConstVectorView value,
                                                      Limits limits) {
  const Extent n = value.size();
  if (n == 0U) {
    return Result<VectorDifferential>::failure(
        ErrorCode::InvalidDomain, "cannot normalize an empty vector");
  }
  if (n > std::numeric_limits<Extent>::max() / n ||
      n * n > std::numeric_limits<Extent>::max() / n) {
    return Result<VectorDifferential>::failure(
        ErrorCode::SizeOverflow, "normalization Hessian size overflows");
  }
  const Extent hsize = n * n * n;
  const Extent matrixSize = n * n;
  if (matrixSize > std::numeric_limits<Extent>::max() - n ||
      hsize > std::numeric_limits<Extent>::max() - n - matrixSize) {
    return Result<VectorDifferential>::failure(
        ErrorCode::SizeOverflow, "normalization result size overflows");
  }
  const Extent resultSize = n + matrixSize + hsize;
  if (resultSize > limits.elements ||
      resultSize > limits.bytes / sizeof(Scalar)) {
    return Result<VectorDifferential>::failure(
        ErrorCode::ResourceLimit, "normalization derivatives exceed limits");
  }
  Scalar scale = 0.0;
  for (Extent i = 0; i < n; ++i) {
    if (!isFinite(value.at(i))) {
      return Result<VectorDifferential>::failure(
          ErrorCode::NonFiniteInput, "vector contains non-finite values");
    }
    scale = std::max(scale, std::abs(value.at(i)));
  }
  if (scale == 0.0) {
    return Result<VectorDifferential>::failure(
        ErrorCode::InvalidDomain, "zero vector has no normalized direction");
  }
  Scalar scaledNorm2 = 0.0;
  for (Extent i = 0; i < n; ++i) {
    scaledNorm2 =
        std::fma(value.at(i) / scale, value.at(i) / scale, scaledNorm2);
  }
  const Scalar scaledNorm = std::sqrt(scaledNorm2);
  if (!isFinite(scaledNorm) || scaledNorm == 0.0) {
    return Result<VectorDifferential>::failure(
        ErrorCode::NonFiniteResult, "scaled vector norm is not representable");
  }
  VectorDifferential out{Vector(n), Matrix(n, n), std::nullopt};
  std::array<Extent, 3> dims{n, n, n};
  auto shape = TensorShape::create(dims);
  if (!shape) {
    return Result<VectorDifferential>::failure(
        shape.status().code(), std::string(shape.status().message()));
  }
  auto hessian =
      Tensor::create(shape.value(), 0.0, TensorLimits(limits.elements));
  if (!hessian) {
    return Result<VectorDifferential>::failure(
        hessian.status().code(), std::string(hessian.status().message()));
  }
  for (Extent i = 0; i < n; ++i) {
    const Scalar ni = (value.at(i) / scale) / scaledNorm;
    out.value.at(i) = ni;
    for (Extent j = 0; j < n; ++j) {
      const Scalar nj = (value.at(j) / scale) / scaledNorm;
      out.jacobian.at(i, j) =
          (((i == j ? 1.0 : 0.0) - ni * nj) / scaledNorm) / scale;
      for (Extent k = 0; k < n; ++k) {
        const Scalar nk = (value.at(k) / scale) / scaledNorm;
        const Scalar numerator = 3.0 * ni * nj * nk - (i == j ? nk : 0.0) -
                                 (i == k ? nj : 0.0) - (j == k ? ni : 0.0);
        const std::array<Index, 3> index{i, j, k};
        hessian.value().at(index) =
            ((numerator / scaledNorm) / scaledNorm / scale) / scale;
        if (!isFinite(hessian.value().at(index))) {
          return Result<VectorDifferential>::failure(
              ErrorCode::NonFiniteResult,
              "normalization Hessian is non-finite");
        }
      }
    }
  }
  out.hessian = std::move(hessian).value();
  return Result<VectorDifferential>::success(std::move(out));
}

Result<Scalar> wrappedAngularResidual(Scalar value, Scalar reference,
                                      Scalar period) {
  if (!isFinite(value) || !isFinite(reference) || !isFinite(period)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "angle or period is non-finite");
  }
  if (!(period > 0.0)) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "period must be positive");
  }
  // Reduce operands separately: value-reference can overflow for finite inputs.
  const Scalar half = period / 2.0;
  if (half == 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "period is too small for a representable half-period interval");
  }
  const Scalar a = std::remainder(value, period);
  const Scalar b = std::remainder(reference, period);
  Scalar residual = std::remainder(a - b, period);
  if (residual >= half) {
    residual -= period;
  }
  if (residual < -half) {
    residual += period;
  }
  if (!isFinite(residual)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "wrapped residual is non-finite");
  }
  return Result<Scalar>::success(residual);
}

Result<Matrix> rigidTransform(ConstMatrixView points, ConstMatrixView rotation,
                              ConstVectorView translation, Tolerance tolerance,
                              Limits limits) {
  if (points.columns() != 3U || rotation.rows() != 3U ||
      rotation.columns() != 3U || translation.size() != 3U) {
    return Result<Matrix>::failure(
        ErrorCode::InvalidExtent,
        "rigid transform requires N-by-3 points and 3D transform");
  }
  if (points.rows() != 0U &&
      points.rows() > std::numeric_limits<Extent>::max() / 3U) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "point output size overflows");
  }
  const Extent count = points.rows() * 3U;
  if (rotation.rows() > std::numeric_limits<Extent>::max() / rotation.rows() ||
      !withinCombined(count, rotation.rows() * rotation.rows(), limits)) {
    return Result<Matrix>::failure(
        ErrorCode::ResourceLimit,
        "point output and validation workspace exceed limits");
  }
  auto proper = properRotation(rotation, tolerance, limits);
  if (!proper) {
    return Result<Matrix>::failure(proper.status().code(),
                                   std::string(proper.status().message()));
  }
  Matrix result(points.rows(), 3U);
  for (Extent r = 0; r < points.rows(); ++r) {
    for (Extent i = 0; i < 3U; ++i) {
      if (!isFinite(translation.at(i))) {
        return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                       "translation is non-finite");
      }
      Scalar sum = translation.at(i);
      for (Extent j = 0; j < 3U; ++j) {
        if (!isFinite(points.at(r, j))) {
          return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                         "point is non-finite");
        }
        sum = std::fma(rotation.at(i, j), points.at(r, j), sum);
      }
      if (!isFinite(sum)) {
        return Result<Matrix>::failure(ErrorCode::NonFiniteResult,
                                       "transformed point is non-finite");
      }
      result.at(r, i) = sum;
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Matrix> rigidResidualJacobian(ConstVectorView position,
                                     ConstMatrixView rotation, Limits limits) {
  if (position.size() != 3U || rotation.rows() != 3U ||
      rotation.columns() != 3U) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "rigid residual Jacobian requires 3D position and rotation");
  }
  if (!withinCombined(18U, 9U, limits)) {
    return Result<Matrix>::failure(
        ErrorCode::ResourceLimit,
        "Jacobian and validation workspace exceed limits");
  }
  auto proper = properRotation(rotation, Tolerance{}, limits);
  if (!proper) {
    return Result<Matrix>::failure(proper.status().code(),
                                   std::string(proper.status().message()));
  }
  for (Extent i = 0; i < 3U; ++i) {
    if (!isFinite(position.at(i))) {
      return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                     "position is non-finite");
    }
  }
  Matrix jacobian(3U, 6U, 0.0);
  const std::array<std::array<Scalar, 3U>, 3U> skew{{
      {{0.0, -position.at(2U), position.at(1U)}},
      {{position.at(2U), 0.0, -position.at(0U)}},
      {{-position.at(1U), position.at(0U), 0.0}},
  }};
  for (Extent i = 0; i < 3U; ++i) {
    for (Extent j = 0; j < 3U; ++j) {
      if (!isFinite(rotation.at(i, j))) {
        return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                       "rotation is non-finite");
      }
      for (Extent k = 0; k < 3U; ++k) {
        jacobian.at(i, j) =
            std::fma(-rotation.at(i, k), skew.at(k).at(j), jacobian.at(i, j));
      }
      jacobian.at(i, j + 3U) = i == j ? 1.0 : 0.0;
    }
  }
  return Result<Matrix>::success(std::move(jacobian));
}

Result<Matrix> localCoordinateCovariance(ConstMatrixView jacobian,
                                         ConstMatrixView covariance,
                                         Limits limits) {
  if (jacobian.columns() != covariance.rows() ||
      covariance.rows() != covariance.columns()) {
    return Result<Matrix>::failure(ErrorCode::DimensionMismatch,
                                   "Jacobian and covariance dimensions differ");
  }
  if (jacobian.rows() != 0U &&
      jacobian.rows() > std::numeric_limits<Extent>::max() / jacobian.rows()) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "coordinate covariance size overflows");
  }
  const Extent output = jacobian.rows() * jacobian.rows();
  if ((jacobian.columns() != 0U &&
       jacobian.rows() >
           std::numeric_limits<Extent>::max() / jacobian.columns()) ||
      (covariance.rows() != 0U &&
       covariance.rows() >
           std::numeric_limits<Extent>::max() / covariance.rows())) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "coordinate covariance exceeds limits");
  }
  const Extent jacobianElements = jacobian.rows() * jacobian.columns();
  const Extent covarianceElements = covariance.rows() * covariance.rows();
  if (covarianceElements > std::numeric_limits<Extent>::max() / 3U ||
      jacobianElements > std::numeric_limits<Extent>::max() - output ||
      3U * covarianceElements >
          std::numeric_limits<Extent>::max() - output - jacobianElements ||
      !within(output + jacobianElements + 3U * covarianceElements, limits)) {
    return Result<Matrix>::failure(
        ErrorCode::ResourceLimit,
        "coordinate covariance workspace exceeds limits");
  }
  auto propagated = covariance::propagateCovariance(jacobian, covariance);
  if (!propagated) {
    return Result<Matrix>::failure(propagated.status().code(),
                                   std::string(propagated.status().message()));
  }
  return Result<Matrix>::success(std::move(propagated.value()));
}
} // namespace vss::math
