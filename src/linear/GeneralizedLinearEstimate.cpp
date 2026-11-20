#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedAdd(Index lhs, Index rhs, Index& result) noexcept {
  if (rhs > std::numeric_limits<Index>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(Index lhs, Index rhs, Index& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Index>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool reserve(Index m, Index n, Index p, Limits limits) noexcept {
  // Conservative peak envelope for covariance factors, exact/stochastic
  // designs, SVD workspaces, and owned parameter diagnostics.
  Index mm = 0U;
  Index mn = 0U;
  Index nn = 0U;
  Index mp = 0U;
  Index np = 0U;
  if (!checkedMultiply(m, m, mm) || !checkedMultiply(m, n, mn) ||
      !checkedMultiply(n, n, nn) || !checkedMultiply(m, p, mp) ||
      !checkedMultiply(n, p, np)) {
    return false;
  }
  Index elements = 0U;
  const auto addScaled = [&elements](Index value, Index scale) {
    Index term = 0U;
    return checkedMultiply(value, scale, term) &&
           checkedAdd(elements, term, elements);
  };
  if (!addScaled(mm, 4U) || !addScaled(mn, 4U) || !addScaled(nn, 5U) ||
      !addScaled(mp, 3U) || !addScaled(np, 3U) ||
      !checkedAdd(elements, m, elements) ||
      !checkedAdd(elements, n, elements) ||
      !checkedAdd(elements, p, elements)) {
    return false;
  }
  Index bytes = 0U;
  if (!checkedMultiply(elements, sizeof(Scalar) * 2U, bytes)) {
    return false;
  }
  return elements <= limits.elements && bytes <= limits.bytes;
}

bool finite(ConstMatrixView value) noexcept {
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      if (!std::isfinite(value.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

long double norm(ConstMatrixView value) noexcept {
  long double scale = 0.0L;
  long double sum = 1.0L;
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      const long double magnitude = std::abs(value.at(row, column));
      if (magnitude == 0.0L) {
        continue;
      }
      if (scale < magnitude) {
        const long double ratio = scale / magnitude;
        sum = 1.0L + (sum * ratio * ratio);
        scale = magnitude;
      } else {
        const long double ratio = magnitude / scale;
        sum += ratio * ratio;
      }
    }
  }
  return scale == 0.0L ? 0.0L : scale * std::sqrt(sum);
}

Matrix transpose(ConstMatrixView value) {
  Matrix result(value.columns(), value.rows(), 0.0);
  for (Index first = 0U; first < value.rows(); ++first) {
    for (Index second = 0U; second < value.columns(); ++second) {
      result.at(second, first) = value.at(first, second);
    }
  }
  return result;
}

bool representable(long double value) noexcept;

bool multiply(ConstMatrixView lhs, ConstMatrixView rhs, Matrix& result) {
  result = Matrix(lhs.rows(), rhs.columns(), 0.0);
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < rhs.columns(); ++column) {
      long double value = 0.0L;
      for (Index inner = 0U; inner < lhs.columns(); ++inner) {
        value += static_cast<long double>(lhs.at(row, inner)) *
                 static_cast<long double>(rhs.at(inner, column));
      }
      if (!representable(value)) {
        return false;
      }
      result.at(row, column) = static_cast<Scalar>(value);
    }
  }
  return true;
}

bool subtractInPlace(Matrix& lhs, ConstMatrixView rhs) {
  for (Index row = 0U; row < lhs.rows(); ++row) {
    for (Index column = 0U; column < lhs.columns(); ++column) {
      const long double value = static_cast<long double>(lhs.at(row, column)) -
                                static_cast<long double>(rhs.at(row, column));
      if (!representable(value)) {
        return false;
      }
      lhs.at(row, column) = static_cast<Scalar>(value);
    }
  }
  return true;
}

bool representable(long double value) noexcept {
  return std::isfinite(value) &&
         std::abs(value) <= std::numeric_limits<Scalar>::max() &&
         (value == 0.0L || static_cast<Scalar>(value) != 0.0);
}

} // namespace

Result<LinearEstimate>
generalizedLinearEstimate(ConstMatrixView design, ConstMatrixView observations,
                          ConstMatrixView observationCovariance,
                          SolvePolicy policy, Limits limits) {
  const Index m = design.rows();
  const Index n = design.columns();
  const Index p = observations.columns();
  if (observations.rows() != m || observationCovariance.rows() != m ||
      observationCovariance.columns() != m) {
    return failure<LinearEstimate>(
        ErrorCode::DimensionMismatch,
        "design, observations, and covariance dimensions must conform");
  }
  if (m == 0U) {
    return failure<LinearEstimate>(ErrorCode::InsufficientSamples,
                                   "GLS requires at least one observation");
  }
  if (!reserve(m, n, p, limits)) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "GLS peak storage exceeds supplied limits");
  }
  if (!finite(design) || !finite(observations)) {
    return failure<LinearEstimate>(
        ErrorCode::NonFiniteInput,
        "GLS design and observations must be finite");
  }

  const Vector zero(m, 0.0);
  const auto whitening = covariance::whitenSupported(
      observationCovariance, zero.view(), policy, limits);
  if (!whitening.hasValue()) {
    return copyFailure<LinearEstimate>(whitening.status());
  }

  Matrix exactTranspose(0U, m, 0.0);
  if (whitening.value().rank < m) {
    exactTranspose = transpose(whitening.value().support_basis.view());
    SolvePolicy basisPolicy = policy;
    basisPolicy.rank_abs = 0.0;
    basisPolicy.rank_rel = 0.0;
    basisPolicy.reciprocal_condition_floor = 0.0;
    basisPolicy.rank = RankPolicy::minimum_norm;
    const auto exactResult =
        nullSpace(exactTranspose.view(), basisPolicy, limits);
    if (!exactResult.hasValue()) {
      return copyFailure<LinearEstimate>(exactResult.status());
    }
    exactTranspose = transpose(exactResult.value().view());
  }

  Matrix constraints;
  Matrix exactObservations;
  if (!multiply(exactTranspose.view(), design, constraints) ||
      !multiply(exactTranspose.view(), observations, exactObservations)) {
    return failure<LinearEstimate>(
        ErrorCode::NonFiniteResult,
        "GLS exact-constraint projection overflowed");
  }
  Matrix particular(n, p, 0.0);
  Matrix exactBasis(n, 0U, 0.0);
  Index exactRank = 0U;
  if (exactTranspose.rows() != 0U) {
    SolvePolicy constraintPolicy = policy;
    constraintPolicy.rank = RankPolicy::minimum_norm;
    const auto constrained = leastSquaresMinNorm(
        constraints.view(), exactObservations.view(), constraintPolicy, limits);
    if (!constrained.hasValue()) {
      return copyFailure<LinearEstimate>(constrained.status());
    }
    const auto& constraintEstimate = constrained.value();
    const auto& constraintBasis = constraintEstimate.identified_basis;
    if (!constraintBasis.has_value()) {
      return failure<LinearEstimate>(
          ErrorCode::InvalidDomain,
          "exact-constraint solve omitted its identified basis");
    }
    particular = constraintEstimate.parameters;
    exactBasis = *constraintBasis;
    exactRank = constraintEstimate.rank;

    Matrix reproduced;
    if (!multiply(constraints.view(), particular.view(), reproduced)) {
      return failure<LinearEstimate>(
          ErrorCode::NonFiniteResult,
          "GLS exact-constraint residual overflowed");
    }
    Matrix residual = exactObservations;
    if (!subtractInPlace(residual, reproduced.view())) {
      return failure<LinearEstimate>(
          ErrorCode::NonFiniteResult,
          "GLS exact-constraint residual overflowed");
    }
    const long double factor =
        128.0L * std::numeric_limits<Scalar>::epsilon() *
        static_cast<long double>(std::max<Index>({1U, m, n}));
    const long double designNorm = norm(constraints.view());
    for (Index rhs = 0U; rhs < p; ++rhs) {
      Matrix residualColumn(residual.rows(), 1U, 0.0);
      Matrix observationColumn(exactObservations.rows(), 1U, 0.0);
      Matrix parameterColumn(particular.rows(), 1U, 0.0);
      for (Index row = 0U; row < residual.rows(); ++row) {
        residualColumn.at(row, 0U) = residual.at(row, rhs);
        observationColumn.at(row, 0U) = exactObservations.at(row, rhs);
      }
      for (Index row = 0U; row < particular.rows(); ++row) {
        parameterColumn.at(row, 0U) = particular.at(row, rhs);
      }
      const long double bound =
          factor * (designNorm * norm(parameterColumn.view()) +
                    norm(observationColumn.view()));
      if (norm(residualColumn.view()) > bound) {
        return failure<LinearEstimate>(
            ErrorCode::InvalidDomain,
            "GLS exact observation constraints are inconsistent");
      }
    }
  }

  Matrix tangent(n, n - exactRank, 0.0);
  if (exactRank == 0U) {
    for (Index coordinate = 0U; coordinate < n; ++coordinate) {
      tangent.at(coordinate, coordinate) = 1.0;
    }
  } else {
    const auto tangentResult = nullSpace(constraints.view(), policy, limits);
    if (!tangentResult.hasValue()) {
      return copyFailure<LinearEstimate>(tangentResult.status());
    }
    tangent = tangentResult.value();
    if (tangent.columns() != n - exactRank) {
      return failure<LinearEstimate>(
          ErrorCode::NoConvergence,
          "GLS exact and tangent rank decisions are inconsistent");
    }
  }

  Matrix whiteDesign;
  Matrix whiteObservations;
  Matrix whiteParticular;
  Matrix tangentDesign;
  if (!multiply(whitening.value().map.view(), design, whiteDesign) ||
      !multiply(whitening.value().map.view(), observations,
                whiteObservations) ||
      !multiply(whiteDesign.view(), particular.view(), whiteParticular) ||
      !multiply(whiteDesign.view(), tangent.view(), tangentDesign)) {
    return failure<LinearEstimate>(
        ErrorCode::NonFiniteResult,
        "GLS whitening products are not representable");
  }
  Matrix tangentObservations = whiteObservations;
  if (!subtractInPlace(tangentObservations, whiteParticular.view())) {
    return failure<LinearEstimate>(
        ErrorCode::NonFiniteResult,
        "GLS whitened residual is not representable");
  }

  SolvePolicy tangentPolicy = policy;
  tangentPolicy.rank = RankPolicy::minimum_norm;
  const auto fitted = leastSquaresMinNorm(
      tangentDesign.view(), tangentObservations.view(), tangentPolicy, limits);
  if (!fitted.hasValue()) {
    return copyFailure<LinearEstimate>(fitted.status());
  }
  const Index tangentRank = fitted.value().rank;
  const Index totalRank = exactRank + tangentRank;
  if (policy.rank == RankPolicy::reject && totalRank < n) {
    return failure<LinearEstimate>(ErrorCode::RankDeficient,
                                   "GLS parameter design is rank deficient");
  }

  Matrix tangentParameters;
  if (!multiply(tangent.view(), fitted.value().parameters.view(),
                tangentParameters)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                   "GLS parameter transform overflowed");
  }
  Matrix parameters = particular;
  for (Index row = 0U; row < n; ++row) {
    for (Index rhs = 0U; rhs < p; ++rhs) {
      parameters.at(row, rhs) += tangentParameters.at(row, rhs);
      if (!std::isfinite(parameters.at(row, rhs))) {
        return failure<LinearEstimate>(
            ErrorCode::NonFiniteResult,
            "GLS parameter estimate is not representable");
      }
    }
  }

  Matrix identified(n, totalRank, 0.0);
  for (Index row = 0U; row < n; ++row) {
    for (Index column = 0U; column < exactRank; ++column) {
      identified.at(row, column) = exactBasis.at(row, column);
    }
  }
  if (tangentRank != 0U) {
    const auto& stochasticEstimate = fitted.value();
    const auto& stochasticBasis = stochasticEstimate.identified_basis;
    if (!stochasticBasis.has_value()) {
      return failure<LinearEstimate>(
          ErrorCode::InvalidDomain,
          "stochastic solve omitted its identified basis");
    }
    Matrix tangentBasis;
    if (!multiply(tangent.view(), stochasticBasis->view(), tangentBasis)) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                     "GLS identified basis overflowed");
    }
    for (Index row = 0U; row < n; ++row) {
      for (Index column = 0U; column < tangentRank; ++column) {
        identified.at(row, exactRank + column) = tangentBasis.at(row, column);
      }
    }
  }

  LinearEstimate estimate{.parameters = std::move(parameters),
                          .covariance = std::nullopt,
                          .identified_basis = std::move(identified),
                          .residual_norms = fitted.value().residual_norms,
                          .rank = totalRank,
                          .rank_cutoff = fitted.value().rank_cutoff};
  if (totalRank == n) {
    Matrix parameterCovariance(n, n, 0.0);
    if (tangentRank != 0U) {
      const auto spectrum =
          factorSVD(tangentDesign.view(), tangentPolicy, limits);
      if (!spectrum.hasValue()) {
        return copyFailure<LinearEstimate>(spectrum.status());
      }
      Matrix tangentCovariance(tangent.columns(), tangent.columns(), 0.0);
      for (Index mode = 0U; mode < tangentRank; ++mode) {
        const long double singular = spectrum.value().singular.at(mode);
        const long double inverseVariance = 1.0L / (singular * singular);
        if (!representable(inverseVariance)) {
          return failure<LinearEstimate>(
              ErrorCode::NonFiniteResult,
              "GLS parameter covariance is not representable");
        }
        for (Index coordinate = 0U; coordinate < tangent.columns();
             ++coordinate) {
          for (Index otherCoordinate = 0U; otherCoordinate < tangent.columns();
               ++otherCoordinate) {
            const long double firstProjection =
                spectrum.value().right_transpose.at(mode, coordinate);
            const long double secondProjection =
                spectrum.value().right_transpose.at(mode, otherCoordinate);
            const long double contribution =
                (firstProjection * secondProjection * inverseVariance);
            const long double updated =
                static_cast<long double>(
                    tangentCovariance.at(coordinate, otherCoordinate)) +
                contribution;
            if (!representable(updated)) {
              return failure<LinearEstimate>(
                  ErrorCode::NonFiniteResult,
                  "GLS tangent covariance is not representable");
            }
            tangentCovariance.at(coordinate, otherCoordinate) =
                static_cast<Scalar>(updated);
          }
        }
      }
      Matrix mappedCovariance;
      const Matrix tangentTranspose = transpose(tangent.view());
      if (!multiply(tangent.view(), tangentCovariance.view(),
                    mappedCovariance) ||
          !multiply(mappedCovariance.view(), tangentTranspose.view(),
                    parameterCovariance)) {
        return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                       "GLS covariance transform overflowed");
      }
    }
    if (!finite(parameterCovariance.view())) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                     "GLS covariance is not finite");
    }
    estimate.covariance = std::move(parameterCovariance);
  }
  return Result<LinearEstimate>::success(std::move(estimate));
}

} // namespace vss::math::linear
