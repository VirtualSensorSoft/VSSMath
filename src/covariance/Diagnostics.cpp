#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
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
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool representable(long double value) noexcept {
  return std::isfinite(value) &&
         std::abs(value) <= std::numeric_limits<Scalar>::max() &&
         (value == 0.0L || static_cast<Scalar>(value) != 0.0);
}

} // namespace

Result<Matrix> marginalCovariance(ConstMatrixView covariance,
                                  std::span<const Index> keep,
                                  SolvePolicy policy, Limits limits) {
  SolvePolicy validationPolicy = policy;
  validationPolicy.rank_abs = 0.0;
  validationPolicy.rank_rel = 0.0;
  validationPolicy.reciprocal_condition_floor = 0.0;
  validationPolicy.rank = RankPolicy::supported_subspace;
  const auto result = conditionalCovariance(
      covariance, keep, std::span<const Index>{}, validationPolicy, limits);
  if (!result.hasValue()) {
    return copyFailure<Matrix>(result.status());
  }
  return Result<Matrix>::success(result.value().covariance);
}

Result<CovarianceResult> inverseSquareRoot(ConstMatrixView covariance,
                                           SolvePolicy policy, Limits limits) {
  if (covariance.rows() != covariance.columns()) {
    return failure<CovarianceResult>(
        ErrorCode::DimensionMismatch,
        "covariance inverse square root requires a square matrix");
  }
  if (covariance.rows() == 0U) {
    return failure<CovarianceResult>(ErrorCode::InsufficientSamples,
                                     "covariance cannot be empty");
  }
  const Vector zero(covariance.rows(), 0.0);
  const auto transform =
      whitenSupported(covariance, zero.view(), policy, limits);
  if (!transform.hasValue()) {
    return copyFailure<CovarianceResult>(transform.status());
  }

  CovarianceResult result{.covariance = transform.value().map,
                          .classification =
                              transform.value().rank == covariance.rows()
                                  ? CovarianceKind::positive_definite
                                  : CovarianceKind::semidefinite_singular,
                          .rank = transform.value().rank,
                          .cutoff = transform.value().cutoff,
                          .reciprocal_condition = std::nullopt};
  if (result.rank == covariance.rows()) {
    long double minimumInverseRoot =
        std::numeric_limits<long double>::infinity();
    long double maximumInverseRoot = 0.0L;
    for (Index mode = 0U; mode < result.rank; ++mode) {
      long double eigenvalue = 0.0L;
      for (Index coordinate = 0U; coordinate < covariance.rows();
           ++coordinate) {
        for (Index other = 0U; other < covariance.rows(); ++other) {
          eigenvalue +=
              static_cast<long double>(
                  transform.value().support_basis.at(coordinate, mode)) *
              static_cast<long double>(
                  transform.value().map.at(coordinate, other)) *
              static_cast<long double>(
                  transform.value().support_basis.at(other, mode));
        }
      }
      if (!(eigenvalue > 0.0L) || !std::isfinite(eigenvalue)) {
        return failure<CovarianceResult>(
            ErrorCode::NonFiniteResult,
            "inverse square root spectrum is not representable");
      }
      minimumInverseRoot = std::min(minimumInverseRoot, eigenvalue);
      maximumInverseRoot = std::max(maximumInverseRoot, eigenvalue);
    }
    const long double ratio = minimumInverseRoot / maximumInverseRoot;
    const long double reciprocal = ratio * ratio;
    if (!representable(reciprocal)) {
      return failure<CovarianceResult>(
          ErrorCode::NonFiniteResult,
          "covariance reciprocal condition is not representable");
    }
    result.reciprocal_condition = static_cast<Scalar>(reciprocal);
  }
  return Result<CovarianceResult>::success(std::move(result));
}

} // namespace vss::math::covariance
