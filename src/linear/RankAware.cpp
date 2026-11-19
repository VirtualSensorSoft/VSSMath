#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedMultiply(Index left, Index right, Index& product) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  product = left * right;
  return true;
}

bool checkedAdd(Index left, Index right, Index& sum) noexcept {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  sum = left + right;
  return true;
}

bool finite(ConstMatrixView matrix) {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool validRankPolicy(RankPolicy policy) {
  switch (policy) {
  case RankPolicy::reject:
  case RankPolicy::supported_subspace:
  case RankPolicy::minimum_norm:
    return true;
  }
  return false;
}

bool outputReservation(Index columns, Index thin, Index rightHandSides,
                       Index& elements, Index& bytes) {
  Index parameters = 0U;
  Index basis = 0U;
  if (!checkedMultiply(columns, rightHandSides, parameters) ||
      !checkedMultiply(columns, thin, basis)) {
    return false;
  }
  Index scalarElements = 0U;
  if (!checkedAdd(parameters, rightHandSides, scalarElements) ||
      !checkedAdd(scalarElements, basis, scalarElements)) {
    return false;
  }
  // The retained projection coefficients are bounded by the thin spectrum.
  if (!checkedAdd(scalarElements, thin, elements)) {
    return false;
  }
  Index scalarBytes = 0U;
  Index wideBytes = 0U;
  if (!checkedMultiply(scalarElements, sizeof(Scalar), scalarBytes) ||
      !checkedMultiply(thin, sizeof(Wide), wideBytes) ||
      !checkedAdd(scalarBytes, wideBytes, bytes)) {
    return false;
  }
  const auto ptrdiffLimit =
      static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max());
  return parameters <= ptrdiffLimit / sizeof(Scalar) &&
         basis <= ptrdiffLimit / sizeof(Scalar) &&
         rightHandSides <= ptrdiffLimit / sizeof(Scalar) &&
         thin <= ptrdiffLimit / sizeof(Wide);
}

bool representableScalar(Wide value) noexcept {
  return detail::representable(value) &&
         (value == 0.0L || static_cast<Scalar>(value) != 0.0);
}

} // namespace

Result<LinearEstimate> leastSquaresMinNorm(ConstMatrixView design,
                                           ConstMatrixView observations,
                                           SolvePolicy policy, Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return failure<LinearEstimate>(ErrorCode::InvalidTolerance,
                                   "least-squares policy is invalid");
  }
  if (!validRankPolicy(policy.rank)) {
    return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                   "least-squares rank policy is invalid");
  }
  if (observations.rows() != design.rows()) {
    return failure<LinearEstimate>(
        ErrorCode::DimensionMismatch,
        "design and observation row counts must match");
  }
  const Index rows = design.rows();
  const Index columns = design.columns();
  const Index rightHandSides = observations.columns();
  const Index thin = std::min(rows, columns);
  Index reservedElements = 0U;
  Index reservedBytes = 0U;
  if (!outputReservation(columns, thin, rightHandSides, reservedElements,
                         reservedBytes) ||
      reservedElements > limits.elements || reservedBytes > limits.bytes) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "least-squares outputs exceed limits");
  }
  if (!finite(observations)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                   "least-squares observations must be finite");
  }

  const Limits factorLimits{.elements = limits.elements - reservedElements,
                            .bytes = limits.bytes - reservedBytes,
                            .evaluations = limits.evaluations};
  const auto factor = factorSVD(design, policy, factorLimits);
  if (!factor.hasValue()) {
    return copyFailure<LinearEstimate>(factor.status());
  }
  const SVD& svd = factor.value();
  const Index rank = svd.rank;
  if (policy.rank == RankPolicy::reject && rank < thin) {
    return failure<LinearEstimate>(ErrorCode::RankDeficient,
                                   "least-squares design is rank deficient");
  }
  if (rank != 0U && policy.reciprocal_condition_floor != 0.0) {
    const Wide reciprocalCondition =
        static_cast<Wide>(svd.singular.at(rank - 1U)) /
        static_cast<Wide>(svd.singular.at(0U));
    if (reciprocalCondition <
        static_cast<Wide>(policy.reciprocal_condition_floor)) {
      return failure<LinearEstimate>(
          ErrorCode::IllConditioned,
          "retained least-squares spectrum is below the condition floor");
    }
  }

  LinearEstimate estimate{.parameters = Matrix(columns, rightHandSides, 0.0),
                          .covariance = std::nullopt,
                          .identified_basis = Matrix(columns, rank, 0.0),
                          .residual_norms = Vector(rightHandSides, 0.0),
                          .rank = rank,
                          .rank_cutoff = svd.cutoff};
  for (Index direction = 0U; direction < rank; ++direction) {
    for (Index coordinate = 0U; coordinate < columns; ++coordinate) {
      estimate.identified_basis->at(coordinate, direction) =
          svd.right_transpose.at(direction, coordinate);
    }
  }

  std::vector<Wide> projected(rank, 0.0L);
  for (Index rhs = 0U; rhs < rightHandSides; ++rhs) {
    for (Index direction = 0U; direction < rank; ++direction) {
      Wide coefficient = 0.0L;
      for (Index row = 0U; row < rows; ++row) {
        coefficient += static_cast<Wide>(svd.left.at(row, direction)) *
                       static_cast<Wide>(observations.at(row, rhs));
      }
      projected[direction] =
          coefficient / static_cast<Wide>(svd.singular.at(direction));
      if (!std::isfinite(projected[direction])) {
        return failure<LinearEstimate>(
            ErrorCode::NonFiniteResult,
            "least-squares spectral coefficient is not finite");
      }
    }
    for (Index coordinate = 0U; coordinate < columns; ++coordinate) {
      Wide parameter = 0.0L;
      for (Index direction = 0U; direction < rank; ++direction) {
        parameter +=
            static_cast<Wide>(svd.right_transpose.at(direction, coordinate)) *
            projected[direction];
      }
      if (!representableScalar(parameter)) {
        return failure<LinearEstimate>(
            ErrorCode::NonFiniteResult,
            "least-squares parameter is not representable");
      }
      estimate.parameters.at(coordinate, rhs) = static_cast<Scalar>(parameter);
    }

    Wide residual = 0.0L;
    for (Index row = 0U; row < rows; ++row) {
      Wide fitted = 0.0L;
      for (Index coordinate = 0U; coordinate < columns; ++coordinate) {
        fitted += static_cast<Wide>(design.at(row, coordinate)) *
                  static_cast<Wide>(estimate.parameters.at(coordinate, rhs));
      }
      residual = std::hypot(
          residual, fitted - static_cast<Wide>(observations.at(row, rhs)));
    }
    if (!representableScalar(residual)) {
      return failure<LinearEstimate>(
          ErrorCode::NonFiniteResult,
          "least-squares residual norm is not representable");
    }
    estimate.residual_norms.at(rhs) = static_cast<Scalar>(residual);
  }
  return Result<LinearEstimate>::success(std::move(estimate));
}

} // namespace vss::math::linear
