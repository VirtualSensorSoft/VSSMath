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

bool nullSpaceStorage(Index rows, Index columns, Index thin, Index rank,
                      Limits limits) {
  const Index nullity = columns - rank;
  const bool needsCompletion = rank != 0U && rank < columns;
  const Index completionRank = needsCompletion ? rank : 0U;
  const Index completionSize = needsCompletion ? columns : 0U;
  Index inputFactors = 0U;
  Index outputElements = 0U;
  Index reflectorMatrix = 0U;
  Index reflectorElements = 0U;
  Index factorBytes = 0U;
  Index outputBytes = 0U;
  Index reflectorBytes = 0U;
  Index elements = 0U;
  Index bytes = 0U;
  if (!checkedMultiply(rows, thin, inputFactors) ||
      !checkedAdd(inputFactors, thin, inputFactors) ||
      !checkedMultiply(thin, columns, outputElements) ||
      !checkedAdd(inputFactors, outputElements, inputFactors) ||
      !checkedMultiply(columns, nullity, outputElements) ||
      !checkedMultiply(completionSize, completionRank, reflectorMatrix) ||
      !checkedAdd(reflectorMatrix, completionRank, reflectorElements) ||
      !checkedAdd(reflectorElements, completionSize, reflectorElements) ||
      !checkedMultiply(inputFactors, sizeof(Scalar), factorBytes) ||
      !checkedMultiply(outputElements, sizeof(Scalar), outputBytes) ||
      !checkedMultiply(reflectorElements, sizeof(Wide), reflectorBytes) ||
      !checkedAdd(inputFactors, outputElements, elements) ||
      !checkedAdd(elements, reflectorElements, elements) ||
      !checkedAdd(factorBytes, outputBytes, bytes) ||
      !checkedAdd(bytes, reflectorBytes, bytes)) {
    return false;
  }
  const auto ptrdiffLimit =
      static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max());
  return elements <= limits.elements && bytes <= limits.bytes &&
         outputElements <= ptrdiffLimit / sizeof(Scalar) &&
         reflectorMatrix <= ptrdiffLimit / sizeof(Wide) &&
         rank <= ptrdiffLimit / sizeof(Wide) &&
         columns <= ptrdiffLimit / sizeof(Wide);
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

Result<Matrix> nullSpace(ConstMatrixView input, SolvePolicy policy,
                         Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return failure<Matrix>(ErrorCode::InvalidTolerance,
                           "null-space rank tolerance is invalid");
  }
  const Index rows = input.rows();
  const Index columns = input.columns();
  const Index thin = std::min(rows, columns);
  const auto factor = factorSVD(input, policy, limits);
  if (!factor.hasValue()) {
    return copyFailure<Matrix>(factor.status());
  }
  const SVD& svd = factor.value();
  const Index rank = svd.rank;
  if (!nullSpaceStorage(rows, columns, thin, rank, limits)) {
    return failure<Matrix>(
        ErrorCode::ResourceLimit,
        "null-space factors, output, and workspace exceed limits");
  }

  Matrix result(columns, columns - rank, 0.0);
  if (rank == 0U) {
    for (Index coordinate = 0U; coordinate < columns; ++coordinate) {
      result.at(coordinate, coordinate) = 1.0;
    }
    return Result<Matrix>::success(std::move(result));
  }
  if (rank == columns) {
    return Result<Matrix>::success(std::move(result));
  }

  // Householder QR of the retained right-singular vectors. Since these columns
  // are orthonormal, pivoting is deterministic and only protects the completion
  // against rounding in the trailing subproblems.
  std::vector<Wide> reflectors(columns * rank, 0.0L);
  std::vector<Wide> tau(rank, 0.0L);
  for (Index row = 0U; row < columns; ++row) {
    for (Index direction = 0U; direction < rank; ++direction) {
      reflectors[(row * rank) + direction] =
          static_cast<Wide>(svd.right_transpose.at(direction, row));
    }
  }

  for (Index step = 0U; step < rank; ++step) {
    Index pivot = step;
    Wide pivotNorm = -1.0L;
    for (Index candidate = step; candidate < rank; ++candidate) {
      Wide norm = 0.0L;
      for (Index row = step; row < columns; ++row) {
        norm = std::hypot(norm, reflectors[(row * rank) + candidate]);
      }
      if (norm > pivotNorm) {
        pivotNorm = norm;
        pivot = candidate;
      }
    }
    if (!(pivotNorm > 0.0L) || !std::isfinite(pivotNorm)) {
      return failure<Matrix>(
          ErrorCode::NoConvergence,
          "null-space completion lost an independent direction");
    }
    if (pivot != step) {
      for (Index row = 0U; row < columns; ++row) {
        std::swap(reflectors[(row * rank) + step],
                  reflectors[(row * rank) + pivot]);
      }
    }

    const Wide first = reflectors[(step * rank) + step];
    const Wide alpha = -std::copysign(pivotNorm, first);
    const Wide denominator = first - alpha;
    if (denominator == 0.0L || alpha == 0.0L || !std::isfinite(alpha)) {
      return failure<Matrix>(ErrorCode::NoConvergence,
                             "null-space Householder reflector is unresolved");
    }
    tau[step] = (alpha - first) / alpha;
    for (Index row = step + 1U; row < columns; ++row) {
      reflectors[(row * rank) + step] /= denominator;
    }
    reflectors[(step * rank) + step] = alpha;

    for (Index candidate = step + 1U; candidate < rank; ++candidate) {
      Wide dot = reflectors[(step * rank) + candidate];
      for (Index row = step + 1U; row < columns; ++row) {
        dot += reflectors[(row * rank) + step] *
               reflectors[(row * rank) + candidate];
      }
      dot *= tau[step];
      reflectors[(step * rank) + candidate] -= dot;
      for (Index row = step + 1U; row < columns; ++row) {
        reflectors[(row * rank) + candidate] -=
            reflectors[(row * rank) + step] * dot;
      }
    }
  }

  std::vector<Wide> vector(columns, 0.0L);
  for (Index outputColumn = 0U; outputColumn < result.columns();
       ++outputColumn) {
    std::fill(vector.begin(), vector.end(), 0.0L);
    vector[rank + outputColumn] = 1.0L;
    for (Index reflector = rank; reflector-- > 0U;) {
      Wide dot = vector[reflector];
      for (Index row = reflector + 1U; row < columns; ++row) {
        dot += reflectors[(row * rank) + reflector] * vector[row];
      }
      dot *= tau[reflector];
      vector[reflector] -= dot;
      for (Index row = reflector + 1U; row < columns; ++row) {
        vector[row] -= reflectors[(row * rank) + reflector] * dot;
      }
    }
    for (Index row = 0U; row < columns; ++row) {
      if (!representableScalar(vector[row])) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "null-space basis is not representable");
      }
      result.at(row, outputColumn) = static_cast<Scalar>(vector[row]);
    }
  }
  return Result<Matrix>::success(std::move(result));
}

} // namespace vss::math::linear
