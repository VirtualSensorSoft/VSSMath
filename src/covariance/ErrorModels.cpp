#include "vss/math/covariance/Conditioning.hpp"

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
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Budget.hpp>
#include <vss/math/covariance/ErrorModels.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool finite(ConstVectorView vector) noexcept {
  for (Index index = 0U; index < vector.size(); ++index) {
    if (!std::isfinite(vector.at(index))) {
      return false;
    }
  }
  return true;
}

bool checkedProduct(Index left, Index right, Index& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

Result<Index> preflight(Index dimension, Limits limits) {
  Index square = 0U;
  Index workspace = 0U;
  if (!checkedProduct(dimension, dimension, square) ||
      !checkedProduct(square, 3U, workspace)) {
    return Result<Index>::failure(ErrorCode::SizeOverflow,
                                  "error covariance workspace overflows");
  }
  if (workspace > limits.elements ||
      workspace > limits.bytes / sizeof(Scalar)) {
    return Result<Index>::failure(ErrorCode::ResourceLimit,
                                  "error covariance exceeds storage limits");
  }
  return Result<Index>::success(square);
}

linear::SymmetricEigenOptions eigenOptions(SolvePolicy policy) {
  return {.rankTolerance = {.absolute = policy.rank_abs,
                            .relative = policy.rank_rel},
          .symmetryTolerance = {.absolute = policy.iteration.abs,
                                .relative = policy.iteration.rel},
          .maximumSweeps =
              std::min<Index>(policy.iteration.max_iterations, 1024U)};
}

SolvePolicy constructionPolicy() {
  SolvePolicy policy{};
  policy.iteration = {.abs = 1.0e-12,
                      .rel = 1.0e-12,
                      .max_iterations = 64U,
                      .max_evaluations = 0U,
                      .bracket = std::nullopt};
  policy.rank_abs = 0.0;
  policy.rank_rel = 1.0e-12;
  policy.reciprocal_condition_floor = 1.0e-10;
  policy.rank = RankPolicy::minimum_norm;
  return policy;
}

bool validPolicy(SolvePolicy policy) noexcept {
  const bool validRank = policy.rank == RankPolicy::reject ||
                         policy.rank == RankPolicy::supported_subspace ||
                         policy.rank == RankPolicy::minimum_norm;
  return validRank && std::isfinite(policy.iteration.abs) &&
         policy.iteration.abs > 0.0 && std::isfinite(policy.iteration.rel) &&
         policy.iteration.rel > 0.0 && policy.iteration.max_iterations > 0U &&
         std::isfinite(policy.rank_abs) && policy.rank_abs >= 0.0 &&
         std::isfinite(policy.rank_rel) && policy.rank_rel >= 0.0 &&
         policy.rank_rel < 1.0 &&
         std::isfinite(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor >= 0.0 &&
         policy.reciprocal_condition_floor <= 1.0;
}

Result<CovarianceResult> classify(Matrix covariance, SolvePolicy policy,
                                  bool requirePsd) {
  const Index dimension = covariance.rows();
  if (dimension == 0U) {
    return Result<CovarianceResult>::success(
        {.covariance = std::move(covariance),
         .classification = CovarianceKind::semidefinite_singular,
         .rank = 0U,
         .cutoff = 0.0,
         .reciprocal_condition = std::nullopt});
  }
  const auto eigen =
      linear::decomposeSymmetricEigen(covariance.view(), eigenOptions(policy));
  if (!eigen) {
    return Result<CovarianceResult>::failure(
        eigen.status().code(), std::string(eigen.status().message()));
  }
  const auto definiteness = eigen.value().definiteness();
  const bool psd =
      definiteness != linear::PositiveDefiniteness::NotPositiveSemidefinite;
  if (requirePsd && !psd) {
    return fail<CovarianceResult>(ErrorCode::NotPositiveSemidefinite,
                                  "error covariance is materially indefinite");
  }
  const Index rank = eigen.value().rank();
  const Scalar cutoff = eigen.value().rankThreshold();
  std::optional<Scalar> reciprocal;
  CovarianceKind kind = CovarianceKind::indefinite;
  if (psd) {
    if (rank < dimension) {
      kind = CovarianceKind::semidefinite_singular;
    } else {
      const Scalar minimum = eigen.value().minimumEigenvalue();
      const Scalar maximum = eigen.value().maximumEigenvalue();
      const Scalar ratio = minimum / maximum;
      reciprocal = ratio;
      kind = ratio < policy.reciprocal_condition_floor
                 ? CovarianceKind::ill_conditioned
                 : CovarianceKind::positive_definite;
    }
  }
  return Result<CovarianceResult>::success(
      {.covariance = std::move(covariance),
       .classification = kind,
       .rank = rank,
       .cutoff = cutoff,
       .reciprocal_condition = reciprocal});
}

Result<Matrix> callEvaluator(const ErrorCovarianceEvaluator& evaluator,
                             ConstVectorView state, Scalar time) {
  try {
    auto result = evaluator(state, time);
    if (!result) {
      const ErrorCode code =
          result.status().code() == ErrorCode::DimensionMismatch
              ? ErrorCode::DimensionMismatch
              : ErrorCode::CallbackFailure;
      return fail<Matrix>(code, "error covariance callback failed");
    }
    return result;
  } catch (...) {
    return fail<Matrix>(ErrorCode::CallbackFailure,
                        "error covariance callback threw an exception");
  }
}

} // namespace

Result<CovarianceResult>
assembleErrorCovariance(const ErrorCovarianceSpec& spec, ConstVectorView state,
                        Scalar time, Limits limits) {
  if (!finite(state) || !std::isfinite(time)) {
    return fail<CovarianceResult>(ErrorCode::NonFiniteInput,
                                  "conditioning state and time must be finite");
  }

  if (spec.evaluate) {
    if (spec.diagonal || spec.loading || !spec.sources.empty() ||
        !spec.cross.empty()) {
      return fail<CovarianceResult>(ErrorCode::InvalidDomain,
                                    "callback and block models are exclusive");
    }
    if (limits.evaluations == 0U) {
      return fail<CovarianceResult>(ErrorCode::ResourceLimit,
                                    "error covariance evaluation cap exceeded");
    }
    auto evaluated = callEvaluator(spec.evaluate, state, time);
    if (!evaluated) {
      return Result<CovarianceResult>::failure(
          evaluated.status().code(), std::string(evaluated.status().message()));
    }
    if (evaluated.value().rows() != evaluated.value().columns()) {
      return fail<CovarianceResult>(ErrorCode::DimensionMismatch,
                                    "callback covariance must be square");
    }
    if (!finite(evaluated.value().view())) {
      return fail<CovarianceResult>(ErrorCode::NonFiniteInput,
                                    "callback covariance is not finite");
    }
    const auto capacity = preflight(evaluated.value().rows(), limits);
    if (!capacity) {
      return Result<CovarianceResult>::failure(
          capacity.status().code(), std::string(capacity.status().message()));
    }
    return classify(std::move(evaluated.value()), constructionPolicy(), true);
  }

  if (spec.sources.empty() && !spec.cross.empty()) {
    return fail<CovarianceResult>(ErrorCode::InvalidDomain,
                                  "cross blocks require declared sources");
  }
  if (!spec.sources.empty() && !spec.loading) {
    return fail<CovarianceResult>(ErrorCode::InvalidDomain,
                                  "source blocks require a loading matrix");
  }
  if (!spec.sources.empty() &&
      spec.dependence != DependencePolicy::exact_joint &&
      spec.dependence != DependencePolicy::declared_independent_blocks) {
    return fail<CovarianceResult>(ErrorCode::InvalidDomain,
                                  "dependence policy is invalid");
  }

  Index dimension = 0U;
  if (spec.diagonal) {
    dimension = spec.diagonal->size();
    for (Index index = 0U; index < dimension; ++index) {
      const Scalar value = spec.diagonal->at(index);
      if (!std::isfinite(value)) {
        return fail<CovarianceResult>(ErrorCode::NonFiniteInput,
                                      "independent variance is not finite");
      }
      if (value < 0.0) {
        return fail<CovarianceResult>(ErrorCode::NotPositiveSemidefinite,
                                      "independent variance is negative");
      }
    }
  } else if (spec.loading) {
    dimension = spec.loading->rows();
  }
  if (spec.loading && spec.diagonal && spec.loading->rows() != dimension) {
    return fail<CovarianceResult>(ErrorCode::DimensionMismatch,
                                  "loading rows must match the diagonal size");
  }
  if (spec.loading && !finite(*spec.loading)) {
    return fail<CovarianceResult>(ErrorCode::NonFiniteInput,
                                  "loading matrix contains a non-finite value");
  }
  const auto capacity = preflight(dimension, limits);
  if (!capacity) {
    return Result<CovarianceResult>::failure(
        capacity.status().code(), std::string(capacity.status().message()));
  }
  Matrix output(dimension, dimension, 0.0);
  if (spec.loading) {
    Index reservedElements = 0U;
    if (!checkedProduct(capacity.value(), 3U, reservedElements)) {
      return fail<CovarianceResult>(ErrorCode::SizeOverflow,
                                    "error covariance workspace overflows");
    }
    const Index reservedBytes = reservedElements * sizeof(Scalar);
    if (reservedElements > limits.elements || reservedBytes > limits.bytes) {
      return fail<CovarianceResult>(ErrorCode::ResourceLimit,
                                    "error covariance exceeds storage limits");
    }
    const Limits budgetLimits{.elements = limits.elements - reservedElements,
                              .bytes = limits.bytes - reservedBytes,
                              .evaluations = limits.evaluations};
    auto budget = uncertaintyContributionBudget(
        *spec.loading, spec.sources, spec.cross, spec.dependence, budgetLimits);
    if (!budget) {
      return Result<CovarianceResult>::failure(
          budget.status().code(), std::string(budget.status().message()));
    }
    output = std::move(budget.value().covariance);
  } else if (!spec.sources.empty() || !spec.cross.empty()) {
    return fail<CovarianceResult>(ErrorCode::InvalidDomain,
                                  "source data require a loading matrix");
  }
  if (spec.diagonal) {
    for (Index index = 0U; index < dimension; ++index) {
      const long double sum =
          static_cast<long double>(output.at(index, index)) +
          spec.diagonal->at(index);
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max()) {
        return fail<CovarianceResult>(ErrorCode::NonFiniteResult,
                                      "assembled covariance is out of range");
      }
      output.at(index, index) = static_cast<Scalar>(sum);
    }
  }
  return classify(std::move(output), constructionPolicy(), true);
}

Result<CovarianceResult>
subtractKnownErrorCovariance(ConstMatrixView observed,
                             ConstMatrixView knownError, SolvePolicy policy,
                             Limits limits) {
  if (!validPolicy(policy)) {
    return fail<CovarianceResult>(
        ErrorCode::InvalidTolerance,
        "covariance classification policy is invalid");
  }
  if (observed.rows() != observed.columns() ||
      knownError.rows() != knownError.columns()) {
    return fail<CovarianceResult>(ErrorCode::DimensionMismatch,
                                  "covariance inputs must be square");
  }
  if (observed.rows() != knownError.rows()) {
    return fail<CovarianceResult>(
        ErrorCode::DimensionMismatch,
        "covariance inputs must have equal dimensions");
  }
  if (!finite(observed) || !finite(knownError)) {
    return fail<CovarianceResult>(
        ErrorCode::NonFiniteInput,
        "covariance input contains a non-finite value");
  }
  const auto capacity = preflight(observed.rows(), limits);
  if (!capacity) {
    return Result<CovarianceResult>::failure(
        capacity.status().code(), std::string(capacity.status().message()));
  }
  const auto options = eigenOptions(policy);
  const auto observedEigen = linear::decomposeSymmetricEigen(observed, options);
  if (!observedEigen) {
    return Result<CovarianceResult>::failure(
        observedEigen.status().code(),
        std::string(observedEigen.status().message()));
  }
  const auto knownEigen = linear::decomposeSymmetricEigen(knownError, options);
  if (!knownEigen) {
    return Result<CovarianceResult>::failure(
        knownEigen.status().code(), std::string(knownEigen.status().message()));
  }
  if (observed.rows() != 0U &&
      (observedEigen.value().definiteness() ==
           linear::PositiveDefiniteness::NotPositiveSemidefinite ||
       knownEigen.value().definiteness() ==
           linear::PositiveDefiniteness::NotPositiveSemidefinite)) {
    return fail<CovarianceResult>(
        ErrorCode::NotPositiveSemidefinite,
        "subtraction inputs must be positive semidefinite");
  }

  const Index dimension = observed.rows();
  Matrix difference(dimension, dimension, 0.0);
  for (Index firstIndex = 0U; firstIndex < dimension; ++firstIndex) {
    for (Index secondIndex = firstIndex; secondIndex < dimension;
         ++secondIndex) {
      const long double upper =
          static_cast<long double>(observed.at(firstIndex, secondIndex)) -
          knownError.at(firstIndex, secondIndex);
      const long double lower =
          static_cast<long double>(observed.at(secondIndex, firstIndex)) -
          knownError.at(secondIndex, firstIndex);
      const long double symmetric = 0.5L * (upper + lower);
      if (!std::isfinite(symmetric) ||
          std::abs(symmetric) > std::numeric_limits<Scalar>::max()) {
        return fail<CovarianceResult>(
            ErrorCode::NonFiniteResult,
            "signed covariance difference is out of range");
      }
      difference.at(firstIndex, secondIndex) = static_cast<Scalar>(symmetric);
      difference.at(secondIndex, firstIndex) = static_cast<Scalar>(symmetric);
    }
  }
  return classify(std::move(difference), policy, false);
}

} // namespace vss::math::covariance
