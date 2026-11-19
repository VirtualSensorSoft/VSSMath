#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool representableScalar(Wide value) noexcept {
  if (!detail::representable(value)) {
    return false;
  }
  return value == 0.0L || static_cast<Scalar>(value) != 0.0;
}

bool checkedSquare(Index size, Index& square) noexcept {
  if (size != 0U && size > std::numeric_limits<Index>::max() / size) {
    return false;
  }
  square = size * size;
  return true;
}

bool preflight(Index size, Limits limits) {
  if (size == 0U) {
    return true;
  }
  Index square = 0U;
  if (!checkedSquare(size, square)) {
    return false;
  }
  detail::StorageBudget budget(limits);
  // Cover canonical inputs, the wide transformed matrix, Cholesky factor,
  // eigensolver work/results, output, and bounded vector workspaces together.
  for (Index dense = 0U; dense < 9U; ++dense) {
    if (!budget.claim<Scalar>(size, size)) {
      return false;
    }
  }
  if (!budget.claim<Wide>(size, size) || !budget.claim<Wide>(size, 8U) ||
      !budget.claim<Scalar>(size, 8U) || !budget.claim<Index>(size, 2U)) {
    return false;
  }
  return square <=
         static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max()) /
             sizeof(Wide);
}

bool finiteAndScale(ConstMatrixView matrix, Scalar& maximum, int& exponent) {
  maximum = 0.0;
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      const Scalar value = matrix.at(row, column);
      if (!std::isfinite(value)) {
        return false;
      }
      maximum = std::max(maximum, std::abs(value));
    }
  }
  exponent = maximum == 0.0 ? 0 : std::ilogb(maximum) + 1;
  return true;
}

bool symmetricWithinAllowance(ConstMatrixView matrix, Scalar maximum) {
  if (maximum == 0.0) {
    return true;
  }
  const Wide allowance =
      64.0L * static_cast<Wide>(std::max<Index>(1U, matrix.rows())) *
      static_cast<Wide>(std::numeric_limits<Scalar>::epsilon());
  const Wide scale = static_cast<Wide>(maximum);
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = row + 1U; column < matrix.columns(); ++column) {
      const Wide upper = static_cast<Wide>(matrix.at(row, column)) / scale;
      const Wide lower =
          static_cast<Wide>(
              matrix.at( // NOLINT(readability-suspicious-call-argument):
                         // Compare the transposed symmetric entry.
                  column, row)) /
          scale;
      if (std::abs(upper - lower) > allowance) {
        return false;
      }
    }
  }
  return true;
}

Matrix canonicalScaled(ConstMatrixView input, int exponent) {
  Matrix result(input.rows(), input.columns(), 0.0);
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = row; column < input.columns(); ++column) {
      const Scalar midpoint = std::midpoint(
          input.at(row, column),
          input.at( // NOLINT(readability-suspicious-call-argument):
                    // Canonicalization averages mirrored input entries.
              column, row));
      const Scalar scaled = std::scalbn(midpoint, -exponent);
      result.at(row, column) = scaled;
      result.at( // NOLINT(readability-suspicious-call-argument):
                 // Mirror the canonical lower-triangle value.
          column, row) = scaled;
    }
  }
  return result;
}

bool exactlyDiagonal(ConstMatrixView matrix) {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (row != column && matrix.at(row, column) != 0.0) {
        return false;
      }
    }
  }
  return true;
}

bool cholesky(ConstMatrixView input, Matrix& lower) {
  const Index size = input.rows();
  lower = Matrix(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column <= row; ++column) {
      Wide value = static_cast<Wide>(input.at(row, column));
      for (Index inner = 0U; inner < column; ++inner) {
        value -=
            static_cast<Wide>(lower.at(row, inner)) *
            static_cast<Wide>(
                lower.at( // NOLINT(readability-suspicious-call-argument):
                          // Symmetric Cholesky product reads L(column, inner).
                    column, inner));
      }
      if (row == column) {
        if (!(value > 0.0L) || !std::isfinite(value)) {
          return false;
        }
        value = std::sqrt(value);
      } else {
        value /= static_cast<Wide>(lower.at(column, column));
      }
      if (!detail::representable(value)) {
        return false;
      }
      lower.at(row, column) = static_cast<Scalar>(value);
    }
  }
  return true;
}

} // namespace

Result<GeneralizedEigen> factorGeneralizedSymmetric(ConstMatrixView a,
                                                    ConstMatrixView b,
                                                    SolvePolicy policy,
                                                    Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return failure<GeneralizedEigen>(ErrorCode::InvalidTolerance,
                                     "generalized eigen policy is invalid");
  }
  switch (policy.rank) {
  case RankPolicy::reject:
  case RankPolicy::supported_subspace:
  case RankPolicy::minimum_norm:
    break;
  default:
    return failure<GeneralizedEigen>(
        ErrorCode::InvalidDomain, "generalized eigen rank policy is invalid");
  }
  if (a.rows() != a.columns() || b.rows() != b.columns() ||
      a.rows() != b.rows()) {
    return failure<GeneralizedEigen>(
        ErrorCode::DimensionMismatch,
        "generalized eigenproblem requires equally sized square matrices");
  }
  const Index size = a.rows();
  if (size == 0U) {
    return Result<GeneralizedEigen>::success(
        {.values = Vector{}, .vectors = Matrix{}, .residual_norms = Vector{}});
  }
  if (!preflight(size, limits)) {
    return failure<GeneralizedEigen>(
        ErrorCode::ResourceLimit, "generalized eigen workspace exceeds limits");
  }

  Scalar maximumA = 0.0;
  Scalar maximumB = 0.0;
  int exponentA = 0;
  int exponentB = 0;
  if (!finiteAndScale(a, maximumA, exponentA) ||
      !finiteAndScale(b, maximumB, exponentB)) {
    return failure<GeneralizedEigen>(ErrorCode::NonFiniteInput,
                                     "generalized eigen inputs must be finite");
  }
  if (!symmetricWithinAllowance(a, maximumA) ||
      !symmetricWithinAllowance(b, maximumB)) {
    return failure<GeneralizedEigen>(
        ErrorCode::NotSymmetric,
        "pencil matrices exceed the symmetry allowance");
  }
  Matrix scaledA = canonicalScaled(a, exponentA);
  Matrix scaledB = canonicalScaled(b, exponentB);

  if (policy.iteration.max_iterations == 0U &&
      !exactlyDiagonal(scaledB.view())) {
    return failure<GeneralizedEigen>(
        ErrorCode::NoConvergence,
        "zero spectral cap requires a diagonal metric and reduced matrix");
  }
  Scalar minimumMetricEigenvalue = 0.0;
  Scalar maximumMetricEigenvalue = 0.0;
  const auto metricEigen = [&]() {
    SymmetricEigenOptions options;
    options.rankTolerance = {.absolute = 0.0, .relative = 0.0};
    options.symmetryTolerance = {.absolute = 0.0, .relative = 0.0};
    options.maximumSweeps = static_cast<Extent>(
        std::clamp<Index>(policy.iteration.max_iterations, 1U, 1024U));
    return decomposeSymmetricEigen(scaledB.view(), options);
  }();
  if (!metricEigen.hasValue()) {
    return copyFailure<GeneralizedEigen>(metricEigen.status());
  }
  minimumMetricEigenvalue = metricEigen.value().minimumEigenvalue();
  maximumMetricEigenvalue = metricEigen.value().maximumEigenvalue();
  if (!(minimumMetricEigenvalue > 0.0)) {
    return failure<GeneralizedEigen>(
        ErrorCode::NotPositiveDefinite,
        "metric has a nonpositive eigen-direction");
  }
  const Wide scaledAbsoluteCutoff =
      std::scalbn(static_cast<Wide>(policy.rank_abs), -exponentB);
  const Wide rankCutoff = std::max(
      scaledAbsoluteCutoff, static_cast<Wide>(policy.rank_rel) *
                                static_cast<Wide>(maximumMetricEigenvalue));
  if (static_cast<Wide>(minimumMetricEigenvalue) <= rankCutoff) {
    return failure<GeneralizedEigen>(ErrorCode::RankDeficient,
                                     "metric is below its numeric rank cutoff");
  }
  const Wide reciprocalCondition = static_cast<Wide>(minimumMetricEigenvalue) /
                                   static_cast<Wide>(maximumMetricEigenvalue);
  if (reciprocalCondition <
      static_cast<Wide>(policy.reciprocal_condition_floor)) {
    return failure<GeneralizedEigen>(
        ErrorCode::IllConditioned,
        "metric is below the requested condition floor");
  }

  Matrix lower;
  if (!cholesky(scaledB.view(), lower)) {
    return failure<GeneralizedEigen>(
        ErrorCode::NoConvergence,
        "metric Cholesky factorization did not converge");
  }

  Index square = 0U;
  static_cast<void>(checkedSquare(size, square));
  std::vector<Wide> transformed(square, 0.0L);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      Wide value = static_cast<Wide>(scaledA.at(row, column));
      for (Index inner = 0U; inner < row; ++inner) {
        value -= static_cast<Wide>(lower.at(row, inner)) *
                 transformed[(inner * size) + column];
      }
      transformed[(row * size) + column] =
          value / static_cast<Wide>(lower.at(row, row));
      if (!std::isfinite(transformed[(row * size) + column])) {
        return failure<GeneralizedEigen>(
            ErrorCode::NonFiniteResult,
            "left triangular transform produced a non-finite value");
      }
    }
  }
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      Wide value = transformed[(row * size) + column];
      for (Index inner = 0U; inner < column; ++inner) {
        value -=
            transformed[(row * size) + inner] *
            static_cast<Wide>(
                lower.at( // NOLINT(readability-suspicious-call-argument):
                          // Right triangular solve reads the L(column, inner)
                          // entry for multiplication by L transpose.
                    column, inner));
      }
      transformed[(row * size) + column] =
          value / static_cast<Wide>(lower.at(column, column));
      if (!std::isfinite(transformed[(row * size) + column])) {
        return failure<GeneralizedEigen>(
            ErrorCode::NonFiniteResult,
            "right triangular transform produced a non-finite value");
      }
    }
  }
  Wide maximumTransformed = 0.0L;
  for (Index row = 0U; row < size; ++row) {
    for (Index column = row; column < size; ++column) {
      Wide midpoint = 0.0L;
      midpoint = std::midpoint(transformed[(row * size) + column],
                               transformed[(column * size) + row]);
      transformed[(row * size) + column] = midpoint;
      transformed[(column * size) + row] = midpoint;
      maximumTransformed = std::max(maximumTransformed, std::abs(midpoint));
    }
  }
  int exponentTransformed = 0;
  if (maximumTransformed != 0.0L) {
    static_cast<void>(std::frexp(maximumTransformed, &exponentTransformed));
  }
  Matrix reduced(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      const Wide value =
          std::scalbn(transformed[(row * size) + column], -exponentTransformed);
      if (!representableScalar(value)) {
        return failure<GeneralizedEigen>(
            ErrorCode::NonFiniteResult,
            "reduced symmetric matrix cannot be represented");
      }
      reduced.at(row, column) = static_cast<Scalar>(value);
    }
  }
  if (policy.iteration.max_iterations == 0U &&
      !exactlyDiagonal(reduced.view())) {
    return failure<GeneralizedEigen>(
        ErrorCode::NoConvergence,
        "zero spectral cap requires a diagonal metric and reduced matrix");
  }
  SymmetricEigenOptions reducedOptions;
  reducedOptions.rankTolerance = {.absolute = 0.0, .relative = 0.0};
  reducedOptions.symmetryTolerance = {.absolute = 0.0, .relative = 0.0};
  reducedOptions.maximumSweeps = static_cast<Extent>(
      std::clamp<Index>(policy.iteration.max_iterations, 1U, 1024U));
  const auto reducedEigen =
      decomposeSymmetricEigen(reduced.view(), reducedOptions);
  if (!reducedEigen.hasValue()) {
    return copyFailure<GeneralizedEigen>(reducedEigen.status());
  }

  GeneralizedEigen output{.values = Vector(size, 0.0),
                          .vectors = Matrix(size, size, 0.0),
                          .residual_norms = Vector(size, 0.0)};
  const Wide metricNormalization = std::sqrt(std::scalbn(1.0L, -exponentB));
  std::vector<Wide> vector(size, 0.0L);
  for (Index eigen = 0U; eigen < size; ++eigen) {
    const Wide eigenvalue = std::scalbn(
        static_cast<Wide>(reducedEigen.value().eigenvalues().at(eigen)),
        exponentA - exponentB + exponentTransformed);
    if (!representableScalar(eigenvalue)) {
      return failure<GeneralizedEigen>(
          ErrorCode::NonFiniteResult,
          "generalized eigenvalue is not representable");
    }
    output.values.at(eigen) = static_cast<Scalar>(eigenvalue);

    for (Index offset = 0U; offset < size; ++offset) {
      const Index row = size - offset - 1U;
      Wide value =
          static_cast<Wide>(reducedEigen.value().eigenvectors().at(row, eigen));
      for (Index inner = row + 1U; inner < size; ++inner) {
        value -= static_cast<Wide>(
                     lower.at( // NOLINT(readability-suspicious-call-argument):
                               // Back substitution solves L transpose times the
                               // vector, so it reads L(inner, row).
                         inner, row)) *
                 vector[inner];
      }
      vector[row] = value / static_cast<Wide>(lower.at(row, row));
    }
    Wide metricNormSquared = 0.0L;
    for (Index row = 0U; row < size; ++row) {
      Wide product = 0.0L;
      for (Index column = 0U; column < size; ++column) {
        product += static_cast<Wide>(scaledB.at(row, column)) * vector[column];
      }
      metricNormSquared += vector[row] * product;
    }
    if (!(metricNormSquared > 0.0L) || !std::isfinite(metricNormSquared)) {
      return failure<GeneralizedEigen>(
          ErrorCode::NonFiniteResult,
          "metric eigenvector normalization failed");
    }
    const Wide normalization =
        metricNormalization / std::sqrt(metricNormSquared);
    for (Index row = 0U; row < size; ++row) {
      const Wide component = vector[row] * normalization;
      if (!representableScalar(component)) {
        return failure<GeneralizedEigen>(
            ErrorCode::NonFiniteResult,
            "eigenvector component is not representable");
      }
      output.vectors.at(row, eigen) = static_cast<Scalar>(component);
    }

    Wide residual = 0.0L;
    for (Index row = 0U; row < size; ++row) {
      Wide aProduct = 0.0L;
      Wide bProduct = 0.0L;
      for (Index column = 0U; column < size; ++column) {
        const Wide component = static_cast<Wide>(
            output.vectors.at( // NOLINT(readability-suspicious-call-argument):
                               // Residual uses the current eigenvector's row
                               // component from the output column.
                column, eigen));
        aProduct += static_cast<Wide>(a.at(row, column)) * component;
        bProduct += static_cast<Wide>(b.at(row, column)) * component;
      }
      residual = std::hypot(residual, aProduct - (eigenvalue * bProduct));
    }
    if (!representableScalar(residual)) {
      return failure<GeneralizedEigen>(
          ErrorCode::NonFiniteResult,
          "eigenpair residual is not representable");
    }
    output.residual_norms.at(eigen) = static_cast<Scalar>(residual);
  }
  return Result<GeneralizedEigen>::success(std::move(output));
}

} // namespace vss::math::linear
