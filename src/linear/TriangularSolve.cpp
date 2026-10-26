#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/SolveOutcomeAccess.hpp>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/TriangularSolve.hpp>

namespace vss::math::linear {
namespace {

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool validRankTolerance(const RankTolerance& tolerance) noexcept {
  return std::isfinite(tolerance.absolute) &&
         std::isfinite(tolerance.relative) && tolerance.absolute >= 0.0 &&
         tolerance.relative >= 0.0;
}

Result<Scalar> validateFactor(ConstMatrixView factor, Extent rightHandSideRows,
                              Extent rightHandSideColumns,
                              TriangularSolveOptions options) {
  if (!validRankTolerance(options.rankTolerance)) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "triangular rank tolerance is invalid");
  }
  if (factor.rows() == 0U || factor.columns() == 0U) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidExtent,
        "triangular factor must be non-empty and square");
  }
  if (factor.rows() != factor.columns()) {
    return Result<Scalar>::failure(ErrorCode::DimensionMismatch,
                                   "triangular factor must be square");
  }
  if (factor.rows() != rightHandSideRows) {
    return Result<Scalar>::failure(
        ErrorCode::DimensionMismatch,
        "triangular factor and right-hand side rows differ");
  }
  if (rightHandSideColumns == 0U) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidExtent,
        "triangular right-hand side must have at least one column");
  }
  Extent resultSize = 0U;
  if (!checkedMultiply(factor.rows(), rightHandSideColumns, resultSize)) {
    return Result<Scalar>::failure(
        ErrorCode::SizeOverflow,
        "triangular solution size overflows its extent representation");
  }

  Scalar scale = options.diagonal == DiagonalMode::Unit ? 1.0 : 0.0;
  const bool lower = options.triangle == Triangle::Lower;
  for (Index row = 0U; row < factor.rows(); ++row) {
    const Index first = lower ? 0U : row;
    const Index last = lower ? row : factor.columns() - 1U;
    for (Index column = first; column <= last; ++column) {
      if (options.diagonal == DiagonalMode::Unit && row == column) {
        continue;
      }
      const Scalar value = factor.at(row, column);
      if (!std::isfinite(value)) {
        return Result<Scalar>::failure(
            ErrorCode::NonFiniteInput,
            "triangular required factor entry is not finite");
      }
      if (row == column) {
        scale = std::max(scale, std::abs(value));
      }
    }
  }

  const Scalar relativeThreshold = options.rankTolerance.relative * scale;
  if (!std::isfinite(relativeThreshold)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "triangular rank threshold is not finite");
  }
  const Scalar threshold =
      std::max(options.rankTolerance.absolute, relativeThreshold);
  if (!std::isfinite(threshold)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "triangular rank threshold is not finite");
  }
  if (options.diagonal == DiagonalMode::Explicit) {
    for (Index row = 0U; row < factor.rows(); ++row) {
      if (!(std::abs(factor.at(row, row)) > threshold)) {
        return Result<Scalar>::failure(
            ErrorCode::Singular,
            "triangular diagonal entry is not above rank threshold");
      }
    }
  }
  return Result<Scalar>::success(threshold);
}

bool finiteVector(ConstVectorView values) noexcept {
  for (Index index = 0U; index < values.size(); ++index) {
    if (!std::isfinite(values.at(index))) {
      return false;
    }
  }
  return true;
}

bool finiteMatrix(ConstMatrixView values) noexcept {
  for (Index row = 0U; row < values.rows(); ++row) {
    for (Index column = 0U; column < values.columns(); ++column) {
      if (!std::isfinite(values.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

Triangle effectiveTriangle(Triangle selected,
                           TransposeMode transpose) noexcept {
  if (transpose == TransposeMode::None) {
    return selected;
  }
  return selected == Triangle::Lower ? Triangle::Upper : Triangle::Lower;
}

Scalar coefficient(ConstMatrixView factor, Index row, Index column,
                   TransposeMode transpose) {
  const Index storedRow = transpose == TransposeMode::None ? row : column;
  const Index storedColumn = transpose == TransposeMode::None ? column : row;
  return factor.at(storedRow, storedColumn);
}

Result<SolveOutcome<Vector>> solveVectorImpl(ConstMatrixView factor,
                                             ConstVectorView rightHandSide,
                                             TriangularSolveOptions options) {
  const auto validation =
      validateFactor(factor, rightHandSide.size(), 1U, options);
  if (!validation.hasValue()) {
    return Result<SolveOutcome<Vector>>::failure(
        validation.status().code(), std::string(validation.status().message()));
  }
  if (!finiteVector(rightHandSide)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::NonFiniteInput,
        "triangular right-hand side contains a non-finite value");
  }

  Vector solution(factor.rows(), 0.0);
  const bool lower =
      effectiveTriangle(options.triangle, options.transpose) == Triangle::Lower;
  for (Index step = 0U; step < factor.rows(); ++step) {
    const Index row = lower ? step : factor.rows() - 1U - step;
    vss::math::detail::CompensatedSum accumulated;
    if (lower) {
      for (Index column = 0U; column < row; ++column) {
        accumulated.add(coefficient(factor, row, column, options.transpose) *
                        solution.at(column));
      }
    } else {
      for (Index column = row + 1U; column < factor.columns(); ++column) {
        accumulated.add(coefficient(factor, row, column, options.transpose) *
                        solution.at(column));
      }
    }
    const Scalar diagonal =
        options.diagonal == DiagonalMode::Unit
            ? 1.0
            : coefficient(factor, row, row, options.transpose);
    const Scalar value =
        (rightHandSide.at(row) - accumulated.total()) / diagonal;
    if (!std::isfinite(value)) {
      return Result<SolveOutcome<Vector>>::failure(
          ErrorCode::NonFiniteResult,
          "triangular solve produced a non-finite solution value");
    }
    solution.at(row) = value;
  }
  return Result<SolveOutcome<Vector>>::success(
      detail::SolveOutcomeAccess<Vector>::make(std::move(solution),
                                               std::nullopt));
}

Result<SolveOutcome<Matrix>> solveMatrixImpl(ConstMatrixView factor,
                                             ConstMatrixView rightHandSides,
                                             TriangularSolveOptions options) {
  const auto validation = validateFactor(factor, rightHandSides.rows(),
                                         rightHandSides.columns(), options);
  if (!validation.hasValue()) {
    return Result<SolveOutcome<Matrix>>::failure(
        validation.status().code(), std::string(validation.status().message()));
  }
  if (!finiteMatrix(rightHandSides)) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::NonFiniteInput,
        "triangular right-hand side contains a non-finite value");
  }

  Matrix solution(factor.rows(), rightHandSides.columns(), 0.0);
  const bool lower =
      effectiveTriangle(options.triangle, options.transpose) == Triangle::Lower;
  for (Index column = 0U; column < rightHandSides.columns(); ++column) {
    for (Index step = 0U; step < factor.rows(); ++step) {
      const Index row = lower ? step : factor.rows() - 1U - step;
      vss::math::detail::CompensatedSum accumulated;
      if (lower) {
        for (Index factorColumn = 0U; factorColumn < row; ++factorColumn) {
          accumulated.add(
              coefficient(factor, row, factorColumn, options.transpose) *
              solution.at(factorColumn, column));
        }
      } else {
        for (Index factorColumn = row + 1U; factorColumn < factor.columns();
             ++factorColumn) {
          accumulated.add(
              coefficient(factor, row, factorColumn, options.transpose) *
              solution.at(factorColumn, column));
        }
      }
      const Scalar diagonal =
          options.diagonal == DiagonalMode::Unit
              ? 1.0
              : coefficient(factor, row, row, options.transpose);
      const Scalar value =
          (rightHandSides.at(row, column) - accumulated.total()) / diagonal;
      if (!std::isfinite(value)) {
        return Result<SolveOutcome<Matrix>>::failure(
            ErrorCode::NonFiniteResult,
            "triangular solve produced a non-finite solution value");
      }
      solution.at(row, column) = value;
    }
  }
  return Result<SolveOutcome<Matrix>>::success(
      detail::SolveOutcomeAccess<Matrix>::make(std::move(solution),
                                               std::nullopt));
}

} // namespace

Result<SolveOutcome<Vector>> solveTriangular(ConstMatrixView factor,
                                             ConstVectorView rightHandSide,
                                             TriangularSolveOptions options) {
  return solveVectorImpl(factor, rightHandSide, options);
}

Result<SolveOutcome<Matrix>> solveTriangular(ConstMatrixView factor,
                                             ConstMatrixView rightHandSides,
                                             TriangularSolveOptions options) {
  return solveMatrixImpl(factor, rightHandSides, options);
}

} // namespace vss::math::linear
