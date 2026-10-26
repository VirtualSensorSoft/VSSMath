#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/ConditionEstimate.hpp>
#include <detail/SolveOutcomeAccess.hpp>
#include <detail/SolveValidation.hpp>
#include <limits>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedSquare(Extent extent, Extent& count) noexcept {
  if (extent != 0U && extent > std::numeric_limits<Extent>::max() / extent) {
    return false;
  }
  count = extent * extent;
  return true;
}

bool validFactorizationOptions(const FactorizationOptions& options) noexcept {
  return std::isfinite(options.rankTolerance.absolute) &&
         std::isfinite(options.rankTolerance.relative) &&
         options.rankTolerance.absolute >= 0.0 &&
         options.rankTolerance.relative >= 0.0 &&
         std::isfinite(options.conditionLimit) && options.conditionLimit > 1.0;
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

bool nearlySymmetric(Scalar lhs, Scalar rhs, Tolerance tolerance) noexcept {
  if (lhs == rhs) {
    return true;
  }
  const Scalar scale = std::max(std::abs(lhs), std::abs(rhs));
  if (scale == 0.0) {
    return true;
  }
  const Scalar normalizedDifference = std::abs((lhs / scale) - (rhs / scale));
  return normalizedDifference <=
         (tolerance.absolute / scale) + tolerance.relative;
}

Result<Scalar> matrixOneNorm(ConstMatrixView matrix) {
  Scalar norm = 0.0;
  for (Index column = 0U; column < matrix.columns(); ++column) {
    vss::math::detail::CompensatedSum columnSum;
    for (Index row = 0U; row < matrix.rows(); ++row) {
      columnSum.add(std::abs(matrix.at(row, column)));
    }
    const Scalar value = columnSum.total();
    if (!std::isfinite(value)) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "Cholesky matrix one-norm is not finite");
    }
    norm = std::max(norm, value);
  }
  return Result<Scalar>::success(norm);
}

Result<Scalar> rankThreshold(ConstMatrixView matrix,
                             const FactorizationOptions& options) {
  Scalar scale = 0.0;
  for (Index diagonal = 0U; diagonal < matrix.rows(); ++diagonal) {
    scale = std::max(scale, std::abs(matrix.at(diagonal, diagonal)));
  }
  const Scalar relative = options.rankTolerance.relative * scale;
  if (!std::isfinite(relative)) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "Cholesky relative rank threshold is not finite");
  }
  const Scalar threshold = std::max(options.rankTolerance.absolute, relative);
  if (!std::isfinite(threshold)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "Cholesky rank threshold is not finite");
  }
  return Result<Scalar>::success(threshold);
}

Result<Vector> solveLower(ConstMatrixView lower,
                          ConstVectorView rightHandSide) {
  const Extent size = lower.rows();
  Vector work(size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    vss::math::detail::CompensatedSum accumulated;
    for (Index column = 0U; column < row; ++column) {
      accumulated.add(lower.at(row, column) * work.at(column));
    }
    const Scalar value =
        (rightHandSide.at(row) - accumulated.total()) / lower.at(row, row);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "Cholesky forward substitution produced a non-finite value");
    }
    work.at(row) = value;
  }
  return Result<Vector>::success(std::move(work));
}

Result<Vector> solveFull(ConstMatrixView lower, ConstVectorView rightHandSide) {
  auto forward = solveLower(lower, rightHandSide);
  if (!forward.hasValue()) {
    return forward;
  }
  Vector work = std::move(forward.value());
  const Extent size = lower.rows();
  for (Index step = 0U; step < size; ++step) {
    const Index row = size - 1U - step;
    vss::math::detail::CompensatedSum accumulated;
    for (Index column = row + 1U; column < size; ++column) {
      const Index transposeRow = column;
      const Index transposeColumn = row;
      const Scalar transposedFactor = lower.at(transposeRow, transposeColumn);
      accumulated.add(transposedFactor * work.at(column));
    }
    const Scalar value =
        (work.at(row) - accumulated.total()) / lower.at(row, row);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "Cholesky backward substitution produced a non-finite value");
    }
    work.at(row) = value;
  }
  return Result<Vector>::success(std::move(work));
}

Result<Matrix> solveFull(ConstMatrixView lower,
                         ConstMatrixView rightHandSides) {
  Matrix solution(lower.rows(), rightHandSides.columns(), 0.0);
  Vector work(lower.rows(), 0.0);
  for (Index rightHandSide = 0U; rightHandSide < rightHandSides.columns();
       ++rightHandSide) {
    for (Index row = 0U; row < lower.rows(); ++row) {
      work.at(row) = rightHandSides.at(row, rightHandSide);
    }
    auto solved = solveFull(lower, work.view());
    if (!solved.hasValue()) {
      return copyFailure<Matrix>(solved.status());
    }
    for (Index row = 0U; row < lower.rows(); ++row) {
      solution.at(row, rightHandSide) = solved.value().at(row);
    }
  }
  return Result<Matrix>::success(std::move(solution));
}

ConditionReport report(Scalar matrixNorm, Scalar condition, Scalar limit) {
  return ConditionReport{.matrixOneNorm = matrixNorm,
                         .estimatedConditionNumber1 = condition,
                         .conditionLimit = limit,
                         .classification =
                             condition >= limit
                                 ? ConditionClassification::IllConditioned
                                 : ConditionClassification::WithinLimit};
}

bool rejectsIllConditioned(const ConditionReport& conditionReport,
                           SolveOptions options) noexcept {
  return conditionReport.classification ==
             ConditionClassification::IllConditioned &&
         options.illConditionedPolicy == IllConditionedSolvePolicy::Reject;
}

} // namespace

CholeskyFactorization::CholeskyFactorization(Extent size, Extent rank,
                                             Scalar rankThreshold,
                                             Tolerance symmetryTolerance,
                                             Matrix lowerFactor,
                                             ConditionReport conditionReport,
                                             Scalar logDeterminant) noexcept
    : size_(size), rank_(rank), rankThreshold_(rankThreshold),
      symmetryTolerance_(symmetryTolerance),
      lowerFactor_(std::move(lowerFactor)), conditionReport_(conditionReport),
      logDeterminant_(logDeterminant) {}

void CholeskyFactorization::resetMovedFrom() noexcept {
  size_ = 0U;
  rank_ = 0U;
  rankThreshold_ = 0.0;
  symmetryTolerance_ = Tolerance{.absolute = 0.0, .relative = 0.0};
  lowerFactor_ = Matrix{};
  conditionReport_ =
      ConditionReport{.matrixOneNorm = 0.0,
                      .estimatedConditionNumber1 = 0.0,
                      .conditionLimit = 0.0,
                      .classification = ConditionClassification::WithinLimit};
  logDeterminant_ = 0.0;
}

CholeskyFactorization::CholeskyFactorization(
    CholeskyFactorization&& other) noexcept
    : size_(other.size_), rank_(other.rank_),
      rankThreshold_(other.rankThreshold_),
      symmetryTolerance_(other.symmetryTolerance_),
      lowerFactor_(std::move(other.lowerFactor_)),
      conditionReport_(other.conditionReport_),
      logDeterminant_(other.logDeterminant_) {
  other.resetMovedFrom();
}

CholeskyFactorization&
CholeskyFactorization::operator=(CholeskyFactorization&& other) noexcept {
  if (this != &other) {
    size_ = other.size_;
    rank_ = other.rank_;
    rankThreshold_ = other.rankThreshold_;
    symmetryTolerance_ = other.symmetryTolerance_;
    lowerFactor_ = std::move(other.lowerFactor_);
    conditionReport_ = other.conditionReport_;
    logDeterminant_ = other.logDeterminant_;
    other.resetMovedFrom();
  }
  return *this;
}

Extent CholeskyFactorization::size() const noexcept { return size_; }

Extent CholeskyFactorization::rank() const noexcept { return rank_; }

Scalar CholeskyFactorization::rankThreshold() const noexcept {
  return rankThreshold_;
}

Tolerance CholeskyFactorization::symmetryTolerance() const noexcept {
  return symmetryTolerance_;
}

const Matrix& CholeskyFactorization::lowerFactor() const noexcept {
  return lowerFactor_;
}

const ConditionReport& CholeskyFactorization::conditionReport() const noexcept {
  return conditionReport_;
}

Scalar CholeskyFactorization::logDeterminant() const noexcept {
  return logDeterminant_;
}

Result<SolveOutcome<Vector>>
CholeskyFactorization::solve(ConstVectorView rightHandSide,
                             SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Vector>>(optionStatus);
  }
  if (size_ == 0U) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::InvalidExtent, "Cholesky factorization is moved-from");
  }
  if (rightHandSide.size() != size_) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::DimensionMismatch,
        "Cholesky factorization and right-hand side dimensions differ");
  }
  if (rejectsIllConditioned(conditionReport_, options)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::IllConditioned,
        "Cholesky condition estimate meets or exceeds the configured condition "
        "limit");
  }
  if (!finite(rightHandSide)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::NonFiniteInput,
        "Cholesky right-hand side contains a non-finite value");
  }
  auto result = solveFull(lowerFactor_.view(), rightHandSide);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Vector>>(result.status());
  }
  return Result<SolveOutcome<Vector>>::success(
      detail::SolveOutcomeAccess<Vector>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<SolveOutcome<Matrix>>
CholeskyFactorization::solve(ConstMatrixView rightHandSides,
                             SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Matrix>>(optionStatus);
  }
  if (size_ == 0U) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::InvalidExtent, "Cholesky factorization is moved-from");
  }
  if (rightHandSides.rows() != size_) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::DimensionMismatch,
        "Cholesky factorization and right-hand side dimensions differ");
  }
  if (rightHandSides.columns() == 0U) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::InvalidExtent,
        "Cholesky matrix right-hand side must have at least one column");
  }
  if (rejectsIllConditioned(conditionReport_, options)) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::IllConditioned,
        "Cholesky condition estimate meets or exceeds the configured condition "
        "limit");
  }
  if (!finite(rightHandSides)) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::NonFiniteInput,
        "Cholesky matrix right-hand side contains a non-finite value");
  }
  auto result = solveFull(lowerFactor_.view(), rightHandSides);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Matrix>>(result.status());
  }
  return Result<SolveOutcome<Matrix>>::success(
      detail::SolveOutcomeAccess<Matrix>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<SolveOutcome<Vector>>
CholeskyFactorization::whiten(ConstVectorView residual,
                              SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Vector>>(optionStatus);
  }
  if (size_ == 0U) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::InvalidExtent, "Cholesky factorization is moved-from");
  }
  if (residual.size() != size_) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::DimensionMismatch,
        "Cholesky factorization and residual dimensions differ");
  }
  if (rejectsIllConditioned(conditionReport_, options)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::IllConditioned,
        "Cholesky condition estimate meets or exceeds the configured condition "
        "limit");
  }
  if (!finite(residual)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::NonFiniteInput,
        "Cholesky residual contains a non-finite value");
  }
  auto result = solveLower(lowerFactor_.view(), residual);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Vector>>(result.status());
  }
  return Result<SolveOutcome<Vector>>::success(
      detail::SolveOutcomeAccess<Vector>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<CholeskyFactorization>
factorizeCholesky(ConstMatrixView matrix,
                  SymmetricFactorizationOptions options) {
  if (!validFactorizationOptions(options.factorization) ||
      !validate(options.symmetryTolerance).ok()) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::InvalidTolerance,
        "Cholesky factorization options contain an invalid tolerance");
  }
  if (matrix.rows() == 0U || matrix.columns() == 0U) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::InvalidExtent,
        "Cholesky factorization requires a non-empty matrix");
  }
  if (matrix.rows() != matrix.columns()) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::DimensionMismatch,
        "Cholesky factorization requires a square matrix");
  }
  Extent elementCount = 0U;
  if (!checkedSquare(matrix.rows(), elementCount)) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::SizeOverflow, "Cholesky factorization matrix size overflows "
                                 "its extent representation");
  }
  static_cast<void>(elementCount);
  if (!finite(matrix)) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::NonFiniteInput,
        "Cholesky factorization input contains a non-finite value");
  }

  const Extent size = matrix.rows();
  Matrix canonical(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column <= row; ++column) {
      const Scalar lower = matrix.at(row, column);
      const Index mirroredRow = column;
      const Index mirroredColumn = row;
      const Scalar upper = matrix.at(mirroredRow, mirroredColumn);
      if (!nearlySymmetric(lower, upper, options.symmetryTolerance)) {
        return Result<CholeskyFactorization>::failure(
            ErrorCode::NotSymmetric,
            "Cholesky factorization input is outside the symmetry tolerance");
      }
      canonical.at(row, column) = lower;
      canonical.at(mirroredRow, mirroredColumn) = lower;
    }
  }

  const auto normResult = matrixOneNorm(canonical.view());
  if (!normResult.hasValue()) {
    return copyFailure<CholeskyFactorization>(normResult.status());
  }
  const auto thresholdResult =
      rankThreshold(canonical.view(), options.factorization);
  if (!thresholdResult.hasValue()) {
    return copyFailure<CholeskyFactorization>(thresholdResult.status());
  }
  const Scalar threshold = thresholdResult.value();
  Matrix lower(size, size, 0.0);
  for (Index column = 0U; column < size; ++column) {
    vss::math::detail::CompensatedSum diagonalSum;
    for (Index index = 0U; index < column; ++index) {
      const Index factorRow = column;
      const Index factorColumn = index;
      const Scalar factor = lower.at(factorRow, factorColumn);
      diagonalSum.add(factor * factor);
    }
    const Scalar diagonal = canonical.at(column, column) - diagonalSum.total();
    if (!std::isfinite(diagonal)) {
      return Result<CholeskyFactorization>::failure(
          ErrorCode::NonFiniteResult,
          "Cholesky diagonal reduction produced a non-finite value");
    }
    if (!(diagonal > threshold)) {
      return Result<CholeskyFactorization>::failure(
          ErrorCode::NotPositiveDefinite,
          "Cholesky pre-square-root pivot is not above the rank threshold");
    }
    const Scalar root = std::sqrt(diagonal);
    if (!std::isfinite(root)) {
      return Result<CholeskyFactorization>::failure(
          ErrorCode::NonFiniteResult,
          "Cholesky square root produced a non-finite value");
    }
    lower.at(column, column) = root;
    for (Index row = column + 1U; row < size; ++row) {
      vss::math::detail::CompensatedSum productSum;
      for (Index index = 0U; index < column; ++index) {
        const Index productRow = column;
        const Index productColumn = index;
        productSum.add(lower.at(row, index) *
                       lower.at(productRow, productColumn));
      }
      const Scalar value =
          (canonical.at(row, column) - productSum.total()) / root;
      if (!std::isfinite(value)) {
        return Result<CholeskyFactorization>::failure(
            ErrorCode::NonFiniteResult,
            "Cholesky lower-factor reduction produced a non-finite value");
      }
      lower.at(row, column) = value;
    }
  }

  vss::math::detail::CompensatedSum logSum;
  for (Index diagonal = 0U; diagonal < size; ++diagonal) {
    logSum.add(std::log(lower.at(diagonal, diagonal)));
  }
  const Scalar logDeterminant = 2.0 * logSum.total();
  if (!std::isfinite(logDeterminant)) {
    return Result<CholeskyFactorization>::failure(
        ErrorCode::NonFiniteResult,
        "Cholesky determinant diagnostic is not finite");
  }

  const auto inverseAction = [&](Index column) -> Result<Vector> {
    Vector basis(size, 0.0);
    basis.at(column) = 1.0;
    return solveFull(lower.view(), basis.view());
  };
  const auto conditionResult = vss::math::detail::estimateOneNormCondition(
      size, normResult.value(), inverseAction);
  if (!conditionResult.hasValue()) {
    return copyFailure<CholeskyFactorization>(conditionResult.status());
  }
  const ConditionReport conditionReport =
      report(normResult.value(), conditionResult.value(),
             options.factorization.conditionLimit);
  return Result<CholeskyFactorization>::success(
      CholeskyFactorization(size, size, threshold, options.symmetryTolerance,
                            std::move(lower), conditionReport, logDeterminant));
}

} // namespace vss::math::linear
