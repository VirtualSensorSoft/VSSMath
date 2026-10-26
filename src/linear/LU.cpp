#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/ConditionEstimate.hpp>
#include <detail/SolveOutcomeAccess.hpp>
#include <detail/SolveValidation.hpp>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LU.hpp>
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

bool validOptions(const FactorizationOptions& options) noexcept {
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
                                     "LU matrix one-norm is not finite");
    }
    norm = std::max(norm, value);
  }
  return Result<Scalar>::success(norm);
}

Result<Scalar> rankThreshold(ConstMatrixView matrix,
                             const FactorizationOptions& options) {
  Scalar scale = 0.0;
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      scale = std::max(scale, std::abs(matrix.at(row, column)));
    }
  }
  const Scalar relative = options.rankTolerance.relative * scale;
  if (!std::isfinite(relative)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "LU relative rank threshold is not finite");
  }
  const Scalar threshold = std::max(options.rankTolerance.absolute, relative);
  if (!std::isfinite(threshold)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "LU rank threshold is not finite");
  }
  return Result<Scalar>::success(threshold);
}

Result<Vector> solveWithFactors(ConstMatrixView lower, ConstMatrixView upper,
                                std::span<const Index> permutation,
                                ConstVectorView rightHandSide) {
  const Extent size = upper.rows();
  if (size == 0U) {
    return Result<Vector>::failure(ErrorCode::InvalidExtent,
                                   "LU factorization is moved-from");
  }
  if (rightHandSide.size() != size) {
    return Result<Vector>::failure(
        ErrorCode::DimensionMismatch,
        "LU factorization and right-hand side dimensions differ");
  }
  for (Index index = 0U; index < rightHandSide.size(); ++index) {
    if (!std::isfinite(rightHandSide.at(index))) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteInput,
          "LU right-hand side contains a non-finite value");
    }
  }

  Vector work(size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    work.at(row) = rightHandSide.at(permutation[row]);
  }
  for (Index row = 0U; row < size; ++row) {
    vss::math::detail::CompensatedSum accumulated;
    for (Index column = 0U; column < row; ++column) {
      accumulated.add(lower.at(row, column) * work.at(column));
    }
    work.at(row) -= accumulated.total();
    if (!std::isfinite(work.at(row))) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "LU forward substitution produced a non-finite value");
    }
  }
  for (Index step = 0U; step < size; ++step) {
    const Index row = size - 1U - step;
    vss::math::detail::CompensatedSum accumulated;
    for (Index column = row + 1U; column < size; ++column) {
      accumulated.add(upper.at(row, column) * work.at(column));
    }
    const Scalar value =
        (work.at(row) - accumulated.total()) / upper.at(row, row);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "LU backward substitution produced a non-finite value");
    }
    work.at(row) = value;
  }
  return Result<Vector>::success(std::move(work));
}

Result<Matrix> solveWithFactors(ConstMatrixView lower, ConstMatrixView upper,
                                std::span<const Index> permutation,
                                ConstMatrixView rightHandSides) {
  const Extent size = upper.rows();
  if (size == 0U) {
    return Result<Matrix>::failure(ErrorCode::InvalidExtent,
                                   "LU factorization is moved-from");
  }
  if (rightHandSides.rows() != size) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "LU factorization and right-hand side dimensions differ");
  }
  if (rightHandSides.columns() == 0U) {
    return Result<Matrix>::failure(
        ErrorCode::InvalidExtent,
        "LU matrix right-hand side must have at least one column");
  }
  for (Index row = 0U; row < rightHandSides.rows(); ++row) {
    for (Index column = 0U; column < rightHandSides.columns(); ++column) {
      if (!std::isfinite(rightHandSides.at(row, column))) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteInput,
            "LU matrix right-hand side contains a non-finite value");
      }
    }
  }

  Matrix solution(size, rightHandSides.columns(), 0.0);
  Vector work(size, 0.0);
  for (Index rightHandSide = 0U; rightHandSide < rightHandSides.columns();
       ++rightHandSide) {
    for (Index row = 0U; row < size; ++row) {
      work.at(row) = rightHandSides.at(permutation[row], rightHandSide);
    }
    for (Index row = 0U; row < size; ++row) {
      vss::math::detail::CompensatedSum accumulated;
      for (Index column = 0U; column < row; ++column) {
        accumulated.add(lower.at(row, column) * work.at(column));
      }
      work.at(row) -= accumulated.total();
      if (!std::isfinite(work.at(row))) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "LU forward substitution produced a non-finite value");
      }
    }
    for (Index step = 0U; step < size; ++step) {
      const Index row = size - 1U - step;
      vss::math::detail::CompensatedSum accumulated;
      for (Index column = row + 1U; column < size; ++column) {
        accumulated.add(upper.at(row, column) * work.at(column));
      }
      const Scalar value =
          (work.at(row) - accumulated.total()) / upper.at(row, row);
      if (!std::isfinite(value)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "LU backward substitution produced a non-finite value");
      }
      work.at(row) = value;
    }
    for (Index row = 0U; row < size; ++row) {
      solution.at(row, rightHandSide) = work.at(row);
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

} // namespace

LUFactorization::LUFactorization(
    Extent size, Scalar rankThreshold, bool singular,
    std::optional<Index> firstUnusablePivot, std::vector<Index> permutation,
    Matrix lowerFactor, Matrix upperFactor, ConditionReport conditionReport,
    int determinantSign, Scalar logAbsDeterminant) noexcept
    : size_(size), rankThreshold_(rankThreshold), singular_(singular),
      firstUnusablePivot_(firstUnusablePivot),
      permutation_(std::move(permutation)),
      lowerFactor_(std::move(lowerFactor)),
      upperFactor_(std::move(upperFactor)), conditionReport_(conditionReport),
      determinantSign_(determinantSign), logAbsDeterminant_(logAbsDeterminant) {
}

void LUFactorization::resetMovedFrom() noexcept {
  size_ = 0U;
  rankThreshold_ = 0.0;
  singular_ = false;
  firstUnusablePivot_.reset();
  permutation_.clear();
  lowerFactor_ = Matrix{};
  upperFactor_ = Matrix{};
  conditionReport_ =
      ConditionReport{.matrixOneNorm = 0.0,
                      .estimatedConditionNumber1 = 0.0,
                      .conditionLimit = 0.0,
                      .classification = ConditionClassification::WithinLimit};
  determinantSign_ = 0;
  logAbsDeterminant_ = 0.0;
}

LUFactorization::LUFactorization(LUFactorization&& other) noexcept
    : size_(other.size_), rankThreshold_(other.rankThreshold_),
      singular_(other.singular_),
      firstUnusablePivot_(other.firstUnusablePivot_),
      permutation_(std::move(other.permutation_)),
      lowerFactor_(std::move(other.lowerFactor_)),
      upperFactor_(std::move(other.upperFactor_)),
      conditionReport_(other.conditionReport_),
      determinantSign_(other.determinantSign_),
      logAbsDeterminant_(other.logAbsDeterminant_) {
  other.resetMovedFrom();
}

LUFactorization& LUFactorization::operator=(LUFactorization&& other) noexcept {
  if (this != &other) {
    size_ = other.size_;
    rankThreshold_ = other.rankThreshold_;
    singular_ = other.singular_;
    firstUnusablePivot_ = other.firstUnusablePivot_;
    permutation_ = std::move(other.permutation_);
    lowerFactor_ = std::move(other.lowerFactor_);
    upperFactor_ = std::move(other.upperFactor_);
    conditionReport_ = other.conditionReport_;
    determinantSign_ = other.determinantSign_;
    logAbsDeterminant_ = other.logAbsDeterminant_;
    other.resetMovedFrom();
  }
  return *this;
}

Extent LUFactorization::size() const noexcept { return size_; }

Scalar LUFactorization::rankThreshold() const noexcept {
  return rankThreshold_;
}

bool LUFactorization::isSingular() const noexcept { return singular_; }

std::optional<Index> LUFactorization::firstUnusablePivot() const noexcept {
  return firstUnusablePivot_;
}

std::span<const Index> LUFactorization::permutation() const noexcept {
  return permutation_;
}

const Matrix& LUFactorization::lowerFactor() const noexcept {
  return lowerFactor_;
}

const Matrix& LUFactorization::upperFactor() const noexcept {
  return upperFactor_;
}

const ConditionReport& LUFactorization::conditionReport() const noexcept {
  return conditionReport_;
}

int LUFactorization::determinantSign() const noexcept {
  return determinantSign_;
}

Scalar LUFactorization::logAbsDeterminant() const noexcept {
  return logAbsDeterminant_;
}

Result<SolveOutcome<Vector>>
LUFactorization::solve(ConstVectorView rightHandSide,
                       SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Vector>>(optionStatus);
  }
  if (size_ == 0U) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::InvalidExtent, "LU factorization is moved-from");
  }
  if (rightHandSide.size() != size_) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::DimensionMismatch,
        "LU factorization and right-hand side dimensions differ");
  }
  if (singular_) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::Singular, "cannot solve with a singular LU factorization");
  }
  if (conditionReport_.classification ==
          ConditionClassification::IllConditioned &&
      options.illConditionedPolicy == IllConditionedSolvePolicy::Reject) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::IllConditioned,
        "LU condition estimate meets or exceeds the configured condition "
        "limit");
  }
  auto result = solveWithFactors(lowerFactor_.view(), upperFactor_.view(),
                                 permutation_, rightHandSide);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Vector>>(result.status());
  }
  return Result<SolveOutcome<Vector>>::success(
      detail::SolveOutcomeAccess<Vector>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<SolveOutcome<Matrix>>
LUFactorization::solve(ConstMatrixView rightHandSides,
                       SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Matrix>>(optionStatus);
  }
  if (size_ == 0U) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::InvalidExtent, "LU factorization is moved-from");
  }
  if (rightHandSides.rows() != size_) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::DimensionMismatch,
        "LU factorization and right-hand side dimensions differ");
  }
  if (singular_) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::Singular, "cannot solve with a singular LU factorization");
  }
  if (conditionReport_.classification ==
          ConditionClassification::IllConditioned &&
      options.illConditionedPolicy == IllConditionedSolvePolicy::Reject) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::IllConditioned,
        "LU condition estimate meets or exceeds the configured condition "
        "limit");
  }
  auto result = solveWithFactors(lowerFactor_.view(), upperFactor_.view(),
                                 permutation_, rightHandSides);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Matrix>>(result.status());
  }
  return Result<SolveOutcome<Matrix>>::success(
      detail::SolveOutcomeAccess<Matrix>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<LUFactorization> factorizeLU(ConstMatrixView matrix,
                                    FactorizationOptions options) {
  if (!validOptions(options)) {
    return Result<LUFactorization>::failure(
        ErrorCode::InvalidTolerance,
        "LU factorization options contain an invalid tolerance");
  }
  if (matrix.rows() == 0U || matrix.columns() == 0U) {
    return Result<LUFactorization>::failure(
        ErrorCode::InvalidExtent,
        "LU factorization requires a non-empty matrix");
  }
  if (matrix.rows() != matrix.columns()) {
    return Result<LUFactorization>::failure(
        ErrorCode::DimensionMismatch,
        "LU factorization requires a square matrix");
  }
  Extent elementCount = 0U;
  if (!checkedSquare(matrix.rows(), elementCount)) {
    return Result<LUFactorization>::failure(
        ErrorCode::SizeOverflow,
        "LU factorization matrix size overflows its extent representation");
  }
  static_cast<void>(elementCount);
  if (!finite(matrix)) {
    return Result<LUFactorization>::failure(
        ErrorCode::NonFiniteInput,
        "LU factorization input contains a non-finite value");
  }
  const auto normResult = matrixOneNorm(matrix);
  if (!normResult.hasValue()) {
    return copyFailure<LUFactorization>(normResult.status());
  }
  const auto thresholdResult = rankThreshold(matrix, options);
  if (!thresholdResult.hasValue()) {
    return copyFailure<LUFactorization>(thresholdResult.status());
  }
  const Extent size = matrix.rows();
  const Scalar threshold = thresholdResult.value();
  Matrix work(size, size);
  Matrix lower(size, size, 0.0);
  Matrix upper(size, size, 0.0);
  std::vector<Index> permutation(size);
  for (Index row = 0U; row < size; ++row) {
    permutation[row] = row;
    lower.at(row, row) = 1.0;
    for (Index column = 0U; column < size; ++column) {
      work.at(row, column) = matrix.at(row, column);
    }
  }

  bool singular = false;
  std::optional<Index> firstUnusablePivot;
  bool oddPermutation = false;
  for (Index pivotColumn = 0U; pivotColumn < size; ++pivotColumn) {
    Index selectedRow = pivotColumn;
    Scalar selectedMagnitude = std::abs(work.at(selectedRow, pivotColumn));
    for (Index row = pivotColumn + 1U; row < size; ++row) {
      const Scalar magnitude = std::abs(work.at(row, pivotColumn));
      if (magnitude > selectedMagnitude ||
          (magnitude == selectedMagnitude &&
           permutation[row] < permutation[selectedRow])) {
        selectedRow = row;
        selectedMagnitude = magnitude;
      }
    }
    if (selectedRow != pivotColumn) {
      for (Index column = 0U; column < size; ++column) {
        std::swap(work.at(pivotColumn, column), work.at(selectedRow, column));
      }
      for (Index column = 0U; column < pivotColumn; ++column) {
        std::swap(lower.at(pivotColumn, column), lower.at(selectedRow, column));
      }
      std::swap(permutation[pivotColumn], permutation[selectedRow]);
      oddPermutation = !oddPermutation;
    }

    const Scalar pivot = work.at(pivotColumn, pivotColumn);
    if (!(std::abs(pivot) > threshold)) {
      singular = true;
      if (!firstUnusablePivot.has_value()) {
        firstUnusablePivot = pivotColumn;
      }
    }
    if (pivot == 0.0) {
      continue;
    }
    for (Index row = pivotColumn + 1U; row < size; ++row) {
      const Scalar multiplier = work.at(row, pivotColumn) / pivot;
      if (!std::isfinite(multiplier)) {
        return Result<LUFactorization>::failure(
            ErrorCode::NonFiniteResult,
            "LU elimination produced a non-finite multiplier");
      }
      lower.at(row, pivotColumn) = multiplier;
      work.at(row, pivotColumn) = 0.0;
      for (Index column = pivotColumn + 1U; column < size; ++column) {
        const Scalar update =
            work.at(row, column) - (multiplier * work.at(pivotColumn, column));
        if (!std::isfinite(update)) {
          return Result<LUFactorization>::failure(
              ErrorCode::NonFiniteResult,
              "LU elimination produced a non-finite update");
        }
        work.at(row, column) = update;
      }
    }
  }

  for (Index row = 0U; row < size; ++row) {
    for (Index column = row; column < size; ++column) {
      upper.at(row, column) = work.at(row, column);
    }
  }

  int determinantSign = oddPermutation ? -1 : 1;
  Scalar logAbsDeterminant = 0.0;
  if (singular) {
    determinantSign = 0;
    logAbsDeterminant = -std::numeric_limits<Scalar>::infinity();
  } else {
    vss::math::detail::CompensatedSum logSum;
    for (Index diagonal = 0U; diagonal < size; ++diagonal) {
      const Scalar value = upper.at(diagonal, diagonal);
      if (value < 0.0) {
        determinantSign = -determinantSign;
      }
      logSum.add(std::log(std::abs(value)));
    }
    logAbsDeterminant = logSum.total();
    if (!std::isfinite(logAbsDeterminant)) {
      return Result<LUFactorization>::failure(
          ErrorCode::NonFiniteResult,
          "LU determinant diagnostic is not finite");
    }
  }

  Scalar condition = std::numeric_limits<Scalar>::infinity();
  if (!singular) {
    const auto inverseAction = [&](Index column) -> Result<Vector> {
      Vector basis(size, 0.0);
      basis.at(column) = 1.0;
      return solveWithFactors(lower.view(), upper.view(), permutation,
                              basis.view());
    };
    const auto estimate = vss::math::detail::estimateOneNormCondition(
        size, normResult.value(), inverseAction);
    if (!estimate.hasValue()) {
      return copyFailure<LUFactorization>(estimate.status());
    }
    condition = estimate.value();
  }

  const ConditionReport conditionReport =
      report(normResult.value(), condition, options.conditionLimit);
  return Result<LUFactorization>::success(LUFactorization(
      size, threshold, singular, firstUnusablePivot, std::move(permutation),
      std::move(lower), std::move(upper), conditionReport, determinantSign,
      logAbsDeterminant));
}

} // namespace vss::math::linear
