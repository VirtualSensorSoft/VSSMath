#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/ConditionEstimate.hpp>
#include <detail/ScaledNorm.hpp>
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
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/QR.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedProduct(Extent lhs, Extent rhs, Extent& product) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  product = lhs * rhs;
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

bool finite(ConstVectorView vector) noexcept {
  for (Index index = 0U; index < vector.size(); ++index) {
    if (!std::isfinite(vector.at(index))) {
      return false;
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
                                     "QR matrix one-norm is not finite");
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
                                   "QR relative rank threshold is not finite");
  }
  const Scalar threshold = std::max(options.rankTolerance.absolute, relative);
  if (!std::isfinite(threshold)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "QR rank threshold is not finite");
  }
  return Result<Scalar>::success(threshold);
}

bool finiteMatrix(ConstMatrixView matrix) noexcept { return finite(matrix); }

Result<Vector> solveWithFactors(ConstMatrixView reflectors,
                                std::span<const Scalar> reflectorCoefficients,
                                std::span<const Scalar> normalizationSigns,
                                ConstMatrixView upper,
                                std::span<const Index> permutation,
                                ConstVectorView rightHandSide) {
  const Extent rows = reflectors.rows();
  const Extent columns = upper.rows();
  if (columns == 0U || rows == 0U) {
    return Result<Vector>::failure(ErrorCode::InvalidExtent,
                                   "QR factorization is moved-from");
  }
  if (rightHandSide.size() != rows) {
    return Result<Vector>::failure(
        ErrorCode::DimensionMismatch,
        "QR factorization and right-hand side dimensions differ");
  }

  Vector work(rows, 0.0);
  for (Index row = 0U; row < rows; ++row) {
    work.at(row) = rightHandSide.at(row);
  }
  for (Index reflector = 0U; reflector < columns; ++reflector) {
    const Scalar tau = reflectorCoefficients[reflector];
    if (tau == 0.0) {
      continue;
    }
    vss::math::detail::CompensatedSum dot;
    for (Index row = reflector; row < rows; ++row) {
      dot.add(reflectors.at(row, reflector) * work.at(row));
    }
    const Scalar factor = tau * dot.total();
    if (!std::isfinite(factor)) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "QR least-squares reflector application is not finite");
    }
    for (Index row = reflector; row < rows; ++row) {
      const Scalar update =
          work.at(row) - (reflectors.at(row, reflector) * factor);
      if (!std::isfinite(update)) {
        return Result<Vector>::failure(
            ErrorCode::NonFiniteResult,
            "QR least-squares reflector update is not finite");
      }
      work.at(row) = update;
    }
  }

  Vector permutedSolution(columns, 0.0);
  for (Index index = 0U; index < columns; ++index) {
    permutedSolution.at(index) = work.at(index) * normalizationSigns[index];
    if (!std::isfinite(permutedSolution.at(index))) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "QR least-squares transformed right-hand side is not finite");
    }
  }
  for (Index step = 0U; step < columns; ++step) {
    const Index row = columns - 1U - step;
    vss::math::detail::CompensatedSum accumulated;
    for (Index column = row + 1U; column < columns; ++column) {
      accumulated.add(upper.at(row, column) * permutedSolution.at(column));
    }
    const Scalar numerator = permutedSolution.at(row) - accumulated.total();
    const Scalar value = numerator / upper.at(row, row);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "QR least-squares triangular solve is not finite");
    }
    permutedSolution.at(row) = value;
  }

  Vector solution(columns, 0.0);
  for (Index column = 0U; column < columns; ++column) {
    solution.at(permutation[column]) = permutedSolution.at(column);
  }
  return Result<Vector>::success(std::move(solution));
}

Result<Matrix> solveWithFactors(ConstMatrixView reflectors,
                                std::span<const Scalar> reflectorCoefficients,
                                std::span<const Scalar> normalizationSigns,
                                ConstMatrixView upper,
                                std::span<const Index> permutation,
                                ConstMatrixView rightHandSides) {
  const Extent rows = reflectors.rows();
  const Extent columns = upper.rows();
  if (columns == 0U || rows == 0U) {
    return Result<Matrix>::failure(ErrorCode::InvalidExtent,
                                   "QR factorization is moved-from");
  }
  if (rightHandSides.rows() != rows) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "QR factorization and right-hand side dimensions differ");
  }
  if (rightHandSides.columns() == 0U) {
    return Result<Matrix>::failure(
        ErrorCode::InvalidExtent,
        "QR matrix right-hand side must have at least one column");
  }

  Matrix solution(columns, rightHandSides.columns(), 0.0);
  Vector work(rows, 0.0);
  Vector permutedSolution(columns, 0.0);
  for (Index rightHandSide = 0U; rightHandSide < rightHandSides.columns();
       ++rightHandSide) {
    for (Index row = 0U; row < rows; ++row) {
      work.at(row) = rightHandSides.at(row, rightHandSide);
    }
    for (Index reflector = 0U; reflector < columns; ++reflector) {
      const Scalar tau = reflectorCoefficients[reflector];
      if (tau == 0.0) {
        continue;
      }
      vss::math::detail::CompensatedSum dot;
      for (Index row = reflector; row < rows; ++row) {
        dot.add(reflectors.at(row, reflector) * work.at(row));
      }
      const Scalar factor = tau * dot.total();
      if (!std::isfinite(factor)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "QR matrix reflector application is not finite");
      }
      for (Index row = reflector; row < rows; ++row) {
        const Scalar update =
            work.at(row) - (reflectors.at(row, reflector) * factor);
        if (!std::isfinite(update)) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "QR matrix reflector update is not finite");
        }
        work.at(row) = update;
      }
    }
    for (Index index = 0U; index < columns; ++index) {
      permutedSolution.at(index) = work.at(index) * normalizationSigns[index];
      if (!std::isfinite(permutedSolution.at(index))) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "QR matrix transformed right-hand side is not finite");
      }
    }
    for (Index step = 0U; step < columns; ++step) {
      const Index row = columns - 1U - step;
      vss::math::detail::CompensatedSum accumulated;
      for (Index column = row + 1U; column < columns; ++column) {
        accumulated.add(upper.at(row, column) * permutedSolution.at(column));
      }
      const Scalar numerator = permutedSolution.at(row) - accumulated.total();
      const Scalar value = numerator / upper.at(row, row);
      if (!std::isfinite(value)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "QR matrix triangular solve is not finite");
      }
      permutedSolution.at(row) = value;
    }
    for (Index column = 0U; column < columns; ++column) {
      solution.at(permutation[column], rightHandSide) =
          permutedSolution.at(column);
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

QRFactorization::QRFactorization(
    Extent rows, Extent columns, Extent rank, Scalar rankThreshold,
    std::vector<Index> permutation, Matrix reflectors,
    std::vector<Scalar> reflectorCoefficients,
    std::vector<Scalar> normalizationSigns, Matrix upperFactor,
    Scalar matrixOneNorm,
    std::optional<ConditionReport> conditionReport) noexcept
    : rows_(rows), columns_(columns), rank_(rank),
      rankThreshold_(rankThreshold), permutation_(std::move(permutation)),
      reflectors_(std::move(reflectors)),
      reflectorCoefficients_(std::move(reflectorCoefficients)),
      normalizationSigns_(std::move(normalizationSigns)),
      upperFactor_(std::move(upperFactor)), matrixOneNorm_(matrixOneNorm),
      conditionReport_(conditionReport) {}

void QRFactorization::resetMovedFrom() noexcept {
  rows_ = 0U;
  columns_ = 0U;
  rank_ = 0U;
  rankThreshold_ = 0.0;
  permutation_.clear();
  reflectors_ = Matrix{};
  reflectorCoefficients_.clear();
  normalizationSigns_.clear();
  upperFactor_ = Matrix{};
  matrixOneNorm_ = 0.0;
  conditionReport_.reset();
}

QRFactorization::QRFactorization(QRFactorization&& other) noexcept
    : rows_(other.rows_), columns_(other.columns_), rank_(other.rank_),
      rankThreshold_(other.rankThreshold_),
      permutation_(std::move(other.permutation_)),
      reflectors_(std::move(other.reflectors_)),
      reflectorCoefficients_(std::move(other.reflectorCoefficients_)),
      normalizationSigns_(std::move(other.normalizationSigns_)),
      upperFactor_(std::move(other.upperFactor_)),
      matrixOneNorm_(other.matrixOneNorm_),
      conditionReport_(other.conditionReport_) {
  other.resetMovedFrom();
}

QRFactorization& QRFactorization::operator=(QRFactorization&& other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    rank_ = other.rank_;
    rankThreshold_ = other.rankThreshold_;
    permutation_ = std::move(other.permutation_);
    reflectors_ = std::move(other.reflectors_);
    reflectorCoefficients_ = std::move(other.reflectorCoefficients_);
    normalizationSigns_ = std::move(other.normalizationSigns_);
    upperFactor_ = std::move(other.upperFactor_);
    matrixOneNorm_ = other.matrixOneNorm_;
    conditionReport_ = other.conditionReport_;
    other.resetMovedFrom();
  }
  return *this;
}

Extent QRFactorization::rows() const noexcept { return rows_; }

Extent QRFactorization::columns() const noexcept { return columns_; }

Extent QRFactorization::rank() const noexcept { return rank_; }

Scalar QRFactorization::rankThreshold() const noexcept {
  return rankThreshold_;
}

bool QRFactorization::hasFullColumnRank() const noexcept {
  return columns_ != 0U && rank_ == columns_;
}

bool QRFactorization::isSquare() const noexcept {
  return rows_ != 0U && rows_ == columns_;
}

std::span<const Index> QRFactorization::permutation() const noexcept {
  return permutation_;
}

const Matrix& QRFactorization::upperFactor() const noexcept {
  return upperFactor_;
}

Scalar QRFactorization::matrixOneNorm() const noexcept {
  return matrixOneNorm_;
}

const std::optional<ConditionReport>&
QRFactorization::conditionReport() const noexcept {
  return conditionReport_;
}

Matrix QRFactorization::economyQ() const {
  if (rows_ == 0U || columns_ == 0U) {
    return Matrix{};
  }
  Matrix q(rows_, columns_, 0.0);
  for (Index column = 0U; column < columns_; ++column) {
    q.at(column, column) = normalizationSigns_[column];
  }
  for (Index step = 0U; step < columns_; ++step) {
    const Index reflector = columns_ - 1U - step;
    const Scalar tau = reflectorCoefficients_[reflector];
    if (tau == 0.0) {
      continue;
    }
    for (Index column = 0U; column < columns_; ++column) {
      vss::math::detail::CompensatedSum dot;
      for (Index row = reflector; row < rows_; ++row) {
        dot.add(reflectors_.at(row, reflector) * q.at(row, column));
      }
      const Scalar factor = tau * dot.total();
      if (!std::isfinite(factor)) {
        return Matrix{};
      }
      for (Index row = reflector; row < rows_; ++row) {
        q.at(row, column) -= reflectors_.at(row, reflector) * factor;
      }
    }
  }
  return q;
}

Result<SolveOutcome<Vector>>
QRFactorization::leastSquares(ConstVectorView rightHandSide,
                              SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Vector>>(optionStatus);
  }
  if (rows_ == 0U || columns_ == 0U) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::InvalidExtent, "QR factorization is moved-from");
  }
  if (rightHandSide.size() != rows_) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::DimensionMismatch,
        "QR factorization and right-hand side dimensions differ");
  }
  if (!finite(rightHandSide)) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::NonFiniteInput,
        "QR right-hand side contains a non-finite value");
  }
  if (!hasFullColumnRank()) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::RankDeficient,
        "cannot solve with a rank-deficient QR factorization");
  }
  if (conditionReport_.has_value() &&
      conditionReport_->classification ==
          ConditionClassification::IllConditioned &&
      options.illConditionedPolicy == IllConditionedSolvePolicy::Reject) {
    return Result<SolveOutcome<Vector>>::failure(
        ErrorCode::IllConditioned, "QR condition estimate meets or exceeds the "
                                   "configured condition limit");
  }
  auto result = solveWithFactors(reflectors_.view(), reflectorCoefficients_,
                                 normalizationSigns_, upperFactor_.view(),
                                 permutation_, rightHandSide);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Vector>>(result.status());
  }
  return Result<SolveOutcome<Vector>>::success(
      detail::SolveOutcomeAccess<Vector>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<SolveOutcome<Matrix>>
QRFactorization::leastSquares(ConstMatrixView rightHandSides,
                              SolveOptions options) const {
  const Status optionStatus = detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<SolveOutcome<Matrix>>(optionStatus);
  }
  if (rows_ == 0U || columns_ == 0U) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::InvalidExtent, "QR factorization is moved-from");
  }
  if (rightHandSides.rows() != rows_) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::DimensionMismatch,
        "QR factorization and right-hand side dimensions differ");
  }
  if (rightHandSides.columns() == 0U) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::InvalidExtent,
        "QR matrix right-hand side must have at least one column");
  }
  if (!finite(rightHandSides)) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::NonFiniteInput,
        "QR matrix right-hand side contains a non-finite value");
  }
  if (!hasFullColumnRank()) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::RankDeficient,
        "cannot solve with a rank-deficient QR factorization");
  }
  if (conditionReport_.has_value() &&
      conditionReport_->classification ==
          ConditionClassification::IllConditioned &&
      options.illConditionedPolicy == IllConditionedSolvePolicy::Reject) {
    return Result<SolveOutcome<Matrix>>::failure(
        ErrorCode::IllConditioned, "QR condition estimate meets or exceeds the "
                                   "configured condition limit");
  }
  auto result = solveWithFactors(reflectors_.view(), reflectorCoefficients_,
                                 normalizationSigns_, upperFactor_.view(),
                                 permutation_, rightHandSides);
  if (!result.hasValue()) {
    return copyFailure<SolveOutcome<Matrix>>(result.status());
  }
  return Result<SolveOutcome<Matrix>>::success(
      detail::SolveOutcomeAccess<Matrix>::make(std::move(result.value()),
                                               conditionReport_));
}

Result<QRFactorization> factorizeQR(ConstMatrixView matrix,
                                    FactorizationOptions options) {
  if (!validOptions(options)) {
    return Result<QRFactorization>::failure(
        ErrorCode::InvalidTolerance,
        "QR factorization options contain an invalid tolerance");
  }
  const Extent rows = matrix.rows();
  const Extent columns = matrix.columns();
  if (rows == 0U || columns == 0U || rows < columns) {
    return Result<QRFactorization>::failure(
        ErrorCode::InvalidExtent,
        "QR factorization requires a non-empty matrix with rows >= columns");
  }
  Extent elementCount = 0U;
  Extent triangularCount = 0U;
  if (!checkedProduct(rows, columns, elementCount) ||
      !checkedProduct(columns, columns, triangularCount)) {
    return Result<QRFactorization>::failure(
        ErrorCode::SizeOverflow,
        "QR factorization matrix size overflows its extent representation");
  }
  static_cast<void>(elementCount);
  static_cast<void>(triangularCount);
  if (!finiteMatrix(matrix)) {
    return Result<QRFactorization>::failure(
        ErrorCode::NonFiniteInput,
        "QR factorization input contains a non-finite value");
  }
  const auto normResult = matrixOneNorm(matrix);
  if (!normResult.hasValue()) {
    return copyFailure<QRFactorization>(normResult.status());
  }
  const auto thresholdResult = rankThreshold(matrix, options);
  if (!thresholdResult.hasValue()) {
    return copyFailure<QRFactorization>(thresholdResult.status());
  }

  Matrix work(rows, columns, 0.0);
  Matrix reflectors(rows, columns, 0.0);
  Matrix upper(columns, columns, 0.0);
  std::vector<Index> permutation(columns);
  std::vector<Scalar> coefficients(columns, 0.0);
  std::vector<Scalar> signs(columns, 1.0);
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      work.at(row, column) = matrix.at(row, column);
    }
  }
  for (Index column = 0U; column < columns; ++column) {
    permutation[column] = column;
  }

  for (Index pivot = 0U; pivot < columns; ++pivot) {
    Index selected = pivot;
    vss::math::detail::ScaledNorm selectedNormAccumulator;
    for (Index row = pivot; row < rows; ++row) {
      selectedNormAccumulator.add(work.at(row, selected));
    }
    Scalar selectedNorm = selectedNormAccumulator.value();
    for (Index candidate = pivot + 1U; candidate < columns; ++candidate) {
      vss::math::detail::ScaledNorm candidateNormAccumulator;
      for (Index row = pivot; row < rows; ++row) {
        candidateNormAccumulator.add(work.at(row, candidate));
      }
      const Scalar candidateNorm = candidateNormAccumulator.value();
      if (!std::isfinite(candidateNorm)) {
        return Result<QRFactorization>::failure(
            ErrorCode::NonFiniteResult,
            "QR trailing column norm is not finite");
      }
      if (candidateNorm > selectedNorm ||
          (candidateNorm == selectedNorm &&
           permutation[candidate] < permutation[selected])) {
        selected = candidate;
        selectedNorm = candidateNorm;
      }
    }
    if (!std::isfinite(selectedNorm)) {
      return Result<QRFactorization>::failure(
          ErrorCode::NonFiniteResult, "QR trailing column norm is not finite");
    }
    if (selected != pivot) {
      for (Index row = 0U; row < rows; ++row) {
        std::swap(work.at(row, pivot), work.at(row, selected));
      }
      std::swap(permutation[pivot], permutation[selected]);
    }

    if (selectedNorm == 0.0) {
      coefficients[pivot] = 0.0;
      signs[pivot] = 1.0;
      upper.at(pivot, pivot) = 0.0;
      continue;
    }

    const Scalar alpha = work.at(pivot, pivot);
    const Scalar householderSign = alpha >= 0.0 ? 1.0 : -1.0;
    for (Index row = pivot; row < rows; ++row) {
      const Scalar normalized = work.at(row, pivot) / selectedNorm;
      if (!std::isfinite(normalized)) {
        return Result<QRFactorization>::failure(
            ErrorCode::NonFiniteResult,
            "QR Householder normalization is not finite");
      }
      reflectors.at(row, pivot) = normalized;
    }
    reflectors.at(pivot, pivot) += householderSign;
    vss::math::detail::CompensatedSum squaredNorm;
    for (Index row = pivot; row < rows; ++row) {
      const Scalar value = reflectors.at(row, pivot);
      squaredNorm.add(value * value);
    }
    const Scalar tau = 2.0 / squaredNorm.total();
    if (!std::isfinite(tau)) {
      return Result<QRFactorization>::failure(
          ErrorCode::NonFiniteResult,
          "QR Householder coefficient is not finite");
    }
    coefficients[pivot] = tau;
    const Scalar rawDiagonal = -householderSign * selectedNorm;
    signs[pivot] = rawDiagonal < 0.0 ? -1.0 : 1.0;

    for (Index column = pivot + 1U; column < columns; ++column) {
      vss::math::detail::CompensatedSum dot;
      for (Index row = pivot; row < rows; ++row) {
        dot.add(reflectors.at(row, pivot) * work.at(row, column));
      }
      const Scalar factor = tau * dot.total();
      if (!std::isfinite(factor)) {
        return Result<QRFactorization>::failure(
            ErrorCode::NonFiniteResult, "QR Householder update is not finite");
      }
      for (Index row = pivot; row < rows; ++row) {
        const Scalar update =
            work.at(row, column) - (reflectors.at(row, pivot) * factor);
        if (!std::isfinite(update)) {
          return Result<QRFactorization>::failure(
              ErrorCode::NonFiniteResult,
              "QR Householder update is not finite");
        }
        work.at(row, column) = update;
      }
    }
    work.at(pivot, pivot) = rawDiagonal;
    for (Index row = pivot + 1U; row < rows; ++row) {
      work.at(row, pivot) = 0.0;
    }
  }

  for (Index row = 0U; row < columns; ++row) {
    for (Index column = row; column < columns; ++column) {
      Scalar value = work.at(row, column) * signs[row];
      if (value == 0.0) {
        value = 0.0;
      }
      if (!std::isfinite(value)) {
        return Result<QRFactorization>::failure(
            ErrorCode::NonFiniteResult, "QR public upper factor is not finite");
      }
      upper.at(row, column) = value;
    }
  }

  const Scalar threshold = thresholdResult.value();
  Extent rank = 0U;
  for (Index diagonal = 0U; diagonal < columns; ++diagonal) {
    if (std::abs(upper.at(diagonal, diagonal)) > threshold) {
      ++rank;
    }
  }

  std::optional<ConditionReport> conditionReport;
  if (rows == columns) {
    Scalar condition = std::numeric_limits<Scalar>::infinity();
    if (rank == columns) {
      const auto inverseAction = [&](Index column) -> Result<Vector> {
        Vector basis(columns, 0.0);
        basis.at(column) = 1.0;
        return solveWithFactors(reflectors.view(), coefficients, signs,
                                upper.view(), permutation, basis.view());
      };
      const auto estimate = vss::math::detail::estimateOneNormCondition(
          columns, normResult.value(), inverseAction);
      if (estimate.hasValue()) {
        condition = estimate.value();
      } else if (estimate.status().code() == ErrorCode::NonFiniteResult) {
        condition = std::numeric_limits<Scalar>::infinity();
      } else {
        return copyFailure<QRFactorization>(estimate.status());
      }
    }
    if (std::isnan(condition)) {
      return Result<QRFactorization>::failure(
          ErrorCode::NonFiniteResult,
          "QR condition diagnostic is not a number");
    }
    conditionReport =
        report(normResult.value(), condition, options.conditionLimit);
  }

  return Result<QRFactorization>::success(QRFactorization(
      rows, columns, rank, threshold, std::move(permutation),
      std::move(reflectors), std::move(coefficients), std::move(signs),
      std::move(upper), normResult.value(), conditionReport));
}

} // namespace vss::math::linear
